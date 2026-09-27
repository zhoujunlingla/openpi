"""Fixed-size adapter for openpi's append-style JAX Gemma KV cache.

The scanned cache layout is [layers, batch, key_length, kv_heads, head_dim].
This module deliberately has no Flax dependency, so its indexing and masks can
be tested independently. It does not change the model's attention mathematics.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def last_valid_index(mask: jax.Array) -> jax.Array:
    """Return physical indices, not token counts; masked camera blocks create holes."""
    if mask.ndim != 2:
        raise ValueError("Expected a [batch, prefix_length] validity mask")
    return jnp.max(jnp.where(mask, jnp.arange(mask.shape[1])[None, :], -1), axis=1)


def pad_cache(cache, extra_slots: int):
    if extra_slots < 1:
        raise ValueError("extra_slots must be positive")
    def pad_one(x):
        if x.ndim != 5:
            raise ValueError("Expected scanned cache [layers,batch,length,heads,dim]")
        return jnp.pad(x, ((0, 0), (0, 0), (0, extra_slots), (0, 0), (0, 0)))
    return jax.tree.map(pad_one, cache)


def next_token_mask(prefix_mask: jax.Array, index: jax.Array, max_tokens: int) -> jax.Array:
    """Attend to valid prefix, earlier generated slots and newly appended token.

    At this call Gemma appends the current token to the END of a fixed-capacity
    cache, not to its eventual logical slot. The last True corresponds to that
    temporary appended token. Unwritten reserved slots must stay masked.
    """
    batch = prefix_mask.shape[0]
    history = jnp.broadcast_to(jnp.arange(max_tokens)[None, :] < index, (batch, max_tokens))
    self_slot = jnp.ones((batch, 1), dtype=jnp.bool_)
    return jnp.concatenate((prefix_mask, history, self_slot), axis=1)[:, None, :]


def compact_cache(appended_cache, prefix_length: int, index: jax.Array):
    """Move the newly appended KV to prefix_length+index and restore fixed shape."""
    def compact_one(x):
        base = x[:, :, :-1, :, :]
        return base.at[:, :, prefix_length + index, :, :].set(x[:, :, -1, :, :])
    return jax.tree.map(compact_one, appended_cache)
