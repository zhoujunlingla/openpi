"""Dependency-light tests. These do not load openpi, real weights, or LIBERO."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from serve_pi05_text_audit import compare_logits, describe_token, reference_layout, write_report


@pytest.mark.parametrize('count', [0, 1, 3])
def test_reference_mask_and_positions_with_holes(count):
    prefix = np.array([[True, False, True, True, False]])
    mask, positions, last = reference_layout(prefix, count, 3)
    plen = prefix.shape[1]
    valid = np.r_[prefix[0], np.arange(3) < count]
    for q in range(plen + 3):
        for k in range(plen + 3):
            causal = k < plen if q < plen else (k < plen or k <= q)
            assert mask[0, q, k] == (valid[q] and valid[k] and causal)
    assert last[0] == (3 if count == 0 else plen + count - 1)
    np.testing.assert_array_equal(positions[0], np.cumsum(valid) - 1)
    # Neither prompt nor image queries can read generated suffix.
    assert not mask[0, :plen, plen:].any()


def test_jax_jit_reference_matches_numpy():
    prefix = np.array([[True, False, True], [False, True, True]])
    fn = jax.jit(lambda v, n: reference_layout(v, n, 4, xp=jnp))
    for count in range(5):
        expected = reference_layout(prefix, count, 4)
        actual = fn(jnp.asarray(prefix), jnp.asarray(count, jnp.int32))
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(np.asarray(a), b)


def test_metrics_identical_and_constant_shift():
    a = np.array([99., 1., 2., 3., 4.])
    same = compare_logits(a, a, [0])
    assert same['raw_max_abs_error'] == 0
    assert same['top1_equal']
    assert same['reference_top1_id'] == 4  # banned ID0 cannot win
    shifted = compare_logits(a + 25, a, [0])
    assert shifted['raw_max_abs_error'] == 25
    assert shifted['centered_max_abs_error'] == 0
    assert shifted['probability_total_variation'] < 1e-12


def test_metrics_nonfinite_rejected():
    with pytest.raises(ValueError, match='Non-finite'):
        compare_logits([np.nan, 1], [2, 1], [])


def test_token_unicode_and_unmapped_preserved():
    class FakeTokenizer:
        def get_piece_size(self): return 3
        def id_to_piece(self, i): return ['<pad>', 'pick', '\ue06e'][i]
        def decode(self, ids): return ''.join(self.id_to_piece(i) for i in ids)
        def is_control(self, i): return i == 0
        def eos_id(self): return 0
    assert describe_token(FakeTokenizer(), 2)['unicode_categories'] == ['Co']
    assert describe_token(FakeTokenizer(), 9)['unmapped']


def test_atomic_report_preserves_escaped_text(tmp_path):
    path = tmp_path / 'subdir' / 'report.json'
    text = '\ue06e\ud800'
    write_report(path, {'text': text})
    assert json.loads(path.read_text())['text'] == text
    assert not path.with_name(path.name + '.tmp').exists()


def test_toy_multilayer_no_cache_matches_incremental_cache_with_holes():
    # A three-layer deterministic attention network, not an openpi model.
    rng = np.random.default_rng(123)
    width, plen, capacity = 8, 6, 4
    prefix = rng.normal(size=(1, plen, width))
    suffix = rng.normal(size=(1, capacity, width))
    valid = np.array([[True, False, True, False, True, True]])
    weights = [[rng.normal(size=(width, width)) * .1 for _ in range(3)] for _ in range(3)]

    def forward(x, mask, kv=None):
        caches = []
        for li, (wq, wk, wv) in enumerate(weights):
            q, k, v = x @ wq, x @ wk, x @ wv
            if kv is not None:
                k = np.concatenate([kv[li][0], k], axis=1)
                v = np.concatenate([kv[li][1], v], axis=1)
            logits = q @ k.swapaxes(-1, -2) / width**.5
            logits = np.where(mask, logits, -1e20)
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            x = np.tanh(x + probs @ v)
            caches.append((k, v))
        return x, caches

    mask0, _, _ = reference_layout(valid, 0, capacity)
    pref_out, cache = forward(prefix, mask0[:, :plen, :plen])
    got = pref_out[:, np.flatnonzero(valid[0])[-1]]
    for count in range(capacity + 1):
        mask, _, last = reference_layout(valid, count, capacity)
        full, _ = forward(np.concatenate([prefix, suffix], axis=1), mask)
        np.testing.assert_allclose(got, full[np.arange(1), last], atol=1e-12, rtol=1e-12)
        if count < capacity:
            step_mask = np.concatenate([valid, np.ones((1, count + 1), bool)], axis=1)[:, None]
            out, cache = forward(suffix[:, count:count+1], step_mask, cache)
            got = out[:, 0]
