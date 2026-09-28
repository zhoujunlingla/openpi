"""Read-only selected parameter comparison for π0.5 base and LIBERO checkpoints."""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

from openpi.models import model as model_lib


def stats(base, tuned):
    a = jnp.asarray(base, dtype=jnp.float32)
    b = jnp.asarray(tuned, dtype=jnp.float32)
    delta = b - a
    return {
        "shape": list(a.shape),
        "base_rms": float(jnp.sqrt(jnp.mean(a * a))),
        "tuned_rms": float(jnp.sqrt(jnp.mean(b * b))),
        "relative_rms_change": float(jnp.sqrt(jnp.mean(delta * delta) / jnp.mean(a * a))),
        "exact_equal_fraction": float(jnp.mean(a == b)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("Restoring base", flush=True)
    a = model_lib.restore_params(args.checkpoint_root / "pi05_base" / "params", dtype=jnp.bfloat16)
    print("Restoring LIBERO", flush=True)
    b = model_lib.restore_params(args.checkpoint_root / "pi05_libero" / "params", dtype=jnp.bfloat16)
    llm_a = a["PaliGemma"]["llm"]
    llm_b = b["PaliGemma"]["llm"]
    emb_a = llm_a["embedder"]["input_embedding"]
    emb_b = llm_b["embedder"]["input_embedding"]
    selected = {
        "embedding_ordinary_rows_0_8192": stats(emb_a[:8192], emb_b[:8192]),
        "embedding_rare_rows_255000_256000": stats(emb_a[255000:256000], emb_b[255000:256000]),
        "embedding_observed_token_255684": stats(emb_a[255684], emb_b[255684]),
        "gemma_final_norm": stats(llm_a["final_norm"]["scale"], llm_b["final_norm"]["scale"]),
        "gemma_attention_q_layer0": stats(
            llm_a["layers"]["attn"]["q_einsum"]["w"][0],
            llm_b["layers"]["attn"]["q_einsum"]["w"][0],
        ),
        "gemma_attention_q_layer17": stats(
            llm_a["layers"]["attn"]["q_einsum"]["w"][17],
            llm_b["layers"]["attn"]["q_einsum"]["w"][17],
        ),
        "gemma_mlp_layer0_first256": stats(
            llm_a["layers"]["mlp"]["gating_einsum"][0, :, :, :256],
            llm_b["layers"]["mlp"]["gating_einsum"][0, :, :, :256],
        ),
        "vision_encoder_norm": stats(
            a["PaliGemma"]["img"]["Transformer"]["encoder_norm"]["scale"],
            b["PaliGemma"]["img"]["Transformer"]["encoder_norm"]["scale"],
        ),
        "action_out_proj_kernel": stats(a["action_out_proj"]["kernel"], b["action_out_proj"]["kernel"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps(selected, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
