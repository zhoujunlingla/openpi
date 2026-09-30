"""Controlled, CPU-compatible text-only parameter interventions for pi05.

No checkpoint is written and no action is executed. Parameters are inputs to
JIT calls, so each intervention actually changes the decoder's weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from flax import nnx, traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.policies import pi05_text_cache as cache_lib
from openpi.policies import pi05_text_debug as text_lib
from openpi.policies import policy_config
from openpi.training import config


class PrefixProbe(nnx.Module):
    def __init__(self, model):
        self.model = model

    def run(self, observation):
        observation = model_lib.preprocess_observation(None, observation, train=False)
        embedded, valid, ar = self.model.embed_prefix(observation)
        hidden, _ = self.model.PaliGemma.llm(
            [embedded, None],
            positions=jnp.cumsum(valid.astype(jnp.int32), axis=1) - 1,
            mask=pi0.make_attn_mask(valid, ar),
        )
        h = hidden[0][jnp.arange(valid.shape[0]), cache_lib.last_valid_index(valid)]
        logits = self.model.PaliGemma.llm(h, method="text_debug_decode")
        return {"hidden": h.astype(jnp.float32), "logits": logits.astype(jnp.float32)}


def is_language_layer(key):
    if key[:3] != ("PaliGemma", "llm", "layers"):
        return False
    return not any(part.endswith("_1") for part in key[3:])


def replace_group(base, libero, name):
    """Shallow copies share arrays; only explicitly selected weights change."""
    reverse = name == "base_inject_libero_language_layers"
    source, target = (libero, base) if reverse else (base, libero)
    result = dict(target)
    changed = []
    if name in {"base", "libero"}:
        return dict(base if name == "base" else libero), changed
    for key in target:
        selected = False
        if name == "libero_restore_base_vision":
            selected = key[:2] == ("PaliGemma", "img")
        elif name == "libero_restore_base_readout":
            selected = key[:3] in {
                ("PaliGemma", "llm", "embedder"),
                ("PaliGemma", "llm", "final_norm"),
            }
        elif name in {"libero_restore_base_language_layers", "base_inject_libero_language_layers"}:
            selected = is_language_layer(key)
        elif name.startswith("libero_restore_base_first_"):
            count = int(name.rsplit("_", 1)[1])
            if is_language_layer(key):
                result[key] = jnp.concatenate([source[key][:count], target[key][count:]], axis=0)
                changed.append({"path": "/".join(key), "layers": [0, count]})
            continue
        else:
            raise ValueError(f"Unknown variant {name}")
        if selected:
            result[key] = source[key]
            changed.append({"path": "/".join(key), "layers": "all"})
    return result, changed


def summarize_logits(logits, sp):
    logits = logits.copy()
    logits[max(0, sp.pad_id())] = -np.inf
    if sp.bos_id() >= 0 and sp.bos_id() != sp.eos_id():
        logits[sp.bos_id()] = -np.inf
    shifted = logits - np.max(logits)
    exp = np.exp(shifted.astype(np.float64))
    probs = exp / exp.sum()
    ids = np.argsort(logits)[-10:][::-1]
    return {
        "top10": [
            {"id": int(i), "piece": sp.id_to_piece(int(i)) if i < sp.get_piece_size() else None,
             "logit": float(logits[i]), "probability": float(probs[i])}
            for i in ids
        ],
        "probability_id_ge_250000": float(probs[250000:].sum()),
    }, probs


def sampled_layer_drift(base, libero):
    profile = {}
    for key in base:
        if not is_language_layer(key):
            continue
        a, b = base[key], libero[key]
        rows = []
        size = int(np.prod(a.shape[1:]))
        indices = np.linspace(0, size - 1, min(size, 16384), dtype=np.int32)
        for layer in range(a.shape[0]):
            av = np.asarray(a[layer].reshape(-1)[indices], dtype=np.float32)
            bv = np.asarray(b[layer].reshape(-1)[indices], dtype=np.float32)
            rms = float(np.sqrt(np.mean(av.astype(np.float64) ** 2)))
            delta = float(np.sqrt(np.mean((av.astype(np.float64) - bv) ** 2)))
            rows.append({"layer": layer, "sample_size": len(indices),
                         "relative_rms_change": delta / max(rms, 1e-30),
                         "bf16_equal_fraction": float(np.mean(av == bv))})
        profile["/".join(key)] = rows
    return profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--libero", required=True)
    p.add_argument("--frame", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--variants", nargs="+", default=[
        "base", "libero", "libero_restore_base_vision", "libero_restore_base_readout",
        "libero_restore_base_language_layers", "base_inject_libero_language_layers",
    ])
    p.add_argument("--prompts", nargs="+", default=["caption", "subtask"])
    args = p.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = config.get_config("pi05_libero")
    print(f"devices={jax.devices()} loading LIBERO policy", flush=True)
    policy = policy_config.create_trained_policy(cfg, args.libero)
    wrapper = text_lib.Pi05TextDebugPolicy(policy, max_tokens=args.max_tokens)
    sp = wrapper._sp
    libero = traverse_util.flatten_dict(nnx.state(policy._model).to_pure_dict())
    with np.load(args.frame, allow_pickle=False) as frame:
        raw = {"observation/" + key: frame[key] for key in ("image", "wrist_image", "state")}
        raw["prompt"] = str(frame["prompt"].item())
    templates = {"caption": "caption en", "answer": "answer en what objects are visible in the image?",
                 "subtask": "Task: {task}\nSubtask:"}
    observations = {}
    for prompt in args.prompts:
        wrapper._template = templates[prompt]
        observations[prompt] = wrapper._prepare_text_observation(raw)
    print("loading base params", flush=True)
    base_all = traverse_util.flatten_dict(model_lib.restore_params(Path(args.base) / "params", dtype=jnp.bfloat16))
    base = {key: base_all[key] for key in libero}
    del base_all
    decoder_graph, _ = nnx.split(wrapper._decoder)
    probe_graph, _ = nnx.split(PrefixProbe(policy._model))

    @jax.jit
    def generate(state, observation):
        return nnx.merge(decoder_graph, state).generate(observation)

    @jax.jit
    def probe(state, observation):
        return nnx.merge(probe_graph, state).run(observation)

    report = {
        "purpose": "same-frame text-only controlled in-memory parameter interventions",
        "frame_sha256": hashlib.sha256(Path(args.frame).read_bytes()).hexdigest(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "base_checkpoint": args.base, "libero_checkpoint": args.libero,
        "jax_version": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "config": repr(cfg.model), "max_tokens": args.max_tokens,
        "task_prompt": raw["prompt"], "actions_executed": False,
        "notes": ["Id>=250000 is an explicit numerical range, not a language quality classifier.",
                  "All results use the same image preprocessing, tokenizer and decoder.",
                  "Base readout restoration also restores the tied input embedding.",
                  "Layer drift is a deterministic sample of up to 16384 values per tensor per layer."],
        "layer_drift": sampled_layer_drift(base, libero), "results": {},
    }
    references = {}
    for name in args.variants:
        print(f"start variant={name}", flush=True)
        flat, replacements = replace_group(base, libero, name)
        model = cfg.model.load(traverse_util.unflatten_dict(flat))
        decoder = text_lib.TextDecoder(model, max_tokens=args.max_tokens, eos_id=sp.eos_id(),
                                      pad_id=sp.pad_id(), bos_id=sp.bos_id())
        _, decoder_state = nnx.split(decoder)
        _, probe_state = nnx.split(PrefixProbe(model))
        item = {"replacements": replacements, "prompts": {}}
        for prompt, (observation, query) in observations.items():
            start = time.perf_counter()
            pref = jax.device_get(probe(probe_state, observation))
            pref_time = time.perf_counter() - start
            logits = np.asarray(pref["logits"])[0]
            hidden = np.asarray(pref["hidden"])[0]
            log_summary, probs = summarize_logits(logits, sp)
            if name == "base":
                references[prompt] = (hidden, probs)
            if prompt in references:
                ref_hidden, ref_probs = references[prompt]
                log_summary["hidden_cosine_vs_base"] = float(
                    np.dot(hidden, ref_hidden) / (np.linalg.norm(hidden) * np.linalg.norm(ref_hidden)))
                mask = ref_probs > 0
                log_summary["kl_base_to_variant_nats"] = float(
                    np.sum(ref_probs[mask] * (np.log(ref_probs[mask]) - np.log(np.maximum(probs[mask], 1e-300)))))
                log_summary["probability_of_base_first_token"] = float(probs[np.argmax(ref_probs)])
            start = time.perf_counter()
            generated = jax.device_get(generate(decoder_state, observation))
            generation_time = time.perf_counter() - start
            ids = np.asarray(generated["ids"])[0, :int(generated["steps"])].tolist()
            text, unmapped, nonlanguage = wrapper._decode_readable(ids)
            item["prompts"][prompt] = {
                "query": query, "raw_ids": ids, "text": text, "text_ascii": ascii(text),
                "unmapped_ids": unmapped, "nonlanguage_ids": nonlanguage,
                "eos_seen": bool(generated["eos_seen"][0]), "prefix_probe_seconds": pref_time,
                "generation_seconds": generation_time, "first_token": log_summary,
            }
            print(json.dumps({"variant": name, "prompt": prompt, "text_ascii": ascii(text),
                              "ids": ids, "seconds": round(pref_time + generation_time, 2)}), flush=True)
        report["results"][name] = item
        output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n")
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
