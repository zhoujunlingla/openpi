"""Diagnose, not repair, the same-checkpoint text probe at openpi 3d8b911.

Run from the openpi root using its existing JAX environment. This adds no model
weights, never feeds diagnostic/teacher-forced text to Action Expert, and audits
only the first observation. The existing observe-mode policy remains in use.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
from pathlib import Path
import subprocess
import time
import traceback
import unicodedata

import numpy as np

EXPECTED_COMMIT = "3d8b911ee2dbd657789e50ddc5a5c9fdb86f07fd"


def reference_layout(prefix_valid, count, capacity: int, xp=np):
    """Independent prefix-LM layout; supports holes in the image/prompt prefix.

    Prefix queries cannot see ANY generated token. Suffix queries see valid
    prefix tokens and only past/self suffix tokens. Unused slots stay masked.
    xp can be numpy or jax.numpy; count may be a JAX scalar.
    """
    batch, plen = prefix_valid.shape
    suffix_valid = xp.broadcast_to(xp.arange(capacity)[None, :] < count, (batch, capacity))
    valid = xp.concatenate((prefix_valid, suffix_valid), axis=1)
    q = xp.arange(plen + capacity)[:, None]
    k = xp.arange(plen + capacity)[None, :]
    structure = ((q < plen) & (k < plen)) | ((q >= plen) & ((k < plen) | (k <= q)))
    mask = structure[None, :, :] & valid[:, :, None] & valid[:, None, :]
    positions = xp.cumsum(valid.astype(xp.int32), axis=1) - 1
    last_prefix = xp.max(xp.where(prefix_valid, xp.arange(plen)[None, :], -1), axis=1)
    last = xp.where(count > 0, plen + count - 1, last_prefix)
    return mask, positions, last


def allowed_logits(logits, banned_ids):
    values = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite raw logits: diagnose model/numerics before interpreting words")
    for token in banned_ids:
        if 0 <= token < values.size:
            values[token] = -np.inf
    if not np.isfinite(values).any():
        raise ValueError("No allowed vocabulary entry")
    return values


def probabilities(values):
    weights = np.exp(values - np.max(values))
    return weights / weights.sum()


def compare_logits(cached, reference, banned_ids):
    """Report numerical evidence, not a hard-coded 'checkpoint good/bad' verdict."""
    a = np.asarray(cached, dtype=np.float64).reshape(-1)
    b = np.asarray(reference, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Vocabulary shapes differ: {a.shape} vs {b.shape}")
    aa, bb = allowed_logits(a, banned_ids), allowed_logits(b, banned_ids)
    pa, pb = probabilities(aa), probabilities(bb)
    best = np.argsort(bb)[-2:][::-1]
    difference = a - b
    return {
        "raw_max_abs_error": float(np.max(np.abs(difference))),
        "raw_rms_error": float(np.sqrt(np.mean(difference**2))),
        "centered_max_abs_error": float(np.max(np.abs(difference - difference.mean()))),
        "probability_total_variation": float(0.5 * np.abs(pa - pb).sum()),
        "cached_top1_id": int(np.argmax(aa)),
        "reference_top1_id": int(np.argmax(bb)),
        "top1_equal": bool(np.argmax(aa) == np.argmax(bb)),
        "reference_top1_margin": float(bb[best[0]] - bb[best[1]]),
    }


def describe_token(sp, token: int):
    result = {"id": int(token)}
    if not 0 <= token < sp.get_piece_size():
        return {**result, "unmapped": True}
    piece = sp.id_to_piece(int(token))
    result.update(
        piece=piece,
        decoded_alone=sp.decode([int(token)]),
        piece_ascii=ascii(piece),
        unicode_categories=sorted({unicodedata.category(c) for c in piece}),
        is_control=bool(sp.is_control(int(token))),
        is_eos=token == sp.eos_id(),
    )
    return result


def top_tokens(logits, sp, banned_ids, count=10):
    vals = allowed_logits(logits, banned_ids)
    probs = probabilities(vals)
    indices = np.argsort(vals)[-min(count, vals.size):][::-1]
    return [
        {**describe_token(sp, int(i)), "logit": float(vals[i]), "probability": float(probs[i])}
        for i in indices if np.isfinite(vals[i])
    ]


def write_report(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    # ASCII escaping keeps even pathological generated Unicode safely representable.
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def build_policy(base, args, runtime):
    # Delayed imports: --help and independent tests need no openpi/Flax install.
    from flax import nnx
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib
    from openpi.models import pi0
    from openpi.policies import pi05_text_cache as cache_lib
    from openpi.policies.pi05_text_debug import Pi05TextDebugPolicy
    from openpi.shared import nnx_utils

    class AuditKernel(nnx.Module):
        def __init__(self, model, cache_capacity):
            self.model = model
            self.cache_capacity = cache_capacity

        def prefix(self, observation):
            obs = model_lib.preprocess_observation(None, observation, train=False)
            emb, valid, ar = self.model.embed_prefix(obs)
            hidden, cache = self.model.PaliGemma.llm(
                [emb, None], positions=jnp.cumsum(valid.astype(jnp.int32), axis=1) - 1,
                mask=pi0.make_attn_mask(valid, ar),
            )
            last = cache_lib.last_valid_index(valid)
            logits = self.model.PaliGemma.llm(
                hidden[0][jnp.arange(valid.shape[0]), last], method="text_debug_decode"
            ).astype(jnp.float32)
            return emb, valid, ar, logits, cache

        def cached_step(self, token, cache, valid, index):
            emb = self.model.PaliGemma.llm(token[:, None], method="embed")
            hidden, appended = self.model.PaliGemma.llm(
                [emb, None], positions=(valid.sum(axis=1).astype(jnp.int32) + index)[:, None],
                mask=cache_lib.next_token_mask(valid, index, self.cache_capacity), kv_cache=cache,
            )
            compact = cache_lib.compact_cache(appended, valid.shape[1], index)
            logits = self.model.PaliGemma.llm(hidden[0][:, 0], method="text_debug_decode")
            return logits.astype(jnp.float32), compact

        def no_cache(self, emb, valid, history, count):
            # Recompute the entire prefix and causal suffix, WITHOUT kv_cache.
            suffix = self.model.PaliGemma.llm(history, method="embed")
            mask, positions, last = reference_layout(valid, count, history.shape[1], xp=jnp)
            hidden, _ = self.model.PaliGemma.llm(
                [jnp.concatenate((emb, suffix), axis=1), None], positions=positions, mask=mask,
            )
            return self.model.PaliGemma.llm(
                hidden[0][jnp.arange(valid.shape[0]), last], method="text_debug_decode"
            ).astype(jnp.float32)

    class AuditPolicy(Pi05TextDebugPolicy):
        def __init__(self):
            super().__init__(
                base, mode="observe", max_tokens=args.max_text_tokens,
                prompt_template=args.prompt_template, log_jsonl=args.trace_jsonl or None,
            )
            self._audit_done = False
            self._audit_kernel = AuditKernel(base._model, args.max_text_tokens)
            self._audit_prefix = nnx_utils.module_jit(self._audit_kernel.prefix)
            self._audit_step = nnx_utils.module_jit(self._audit_kernel.cached_step)
            self._audit_reference = nnx_utils.module_jit(self._audit_kernel.no_cache)

        def _plan(self, raw_obs):
            plan = super()._plan(raw_obs)
            if self._audit_done:
                return plan
            self._audit_done = True
            report = {
                "purpose": "numerical/tokenizer diagnosis, NOT a restored subtask planner",
                "runtime": runtime, "production_plan": plan,
                "teacher_text_is_diagnostic_only": args.teacher_text,
                "status": "started", "comparisons": [],
            }
            write_report(args.report, report)
            try:
                observation, query = self._prepare_text_observation(raw_obs)
                emb, valid, ar, current, cache = self._audit_prefix(observation)
                valid_np = np.asarray(jax.device_get(valid))
                if valid_np.shape[0] != 1 or not valid_np.any():
                    raise ValueError("Audit expects one nonempty LIBERO observation")
                if np.asarray(jax.device_get(ar)).any():
                    raise ValueError("This independent reference expects a bidirectional input prefix")
                sp = self._sp
                pad_id, bos_id, eos_id = max(0, sp.pad_id()), sp.bos_id(), sp.eos_id()
                banned = [pad_id] + ([bos_id] if bos_id >= 0 and bos_id != eos_id else [])
                first = np.asarray(jax.device_get(current))[0]
                production_ids = list(plan["text_token_ids"])
                teacher_ids = list(sp.encode(args.teacher_text))[:args.steps]
                if not teacher_ids:
                    raise ValueError("Diagnostic teacher text must encode at least one token")
                history = jnp.full((1, len(teacher_ids)), pad_id, dtype=jnp.int32)
                report.update(
                    high_level_query=query,
                    sp_vocab_size=int(sp.get_piece_size()), model_vocab_size=int(first.size),
                    tokenizer_special_ids={"pad": int(sp.pad_id()), "bos": int(bos_id), "eos": int(eos_id)},
                    prefix_physical_length=int(valid_np.shape[1]), prefix_valid_count=int(valid_np.sum()),
                    last_valid_physical_index=int(np.flatnonzero(valid_np[0])[-1]),
                    production_token_details=[describe_token(sp, i) for i in production_ids],
                    first_token_top10_before_any_cache_update=top_tokens(first, sp, banned),
                    production_first_id_matches_independent_prefill=(
                        bool(production_ids[0] == np.argmax(allowed_logits(first, banned)))
                        if production_ids else None
                    ),
                    teacher_forced_ids=teacher_ids,
                    teacher_forced_token_details=[describe_token(sp, i) for i in teacher_ids],
                )
                # Same original capacity and the same repository cache helpers.
                cache = cache_lib.pad_cache(cache, args.max_text_tokens)
                for count in range(len(teacher_ids) + 1):
                    reference = self._audit_reference(emb, valid, history, jnp.asarray(count, jnp.int32))
                    a = np.asarray(jax.device_get(current))[0]
                    b = np.asarray(jax.device_get(reference))[0]
                    metrics = compare_logits(a, b, banned)
                    metrics.update(
                        teacher_tokens_already_fed=count,
                        cached_top5=top_tokens(a, sp, banned, 5),
                        reference_top5=top_tokens(b, sp, banned, 5),
                    )
                    report["comparisons"].append(metrics)
                    write_report(args.report, report)
                    print(f"[AUDIT] history={count} top1_equal={metrics['top1_equal']} "
                          f"TV={metrics['probability_total_variation']:.6g} "
                          f"max_abs={metrics['raw_max_abs_error']:.6g}", flush=True)
                    if count < len(teacher_ids):
                        token = jnp.asarray([teacher_ids[count]], dtype=jnp.int32)
                        history = history.at[:, count].set(token)
                        current, cache = self._audit_step(token, cache, valid, jnp.asarray(count, jnp.int32))
                report["status"] = "completed"
                report["interpretation_warning"] = (
                    "Agreement only checks numerical decoding for this observation/history. "
                    "It does not validate the prompt, vocabulary semantics, training, or subtask ability. "
                    "BF16 kernel differences and small top1 margins can cause harmless argmax flips."
                )
            except Exception:
                report["status"] = "error"
                report["traceback"] = traceback.format_exc()
                write_report(args.report, report)
                raise  # Do not hide an incomplete diagnostic by executing actions.
            write_report(args.report, report)
            print(f"[AUDIT REPORT] {args.report.resolve()}", flush=True)
            return plan

    return AuditPolicy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--report", type=Path, default=Path("data/libero/pi05_text_audit.json"))
    parser.add_argument("--trace-jsonl", default="data/libero/pi05_text_audit_trace.jsonl")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--teacher-text", default="pick up the black bowl")
    parser.add_argument("--max-text-tokens", type=int, default=24)
    parser.add_argument("--prompt-template", default="Task: {task}\nSubtask:")
    parser.add_argument("--num-action-steps", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.steps <= args.max_text_tokens <= 128:
        parser.error("Require 1 <= steps <= max-text-tokens <= 128")
    if args.num_action_steps < 1 or not 1 <= args.port <= 65535:
        parser.error("Invalid action steps or port")
    if args.report.exists():
        parser.error(f"Report already exists: {args.report}. Choose another --report path.")
    logging.basicConfig(level=logging.INFO)
    from openpi.models import gemma, pi0
    from openpi.policies import pi05_text_debug, pi05_text_cache, policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as training_config

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    cfg = training_config.get_config(args.config)
    runtime = {
        "expected_commit": EXPECTED_COMMIT, "runtime_git_commit": sha,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": args.checkpoint, "config_name": args.config, "model_config": repr(cfg.model),
        "import_paths": {m.__name__: inspect.getfile(m) for m in (gemma, pi0, pi05_text_debug, pi05_text_cache)},
        "timestamp_unix": time.time(),
    }
    print(json.dumps(runtime, indent=2, ensure_ascii=False), flush=True)
    if sha != EXPECTED_COMMIT and subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_COMMIT, "HEAD"], check=False
    ).returncode != 0:
        logging.warning("HEAD is not based on the audited commit; confirm source compatibility")
    base = policy_config.create_trained_policy(
        cfg, args.checkpoint, sample_kwargs={"num_steps": args.num_action_steps}
    )
    policy = build_policy(base, args, runtime)
    print("Audit first observation only; Action Expert stays in OBSERVE mode.", flush=True)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
