"""Experimental pi05 text probe and two-pass action wrapper.

Not an official high-level pi05 implementation and not a guarantee that a released
checkpoint retains useful language generation. Supported backend: openpi JAX.
Run install.py first to expose Gemma's existing tied vocabulary projection.
An optional second pi05 policy can consume the text as an action prompt.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import json
from pathlib import Path
import time
import unicodedata
import uuid
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy

from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.models import tokenizer as tokenizer_lib
from openpi.policies import pi05_text_cache as cache_lib
from openpi.shared import nnx_utils


class TextDecoder(nnx.Module):
    """Greedy autoregression through the existing VLM, never through Action Expert."""

    def __init__(self, model, *, max_tokens: int, eos_id: int, pad_id: int, bos_id: int):
        if max_tokens < 1 or max_tokens > 128:
            raise ValueError("For this debug implementation use 1 <= max_tokens <= 128")
        if not getattr(model, "pi05", False):
            raise ValueError("This wrapper requires a JAX Pi0 model with pi05=True")
        if eos_id < 0:
            raise ValueError("The tokenizer must define an EOS token")
        self.model = model  # shares the existing model, no second checkpoint load
        self.max_tokens = max_tokens
        self.eos_id = eos_id
        self.pad_id = max(0, pad_id)
        self.bos_id = bos_id

    def generate(self, observation):
        observation = model_lib.preprocess_observation(None, observation, train=False)
        embedded, valid, ar = self.model.embed_prefix(observation)
        positions = jnp.cumsum(valid.astype(jnp.int32), axis=1) - 1
        hidden, cache = self.model.PaliGemma.llm(
            [embedded, None],
            positions=positions,
            mask=pi0.make_attn_mask(valid, ar),
        )
        batch, prefix_length = valid.shape
        last = cache_lib.last_valid_index(valid)
        h = hidden[0][jnp.arange(batch), last]
        logits = self.model.PaliGemma.llm(h, method="text_debug_decode").astype(jnp.float32)
        cache = cache_lib.pad_cache(cache, self.max_tokens)
        generated = jnp.full((batch, self.max_tokens), self.pad_id, dtype=jnp.int32)
        finished = jnp.zeros((batch,), dtype=jnp.bool_)
        prefix_counts = valid.sum(axis=1).astype(jnp.int32)

        def continue_decode(carry):
            index, _, _, _, done = carry
            return (index < self.max_tokens) & ~jnp.all(done)

        def decode_one(carry):
            index, current_logits, current_cache, ids, done = carry
            # Suppress only structural padding/BOS, not words or action vocabulary.
            current_logits = current_logits.at[:, self.pad_id].set(-jnp.inf)
            if self.bos_id >= 0 and self.bos_id != self.eos_id:
                current_logits = current_logits.at[:, self.bos_id].set(-jnp.inf)
            token = jnp.argmax(current_logits, axis=-1).astype(jnp.int32)
            token = jnp.where(done, self.pad_id, token)
            ids = ids.at[:, index].set(token)
            done = done | (token == self.eos_id)
            token_emb = self.model.PaliGemma.llm(token[:, None], method="embed")
            next_hidden, appended = self.model.PaliGemma.llm(
                [token_emb, None],
                positions=(prefix_counts + index)[:, None],
                mask=cache_lib.next_token_mask(valid, index, self.max_tokens),
                kv_cache=current_cache,
            )
            new_cache = cache_lib.compact_cache(appended, prefix_length, index)
            next_logits = self.model.PaliGemma.llm(
                next_hidden[0][:, 0, :], method="text_debug_decode"
            ).astype(jnp.float32)
            return index + 1, next_logits, new_cache, ids, done

        length, _, _, ids, stopped = jax.lax.while_loop(
            continue_decode,
            decode_one,
            (jnp.asarray(0, dtype=jnp.int32), logits, cache, generated, finished),
        )
        return {"ids": ids, "steps": length, "eos_seen": stopped}


class Pi05TextDebugPolicy(base_policy.BasePolicy):
    """Modes:

    observe: text is a probe; action still uses the original instruction.
    condition: generated text replaces the low-level instruction.

    One-call: infer(raw_obs), server prints BEFORE starting action inference.
    Two-call: infer({...raw_obs, '_text_stage':'plan'}) -> plan_id;
              infer({'_text_stage':'act', '_text_plan_id':plan_id}) -> actions.
    Two-call mode retains the EXACT observation from plan, single-use, bounded.
    Do not advance the simulator between the plan and act calls.
    """

    def __init__(
        self,
        policy,
        *,
        mode: str = "observe",
        max_tokens: int = 24,
        prompt_template: str = "Task: {task}\nSubtask:",
        log_jsonl: str | None = None,
        action_policy=None,
        text_source: str = "same_pi05_checkpoint_vlm_probe",
    ):
        if mode not in {"observe", "condition"}:
            raise ValueError("mode must be observe or condition")
        if getattr(policy, "_is_pytorch_model", True):
            raise ValueError("JAX checkpoints only; PyTorch needs a separate implementation")
        if not getattr(policy._model, "pi05", False):
            raise ValueError("Use config pi05_libero and a compatible pi05 checkpoint")
        action_policy = action_policy or policy
        if getattr(action_policy, "_is_pytorch_model", True) or not getattr(action_policy._model, "pi05", False):
            raise ValueError("Action policy must also be a JAX pi05 policy")
        self._policy = policy
        self._action_policy = action_policy
        self._text_source = text_source
        self._mode = mode
        self._template = prompt_template
        self._max_tokens = max_tokens
        self._max_input_tokens = policy._model.max_token_len
        # Use the exact tokenizer distributed with openpi, not another Gemma tokenizer.
        tok = tokenizer_lib.PaligemmaTokenizer(self._max_input_tokens)
        self._sp = tok._tokenizer
        self._decoder = TextDecoder(
            policy._model,
            max_tokens=max_tokens,
            eos_id=self._sp.eos_id(),
            pad_id=self._sp.pad_id(),
            bos_id=self._sp.bos_id(),
        )
        self._decode = nnx_utils.module_jit(self._decoder.generate)
        self._pending = OrderedDict()
        self._log_path = Path(log_jsonl) if log_jsonl else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def metadata(self):
        return {
            **self._action_policy.metadata,
            "text_debug": {
                "mode": self._mode,
                "source": self._text_source,
                "max_tokens": self._max_tokens,
                "prompt_template": self._template,
                "two_phase_available": True,
            },
        }

    def _prepare_text_observation(self, raw_obs: dict):
        task = raw_obs.get("prompt")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Provide the original task as a nonempty string in obs['prompt']")
        # Retain official image/state preprocessing but independently encode the HL query.
        # For pi05_libero the official discrete_state_input flag is False. Do NOT
        # silently turn it on or invent additional state tokens for this checkpoint.
        data = self._policy._input_transform(copy.deepcopy(raw_obs))
        query = self._template.format(task=task.strip().replace("_", " "))
        token_ids = self._sp.encode(query, add_bos=True) + self._sp.encode("\n")
        if len(token_ids) > self._max_input_tokens:
            raise ValueError(
                f"HL prompt has {len(token_ids)} tokens but limit is {self._max_input_tokens}; "
                "shorten the template. Refusing silent truncation."
            )
        pad_id = max(0, self._sp.pad_id())
        padded = np.full(self._max_input_tokens, pad_id, dtype=np.int32)
        valid = np.zeros(self._max_input_tokens, dtype=np.bool_)
        padded[: len(token_ids)] = token_ids
        valid[: len(token_ids)] = True
        data["tokenized_prompt"] = padded
        data["tokenized_prompt_mask"] = valid
        data.pop("token_ar_mask", None)
        data.pop("token_loss_mask", None)
        batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], data)
        return model_lib.Observation.from_dict(batched), query

    def _decode_readable(self, ids: list[int]):
        """Preserve unrepresentable IDs explicitly rather than hiding them."""
        content = []
        for token in ids:
            if token == self._sp.eos_id():
                break
            content.append(token)
        segments, segment, unmapped, nonlanguage = [], [], [], []
        for token in content:
            if 0 <= token < self._sp.get_piece_size():
                piece = self._sp.id_to_piece(token)
                if (
                    self._sp.is_control(token)
                    or (piece.startswith("<") and piece.endswith(">"))
                    or any(unicodedata.category(char) in {"Cc", "Cf", "Co", "Cs"} for char in piece)
                ):
                    nonlanguage.append(token)
                segment.append(token)
            else:
                if segment:
                    segments.append(self._sp.decode(segment))
                    segment = []
                segments.append(f"<UNMAPPED_TOKEN_{token}>")
                unmapped.append(token)
        if segment:
            segments.append(self._sp.decode(segment))
        return "".join(segments).strip(), unmapped, nonlanguage

    def _plan(self, raw_obs):
        observation, query = self._prepare_text_observation(raw_obs)
        start = time.perf_counter()
        output = jax.device_get(self._decode(observation))  # synchronizes before timing/printing
        n = int(output["steps"])
        ids = np.asarray(output["ids"])[0, :n].astype(np.int32).tolist()
        text, unmapped, nonlanguage = self._decode_readable(ids)
        result = {
            "plan_id": uuid.uuid4().hex,
            "original_prompt": raw_obs["prompt"],
            "subtask": text,
            "text_token_ids": ids,
            "text_unmapped_ids": unmapped,
            "text_nonlanguage_ids": nonlanguage,
            "text_eos_seen": bool(output["eos_seen"][0]),
            "text_hit_token_limit": n == self._max_tokens and not bool(output["eos_seen"][0]),
            "text_generation_ms": (time.perf_counter() - start) * 1000,
            "high_level_query": query,
            "text_mode": self._mode,
            "text_source": self._text_source,
        }
        print(f"[VLM BEFORE ACTION] mode={self._mode} id={result['plan_id']} text={text!r}", flush=True)
        print(f"[VLM RAW IDS] {ids}", flush=True)
        return result

    def _act(self, raw_obs, plan, noise=None):
        if self._mode == "condition":
            if not plan["subtask"] or plan["text_unmapped_ids"] or plan["text_nonlanguage_ids"]:
                raise ValueError(
                    "VLM produced empty/unmapped/nonlanguage text. No fallback was executed. "
                    "Use observe mode to inspect raw tokens and prompt/checkpoint compatibility."
                )
            action_prompt = plan["subtask"]
        else:
            action_prompt = raw_obs["prompt"]
        action_obs = copy.deepcopy(raw_obs)
        action_obs["prompt"] = action_prompt
        print(f"[ACTION START] conditioned_on={action_prompt!r}", flush=True)
        start = time.perf_counter()
        result = self._action_policy.infer(action_obs, noise=noise)
        # Add strings ONLY AFTER the existing numpy/JAX/output-transform pipeline.
        result.update(plan)
        result["action_prompt"] = action_prompt
        result["action_wall_ms"] = (time.perf_counter() - start) * 1000
        if self._log_path:
            record = {k: v for k, v in result.items() if k not in {"state", "actions"}}
            record["actions"] = np.asarray(result["actions"]).tolist()
            record["timestamp_unix"] = time.time()
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return result

    def infer(self, obs: dict, *, noise=None) -> dict[str, Any]:
        stage = obs.get("_text_stage", "both")
        if stage == "act":
            key = obs.get("_text_plan_id")
            if key not in self._pending:
                raise ValueError("Unknown/expired/already consumed plan_id")
            saved_obs, plan = self._pending.pop(key)
            return self._act(saved_obs, plan, noise=noise)
        if stage not in {"plan", "both"}:
            raise ValueError("_text_stage must be plan, act or both")
        raw_obs = copy.deepcopy({k: v for k, v in obs.items() if not k.startswith("_text_")})
        plan = self._plan(raw_obs)
        if stage == "plan":
            if len(self._pending) >= 32:
                self._pending.popitem(last=False)
            self._pending[plan["plan_id"]] = (raw_obs, plan)
            return plan
        return self._act(raw_obs, plan, noise=noise)

    def reset(self):
        self._pending.clear()
        self._policy.reset()
        if self._action_policy is not self._policy:
            self._action_policy.reset()
