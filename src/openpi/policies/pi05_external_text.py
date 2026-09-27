"""Attach an independently trained text VLM to the pi05 action policy.

The VLM and action policy have separate checkpoints. The VLM is called once per
action chunk; observe mode keeps the original task as the action input.
"""

from __future__ import annotations

import base64
import io
import json
import time
import unicodedata
import urllib.error
import urllib.request
import uuid

import numpy as np
from PIL import Image

from openpi.policies.pi05_text_debug import Pi05TextDebugPolicy


class PaliGemmaTextClient:
    def __init__(self, url: str, *, timeout: float = 120.0):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.url = url
        self.timeout = timeout

    def generate(self, image: np.ndarray, task: str) -> dict:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("observation/image must be an RGB uint8 image")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("prompt must be a nonempty string")
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        payload = json.dumps({
            "image_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "task": task,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read(65536))
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"PaliGemma text service returned HTTP {exc.code}: {detail}") from exc
        if result.get("source") != "external_paligemma_mix":
            raise ValueError("Unexpected text source from PaliGemma service")
        if not isinstance(result.get("text"), str) or not result["text"].strip():
            raise ValueError("PaliGemma returned empty text; action inference was not started")
        if not isinstance(result.get("token_ids"), list) or not all(
            isinstance(token, int) and token >= 0 for token in result["token_ids"]
        ):
            raise ValueError("PaliGemma response has invalid token_ids")
        if not isinstance(result.get("model"), str) or not result["model"]:
            raise ValueError("PaliGemma response has no model identity")
        return result


class Pi05ExternalTextPolicy(Pi05TextDebugPolicy):
    """Use external PaliGemma text at every plan while retaining pi05 actions."""

    def __init__(self, policy, *, text_url: str, mode: str = "observe", log_jsonl: str | None = None,
                 timeout: float = 120.0):
        # The parent supplies the tested plan/act handshake and action logging.
        # Its native text decoder is constructed but never called here.
        super().__init__(policy, mode=mode, log_jsonl=log_jsonl)
        self._text_client = PaliGemmaTextClient(text_url, timeout=timeout)

    @property
    def metadata(self):
        metadata = super().metadata
        metadata["text_debug"].pop("max_tokens", None)
        metadata["text_debug"].pop("prompt_template", None)
        metadata["text_debug"].update(
            source="external_paligemma_mix",
            text_service_url=self._text_client.url,
            action_checkpoint="pi05_jax_checkpoint",
        )
        return metadata

    def _plan(self, raw_obs):
        start = time.perf_counter()
        response = self._text_client.generate(raw_obs["observation/image"], raw_obs["prompt"])
        text = response["text"].strip()
        invalid_chars = [
            f"U+{ord(char):04X}" for char in text
            if unicodedata.category(char) in {"Cc", "Cf", "Co", "Cs"}
        ]
        result = {
            "plan_id": uuid.uuid4().hex,
            "original_prompt": raw_obs["prompt"],
            "subtask": text,
            "text_token_ids": response["token_ids"],
            "text_unmapped_ids": [],
            "text_nonlanguage_ids": [],
            "text_invalid_unicode": invalid_chars,
            "text_eos_seen": bool(response.get("eos_seen", False)),
            "text_hit_token_limit": bool(response.get("hit_token_limit", False)),
            "text_generation_ms": (time.perf_counter() - start) * 1000,
            "vlm_generation_ms": response.get("generation_ms"),
            "high_level_query": response["high_level_query"],
            "text_mode": self._mode,
            "text_source": "external_paligemma_mix",
            "vlm_model": response["model"],
            "vlm_revision": response.get("revision"),
            "vlm_image_sha256": response.get("image_sha256"),
        }
        print(
            f"[VLM BEFORE ACTION] source={result['text_source']} mode={self._mode} "
            f"id={result['plan_id']} text={text!r}", flush=True
        )
        print(f"[VLM RAW IDS] {result['text_token_ids']}", flush=True)
        return result

    def _act(self, raw_obs, plan, noise=None):
        if self._mode == "condition" and plan["text_invalid_unicode"]:
            raise ValueError("PaliGemma text contains control/private Unicode; action inference was not started")
        return super()._act(raw_obs, plan, noise=noise)
