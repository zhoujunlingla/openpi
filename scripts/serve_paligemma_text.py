"""Serve a separately trained PaliGemma mix checkpoint as a local text VLM.

Run this process on a different GPU from the pi05 action server. Model access is
subject to the checkpoint's license; this script does not download or substitute
another model when access fails.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import io
import json
import logging
from pathlib import Path
import time

from PIL import Image


def build_prompt(template: str, task: str) -> str:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a nonempty string")
    prompt = template.format(task=task.strip().replace("_", " "))
    if not prompt.strip():
        raise ValueError("prompt template produced an empty prompt")
    return prompt


def decode_request(body: bytes) -> tuple[Image.Image, str, str]:
    data = json.loads(body)
    if not isinstance(data, dict) or not isinstance(data.get("image_png_base64"), str):
        raise ValueError("Request must contain image_png_base64 and task")
    image_bytes = base64.b64decode(data["image_png_base64"], validate=True)
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    if image.width > 1024 or image.height > 1024:
        raise ValueError("Image is too large for this text service")
    return image.convert("RGB"), data.get("task"), hashlib.sha256(image_bytes).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/paligemma-3b-mix-224")
    parser.add_argument("--revision", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--prompt-template",
        default="answer en what should the robot do next to complete this task: {task}?\n",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1 or not 1 <= args.port <= 65535:
        parser.error("Invalid max-new-tokens or port")
    logging.basicConfig(level=logging.INFO)
    import torch
    from transformers import AutoProcessor
    from transformers import PaliGemmaForConditionalGeneration

    revision = None if Path(args.model).is_dir() else args.revision or None
    logging.info("Loading text model %s revision %s on %s", args.model, revision, args.device)
    processor = AutoProcessor.from_pretrained(args.model)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model, revision=revision, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    eos_id = processor.tokenizer.eos_token_id

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path != "/generate":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 5_000_000:
                    raise ValueError("Request body must be between 1 and 5 MB")
                image, task, image_sha256 = decode_request(self.rfile.read(size))
                prompt = build_prompt(args.prompt_template, task)
                start = time.perf_counter()
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
                input_length = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                ids = output[0][input_length:].detach().cpu().tolist()
                text = processor.decode(ids, skip_special_tokens=True).strip()
                eos_seen = eos_id in ids if eos_id is not None else False
                result = {
                    "source": "external_paligemma_mix",
                    "model": args.model,
                    "revision": revision,
                    "high_level_query": prompt,
                    "text": text,
                    "token_ids": ids,
                    "eos_seen": eos_seen,
                    "hit_token_limit": len(ids) >= args.max_new_tokens and not eos_seen,
                    "generation_ms": (time.perf_counter() - start) * 1000,
                    "image_sha256": image_sha256,
                }
                if not text:
                    raise ValueError("PaliGemma generated empty text")
                response = json.dumps(result, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
            except Exception as exc:
                logging.exception("Text generation request failed")
                response = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
                self.send_response(422)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = HTTPServer((args.host, args.port), Handler)
    logging.info("PaliGemma text service listening on %s:%d", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
