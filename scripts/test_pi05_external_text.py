"""Protocol and action-input tests; no external model weights are loaded."""

# ruff: noqa: SLF001

import base64
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import io
import json
import threading

import numpy as np
from PIL import Image
import pytest
from serve_paligemma_text import build_prompt
from serve_paligemma_text import decode_request

from openpi.policies.pi05_external_text import PaliGemmaTextClient
from openpi.policies.pi05_external_text import Pi05ExternalTextPolicy


def test_prompt_and_png_request():
    image = Image.new("RGB", (8, 8), (1, 2, 3))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = json.dumps({
        "image_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "task": "pick_up_bowl",
    }).encode()
    decoded, task, digest = decode_request(payload)
    assert decoded.size == image.size
    assert task == "pick_up_bowl"
    assert len(digest) == 64
    assert build_prompt("answer en {task}?\n", task) == "answer en pick up bowl?\n"


def test_http_client_sends_image_and_requires_real_text():
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            captured["path"] = self.path
            captured["image"], captured["task"], _ = decode_request(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            body = json.dumps({
                "source": "external_paligemma_mix", "model": "test-model",
                "text": "reach toward the bowl", "token_ids": [1, 2],
                "high_level_query": "answer en ...", "generation_ms": 3,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = PaliGemmaTextClient(f"http://127.0.0.1:{server.server_port}/generate")
        response = client.generate(np.zeros((8, 8, 3), dtype=np.uint8), "pick up bowl")
        assert response["text"] == "reach toward the bowl"
        assert captured["path"] == "/generate"
        assert captured["task"] == "pick up bowl"
        assert captured["image"].size == (8, 8)
    finally:
        server.shutdown()
        thread.join()


class FakeActionPolicy:
    def __init__(self):
        self.action_prompt = None

    def infer(self, observation, *, noise=None):
        self.action_prompt = observation["prompt"]
        return {"actions": np.zeros((10, 7), dtype=np.float32)}


@pytest.mark.parametrize(("mode", "expected"), [
    ("observe", "pick up the black bowl"),
    ("condition", "reach toward the black bowl"),
])
def test_action_receives_recorded_language_input(mode, expected):
    policy = object.__new__(Pi05ExternalTextPolicy)
    policy._mode = mode
    policy._log_path = None
    policy._policy = FakeActionPolicy()
    plan = {
        "subtask": "reach toward the black bowl",
        "text_unmapped_ids": [], "text_nonlanguage_ids": [],
        "text_invalid_unicode": [],
    }
    result = policy._act({"prompt": "pick up the black bowl"}, plan)
    assert policy._policy.action_prompt == expected
    assert result["action_prompt"] == expected


def test_invalid_generated_text_stops_conditioned_action():
    policy = object.__new__(Pi05ExternalTextPolicy)
    policy._mode = "condition"
    policy._policy = FakeActionPolicy()
    with pytest.raises(ValueError, match="control/private"):
        policy._act({"prompt": "task"}, {"text_invalid_unicode": ["U+E814"]})
    assert policy._policy.action_prompt is None
