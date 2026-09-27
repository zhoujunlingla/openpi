"""Serve pi05 actions with a separate PaliGemma mix text generator.

Start scripts/serve_paligemma_text.py first. The text service runs on its own
GPU; this JAX action server calls it once per action chunk before inference.
"""

import argparse
import logging
import subprocess

from openpi.policies import policy_config
from openpi.policies.pi05_external_text import Pi05ExternalTextPolicy
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--text-service-url", default="http://127.0.0.1:8020/generate")
    parser.add_argument("--text-timeout", type=float, default=120.0)
    parser.add_argument("--mode", choices=("observe", "condition"), default="observe")
    parser.add_argument("--num-action-steps", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--log-jsonl", default="data/libero/pi05_external_text_trace.jsonl")
    args = parser.parse_args()
    if args.num_action_steps < 1:
        parser.error("num-action-steps must be positive")
    logging.basicConfig(level=logging.INFO)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    print(f"openpi commit: {sha}", flush=True)
    cfg = training_config.get_config(args.config)
    base = policy_config.create_trained_policy(
        cfg, args.checkpoint, sample_kwargs={"num_steps": args.num_action_steps}
    )
    policy = Pi05ExternalTextPolicy(
        base, text_url=args.text_service_url, mode=args.mode,
        log_jsonl=args.log_jsonl or None, timeout=args.text_timeout,
    )
    print(
        f"pi05 actions use checkpoint={args.checkpoint}; VLM text uses {args.text_service_url}; "
        f"mode={args.mode}", flush=True
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
