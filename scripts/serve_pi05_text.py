"""Start the experimental JAX pi05 text-debug server from the openpi root."""
import argparse
import logging
from pathlib import Path
import subprocess

from openpi.policies import policy_config
from openpi.policies.pi05_text_debug import Pi05TextDebugPolicy
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--mode", choices=("observe", "condition"), default="observe")
    parser.add_argument("--max-text-tokens", type=int, default=24)
    parser.add_argument("--prompt-template", default="Task: {task}\nSubtask:")
    parser.add_argument("--num-action-steps", type=int, default=10)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--log-jsonl", default="data/libero/pi05_text_trace.jsonl")
    args = parser.parse_args()
    if args.num_action_steps < 1:
        parser.error("num-action-steps must be positive")
    logging.basicConfig(level=logging.INFO)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        print(f"openpi commit: {sha}", flush=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("openpi commit: unknown (not running in a Git checkout)", flush=True)
    cfg = training_config.get_config(args.config)
    print(f"Using model config: {cfg.model}", flush=True)
    base = policy_config.create_trained_policy(
        cfg, args.checkpoint, sample_kwargs={"num_steps": args.num_action_steps}
    )
    wrapped = Pi05TextDebugPolicy(
        base,
        mode=args.mode,
        max_tokens=args.max_text_tokens,
        prompt_template=args.prompt_template,
        log_jsonl=args.log_jsonl or None,
    )
    print("Text output is an unvalidated probe of this checkpoint, not a restored official planner.", flush=True)
    print(f"Listening on {args.host}:{args.port}; trace={Path(args.log_jsonl).resolve()}", flush=True)
    websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host=args.host, port=args.port, metadata=wrapped.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
