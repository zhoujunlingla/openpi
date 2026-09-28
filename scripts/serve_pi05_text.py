"""Start the experimental JAX pi05 text-debug server from the openpi root."""
import argparse
import logging
from pathlib import Path
import subprocess

from openpi.policies import policy_config
from openpi.policies.pi05_text_debug import Pi05TextDebugPolicy
from openpi.serving import websocket_policy_server
from openpi.shared import download
from openpi.training import checkpoints
from openpi.training import config as training_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument(
        "--norm-stats-checkpoint",
        default=None,
        help="Read dataset normalization statistics from this checkpoint (useful for pi05_base on LIBERO)",
    )
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
    norm_stats = None
    if args.norm_stats_checkpoint:
        stats_path = download.maybe_download(args.norm_stats_checkpoint)
        asset_id = cfg.data.assets.asset_id or cfg.data.repo_id
        norm_stats = checkpoints.load_norm_stats(stats_path / "assets", asset_id)
    print(f"Using model config: {cfg.model}", flush=True)
    base = policy_config.create_trained_policy(
        cfg, args.checkpoint, sample_kwargs={"num_steps": args.num_action_steps}, norm_stats=norm_stats
    )
    wrapped = Pi05TextDebugPolicy(
        base,
        mode=args.mode,
        max_tokens=args.max_text_tokens,
        prompt_template=args.prompt_template,
        log_jsonl=args.log_jsonl or None,
    )
    print("Text output is an unvalidated probe of this checkpoint, not a restored official planner.", flush=True)
    print(f"Model checkpoint={args.checkpoint}; norm stats checkpoint={args.norm_stats_checkpoint or args.checkpoint}", flush=True)
    print(f"Listening on {args.host}:{args.port}; trace={Path(args.log_jsonl).resolve()}", flush=True)
    websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host=args.host, port=args.port, metadata=wrapped.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
