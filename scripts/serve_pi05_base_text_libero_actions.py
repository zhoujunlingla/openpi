"""Generate text with pi05_base, then run pi05_libero actions on that text.

The two checkpoints are loaded into separate JAX policies. In condition mode,
each action chunk receives exactly the new text; no fallback is applied.
"""

import argparse
import logging
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
    parser.add_argument("--text-checkpoint", default="gs://openpi-assets/checkpoints/pi05_base")
    parser.add_argument("--action-checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--mode", choices=("observe", "condition"), default="condition")
    parser.add_argument("--max-text-tokens", type=int, default=16)
    parser.add_argument("--prompt-template", default="Task: {task}\nSubtask:")
    parser.add_argument("--num-action-steps", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8022)
    parser.add_argument("--log-jsonl", default="data/libero/pi05_base_to_libero_condition_trace.jsonl")
    args = parser.parse_args()
    if args.num_action_steps < 1:
        parser.error("--num-action-steps must be positive")
    logging.basicConfig(level=logging.INFO)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    print(f"openpi commit: {sha}", flush=True)

    cfg = training_config.get_config(args.config)
    action_path = download.maybe_download(args.action_checkpoint)
    asset_id = cfg.data.assets.asset_id or cfg.data.repo_id
    norm_stats = checkpoints.load_norm_stats(action_path / "assets", asset_id)
    text_policy = policy_config.create_trained_policy(
        cfg,
        args.text_checkpoint,
        norm_stats=norm_stats,
        sample_kwargs={"num_steps": args.num_action_steps},
    )
    action_policy = policy_config.create_trained_policy(
        cfg, action_path, sample_kwargs={"num_steps": args.num_action_steps}
    )
    policy = Pi05TextDebugPolicy(
        text_policy,
        action_policy=action_policy,
        text_source="pi05_base_checkpoint_vlm_probe",
        mode=args.mode,
        max_tokens=args.max_text_tokens,
        prompt_template=args.prompt_template,
        log_jsonl=args.log_jsonl or None,
    )
    print(
        f"Text checkpoint={args.text_checkpoint}; action checkpoint={action_path}; "
        f"mode={args.mode}; log={args.log_jsonl}",
        flush=True,
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
