"""Compare text probes from base and LIBERO π0.5 on exactly one saved observation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from openpi.policies import policy_config
from openpi.policies.pi05_text_debug import Pi05TextDebugPolicy
from openpi.training import checkpoints
from openpi.training import config as training_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.frame, allow_pickle=False) as frame:
        raw_obs = {
            "observation/image": frame["image"],
            "observation/wrist_image": frame["wrist_image"],
            "observation/state": frame["state"],
            "prompt": str(frame["prompt"].item()),
        }
    cfg = training_config.get_config("pi05_libero")
    asset_id = cfg.data.assets.asset_id or cfg.data.repo_id
    stats = checkpoints.load_norm_stats(args.checkpoint_root / "pi05_libero" / "assets", asset_id)
    probes = {
        "official_caption": "caption en",
        "official_answer": "answer en what objects are visible in the image?",
        "subtask": "Task: {task}\nSubtask:",
    }
    result = {
        "purpose": "same-frame base-vs-LIBERO text probe; no action decoder generates text",
        "frame_sha256": hashlib.sha256(args.frame.read_bytes()).hexdigest(),
        "prompt": raw_obs["prompt"],
        "config": repr(cfg.model),
        "results": {},
    }
    for name in ("pi05_base", "pi05_libero"):
        print(f"Loading {name}", flush=True)
        policy = policy_config.create_trained_policy(
            cfg, args.checkpoint_root / name, norm_stats=stats, sample_kwargs={"num_steps": 10}
        )
        variants = {}
        for label, template in probes.items():
            wrapper = Pi05TextDebugPolicy(policy, max_tokens=16, prompt_template=template)
            plan = wrapper._plan(raw_obs)  # noqa: SLF001 - read-only diagnostic stage
            variants[label] = {
                "query_template": template,
                "raw_ids": plan["text_token_ids"],
                "text_ascii": ascii(plan["subtask"]),
                "eos_seen": plan["text_eos_seen"],
                "nonlanguage_ids": plan["text_nonlanguage_ids"],
                "unmapped_ids": plan["text_unmapped_ids"],
            }
            print(f"{name} {label}: {variants[label]['text_ascii']}", flush=True)
        result["results"][name] = variants
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
