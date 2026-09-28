"""Probe readable VLM tokens from a LeRobot π0.5 checkpoint on one saved LIBERO frame.

This is an experimental Gemma vocabulary projection. The released policy API returns
continuous actions and does not expose an official subtask predictor.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import unicodedata

import numpy as np
import sentencepiece as spm
import torch
from safetensors import safe_open

from lerobot.configs import PreTrainedConfig
from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
from lerobot.policies.pi05 import PI05Policy


REPO_ID = "lerobot/pi05_libero_base"
REVISION = "a217bfd3b14673cf2ce597e69997ab21866438dd"
SHA256 = "21b8711787c4a75861b02cff6aa81675a3a943d32b435a68262ac4461e476ba4"
VLM_Q_PREFIX = "paligemma_with_expert.paligemma.model.language_model.layers"


def readable(sp, ids):
    parts = []
    unmapped = []
    controls = []
    segment = []
    for token in ids:
        if token == sp.eos_id():
            break
        if 0 <= token < sp.get_piece_size():
            piece = sp.id_to_piece(token)
            if sp.is_control(token) or (piece.startswith("<") and piece.endswith(">")) or any(
                unicodedata.category(ch) in {"Cc", "Cf", "Co", "Cs"} for ch in piece
            ):
                controls.append(token)
            segment.append(token)
        else:
            if segment:
                parts.append(sp.decode(segment))
                segment = []
            parts.append(f"<UNMAPPED_TOKEN_{token}>")
            unmapped.append(token)
    if segment:
        parts.append(sp.decode(segment))
    return "".join(parts).strip(), unmapped, controls


def generate(policy, images, img_masks, sp, query, max_tokens):
    core = policy.model
    device = next(core.parameters()).device
    token_ids = sp.encode(query, add_bos=True) + sp.encode("\n")
    if len(token_ids) > policy.config.tokenizer_max_length:
        raise ValueError(f"Prompt has {len(token_ids)} tokens, limit is {policy.config.tokenizer_max_length}")
    tokens = torch.full((1, policy.config.tokenizer_max_length), max(0, sp.pad_id()), dtype=torch.long, device=device)
    masks = torch.zeros_like(tokens, dtype=torch.bool)
    tokens[0, : len(token_ids)] = torch.tensor(token_ids, device=device)
    masks[0, : len(token_ids)] = True
    embeddings, pad_masks, ar_masks = core.embed_prefix(images, img_masks, tokens, masks)
    attention = prepare_attention_masks_4d(make_att_2d_masks(pad_masks, ar_masks))
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    vlm = core.paligemma_with_expert
    vlm.paligemma.model.language_model.config._attn_implementation = "eager"
    outputs, cache = vlm.forward(
        attention_mask=attention,
        position_ids=position_ids,
        inputs_embeds=[embeddings, None],
        past_key_values=None,
        use_cache=True,
    )
    last = int(pad_masks[0].sum().item()) - 1
    hidden = outputs[0][:, last, :]
    ids = []
    for step in range(max_tokens):
        logits = vlm.paligemma.lm_head(hidden.float())
        logits[:, max(0, sp.pad_id())] = -torch.inf
        logits[:, sp.bos_id()] = -torch.inf
        token = torch.argmax(logits, dim=-1)
        token_id = int(token.item())
        ids.append(token_id)
        if token_id == sp.eos_id():
            break
        token_embedding = vlm.embed_language_tokens(token[:, None])
        keys = torch.cat(
            [pad_masks, torch.ones((1, step + 1), dtype=torch.bool, device=device)], dim=1
        )
        one_query_attention = prepare_attention_masks_4d(keys[:, None, :])
        next_position = pad_masks.sum(dim=1, keepdim=True) + step
        outputs, cache = vlm.forward(
            attention_mask=one_query_attention,
            position_ids=next_position,
            inputs_embeds=[token_embedding, None],
            past_key_values=cache,
            use_cache=True,
        )
        hidden = outputs[0][:, -1, :]
    text, unmapped, controls = readable(sp, ids)
    return {
        "query": query,
        "prompt_token_ids": token_ids,
        "raw_ids": ids,
        "decoded_text": text,
        "unmapped_ids": unmapped,
        "control_or_nonlanguage_ids": controls,
        "eos_seen": sp.eos_id() in ids,
        "hit_token_limit": len(ids) == max_tokens and sp.eos_id() not in ids,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()
    weight_file = args.model / "model.safetensors"
    if not weight_file.is_file():
        parser.error("--model must contain model.safetensors")
    digest = hashlib.sha256()
    with weight_file.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != SHA256:
        raise ValueError(f"Unexpected model SHA256: {actual_sha256}")
    config = PreTrainedConfig.from_pretrained(args.model)
    config.device = args.device
    policy = PI05Policy.from_pretrained(args.model, config=config, local_files_only=True, strict=True).eval()
    vlm_layers = policy.model.paligemma_with_expert.paligemma.model.language_model.layers
    checks = [
        (f"{VLM_Q_PREFIX}.0.self_attn.q_proj.weight", vlm_layers[0].self_attn.q_proj.weight),
        (f"{VLM_Q_PREFIX}.17.self_attn.q_proj.weight", vlm_layers[17].self_attn.q_proj.weight),
        ("action_out_proj.weight", policy.model.action_out_proj.weight),
    ]
    with safe_open(str(weight_file), framework="pt", device="cpu") as checkpoint:
        for key, parameter in checks:
            expected = checkpoint.get_tensor(key).reshape(-1)[:32]
            actual = parameter.detach().cpu().reshape(-1)[:32]
            if not torch.equal(actual, expected):
                raise RuntimeError(f"The model parameter {key} was not loaded exactly")
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    with np.load(args.frame, allow_pickle=False) as frame:
        task = str(frame["prompt"].item())
        image = frame["image"].copy()
        wrist_image = frame["wrist_image"].copy()
    camera_keys = [k for k in config.image_features if "empty_camera" not in k]
    if len(camera_keys) != 2:
        raise ValueError(f"Expected 2 real LIBERO cameras; got {camera_keys}")
    batch = {
        camera_keys[0]: torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255,
        camera_keys[1]: torch.from_numpy(wrist_image).permute(2, 0, 1).unsqueeze(0).float() / 255,
    }
    images, masks = policy._preprocess_images(batch)
    probes = {
        "caption": "caption en",
        "answer": "answer en what objects are visible in the image?",
        "subtask": f"Task: {task.strip().replace('_', ' ')}\nSubtask:",
    }
    result = {
        "model_repo": REPO_ID,
        "model_revision": REVISION,
        "model_sha256": actual_sha256,
        "frame_sha256": hashlib.sha256(args.frame.read_bytes()).hexdigest(),
        "task": task,
        "model_config_chunk_size": config.chunk_size,
        "checkpoint_weight_load_verified": True,
        "probe_type": "experimental VLM Gemma vocabulary projection; not official task description head",
        "results": {},
    }
    with torch.inference_mode():
        for name, query in probes.items():
            start = time.perf_counter()
            result["results"][name] = generate(policy, images, masks, sp, query, args.max_tokens)
            result["results"][name]["seconds"] = round(time.perf_counter() - start, 3)
            print(f"{name}: {result['results'][name]['decoded_text']!r}", flush=True)
            print(f"{name} raw_ids: {result['results'][name]['raw_ids']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
