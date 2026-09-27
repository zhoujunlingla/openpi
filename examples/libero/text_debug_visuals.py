"""LIBERO-side text logging / optional subtitles. Compatible with Python 3.8.

Subtitles are drawn on a NEW array, never on the image sent to the policy.
The server must be serve_pi05_text.py when using two_phase=True.
"""
import textwrap
import numpy as np


def infer_with_text(client, element, two_phase=False, pause=False):
    if two_phase or pause:
        plan = client.infer(dict(element, _text_stage="plan"))
        print("[VLM BEFORE ACTION] mode={} id={} text={!r}".format(
            plan.get("text_mode"), plan["plan_id"], plan.get("subtask", "")), flush=True)
        if pause:
            input("Text is above. Press Enter to run Action Expert (Ctrl-C to stop): ")
        result = client.infer({"_text_stage": "act", "_text_plan_id": plan["plan_id"]})
    else:
        result = client.infer(element)
        print("[VLM BEFORE SIM STEP] mode={} text={!r}".format(
            result.get("text_mode"), result.get("subtask", "<no text field>")), flush=True)
    print("[LOW-LEVEL PROMPT] {!r}".format(result.get("action_prompt", element["prompt"])), flush=True)
    return result


def annotated_frame(rgb, result, step, enabled=True):
    frame = np.array(rgb, copy=True)
    if not enabled:
        return frame
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame  # console logging remains available without Pillow
    image = Image.fromarray(frame)
    band = 192
    canvas = Image.new("RGB", (image.width, image.height + band), (0, 0, 0))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    mode = result.get("text_mode", "unknown")
    text = result.get("subtask", "<no generated text>")
    action_prompt = result.get("action_prompt", "<no action prompt>")
    plan_id = str(result.get("plan_id", "unknown"))[:8]
    title = "step={} mode={} id={}".format(step, mode, plan_id)
    # Pillow's default font is ASCII-only. Escape other Unicode characters in
    # the video; the console and server JSONL retain the exact decoded text.
    lines = [title]
    for label, value in (("VLM", text), ("ACTION_INPUT", action_prompt)):
        visible = (label + ": " + str(value).replace("\n", " ")).encode("ascii", "backslashreplace").decode("ascii")
        lines.extend(textwrap.wrap(visible, width=max(24, image.width // 6 - 2)))
    if len(lines) > 13:
        lines = lines[:13]
        lines[-1] = lines[-1][:-3] + "..."
    for index, line in enumerate(lines):
        draw.text((5, image.height + 5 + 14 * index), line, font=font, fill=(255, 255, 255))
    return np.asarray(canvas)
