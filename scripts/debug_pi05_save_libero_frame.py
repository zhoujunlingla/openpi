"""Save one deterministic LIBERO frame for checkpoint comparison."""

import sys

from libero.libero import benchmark
import numpy as np
from openpi_client import image_tools

sys.path.insert(0, "examples/libero")
from main import LIBERO_DUMMY_ACTION
from main import _get_libero_env
from main import _quat2axisangle


def main():
    task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(0)
    env, prompt = _get_libero_env(task, 256, 7)
    try:
        obs = env.reset()
        for _ in range(10):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        def prepare(x):
            image = np.ascontiguousarray(x[::-1, ::-1])
            return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))

        np.savez_compressed(
            sys.argv[1],
            image=prepare(obs["agentview_image"]),
            wrist_image=prepare(obs["robot0_eye_in_hand_image"]),
            state=np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"].copy()),
                    obs["robot0_gripper_qpos"],
                )
            ),
            prompt=np.asarray(prompt),
        )
        print(f"Saved LIBERO frame; prompt={prompt!r}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
