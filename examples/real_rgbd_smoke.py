from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from manisoftrl import ManiSoftRLEnv, __version__


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--case-root", required=True)
    parser.add_argument("--case", default="0")
    parser.add_argument("--seed", type=int, default=70700000)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--depth-max-m", type=float, default=10.0)
    args = parser.parse_args()

    env = ManiSoftRLEnv(
        case_dirs=args.case_root,
        runtime_root=args.runtime_root,
        asset_root=args.asset_root,
        default_case=args.case,
        horizon=1,
        observation_mode="hybrid_rgbd",
        image_size=(args.image_size, args.image_size),
        depth_max_m=args.depth_max_m,
        camera_name="third",
    )
    try:
        observation, info = env.reset(seed=args.seed)
        reset_contains = bool(env.observation_space.contains(observation))
        next_observation, reward, terminated, truncated, next_info = env.step(
            np.zeros(19, dtype=np.float32)
        )
        step_contains = bool(env.observation_space.contains(next_observation))
        payload = {
            "status": "pass",
            "version": __version__,
            "contract_sha256": info["contract_sha256"],
            "case_id": info["case_id"],
            "seed": info["episode_seed"],
            "state_shape": list(observation["state"].shape),
            "rgb_shape": list(observation["rgb"].shape),
            "rgb_dtype": str(observation["rgb"].dtype),
            "depth_shape": list(observation["depth"].shape),
            "depth_dtype": str(observation["depth"].dtype),
            "reset_state_sha256": _array_sha256(observation["state"]),
            "reset_rgb_sha256": _array_sha256(observation["rgb"]),
            "reset_depth_sha256": _array_sha256(observation["depth"]),
            "step_state_sha256": _array_sha256(next_observation["state"]),
            "step_rgb_sha256": _array_sha256(next_observation["rgb"]),
            "step_depth_sha256": _array_sha256(next_observation["depth"]),
            "reset_depth_min_m": float(np.min(observation["depth"])),
            "reset_depth_max_m": float(np.max(observation["depth"])),
            "step_depth_min_m": float(np.min(next_observation["depth"])),
            "step_depth_max_m": float(np.max(next_observation["depth"])),
            "reset_space_contains": reset_contains,
            "step_space_contains": step_contains,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": next_info["termination_reason"],
            "scene_sha256": info["scene_sha256"],
            "depth_units": info["contract"]["depth_units"],
            "depth_max_m": info["contract"]["depth_max_m"],
        }
        if not reset_contains or not step_contains:
            raise RuntimeError("RGB-D observation is outside its Gymnasium space")
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
