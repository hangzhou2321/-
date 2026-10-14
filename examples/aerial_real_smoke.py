from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from manisoftrl import (
    AERIAL_ACTION_DIM,
    AerialRefuelEnv,
    __version__,
    make_aerial_cases,
)


def _sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _save_preview(output_dir: Path, name: str, observation: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = observation["rgb"]
    depth = observation["depth"][..., 0]
    np.save(output_dir / f"{name}_depth_m.npy", depth)
    Image.fromarray(rgb).save(output_dir / f"{name}_rgb.png")
    minimum = float(np.min(depth))
    maximum = float(np.max(depth))
    scale = max(maximum - minimum, 1.0e-6)
    gray = np.asarray(255.0 * (1.0 - (depth - minimum) / scale), dtype=np.uint8)
    preview = np.concatenate((rgb, np.repeat(gray[..., None], 3, axis=2)), axis=1)
    Image.fromarray(preview).save(output_dir / f"{name}_rgb_depth_preview.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--case", default="0")
    parser.add_argument("--seed", type=int, default=80700000)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--action-index", type=int, default=-1)
    parser.add_argument("--action-value", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action_index < -1 or args.action_index >= AERIAL_ACTION_DIM:
        raise ValueError("action-index must be -1 or in [0, 17]")
    if not -1.0 <= args.action_value <= 1.0:
        raise ValueError("action-value must be in [-1, 1]")

    env = AerialRefuelEnv(
        cases=make_aerial_cases(8, seed=20260807),
        default_case=args.case,
        runtime_root=args.runtime_root,
        asset_root=args.asset_root,
        horizon=1,
        observation_mode="hybrid_rgbd",
        image_size=(args.image_size, args.image_size),
    )
    try:
        observation, info = env.reset(seed=args.seed)
        action = np.zeros(AERIAL_ACTION_DIM, dtype=np.float32)
        if args.action_index >= 0:
            action[args.action_index] = args.action_value
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        if args.output_dir is not None:
            _save_preview(args.output_dir, "reset", observation)
            _save_preview(args.output_dir, "step", next_observation)
        payload = {
            "status": "pass",
            "version": __version__,
            "case_id": info["case_id"],
            "seed": info["episode_seed"],
            "action_index": args.action_index,
            "action_value": args.action_value,
            "contract_sha256": info["contract_sha256"],
            "case_sha256": info["case_sha256"],
            "reset_space_contains": bool(env.observation_space.contains(observation)),
            "step_space_contains": bool(
                env.observation_space.contains(next_observation)
            ),
            "state_shape": list(observation["state"].shape),
            "rgb_shape": list(observation["rgb"].shape),
            "depth_shape": list(observation["depth"].shape),
            "reset_state_sha256": _sha256(observation["state"]),
            "reset_rgb_sha256": _sha256(observation["rgb"]),
            "reset_depth_sha256": _sha256(observation["depth"]),
            "step_state_sha256": _sha256(next_observation["state"]),
            "step_rgb_sha256": _sha256(next_observation["rgb"]),
            "step_depth_sha256": _sha256(next_observation["depth"]),
            "reset_depth_min_m": float(np.min(observation["depth"])),
            "reset_depth_max_m": float(np.max(observation["depth"])),
            "step_depth_min_m": float(np.min(next_observation["depth"])),
            "step_depth_max_m": float(np.max(next_observation["depth"])),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": next_info["termination_reason"],
            "insertion_depth_m": next_info["insertion_depth_m"],
            "lateral_error_m": next_info["lateral_error_m"],
            "orientation_error_rad": next_info["orientation_error_rad"],
            "step_peak_contact_force_n": next_info["contact"][
                "step_peak_contact_force_n"
            ],
            "continuum_integrity_valid": next_info[
                "continuum_integrity_valid"
            ],
        }
        if not payload["reset_space_contains"] or not payload["step_space_contains"]:
            raise RuntimeError("aerial RGB-D observation is outside its space")
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
