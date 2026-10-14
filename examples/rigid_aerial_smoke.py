from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from manisoftrl import RigidFlyingBoomEnv, make_rigid_aerial_cases


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remote rigid flying-boom dynamics/contact/RGB-D smoke"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="0")
    parser.add_argument("--case-seed", type=int, default=20260808)
    parser.add_argument("--episode-seed", type=int, default=80730000)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--left", type=float, default=0.0)
    parser.add_argument("--right", type=float, default=0.0)
    parser.add_argument("--extension", type=float, default=1.0)
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = RigidFlyingBoomEnv(
        cases=make_rigid_aerial_cases(8, seed=args.case_seed),
        observation_mode="hybrid_rgbd",
        image_size=(256, 256),
        horizon=args.steps,
    )
    action = np.asarray(
        (args.left, args.right, args.extension), dtype=np.float32
    )
    rows: list[dict[str, object]] = []
    try:
        observation, info = env.reset(
            seed=args.episode_seed, options={"case": args.case}
        )
        Image.fromarray(observation["rgb"]).save(output / "reset_rgb.png")
        np.save(output / "reset_depth_m.npy", observation["depth"])
        terminated = truncated = False
        for _ in range(args.steps):
            observation, reward, terminated, truncated, info = env.step(action)
            rows.append(
                {
                    "step": int(info["step_count"]),
                    "reward": float(reward),
                    "insertion_depth_m": float(info["insertion_depth_m"]),
                    "lateral_error_m": float(info["lateral_error_m"]),
                    "orientation_error_rad": float(info["orientation_error_rad"]),
                    "extension_m": float(info["boom_extension_m"]),
                    "extension_velocity_m_s": float(
                        info["boom_extension_velocity_m_s"]
                    ),
                    "step_peak_contact_force_n": float(
                        info["contact"]["step_peak_contact_force_n"]
                    ),
                    "termination_reason": info["termination_reason"],
                }
            )
            if terminated or truncated:
                break
        Image.fromarray(observation["rgb"]).save(output / "final_rgb.png")
        np.save(output / "final_depth_m.npy", observation["depth"])
        report = {
            "schema_version": 1,
            "formal_evaluation": False,
            "scope": "rigid_boom_remote_integration_smoke",
            "case": str(args.case),
            "case_seed": args.case_seed,
            "episode_seed": args.episode_seed,
            "action": action.tolist(),
            "environment_contract_sha256": env.contract_sha256,
            "contract": env.contract,
            "observation_shapes": {
                key: list(value.shape) for key, value in observation.items()
            },
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "official_success": bool(info["official_success"]),
            "termination_reason": info["termination_reason"],
            "rows": rows,
        }
        report_path = output / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "report_sha256": _sha256(report_path),
                    "official_success": report["official_success"],
                    "termination_reason": report["termination_reason"],
                    "steps": len(rows),
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
