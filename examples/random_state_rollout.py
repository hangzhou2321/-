from __future__ import annotations

import argparse
from pathlib import Path

from manisoftrl import ManiSoftRLEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()

    env = ManiSoftRLEnv(
        args.case_root,
        runtime_root=args.runtime_root,
        asset_root=args.asset_root,
        default_case=args.case,
        horizon=args.horizon,
    )
    try:
        observation, info = env.reset(seed=args.seed)
        print("reset", observation.shape, info["case_id"], info["scene_sha256"])
        while True:
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            print(
                "step",
                info["step_count"],
                reward,
                info["official_success"],
                info["contact"],
            )
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
