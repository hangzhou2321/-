from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manisoftrl import make_vector_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()

    envs = make_vector_env(
        args.num_envs,
        {
            "case_dirs": args.case_root,
            "runtime_root": args.runtime_root,
            "asset_root": args.asset_root,
            "horizon": args.horizon,
        },
        asynchronous=True,
    )
    try:
        observations, _ = envs.reset(seed=args.seed)
        next_observations, rewards, terminated, truncated, _ = envs.step(
            np.zeros((args.num_envs, 19), dtype=np.float32)
        )
        print(
            json.dumps(
                {
                    "reset_shape": list(observations.shape),
                    "step_shape": list(next_observations.shape),
                    "rewards": rewards.tolist(),
                    "terminated": terminated.tolist(),
                    "truncated": truncated.tolist(),
                },
                sort_keys=True,
            )
        )
    finally:
        envs.close()


if __name__ == "__main__":
    main()
