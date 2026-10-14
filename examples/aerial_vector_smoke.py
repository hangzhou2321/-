from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manisoftrl import (
    AERIAL_ACTION_DIM,
    make_aerial_cases,
    make_aerial_vector_env,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--case", default="0")
    parser.add_argument("--seed", type=int, default=80700000)
    args = parser.parse_args()

    envs = make_aerial_vector_env(
        args.num_envs,
        {
            "cases": make_aerial_cases(8, seed=20260807),
            "default_case": args.case,
            "runtime_root": args.runtime_root,
            "asset_root": args.asset_root,
            "horizon": 1,
            "observation_mode": "state",
        },
        asynchronous=True,
    )
    try:
        observations, _ = envs.reset(seed=args.seed)
        next_observations, rewards, terminated, truncated, _ = envs.step(
            np.zeros((args.num_envs, AERIAL_ACTION_DIM), dtype=np.float32)
        )
        print(
            json.dumps(
                {
                    "status": "pass",
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
