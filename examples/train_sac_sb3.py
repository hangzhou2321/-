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
    parser.add_argument("--horizon", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("sac_state_baseline"))
    args = parser.parse_args()

    try:
        from stable_baselines3 import SAC
    except ImportError as error:
        raise SystemExit("install the baseline extra: pip install -e .[baseline]") from error

    env = ManiSoftRLEnv(
        args.case_root,
        runtime_root=args.runtime_root,
        asset_root=args.asset_root,
        default_case=args.case,
        horizon=args.horizon,
        observation_mode="state",
    )
    try:
        model = SAC("MlpPolicy", env, seed=args.seed, verbose=1)
        model.learn(total_timesteps=args.steps)
        model.save(args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
