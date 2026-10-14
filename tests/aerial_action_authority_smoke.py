from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manisoftrl import AERIAL_ACTION_DIM, AerialRefuelEnv, make_aerial_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    probes = ((1, 0.25),)
    rows: list[dict[str, object]] = []
    for channel, amplitude in probes:
        env = AerialRefuelEnv(
            cases=make_aerial_cases(1),
            runtime_root=args.runtime_root,
            asset_root=args.asset_root,
            horizon=2 * args.steps,
            observation_mode="state",
        )
        try:
            observation, _ = env.reset(seed=80710000)
            action = np.zeros(AERIAL_ACTION_DIM, dtype=np.float32)
            action[-1] = -1.0
            action[:18].reshape(6, 3)[:, channel] = amplitude
            best = {
                "lateral_error_m": float("inf"),
                "insertion_depth_m": 0.0,
                "step": 0,
                "tip_relative_m": observation[60:63].astype(float).tolist(),
            }
            reason = None
            success = None
            for step in range(1, 2 * args.steps + 1):
                if step == args.steps + 1:
                    action[-1] = 1.0
                observation, _, terminated, truncated, info = env.step(action)
                lateral = float(info["lateral_error_m"])
                if lateral < float(best["lateral_error_m"]):
                    best = {
                        "lateral_error_m": lateral,
                        "insertion_depth_m": float(info["insertion_depth_m"]),
                        "step": step,
                        "tip_relative_m": observation[60:63]
                        .astype(float)
                        .tolist(),
                    }
                if terminated or truncated:
                    reason = info["termination_reason"]
                    if info["official_success"]:
                        success = {
                            "insertion_depth_m": float(info["insertion_depth_m"]),
                            "lateral_error_m": lateral,
                            "step": step,
                            "peak_contact_force_n": float(
                                info["contact"]["episode_peak_contact_force_n"]
                            ),
                        }
                    break
            rows.append(
                {
                    "channel": channel,
                    "amplitude": amplitude,
                    "best": best,
                    "success": success,
                    "termination_reason": reason,
                }
            )
        finally:
            env.close()

    reachable = [
        row
        for row in rows
        if row["termination_reason"] == "official_success"
    ]
    payload = {
        "status": "pass" if reachable else "fail",
        "reachable_count": len(reachable),
        "best": min(rows, key=lambda row: float(row["best"]["lateral_error_m"])),
        "reachable": reachable,
    }
    print(json.dumps(payload, sort_keys=True))
    if not reachable:
        raise RuntimeError("no real 19D bend-and-extension probe reached the success geometry")


if __name__ == "__main__":
    main()
