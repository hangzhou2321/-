from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manisoftrl import AerialRefuelEnv, make_aerial_cases
from manisoftrl.aerial import AERIAL_NOZZLE_RETRACTED_LENGTH_M


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args()

    env = AerialRefuelEnv(
        cases=make_aerial_cases(1),
        runtime_root=args.runtime_root,
        asset_root=args.asset_root,
        horizon=2,
        observation_mode="state",
    )
    try:
        env.reset(seed=80700000)
        backend = env._backend
        if backend is None:
            raise RuntimeError("aerial backend was not created")
        simulator = backend._backend._mujoco_simulator
        position, axis, _, _, quaternion = backend.case.motion(0.0)

        aligned_tip = position + axis * 0.060
        aligned_pose = np.concatenate(
            (aligned_tip - axis * AERIAL_NOZZLE_RETRACTED_LENGTH_M, quaternion)
        )
        simulator.step(gripper_pose=aligned_pose)
        aligned = backend.snapshot()
        if not env._success_now(aligned):
            raise RuntimeError("aligned 60 mm insertion did not satisfy the predicate")

        local_x_world = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        collision_tip = position + axis * 0.020 + local_x_world * 0.065
        collision_pose = np.concatenate(
            (collision_tip - axis * AERIAL_NOZZLE_RETRACTED_LENGTH_M, quaternion)
        )
        for _ in range(8):
            simulator.step(gripper_pose=collision_pose)
        collision = backend.snapshot()
        if collision.contact_force_n <= 0.0:
            raise RuntimeError("offset nozzle did not produce receptacle contact force")

        print(
            json.dumps(
                {
                    "status": "pass",
                    "aligned_insertion_depth_m": aligned.insertion_depth_m,
                    "aligned_lateral_error_m": aligned.lateral_error_m,
                    "aligned_orientation_error_rad": aligned.orientation_error_rad,
                    "aligned_contact_force_n": aligned.contact_force_n,
                    "offset_insertion_depth_m": collision.insertion_depth_m,
                    "offset_lateral_error_m": collision.lateral_error_m,
                    "offset_contact_force_n": collision.contact_force_n,
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
