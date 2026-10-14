from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manisoftrl import AerialRefuelEnv, make_aerial_cases
from manisoftrl.aerial import (
    AERIAL_AIRCRAFT_VISUAL_SCALE,
    AERIAL_BOOM_START_M,
    AERIAL_F16_REFUEL_ANCHOR_MODEL_M,
    AERIAL_KC135_BOOM_PIVOT_MODEL_M,
    AERIAL_KC135_BOOM_TIP_MODEL_M,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify FlightGear aircraft anchors in the ManiSoft scene"
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--aircraft-visual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", default="1")
    parser.add_argument("--case-seed", type=int, default=20260808)
    parser.add_argument("--episode-seed", type=int, default=80720000)
    args = parser.parse_args()
    output_path = (
        None if args.output is None else args.output.expanduser().resolve()
    )

    env = AerialRefuelEnv(
        cases=make_aerial_cases(8, seed=args.case_seed),
        default_case=args.case,
        runtime_root=args.runtime_root,
        aircraft_visual_root=args.aircraft_visual_root,
        observation_mode="state",
    )
    try:
        _, _ = env.reset(seed=args.episode_seed, options={"case": args.case})
        backend = env._backend
        if backend is None:
            raise RuntimeError("aerial backend was not constructed")
        mujoco = backend._mujoco
        model = backend._model
        data = backend._data

        def site_position(name: str) -> np.ndarray:
            site_id = int(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            )
            if site_id < 0:
                raise RuntimeError(f"missing ManiSoft scene site {name!r}")
            return np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()

        hard_boom_pivot = site_position("kc135_boom_pivot_visual_anchor")
        hard_boom_tip = site_position("kc135_boom_tip_visual_anchor")
        soft_segment_root = np.asarray(AERIAL_BOOM_START_M, dtype=np.float64)
        f16_refuel_anchor = site_position("f16_refuel_visual_anchor")
        receptacle_mouth = site_position("receptacle_mouth")
        hard_boom_error = float(np.linalg.norm(hard_boom_tip - soft_segment_root))
        f16_error = float(np.linalg.norm(f16_refuel_anchor - receptacle_mouth))
        tolerance_m = 1.0e-6
        report = {
            "status": "pass",
            "case": str(args.case),
            "episode_seed": int(args.episode_seed),
            "flightgear_to_model_coordinates": "(x,z,-y)",
            "aircraft_uniform_scale": AERIAL_AIRCRAFT_VISUAL_SCALE,
            "kc135_boom_pivot_model_m": list(AERIAL_KC135_BOOM_PIVOT_MODEL_M),
            "kc135_boom_tip_model_m": list(AERIAL_KC135_BOOM_TIP_MODEL_M),
            "f16_refuel_anchor_model_m": list(AERIAL_F16_REFUEL_ANCHOR_MODEL_M),
            "kc135_boom_pivot_world_m": hard_boom_pivot.tolist(),
            "kc135_boom_tip_world_m": hard_boom_tip.tolist(),
            "manisoft_soft_segment_root_world_m": soft_segment_root.tolist(),
            "f16_refuel_anchor_world_m": f16_refuel_anchor.tolist(),
            "manisoft_receptacle_mouth_world_m": receptacle_mouth.tolist(),
            "kc135_tip_to_soft_root_error_m": hard_boom_error,
            "f16_anchor_to_receptacle_error_m": f16_error,
            "binding_tolerance_m": tolerance_m,
            "environment_contract_sha256": env.contract_sha256,
        }
        if max(hard_boom_error, f16_error) > tolerance_m:
            raise RuntimeError(json.dumps(report, sort_keys=True))
    finally:
        env.close()

    payload = json.dumps(report, indent=2, sort_keys=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
