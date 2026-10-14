from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from manisoftrl import (
    AerialRefuelEnv,
    AerialSynergyActionWrapper,
    RigidFlyingBoomEnv,
    RigidStateScaleWrapper,
    make_aerial_cases,
    make_rigid_aerial_cases,
)
from manisoftrl.promotion import compare_paired_outcomes


UNSAFE_REASONS = {
    "hard_contact_force",
    "continuum_integrity_failure",
    "mechanism_integrity_failure",
    "disconnect_after_engagement",
    "out_of_workspace",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _evaluate(model: Any, env: AerialRefuelEnv, seed_base: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(env.cases):
        episode_seed = int(seed_base + index)
        observation, info = env.reset(
            seed=episode_seed,
            options={"case": case_id},
        )
        episode_return = 0.0
        peak_force_n = 0.0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            peak_force_n = max(
                peak_force_n,
                float(info["contact"]["episode_peak_contact_force_n"]),
            )
        rows.append(
            {
                "case": case_id,
                "seed": episode_seed,
                "official_success": bool(info["official_success"]),
                "termination_reason": info["termination_reason"],
                "episode_return": episode_return,
                "episode_steps": int(info["step_count"]),
                "peak_contact_force_n": peak_force_n,
                "final_insertion_depth_m": float(info["insertion_depth_m"]),
                "final_lateral_error_m": float(info["lateral_error_m"]),
                "final_orientation_error_rad": float(
                    info["orientation_error_rad"]
                ),
                "final_boom_extension_m": float(info["boom_extension_m"]),
            }
        )
    return rows


def _outcomes(rows: list[dict[str, Any]]) -> dict[str, bool]:
    return {str(row["case"]): bool(row["official_success"]) for row in rows}


def _safety(rows: list[dict[str, Any]], safe_force_n: float) -> dict[str, Any]:
    return {
        "unsafe_terminations": sum(
            row["termination_reason"] in UNSAFE_REASONS for row in rows
        ),
        "safe_force_exceedances": sum(
            float(row["peak_contact_force_n"]) > safe_force_n for row in rows
        ),
        "max_peak_contact_force_n": max(
            float(row["peak_contact_force_n"]) for row in rows
        ),
    }


def _paired_report(
    *,
    parent_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    parent_sha256: str,
    candidate_sha256: str,
    contract_sha256: str,
    safe_force_n: float,
    timestep: int,
) -> dict[str, Any]:
    success_decision = compare_paired_outcomes(
        _outcomes(parent_rows),
        _outcomes(candidate_rows),
    )
    parent_safety = _safety(parent_rows, safe_force_n)
    candidate_safety = _safety(candidate_rows, safe_force_n)
    safety_no_regression = bool(
        candidate_safety["unsafe_terminations"]
        <= parent_safety["unsafe_terminations"]
        and candidate_safety["safe_force_exceedances"]
        <= parent_safety["safe_force_exceedances"]
    )
    promote = bool(success_decision.promote and safety_no_regression)
    reason = (
        success_decision.reason
        if safety_no_regression
        else "candidate_safety_regression"
    )
    return {
        "formal_evaluation": False,
        "evaluation_scope": "aerial_training_canary",
        "common_random_numbers": True,
        "single_frozen_checkpoint_per_role": True,
        "success_predicate": "aerial_terminal_docking_v1",
        "environment_contract_sha256": contract_sha256,
        "parent_checkpoint_sha256": parent_sha256,
        "candidate_checkpoint_sha256": candidate_sha256,
        "training_timestep": timestep,
        "parent": {"rows": parent_rows, "safety": parent_safety},
        "candidate": {"rows": candidate_rows, "safety": candidate_safety},
        "decision": {
            **asdict(success_decision),
            "safety_no_regression": safety_no_regression,
            "promote": promote,
            "reason": reason,
        },
    }


def _make_env(
    *,
    runtime_root: Path | None,
    asset_root: Path | None,
    cases: int,
    case_seed: int,
    horizon: int,
    default_case: str | None,
    action_mode: str,
    mechanism: str,
) -> AerialRefuelEnv | AerialSynergyActionWrapper:
    if mechanism == "rigid":
        env = RigidFlyingBoomEnv(
            cases=make_rigid_aerial_cases(cases, seed=case_seed),
            default_case=default_case,
            horizon=horizon,
            observation_mode="state",
        )
        return RigidStateScaleWrapper(env)
    else:
        if runtime_root is None:
            raise ValueError("runtime_root is required for the continuum mechanism")
        env = AerialRefuelEnv(
            cases=make_aerial_cases(cases, seed=case_seed),
            default_case=default_case,
            runtime_root=runtime_root,
            asset_root=asset_root,
            horizon=horizon,
            observation_mode="state",
        )
    if action_mode == "synergy4":
        return AerialSynergyActionWrapper(env)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure online SAC for flying-boom terminal docking"
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--train-cases", type=int, default=1)
    parser.add_argument("--eval-cases", type=int, default=8)
    parser.add_argument("--train-case-seed", type=int, default=20260807)
    parser.add_argument("--eval-case-seed", type=int, default=20260808)
    parser.add_argument("--eval-seed-base", type=int, default=80720000)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--seed", type=int, default=8072)
    parser.add_argument("--learning-starts", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--td3-noise-scale", type=float, default=1.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--algorithm",
        choices=("SAC", "TD3"),
        default="SAC",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        help="fixed SAC entropy coefficient; omit to use SB3 auto tuning",
    )
    parser.add_argument(
        "--action-mode",
        choices=("synergy4", "direct19", "direct3"),
        default="synergy4",
    )
    parser.add_argument(
        "--mechanism",
        choices=("continuum", "rigid"),
        default="continuum",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--parent-replay-buffer", type=Path)
    args = parser.parse_args()

    if min(
        args.steps,
        args.eval_every,
        args.train_cases,
        args.eval_cases,
        args.horizon,
        args.torch_threads,
    ) < 1:
        raise SystemExit("steps, case counts, horizon, and eval interval must be positive")
    if args.resume and args.parent_checkpoint is not None:
        raise SystemExit("--resume and --parent-checkpoint are mutually exclusive")
    if args.parent_replay_buffer is not None and args.parent_checkpoint is None:
        raise SystemExit("--parent-replay-buffer requires --parent-checkpoint")
    if args.mechanism == "rigid" and args.action_mode != "direct3":
        raise SystemExit("the rigid mechanism requires --action-mode direct3")
    if args.mechanism == "continuum" and args.action_mode == "direct3":
        raise SystemExit("the continuum mechanism requires synergy4 or direct19")
    if args.mechanism == "continuum" and args.runtime_root is None:
        raise SystemExit("--runtime-root is required for the continuum mechanism")
    if args.ent_coef is not None and args.ent_coef <= 0.0:
        raise SystemExit("--ent-coef must be positive")
    if args.learning_rate <= 0.0 or args.td3_noise_scale <= 0.0:
        raise SystemExit("learning rate and TD3 noise scale must be positive")
    if args.algorithm == "TD3" and args.ent_coef is not None:
        raise SystemExit("--ent-coef only applies to SAC")

    try:
        import torch
        from stable_baselines3 import SAC, TD3
        from stable_baselines3.common.noise import NormalActionNoise
    except ImportError as error:
        raise SystemExit("install the baseline extra: pip install -e .[baseline]") from error
    torch.set_num_threads(args.torch_threads)
    algorithm_class = SAC if args.algorithm == "SAC" else TD3

    output = args.output.expanduser().resolve()
    state_path = output / "trainer_state.json"
    latest_path = output / "latest.zip"
    replay_path = output / "replay_buffer.pkl"
    checkpoint_dir = output / "checkpoints"
    report_dir = output / "evaluations"
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit(f"output is not empty; use --resume or a new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    train_env = _make_env(
        runtime_root=args.runtime_root,
        asset_root=args.asset_root,
        cases=args.train_cases,
        case_seed=args.train_case_seed,
        horizon=args.horizon,
        default_case=None,
        action_mode=args.action_mode,
        mechanism=args.mechanism,
    )
    eval_env = _make_env(
        runtime_root=args.runtime_root,
        asset_root=args.asset_root,
        cases=args.eval_cases,
        case_seed=args.eval_case_seed,
        horizon=args.horizon,
        default_case="0",
        action_mode=args.action_mode,
        mechanism=args.mechanism,
    )
    td3_action_noise = None
    td3_noise_sigma: list[float] | float | None = None
    if args.algorithm == "TD3":
        action_dim = int(train_env.action_space.shape[0])
        noise_sigma = np.full(action_dim, 0.10, dtype=np.float32)
        if args.mechanism == "rigid" and action_dim == 3:
            noise_sigma = np.asarray((0.02, 0.02, 0.20), dtype=np.float32)
        noise_sigma *= float(args.td3_noise_scale)
        td3_action_noise = NormalActionNoise(
            mean=np.zeros(action_dim, dtype=np.float32),
            sigma=noise_sigma,
        )
        td3_noise_sigma = noise_sigma.tolist()
    try:
        training_contract_sha256 = train_env.contract_sha256
        evaluation_contract_sha256 = eval_env.contract_sha256
        if args.resume:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["training_contract_sha256"] != training_contract_sha256:
                raise RuntimeError("resume training contract does not match")
            if state["evaluation_contract_sha256"] != evaluation_contract_sha256:
                raise RuntimeError("resume evaluation contract does not match")
            if state.get("algorithm", "SAC") != args.algorithm:
                raise RuntimeError("resume algorithm does not match")
            model = algorithm_class.load(
                latest_path, env=train_env, device=args.device
            )
            model.learning_rate = args.learning_rate
            model.lr_schedule = lambda _: args.learning_rate
            if td3_action_noise is not None:
                model.action_noise = td3_action_noise
            if replay_path.is_file():
                model.load_replay_buffer(replay_path)
            parent_rows = state["parent_rows"]
            parent_sha256 = state["parent_checkpoint_sha256"]
        else:
            if args.parent_checkpoint is None:
                algorithm_kwargs: dict[str, Any] = {}
                if args.algorithm == "SAC":
                    algorithm_kwargs["ent_coef"] = (
                        "auto" if args.ent_coef is None else args.ent_coef
                    )
                else:
                    algorithm_kwargs["action_noise"] = td3_action_noise
                model = algorithm_class(
                    "MlpPolicy",
                    train_env,
                    seed=args.seed,
                    device=args.device,
                    learning_starts=args.learning_starts,
                    buffer_size=100_000,
                    batch_size=64,
                    learning_rate=args.learning_rate,
                    gamma=0.99,
                    tau=0.005,
                    policy_kwargs={"net_arch": [128, 128]},
                    verbose=1,
                    **algorithm_kwargs,
                )
                model.save(output / "parent_initial.zip")
            else:
                source = args.parent_checkpoint.expanduser().resolve(strict=True)
                shutil.copy2(source, output / "parent_initial.zip")
                model = algorithm_class.load(
                    source, env=train_env, device=args.device
                )
                model.learning_rate = args.learning_rate
                model.lr_schedule = lambda _: args.learning_rate
                if td3_action_noise is not None:
                    model.action_noise = td3_action_noise
                if args.parent_replay_buffer is not None:
                    model.load_replay_buffer(
                        args.parent_replay_buffer.expanduser().resolve(strict=True)
                    )
            frozen_parent = algorithm_class.load(
                output / "parent_initial.zip",
                device=args.device,
            )
            parent_rows = _evaluate(frozen_parent, eval_env, args.eval_seed_base)
            parent_sha256 = _sha256(output / "parent_initial.zip")
            model.save(latest_path)
            model.save_replay_buffer(replay_path)
            state = {
                "schema_version": 1,
                "algorithm": args.algorithm,
                "training_contract_sha256": training_contract_sha256,
                "evaluation_contract_sha256": evaluation_contract_sha256,
                "parent_checkpoint_sha256": parent_sha256,
                "parent_rows": parent_rows,
                "promoted_timestep": int(model.num_timesteps),
                "latest_timestep": int(model.num_timesteps),
            }
            _write_json(state_path, state)

        manifest = {
            "algorithm": args.algorithm,
            "policy": "MlpPolicy",
            "pure_online_rl": True,
            "behaviour_cloning": False,
            "expert_actions": False,
            "runtime_fsm": False,
            "runtime_waypoints": False,
            "observation_mode": "state",
            "action_mode": args.action_mode,
            "mechanism": args.mechanism,
            "entropy_coefficient": (
                None
                if args.algorithm == "TD3"
                else "auto"
                if args.ent_coef is None
                else args.ent_coef
            ),
            "learning_rate": args.learning_rate,
            "td3_action_noise_sigma": td3_noise_sigma,
            "seed": args.seed,
            "train_cases": args.train_cases,
            "eval_cases": args.eval_cases,
            "train_case_seed": args.train_case_seed,
            "eval_case_seed": args.eval_case_seed,
            "eval_seed_base": args.eval_seed_base,
            "horizon": args.horizon,
            "torch_threads": args.torch_threads,
            "network": [128, 128],
            "batch_size": 64,
            "training_contract_sha256": training_contract_sha256,
            "evaluation_contract_sha256": evaluation_contract_sha256,
            "parent_replay_buffer": (
                None
                if args.parent_replay_buffer is None
                else str(args.parent_replay_buffer.expanduser().resolve())
            ),
        }
        _write_json(output / "training_manifest.json", manifest)

        target_timestep = int(model.num_timesteps + args.steps)
        last_report: dict[str, Any] | None = None
        while model.num_timesteps < target_timestep:
            chunk = min(args.eval_every, target_timestep - model.num_timesteps)
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            timestep = int(model.num_timesteps)
            checkpoint = checkpoint_dir / f"step_{timestep:09d}.zip"
            model.save(checkpoint)
            model.save(latest_path)
            model.save_replay_buffer(replay_path)

            frozen_candidate = algorithm_class.load(
                checkpoint, device=args.device
            )
            candidate_rows = _evaluate(
                frozen_candidate,
                eval_env,
                args.eval_seed_base,
            )
            candidate_sha256 = _sha256(checkpoint)
            report = _paired_report(
                parent_rows=parent_rows,
                candidate_rows=candidate_rows,
                parent_sha256=parent_sha256,
                candidate_sha256=candidate_sha256,
                contract_sha256=evaluation_contract_sha256,
                safe_force_n=eval_env.safe_contact_force_n,
                timestep=timestep,
            )
            _write_json(report_dir / f"step_{timestep:09d}.json", report)
            last_report = report

            if report["decision"]["promote"]:
                shutil.copy2(checkpoint, output / "promoted.zip")
                parent_rows = candidate_rows
                parent_sha256 = candidate_sha256
                state["promoted_timestep"] = timestep
            state.update(
                {
                    "parent_checkpoint_sha256": parent_sha256,
                    "parent_rows": parent_rows,
                    "latest_checkpoint_sha256": _sha256(latest_path),
                    "latest_timestep": timestep,
                    "last_report": str(
                        report_dir / f"step_{timestep:09d}.json"
                    ),
                }
            )
            _write_json(state_path, state)
            print(
                json.dumps(
                    {
                        "timestep": timestep,
                        "candidate_successes": sum(_outcomes(candidate_rows).values()),
                        "parent_successes": sum(_outcomes(parent_rows).values()),
                        "promote": report["decision"]["promote"],
                        "reason": report["decision"]["reason"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        summary = {
            "status": "complete",
            "latest_timestep": int(model.num_timesteps),
            "promoted_timestep": state["promoted_timestep"],
            "promoted_successes": sum(_outcomes(parent_rows).values()),
            "evaluation_cases": len(parent_rows),
            "last_decision": None if last_report is None else last_report["decision"],
        }
        _write_json(output / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True))
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
