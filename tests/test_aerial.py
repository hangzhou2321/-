from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from manisoftrl import (
    AERIAL_ACTION_DIM,
    AERIAL_ENV_ID,
    AERIAL_STATE_DIM,
    RIGID_AERIAL_ACTION_DIM,
    RIGID_AERIAL_ENV_ID,
    AerialRefuelCase,
    AerialRefuelEnv,
    AerialSynergyActionWrapper,
    RigidFlyingBoomEnv,
    RigidStateScaleWrapper,
    make_aerial_vector_env,
    make_aerial_cases,
    make_rigid_aerial_cases,
)
from manisoftrl.aerial import AerialSnapshot


class FakeAerialBackend:
    def __init__(
        self,
        case: AerialRefuelCase,
        seed: int,
        enable_renderer: bool,
        image_size: tuple[int, int],
        camera_name: str,
        update_interval: int,
    ) -> None:
        self.case = case
        self.seed = seed
        self.enable_renderer = enable_renderer
        self.image_size = image_size
        self.camera_name = camera_name
        self.update_interval = update_interval
        self.step_count = 0
        self.action = np.zeros(AERIAL_ACTION_DIM, dtype=np.float32)
        self.closed = False

    def snapshot(self) -> AerialSnapshot:
        time_s = self.step_count * self.update_interval * 0.00005
        position, axis, velocity, angular_velocity, _ = self.case.motion(time_s)
        inserted = bool(self.action[0] > 0.5)
        insertion_depth = 0.060 if inserted else 0.0
        tip = position + axis * insertion_depth if inserted else position - axis * 0.10
        nodes = np.linspace(
            np.asarray((0.0, 0.72, 0.90)), tip, 21, dtype=np.float64
        )
        contact = 40.0 if self.action[1] > 0.5 else 0.0
        return AerialSnapshot(
            boom_positions_m=nodes,
            boom_velocities_m_s=np.zeros_like(nodes),
            tip_position_m=tip,
            tip_velocity_m_s=np.zeros(3, dtype=np.float64),
            tip_axis=axis.copy(),
            receptacle_position_m=position,
            receptacle_axis=axis,
            receptacle_linear_velocity_m_s=velocity,
            receptacle_angular_velocity_rad_s=angular_velocity,
            insertion_depth_m=insertion_depth,
            lateral_error_m=0.0,
            orientation_error_rad=0.0,
            contact_force_n=contact,
            step_peak_contact_force_n=contact,
            episode_peak_contact_force_n=contact,
            boom_extension_m=0.0,
            boom_extension_velocity_m_s=0.0,
            continuum_integrity_valid=bool(self.action[2] <= 0.5),
            physics_time_s=time_s,
        )

    def advance(self, action: np.ndarray) -> None:
        self.action = np.asarray(action, dtype=np.float32).copy()
        self.step_count += 1

    def render_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.enable_renderer:
            raise RuntimeError("renderer disabled")
        rgb = np.full(
            (*self.image_size, 3), 20 + self.step_count, dtype=np.uint8
        )
        depth = np.full(
            (*self.image_size, 1), 1.5 + 0.1 * self.step_count, dtype=np.float32
        )
        return rgb, depth

    def close(self) -> None:
        self.closed = True


def _factory(
    case: AerialRefuelCase,
    seed: int,
    enable_renderer: bool,
    image_size: tuple[int, int],
    camera_name: str,
    update_interval: int,
) -> FakeAerialBackend:
    return FakeAerialBackend(
        case,
        seed,
        enable_renderer,
        image_size,
        camera_name,
        update_interval,
    )


def _env(**kwargs: Any) -> AerialRefuelEnv:
    return AerialRefuelEnv(
        backend_factory=_factory,
        backend_id="fake_aerial_backend_v1",
        **kwargs,
    )


def test_aerial_canary_panel_is_deterministic_and_case_zero_is_static() -> None:
    first = make_aerial_cases(8, seed=20260807)
    second = make_aerial_cases(8, seed=20260807)
    assert tuple(first) == tuple(str(index) for index in range(8))
    assert {key: value.to_dict() for key, value in first.items()} == {
        key: value.to_dict() for key, value in second.items()
    }
    position0, axis0, velocity0, angular0, _ = first["0"].motion(0.0)
    position1, axis1, velocity1, angular1, _ = first["0"].motion(3.0)
    np.testing.assert_array_equal(position0, position1)
    np.testing.assert_array_equal(axis0, axis1)
    np.testing.assert_array_equal(velocity0, velocity1)
    np.testing.assert_array_equal(angular0, angular1)
    assert np.linalg.norm(axis0) == 1.0


def test_aerial_state_env_is_gymnasium_compliant() -> None:
    env = _env(horizon=3)
    try:
        check_env(env, skip_render_check=True)
        observation, info = env.reset(seed=7, options={"case": 0})
        assert observation.shape == (AERIAL_STATE_DIM,)
        assert observation.dtype == np.float32
        assert env.action_space.shape == (AERIAL_ACTION_DIM,)
        assert info["case_id"] == "0"
        assert info["contract"]["case_in_observation"] is False
        assert info["contract"]["runtime_fsm"] is False
        assert info["contract_sha256"] == env.contract_sha256
    finally:
        env.close()


def test_aerial_success_requires_dwell() -> None:
    env = _env(horizon=4, dwell_steps=2)
    try:
        env.reset(seed=1)
        action = np.zeros(AERIAL_ACTION_DIM, dtype=np.float32)
        action[0] = 1.0
        _, _, terminated, truncated, info = env.step(action)
        assert terminated is False
        assert truncated is False
        assert info["dwell_steps"] == 1
        _, reward, terminated, truncated, info = env.step(action)
        assert terminated is True
        assert truncated is False
        assert reward > 0.0
        assert info["official_success"] is True
        assert info["termination_reason"] == "official_success"
    finally:
        env.close()


def test_aerial_hard_force_and_time_limit_are_separate() -> None:
    force_env = _env(horizon=3)
    time_env = _env(horizon=1)
    try:
        force_env.reset(seed=2)
        force_action = np.zeros(AERIAL_ACTION_DIM, dtype=np.float32)
        force_action[1] = 1.0
        _, _, terminated, truncated, info = force_env.step(force_action)
        assert terminated is True
        assert truncated is False
        assert info["termination_reason"] == "hard_contact_force"

        time_env.reset(seed=2)
        _, _, terminated, truncated, info = time_env.step(
            np.zeros(AERIAL_ACTION_DIM, dtype=np.float32)
        )
        assert terminated is False
        assert truncated is True
        assert info["termination_reason"] == "time_limit"
    finally:
        force_env.close()
        time_env.close()


def test_aerial_disconnect_after_engagement_terminates() -> None:
    env = _env(horizon=4, dwell_steps=3)
    try:
        env.reset(seed=4)
        engage = np.zeros(AERIAL_ACTION_DIM, dtype=np.float32)
        engage[0] = 1.0
        _, _, terminated, _, _ = env.step(engage)
        assert terminated is False
        _, _, terminated, truncated, info = env.step(
            np.zeros(AERIAL_ACTION_DIM, dtype=np.float32)
        )
        assert terminated is True
        assert truncated is False
        assert info["termination_reason"] == "disconnect_after_engagement"
    finally:
        env.close()


def test_aerial_hybrid_rgbd_and_registered_entry_point() -> None:
    env = gym.make(
        AERIAL_ENV_ID,
        backend_factory=_factory,
        backend_id="fake_aerial_backend_v1",
        observation_mode="hybrid_rgbd",
        image_size=(32, 40),
        horizon=1,
    )
    try:
        observation, info = env.reset(seed=3)
        assert set(observation) == {"state", "rgb", "depth"}
        assert observation["state"].shape == (AERIAL_STATE_DIM,)
        assert observation["rgb"].shape == (32, 40, 3)
        assert observation["depth"].shape == (32, 40, 1)
        assert env.observation_space.contains(observation)
        assert env.spec is not None
        assert env.spec.id == AERIAL_ENV_ID
        assert info["contract"]["camera_name"] == "boom_tip_camera"
    finally:
        env.close()


def test_aerial_contract_hash_changes_with_motion_panel() -> None:
    static_env = _env(cases=make_aerial_cases(1), horizon=2)
    moving_env = _env(cases=make_aerial_cases(8), horizon=2)
    try:
        assert static_env.contract_sha256 != moving_env.contract_sha256
    finally:
        static_env.close()
        moving_env.close()


def test_aerial_synergy_action_expands_to_smooth_19d_control() -> None:
    base = _env(horizon=1)
    env = AerialSynergyActionWrapper(base)
    try:
        env.reset(seed=9)
        action = np.asarray((0.1, -0.2, 0.3, 0.4), dtype=np.float32)
        env.step(action)
        assert base._backend is not None
        expected = np.empty(AERIAL_ACTION_DIM, dtype=np.float32)
        expected[:18] = np.tile(action[:3], 6)
        expected[18] = action[3]
        np.testing.assert_array_equal(base._backend.action, expected)
        assert env.action_space.shape == (4,)
        assert env.contract_sha256 != base.contract_sha256
    finally:
        env.close()


def test_rigid_aerial_contract_and_registered_entry_point() -> None:
    cases = make_rigid_aerial_cases(2, seed=20260807)
    env = gym.make(
        RIGID_AERIAL_ENV_ID,
        cases=cases,
        backend_factory=_factory,
        backend_id="fake_rigid_backend_v1",
        horizon=1,
    )
    try:
        observation, info = env.reset(seed=11, options={"case": 0})
        assert observation.shape == (AERIAL_STATE_DIM,)
        assert env.action_space.shape == (RIGID_AERIAL_ACTION_DIM,)
        assert info["contract"]["action_dim"] == RIGID_AERIAL_ACTION_DIM
        assert info["contract"]["fidelity_claim"] == (
            "uncalibrated_engineering_simulator"
        )
        assert info["contract"]["runtime_fsm"] is False
        assert cases["0"].receptacle_position_m == (0.0, -9.20, 0.0)
        _, _, terminated, truncated, info = env.step(
            np.zeros(RIGID_AERIAL_ACTION_DIM, dtype=np.float32)
        )
        assert terminated is False
        assert truncated is True
        assert info["mechanism_integrity_valid"] is True
        assert env.spec is not None
        assert env.spec.id == RIGID_AERIAL_ENV_ID
    finally:
        env.close()


def test_continuum_synergy_wrapper_rejects_rigid_mechanism() -> None:
    env = RigidFlyingBoomEnv(
        cases=make_rigid_aerial_cases(1),
        backend_factory=_factory,
        backend_id="fake_rigid_backend_v1",
        horizon=1,
    )
    try:
        with pytest.raises(ValueError, match="19D continuum"):
            AerialSynergyActionWrapper(env)
    finally:
        env.close()


def test_rigid_state_scale_is_fixed_and_markov() -> None:
    base = RigidFlyingBoomEnv(
        cases=make_rigid_aerial_cases(1),
        backend_factory=_factory,
        backend_id="fake_rigid_backend_v1",
        horizon=1,
    )
    env = RigidStateScaleWrapper(base)
    try:
        observation, _ = env.reset(seed=13)
        assert observation.shape == (AERIAL_STATE_DIM,)
        assert observation.dtype == np.float32
        assert env.observation_space.contains(observation)
        assert env.contract["stateful"] is False
        assert env.contract["history"] is False
        assert env.contract_sha256 != base.contract_sha256
    finally:
        env.close()


def test_aerial_sync_vector_env() -> None:
    vector = make_aerial_vector_env(
        2,
        {
            "backend_factory": _factory,
            "backend_id": "fake_aerial_backend_v1",
            "horizon": 1,
        },
        asynchronous=False,
    )
    try:
        observations, _ = vector.reset(seed=10)
        assert observations.shape == (2, AERIAL_STATE_DIM)
        _, _, terminated, truncated, _ = vector.step(
            np.zeros((2, AERIAL_ACTION_DIM), dtype=np.float32)
        )
        assert not np.any(terminated)
        assert np.all(truncated)
    finally:
        vector.close()


def test_aerial_async_vector_env() -> None:
    vector = make_aerial_vector_env(
        2,
        {
            "backend_factory": _factory,
            "backend_id": "fake_aerial_backend_v1",
            "horizon": 1,
        },
        asynchronous=True,
    )
    try:
        observations, _ = vector.reset(seed=20)
        assert observations.shape == (2, AERIAL_STATE_DIM)
        _, _, terminated, truncated, _ = vector.step(
            np.zeros((2, AERIAL_ACTION_DIM), dtype=np.float32)
        )
        assert not np.any(terminated)
        assert np.all(truncated)
    finally:
        vector.close()
