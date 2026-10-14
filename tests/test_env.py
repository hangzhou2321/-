from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from manisoftrl import ENV_ID, ManiSoftRLEnv, make_vector_env
from manisoftrl.env import OBSERVATION_DIM, STATE_LAYOUT, _ManiSoftCollectionEpisode
from manisoftrl.runtime import _install_optional_import_shims


def test_optional_import_shims_are_idempotent() -> None:
    _install_optional_import_shims()
    _install_optional_import_shims()


def _make_cases(tmp_path: Path, count: int = 3) -> Path:
    root = tmp_path / "cases"
    for case in range(count):
        case_dir = root / str(case)
        case_dir.mkdir(parents=True)
        (case_dir / "environment.yaml").write_text(
            f"case: {case}\n", encoding="utf-8"
        )
    return root


class FakeEpisode:
    def __init__(self, task_dir: Path, seed: int, enable_renderer: bool) -> None:
        self.task_dir = task_dir
        self.seed = seed
        self.enable_renderer = enable_renderer
        self.steps = 0
        self.closed = False

    def state(self) -> np.ndarray:
        state = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        state[0] = float(int(self.task_dir.name))
        state[1] = float(self.seed % 997)
        state[2] = float(self.steps)
        return state

    def initial_info(self) -> dict[str, Any]:
        return {
            "official_success": False,
            "reward_terms": {"total": 0.0},
            "contact": {"force_available": False},
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        self.steps += 1
        success = bool(action[-1] > 0.5)
        reward = float(action.sum())
        return self.state(), reward, success, {
            "official_success": success,
            "reward_terms": {"total": reward},
            "contact": {"force_available": False},
        }

    def render(self, camera_name: str | None) -> np.ndarray:
        if not self.enable_renderer:
            raise RuntimeError("renderer disabled")
        value = (17 if camera_name == "third" else 9) + self.steps
        return np.full((4, 5, 3), value, dtype=np.uint8)

    def render_rgbd(
        self, camera_name: str | None
    ) -> tuple[np.ndarray, np.ndarray]:
        rgb = self.render(camera_name)
        depth = np.full((4, 5, 1), 1.5 + self.steps, dtype=np.float32)
        return rgb, depth

    def close(self) -> None:
        self.closed = True


def _fake_factory(
    task_dir: Path, seed: int, enable_renderer: bool
) -> FakeEpisode:
    return FakeEpisode(task_dir, seed, enable_renderer)


def test_state_env_is_gymnasium_compliant_and_case_is_public(tmp_path: Path) -> None:
    root = _make_cases(tmp_path)
    env = ManiSoftRLEnv(
        root,
        horizon=2,
        default_case=1,
        episode_factory=_fake_factory,
    )
    try:
        observation, info = env.reset(seed=123, options={"case": 2})
        assert observation.shape == (85,)
        assert observation.dtype == np.float32
        assert info["case_id"] == "2"
        assert info["episode_seed"] == 123
        assert info["contract"]["clock_in_observation"] is False
        assert info["contract"] == env.contract
        assert info["contract_sha256"] == env.contract_sha256
        assert len(info["contract_sha256"]) == 64
        assert env.action_space.shape == (19,)
        assert env.observation_space.shape == (85,)

        _, _, terminated, truncated, info = env.step(
            np.zeros(19, dtype=np.float32)
        )
        assert terminated is False
        assert truncated is False
        assert info["termination_reason"] is None

        _, _, terminated, truncated, info = env.step(
            np.zeros(19, dtype=np.float32)
        )
        assert terminated is False
        assert truncated is True
        assert info["termination_reason"] == "time_limit"
        with pytest.raises(RuntimeError, match="finished"):
            env.step(np.zeros(19, dtype=np.float32))
    finally:
        env.close()


def test_registered_gymnasium_entry_point(tmp_path: Path) -> None:
    env = gym.make(
        ENV_ID,
        case_dirs=_make_cases(tmp_path, 1),
        horizon=1,
        episode_factory=_fake_factory,
    )
    try:
        observation, info = env.reset(seed=5)
        assert observation.shape == (85,)
        assert env.spec is not None
        assert env.spec.id == ENV_ID
        assert info["contract_sha256"] == env.unwrapped.contract_sha256
    finally:
        env.close()


def test_contract_hash_changes_with_policy_facing_contract(tmp_path: Path) -> None:
    root = _make_cases(tmp_path, 1)
    state_env = ManiSoftRLEnv(root, horizon=2, episode_factory=_fake_factory)
    visual_env = ManiSoftRLEnv(
        root,
        horizon=2,
        observation_mode="hybrid_rgbd",
        image_size=(6, 8),
        episode_factory=_fake_factory,
    )
    try:
        assert state_env.contract_sha256 != visual_env.contract_sha256
        assert state_env.contract["camera_name"] is None
        assert visual_env.contract["rgb_shape"] == [6, 8, 3]
        assert visual_env.contract["depth_shape"] == [6, 8, 1]
    finally:
        state_env.close()
        visual_env.close()


def test_official_success_terminates_but_does_not_truncate(tmp_path: Path) -> None:
    env = ManiSoftRLEnv(
        _make_cases(tmp_path, 1), horizon=1, episode_factory=_fake_factory
    )
    try:
        env.reset(seed=7)
        action = np.zeros(19, dtype=np.float32)
        action[-1] = 1.0
        _, _, terminated, truncated, info = env.step(action)
        assert terminated is True
        assert truncated is False
        assert info["official_success"] is True
        assert info["termination_reason"] == "official_success"
    finally:
        env.close()


def test_reset_seed_repeats_case_and_physics_seed(tmp_path: Path) -> None:
    env = ManiSoftRLEnv(
        _make_cases(tmp_path), horizon=2, episode_factory=_fake_factory
    )
    try:
        first, first_info = env.reset(seed=991)
        second, second_info = env.reset(seed=991)
        np.testing.assert_array_equal(first, second)
        assert first_info["case_id"] == second_info["case_id"]
        assert first_info["episode_seed"] == second_info["episode_seed"] == 991
    finally:
        env.close()


def test_env_checker_and_action_validation(tmp_path: Path) -> None:
    env = ManiSoftRLEnv(
        _make_cases(tmp_path, 1), horizon=3, episode_factory=_fake_factory
    )
    try:
        check_env(env, skip_render_check=True)
        env.reset(seed=1)
        with pytest.raises(ValueError, match="19D"):
            env.step(np.zeros(18, dtype=np.float32))
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            env.step(np.full(19, 2.0, dtype=np.float32))
    finally:
        env.close()


def test_rgb_hybrid_and_render_modes(tmp_path: Path) -> None:
    root = _make_cases(tmp_path, 1)
    rgb_env = ManiSoftRLEnv(
        root,
        observation_mode="rgb",
        image_size=(6, 8),
        episode_factory=_fake_factory,
    )
    hybrid_env = ManiSoftRLEnv(
        root,
        observation_mode="hybrid",
        image_size=(6, 8),
        episode_factory=_fake_factory,
    )
    render_env = ManiSoftRLEnv(
        root,
        render_mode="rgb_array",
        image_size=(6, 8),
        episode_factory=_fake_factory,
    )
    depth_env = ManiSoftRLEnv(
        root,
        observation_mode="depth",
        image_size=(6, 8),
        depth_max_m=2.0,
        episode_factory=_fake_factory,
    )
    rgbd_env = ManiSoftRLEnv(
        root,
        observation_mode="rgbd",
        image_size=(6, 8),
        episode_factory=_fake_factory,
    )
    hybrid_rgbd_env = ManiSoftRLEnv(
        root,
        observation_mode="hybrid_rgbd",
        image_size=(6, 8),
        episode_factory=_fake_factory,
    )
    try:
        rgb, _ = rgb_env.reset(seed=1)
        assert rgb.shape == (6, 8, 3)
        assert rgb.dtype == np.uint8
        assert rgb_env.observation_space.contains(rgb)

        hybrid, _ = hybrid_env.reset(seed=1)
        assert set(hybrid) == {"state", "rgb"}
        assert hybrid["state"].shape == (85,)
        assert hybrid["rgb"].shape == (6, 8, 3)
        assert hybrid_env.observation_space.contains(hybrid)

        state, _ = render_env.reset(seed=1)
        assert state.shape == (85,)
        before = render_env.render()
        assert before.shape == (6, 8, 3)
        render_env.step(np.zeros(19, dtype=np.float32))
        after = render_env.render()
        assert not np.array_equal(before, after)

        depth, depth_info = depth_env.reset(seed=1)
        assert depth.shape == (6, 8, 1)
        assert depth.dtype == np.float32
        assert depth_env.observation_space.contains(depth)
        assert depth_info["contract"]["depth_units"] == "metres"
        depth, *_ = depth_env.step(np.zeros(19, dtype=np.float32))
        assert np.all(depth == 2.0)

        rgbd, _ = rgbd_env.reset(seed=1)
        assert set(rgbd) == {"rgb", "depth"}
        assert rgbd["rgb"].shape == (6, 8, 3)
        assert rgbd["depth"].shape == (6, 8, 1)
        assert rgbd_env.observation_space.contains(rgbd)

        hybrid_rgbd, _ = hybrid_rgbd_env.reset(seed=1)
        assert set(hybrid_rgbd) == {"state", "rgb", "depth"}
        assert hybrid_rgbd["state"].shape == (85,)
        assert hybrid_rgbd_env.observation_space.contains(hybrid_rgbd)
    finally:
        rgb_env.close()
        hybrid_env.close()
        render_env.close()
        depth_env.close()
        rgbd_env.close()
        hybrid_rgbd_env.close()


def test_rgbd_fails_closed_when_episode_has_no_depth(tmp_path: Path) -> None:
    class RgbOnlyEpisode(FakeEpisode):
        render_rgbd = None

    def rgb_only_factory(
        task_dir: Path, seed: int, enable_renderer: bool
    ) -> RgbOnlyEpisode:
        return RgbOnlyEpisode(task_dir, seed, enable_renderer)

    env = ManiSoftRLEnv(
        _make_cases(tmp_path, 1),
        observation_mode="rgbd",
        episode_factory=rgb_only_factory,
    )
    try:
        with pytest.raises(RuntimeError, match="does not expose metric depth"):
            env.reset(seed=1)
    finally:
        env.close()


def test_sync_vector_env(tmp_path: Path) -> None:
    vector = make_vector_env(
        2,
        {
            "case_dirs": _make_cases(tmp_path, 1),
            "horizon": 1,
            "episode_factory": _fake_factory,
        },
        asynchronous=False,
    )
    try:
        observations, _ = vector.reset(seed=10)
        assert observations.shape == (2, 85)
        _, _, terminated, truncated, _ = vector.step(
            np.zeros((2, 19), dtype=np.float32)
        )
        assert not np.any(terminated)
        assert np.all(truncated)
    finally:
        vector.close()


def test_async_vector_env(tmp_path: Path) -> None:
    vector = make_vector_env(
        2,
        {
            "case_dirs": _make_cases(tmp_path, 1),
            "horizon": 1,
            "episode_factory": _fake_factory,
        },
        asynchronous=True,
    )
    try:
        observations, _ = vector.reset(seed=20)
        assert observations.shape == (2, 85)
        _, _, terminated, truncated, _ = vector.step(
            np.zeros((2, 19), dtype=np.float32)
        )
        assert not np.any(terminated)
        assert np.all(truncated)
    finally:
        vector.close()


class _RobotState:
    def __init__(self) -> None:
        self.element_positions = np.arange(63, dtype=np.float64).reshape(21, 3) / 100
        self.element_velocities = np.full((21, 3), 0.25, dtype=np.float64)
        self.element_lengths = np.ones(20, dtype=np.float64)
        self.num_elements = 20


class _PhysicalBackend:
    def __init__(self) -> None:
        self.softrobot_state = _RobotState()
        self.object_positions = [
            np.asarray([1.5, 0.5, 0.5]),
            np.asarray([2.0, 2.0, 2.0]),
            np.asarray([0.5, 0.5, 0.5]),
        ]
        basket = np.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        self.object_bboxes = [basket.copy(), basket.copy(), basket.copy()]
        self.gripper_positions = [np.asarray([1.4, 0.5, 0.5])]
        self.grasp_state = False

    def collision_matrix(self) -> list[list[bool]]:
        matrix = np.zeros((5, 5), dtype=bool)
        matrix[1, 2] = True
        return matrix.tolist()

    def gripper_object_contact_forces_for_object(
        self, object_index: int
    ) -> list[dict[str, float]]:
        assert object_index == 0
        return [
            {
                "force_magnitude": 3.0,
                "normal_force": 2.0,
                "tangential_force_magnitude": 1.0,
            }
        ]


class _PhysicalEnv:
    def __init__(self) -> None:
        self._backend = _PhysicalBackend()
        self._target_object_id = 0
        self._basket_id = 2

    def is_finished(self) -> bool:
        target = self._backend.object_positions[0]
        basket = self._backend.object_bboxes[2]
        return bool(
            basket[0, 0] < target[0] < basket[0, 1]
            and basket[1, 0] < target[1] < basket[1, 1]
            and target[2] < basket[2, 1]
            and not self._backend.grasp_state
        )

    def pick_place_step(self, torque: list[tuple[float, ...]], command: int) -> None:
        assert len(torque) == 20
        assert command == 0
        self._backend.object_positions[0] = np.asarray([0.5, 0.5, 0.5])
        self._backend.gripper_positions[0] = np.asarray([0.5, 0.5, 0.5])


class _Muscle:
    def activate(self, action: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        assert action.shape == (6, 3)
        return np.zeros((len(lengths), 3), dtype=np.float64)


def test_route_b_physical_contract_and_diagnostics() -> None:
    episode = object.__new__(_ManiSoftCollectionEpisode)
    episode.task_dir = Path("case0")
    episode._temp_dir = None
    episode.env = _PhysicalEnv()
    episode.muscle = _Muscle()
    episode.previous_potential = episode._potential()

    state = episode.state()
    assert state.shape == (85,)
    np.testing.assert_array_equal(
        state[STATE_LAYOUT["gripper_closed"]], np.asarray([0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        state[STATE_LAYOUT["target_position"]], [1.5, 0.5, 0.5]
    )
    assert episode.initial_info()["official_success"] is False

    next_state, reward, success, info = episode.step(
        np.zeros(19, dtype=np.float32)
    )
    assert next_state.shape == (85,)
    assert success is True
    assert info["official_success"] is True
    assert info["reward_terms"]["success_bonus"] == 100.0
    assert info["reward_terms"]["total"] == pytest.approx(reward)
    assert info["contact"]["force_available"] is True
    assert info["contact"]["normal_force_n"] == 2.0
