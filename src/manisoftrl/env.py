from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from PIL import Image

from .blender_depth import render_metric_depth
from .runtime import configure_manisoft_runtime


OBSERVATION_DIM = 85
ARM_ACTION_DIM = 18
ACTION_DIM = 19
CONTRACT_ID = "manisoftrl-coll-85d19d-v1"

STATE_LAYOUT: dict[str, slice] = {
    "soft_arm_positions_10x3": slice(0, 30),
    "soft_arm_velocities_10x3": slice(30, 60),
    "tip_position": slice(60, 63),
    "target_position": slice(63, 66),
    "basket_center": slice(66, 69),
    "basket_bbox": slice(69, 75),
    "gripper_position": slice(75, 78),
    "target_minus_tip": slice(78, 81),
    "basket_minus_target": slice(81, 84),
    "gripper_closed": slice(84, 85),
}

_REWARD_KEYS = (
    "potential_delta",
    "arm_action_penalty",
    "closed_far_penalty",
    "success_bonus",
    "total",
)


class CollectionEpisode(Protocol):
    def state(self) -> npt.NDArray[np.float32]: ...

    def initial_info(self) -> dict[str, Any]: ...

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, dict[str, Any]]: ...

    def render(self, camera_name: str | None) -> npt.NDArray[np.uint8]: ...

    def render_rgbd(
        self, camera_name: str | None
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]: ...

    def close(self) -> None: ...


EpisodeFactory = Callable[[Path, int, bool], CollectionEpisode]


def _case_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def discover_case_dirs(case_root: str | Path) -> dict[str, Path]:
    """Discover one case or the immediate case directories below a root."""
    root = Path(case_root).expanduser().resolve(strict=True)
    if (root / "environment.yaml").is_file():
        return {root.name: root}

    cases = {
        child.name: child.resolve()
        for child in root.iterdir()
        if child.is_dir() and (child / "environment.yaml").is_file()
    }
    if not cases:
        raise ValueError(
            f"no COLL cases found under {root}; expected environment.yaml in "
            "the root or its immediate child directories"
        )
    return dict(sorted(cases.items(), key=lambda item: _case_sort_key(item[0])))


def _normalize_case_dirs(
    value: Mapping[str | int, str | Path] | str | Path,
) -> dict[str, Path]:
    if isinstance(value, (str, Path)):
        return discover_case_dirs(value)

    result: dict[str, Path] = {}
    for raw_case, raw_path in value.items():
        case = str(raw_case)
        if case in result:
            raise ValueError(f"duplicate normalized case id {case!r}")
        path = Path(raw_path).expanduser().resolve(strict=True)
        if not (path / "environment.yaml").is_file():
            raise FileNotFoundError(path / "environment.yaml")
        result[case] = path
    if not result:
        raise ValueError("case_dirs cannot be empty")
    return dict(sorted(result.items(), key=lambda item: _case_sort_key(item[0])))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ManiSoftCollectionEpisode:
    """The proven Route-B 85D/19D COLL loop, without its SAC trainer."""

    def __init__(self, task_dir: Path, seed: int, enable_renderer: bool) -> None:
        if enable_renderer and importlib.util.find_spec("bpy") is None:
            raise RuntimeError(
                "visual observation requires ManiSoft's Blender renderer and an "
                "interpreter with bpy; state mode does not require bpy"
            )

        try:
            from manisoft.envs import COLLEnvironment
            from manisoft.muscle import SplineMuscle
            from manisoft.utils import load_yaml
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "ManiSoft runtime is unavailable. Pass runtime_root/asset_root "
                "or install ManiSoft and its simulator dependencies."
            ) from error

        self.task_dir = task_dir
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        config = copy.deepcopy(load_yaml(task_dir / "environment.yaml"))
        config.setdefault("environment", {})["rng"] = np.random.default_rng(seed)
        renderer_config = config.get("renderer")
        if enable_renderer:
            if not renderer_config or not renderer_config.get("cameras"):
                raise RuntimeError(
                    f"vision requested but {task_dir / 'environment.yaml'} has no "
                    "renderer camera"
                )
            self._temp_dir = tempfile.TemporaryDirectory(prefix="manisoftrl-render-")
            for camera in renderer_config["cameras"]:
                (Path(self._temp_dir.name) / str(camera["name"])).mkdir(parents=True)
        else:
            config.pop("renderer", None)

        try:
            self.env = COLLEnvironment.from_dict(config)
        except NameError as error:
            self.close()
            raise RuntimeError(
                "ManiSoft renderer could not start because bpy is unavailable"
            ) from error

        robot = self.env._backend.softrobot_state
        self.muscle = SplineMuscle(
            robot_length=float(np.sum(robot.element_lengths)),
            robot_num_elements=int(robot.num_elements),
            number_of_control_points=6,
            muscle_torque_scale=75.0,
        )
        self.previous_potential = self._potential()

    def _physical(
        self,
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        backend = self.env._backend
        target = np.asarray(
            backend.object_positions[int(self.env._target_object_id)],
            dtype=np.float64,
        )
        basket = np.asarray(
            backend.object_bboxes[int(self.env._basket_id)],
            dtype=np.float64,
        )
        tip = np.asarray(
            backend.softrobot_state.element_positions[-1], dtype=np.float64
        )
        grippers = getattr(backend, "gripper_positions", [])
        gripper = (
            np.asarray(grippers[0], dtype=np.float64) if grippers else tip.copy()
        )
        center = np.mean(basket, axis=1)
        return backend, target, basket, tip, gripper, center

    def state(self) -> npt.NDArray[np.float32]:
        backend, target, basket, tip, gripper, center = self._physical()
        robot = backend.softrobot_state
        positions = np.asarray(robot.element_positions, dtype=np.float64)
        velocities = np.asarray(robot.element_velocities, dtype=np.float64)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or velocities.shape != positions.shape
        ):
            raise RuntimeError("malformed soft-arm state")
        indices = np.linspace(0, positions.shape[0] - 1, 10).astype(int)
        result = np.concatenate(
            (
                positions[indices].reshape(-1),
                velocities[indices].reshape(-1),
                tip,
                target,
                center,
                basket.reshape(-1),
                gripper,
                target - tip,
                center - target,
                np.asarray([float(bool(backend.grasp_state))]),
            )
        ).astype(np.float32)
        if result.shape != (OBSERVATION_DIM,) or not np.all(np.isfinite(result)):
            raise RuntimeError(f"invalid {result.shape} physical observation")
        return result

    def _official(self) -> tuple[bool, np.ndarray, bool]:
        backend, target, basket, _, _, _ = self._physical()
        margins = np.asarray(
            (
                target[0] - basket[0, 0],
                basket[0, 1] - target[0],
                target[1] - basket[1, 0],
                basket[1, 1] - target[1],
                basket[2, 1] - target[2],
            ),
            dtype=np.float64,
        )
        gripper_open = not bool(backend.grasp_state)
        raw = bool(self.env.is_finished())
        if raw != bool(np.all(margins > 0.0) and gripper_open):
            raise RuntimeError("official predicate audit mismatch")
        return raw, margins, gripper_open

    def _potential(self) -> float:
        backend, target, basket, _, gripper, center = self._physical()
        goal = np.asarray((center[0], center[1], basket[2, 1] - 0.01))
        reach_distance = float(np.linalg.norm(gripper - target))
        goal_distance = float(np.linalg.norm(target - goal))
        gripper_closed = float(bool(backend.grasp_state))
        close_near_target = math.exp(-((reach_distance / 0.05) ** 2))
        return (
            -6.0 * reach_distance
            - 3.0 * goal_distance
            + 0.5 * gripper_closed * close_near_target
        )

    def _contact_info(self) -> dict[str, Any]:
        backend = self.env._backend
        target_id = int(self.env._target_object_id)
        soft_contact = False
        gripper_contact = False
        contact_available = False
        try:
            matrix = np.asarray(backend.collision_matrix(), dtype=bool)
            target_index = target_id + 2
            if matrix.ndim == 2 and matrix.shape[0] > target_index:
                soft_contact = bool(matrix[0, target_index])
                gripper_contact = bool(matrix[1, target_index])
                contact_available = True
        except (AttributeError, IndexError, TypeError, ValueError):
            pass

        reader = getattr(
            backend, "gripper_object_contact_forces_for_object", None
        )
        if not callable(reader):
            return {
                "contact_available": contact_available,
                "force_available": False,
                "source": "collision_matrix" if contact_available else "unavailable",
                "target_contact": bool(soft_contact or gripper_contact),
                "soft_arm_contact": soft_contact,
                "gripper_contact": gripper_contact,
                "contact_count": None,
                "force_magnitude_n": None,
                "normal_force_n": None,
                "tangential_force_n": None,
            }

        records = reader(target_id) or []
        magnitude = float(
            sum(float(row["force_magnitude"]) for row in records)
        )
        normal = float(sum(float(row["normal_force"]) for row in records))
        tangential = float(
            sum(float(row["tangential_force_magnitude"]) for row in records)
        )
        values = np.asarray((magnitude, normal, tangential), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("nonfinite target contact-force diagnostic")
        return {
            "contact_available": True,
            "force_available": True,
            "source": "native_gripper_object_contact_forces",
            "target_contact": bool(soft_contact or gripper_contact or records),
            "soft_arm_contact": soft_contact,
            "gripper_contact": bool(gripper_contact or records),
            "contact_count": len(records),
            "force_magnitude_n": magnitude,
            "normal_force_n": normal,
            "tangential_force_n": tangential,
        }

    def _info(self, reward_terms: Mapping[str, float]) -> dict[str, Any]:
        success, margins, gripper_open = self._official()
        _, target, basket, tip, gripper, center = self._physical()
        goal = np.asarray((center[0], center[1], basket[2, 1] - 0.01))
        return {
            "official_success": success,
            "official_margins_m": margins.tolist(),
            "official_open_margin_m": (
                float(np.min(margins)) if gripper_open else None
            ),
            "gripper_open": gripper_open,
            "reach_distance_m": float(np.linalg.norm(gripper - target)),
            "goal_distance_m": float(np.linalg.norm(target - goal)),
            "tip_target_distance_m": float(np.linalg.norm(tip - target)),
            "reward_terms": {key: float(reward_terms[key]) for key in _REWARD_KEYS},
            "contact": self._contact_info(),
        }

    def initial_info(self) -> dict[str, Any]:
        return self._info({key: 0.0 for key in _REWARD_KEYS})

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be finite 19D")
        if float(np.max(np.abs(action))) > 1.0 + 1e-6:
            raise ValueError("action exceeds [-1, 1]")

        lengths = np.asarray(
            self.env._backend.softrobot_state.element_lengths, dtype=np.float64
        )
        torque = self.muscle.activate(
            action[:ARM_ACTION_DIM].reshape(6, 3), lengths
        )
        gripper_command = 1 if action[-1] > 0.0 else 0
        self.env.pick_place_step(
            [tuple(row) for row in np.asarray(torque)], gripper_command
        )

        success, _, _ = self._official()
        potential = self._potential()
        potential_delta = 20.0 * (potential - self.previous_potential)
        action_penalty = -0.0005 * float(
            np.square(action[:ARM_ACTION_DIM]).mean()
        )
        backend, target, _, _, gripper, _ = self._physical()
        reach_distance = float(np.linalg.norm(gripper - target))
        gripper_closed = float(bool(backend.grasp_state))
        closed_far_penalty = (
            -0.05 * gripper_closed * min(reach_distance / 0.1, 1.0)
        )
        success_bonus = 100.0 if success else 0.0
        reward = potential_delta + action_penalty + closed_far_penalty + success_bonus
        reward_terms = {
            "potential_delta": potential_delta,
            "arm_action_penalty": action_penalty,
            "closed_far_penalty": closed_far_penalty,
            "success_bonus": success_bonus,
            "total": reward,
        }
        self.previous_potential = potential
        info = self._info(reward_terms)
        info["action"] = {
            "arm_max_abs": float(np.max(np.abs(action[:ARM_ACTION_DIM]))),
            "gripper_scalar": float(action[-1]),
            "gripper_command": gripper_command,
        }
        return self.state(), float(reward), success, info

    def render(self, camera_name: str | None) -> npt.NDArray[np.uint8]:
        if self._temp_dir is None:
            raise RuntimeError("renderer was not enabled for this episode")
        images, _ = self.env.get_state(
            has_image=True,
            image_folder=Path(self._temp_dir.name),
            save_image=False,
        )
        if not isinstance(images, dict) or not images:
            raise RuntimeError("ManiSoft renderer returned no RGB frame")
        selected = next(iter(images)) if camera_name is None else camera_name
        if selected not in images:
            raise KeyError(
                f"camera {selected!r} not found; available cameras: "
                f"{sorted(images)}"
            )
        frame = np.asarray(images[selected])
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise RuntimeError(f"invalid RGB frame shape {frame.shape}")
        return np.asarray(frame[..., :3], dtype=np.uint8)

    def render_rgbd(
        self, camera_name: str | None
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
        rgb = self.render(camera_name)
        if self._temp_dir is None:
            raise RuntimeError("renderer was not enabled for this episode")
        depth = render_metric_depth(
            camera_name,
            Path(self._temp_dir.name) / "metric-depth",
        )
        return rgb, depth

    def close(self) -> None:
        env = getattr(self, "env", None)
        close = getattr(env, "close", None)
        if callable(close):
            close()
        temp_dir = getattr(self, "_temp_dir", None)
        if temp_dir is not None:
            temp_dir.cleanup()
            self._temp_dir = None


class ManiSoftRLEnv(gym.Env):
    """Standard Gymnasium facade for the Route-B COLL physical task."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        case_dirs: Mapping[str | int, str | Path] | str | Path,
        *,
        horizon: int = 1500,
        default_case: str | int | None = None,
        observation_mode: str = "state",
        render_mode: str | None = None,
        image_size: tuple[int, int] = (128, 128),
        camera_name: str | None = "third",
        depth_max_m: float = 10.0,
        runtime_root: str | Path | None = None,
        asset_root: str | Path | None = None,
        episode_factory: EpisodeFactory | None = None,
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError("horizon must be positive")
        observation_modes = {
            "state",
            "rgb",
            "depth",
            "rgbd",
            "hybrid",
            "hybrid_rgbd",
        }
        if observation_mode not in observation_modes:
            raise ValueError(
                "observation_mode must be state, rgb, depth, rgbd, hybrid, "
                "or hybrid_rgbd"
            )
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")
        if len(image_size) != 2 or min(image_size) < 1:
            raise ValueError("image_size must be positive (height, width)")
        if not math.isfinite(depth_max_m) or depth_max_m <= 0.0:
            raise ValueError("depth_max_m must be finite and positive")
        if asset_root is not None and runtime_root is None:
            raise ValueError("asset_root requires runtime_root")
        if runtime_root is not None:
            configure_manisoft_runtime(runtime_root, asset_root)

        self._case_dirs = _normalize_case_dirs(case_dirs)
        self.horizon = int(horizon)
        self.default_case = None if default_case is None else str(default_case)
        if self.default_case is not None and self.default_case not in self._case_dirs:
            raise KeyError(
                f"default case {self.default_case!r} not in {tuple(self._case_dirs)}"
            )
        self.observation_mode = observation_mode
        self.render_mode = render_mode
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.camera_name = camera_name
        self.depth_max_m = float(depth_max_m)
        self._episode_factory = episode_factory or _ManiSoftCollectionEpisode
        self._episode: CollectionEpisode | None = None
        self._episode_done = True
        self._step_count = 0
        self._case_id: str | None = None
        self._episode_seed: int | None = None
        self._last_rgb: npt.NDArray[np.uint8] | None = None
        self._last_depth: npt.NDArray[np.float32] | None = None

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
        finite_limit = np.finfo(np.float32).max
        state_space = spaces.Box(
            low=-finite_limit,
            high=finite_limit,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )
        rgb_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.image_size[0], self.image_size[1], 3),
            dtype=np.uint8,
        )
        depth_space = spaces.Box(
            low=0.0,
            high=self.depth_max_m,
            shape=(self.image_size[0], self.image_size[1], 1),
            dtype=np.float32,
        )
        if observation_mode == "state":
            self.observation_space = state_space
        elif observation_mode == "rgb":
            self.observation_space = rgb_space
        elif observation_mode == "depth":
            self.observation_space = depth_space
        elif observation_mode == "rgbd":
            self.observation_space = spaces.Dict(
                {"rgb": rgb_space, "depth": depth_space}
            )
        elif observation_mode == "hybrid":
            self.observation_space = spaces.Dict(
                {"state": state_space, "rgb": rgb_space}
            )
        else:
            self.observation_space = spaces.Dict(
                {"state": state_space, "rgb": rgb_space, "depth": depth_space}
            )

    @property
    def cases(self) -> tuple[str, ...]:
        return tuple(self._case_dirs)

    @property
    def selected_case(self) -> str | None:
        return self._case_id

    @property
    def contract(self) -> dict[str, Any]:
        """Return the policy-facing environment contract as JSON-safe data."""
        uses_state = self.observation_mode in {"state", "hybrid", "hybrid_rgbd"}
        uses_rgb = self.observation_mode in {"rgb", "rgbd", "hybrid", "hybrid_rgbd"}
        uses_depth = self.observation_mode in {"depth", "rgbd", "hybrid_rgbd"}
        observation_labels = {
            "state": "85D_current_physical_state",
            "rgb": "rgb_uint8",
            "depth": "metric_depth_float32_metres",
            "rgbd": "rgb_uint8_plus_metric_depth_float32_metres",
            "hybrid": "85D_state_plus_rgb_uint8",
            "hybrid_rgbd": "85D_state_plus_rgb_uint8_plus_metric_depth_float32_metres",
        }
        return {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "observation": observation_labels[self.observation_mode],
            "observation_mode": self.observation_mode,
            "state_dim": OBSERVATION_DIM if uses_state else None,
            "rgb_shape": [*self.image_size, 3] if uses_rgb else None,
            "rgb_dtype": "uint8" if uses_rgb else None,
            "depth_shape": [*self.image_size, 1] if uses_depth else None,
            "depth_dtype": "float32" if uses_depth else None,
            "depth_units": "metres" if uses_depth else None,
            "depth_max_m": self.depth_max_m if uses_depth else None,
            "camera_name": self.camera_name if uses_rgb or uses_depth else None,
            "action": "18D_soft_arm_plus_1D_gripper",
            "action_dim": ACTION_DIM,
            "action_range": [-1.0, 1.0],
            "reward": "route_b_potential_delta_v1",
            "official_success": "raw_COLLEnvironment.is_finished",
            "horizon": self.horizon,
            "case_in_observation": False,
            "stage_in_observation": False,
            "clock_in_observation": False,
            "runtime_fsm": False,
            "runtime_waypoints": False,
            "runtime_expert_actions": False,
        }

    @property
    def contract_sha256(self) -> str:
        payload = json.dumps(
            self.contract,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _select_case(self, options: Mapping[str, Any]) -> str:
        unknown = set(options) - {"case"}
        if unknown:
            raise ValueError(f"unsupported reset options: {sorted(unknown)}")
        requested = options.get("case", self.default_case)
        if requested is None:
            index = int(self.np_random.integers(0, len(self._case_dirs)))
            return self.cases[index]
        case = str(requested)
        if case not in self._case_dirs:
            raise KeyError(f"unknown case {case!r}; available cases: {self.cases}")
        return case

    @staticmethod
    def _validate_state(value: npt.ArrayLike) -> npt.NDArray[np.float32]:
        state = np.asarray(value, dtype=np.float32).reshape(-1)
        if state.shape != (OBSERVATION_DIM,) or not np.all(np.isfinite(state)):
            raise RuntimeError(f"episode returned invalid state {state.shape}")
        return state

    @staticmethod
    def _validate_action(value: npt.ArrayLike) -> npt.NDArray[np.float32]:
        action = np.asarray(value, dtype=np.float32).reshape(-1)
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be finite 19D")
        if float(np.max(np.abs(action))) > 1.0 + 1e-6:
            raise ValueError("action exceeds [-1, 1]")
        return action

    def _prepare_rgb(self, value: npt.ArrayLike) -> npt.NDArray[np.uint8]:
        frame = np.asarray(value)
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise RuntimeError(f"episode returned invalid RGB shape {frame.shape}")
        frame = np.asarray(frame[..., :3], dtype=np.uint8)
        expected = (self.image_size[0], self.image_size[1])
        if frame.shape[:2] != expected:
            image = Image.fromarray(frame)
            image = image.resize(
                (self.image_size[1], self.image_size[0]),
                resample=Image.Resampling.BILINEAR,
            )
            frame = np.asarray(image, dtype=np.uint8)
        self._last_rgb = frame.copy()
        return frame.copy()

    def _prepare_depth(self, value: npt.ArrayLike) -> npt.NDArray[np.float32]:
        depth = np.asarray(value, dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[..., None]
        if depth.ndim != 3 or depth.shape[2] != 1:
            raise RuntimeError(
                f"episode returned invalid depth shape {depth.shape}; expected HxWx1"
            )
        if np.any(np.isnan(depth)) or np.any(depth < 0.0):
            raise RuntimeError("episode returned negative or NaN depth")
        depth = np.nan_to_num(
            depth,
            nan=0.0,
            posinf=self.depth_max_m,
            neginf=0.0,
        )
        depth = np.clip(depth, 0.0, self.depth_max_m)
        expected = (self.image_size[0], self.image_size[1])
        if depth.shape[:2] != expected:
            image = Image.fromarray(depth[..., 0])
            image = image.resize(
                (self.image_size[1], self.image_size[0]),
                resample=Image.Resampling.NEAREST,
            )
            depth = np.asarray(image, dtype=np.float32)[..., None]
        self._last_depth = depth.copy()
        return depth.copy()

    def _rgb(self) -> npt.NDArray[np.uint8]:
        if self._episode is None:
            raise RuntimeError("reset must be called before rendering")
        return self._prepare_rgb(self._episode.render(self.camera_name))

    def _rgbd(
        self,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
        if self._episode is None:
            raise RuntimeError("reset must be called before rendering")
        render_rgbd = getattr(self._episode, "render_rgbd", None)
        if not callable(render_rgbd):
            raise RuntimeError(
                "the selected episode renderer does not expose metric depth"
            )
        rgb, depth = render_rgbd(self.camera_name)
        return self._prepare_rgb(rgb), self._prepare_depth(depth)

    def _observation(self, state: npt.NDArray[np.float32]) -> Any:
        if self.observation_mode == "state":
            return state.copy()
        if self.observation_mode == "rgb":
            return self._rgb()
        if self.observation_mode == "hybrid":
            return {"state": state.copy(), "rgb": self._rgb()}
        rgb, depth = self._rgbd()
        if self.observation_mode == "depth":
            return depth
        if self.observation_mode == "rgbd":
            return {"rgb": rgb, "depth": depth}
        return {"state": state.copy(), "rgb": rgb, "depth": depth}

    def _platform_info(
        self,
        info: Mapping[str, Any],
        termination_reason: str | None,
    ) -> dict[str, Any]:
        if self._case_id is None or self._episode_seed is None:
            raise RuntimeError("episode metadata is unavailable")
        task_dir = self._case_dirs[self._case_id]
        contract = self.contract
        result = dict(info)
        result.update(
            {
                "case_id": self._case_id,
                "episode_seed": self._episode_seed,
                "task_dir": str(task_dir),
                "scene_sha256": _sha256_file(task_dir / "environment.yaml"),
                "step_count": self._step_count,
                "horizon": self.horizon,
                "termination_reason": termination_reason,
                "contract": contract,
                "contract_sha256": self.contract_sha256,
            }
        )
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        self.close()
        case = self._select_case(options or {})
        episode_seed = (
            int(seed)
            if seed is not None
            else int(self.np_random.integers(0, np.iinfo(np.uint32).max))
        )
        needs_renderer = self.observation_mode != "state" or (
            self.render_mode == "rgb_array"
        )
        episode = self._episode_factory(
            self._case_dirs[case], episode_seed, needs_renderer
        )
        self._episode = episode
        self._episode_done = False
        self._step_count = 0
        self._case_id = case
        self._episode_seed = episode_seed
        self._last_rgb = None
        self._last_depth = None
        state = self._validate_state(episode.state())
        observation = self._observation(state)
        info = self._platform_info(episode.initial_info(), None)
        return observation, info

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._episode is None:
            raise RuntimeError("reset must be called before step")
        if self._episode_done:
            raise RuntimeError("episode is finished; call reset before step")
        validated = self._validate_action(action)
        state, reward, success, info = self._episode.step(validated)
        state = self._validate_state(state)
        if not np.isfinite(reward):
            raise RuntimeError("episode returned nonfinite reward")
        self._step_count += 1
        terminated = bool(success)
        truncated = bool(not terminated and self._step_count >= self.horizon)
        reason = "official_success" if terminated else "time_limit" if truncated else None
        self._episode_done = terminated or truncated
        self._last_rgb = None
        self._last_depth = None
        observation = self._observation(state)
        result_info = self._platform_info(info, reason)
        return observation, float(reward), terminated, truncated, result_info

    def render(self) -> npt.NDArray[np.uint8]:
        if self.render_mode != "rgb_array":
            raise RuntimeError("construct the environment with render_mode='rgb_array'")
        if self._last_rgb is not None:
            return self._last_rgb.copy()
        return self._rgb()

    def close(self) -> None:
        if self._episode is not None:
            self._episode.close()
        self._episode = None
        self._episode_done = True
        self._last_rgb = None
        self._last_depth = None
