from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from PIL import Image


AERIAL_ACTION_DIM = 19
AERIAL_STATE_DIM = 82
AERIAL_CONTRACT_ID = "manisoftrl-flying-boom-82d19d-v6"
AERIAL_CASE_SET_ID = "manisoftrl-flying-boom-motion-v2"
RIGID_AERIAL_ACTION_DIM = 3
RIGID_AERIAL_CONTRACT_ID = "manisoftrl-rigid-flying-boom-82d3d-v5"
RIGID_AERIAL_CASE_SET_ID = "manisoftrl-rigid-flying-boom-motion-v1"
RIGID_AERIAL_PHYSICS_DT_S = 0.001
RIGID_AERIAL_DEFAULT_UPDATE_INTERVAL = 20
RIGID_BOOM_OUTER_LENGTH_M = 7.057574093
RIGID_BOOM_TELESCOPIC_LENGTH_M = 2.838961916
RIGID_BOOM_INITIAL_EXTENSION_M = 2.02
RIGID_BOOM_EXTENSION_MAX_M = RIGID_BOOM_TELESCOPIC_LENGTH_M
RIGID_BOOM_EXTENSION_RATE_LIMIT_M_S = 0.30
RIGID_RECEPTACLE_POSITION_M = (0.0, -9.20, 0.0)
AERIAL_PHYSICS_DT_S = 0.00005
AERIAL_DEFAULT_UPDATE_INTERVAL = 28
AERIAL_DEFAULT_HORIZON = 800
AERIAL_BOOM_START_M = (0.0, 0.72, 0.90)
AERIAL_BOOM_DIRECTION = (0.0, -1.0, 0.0)
AERIAL_BOOM_LENGTH_M = 1.0
AERIAL_BOOM_RADIUS_M = 0.030
AERIAL_BOOM_NUM_ELEMENTS = 20
AERIAL_BOOM_DENSITY_KG_M3 = 850.0
AERIAL_BOOM_YOUNGS_MODULUS_PA = 10_000_000.0
AERIAL_BOOM_POISSON_RATIO = 0.5
AERIAL_BOOM_DAMPING_CONSTANT = 7.0
AERIAL_MUSCLE_TORQUE_SCALE = 75.0
AERIAL_MUSCLE_ACTIVATION_LIMIT = 0.010
AERIAL_NOZZLE_RETRACTED_LENGTH_M = 0.120
AERIAL_BOOM_EXTENSION_MAX_M = 0.220
AERIAL_BOOM_EXTENSION_COMMAND_MAX_M = 0.190
AERIAL_BOOM_EXTENSION_RATE_LIMIT_M_S = 0.500
AERIAL_PORT_HALF_OPENING_M = 0.050
AERIAL_PORT_TUNNEL_DEPTH_M = 0.120
AERIAL_AIRCRAFT_VISUAL_SCALE = 0.08
AERIAL_F16_REFUEL_ANCHOR_MODEL_M = (0.1054, 0.6500, 0.0)
AERIAL_KC135_BOOM_PIVOT_MODEL_M = (15.10, -0.13, 0.0)
AERIAL_KC135_BOOM_TIP_MODEL_M = (22.61216417, -5.74430750, -0.03131101)
AERIAL_REWARD_ID = "aerial_geometric_potential_v1"
RIGID_AERIAL_REWARD_ID = "rigid_aerial_tracking_potential_v2"

AERIAL_STATE_LAYOUT: dict[str, slice] = {
    "boom_positions_relative_to_receptacle_10x3": slice(0, 30),
    "boom_velocities_relative_to_receptacle_10x3": slice(30, 60),
    "tip_position_relative_to_receptacle": slice(60, 63),
    "tip_velocity_relative_to_receptacle": slice(63, 66),
    "tip_axis": slice(66, 69),
    "receptacle_axis": slice(69, 72),
    "receptacle_linear_velocity": slice(72, 75),
    "receptacle_angular_velocity": slice(75, 78),
    "insertion_depth_m": slice(78, 79),
    "contact_force_ratio": slice(79, 80),
    "boom_extension_m": slice(80, 81),
    "boom_extension_velocity_m_s": slice(81, 82),
}

_BASE_RECEPTACLE_ROTATION = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)


def _array3(value: tuple[float, float, float]) -> npt.NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("motion fields must be finite 3D vectors")
    return result


def _rotation_xyz(angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    roll, pitch, yaw = map(float, angles)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float64)
    ry = np.asarray(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float64)
    rz = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ ry @ rx


def _matrix_to_wxyz(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ),
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            quat = np.asarray(
                (
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ),
                dtype=np.float64,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            quat = np.asarray(
                (
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ),
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            quat = np.asarray(
                (
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ),
                dtype=np.float64,
            )
    quat /= max(float(np.linalg.norm(quat)), 1.0e-12)
    return quat


@dataclass(frozen=True)
class AerialRefuelCase:
    case_id: str
    receptacle_position_m: tuple[float, float, float] = (0.04, -0.52, 0.90)
    translation_amplitude_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translation_frequency_hz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translation_phase_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_amplitude_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_frequency_hz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_phase_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not str(self.case_id):
            raise ValueError("case_id cannot be empty")
        for value in (
            self.receptacle_position_m,
            self.translation_amplitude_m,
            self.translation_frequency_hz,
            self.translation_phase_rad,
            self.rotation_amplitude_rad,
            self.rotation_frequency_hz,
            self.rotation_phase_rad,
        ):
            _array3(value)
        if np.any(_array3(self.translation_amplitude_m) < 0.0):
            raise ValueError("translation amplitudes must be nonnegative")
        if np.any(_array3(self.translation_frequency_hz) < 0.0):
            raise ValueError("translation frequencies must be nonnegative")
        if np.any(_array3(self.rotation_frequency_hz) < 0.0):
            raise ValueError("rotation frequencies must be nonnegative")
        if np.max(np.abs(_array3(self.rotation_amplitude_rad))) > math.radians(15):
            raise ValueError("rotation amplitudes exceed the 15 degree MVP bound")

    def motion(
        self, time_s: float
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        time_s = float(time_s)
        translation_omega = 2.0 * math.pi * _array3(self.translation_frequency_hz)
        translation_angle = translation_omega * time_s + _array3(
            self.translation_phase_rad
        )
        position = _array3(self.receptacle_position_m) + _array3(
            self.translation_amplitude_m
        ) * np.sin(translation_angle)
        linear_velocity = (
            _array3(self.translation_amplitude_m)
            * translation_omega
            * np.cos(translation_angle)
        )

        rotation_omega = 2.0 * math.pi * _array3(self.rotation_frequency_hz)
        rotation_angle = rotation_omega * time_s + _array3(self.rotation_phase_rad)
        euler = _array3(self.rotation_amplitude_rad) * np.sin(rotation_angle)
        euler_velocity = (
            _array3(self.rotation_amplitude_rad)
            * rotation_omega
            * np.cos(rotation_angle)
        )
        rotation = _BASE_RECEPTACLE_ROTATION @ _rotation_xyz(euler)
        axis = rotation[:, 2].copy()
        angular_velocity = _BASE_RECEPTACLE_ROTATION @ euler_velocity
        quaternion = _matrix_to_wxyz(rotation)
        return position, axis, linear_velocity, angular_velocity, quaternion

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_aerial_cases(
    count: int = 8, *, seed: int = 20260807
) -> dict[str, AerialRefuelCase]:
    """Create the versioned deterministic motion panel; case 0 is static."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(int(seed))
    result: dict[str, AerialRefuelCase] = {"0": AerialRefuelCase(case_id="0")}
    for index in range(1, count):
        translation_amplitude = rng.uniform(
            (0.004, 0.002, 0.004), (0.020, 0.010, 0.020)
        )
        translation_frequency = rng.uniform(0.25, 1.00, size=3)
        translation_phase = rng.uniform(0.0, 2.0 * math.pi, size=3)
        rotation_amplitude = np.deg2rad(rng.uniform(0.5, 3.0, size=3))
        rotation_frequency = rng.uniform(0.20, 0.80, size=3)
        rotation_phase = rng.uniform(0.0, 2.0 * math.pi, size=3)
        result[str(index)] = AerialRefuelCase(
            case_id=str(index),
            translation_amplitude_m=tuple(map(float, translation_amplitude)),
            translation_frequency_hz=tuple(map(float, translation_frequency)),
            translation_phase_rad=tuple(map(float, translation_phase)),
            rotation_amplitude_rad=tuple(map(float, rotation_amplitude)),
            rotation_frequency_hz=tuple(map(float, rotation_frequency)),
            rotation_phase_rad=tuple(map(float, rotation_phase)),
        )
    return result


def make_rigid_aerial_cases(
    count: int = 8, *, seed: int = 20260807
) -> dict[str, AerialRefuelCase]:
    """Create rigid-boom canaries around the full-scale receiver anchor."""
    return {
        case_id: replace(spec, receptacle_position_m=RIGID_RECEPTACLE_POSITION_M)
        for case_id, spec in make_aerial_cases(count, seed=seed).items()
    }


def _case_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _case_set_sha256(cases: Mapping[str, AerialRefuelCase]) -> str:
    payload = {
        case: cases[case].to_dict()
        for case in sorted(cases, key=_case_sort_key)
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def _boom_tip_asset_sha256() -> str:
    path = Path(__file__).resolve().parent / "assets" / "boom_tip" / "config.xml"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class AerialSnapshot:
    boom_positions_m: npt.NDArray[np.float64]
    boom_velocities_m_s: npt.NDArray[np.float64]
    tip_position_m: npt.NDArray[np.float64]
    tip_velocity_m_s: npt.NDArray[np.float64]
    tip_axis: npt.NDArray[np.float64]
    receptacle_position_m: npt.NDArray[np.float64]
    receptacle_axis: npt.NDArray[np.float64]
    receptacle_linear_velocity_m_s: npt.NDArray[np.float64]
    receptacle_angular_velocity_rad_s: npt.NDArray[np.float64]
    insertion_depth_m: float
    lateral_error_m: float
    orientation_error_rad: float
    contact_force_n: float
    step_peak_contact_force_n: float
    episode_peak_contact_force_n: float
    boom_extension_m: float
    boom_extension_velocity_m_s: float
    continuum_integrity_valid: bool
    physics_time_s: float


class AerialBackend(Protocol):
    def snapshot(self) -> AerialSnapshot: ...

    def advance(self, action: npt.NDArray[np.float32]) -> None: ...

    def render_rgbd(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]: ...

    def close(self) -> None: ...


AerialBackendFactory = Callable[
    [AerialRefuelCase, int, bool, tuple[int, int], str, int], AerialBackend
]


def _normalize_cases(
    value: Mapping[str | int, AerialRefuelCase] | None,
) -> dict[str, AerialRefuelCase]:
    source = make_aerial_cases() if value is None else value
    result: dict[str, AerialRefuelCase] = {}
    for raw_case, spec in source.items():
        case = str(raw_case)
        if case in result:
            raise ValueError(f"duplicate normalized case id {case!r}")
        if not isinstance(spec, AerialRefuelCase):
            raise TypeError("cases must map ids to AerialRefuelCase values")
        if spec.case_id != case:
            raise ValueError(f"case key {case!r} does not match spec {spec.case_id!r}")
        result[case] = spec
    if not result:
        raise ValueError("cases cannot be empty")
    return dict(sorted(result.items(), key=lambda item: _case_sort_key(item[0])))


class AerialRefuelEnv(gym.Env):
    """Terminal docking shared by the continuum and rigid flying-boom backends."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        cases: Mapping[str | int, AerialRefuelCase] | None = None,
        default_case: str | int | None = "0",
        runtime_root: str | Path | None = None,
        asset_root: str | Path | None = None,
        aircraft_visual_root: str | Path | None = None,
        horizon: int = AERIAL_DEFAULT_HORIZON,
        update_interval: int = AERIAL_DEFAULT_UPDATE_INTERVAL,
        observation_mode: str = "state",
        render_mode: str | None = None,
        image_size: tuple[int, int] = (128, 128),
        camera_name: str = "boom_tip_camera",
        depth_max_m: float = 5.0,
        required_insertion_depth_m: float = 0.050,
        lateral_tolerance_m: float = 0.035,
        orientation_tolerance_rad: float = math.radians(12.0),
        safe_contact_force_n: float = 15.0,
        hard_contact_force_n: float = 30.0,
        dwell_steps: int = 5,
        workspace_radius_m: float = 0.60,
        reward_scale: float = 1.0,
        actuation_mode: str = "manisoft_continuum",
        backend_factory: AerialBackendFactory | None = None,
        backend_id: str | None = None,
    ) -> None:
        super().__init__()
        if horizon < 1 or update_interval < 1 or dwell_steps < 1:
            raise ValueError("horizon, update_interval, and dwell_steps must be positive")
        observation_modes = {
            "state",
            "rgb",
            "depth",
            "rgbd",
            "hybrid",
            "hybrid_rgbd",
        }
        if observation_mode not in observation_modes:
            raise ValueError(f"unsupported observation_mode {observation_mode!r}")
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")
        if len(image_size) != 2 or min(image_size) < 1:
            raise ValueError("image_size must be positive (height, width)")
        thresholds = np.asarray(
            (
                depth_max_m,
                required_insertion_depth_m,
                lateral_tolerance_m,
                orientation_tolerance_rad,
                safe_contact_force_n,
                hard_contact_force_n,
                workspace_radius_m,
                reward_scale,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(thresholds)) or np.any(thresholds <= 0.0):
            raise ValueError("all task thresholds must be finite and positive")
        if hard_contact_force_n < safe_contact_force_n:
            raise ValueError("hard_contact_force_n cannot be below the safe limit")
        if actuation_mode not in {"manisoft_continuum", "rigid_flying_boom"}:
            raise ValueError(f"unsupported actuation_mode {actuation_mode!r}")
        if actuation_mode == "manisoft_continuum" and reward_scale != 1.0:
            raise ValueError("reward_scale is fixed at 1.0 for the continuum contract")
        if (
            actuation_mode == "manisoft_continuum"
            and backend_factory is None
            and runtime_root is None
        ):
            raise ValueError("runtime_root is required for the ManiSoft aerial backend")
        if asset_root is not None and runtime_root is None:
            raise ValueError("asset_root requires runtime_root")
        if aircraft_visual_root is not None and actuation_mode != "manisoft_continuum":
            raise ValueError("aircraft visuals currently support the continuum backend only")

        if cases is None and actuation_mode == "rigid_flying_boom":
            cases = make_rigid_aerial_cases()
        self._cases = _normalize_cases(cases)
        self.default_case = None if default_case is None else str(default_case)
        if self.default_case is not None and self.default_case not in self._cases:
            raise KeyError(f"unknown default case {self.default_case!r}")
        self.runtime_root = None if runtime_root is None else Path(runtime_root)
        self.asset_root = None if asset_root is None else Path(asset_root)
        self.aircraft_visual_root = (
            None if aircraft_visual_root is None else Path(aircraft_visual_root)
        )
        if self.aircraft_visual_root is not None:
            missing = [
                name
                for name in ("KC135.obj", "f16.obj")
                if not (self.aircraft_visual_root / name).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"aircraft visual root is missing: {', '.join(missing)}"
                )
        self.horizon = int(horizon)
        self.update_interval = int(update_interval)
        self.observation_mode = observation_mode
        self.render_mode = render_mode
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.camera_name = str(camera_name)
        self.depth_max_m = float(depth_max_m)
        self.required_insertion_depth_m = float(required_insertion_depth_m)
        self.lateral_tolerance_m = float(lateral_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.safe_contact_force_n = float(safe_contact_force_n)
        self.hard_contact_force_n = float(hard_contact_force_n)
        self.dwell_steps = int(dwell_steps)
        self.workspace_radius_m = float(workspace_radius_m)
        self.reward_scale = float(reward_scale)
        self.actuation_mode = actuation_mode
        self.action_dim = (
            AERIAL_ACTION_DIM
            if actuation_mode == "manisoft_continuum"
            else RIGID_AERIAL_ACTION_DIM
        )
        self.boom_extension_max_m = (
            AERIAL_BOOM_EXTENSION_MAX_M
            if actuation_mode == "manisoft_continuum"
            else RIGID_BOOM_EXTENSION_MAX_M
        )
        self._backend_factory = backend_factory
        self.backend_id = backend_id or (
            "ManiSoftSimBackend+PyElastica_CosseratRod"
            if backend_factory is None and actuation_mode == "manisoft_continuum"
            else "MuJoCo_rigid_gimbal_telescopic_boom"
            if backend_factory is None
            else "custom"
        )

        self._backend: AerialBackend | None = None
        self._case_id: str | None = None
        self._episode_seed: int | None = None
        self._step_count = 0
        self._done = True
        self._dwell = 0
        self._engaged_once = False
        self._previous_potential = 0.0
        self._last_rgb: npt.NDArray[np.uint8] | None = None
        self._last_depth: npt.NDArray[np.float32] | None = None

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        finite = np.finfo(np.float32).max
        state_space = spaces.Box(
            low=-finite,
            high=finite,
            shape=(AERIAL_STATE_DIM,),
            dtype=np.float32,
        )
        rgb_space = spaces.Box(
            low=0,
            high=255,
            shape=(*self.image_size, 3),
            dtype=np.uint8,
        )
        depth_space = spaces.Box(
            low=0.0,
            high=self.depth_max_m,
            shape=(*self.image_size, 1),
            dtype=np.float32,
        )
        if observation_mode == "state":
            self.observation_space = state_space
        elif observation_mode == "rgb":
            self.observation_space = rgb_space
        elif observation_mode == "depth":
            self.observation_space = depth_space
        elif observation_mode == "rgbd":
            self.observation_space = spaces.Dict({"rgb": rgb_space, "depth": depth_space})
        elif observation_mode == "hybrid":
            self.observation_space = spaces.Dict({"state": state_space, "rgb": rgb_space})
        else:
            self.observation_space = spaces.Dict(
                {"state": state_space, "rgb": rgb_space, "depth": depth_space}
            )

    @property
    def cases(self) -> tuple[str, ...]:
        return tuple(self._cases)

    @property
    def selected_case(self) -> str | None:
        return self._case_id

    @property
    def contract(self) -> dict[str, Any]:
        uses_state = self.observation_mode in {"state", "hybrid", "hybrid_rgbd"}
        uses_rgb = self.observation_mode in {"rgb", "rgbd", "hybrid", "hybrid_rgbd"}
        uses_depth = self.observation_mode in {"depth", "rgbd", "hybrid_rgbd"}
        if self.actuation_mode == "rigid_flying_boom":
            return {
                "schema_version": 1,
                "contract_id": RIGID_AERIAL_CONTRACT_ID,
                "case_set_id": RIGID_AERIAL_CASE_SET_ID,
                "case_set_sha256": _case_set_sha256(self._cases),
                "task": "rigid_flying_boom_terminal_docking",
                "aircraft_autopilots": "frozen_precontact_formation_hold",
                "backend": self.backend_id,
                "fidelity_claim": "uncalibrated_engineering_simulator",
                "observation_mode": self.observation_mode,
                "state_dim": AERIAL_STATE_DIM if uses_state else None,
                "state_layout": {
                    key: [value.start, value.stop]
                    for key, value in AERIAL_STATE_LAYOUT.items()
                },
                "rgb_shape": [*self.image_size, 3] if uses_rgb else None,
                "rgb_dtype": "uint8" if uses_rgb else None,
                "depth_shape": [*self.image_size, 1] if uses_depth else None,
                "depth_dtype": "float32" if uses_depth else None,
                "depth_units": "metres" if uses_depth else None,
                "depth_max_m": self.depth_max_m if uses_depth else None,
                "camera_name": self.camera_name if uses_rgb or uses_depth else None,
                "action": "2D_left_right_ruddevator_plus_1D_telescopic_rate",
                "action_dim": RIGID_AERIAL_ACTION_DIM,
                "action_range": [-1.0, 1.0],
                "reward": RIGID_AERIAL_REWARD_ID,
                "reward_scale": self.reward_scale,
                "boom_model": {
                    "mechanism": "two_axis_gimbal_fixed_outer_boom_telescopic_tube",
                    "outer_length_m": RIGID_BOOM_OUTER_LENGTH_M,
                    "telescopic_tube_length_m": RIGID_BOOM_TELESCOPIC_LENGTH_M,
                    "initial_extension_m": RIGID_BOOM_INITIAL_EXTENSION_M,
                    "ruddevator_terminal_command_limit_deg": 1.5,
                    "gimbal_mechanical_limit_deg": 25.0,
                    "air_density_kg_m3": 0.459,
                    "relative_airspeed_m_s": 200.0,
                    "gravity_compensated_tanker_frame": True,
                    "aerodynamic_parameters_calibrated": False,
                    "geometry_source": "FlightGear_Aircraft-2024_KC135.zip",
                    "geometry_source_sha256": (
                        "5584ba0a9fdda86fb289688339202a0dee0dae4c61e40b4463608bdb788bee12"
                    ),
                },
                "receiver_reference": {
                    "frame": "receiver_receptacle_mouth",
                    "position_m": list(RIGID_RECEPTACLE_POSITION_M),
                    "f16_refuel_anchor_aircraft_frame_m": [0.1054, 0.0, 0.6500],
                    "geometry_source": "FlightGear_Aircraft-2024_f16.zip",
                    "geometry_source_sha256": (
                        "6911d6ed92f6ed2dd5dd68068cbfc17e3581b36033925f617d5a5b19033fb052"
                    ),
                },
                "receptacle_geometry": {
                    "telescopic_extension_joint_range_m": [
                        0.0,
                        RIGID_BOOM_EXTENSION_MAX_M,
                    ],
                    "telescopic_extension_rate_limit_m_s": (
                        RIGID_BOOM_EXTENSION_RATE_LIMIT_M_S
                    ),
                    "port_half_opening_m": AERIAL_PORT_HALF_OPENING_M,
                    "tunnel_depth_m": AERIAL_PORT_TUNNEL_DEPTH_M,
                },
                "physics_dt_s": RIGID_AERIAL_PHYSICS_DT_S,
                "physics_substeps_per_control_step": self.update_interval,
                "control_period_s": (
                    RIGID_AERIAL_PHYSICS_DT_S * self.update_interval
                ),
                "horizon": self.horizon,
                "success": {
                    "insertion_depth_min_m": self.required_insertion_depth_m,
                    "lateral_error_max_m": self.lateral_tolerance_m,
                    "orientation_error_max_rad": self.orientation_tolerance_rad,
                    "contact_force_max_n": self.safe_contact_force_n,
                    "dwell_control_steps": self.dwell_steps,
                },
                "failure": {
                    "hard_contact_force_n": self.hard_contact_force_n,
                    "workspace_radius_m": self.workspace_radius_m,
                    "disconnect_after_engagement": True,
                    "mechanism_integrity": True,
                },
                "case_in_observation": False,
                "stage_in_observation": False,
                "clock_in_observation": False,
                "runtime_fsm": False,
                "runtime_waypoints": False,
                "runtime_expert_actions": False,
            }
        return {
            "schema_version": 1,
            "contract_id": AERIAL_CONTRACT_ID,
            "case_set_id": AERIAL_CASE_SET_ID,
            "case_set_sha256": _case_set_sha256(self._cases),
            "task": "flying_boom_terminal_docking",
            "aircraft_autopilots": "frozen_precontact_formation_hold",
            "backend": self.backend_id,
            "observation_mode": self.observation_mode,
            "state_dim": AERIAL_STATE_DIM if uses_state else None,
            "state_layout": {key: [value.start, value.stop] for key, value in AERIAL_STATE_LAYOUT.items()},
            "rgb_shape": [*self.image_size, 3] if uses_rgb else None,
            "rgb_dtype": "uint8" if uses_rgb else None,
            "depth_shape": [*self.image_size, 1] if uses_depth else None,
            "depth_dtype": "float32" if uses_depth else None,
            "depth_units": "metres" if uses_depth else None,
            "depth_max_m": self.depth_max_m if uses_depth else None,
            "camera_name": self.camera_name if uses_rgb or uses_depth else None,
            "aircraft_visuals": (
                {
                    "role": "presentation_only_no_collision_or_success_effect",
                    "tanker": "FlightGear KC-135E",
                    "receiver": "FlightGear F-16",
                    "source_catalog": "demo/aircraft_assets.json",
                    "uniform_scale": AERIAL_AIRCRAFT_VISUAL_SCALE,
                    "flightgear_to_model_coordinates": "(x,z,-y)",
                    "receiver_refuel_anchor_flightgear_m": [0.1054, 0.0, 0.6500],
                    "receiver_refuel_anchor_model_m": list(
                        AERIAL_F16_REFUEL_ANCHOR_MODEL_M
                    ),
                    "tanker_boom_pivot_flightgear_m": [15.10, 0.0, -0.13],
                    "tanker_boom_pivot_model_m": list(
                        AERIAL_KC135_BOOM_PIVOT_MODEL_M
                    ),
                    "tanker_boom_tip_model_m": list(
                        AERIAL_KC135_BOOM_TIP_MODEL_M
                    ),
                    "binding": (
                        "f16_anchor_to_receptacle_mouth_and_"
                        "kc135_boom_tip_to_continuum_root"
                    ),
                    "runtime_binding_tolerance_m": 1.0e-6,
                }
                if self.aircraft_visual_root is not None
                else None
            ),
            "action": "18D_normalized_SplineMuscle_plus_1D_telescopic_rate",
            "action_dim": AERIAL_ACTION_DIM,
            "action_range": [-1.0, 1.0],
            "reward": AERIAL_REWARD_ID,
            "boom_model": {
                "start_m": list(AERIAL_BOOM_START_M),
                "direction": list(AERIAL_BOOM_DIRECTION),
                "length_m": AERIAL_BOOM_LENGTH_M,
                "radius_m": AERIAL_BOOM_RADIUS_M,
                "num_elements": AERIAL_BOOM_NUM_ELEMENTS,
                "density_kg_m3": AERIAL_BOOM_DENSITY_KG_M3,
                "youngs_modulus_pa": AERIAL_BOOM_YOUNGS_MODULUS_PA,
                "poisson_ratio": AERIAL_BOOM_POISSON_RATIO,
                "damping_constant": AERIAL_BOOM_DAMPING_CONSTANT,
                "muscle_torque_scale": AERIAL_MUSCLE_TORQUE_SCALE,
                "normalized_muscle_activation_limit": AERIAL_MUSCLE_ACTIVATION_LIMIT,
                "gravity_compensated_relative_frame": True,
            },
            "receptacle_geometry": {
                "retracted_nozzle_length_m": AERIAL_NOZZLE_RETRACTED_LENGTH_M,
                "telescopic_extension_joint_range_m": [
                    0.0,
                    AERIAL_BOOM_EXTENSION_MAX_M,
                ],
                "telescopic_extension_command_range_m": [
                    0.0,
                    AERIAL_BOOM_EXTENSION_COMMAND_MAX_M,
                ],
                "telescopic_extension_rate_limit_m_s": (
                    AERIAL_BOOM_EXTENSION_RATE_LIMIT_M_S
                ),
                "port_half_opening_m": AERIAL_PORT_HALF_OPENING_M,
                "tunnel_depth_m": AERIAL_PORT_TUNNEL_DEPTH_M,
                "boom_tip_asset_sha256": _boom_tip_asset_sha256(),
            },
            "physics_dt_s": AERIAL_PHYSICS_DT_S,
            "physics_substeps_per_control_step": self.update_interval,
            "control_period_s": AERIAL_PHYSICS_DT_S * self.update_interval,
            "horizon": self.horizon,
            "success": {
                "insertion_depth_min_m": self.required_insertion_depth_m,
                "lateral_error_max_m": self.lateral_tolerance_m,
                "orientation_error_max_rad": self.orientation_tolerance_rad,
                "contact_force_max_n": self.safe_contact_force_n,
                "dwell_control_steps": self.dwell_steps,
            },
            "failure": {
                "hard_contact_force_n": self.hard_contact_force_n,
                "workspace_radius_m": self.workspace_radius_m,
                "disconnect_after_engagement": True,
                "continuum_integrity": True,
            },
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
            return self.cases[int(self.np_random.integers(0, len(self._cases)))]
        case = str(requested)
        if case not in self._cases:
            raise KeyError(f"unknown case {case!r}; available cases: {self.cases}")
        return case

    def _make_backend(
        self, case: AerialRefuelCase, seed: int, enable_renderer: bool
    ) -> AerialBackend:
        if self._backend_factory is not None:
            return self._backend_factory(
                case,
                seed,
                enable_renderer,
                self.image_size,
                self.camera_name,
                self.update_interval,
            )
        if self.actuation_mode == "manisoft_continuum":
            from .aerial_manisoft import ManiSoftAerialBackend

            assert self.runtime_root is not None
            return ManiSoftAerialBackend(
                case=case,
                seed=seed,
                enable_renderer=enable_renderer,
                image_size=self.image_size,
                camera_name=self.camera_name,
                update_interval=self.update_interval,
                runtime_root=self.runtime_root,
                asset_root=self.asset_root,
                aircraft_visual_root=self.aircraft_visual_root,
            )
        from .aerial_rigid import RigidFlyingBoomBackend

        return RigidFlyingBoomBackend(
            case=case,
            seed=seed,
            enable_renderer=enable_renderer,
            image_size=self.image_size,
            camera_name=self.camera_name,
            update_interval=self.update_interval,
        )

    def _validate_snapshot(self, snapshot: AerialSnapshot) -> None:
        arrays = (
            snapshot.boom_positions_m,
            snapshot.boom_velocities_m_s,
            snapshot.tip_position_m,
            snapshot.tip_velocity_m_s,
            snapshot.tip_axis,
            snapshot.receptacle_position_m,
            snapshot.receptacle_axis,
            snapshot.receptacle_linear_velocity_m_s,
            snapshot.receptacle_angular_velocity_rad_s,
        )
        if snapshot.boom_positions_m.ndim != 2 or snapshot.boom_positions_m.shape[1] != 3:
            raise RuntimeError("backend returned malformed boom positions")
        if snapshot.boom_velocities_m_s.shape != snapshot.boom_positions_m.shape:
            raise RuntimeError("backend returned malformed boom velocities")
        if snapshot.boom_positions_m.shape[0] < 10:
            raise RuntimeError("backend must expose at least 10 boom nodes")
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise RuntimeError("backend returned nonfinite aerial state")
        scalars = np.asarray(
            (
                snapshot.insertion_depth_m,
                snapshot.lateral_error_m,
                snapshot.orientation_error_rad,
                snapshot.contact_force_n,
                snapshot.step_peak_contact_force_n,
                snapshot.episode_peak_contact_force_n,
                snapshot.boom_extension_m,
                snapshot.boom_extension_velocity_m_s,
                snapshot.physics_time_s,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)):
            raise RuntimeError("backend returned invalid aerial diagnostics")
        nonnegative = np.delete(scalars, 7)
        if np.any(nonnegative < 0.0):
            raise RuntimeError("backend returned negative aerial diagnostics")
        if snapshot.boom_extension_m > self.boom_extension_max_m + 1.0e-3:
            raise RuntimeError(
                "backend returned out-of-range boom extension "
                f"{snapshot.boom_extension_m:.9f} m"
            )

    def _state(self, snapshot: AerialSnapshot) -> npt.NDArray[np.float32]:
        self._validate_snapshot(snapshot)
        indices = np.linspace(0, snapshot.boom_positions_m.shape[0] - 1, 10).astype(int)
        position_relative = (
            snapshot.boom_positions_m[indices] - snapshot.receptacle_position_m
        )
        velocity_relative = (
            snapshot.boom_velocities_m_s[indices]
            - snapshot.receptacle_linear_velocity_m_s
        )
        tip_relative = snapshot.tip_position_m - snapshot.receptacle_position_m
        tip_velocity_relative = (
            snapshot.tip_velocity_m_s - snapshot.receptacle_linear_velocity_m_s
        )
        result = np.concatenate(
            (
                position_relative.reshape(-1),
                velocity_relative.reshape(-1),
                tip_relative,
                tip_velocity_relative,
                snapshot.tip_axis,
                snapshot.receptacle_axis,
                snapshot.receptacle_linear_velocity_m_s,
                snapshot.receptacle_angular_velocity_rad_s,
                np.asarray((snapshot.insertion_depth_m,), dtype=np.float64),
                np.asarray(
                    (snapshot.contact_force_n / self.safe_contact_force_n,),
                    dtype=np.float64,
                ),
                np.asarray(
                    (
                        snapshot.boom_extension_m,
                        snapshot.boom_extension_velocity_m_s,
                    ),
                    dtype=np.float64,
                ),
            )
        ).astype(np.float32)
        if result.shape != (AERIAL_STATE_DIM,) or not np.all(np.isfinite(result)):
            raise RuntimeError(f"invalid aerial state {result.shape}")
        return result

    def _prepare_rgb(self, value: npt.ArrayLike) -> npt.NDArray[np.uint8]:
        frame = np.asarray(value)
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise RuntimeError(f"backend returned invalid RGB shape {frame.shape}")
        frame = np.asarray(frame[..., :3], dtype=np.uint8)
        if frame.shape[:2] != self.image_size:
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
            raise RuntimeError(f"backend returned invalid depth shape {depth.shape}")
        if np.any(np.isnan(depth)) or np.any(depth < 0.0):
            raise RuntimeError("backend returned negative or NaN depth")
        depth = np.nan_to_num(depth, posinf=self.depth_max_m)
        depth = np.clip(depth, 0.0, self.depth_max_m)
        if depth.shape[:2] != self.image_size:
            image = Image.fromarray(depth[..., 0])
            image = image.resize(
                (self.image_size[1], self.image_size[0]),
                resample=Image.Resampling.NEAREST,
            )
            depth = np.asarray(image, dtype=np.float32)[..., None]
        self._last_depth = depth.copy()
        return depth.copy()

    def _rgbd(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
        if self._backend is None:
            raise RuntimeError("reset must be called before rendering")
        render_rgbd = getattr(self._backend, "render_rgbd", None)
        if not callable(render_rgbd):
            raise RuntimeError("the aerial backend does not expose synchronized RGB-D")
        rgb, depth = render_rgbd()
        return self._prepare_rgb(rgb), self._prepare_depth(depth)

    def _observation(self, snapshot: AerialSnapshot) -> Any:
        state = self._state(snapshot)
        if self.observation_mode == "state":
            return state
        rgb, depth = self._rgbd()
        if self.observation_mode == "rgb":
            return rgb
        if self.observation_mode == "depth":
            return depth
        if self.observation_mode == "rgbd":
            return {"rgb": rgb, "depth": depth}
        if self.observation_mode == "hybrid":
            return {"state": state, "rgb": rgb}
        return {"state": state, "rgb": rgb, "depth": depth}

    def _success_now(self, snapshot: AerialSnapshot) -> bool:
        return bool(
            snapshot.insertion_depth_m >= self.required_insertion_depth_m
            and snapshot.lateral_error_m <= self.lateral_tolerance_m
            and snapshot.orientation_error_rad <= self.orientation_tolerance_rad
            and snapshot.step_peak_contact_force_n <= self.safe_contact_force_n
            and snapshot.continuum_integrity_valid
        )

    def _failure_reason(self, snapshot: AerialSnapshot) -> str | None:
        if not snapshot.continuum_integrity_valid:
            return (
                "continuum_integrity_failure"
                if self.actuation_mode == "manisoft_continuum"
                else "mechanism_integrity_failure"
            )
        if snapshot.step_peak_contact_force_n > self.hard_contact_force_n:
            return "hard_contact_force"
        distance = float(
            np.linalg.norm(snapshot.tip_position_m - snapshot.receptacle_position_m)
        )
        if distance > self.workspace_radius_m:
            return "out_of_workspace"
        signed_axial_distance = float(
            np.dot(
                snapshot.tip_position_m - snapshot.receptacle_position_m,
                snapshot.receptacle_axis,
            )
        )
        if self._engaged_once and signed_axial_distance < -0.020:
            return "disconnect_after_engagement"
        return None

    def _potential(self, snapshot: AerialSnapshot) -> float:
        distance = float(
            np.linalg.norm(snapshot.tip_position_m - snapshot.receptacle_position_m)
        )
        depth_score = float(
            np.clip(
                snapshot.insertion_depth_m / self.required_insertion_depth_m,
                0.0,
                1.0,
            )
        )
        return (
            -4.0 * distance
            - 4.0 * snapshot.lateral_error_m
            - 0.5 * snapshot.orientation_error_rad
            + 4.0 * depth_score
        )

    def _info(
        self,
        snapshot: AerialSnapshot,
        reward_terms: Mapping[str, float],
        termination_reason: str | None,
    ) -> dict[str, Any]:
        if self._case_id is None or self._episode_seed is None:
            raise RuntimeError("episode metadata is unavailable")
        return {
            "case_id": self._case_id,
            "case_sha256": hashlib.sha256(
                json.dumps(
                    self._cases[self._case_id].to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "episode_seed": self._episode_seed,
            "step_count": self._step_count,
            "physics_time_s": snapshot.physics_time_s,
            "termination_reason": termination_reason,
            "official_success": bool(self._dwell >= self.dwell_steps),
            "success_predicate": "aerial_terminal_docking_v1",
            "dwell_steps": self._dwell,
            "insertion_depth_m": snapshot.insertion_depth_m,
            "lateral_error_m": snapshot.lateral_error_m,
            "orientation_error_rad": snapshot.orientation_error_rad,
            "boom_extension_m": snapshot.boom_extension_m,
            "boom_extension_velocity_m_s": snapshot.boom_extension_velocity_m_s,
            "contact": {
                "contact_force_n": snapshot.contact_force_n,
                "step_peak_contact_force_n": snapshot.step_peak_contact_force_n,
                "episode_peak_contact_force_n": snapshot.episode_peak_contact_force_n,
                "safe_limit_n": self.safe_contact_force_n,
                "hard_limit_n": self.hard_contact_force_n,
            },
            "continuum_integrity_valid": snapshot.continuum_integrity_valid,
            "mechanism_integrity_valid": snapshot.continuum_integrity_valid,
            "reward_terms": {key: float(value) for key, value in reward_terms.items()},
            "contract": self.contract,
            "contract_sha256": self.contract_sha256,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        self.close()
        case_id = self._select_case(options or {})
        episode_seed = (
            int(seed)
            if seed is not None
            else int(self.np_random.integers(0, np.iinfo(np.uint32).max))
        )
        enable_renderer = self.observation_mode != "state" or self.render_mode == "rgb_array"
        backend = self._make_backend(
            self._cases[case_id], episode_seed, enable_renderer
        )
        self._backend = backend
        self._case_id = case_id
        self._episode_seed = episode_seed
        self._step_count = 0
        self._done = False
        self._dwell = 0
        self._engaged_once = False
        self._last_rgb = None
        self._last_depth = None
        snapshot = backend.snapshot()
        self._validate_snapshot(snapshot)
        self._previous_potential = self._potential(snapshot)
        reward_terms = {
            "potential_delta": 0.0,
            "action_penalty": 0.0,
            "contact_penalty": 0.0,
            "success_bonus": 0.0,
            "failure_penalty": 0.0,
            "total": 0.0,
        }
        if self.actuation_mode == "rigid_flying_boom":
            reward_terms["tracking_penalty"] = 0.0
            reward_terms["ruddevator_penalty"] = 0.0
        observation = self._observation(snapshot)
        return observation, self._info(snapshot, reward_terms, None)

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._backend is None:
            raise RuntimeError("reset must be called before step")
        if self._done:
            raise RuntimeError("episode is finished; call reset before step")
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape != (self.action_dim,) or not np.all(
            np.isfinite(action_array)
        ):
            raise ValueError(f"action must be finite {self.action_dim}D")
        if float(np.max(np.abs(action_array))) > 1.0 + 1.0e-6:
            raise ValueError("action exceeds [-1, 1]")

        self._backend.advance(action_array)
        snapshot = self._backend.snapshot()
        self._validate_snapshot(snapshot)
        self._step_count += 1
        if snapshot.insertion_depth_m >= 0.010:
            self._engaged_once = True
        failure_reason = self._failure_reason(snapshot)
        safe_insert = self._success_now(snapshot) and failure_reason is None
        self._dwell = self._dwell + 1 if safe_insert else 0
        success = self._dwell >= self.dwell_steps
        terminated = bool(success or failure_reason is not None)
        truncated = bool(not terminated and self._step_count >= self.horizon)
        termination_reason = (
            "official_success"
            if success
            else failure_reason
            if failure_reason is not None
            else "time_limit"
            if truncated
            else None
        )

        potential = self._potential(snapshot)
        potential_delta = 20.0 * (potential - self._previous_potential)
        self._previous_potential = potential
        action_penalty = -0.0005 * float(np.mean(np.square(action_array)))
        tracking_penalty = 0.0
        ruddevator_penalty = 0.0
        if self.actuation_mode == "rigid_flying_boom":
            relative = snapshot.tip_position_m - snapshot.receptacle_position_m
            axial_gap_m = max(
                -float(np.dot(relative, snapshot.receptacle_axis)), 0.0
            )
            tracking_penalty = (
                -0.25 * (snapshot.lateral_error_m / 0.050) ** 2
                -0.05
                * (snapshot.orientation_error_rad / self.orientation_tolerance_rad)
                ** 2
                -0.05 * (axial_gap_m / 0.20)
            )
            ruddevator_penalty = -0.01 * float(
                np.mean(np.square(action_array[:2]))
            )
        contact_penalty = -0.02 * float(
            (snapshot.step_peak_contact_force_n / self.safe_contact_force_n) ** 2
        )
        success_bonus = 100.0 if success else 0.0
        failure_penalty = -100.0 if failure_reason is not None else 0.0
        reward = (
            potential_delta
            + action_penalty
            + tracking_penalty
            + ruddevator_penalty
            + contact_penalty
            + success_bonus
            + failure_penalty
        )
        if self.reward_scale != 1.0:
            potential_delta *= self.reward_scale
            action_penalty *= self.reward_scale
            tracking_penalty *= self.reward_scale
            ruddevator_penalty *= self.reward_scale
            contact_penalty *= self.reward_scale
            success_bonus *= self.reward_scale
            failure_penalty *= self.reward_scale
            reward *= self.reward_scale
        reward_terms = {
            "potential_delta": potential_delta,
            "action_penalty": action_penalty,
            "tracking_penalty": tracking_penalty,
            "ruddevator_penalty": ruddevator_penalty,
            "contact_penalty": contact_penalty,
            "success_bonus": success_bonus,
            "failure_penalty": failure_penalty,
            "total": reward,
        }
        self._done = terminated or truncated
        self._last_rgb = None
        self._last_depth = None
        observation = self._observation(snapshot)
        info = self._info(snapshot, reward_terms, termination_reason)
        return observation, float(reward), terminated, truncated, info

    def render(self) -> npt.NDArray[np.uint8]:
        if self.render_mode != "rgb_array":
            raise RuntimeError("construct the environment with render_mode='rgb_array'")
        if self._last_rgb is not None:
            return self._last_rgb.copy()
        rgb, _ = self._rgbd()
        return rgb

    def close(self) -> None:
        backend = self._backend
        self._backend = None
        if backend is not None:
            backend.close()
        self._done = True
        self._last_rgb = None
        self._last_depth = None


class RigidFlyingBoomEnv(AerialRefuelEnv):
    """Full-scale rigid/telescopic flying-boom MVP with 3D direct control."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("actuation_mode", "rigid_flying_boom")
        kwargs.setdefault("update_interval", RIGID_AERIAL_DEFAULT_UPDATE_INTERVAL)
        kwargs.setdefault("reward_scale", 0.01)
        super().__init__(**kwargs)
