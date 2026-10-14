from __future__ import annotations

import hashlib
import json
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from .aerial import AERIAL_ACTION_DIM, AERIAL_STATE_DIM, AerialRefuelEnv


RIGID_STATE_SCALE = np.ones(AERIAL_STATE_DIM, dtype=np.float32)
RIGID_STATE_SCALE[0:30] = 10.0
RIGID_STATE_SCALE[30:60] = 2.0
RIGID_STATE_SCALE[63:66] = 2.0
RIGID_STATE_SCALE[72:75] = 2.0
RIGID_STATE_SCALE[78] = 0.20
RIGID_STATE_SCALE[80] = 3.0


class AerialSynergyActionWrapper(gym.ActionWrapper):
    """Map three smooth boom modes plus extension rate into the 19D actuator."""

    def __init__(self, env: AerialRefuelEnv) -> None:
        if env.action_space.shape != (AERIAL_ACTION_DIM,):
            raise ValueError(
                "AerialSynergyActionWrapper only supports the 19D continuum boom"
            )
        super().__init__(env)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

    def action(self, action: npt.ArrayLike) -> npt.NDArray[np.float32]:
        value = np.asarray(action, dtype=np.float32).reshape(-1)
        if value.shape != (4,) or not np.all(np.isfinite(value)):
            raise ValueError("synergy action must be finite 4D")
        if float(np.max(np.abs(value))) > 1.0 + 1.0e-6:
            raise ValueError("synergy action exceeds [-1, 1]")
        expanded = np.empty(AERIAL_ACTION_DIM, dtype=np.float32)
        expanded[:18] = np.tile(value[:3], 6)
        expanded[18] = value[3]
        return expanded

    @property
    def cases(self) -> tuple[str, ...]:
        return self.env.cases

    @property
    def safe_contact_force_n(self) -> float:
        return self.env.safe_contact_force_n

    @property
    def contract(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "base_environment_contract_sha256": self.env.contract_sha256,
            "policy_action": "4D_three_tiled_smooth_boom_modes_plus_extension_rate",
            "mapping": {
                "boom": "tile(action[0:3], 6)",
                "extension_rate": "action[3]",
                "range": [-1.0, 1.0],
            },
        }

    @property
    def contract_sha256(self) -> str:
        payload = json.dumps(
            self.contract,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class RigidStateScaleWrapper(gym.ObservationWrapper):
    """Fixed Markov scaling for full-scale rigid-boom state policies."""

    def __init__(self, env: AerialRefuelEnv) -> None:
        if env.actuation_mode != "rigid_flying_boom":
            raise ValueError("RigidStateScaleWrapper requires the rigid mechanism")
        if env.observation_mode != "state":
            raise ValueError("RigidStateScaleWrapper currently supports state mode only")
        super().__init__(env)
        finite = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-finite,
            high=finite,
            shape=(AERIAL_STATE_DIM,),
            dtype=np.float32,
        )

    def observation(self, observation: npt.ArrayLike) -> npt.NDArray[np.float32]:
        state = np.asarray(observation, dtype=np.float32)
        if state.shape != (AERIAL_STATE_DIM,) or not np.all(np.isfinite(state)):
            raise RuntimeError("rigid state scaler received an invalid observation")
        return np.asarray(state / RIGID_STATE_SCALE, dtype=np.float32)

    @property
    def cases(self) -> tuple[str, ...]:
        return self.env.cases

    @property
    def safe_contact_force_n(self) -> float:
        return self.env.safe_contact_force_n

    @property
    def contract(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "base_environment_contract_sha256": self.env.contract_sha256,
            "policy_observation": "82D_fixed_scaled_rigid_physical_state",
            "state_divisor": RIGID_STATE_SCALE.tolist(),
            "stateful": False,
            "history": False,
            "normalization_statistics_fitted": False,
        }

    @property
    def contract_sha256(self) -> str:
        payload = json.dumps(
            self.contract,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
