from gymnasium.envs.registration import register, registry

from .aerial import (
    AERIAL_ACTION_DIM,
    AERIAL_CASE_SET_ID,
    AERIAL_CONTRACT_ID,
    AERIAL_STATE_DIM,
    AERIAL_STATE_LAYOUT,
    RIGID_AERIAL_ACTION_DIM,
    RIGID_AERIAL_CASE_SET_ID,
    RIGID_AERIAL_CONTRACT_ID,
    AerialRefuelCase,
    AerialRefuelEnv,
    RigidFlyingBoomEnv,
    make_aerial_cases,
    make_rigid_aerial_cases,
)
from .aerial_rl import AerialSynergyActionWrapper, RigidStateScaleWrapper
from .env import (
    ACTION_DIM,
    ARM_ACTION_DIM,
    CONTRACT_ID,
    OBSERVATION_DIM,
    STATE_LAYOUT,
    ManiSoftRLEnv,
    discover_case_dirs,
)
from .promotion import (
    PromotionDecision,
    compare_paired_outcomes,
    gate_paired_report,
)
from .vector import make_aerial_vector_env, make_vector_env

ENV_ID = "ManiSoft-COLL-v0"
if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="manisoftrl.env:ManiSoftRLEnv")
AERIAL_ENV_ID = "ManiSoft-AerialRefuel-v0"
if AERIAL_ENV_ID not in registry:
    register(id=AERIAL_ENV_ID, entry_point="manisoftrl.aerial:AerialRefuelEnv")
RIGID_AERIAL_ENV_ID = "ManiSoft-AerialRigidBoom-v0"
if RIGID_AERIAL_ENV_ID not in registry:
    register(
        id=RIGID_AERIAL_ENV_ID,
        entry_point="manisoftrl.aerial:RigidFlyingBoomEnv",
    )

__all__ = [
    "ACTION_DIM",
    "AERIAL_ACTION_DIM",
    "AerialSynergyActionWrapper",
    "AERIAL_CASE_SET_ID",
    "AERIAL_CONTRACT_ID",
    "AERIAL_ENV_ID",
    "AERIAL_STATE_DIM",
    "AERIAL_STATE_LAYOUT",
    "RIGID_AERIAL_ACTION_DIM",
    "RIGID_AERIAL_CASE_SET_ID",
    "RIGID_AERIAL_CONTRACT_ID",
    "RIGID_AERIAL_ENV_ID",
    "ARM_ACTION_DIM",
    "CONTRACT_ID",
    "ENV_ID",
    "OBSERVATION_DIM",
    "PromotionDecision",
    "STATE_LAYOUT",
    "ManiSoftRLEnv",
    "AerialRefuelCase",
    "AerialRefuelEnv",
    "RigidFlyingBoomEnv",
    "RigidStateScaleWrapper",
    "compare_paired_outcomes",
    "discover_case_dirs",
    "gate_paired_report",
    "make_vector_env",
    "make_aerial_vector_env",
    "make_aerial_cases",
    "make_rigid_aerial_cases",
]

__version__ = "0.7.0"
