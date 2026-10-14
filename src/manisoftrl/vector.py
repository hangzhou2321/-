from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv

from .aerial import AerialRefuelEnv
from .env import ManiSoftRLEnv


def _build_env(kwargs: dict[str, Any]) -> ManiSoftRLEnv:
    return ManiSoftRLEnv(**kwargs)


def _build_aerial_env(kwargs: dict[str, Any]) -> AerialRefuelEnv:
    return AerialRefuelEnv(**kwargs)


def _vector_env(
    num_envs: int,
    env_kwargs: Mapping[str, Any],
    asynchronous: bool,
    builder: Any,
) -> VectorEnv:
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    factories = [partial(builder, dict(env_kwargs)) for _ in range(int(num_envs))]
    vector_type = AsyncVectorEnv if asynchronous else SyncVectorEnv
    return vector_type(factories)


def make_vector_env(
    num_envs: int,
    env_kwargs: Mapping[str, Any],
    *,
    asynchronous: bool = True,
) -> VectorEnv:
    """Build standard Gymnasium vector environments from one environment config."""
    return _vector_env(num_envs, env_kwargs, asynchronous, _build_env)


def make_aerial_vector_env(
    num_envs: int,
    env_kwargs: Mapping[str, Any],
    *,
    asynchronous: bool = True,
) -> VectorEnv:
    """Build process-safe vector environments for flying-boom docking."""
    return _vector_env(num_envs, env_kwargs, asynchronous, _build_aerial_env)
