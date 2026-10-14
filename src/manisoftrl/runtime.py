from __future__ import annotations

import importlib.util
import hashlib
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any


_CONFIGURED_RUNTIME: tuple[Path, Path] | None = None


def _configure_numba_cache(runtime: Path) -> Path:
    """Avoid loading movable/stale Numba binaries from the source checkout."""
    configured = os.environ.get("NUMBA_CACHE_DIR")
    if configured:
        path = Path(configured).expanduser().resolve()
    else:
        identity = hashlib.sha256(str(runtime).encode("utf-8")).hexdigest()[:12]
        base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
        path = (
            base
            / "manisoftrl"
            / "numba"
            / f"py{sys.version_info.major}{sys.version_info.minor}-{identity}"
        ).resolve()
        os.environ["NUMBA_CACHE_DIR"] = str(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _install_optional_import_shims() -> None:
    """Provide only the optional imports needed to load the COLL environment."""
    if "pyvista" not in sys.modules and importlib.util.find_spec("pyvista") is None:
        sys.modules["pyvista"] = types.ModuleType("pyvista")

    if (
        "stable_baselines3" not in sys.modules
        and importlib.util.find_spec("stable_baselines3") is None
    ):
        module = types.ModuleType("stable_baselines3")

        class UnavailableSAC:
            @staticmethod
            def load(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    "stable-baselines3 is not installed; install manisoftrl[baseline] "
                    "before loading an SB3 checkpoint"
                )

        module.SAC = UnavailableSAC
        sys.modules["stable_baselines3"] = module


def configure_manisoft_runtime(
    runtime_root: str | Path,
    asset_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Make an existing ManiSoft checkout importable without copying it.

    ManiSoft resolves several assets relative to the process working directory,
    matching the proven Route-B launcher. One process must therefore use one
    runtime/asset pair. Parallel environments should use separate processes.
    """
    global _CONFIGURED_RUNTIME

    runtime = Path(runtime_root).expanduser().resolve(strict=True)
    assets = (
        runtime
        if asset_root is None
        else Path(asset_root).expanduser().resolve(strict=True)
    )
    requested = (runtime, assets)
    if _CONFIGURED_RUNTIME is not None and _CONFIGURED_RUNTIME != requested:
        raise RuntimeError(
            "this process is already configured for a different ManiSoft runtime: "
            f"{_CONFIGURED_RUNTIME}"
        )

    roots = (
        runtime,
        runtime / "third_party" / "pyelastica",
        runtime / "third_party" / "liegroups",
    )
    _configure_numba_cache(runtime)
    for root in reversed(roots):
        value = str(root)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)

    os.chdir(assets)
    _install_optional_import_shims()
    _CONFIGURED_RUNTIME = requested
    return requested
