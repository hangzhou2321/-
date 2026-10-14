# Publication validation — 2026-10-14

This publication packages the existing reusable ManiSoft-RL source snapshot.
All files under `src/manisoftrl` selected for publication were compared with the
local development package. Core source files were identical. Public examples
already use portable runtime arguments in place of private workstation paths.

## Checks executed

- Installed the package into a fresh Python 3.12 virtual environment with
  `pip install -e ".[test,rigid-aerial]"`.
- Ran `python -m pytest`: **33 passed**.
- Ran `python examples/rigid_aerial_smoke.py --output artifacts/publication_smoke_20261014 --steps 20`.
  The real MuJoCo simulation completed 20 steps and exported RGB, metric depth,
  and a JSON report. The result was `official_success=false` and
  `termination_reason=time_limit`. This is an execution smoke check, not a
  successful learned docking result.

## Scope and exclusions

The selected source tree includes environments, training/replay entry points,
tests, task contracts, and third-party notices. Local and external-drive
experiment archives were inspected for provenance but are not republished.
Datasets, checkpoints, replay buffers, external simulator checkouts, generated
builds, caches, credentials, and aircraft asset packages are excluded.

Full soft-continuum physics, pretrained-policy evaluation, Blender rendering,
and formal aerial success rates were not revalidated for this publication.
The unit tests exercise package contracts without the external ManiSoft runtime.
Historical research proposals and experimental VLA branches are not represented
as implemented or validated capabilities of this release.
