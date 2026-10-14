# ManiSoft-RL

[中文说明](README.zh-CN.md) · [Publication validation](PUBLICATION_VALIDATION.md)

ManiSoft-RL is a small Gymnasium package for reproducible reinforcement-learning
experiments in soft-continuum manipulation and flying-boom docking. It provides
explicit environment contracts, vectorized execution, RGB-D observation paths,
and a fail-closed parent/candidate promotion gate. It does not replace upstream
trainers or frozen evaluation scripts.

This is a community-maintained adapter. The upstream
[ManiSoft](https://github.com/buaa-colalab/ManiSoft) simulator is an external
MIT-licensed runtime and is not copied into this repository.

The implemented state contract is deliberately narrow:

- observation: 85D current physical state;
- action: 18D soft-arm activation plus one learned gripper scalar;
- success: raw `COLLEnvironment.is_finished()` with a margin audit;
- no case id, stage, clock, waypoint, FSM, expert action, or action overlay in
  the policy input/runtime loop;
- official success yields `terminated=True`;
- the platform horizon yields `truncated=True` only when success is false.

This is the Route-B 85D contract. It is not B's separate 83D Can0 contract and
not asython's historical 86D observation with a time-progress feature.

## Install

```powershell
python -m pip install -e .
python -m pip install -e ".[rigid-aerial,baseline]"
```

ManiSoft itself remains an external runtime. Pass the checkout and asset roots
when constructing the environment; no source is copied into this package.
The launcher also isolates Numba's cache under the user's local cache directory.
This prevents moved or stale PyElastica machine-code caches in an old checkout
from crashing the Python process; an explicitly set `NUMBA_CACHE_DIR` is kept.

The registered Gymnasium ids are `ManiSoft-COLL-v0`,
`ManiSoft-AerialRefuel-v0`, and `ManiSoft-AerialRigidBoom-v0`, so class
constructors and standard entry points are both supported:

```python
import gymnasium as gym
import manisoftrl

env = gym.make(manisoftrl.ENV_ID, case_dirs=case_root)
```

## State environment

```python
import numpy as np
from manisoftrl import ManiSoftRLEnv

env = ManiSoftRLEnv(
    case_dirs="/path/to/ManiSoft/data/COLL/eval/can",
    runtime_root="/path/to/ManiSoft",
    asset_root="/path/to/ManiSoft",
    horizon=1500,
)

obs, info = env.reset(seed=70700007, options={"case": 7})
assert obs.shape == (85,)
print(info["contract_sha256"])

obs, reward, terminated, truncated, info = env.step(
    np.zeros(19, dtype=np.float32)
)
print(info["reward_terms"], info["contact"])
env.close()
```

`case_dirs` may be one case directory, a directory whose immediate children are
cases, or an explicit `{case_id: directory}` mapping. `env.cases` exposes the
resolved public case ids. Reusing the same `seed` and `case` supplies the common
random-number boundary; it does not by itself make two different runtimes
physically equivalent.

`env.contract` is a JSON-safe description of the policy-facing observation,
action, reward, horizon, success predicate, and forbidden runtime scaffolds.
`env.contract_sha256` is its stable fingerprint. Vision shape, camera, units,
and depth range are included, so changing a policy-facing setting changes the
fingerprint.

## Flying-boom terminal docking

```python
import numpy as np
from manisoftrl import AerialRefuelEnv, make_aerial_cases

env = AerialRefuelEnv(
    cases=make_aerial_cases(8, seed=20260807),
    runtime_root=runtime_root,
    asset_root=asset_root,
    observation_mode="hybrid_rgbd",
)
obs, info = env.reset(seed=80700000, options={"case": 1})
assert obs["state"].shape == (82,)
assert obs["rgb"].shape == (128, 128, 3)
assert obs["depth"].shape == (128, 128, 1)
obs, reward, terminated, truncated, info = env.step(
    np.zeros(19, dtype=np.float32)
)
env.close()
```

This adapter uses the original `ManiSoftSimBackend`, PyElastica
`CosseratRod`, and 18D `SplineMuscle`, plus one telescopic-extension-rate
channel. The normalized 19D environment action is therefore 18D bending plus
1D extension; there is no gripper. State is 82D after adding extension position
and velocity. The receiver receptacle is an independent prescribed
six-degree-of-freedom MuJoCo body. Case, stage, and clock are excluded. See
`AERIAL_REFUELING_VISION_CONTRACT.md` for the exact contract.

For online RL, `AerialSynergyActionWrapper` exposes a smaller 4D policy action:
three smooth bending modes are tiled over the six spline control points and the
fourth value is extension rate. The underlying environment remains directly
controllable in 19D. The wrapper is an actuator basis, not a waypoint, expert,
FSM, or action overlay.

### Rigid KC-135-style boom branch

`ManiSoft-AerialRigidBoom-v0` is the independent realistic-mechanism MVP. It
does not replace the continuum research branch. Its 3D normalized action is
left ruddevator, right ruddevator, and telescopic extension rate; the MuJoCo
mechanism has pitch/yaw gimbals, a fixed outer boom, a sliding rigid tube,
force contact, and synchronized eye-in-hand RGB-D.

```python
import gymnasium as gym
import numpy as np
import manisoftrl

env = gym.make(
    manisoftrl.RIGID_AERIAL_ENV_ID,
    observation_mode="hybrid_rgbd",
)
obs, info = env.reset(seed=80730000, options={"case": 0})
obs, reward, terminated, truncated, info = env.step(
    np.zeros(3, dtype=np.float32)
)
env.close()
```

The full-scale outer/telescopic lengths come from the FlightGear 2024 KC-135
mesh, and the receiver reference is tied to the FlightGear F-16 refuel anchor.
The aerodynamic coefficients are deliberately labelled uncalibrated in the
hashed contract. This is an engineering simulator, not yet a validated digital
twin: aircraft 6-DoF, wake coupling, and flight-data calibration remain future
contract changes.

State-policy training uses `RigidStateScaleWrapper`, a fixed, stateless,
82D divisor recorded in a separate hash. The public environment still exposes
the raw metre/radian state. This avoids fitting hidden normalization statistics
and keeps checkpoint preprocessing reproducible.

## No-forgetting promotion gate

Higher total success is not enough. Promotion requires

```text
parent_successes ⊂ candidate_successes
```

That means every parent success is retained and at least one new success is
added. Any lost parent case rejects the candidate, even if its total success
count is higher. Parent and candidate rows must have identical case/seed keys;
the Route-B report adapter also requires common random numbers and, by default,
a formal evaluation with one frozen checkpoint per role.

```powershell
manisoftrl-promotion <paired-report.json>
```

Exit code `0` means the candidate passed. Exit code `2` means rejection or an
invalid evidence contract. `--allow-nonformal` is only for a local development
decision and never implies official-task success.

The gate recomputes success sets from raw rows instead of trusting summary
counts. It does not claim to make every policy update safe or better; it keeps
an accepted parent protected by refusing to promote a regressing candidate.
Formal reports must also contain `environment_contract_sha256`, copied from the
evaluating environment. This prevents results from silently crossing state,
camera, image-size, horizon, reward, or success-predicate contracts.

## RGB, metric depth, and hybrid observations

```python
vision_env = ManiSoftRLEnv(
    case_dirs=case_root,
    runtime_root=runtime_root,
    asset_root=asset_root,
    observation_mode="hybrid_rgbd",
    image_size=(128, 128),
    camera_name="third",
    depth_max_m=10.0,
)
obs, info = vision_env.reset(seed=42, options={"case": 0})
assert obs["state"].shape == (85,)
assert obs["rgb"].shape == (128, 128, 3)
assert obs["depth"].shape == (128, 128, 1)
assert obs["depth"].dtype == np.float32
```

Observation modes are `state`, `rgb`, `depth`, `rgbd`, `hybrid` (state+RGB),
and `hybrid_rgbd` (state+RGB+depth). RGB remains `uint8`; depth is a separate
float32 metre channel clipped to `depth_max_m`, so RGB-D is represented as a
Gymnasium `Dict` rather than mixing incompatible dtypes into one tensor.

The visual path uses ManiSoft's existing Blender scene. The package adds a
non-invasive Blender Z-pass adapter and does not modify ManiSoft source. It must
run in an interpreter that provides `bpy`; ordinary state mode does not need
`bpy`. If the selected episode renderer cannot provide a real metric Z-buffer,
depth modes fail closed. The API does not synthesize depth from the 85D state.

The aerial adapter uses its own MuJoCo eye-in-hand camera and metric depth path;
it does not depend on the COLL Blender adapter.

## Vector environments

```python
from manisoftrl import make_vector_env

envs = make_vector_env(
    4,
    {
        "case_dirs": case_root,
        "runtime_root": runtime_root,
        "asset_root": asset_root,
        "horizon": 1500,
    },
    asynchronous=True,
)
```

Use process-based vectorization for real ManiSoft simulation. One process is
restricted to one runtime/asset pair because ManiSoft resolves assets from the
process working directory.

`examples/vector_state_smoke.py` is the smallest real two-process check.
`make_aerial_vector_env` and `examples/aerial_vector_smoke.py` provide the same
process boundary for the flying-boom environment.

## Baselines and validation

The existing Route-B SAC/checkpoint/evaluator remains the COLL training path.
Flying-boom docking has an independent pure-online SAC/TD3 entry point with replay,
resume, frozen checkpoint evaluation, and protected parent promotion:

```powershell
python -m pip install -e ".[baseline]"
python examples/train_sac_sb3.py --runtime-root <ManiSoft> --case-root <cases>
python examples/train_aerial_sac.py --runtime-root <ManiSoft> --output <run>
python examples/train_aerial_sac.py --algorithm TD3 --mechanism rigid --action-mode direct3 --learning-starts 0 --output <run>
```

`train_aerial_sac.py` defaults to SAC, the 4D smooth-synergy policy action, and
state observations. `--action-mode direct19` keeps fully independent continuum
actuator outputs; the rigid mechanism requires direct 3D control.
The promoted state checkpoint can be replayed with synchronized RGB-D using
`examples/render_aerial_sac.py`; that replay does not turn it into a visual
policy. The script also writes a presentation-only 1280 x 720 HUD animation;
the raw RGB-D files remain separate evidence. The eight motions created by
`make_aerial_cases(8)` are generated canaries, not eight real flight cases.
There is currently no frozen formal aerial case set and no claimed case count.

The visual aircraft is deliberately outside the collision/success contract.
See `demo/README.md` and `demo/aircraft_assets.json` for the remotely acquired
FlightGear F-16/KC-135/A-10 packages, the F-16's declared refuel anchor, the
optional high-poly visual upgrade, and the shortest honest demonstration
sequence.

Generated checkpoints, replay buffers, and experiment evidence are intentionally
not bundled with the Python package. The repository focuses on the reusable
environment, contract, promotion gate, examples, and tests.

Run the platform tests with:

```powershell
python -m pytest
```

The unit suite validates the package-level contracts without requiring private
experiment data. Real COLL and RGB-D integration additionally requires the
external ManiSoft runtime and its separately distributed assets.

COLL and flying-boom docking remain separate contracts. Adding task-specific
policy logic to the common layer, or pretending that either adapter is a
universal task interface, is outside this MVP.

## Contributing

Run `python -m pytest` before opening a pull request. Keep policy-facing
contracts explicit, add regression coverage for behavior changes, and do not
commit datasets, checkpoints, credentials, or generated experiment outputs.

## License

ManiSoft-RL is released under the MIT License. See `LICENSE`. External runtimes,
datasets, and optional visual assets retain their own licenses; see `NOTICE.md`.
