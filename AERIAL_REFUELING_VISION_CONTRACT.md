# Aerial-refuelling simulation MVP contract

Contract date: 2026-08-07. Environment id: `ManiSoft-AerialRefuel-v0`.

This is the implemented flying-boom terminal-docking simulation contract. It
is not whole-aircraft control, real-camera, fuel-transfer, hardware, safety, or
flight-certification evidence.

## Boundary

- Aircraft autopilots hold the pre-contact formation outside this environment.
- The environment action is 19D: 18 normalized ManiSoft `SplineMuscle`
  activations plus one telescopic-extension-rate command; there is no gripper.
- The receiver receptacle is a prescribed six-degree-of-freedom MuJoCo mocap
  body driven by a frozen case.
- The boom is the original `ManiSoftSimBackend` PyElastica `CosseratRod`; it is
  not replaced by a joint or kinematic surrogate.
- Receiver motion affects contact and vision. Boom forces do not yet feed back
  into aircraft flight dynamics.
- There is no runtime FSM, waypoint, expert action, stage variable, case id, or
  clock in the policy input.

## Action and state

Environment action is float32 `19D` in `[-1, 1]`. The first 18 values are
reshaped to `6 x 3`, scaled to physical activation `[-0.01, 0.01]`, and passed
to the existing `SplineMuscle`. The last value commands telescopic speed in
`[-0.5, 0.5] m/s`; extension is limited to `0.19 m` inside a `0.22 m`
mechanical joint.

The SAC baseline defaults to a separate 4D policy-action contract:
three bending values are tiled over all six spline control points and the last
value commands extension rate. `direct19` remains available. This continuous
actuator basis contains no expert action, task stage, waypoint, or FSM.

State is float32 `82D`:

- 30D: ten boom-node positions relative to the receptacle;
- 30D: the same nodes' velocities relative to the receptacle;
- 3D: tip position relative to the receptacle;
- 3D: tip velocity relative to the receptacle;
- 3D: tip axis;
- 3D: receptacle axis;
- 3D: receptacle linear velocity;
- 3D: receptacle angular velocity;
- 1D: insertion depth in metres;
- 1D: contact force divided by the safe limit.
- 1D: telescopic extension in metres;
- 1D: telescopic extension velocity in metres per second.

## Vision

Observation modes are `state`, `rgb`, `depth`, `rgbd`, `hybrid`, and
`hybrid_rgbd`. The primary eye-in-hand camera is `boom_tip_camera`. MuJoCo
renders synchronized uint8 RGB and float32 metric depth from the same scene
state. Missing RGB-D support is an error; state-derived pseudo-depth is
prohibited.

## Simulation predicate

Success requires all of the following for five consecutive control steps:

- insertion depth at least `0.050 m`;
- lateral error at most `0.035 m`;
- orientation error at most `12 degrees`;
- peak contact force in the control step at most `15 N`;
- valid finite continuum state.

An episode terminates on success, continuum-integrity failure, contact force
above `30 N`, leaving the `0.60 m` terminal workspace, or disconnecting more
than `0.020 m` behind the mouth after engagement. Horizon exhaustion is the
only truncation.

These are simulation-MVP thresholds, not claimed aircraft engineering limits.
Changing them changes `env.contract_sha256` and creates a different result
contract.

## Cases and evaluation

`make_aerial_cases()` deterministically creates the versioned motion panel;
case `0` is static and the remaining cases use bounded three-axis translation
and rotation. The default eight generated motions are development canaries,
not real flight cases and not a formal benchmark. A formal result requires a
separately designed and frozen scenario suite, common random numbers, a single
checkpoint per role, the same observation/camera contract, and the promotion
gate. Its size has not been selected; this project currently has no formal
aerial case set.

State, RGB, RGB-D, and hybrid policies remain separate result contracts.
Promotion requires strict success-set expansion plus no force, collision, or
disconnect regression.

## Current evidence boundary

Remote evidence now covers real online SAC as well as construction and sensor
smokes. Training used only environment interaction: no behaviour cloning,
expert actions, demonstrations, task FSM, stages, or waypoints. The protected
state-policy checkpoint at step `9000` has SHA-256
`78ea55223d48f826df3ed0a91b1c777e8448a78c6c68250ac61efd11de5c0f0f`.

On the independent frozen eight-case motion canary (case seed `20260808`,
episode seeds `80720000..80720007`) that checkpoint scored `8/8` official
successes with `0 N` peak contact force in every case. A same-contract
candidate retained all eight but added none, so the strict-expansion gate kept
the parent. This is only a generated-motion canary result.

The state policy was also replayed on moving case `1` while synchronized RGB-D
was rendered. It succeeded in 772 steps at `51.21 mm` insertion, `4.32 mm`
lateral error, and `0 N` peak contact. The policy consumed only the 82D state;
no RGB-D policy has yet been trained. A formal aerial scenario-suite
evaluation, aircraft force feedback, fuel-transfer physics, real sensors, and
hardware evidence remain zero.
