# Flying-boom demo

## One centre

The scene centre is the moving receiver-receptacle mouth frame, not either
aircraft's centre of mass and not the boom base. Its origin is the mouth centre;
its positive insertion axis is the receptacle axis used by the environment.
The RGB-D camera, reticle, lateral error, orientation error, insertion depth,
success predicate, and any presentation-only aircraft shell must all resolve to
this frame.

This keeps the demonstration honest: the learned task is terminal docking after
the two aircraft have already reached pre-contact formation. Whole-aircraft
flight control is outside the current contract.

## Selected aircraft models

The default engineering asset is FlightGear's official **F-16** package:

- source: <https://wiki.flightgear.org/F16>;
- direct package: <https://mirrors.ibiblio.org/flightgear/ftp/Aircraft-2024/f16.zip>;
- package license: GPLv2 or later, stated by the source and included in the zip;
- model path: `f16/Models/f16.ac` with a `2048 x 2048` RGBA body texture;
- main AC3D body: 10,627 vertices and 15,834 surfaces;
- exact boom receptacle anchor supplied by the model:
  `(0.1054, 0.0, 0.6500) m` in the FlightGear aircraft frame;
- remote package SHA-256:
  `6911d6ed92f6ed2dd5dd68068cbfc17e3581b36033925f617d5a5b19033fb052`.

This model is task-correct: the FlightGear package declares a boom refuelling
receptacle and exposes its anchor. Bind the ManiSoft receptacle-mouth frame to
that anchor after converting the external AC3D presentation asset. Do not infer
the anchor by eye.

The verified FlightGear-to-AC3D/OBJ coordinate mapping is
`(x_fg, y_fg, z_fg) -> (x_fg, z_fg, -y_fg)`. Therefore the F-16 refuel anchor
used by the renderer is `(0.1054, 0.6500, 0.0) m` in the converted model, not
`(-0.1054, 0.6500, 0.0)`. Runtime construction checks that this transformed
point and the ManiSoft receptacle mouth agree within `1e-6 m`.

FlightGear's official **KC-135E** package is retained only for the three-second
wide context shot. Its main body is much lighter (4,237 vertices and 3,865
surfaces), so it is not the close-up hero asset. Source:
<https://wiki.flightgear.org/Boeing_KC-135E>; package:
<https://mirrors.ibiblio.org/flightgear/ftp/Aircraft-2024/KC135.zip>; remote
package SHA-256:
`5584ba0a9fdda86fb289688339202a0dee0dae4c61e40b4463608bdb788bee12`.

The KC-135 hard boom is embedded in `KC135.ac`. Its FlightGear hinge is
`(15.10, 0.0, -0.13) m`, which converts to
`(15.10, -0.13, 0.0) m` in the OBJ. The centre of the terminal tube ring,
computed from the source AC3D vertices, is
`(22.6121641667, -5.7443075, -0.0313110108) m`. The ManiSoft scene binds that
hard-boom endpoint to the continuum segment root and checks the binding at
every environment construction. The source hard boom remains visual-only; the
controlled compliant terminal segment is the ManiSoft Cosserat rod.

An alternate receiver is FlightGear's **A-10** package. It is substantially
denser than the current F-16 body (30,313 vertices and 32,985 surfaces), has a
dedicated `A-10-fuel-aar.ac` refuelling component, and includes GPLv2 license
text. Its 9,838,104-byte archive was acquired directly on the remote host with
SHA-256
`367ba0c86dcc805f1e5e7abad3ce49962e75e4e33070acfda37e033179c0925b`.
Use it for an alternate receiver-context shot, not as a replacement for the
F-16 anchor until its receptacle transform has been extracted and frozen.

Two further official packages are recorded but not claimed as acquired:
FlightGear `F-15.zip` as another boom receiver and `DC-10.zip`, whose aircraft
set includes KC-10A, as a second tanker. A partial F-15 transfer was removed;
an incomplete archive is not evidence of an available model.

Both archives were downloaded directly on the remote host without a proxy and
are kept outside the Python package under `.downloads/flightgear_2024/`. They
are converted to presentation-only OBJ meshes by `demo/ac3d_to_obj.py` and can
be inserted into the MuJoCo or Blender rollout renderer without changing the
physics. Their exact machine-readable acquisition record is
`aircraft_assets.json`.

For a later photorealistic close-up, the selected upgrade is **F 16 High Poly
(Subdive Ready)** by Andante: 300.5k triangles, 150.5k vertices, and CC BY at
<https://sketchfab.com/3d-models/f-16-high-poly-subdive-ready-eb37548347bf408eb4d5547ee7b67322>.
Sketchfab requires an authenticated download, so the project must not scrape it
or claim it as acquired. If used, preserve the archive, license, attribution,
source URL, download date, and SHA-256 together.

Do not use the free CGTrader KC-135 candidate in a visual-policy environment:
its listing says `Royalty Free License (no AI)`.

The aircraft mesh is presentation geometry. It must not change collision,
reward, success, camera calibration, or the checkpoint contract. To use it for
visual-policy training later, freeze a new visual contract and retrain/evaluate
under that contract.

## Short demonstration

1. **Context, 3 seconds:** one wide receiver-aircraft shot, with the receptacle
   frame bound to the F-16 package's declared refuel anchor and the boom entering
   the frame.
2. **Evidence, 15 seconds:** the real rollout in `rollout_demo.gif`, centred on
   the receptacle, with synchronized RGB, metric depth, insertion, alignment,
   and force telemetry.
3. **Result, 2 seconds:** hold `final_demo.png` and show the exact success values,
   case/seed, checkpoint hash, and environment-contract hash.

Run the actual replay on the remote runtime:

```powershell
python examples/render_aerial_sac.py `
  --runtime-root <remote-ManiSoft> `
  --checkpoint evidence/aerial_sac_state_v1/promoted.zip `
  --output <remote-output> `
  --case 1 `
  --case-seed 20260808 `
  --episode-seed 80720000
```

The script retains raw RGB, metric depth, the raw side-by-side animation, and a
separate 1280 x 720 presentation animation. The HUD is a presentation layer;
the JSON report and raw sensor arrays remain the evidence.

Verify the two source-model bindings directly in the ManiSoft scene with:

```powershell
python examples/verify_aerial_anchors.py `
  --runtime-root <remote-ManiSoft> `
  --aircraft-visual-root <remote-converted-aircraft> `
  --output <remote-output>/anchor_verification.json
```

For a perspective Blender render and an editable `.blend` scene, run:

```powershell
python examples/render_aerial_blender.py `
  --runtime-root <remote-ManiSoft> `
  --aircraft-visual-root <remote-converted-aircraft> `
  --checkpoint <promoted.zip> `
  --output <remote-output> `
  --case 1 `
  --camera-mode overview
```

This entry point replays the same deterministic state policy, exports the
trajectory, and renders it with Blender EEVEE Next. The physical boom radius
remains 30 mm; only the aircraft-context presentation curve is reduced to an
8 mm radius so the scaled aircraft and boom remain visually proportional.

## Next research idea

The useful research centre is **receptacle-frame RGB-D residual perception with
protected online RL**, not full-flight control yet:

- keep the current state checkpoint as the protected parent;
- add a visual encoder to one unified actor, with state and RGB-D as an explicit
  hybrid contract first;
- train only from environment reward and current-policy interaction: no expert
  actions, behaviour cloning, waypoint, FSM, or stage input;
- continually revisit the frozen parent-success scenarios and add new motion,
  lighting, camera, and port-pose scenarios;
- promote a candidate only when paired evaluation retains every protected
  success and adds at least one new success without force regression.

Replay protects exposure; the promotion gate protects deployment. Neither one
alone is a mathematical guarantee against catastrophic forgetting.
