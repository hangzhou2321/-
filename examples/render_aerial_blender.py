from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from manisoftrl import (
    AERIAL_STATE_LAYOUT,
    AerialRefuelEnv,
    AerialSynergyActionWrapper,
    make_aerial_cases,
)


_AIRCRAFT_SCALE = 0.08
_PRESENTATION_BOOM_RADIUS_M = 0.008
_MODEL_TO_WORLD_ROTATION = np.asarray(
    ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)
_F16_REFUEL_ANCHOR_MODEL_M = np.asarray((0.1054, 0.6500, 0.0))
_KC135_BOOM_TIP_MODEL_M = np.asarray(
    (22.6121641667, -5.7443075, -0.0313110108)
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect_trajectory(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        from stable_baselines3 import SAC
    except ImportError as error:
        raise SystemExit("install the baseline extra: pip install -e .[baseline]") from error

    cases = make_aerial_cases(8, seed=args.case_seed)
    case_id = str(args.case)
    if case_id not in cases:
        raise KeyError(f"unknown case {case_id!r}")
    base = AerialRefuelEnv(
        cases=cases,
        default_case=case_id,
        runtime_root=args.runtime_root,
        horizon=args.horizon,
        observation_mode="state",
    )
    env = AerialSynergyActionWrapper(base)
    model = SAC.load(args.checkpoint, device=args.device)
    points: list[np.ndarray] = []
    receptacles: list[np.ndarray] = []
    steps: list[int] = []
    insertions: list[float] = []

    def capture(observation: np.ndarray, info: dict[str, Any]) -> None:
        state = np.asarray(observation, dtype=np.float64)
        receptacle = cases[case_id].motion(float(info["physics_time_s"]))[0]
        nodes = state[
            AERIAL_STATE_LAYOUT["boom_positions_relative_to_receptacle_10x3"]
        ].reshape(10, 3)
        tip = state[AERIAL_STATE_LAYOUT["tip_position_relative_to_receptacle"]]
        points.append(np.vstack((nodes, tip)) + receptacle)
        receptacles.append(np.asarray(receptacle, dtype=np.float64))
        steps.append(int(info["step_count"]))
        insertions.append(float(info["insertion_depth_m"]))

    try:
        observation, info = env.reset(
            seed=args.episode_seed,
            options={"case": case_id},
        )
        capture(observation, info)
        terminated = False
        truncated = False
        episode_return = 0.0
        while not terminated and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            if info["step_count"] % args.frame_stride == 0:
                capture(observation, info)
        if steps[-1] != int(info["step_count"]):
            capture(observation, info)
        report = {
            "official_success": bool(info["official_success"]),
            "termination_reason": info["termination_reason"],
            "episode_steps": int(info["step_count"]),
            "episode_return": episode_return,
            "insertion_depth_m": float(info["insertion_depth_m"]),
            "lateral_error_m": float(info["lateral_error_m"]),
            "orientation_error_rad": float(info["orientation_error_rad"]),
            "boom_extension_m": float(info["boom_extension_m"]),
            "peak_contact_force_n": float(
                info["contact"]["episode_peak_contact_force_n"]
            ),
            "environment_contract_sha256": base.contract_sha256,
        }
        if not report["official_success"]:
            raise RuntimeError("frozen policy did not succeed during trajectory export")
    finally:
        env.close()

    trajectory = {
        "boom_points_world_m": np.asarray(points, dtype=np.float64),
        "receptacle_position_m": np.asarray(receptacles, dtype=np.float64),
        "step_count": np.asarray(steps, dtype=np.int32),
        "insertion_depth_m": np.asarray(insertions, dtype=np.float64),
    }
    return trajectory, report


def _material(
    bpy: Any,
    name: str,
    colour: tuple[float, float, float, float],
    *,
    metallic: float = 0.0,
    roughness: float = 0.45,
) -> Any:
    material = bpy.data.materials.new(name)
    material.diffuse_color = colour
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = colour
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return material


def _load_obj_mesh(bpy: Any, path: Path, name: str, material: Any) -> Any:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("v "):
            values = line.split()
            vertices.append((float(values[1]), float(values[2]), float(values[3])))
        elif line.startswith("f "):
            values = line.split()
            faces.append(tuple(int(value.split("/")[0]) - 1 for value in values[1:4]))
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.append(material)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    aircraft = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(aircraft)
    return aircraft


def _keyframe_linear(bpy: Any) -> None:
    for action in bpy.data.actions:
        for curve in action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"


def _presentation_hose_points(
    physical_points: np.ndarray,
    receptacle: np.ndarray,
    insertion_depth_m: float,
    *,
    count: int = 33,
) -> np.ndarray:
    start = np.asarray(physical_points[0], dtype=np.float64)
    physical_tip = np.asarray(physical_points[-1], dtype=np.float64)
    relative = physical_tip - receptacle
    axial = -float(relative[1])
    height = (
        -min(insertion_depth_m, 0.015)
        if axial >= 0.0
        else -axial
    )
    end = np.asarray(
        (
            receptacle[0] + relative[0],
            receptacle[1] + relative[2],
            receptacle[2] + height,
        ),
        dtype=np.float64,
    )
    chord = end - start
    control_1 = start + 0.34 * chord + np.asarray((0.045, 0.0, 0.055))
    control_2 = end + np.asarray((0.030, 0.0, 0.130))
    values = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return np.asarray(
        [
            (1.0 - value) ** 3 * start
            + 3.0 * (1.0 - value) ** 2 * value * control_1
            + 3.0 * (1.0 - value) * value**2 * control_2
            + value**3 * end
            for value in values
        ],
        dtype=np.float64,
    )


def _build_and_render(
    args: argparse.Namespace,
    trajectory: dict[str, np.ndarray],
) -> dict[str, Any]:
    try:
        import bpy
        from mathutils import Vector
    except ImportError as error:
        raise SystemExit("this entry point requires a Python environment with bpy") from error

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.fps = 25
    scene.render.fps_base = 2.0
    scene.render.film_transparent = False

    world = bpy.data.worlds.new("Flight sky")
    world.use_nodes = True
    scene.world = world
    nodes = world.node_tree.nodes
    nodes.clear()
    coordinates = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    mapping = nodes.new("ShaderNodeMapRange")
    mapping.inputs["From Min"].default_value = -1.0
    mapping.inputs["From Max"].default_value = 1.0
    gradient = nodes.new("ShaderNodeValToRGB")
    gradient.color_ramp.elements[0].color = (0.015, 0.055, 0.14, 1.0)
    gradient.color_ramp.elements[1].color = (0.34, 0.62, 0.88, 1.0)
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 0.45
    output = nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(coordinates.outputs["Normal"], separate.inputs["Vector"])
    world.node_tree.links.new(separate.outputs["Z"], mapping.inputs["Value"])
    world.node_tree.links.new(mapping.outputs["Result"], gradient.inputs["Fac"])
    world.node_tree.links.new(gradient.outputs["Color"], background.inputs["Color"])
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])

    tanker_material = _material(bpy, "KC-135 aluminium", (0.62, 0.67, 0.73, 1.0), metallic=0.55, roughness=0.28)
    receiver_material = _material(bpy, "F-16 grey", (0.22, 0.27, 0.34, 1.0), metallic=0.35, roughness=0.36)
    boom_material = _material(bpy, "Soft boom", (0.95, 0.20, 0.035, 1.0), roughness=0.32)
    port_material = _material(bpy, "Receiver port", (0.95, 0.62, 0.05, 1.0), metallic=0.25, roughness=0.30)

    visual_root = args.aircraft_visual_root
    tanker = _load_obj_mesh(bpy, visual_root / "KC135.obj", "KC-135", tanker_material)
    receiver = _load_obj_mesh(bpy, visual_root / "f16.obj", "F-16", receiver_material)
    aircraft_rotation = (0.5, 0.5, -0.5, -0.5)
    for aircraft in (tanker, receiver):
        aircraft.rotation_mode = "QUATERNION"
        aircraft.rotation_quaternion = aircraft_rotation
        aircraft.scale = (_AIRCRAFT_SCALE,) * 3
    receptacles = trajectory["receptacle_position_m"]
    boom_roots = trajectory["boom_points_world_m"][:, 0]
    tanker.location = boom_roots[0] - _MODEL_TO_WORLD_ROTATION @ (
        _KC135_BOOM_TIP_MODEL_M * _AIRCRAFT_SCALE
    )
    receiver_anchor_offset = _MODEL_TO_WORLD_ROTATION @ (
        _F16_REFUEL_ANCHOR_MODEL_M * _AIRCRAFT_SCALE
    )
    for frame_index, receptacle in enumerate(receptacles, start=1):
        receiver.location = receptacle - receiver_anchor_offset
        receiver.keyframe_insert(data_path="location", frame=frame_index)

    presentation_points = np.asarray(
        [
            _presentation_hose_points(points, receptacle, insertion)
            for points, receptacle, insertion in zip(
                trajectory["boom_points_world_m"],
                receptacles,
                trajectory["insertion_depth_m"],
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    curve_data = bpy.data.curves.new("Soft boom trajectory", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 2
    curve_data.bevel_depth = _PRESENTATION_BOOM_RADIUS_M
    curve_data.bevel_resolution = 4
    spline = curve_data.splines.new("POLY")
    point_count = presentation_points.shape[1]
    spline.points.add(point_count - 1)
    boom = bpy.data.objects.new("Soft boom", curve_data)
    bpy.context.collection.objects.link(boom)
    curve_data.materials.append(boom_material)
    for frame_index, frame_points in enumerate(
        presentation_points, start=1
    ):
        for point, coordinate in zip(spline.points, frame_points, strict=True):
            point.co = (*map(float, coordinate), 1.0)
            point.keyframe_insert(data_path="co", frame=frame_index)

    nozzle_material = _material(
        bpy,
        "Nozzle",
        (0.62, 0.66, 0.70, 1.0),
        metallic=0.75,
        roughness=0.22,
    )
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.009)
    nozzle_tip = bpy.context.object
    nozzle_tip.name = "Nozzle tip"
    nozzle_tip.data.materials.append(nozzle_material)
    for frame_index, frame_points in enumerate(
        presentation_points, start=1
    ):
        nozzle_tip.location = frame_points[-1]
        nozzle_tip.keyframe_insert(data_path="location", frame=frame_index)

    recess_material = _material(
        bpy,
        "Receptacle recess",
        (0.025, 0.035, 0.045, 1.0),
        metallic=0.55,
        roughness=0.20,
    )
    bpy.ops.mesh.primitive_cube_add()
    port_frame = bpy.context.object
    port_frame.name = "F-16 receptacle frame"
    port_frame.dimensions = (0.072, 0.120, 0.005)
    port_frame.data.materials.append(port_material)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    frame_bevel = port_frame.modifiers.new("Rounded frame", "BEVEL")
    frame_bevel.width = 0.012
    frame_bevel.segments = 5
    bpy.ops.mesh.primitive_cube_add()
    port_recess = bpy.context.object
    port_recess.name = "F-16 receptacle recess"
    port_recess.dimensions = (0.050, 0.092, 0.006)
    port_recess.data.materials.append(recess_material)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    recess_bevel = port_recess.modifiers.new("Rounded recess", "BEVEL")
    recess_bevel.width = 0.009
    recess_bevel.segments = 5
    for frame_index, receptacle in enumerate(receptacles, start=1):
        port_frame.location = receptacle + np.asarray((0.0, 0.0, 0.001))
        port_recess.location = receptacle + np.asarray((0.0, 0.0, 0.005))
        port_frame.keyframe_insert(data_path="location", frame=frame_index)
        port_recess.keyframe_insert(data_path="location", frame=frame_index)

    bpy.ops.object.light_add(type="SUN", location=(2.0, -2.0, 5.0))
    sun = bpy.context.object
    sun.rotation_euler = (math.radians(28.0), math.radians(-18.0), math.radians(35.0))
    sun.data.energy = 3.0
    sun.data.angle = math.radians(4.0)
    bpy.ops.object.light_add(type="AREA", location=(2.5, -1.0, 4.5))
    fill = bpy.context.object
    fill.data.energy = 900.0
    fill.data.shape = "DISK"
    fill.data.size = 5.0
    fill.rotation_euler = (math.radians(18.0), 0.0, math.radians(120.0))

    frame_count = int(trajectory["boom_points_world_m"].shape[0])
    scene.frame_start = 1
    scene.frame_end = frame_count

    def make_camera(
        name: str,
        location: tuple[float, float, float],
        target: tuple[float, float, float] | np.ndarray,
        lens: float,
        focus: Any,
    ) -> Any:
        camera_data = bpy.data.cameras.new(name)
        camera = bpy.data.objects.new(name, camera_data)
        bpy.context.collection.objects.link(camera)
        camera.location = location
        target_vector = Vector(tuple(map(float, target)))
        camera.rotation_euler = (target_vector - camera.location).to_track_quat(
            "-Z", "Y"
        ).to_euler()
        camera_data.lens = lens
        camera_data.dof.use_dof = True
        camera_data.dof.focus_object = focus
        camera_data.dof.aperture_fstop = 8.0
        return camera

    camera_cuts: list[int] = []
    if args.camera_mode == "overview":
        wide = make_camera(
            "01 Wide formation",
            (7.6, -2.8, 4.1),
            (0.0, 1.55, 1.02),
            52.0,
            boom,
        )
        approach = make_camera(
            "02 Soft hose approach",
            (5.2, -2.4, 2.8),
            (0.0, 0.12, 0.96),
            58.0,
            boom,
        )
        close = make_camera(
            "03 F-16 receptacle close-up",
            (2.4, -1.8, 1.9),
            receptacles[-1],
            65.0,
            port_recess,
        )
        approach_frame = max(2, int(round(frame_count * 0.34)))
        close_frame = max(approach_frame + 1, int(round(frame_count * 0.72)))
        scene.camera = wide
        scene.timeline_markers.new("Wide formation", frame=1).camera = wide
        scene.timeline_markers.new(
            "Soft hose approach", frame=approach_frame
        ).camera = approach
        scene.timeline_markers.new(
            "F-16 receptacle close-up", frame=close_frame
        ).camera = close
        camera_cuts = [1, approach_frame, close_frame]
    else:
        scene.camera = make_camera(
            "Cinematic camera",
            (7.6, -2.8, 4.1),
            (0.0, 1.55, 1.02),
            52.0,
            boom,
        )
        camera_cuts = [1]
    _keyframe_linear(bpy)

    output = args.output
    scene.render.image_settings.file_format = "PNG"
    scene.frame_set(1)
    scene.render.filepath = str(output / "preview_start.png")
    bpy.ops.render.render(write_still=True)
    if args.camera_mode == "overview":
        scene.frame_set(camera_cuts[1])
        scene.render.filepath = str(output / "preview_approach.png")
        bpy.ops.render.render(write_still=True)
        scene.frame_set(camera_cuts[2])
        scene.render.filepath = str(output / "preview_close.png")
        bpy.ops.render.render(write_still=True)
    scene.frame_set(frame_count)
    scene.render.filepath = str(output / "preview_final.png")
    bpy.ops.render.render(write_still=True)

    video_path = output / (
        "overview_demo.mp4" if args.camera_mode == "overview" else "blender_demo.mp4"
    )
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.audio_codec = "NONE"
    scene.render.filepath = str(video_path)
    blend_path = output / (
        "aerial_refuel_overview.blend"
        if args.camera_mode == "overview"
        else "aerial_refuel_3d.blend"
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), check_existing=False)
    bpy.ops.render.render(animation=True)
    return {
        "renderer": "Blender EEVEE Next",
        "frames": frame_count,
        "fps": 12.5,
        "resolution": [args.width, args.height],
        "camera_mode": args.camera_mode,
        "camera_cut_frames": camera_cuts,
        "presentation_boom_radius_m": _PRESENTATION_BOOM_RADIUS_M,
        "physical_boom_radius_m": 0.030,
        "presentation_aircraft_scale": _AIRCRAFT_SCALE,
        "presentation_curve": "cubic_soft_hose_with_top_port_terminal_bend",
        "presentation_frame_mapping": (
            "simulation_minus_y_insertion_axis_to_f16_top_port_minus_z_normal"
        ),
        "f16_refuel_anchor_source_m": [0.1054, 0.0, 0.6500],
        "blend_artifact": blend_path.name,
        "video_artifact": video_path.name,
        "video_sha256": _sha256(video_path),
        "blend_sha256": _sha256(blend_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay frozen aerial SAC and render a 3D Blender presentation"
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--aircraft-visual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="1")
    parser.add_argument("--case-seed", type=int, default=20260808)
    parser.add_argument("--episode-seed", type=int, default=80720000)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--camera-mode",
        choices=("perspective", "overview"),
        default="perspective",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.runtime_root = args.runtime_root.expanduser().resolve(strict=True)
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    args.aircraft_visual_root = args.aircraft_visual_root.expanduser().resolve(strict=True)
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("KC135.obj", "f16.obj"):
        if not (args.aircraft_visual_root / name).is_file():
            raise FileNotFoundError(f"missing Blender presentation mesh {name}")

    trajectory, report = _collect_trajectory(args)
    np.savez_compressed(args.output / "trajectory.npz", **trajectory)
    report.update(_build_and_render(args, trajectory))
    report.update(
        {
            "status": "pass",
            "policy_observation": "82D_state_only",
            "aircraft_models": ["FlightGear KC-135E", "FlightGear F-16"],
            "aircraft_models_role": "presentation_only",
        }
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
