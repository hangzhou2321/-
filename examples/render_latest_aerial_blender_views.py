"""Render the latest remote aerial-refuelling Blender scene from two views."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import bpy
from mathutils import Vector


FAR_CAMERA_POSITION = Vector((119.67485046386719, 39.35637664794922, 62.45428466796875))
FAR_CAMERA_TARGET = Vector((6.120243072509766, -0.38773632049560547, -0.0007495880126953125))
CLOSE_CAMERA_POSITION = Vector((21.05, -4.62, -2.53131103515625))
CLOSE_CAMERA_TARGET = Vector((21.05, -4.62, -0.03131100535392761))
CLOSE_VIEW_MESHES = {
    "B_EXACT_GRAY_SOFTROBOT",
    "KC135_EXACT_SOURCE_NOZZLE_HEAD",
    "ATP_B_RECEPTACLE_AFT_BORE_MIN_ENVELOPE",
    "ATP_B_RECEPTACLE_DARK_BACKSTOP_PRESENTATION",
    "ATP_B_RECEPTACLE_SLIDING_VALVE_MIN_TRAVEL",
    "ATP_B_RECEPTACLE_TULIP_MIN_ENVELOPE",
}


def _look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def _camera(name: str, position: Vector, target: Vector, lens_mm: float) -> bpy.types.Object:
    data = bpy.data.cameras.get(name) or bpy.data.cameras.new(name)
    camera = bpy.data.objects.get(name) or bpy.data.objects.new(name, data)
    if camera.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(camera)
    camera.location = position
    camera.data.lens = lens_mm
    _look_at(camera, target)
    return camera


def _close_view_light() -> None:
    data = bpy.data.lights.get("soft_arm_close_light") or bpy.data.lights.new("soft_arm_close_light", "AREA")
    light = bpy.data.objects.get("soft_arm_close_light") or bpy.data.objects.new("soft_arm_close_light", data)
    if light.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(light)
    light.location = CLOSE_CAMERA_POSITION
    light.data.energy = 900.0
    light.data.shape = "DISK"
    light.data.size = 3.0
    _look_at(light, CLOSE_CAMERA_TARGET)


def _render_still(camera: bpy.types.Object, frame: int, output: Path) -> None:
    scene = bpy.context.scene
    scene.camera = camera
    scene.frame_set(frame)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def _render_video(camera: bpy.types.Object, output: Path) -> None:
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.audio_codec = "NONE"
    scene.render.filepath = str(output)
    bpy.ops.render.render(animation=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.fps = 24

    far = _camera("latest_far_view", FAR_CAMERA_POSITION, FAR_CAMERA_TARGET, 50.0)
    close = _camera("latest_soft_arm_close_view", CLOSE_CAMERA_POSITION, CLOSE_CAMERA_TARGET, 43.0)
    for frame, name in ((1, "far_start.png"), (72, "far_approach.png"), (120, "far_precontact.png")):
        _render_still(far, frame, output / name)
    if not args.preview_only:
        _render_video(far, output / "far_view.mp4")

    original_visibility = {obj.name: obj.hide_render for obj in bpy.data.objects if obj.type == "MESH"}
    _close_view_light()
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.hide_render = obj.name not in CLOSE_VIEW_MESHES
    for frame, name in ((1, "soft_arm_start.png"), (72, "soft_arm_approach.png"), (120, "soft_arm_precontact.png")):
        _render_still(close, frame, output / name)
    if not args.preview_only:
        _render_video(close, output / "soft_arm_close_view.mp4")
    for name, hidden in original_visibility.items():
        bpy.data.objects[name].hide_render = hidden


if __name__ == "__main__":
    main()
