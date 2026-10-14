from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import numpy.typing as npt


def render_metric_depth(
    camera_name: str | None,
    output_dir: str | Path,
) -> npt.NDArray[np.float32]:
    """Render Blender's camera Z-pass without changing ManiSoft's renderer.

    The returned array is top-left-origin ``H x W x 1`` float32 depth. ManiSoft
    scene coordinates are metres, so the unmodified Blender Z-pass is metric.
    This function must run in a Python interpreter that provides ``bpy``.
    """

    try:
        import bpy
    except ImportError as error:
        raise RuntimeError(
            "metric depth requires a Blender Python runtime with bpy"
        ) from error

    scene = bpy.context.scene
    camera = scene.camera if camera_name is None else bpy.data.objects.get(camera_name)
    if camera is None or camera.type != "CAMERA":
        available = sorted(
            item.name for item in bpy.data.objects if item.type == "CAMERA"
        )
        raise KeyError(
            f"camera {camera_name!r} not found; available cameras: {available}"
        )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    token = f"manisoftrl_depth_{uuid.uuid4().hex}_"
    old_camera = scene.camera
    old_use_nodes = scene.use_nodes
    view_layer = scene.view_layers[0]
    old_use_pass_z = view_layer.use_pass_z
    depth_image = None
    created_paths: list[Path] = []
    render_layers = None
    file_output = None

    try:
        scene.camera = camera
        view_layer.use_pass_z = True
        scene.use_nodes = True
        nodes = scene.node_tree.nodes
        links = scene.node_tree.links
        render_layers = nodes.new("CompositorNodeRLayers")
        file_output = nodes.new("CompositorNodeOutputFile")
        file_output.base_path = str(destination)
        file_output.format.file_format = "OPEN_EXR"
        file_output.format.color_mode = "BW"
        file_output.format.color_depth = "32"
        file_output.file_slots[0].path = token
        links.new(render_layers.outputs["Depth"], file_output.inputs[0])

        bpy.ops.render.render()
        created_paths = sorted(destination.glob(f"{token}*.exr"))
        if len(created_paths) != 1:
            raise RuntimeError(
                f"Blender depth pass produced {len(created_paths)} files"
            )
        depth_image = bpy.data.images.load(
            str(created_paths[0]), check_existing=False
        )
        width, height = map(int, depth_image.size)
        channels = int(depth_image.channels)
        pixels = np.asarray(depth_image.pixels[:], dtype=np.float32)
        expected = width * height * channels
        if width < 1 or height < 1 or channels < 1 or pixels.size != expected:
            raise RuntimeError(
                "Blender returned a malformed depth image: "
                f"size={(width, height)}, channels={channels}, pixels={pixels.size}"
            )
        depth = pixels.reshape(height, width, channels)[..., 0]
        depth = np.flipud(depth).copy()[..., None]
        if np.any(np.isnan(depth)) or np.any(depth < 0.0):
            raise RuntimeError("Blender returned invalid negative/NaN depth")
        return depth.astype(np.float32, copy=False)
    finally:
        if depth_image is not None:
            bpy.data.images.remove(depth_image)
        if scene.use_nodes and scene.node_tree is not None:
            nodes = scene.node_tree.nodes
            if file_output is not None:
                nodes.remove(file_output)
            if render_layers is not None:
                nodes.remove(render_layers)
        scene.camera = old_camera
        view_layer.use_pass_z = old_use_pass_z
        scene.use_nodes = old_use_nodes
        for path in created_paths:
            path.unlink(missing_ok=True)
