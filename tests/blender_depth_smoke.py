from __future__ import annotations

import json
import tempfile
from pathlib import Path

import bpy
import numpy as np

from manisoftrl.blender_depth import render_metric_depth


scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x = 32
scene.render.resolution_y = 24
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"

with tempfile.TemporaryDirectory(prefix="manisoftrl-depth-smoke-") as directory:
    depth = render_metric_depth("Camera", Path(directory))

assert depth.shape == (24, 32, 1)
assert depth.dtype == np.float32
assert np.all(np.isfinite(depth))
assert float(depth.min()) >= 0.0
assert float(depth.max()) > float(depth.min())
print(
    json.dumps(
        {
            "status": "pass",
            "shape": list(depth.shape),
            "dtype": str(depth.dtype),
            "minimum_m": float(depth.min()),
            "maximum_m": float(depth.max()),
        }
    )
)
