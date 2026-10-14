from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np
import numpy.typing as npt

from .aerial import (
    AERIAL_ACTION_DIM,
    AERIAL_AIRCRAFT_VISUAL_SCALE,
    AERIAL_BOOM_DAMPING_CONSTANT,
    AERIAL_BOOM_DENSITY_KG_M3,
    AERIAL_BOOM_DIRECTION,
    AERIAL_BOOM_EXTENSION_COMMAND_MAX_M,
    AERIAL_BOOM_EXTENSION_RATE_LIMIT_M_S,
    AERIAL_BOOM_LENGTH_M,
    AERIAL_BOOM_NUM_ELEMENTS,
    AERIAL_BOOM_POISSON_RATIO,
    AERIAL_BOOM_RADIUS_M,
    AERIAL_BOOM_START_M,
    AERIAL_BOOM_YOUNGS_MODULUS_PA,
    AERIAL_MUSCLE_TORQUE_SCALE,
    AERIAL_MUSCLE_ACTIVATION_LIMIT,
    AERIAL_F16_REFUEL_ANCHOR_MODEL_M,
    AERIAL_KC135_BOOM_PIVOT_MODEL_M,
    AERIAL_KC135_BOOM_TIP_MODEL_M,
    AERIAL_PORT_HALF_OPENING_M,
    AERIAL_PORT_TUNNEL_DEPTH_M,
    AERIAL_PHYSICS_DT_S,
    AerialRefuelCase,
    AerialSnapshot,
)
from .runtime import configure_manisoft_runtime


_BOOM_START_M = np.asarray(AERIAL_BOOM_START_M, dtype=np.float64)
_BOOM_DIRECTION = np.asarray(AERIAL_BOOM_DIRECTION, dtype=np.float64)
_PRESENTATION_BOOM_RADIUS_M = 0.008
_MODEL_TO_WORLD_ROTATION = np.asarray(
    ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)
_RECEPTACLE_BASE_ROTATION = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)
_F16_REFUEL_ANCHOR_MODEL_M = np.asarray(
    AERIAL_F16_REFUEL_ANCHOR_MODEL_M, dtype=np.float64
)
_KC135_BOOM_TIP_MODEL_M = np.asarray(
    AERIAL_KC135_BOOM_TIP_MODEL_M, dtype=np.float64
)
_KC135_BOOM_PIVOT_MODEL_M = np.asarray(
    AERIAL_KC135_BOOM_PIVOT_MODEL_M, dtype=np.float64
)


def _matrix_to_wxyz(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ),
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            ) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ),
                dtype=np.float64,
            )
        elif index == 1:
            scale = math.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            ) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ),
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            ) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ),
                dtype=np.float64,
            )
    return quaternion / max(float(np.linalg.norm(quaternion)), 1.0e-12)


def _pretty_xml(root: ET.Element) -> str:
    parsed = minidom.parseString(ET.tostring(root, encoding="utf-8"))
    return "\n".join(
        line for line in parsed.toprettyxml(indent="  ").splitlines() if line.strip()
    )


def _add_receptacle_box(
    body: ET.Element,
    *,
    name: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
) -> None:
    ET.SubElement(
        body,
        "geom",
        name=name,
        type="box",
        pos=" ".join(map(str, position)),
        size=" ".join(map(str, size)),
        rgba="0.82 0.66 0.16 0",
        density="0",
        friction="1.5 0.05 0.02",
        solref="0.003 1",
        solimp="0.95 0.995 0.001",
        contype="1",
        conaffinity="1",
    )


def _scene_xml(
    boom_tip_config: Path,
    receptacle_position: npt.NDArray[np.float64],
    receptacle_quaternion: npt.NDArray[np.float64],
    aircraft_visual_root: Path | None,
) -> str:
    model = ET.Element("mujoco", model="manisoftrl_flying_boom_terminal_docking")
    ET.SubElement(model, "include", file=boom_tip_config.resolve().as_posix())
    ET.SubElement(
        model,
        "option",
        timestep=str(AERIAL_PHYSICS_DT_S),
        gravity="0 0 0",
        integrator="Euler",
    )
    visual = ET.SubElement(model, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "map", znear="0.01", zfar="8.0")
    ET.SubElement(visual, "rgba", haze="0.38 0.48 0.62 1")
    default = ET.SubElement(model, "default")
    ET.SubElement(
        default,
        "geom",
        conaffinity="1",
        condim="3",
        friction="1.5 0.05 0.02",
    )
    assets = ET.SubElement(model, "asset")
    if aircraft_visual_root is not None:
        ET.SubElement(
            assets,
            "mesh",
            name="kc135_visual_mesh",
            file=(aircraft_visual_root / "KC135.obj").resolve().as_posix(),
            scale="0.08 0.08 0.08",
        )
        ET.SubElement(
            assets,
            "mesh",
            name="f16_visual_mesh",
            file=(aircraft_visual_root / "f16.obj").resolve().as_posix(),
            scale="0.08 0.08 0.08",
        )
    worldbody = ET.SubElement(model, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        pos="1.5 1.0 2.5",
        dir="-0.3 -0.4 -1",
        directional="true",
        diffuse="0.85 0.85 0.82",
    )
    ET.SubElement(
        worldbody,
        "light",
        pos="-1.0 0.3 1.4",
        dir="0.4 -0.2 -0.6",
        directional="true",
        diffuse="0.40 0.48 0.62",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="far_scene_camera",
        pos="4.0 0.8 1.1",
        xyaxes="0 1 0 0 0 1",
        fovy="48",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="far_perspective_camera",
        pos="4.2 -2.2 2.4",
        xyaxes="0.594 0.804 0 -0.195 0.144 0.970",
        fovy="48",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="receptacle_close_camera",
        pos="1.35 -1.45 1.45",
        xyaxes="0.578881 0.815412 0 -0.264107 0.187496 0.946093",
        fovy="48",
    )
    if aircraft_visual_root is not None:
        tanker_position = _BOOM_START_M - _MODEL_TO_WORLD_ROTATION @ (
            _KC135_BOOM_TIP_MODEL_M * AERIAL_AIRCRAFT_VISUAL_SCALE
        )
        tanker = ET.SubElement(
            worldbody,
            "body",
            name="kc135_visual",
            pos=" ".join(map(str, tanker_position)),
            quat="0.5 0.5 -0.5 -0.5",
        )
        ET.SubElement(
            tanker,
            "geom",
            name="kc135_visual_geom",
            type="mesh",
            mesh="kc135_visual_mesh",
            rgba="0.68 0.72 0.78 1",
            density="0",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            tanker,
            "site",
            name="kc135_boom_pivot_visual_anchor",
            pos=" ".join(
                map(
                    str,
                    _KC135_BOOM_PIVOT_MODEL_M * AERIAL_AIRCRAFT_VISUAL_SCALE,
                )
            ),
            size="0.001",
            rgba="0 0 0 0",
        )
        ET.SubElement(
            tanker,
            "site",
            name="kc135_boom_tip_visual_anchor",
            pos=" ".join(
                map(
                    str,
                    _KC135_BOOM_TIP_MODEL_M * AERIAL_AIRCRAFT_VISUAL_SCALE,
                )
            ),
            size="0.001",
            rgba="0 0 0 0",
        )
        for index in range(AERIAL_BOOM_NUM_ELEMENTS):
            segment = ET.SubElement(
                worldbody,
                "body",
                name=f"soft_boom_visual_{index:02d}",
                mocap="true",
                pos="0 0 0",
            )
            ET.SubElement(
                segment,
                "inertial",
                pos="0 0 0",
                mass="0.001",
                diaginertia="0.000001 0.000001 0.000001",
            )
            ET.SubElement(
                segment,
                "geom",
                name=f"soft_boom_visual_geom_{index:02d}",
                type="capsule",
                size=f"{_PRESENTATION_BOOM_RADIUS_M} 0.025",
                rgba="0.92 0.38 0.08 1",
                density="0",
                contype="0",
                conaffinity="0",
            )
    receptacle = ET.SubElement(
        worldbody,
        "body",
        name="receiver_receptacle",
        mocap="true",
        pos=" ".join(map(str, receptacle_position)),
        quat=" ".join(map(str, receptacle_quaternion)),
    )
    ET.SubElement(
        receptacle,
        "inertial",
        pos=f"0 0 {AERIAL_PORT_TUNNEL_DEPTH_M / 2.0}",
        mass="1.0",
        diaginertia="0.02 0.02 0.02",
    )
    ET.SubElement(
        receptacle,
        "site",
        name="receptacle_mouth",
        pos="0 0 0",
        size="0.001",
        rgba="0 0 0 0",
    )
    outer = 0.120
    opening = AERIAL_PORT_HALF_OPENING_M
    wall = (outer - opening) / 2.0
    center = (outer + opening) / 2.0
    half_depth = AERIAL_PORT_TUNNEL_DEPTH_M / 2.0
    _add_receptacle_box(
        receptacle,
        name="receptacle_left",
        position=(-center, 0.0, half_depth),
        size=(wall, outer, half_depth),
    )
    _add_receptacle_box(
        receptacle,
        name="receptacle_right",
        position=(center, 0.0, half_depth),
        size=(wall, outer, half_depth),
    )
    _add_receptacle_box(
        receptacle,
        name="receptacle_bottom",
        position=(0.0, -center, half_depth),
        size=(opening, wall, half_depth),
    )
    _add_receptacle_box(
        receptacle,
        name="receptacle_top",
        position=(0.0, center, half_depth),
        size=(opening, wall, half_depth),
    )
    if aircraft_visual_root is not None:
        receiver_rotation = _RECEPTACLE_BASE_ROTATION.T @ _MODEL_TO_WORLD_ROTATION
        receiver_position = -receiver_rotation @ (
            _F16_REFUEL_ANCHOR_MODEL_M * AERIAL_AIRCRAFT_VISUAL_SCALE
        )
        receiver_aircraft = ET.SubElement(
            receptacle,
            "body",
            name="f16_visual",
            pos=" ".join(map(str, receiver_position)),
            quat=" ".join(map(str, _matrix_to_wxyz(receiver_rotation))),
        )
        ET.SubElement(
            receiver_aircraft,
            "inertial",
            pos="0 0 0",
            mass="0.001",
            diaginertia="0.000001 0.000001 0.000001",
        )
        ET.SubElement(
            receiver_aircraft,
            "geom",
            name="f16_visual_geom",
            type="mesh",
            mesh="f16_visual_mesh",
            rgba="0.35 0.40 0.48 1",
            density="0",
            contype="0",
            conaffinity="0",
        )
        anchor = _F16_REFUEL_ANCHOR_MODEL_M * AERIAL_AIRCRAFT_VISUAL_SCALE
        ET.SubElement(
            receiver_aircraft,
            "site",
            name="f16_refuel_visual_anchor",
            pos=" ".join(map(str, anchor)),
            size="0.001",
            rgba="0 0 0 0",
        )
        port_y = float(anchor[1] + 0.002)
        port_colour = "0.92 0.62 0.08 1"
        ET.SubElement(
            receiver_aircraft,
            "geom",
            name="f16_receptacle_recess_visual",
            type="box",
            pos=f"{anchor[0]} {port_y} {anchor[2]}",
            size="0.060 0.0015 0.036",
            rgba="0.025 0.035 0.045 1",
            density="0",
            contype="0",
            conaffinity="0",
        )
        for name, offset, size in (
            ("fore", (-0.057, 0.0), (0.003, 0.036)),
            ("aft", (0.057, 0.0), (0.003, 0.036)),
            ("left", (0.0, -0.033), (0.057, 0.003)),
            ("right", (0.0, 0.033), (0.057, 0.003)),
        ):
            ET.SubElement(
                receiver_aircraft,
                "geom",
                name=f"f16_receptacle_frame_{name}_visual",
                type="box",
                pos=(
                    f"{anchor[0] + offset[0]} {port_y + 0.001} "
                    f"{anchor[2] + offset[1]}"
                ),
                size=f"{size[0]} 0.0015 {size[1]}",
                rgba=port_colour,
                density="0",
                contype="0",
                conaffinity="0",
            )
    if aircraft_visual_root is None:
        ET.SubElement(
            receptacle,
            "geom",
            name="receiver_panel_visual",
            type="box",
            pos=f"0 0 {AERIAL_PORT_TUNNEL_DEPTH_M + 0.035}",
            size="0.34 0.24 0.025",
            rgba="0.16 0.24 0.36 1",
            density="0",
            contype="0",
            conaffinity="0",
        )
    return _pretty_xml(model)


class ManiSoftAerialBackend:
    """Exact ManiSoft continuum boom coupled to a moving MuJoCo receptacle."""

    def __init__(
        self,
        *,
        case: AerialRefuelCase,
        seed: int,
        enable_renderer: bool,
        image_size: tuple[int, int],
        camera_name: str,
        update_interval: int,
        runtime_root: str | Path,
        asset_root: str | Path | None,
        aircraft_visual_root: str | Path | None,
    ) -> None:
        runtime, _ = configure_manisoft_runtime(runtime_root, asset_root)
        from manisoft.arguments import SoftRobotArgument
        from manisoft.asset import Asset, _Asset
        from manisoft.backend.manisoft_sim import (
            ManiSoftSimBackend,
            directions_to_wxyz,
            read_contact_force,
        )
        from manisoft.muscle import SplineMuscle

        import mujoco

        self.case = case
        self.seed = int(seed)
        self.update_interval = int(update_interval)
        self.camera_name = str(camera_name)
        self._mujoco = mujoco
        self._read_contact_force = read_contact_force
        self._physics_substeps = 0
        self._time_s = 0.0
        self._last_step_peak_contact_force_n = 0.0
        self._episode_peak_contact_force_n = 0.0
        self._last_step_integrity_valid = True

        normal = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        binormal = np.cross(_BOOM_DIRECTION, normal)
        binormal /= max(float(np.linalg.norm(binormal)), 1.0e-12)
        initial_quaternion = np.asarray(
            directions_to_wxyz(
                np.stack((normal, binormal, _BOOM_DIRECTION), axis=0)
            ),
            dtype=np.float64,
        )
        receptacle_position, _, _, _, receptacle_quaternion = case.motion(0.0)
        boom_tip_config = (
            Path(__file__).resolve().parent / "assets" / "boom_tip" / "config.xml"
        ).resolve(strict=True)
        scene_xml = _scene_xml(
            boom_tip_config,
            receptacle_position,
            receptacle_quaternion,
            None
            if aircraft_visual_root is None
            else Path(aircraft_visual_root).resolve(strict=True),
        )

        class AerialBackend(ManiSoftSimBackend):
            def _get_xml_string(
                backend_self,
                gripper,
                object_configs,
                dt,
                enable_gravity,
                enable_friction,
                enable_damping,
            ):
                return scene_xml

        boom_tip_asset = _Asset("BOOM_TIP", boom_tip_config.parent)
        softrobot_config = SoftRobotArgument(
            start=tuple(_BOOM_START_M),
            direction=tuple(_BOOM_DIRECTION),
            normal=tuple(normal),
            length=AERIAL_BOOM_LENGTH_M,
            radius=AERIAL_BOOM_RADIUS_M,
            num_elements=AERIAL_BOOM_NUM_ELEMENTS,
            density=AERIAL_BOOM_DENSITY_KG_M3,
            youngs_modulus=AERIAL_BOOM_YOUNGS_MODULUS_PA,
            poisson_ratio=AERIAL_BOOM_POISSON_RATIO,
            damping_constant=AERIAL_BOOM_DAMPING_CONSTANT,
            asset=Asset.SOFTROBOT,
            size=(-1.0, -0.8),
            gripper=boom_tip_asset,
        )
        self._backend = AerialBackend(
            softrobot_config=softrobot_config,
            object_configs=(),
            dt=AERIAL_PHYSICS_DT_S,
            enable_gravity=False,
            enable_friction=False,
            enable_damping=True,
            fix_robot_end=True,
        )
        robot = self._backend.softrobot_state
        self._muscle = SplineMuscle(
            robot_length=float(np.sum(robot.element_lengths)),
            robot_num_elements=int(robot.num_elements),
            number_of_control_points=6,
            muscle_torque_scale=AERIAL_MUSCLE_TORQUE_SCALE,
        )

        simulator = self._backend._mujoco_simulator
        self._model = simulator.model
        self._data = simulator.data
        self._boom_visual_mocap_ids: list[int] = []
        if aircraft_visual_root is not None:
            for index in range(AERIAL_BOOM_NUM_ELEMENTS):
                body_id = int(
                    mujoco.mj_name2id(
                        self._model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        f"soft_boom_visual_{index:02d}",
                    )
                )
                if body_id < 0:
                    raise RuntimeError("soft boom presentation geometry is incomplete")
                mocap_id = int(self._model.body_mocapid[body_id])
                if mocap_id < 0:
                    raise RuntimeError("soft boom presentation geometry is incomplete")
                self._boom_visual_mocap_ids.append(mocap_id)
        self._boom_body_id = int(
            mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, "boom_tip_body"
            )
        )
        self._receiver_body_id = int(
            mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, "receiver_receptacle"
            )
        )
        self._tip_site_id = int(
            mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_SITE, "boom_nozzle_tip"
            )
        )
        self._receptacle_mouth_site_id = int(
            mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_SITE, "receptacle_mouth"
            )
        )
        self._kc135_boom_tip_visual_site_id = int(
            mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_SITE,
                "kc135_boom_tip_visual_anchor",
            )
        )
        self._f16_refuel_visual_site_id = int(
            mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_SITE,
                "f16_refuel_visual_anchor",
            )
        )
        extension_joint_id = int(
            mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, "boom_extension_joint"
            )
        )
        self._extension_actuator_id = int(
            mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator"
            )
        )
        self._mocap_id = int(self._model.body_mocapid[self._receiver_body_id])
        if min(
            self._boom_body_id,
            self._receiver_body_id,
            self._tip_site_id,
            self._receptacle_mouth_site_id,
            extension_joint_id,
            self._extension_actuator_id,
            self._mocap_id,
        ) < 0:
            raise RuntimeError("aerial MuJoCo scene is missing required bodies/sites")
        self._extension_qpos_address = int(self._model.jnt_qposadr[extension_joint_id])
        self._extension_dof_address = int(self._model.jnt_dofadr[extension_joint_id])
        self._extension_commanded_m = float(
            self._data.qpos[self._extension_qpos_address]
        )
        self._set_receptacle(0.0)
        self._update_boom_visuals()
        if aircraft_visual_root is not None:
            if min(
                self._kc135_boom_tip_visual_site_id,
                self._f16_refuel_visual_site_id,
            ) < 0:
                raise RuntimeError("aircraft visual anchor sites are missing")
            kc135_error_m = float(
                np.linalg.norm(
                    self._data.site_xpos[self._kc135_boom_tip_visual_site_id]
                    - _BOOM_START_M
                )
            )
            f16_error_m = float(
                np.linalg.norm(
                    self._data.site_xpos[self._f16_refuel_visual_site_id]
                    - self._data.site_xpos[self._receptacle_mouth_site_id]
                )
            )
            if max(kc135_error_m, f16_error_m) > 1.0e-6:
                raise RuntimeError(
                    "aircraft visual anchors are not bound to the task: "
                    f"kc135_error_m={kc135_error_m:.9g}, "
                    f"f16_error_m={f16_error_m:.9g}"
                )
        self._renderer = (
            mujoco.Renderer(
                self._model,
                height=int(image_size[0]),
                width=int(image_size[1]),
            )
            if enable_renderer
            else None
        )

    def _set_receptacle(self, time_s: float) -> None:
        position, _, _, _, quaternion = self.case.motion(time_s)
        self._data.mocap_pos[self._mocap_id] = position
        self._data.mocap_quat[self._mocap_id] = quaternion
        self._mujoco.mj_forward(self._model, self._data)

    def _update_boom_visuals(self) -> None:
        if not self._boom_visual_mocap_ids:
            return
        positions = np.asarray(
            self._backend.softrobot_state.element_positions,
            dtype=np.float64,
        )
        segment_count = min(len(positions) - 1, len(self._boom_visual_mocap_ids))
        for index, mocap_id in enumerate(self._boom_visual_mocap_ids):
            if index >= segment_count:
                self._data.mocap_pos[mocap_id] = (0.0, 0.0, -10.0)
                self._data.mocap_quat[mocap_id] = (1.0, 0.0, 0.0, 0.0)
                continue
            start = positions[index]
            end = positions[index + 1]
            direction = end - start
            direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
            quaternion = np.asarray(
                (1.0 + direction[2], -direction[1], direction[0], 0.0),
                dtype=np.float64,
            )
            if float(np.linalg.norm(quaternion)) < 1.0e-8:
                quaternion = np.asarray((0.0, 1.0, 0.0, 0.0), dtype=np.float64)
            else:
                quaternion /= float(np.linalg.norm(quaternion))
            self._data.mocap_pos[mocap_id] = 0.5 * (start + end)
            self._data.mocap_quat[mocap_id] = quaternion
        self._mujoco.mj_forward(self._model, self._data)

    def _contact_force_n(self) -> float:
        total = 0.0
        for contact_index in range(int(self._data.ncon)):
            contact = self._data.contact[contact_index]
            names = {
                self._mujoco.mj_id2name(
                    self._model,
                    self._mujoco.mjtObj.mjOBJ_GEOM,
                    int(contact.geom1),
                )
                or "",
                self._mujoco.mj_id2name(
                    self._model,
                    self._mujoco.mjtObj.mjOBJ_GEOM,
                    int(contact.geom2),
                )
                or "",
            }
            if not any(name.startswith("receptacle_") for name in names):
                continue
            record = self._read_contact_force(
                self._model, self._data, contact_index
            )
            top_ids = {
                int(record["top_body1_id"]),
                int(record["top_body2_id"]),
            }
            if self._boom_body_id in top_ids and self._receiver_body_id in top_ids:
                total += float(record["force_magnitude"])
        return float(total)

    def _continuum_integrity_valid(self) -> bool:
        rod = self._backend._softrobot
        lengths = np.asarray(rod.lengths, dtype=np.float64)
        finite = bool(
            np.all(np.isfinite(lengths))
            and np.all(np.isfinite(rod.position_collection))
            and np.all(np.isfinite(rod.director_collection))
            and np.all(np.isfinite(rod.velocity_collection))
            and np.all(np.isfinite(rod.omega_collection))
        )
        return bool(
            finite
            and lengths.shape == (AERIAL_BOOM_NUM_ELEMENTS,)
            and float(np.min(lengths)) >= 0.040
            and float(np.max(lengths)) <= 0.060
            and 0.90 <= float(np.sum(lengths)) <= 1.10
        )

    def _tip_pose(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        position = np.asarray(
            self._data.site_xpos[self._tip_site_id], dtype=np.float64
        ).copy()
        rotation = np.asarray(
            self._data.xmat[self._boom_body_id], dtype=np.float64
        ).reshape(3, 3)
        axis = rotation[:, 2].copy()
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        spatial_velocity = np.zeros(6, dtype=np.float64)
        self._mujoco.mj_objectVelocity(
            self._model,
            self._data,
            self._mujoco.mjtObj.mjOBJ_SITE,
            self._tip_site_id,
            spatial_velocity,
            0,
        )
        return position, axis, spatial_velocity[3:].copy()

    def snapshot(self) -> AerialSnapshot:
        robot = self._backend.softrobot_state
        positions = np.asarray(robot.element_positions, dtype=np.float64)
        velocities = np.asarray(robot.element_velocities, dtype=np.float64)
        tip, tip_axis, tip_velocity = self._tip_pose()
        (
            receptacle_position,
            receptacle_axis,
            receptacle_velocity,
            receptacle_angular_velocity,
            _,
        ) = self.case.motion(self._time_s)
        relative = tip - receptacle_position
        axial = float(np.dot(relative, receptacle_axis))
        lateral_vector = relative - axial * receptacle_axis
        lateral_error = float(np.linalg.norm(lateral_vector))
        alignment = float(np.clip(np.dot(tip_axis, receptacle_axis), -1.0, 1.0))
        orientation_error = float(math.acos(alignment))
        insertion_depth = (
            max(axial, 0.0)
            if lateral_error <= AERIAL_PORT_HALF_OPENING_M + 0.015
            else 0.0
        )
        contact_force = self._contact_force_n()
        return AerialSnapshot(
            boom_positions_m=positions,
            boom_velocities_m_s=velocities,
            tip_position_m=tip,
            tip_velocity_m_s=tip_velocity,
            tip_axis=tip_axis,
            receptacle_position_m=receptacle_position,
            receptacle_axis=receptacle_axis,
            receptacle_linear_velocity_m_s=receptacle_velocity,
            receptacle_angular_velocity_rad_s=receptacle_angular_velocity,
            insertion_depth_m=insertion_depth,
            lateral_error_m=lateral_error,
            orientation_error_rad=orientation_error,
            contact_force_n=contact_force,
            step_peak_contact_force_n=max(
                contact_force, self._last_step_peak_contact_force_n
            ),
            episode_peak_contact_force_n=max(
                contact_force, self._episode_peak_contact_force_n
            ),
            boom_extension_m=float(self._data.qpos[self._extension_qpos_address]),
            boom_extension_velocity_m_s=float(
                self._data.qvel[self._extension_dof_address]
            ),
            continuum_integrity_valid=bool(
                self._last_step_integrity_valid
                and self._continuum_integrity_valid()
            ),
            physics_time_s=self._time_s,
        )

    def advance(self, action: npt.NDArray[np.float32]) -> None:
        action_array = np.asarray(action, dtype=np.float64).reshape(AERIAL_ACTION_DIM)
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action contains a non-finite value")
        action_array = np.clip(action_array, -1.0, 1.0)
        activation = action_array[:18] * AERIAL_MUSCLE_ACTIVATION_LIMIT
        extension_rate_command_m_s = (
            action_array[18] * AERIAL_BOOM_EXTENSION_RATE_LIMIT_M_S
        )
        robot = self._backend.softrobot_state
        torque = self._muscle.activate(
            activation.reshape(6, 3),
            np.asarray(robot.element_lengths, dtype=np.float64),
        )
        self._last_step_peak_contact_force_n = 0.0
        self._last_step_integrity_valid = True
        for _ in range(self.update_interval):
            next_time = (self._physics_substeps + 1) * AERIAL_PHYSICS_DT_S
            self._set_receptacle(next_time)
            previous_extension_m = self._extension_commanded_m
            self._extension_commanded_m = float(
                np.clip(
                    self._extension_commanded_m
                    + extension_rate_command_m_s * AERIAL_PHYSICS_DT_S,
                    0.0,
                    AERIAL_BOOM_EXTENSION_COMMAND_MAX_M,
                )
            )
            extension_velocity_m_s = (
                self._extension_commanded_m - previous_extension_m
            ) / AERIAL_PHYSICS_DT_S
            self._data.qpos[
                self._extension_qpos_address
            ] = self._extension_commanded_m
            self._data.qvel[
                self._extension_dof_address
            ] = extension_velocity_m_s
            self._data.ctrl[
                self._extension_actuator_id
            ] = self._extension_commanded_m
            self._mujoco.mj_forward(self._model, self._data)
            self._backend.step([tuple(row) for row in np.asarray(torque)])
            self._data.qpos[
                self._extension_qpos_address
            ] = self._extension_commanded_m
            self._data.qvel[
                self._extension_dof_address
            ] = extension_velocity_m_s
            self._mujoco.mj_forward(self._model, self._data)
            self._physics_substeps += 1
            self._time_s = next_time
            contact_force = self._contact_force_n()
            self._last_step_peak_contact_force_n = max(
                self._last_step_peak_contact_force_n,
                contact_force,
            )
            self._episode_peak_contact_force_n = max(
                self._episode_peak_contact_force_n,
                contact_force,
            )
            self._last_step_integrity_valid = bool(
                self._last_step_integrity_valid
                and self._continuum_integrity_valid()
            )
        self._update_boom_visuals()

    def render_rgbd(
        self,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
        if self._renderer is None:
            raise RuntimeError("renderer was not enabled for this aerial backend")
        self._renderer.update_scene(self._data, camera=self.camera_name)
        self._renderer.disable_depth_rendering()
        rgb = np.asarray(self._renderer.render(), dtype=np.uint8).copy()
        self._renderer.enable_depth_rendering()
        depth = np.asarray(self._renderer.render(), dtype=np.float32).copy()
        self._renderer.disable_depth_rendering()
        return rgb, depth[..., None]

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
