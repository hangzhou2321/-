from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import numpy.typing as npt

from .aerial import (
    AERIAL_PORT_HALF_OPENING_M,
    AERIAL_PORT_TUNNEL_DEPTH_M,
    RIGID_AERIAL_ACTION_DIM,
    RIGID_AERIAL_PHYSICS_DT_S,
    RIGID_BOOM_EXTENSION_MAX_M,
    RIGID_BOOM_EXTENSION_RATE_LIMIT_M_S,
    RIGID_BOOM_INITIAL_EXTENSION_M,
    RIGID_BOOM_OUTER_LENGTH_M,
    RIGID_BOOM_TELESCOPIC_LENGTH_M,
    AerialRefuelCase,
    AerialSnapshot,
)


_AIR_DENSITY_KG_M3 = 0.459
_RELATIVE_AIRSPEED_M_S = 200.0
_RUDDEVATOR_AREA_M2 = 0.35
_RUDDEVATOR_MOMENT_ARM_M = 4.5
_LIFT_CURVE_PER_RAD = 0.80
_RUDDEVATOR_LIMIT_RAD = math.radians(1.5)
_EXTENSION_SERVO_GAIN_N_S_M = 6_000.0
_EXTENSION_FORCE_LIMIT_N = 2_000.0


def _pretty_xml(root: ET.Element) -> str:
    parsed = minidom.parseString(ET.tostring(root, encoding="utf-8"))
    return "\n".join(
        line for line in parsed.toprettyxml(indent="  ").splitlines() if line.strip()
    )


def _vec(value: npt.ArrayLike) -> str:
    return " ".join(f"{float(item):.12g}" for item in np.asarray(value).reshape(-1))


def _add_port_wall(
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
        pos=_vec(position),
        size=_vec(size),
        rgba="0.82 0.66 0.16 1",
        density="0",
        friction="1.5 0.05 0.02",
        solref="0.04 1",
        solimp="0.95 0.995 0.001",
        contype="1",
        conaffinity="1",
    )


def _scene_xml(
    receptacle_position: npt.NDArray[np.float64],
    receptacle_quaternion: npt.NDArray[np.float64],
) -> str:
    model = ET.Element("mujoco", model="manisoftrl_rigid_flying_boom")
    ET.SubElement(model, "compiler", angle="degree", autolimits="true")
    ET.SubElement(
        model,
        "option",
        timestep=str(RIGID_AERIAL_PHYSICS_DT_S),
        gravity="0 0 0",
        integrator="implicitfast",
    )
    visual = ET.SubElement(model, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    ET.SubElement(visual, "map", znear="0.01", zfar="30")
    ET.SubElement(visual, "rgba", haze="0.36 0.47 0.62 1")
    default = ET.SubElement(model, "default")
    ET.SubElement(
        default,
        "geom",
        conaffinity="1",
        condim="3",
        friction="1.2 0.04 0.02",
    )
    world = ET.SubElement(model, "worldbody")
    ET.SubElement(
        world,
        "light",
        pos="2 2 5",
        dir="-0.2 -0.4 -1",
        directional="true",
        diffuse="0.9 0.9 0.86",
    )
    ET.SubElement(
        world,
        "light",
        pos="-3 -4 1",
        dir="0.4 0.6 -0.2",
        directional="true",
        diffuse="0.35 0.45 0.62",
    )
    ET.SubElement(
        world,
        "geom",
        name="sky_reference",
        type="plane",
        pos="0 0 -5",
        size="30 30 0.1",
        rgba="0.16 0.28 0.43 1",
        contype="0",
        conaffinity="0",
    )

    boom = ET.SubElement(world, "body", name="boom_outer", pos="0 0 0")
    ET.SubElement(
        boom,
        "joint",
        name="boom_yaw",
        type="hinge",
        axis="0 0 1",
        range="-25 25",
        damping="180",
        stiffness="900",
    )
    ET.SubElement(
        boom,
        "joint",
        name="boom_pitch",
        type="hinge",
        axis="1 0 0",
        range="-25 25",
        damping="180",
        stiffness="900",
    )
    ET.SubElement(
        boom,
        "geom",
        name="outer_boom_visual",
        type="capsule",
        fromto=f"0 0 0 0 {-RIGID_BOOM_OUTER_LENGTH_M:.12g} 0",
        size="0.12",
        mass="480",
        rgba="0.70 0.72 0.74 1",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        boom,
        "geom",
        name="left_ruddevator_visual",
        type="box",
        pos=f"-0.55 {-RIGID_BOOM_OUTER_LENGTH_M * 0.78:.12g} 0",
        size="0.48 0.55 0.035",
        rgba="0.72 0.18 0.12 1",
        mass="18",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        boom,
        "geom",
        name="right_ruddevator_visual",
        type="box",
        pos=f"0.55 {-RIGID_BOOM_OUTER_LENGTH_M * 0.78:.12g} 0",
        size="0.48 0.55 0.035",
        rgba="0.12 0.30 0.72 1",
        mass="18",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(boom, "site", name="boom_pivot", pos="0 0 0", size="0.001")
    ET.SubElement(
        boom,
        "site",
        name="outer_boom_end",
        pos=f"0 {-RIGID_BOOM_OUTER_LENGTH_M:.12g} 0",
        size="0.001",
    )

    tube_origin_y = -(RIGID_BOOM_OUTER_LENGTH_M - RIGID_BOOM_TELESCOPIC_LENGTH_M)
    tube = ET.SubElement(
        boom,
        "body",
        name="telescopic_tube",
        pos=f"0 {tube_origin_y:.12g} 0",
    )
    ET.SubElement(
        tube,
        "joint",
        name="boom_extension",
        type="slide",
        axis="0 -1 0",
        range=f"0 {RIGID_BOOM_EXTENSION_MAX_M:.12g}",
        damping="2500",
    )
    ET.SubElement(
        tube,
        "geom",
        name="telescopic_tube_visual",
        type="capsule",
        fromto=f"0 0 0 0 {-RIGID_BOOM_TELESCOPIC_LENGTH_M:.12g} 0",
        size="0.075",
        mass="75",
        rgba="0.42 0.45 0.48 1",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        tube,
        "geom",
        name="boom_nozzle_contact",
        type="capsule",
        fromto=(
            f"0 {-RIGID_BOOM_TELESCOPIC_LENGTH_M + 0.12:.12g} 0 "
            f"0 {-RIGID_BOOM_TELESCOPIC_LENGTH_M:.12g} 0"
        ),
        size="0.026",
        mass="4",
        rgba="0.12 0.12 0.13 1",
        friction="1.0 0.03 0.01",
        solref="0.04 1",
        solimp="0.95 0.995 0.001",
        contype="1",
        conaffinity="1",
    )
    ET.SubElement(
        tube,
        "site",
        name="boom_nozzle_tip",
        pos=f"0 {-RIGID_BOOM_TELESCOPIC_LENGTH_M:.12g} 0",
        size="0.001",
        rgba="0 0 0 0",
    )
    ET.SubElement(
        tube,
        "camera",
        name="boom_tip_camera",
        pos=f"0 {-RIGID_BOOM_TELESCOPIC_LENGTH_M + 0.18:.12g} 0.075",
        xyaxes="1 0 0 0 0 -1",
        fovy="58",
    )
    ET.SubElement(
        world,
        "camera",
        name="scene_camera",
        pos="7 2 5",
        xyaxes="0.82 -0.57 0 0.22 0.32 0.92",
        fovy="48",
    )

    receiver = ET.SubElement(
        world,
        "body",
        name="receiver_receptacle",
        mocap="true",
        pos=_vec(receptacle_position),
        quat=_vec(receptacle_quaternion),
    )
    ET.SubElement(
        receiver,
        "inertial",
        pos=f"0 0 {AERIAL_PORT_TUNNEL_DEPTH_M / 2.0:.12g}",
        mass="1",
        diaginertia="0.02 0.02 0.02",
    )
    ET.SubElement(
        receiver,
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
    _add_port_wall(
        receiver,
        name="receptacle_left",
        position=(-center, 0.0, half_depth),
        size=(wall, outer, half_depth),
    )
    _add_port_wall(
        receiver,
        name="receptacle_right",
        position=(center, 0.0, half_depth),
        size=(wall, outer, half_depth),
    )
    _add_port_wall(
        receiver,
        name="receptacle_bottom",
        position=(0.0, -center, half_depth),
        size=(opening, wall, half_depth),
    )
    _add_port_wall(
        receiver,
        name="receptacle_top",
        position=(0.0, center, half_depth),
        size=(opening, wall, half_depth),
    )
    ET.SubElement(
        receiver,
        "geom",
        name="receiver_panel_visual",
        type="box",
        pos=f"0 0 {AERIAL_PORT_TUNNEL_DEPTH_M + 0.035:.12g}",
        size="0.50 0.34 0.025",
        rgba="0.16 0.24 0.36 1",
        density="0",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        receiver,
        "geom",
        name="receiver_fuselage_context",
        type="ellipsoid",
        pos="0 0 1.0",
        size="1.15 0.85 1.8",
        rgba="0.22 0.31 0.44 1",
        density="0",
        contype="0",
        conaffinity="0",
    )
    return _pretty_xml(model)


class RigidFlyingBoomBackend:
    """MuJoCo rigid boom driven by ruddevator moments and a force-limited tube."""

    def __init__(
        self,
        *,
        case: AerialRefuelCase,
        seed: int,
        enable_renderer: bool,
        image_size: tuple[int, int],
        camera_name: str,
        update_interval: int,
    ) -> None:
        import mujoco

        self.case = case
        self.seed = int(seed)
        self.update_interval = int(update_interval)
        self.camera_name = str(camera_name)
        self._mujoco = mujoco
        self._time_s = 0.0
        self._last_step_peak_contact_force_n = 0.0
        self._episode_peak_contact_force_n = 0.0
        receptacle_position, _, _, _, receptacle_quaternion = case.motion(0.0)
        self._model = mujoco.MjModel.from_xml_string(
            _scene_xml(receptacle_position, receptacle_quaternion)
        )
        self._data = mujoco.MjData(self._model)

        self._receiver_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "receiver_receptacle")
        self._mocap_id = int(self._model.body_mocapid[self._receiver_body_id])
        self._pivot_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "boom_pivot")
        self._outer_end_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "outer_boom_end")
        self._tip_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "boom_nozzle_tip")
        self._tube_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "telescopic_tube")
        self._yaw_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "boom_yaw")
        self._pitch_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "boom_pitch")
        self._extension_joint_id = self._id(
            mujoco.mjtObj.mjOBJ_JOINT, "boom_extension"
        )
        self._yaw_qpos = int(self._model.jnt_qposadr[self._yaw_joint_id])
        self._pitch_qpos = int(self._model.jnt_qposadr[self._pitch_joint_id])
        self._extension_qpos = int(self._model.jnt_qposadr[self._extension_joint_id])
        self._yaw_dof = int(self._model.jnt_dofadr[self._yaw_joint_id])
        self._pitch_dof = int(self._model.jnt_dofadr[self._pitch_joint_id])
        self._extension_dof = int(self._model.jnt_dofadr[self._extension_joint_id])
        if self._mocap_id < 0:
            raise RuntimeError("receiver_receptacle is not a MuJoCo mocap body")

        self._data.qpos[self._extension_qpos] = RIGID_BOOM_INITIAL_EXTENSION_M
        self._set_receptacle(0.0)
        mujoco.mj_forward(self._model, self._data)
        self._renderer = (
            mujoco.Renderer(
                self._model,
                height=int(image_size[0]),
                width=int(image_size[1]),
            )
            if enable_renderer
            else None
        )

    def _id(self, object_type: object, name: str) -> int:
        identifier = int(self._mujoco.mj_name2id(self._model, object_type, name))
        if identifier < 0:
            raise RuntimeError(f"rigid aerial scene is missing {name!r}")
        return identifier

    def _set_receptacle(self, time_s: float) -> None:
        position, _, _, _, quaternion = self.case.motion(time_s)
        self._data.mocap_pos[self._mocap_id] = position
        self._data.mocap_quat[self._mocap_id] = quaternion

    def _site_velocity(self, site_id: int) -> npt.NDArray[np.float64]:
        spatial = np.zeros(6, dtype=np.float64)
        self._mujoco.mj_objectVelocity(
            self._model,
            self._data,
            self._mujoco.mjtObj.mjOBJ_SITE,
            site_id,
            spatial,
            0,
        )
        return spatial[3:].copy()

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
            if "boom_nozzle_contact" not in names or not any(
                name.startswith("receptacle_") for name in names
            ):
                continue
            force = np.zeros(6, dtype=np.float64)
            self._mujoco.mj_contactForce(
                self._model, self._data, contact_index, force
            )
            total += float(np.linalg.norm(force[:3]))
        return total

    def _integrity_valid(self) -> bool:
        qpos = np.asarray(self._data.qpos, dtype=np.float64)
        qvel = np.asarray(self._data.qvel, dtype=np.float64)
        extension = float(self._data.qpos[self._extension_qpos])
        angle_limit = math.radians(25.0) + 1.0e-3
        return bool(
            np.all(np.isfinite(qpos))
            and np.all(np.isfinite(qvel))
            and -1.0e-4 <= extension <= RIGID_BOOM_EXTENSION_MAX_M + 1.0e-4
            and abs(float(self._data.qpos[self._yaw_qpos])) <= angle_limit
            and abs(float(self._data.qpos[self._pitch_qpos])) <= angle_limit
        )

    def snapshot(self) -> AerialSnapshot:
        pivot = np.asarray(self._data.site_xpos[self._pivot_site_id], dtype=np.float64)
        outer_end = np.asarray(
            self._data.site_xpos[self._outer_end_site_id], dtype=np.float64
        )
        tip = np.asarray(self._data.site_xpos[self._tip_site_id], dtype=np.float64)
        pivot_velocity = self._site_velocity(self._pivot_site_id)
        outer_velocity = self._site_velocity(self._outer_end_site_id)
        tip_velocity = self._site_velocity(self._tip_site_id)
        outer_positions = np.linspace(pivot, outer_end, 15, dtype=np.float64)
        tube_positions = np.linspace(outer_end, tip, 7, dtype=np.float64)[1:]
        outer_velocities = np.linspace(
            pivot_velocity, outer_velocity, 15, dtype=np.float64
        )
        tube_velocities = np.linspace(
            outer_velocity, tip_velocity, 7, dtype=np.float64
        )[1:]
        positions = np.concatenate((outer_positions, tube_positions), axis=0)
        velocities = np.concatenate((outer_velocities, tube_velocities), axis=0)

        tube_rotation = np.asarray(
            self._data.xmat[self._tube_body_id], dtype=np.float64
        ).reshape(3, 3)
        tip_axis = tube_rotation @ np.asarray((0.0, -1.0, 0.0))
        tip_axis /= max(float(np.linalg.norm(tip_axis)), 1.0e-12)
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
            tip_position_m=tip.copy(),
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
            boom_extension_m=float(self._data.qpos[self._extension_qpos]),
            boom_extension_velocity_m_s=float(
                self._data.qvel[self._extension_dof]
            ),
            continuum_integrity_valid=self._integrity_valid(),
            physics_time_s=self._time_s,
        )

    def advance(self, action: npt.NDArray[np.float32]) -> None:
        action_array = np.asarray(action, dtype=np.float64).reshape(
            RIGID_AERIAL_ACTION_DIM
        )
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action contains a non-finite value")
        left, right, extension = np.clip(action_array, -1.0, 1.0)
        dynamic_pressure_pa = (
            0.5 * _AIR_DENSITY_KG_M3 * _RELATIVE_AIRSPEED_M_S**2
        )
        torque_per_rad = (
            dynamic_pressure_pa
            * _RUDDEVATOR_AREA_M2
            * _RUDDEVATOR_MOMENT_ARM_M
            * _LIFT_CURVE_PER_RAD
        )
        left_rad = float(left) * _RUDDEVATOR_LIMIT_RAD
        right_rad = float(right) * _RUDDEVATOR_LIMIT_RAD
        pitch_torque_nm = torque_per_rad * 0.5 * (left_rad + right_rad)
        yaw_torque_nm = torque_per_rad * 0.5 * (left_rad - right_rad)
        desired_extension_velocity = (
            float(extension) * RIGID_BOOM_EXTENSION_RATE_LIMIT_M_S
        )

        self._last_step_peak_contact_force_n = 0.0
        for _ in range(self.update_interval):
            next_time = self._time_s + RIGID_AERIAL_PHYSICS_DT_S
            self._set_receptacle(next_time)
            self._data.qfrc_applied[:] = 0.0
            self._data.qfrc_applied[self._pitch_dof] = pitch_torque_nm
            self._data.qfrc_applied[self._yaw_dof] = yaw_torque_nm
            extension_force_n = _EXTENSION_SERVO_GAIN_N_S_M * (
                desired_extension_velocity
                - float(self._data.qvel[self._extension_dof])
            )
            self._data.qfrc_applied[self._extension_dof] = float(
                np.clip(
                    extension_force_n,
                    -_EXTENSION_FORCE_LIMIT_N,
                    _EXTENSION_FORCE_LIMIT_N,
                )
            )
            self._mujoco.mj_step(self._model, self._data)
            self._time_s = next_time
            contact_force = self._contact_force_n()
            self._last_step_peak_contact_force_n = max(
                self._last_step_peak_contact_force_n, contact_force
            )
            self._episode_peak_contact_force_n = max(
                self._episode_peak_contact_force_n, contact_force
            )

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
