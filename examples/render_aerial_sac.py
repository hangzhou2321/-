from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from manisoftrl import AerialRefuelEnv, AerialSynergyActionWrapper, make_aerial_cases


_CANVAS_SIZE = (1280, 720)
_RGB_BOX = (44, 108, 652, 716)


@lru_cache(maxsize=None)
def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _depth_preview(
    observation: dict[str, np.ndarray], *, maximum_m: float = 0.40
) -> Image.Image:
    depth = np.asarray(observation["depth"][..., 0], dtype=np.float32)
    clipped = np.clip(depth, 0.0, maximum_m) / maximum_m
    near = np.asarray((246, 196, 74), dtype=np.float32)
    far = np.asarray((24, 49, 78), dtype=np.float32)
    colour = near + clipped[..., None] * (far - near)
    return Image.fromarray(np.asarray(np.clip(colour, 0, 255), dtype=np.uint8))


def _raw_preview(observation: dict[str, np.ndarray]) -> Image.Image:
    rgb = observation["rgb"]
    depth = observation["depth"][..., 0]
    minimum = float(np.min(depth))
    maximum = float(np.max(depth))
    scale = max(maximum - minimum, 1.0e-6)
    gray = np.asarray(255.0 * (1.0 - (depth - minimum) / scale), dtype=np.uint8)
    combined = np.concatenate((rgb, np.repeat(gray[..., None], 3, axis=2)), axis=1)
    return Image.fromarray(combined)


def _status(info: dict[str, Any]) -> tuple[str, tuple[int, int, int]]:
    if info["official_success"]:
        return "SUCCESS", (62, 211, 137)
    if info["termination_reason"] not in {None, "time_limit"}:
        return "FAILED", (244, 92, 92)
    if float(info["insertion_depth_m"]) > 0.0:
        return "INSERTING", (246, 196, 74)
    return "APPROACH", (80, 174, 255)


def _presentation_rgb(
    observation: dict[str, np.ndarray], camera_name: str
) -> np.ndarray:
    rgb = np.asarray(observation["rgb"], dtype=np.uint8).copy()
    if camera_name == "boom_tip_camera":
        return rgb
    depth = np.asarray(observation["depth"][..., 0], dtype=np.float32)
    background = (depth >= float(np.max(depth)) - 1.0e-4) & (
        np.max(rgb, axis=2) <= 3
    )
    height, width = rgb.shape[:2]
    top = np.asarray((38, 92, 158), dtype=np.float32)
    bottom = np.asarray((174, 207, 232), dtype=np.float32)
    blend = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    sky = top + blend * (bottom - top)
    sky = np.broadcast_to(sky, (height, width, 3)).astype(np.uint8)
    rgb[background] = sky[background]
    return rgb


def _bar(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    label: str,
    value: float,
    limit: float,
    unit: str,
    lower_is_better: bool,
) -> None:
    x0, x1 = 722, 1218
    fraction = float(np.clip(value / max(limit, 1.0e-9), 0.0, 1.0))
    safe = value <= limit if lower_is_better else value >= limit
    colour = (62, 211, 137) if safe else (246, 196, 74)
    draw.text((x0, y), label, font=_font(22), fill=(216, 226, 238))
    text = f"{value:.3f} {unit} / {limit:.3f} {unit}"
    right = draw.textbbox((0, 0), text, font=_font(20))[2]
    draw.text((x1 - right, y + 2), text, font=_font(20), fill=(160, 178, 198))
    top = y + 34
    draw.rounded_rectangle((x0, top, x1, top + 14), radius=7, fill=(38, 54, 72))
    draw.rounded_rectangle(
        (x0, top, x0 + max(8, int((x1 - x0) * fraction)), top + 14),
        radius=7,
        fill=colour,
    )


def _demo_frame(
    observation: dict[str, np.ndarray],
    info: dict[str, Any],
    *,
    episode_return: float,
    camera_name: str,
) -> Image.Image:
    canvas = Image.new("RGB", _CANVAS_SIZE, (9, 18, 31))
    draw = ImageDraw.Draw(canvas)
    rgb = Image.fromarray(_presentation_rgb(observation, camera_name)).resize(
        (608, 608), Image.Resampling.LANCZOS
    )
    canvas.paste(rgb, (_RGB_BOX[0], _RGB_BOX[1]))
    draw.rounded_rectangle(_RGB_BOX, radius=8, outline=(93, 121, 151), width=2)

    if camera_name == "boom_tip_camera":
        cx = (_RGB_BOX[0] + _RGB_BOX[2]) // 2
        cy = (_RGB_BOX[1] + _RGB_BOX[3]) // 2
        draw.ellipse((cx - 28, cy - 28, cx + 28, cy + 28), outline=(62, 211, 137), width=3)
        draw.line((cx - 48, cy, cx - 14, cy), fill=(62, 211, 137), width=3)
        draw.line((cx + 14, cy, cx + 48, cy), fill=(62, 211, 137), width=3)
        draw.line((cx, cy - 48, cx, cy - 14), fill=(62, 211, 137), width=3)
        draw.line((cx, cy + 14, cx, cy + 48), fill=(62, 211, 137), width=3)

    draw.text((44, 30), "ManiSoft-RL", font=_font(36), fill=(242, 247, 252))
    draw.text(
        (274, 39),
        "FLYING-BOOM TERMINAL DOCKING",
        font=_font(23),
        fill=(142, 164, 188),
    )
    draw.text(
        (44, 82),
        {
            "boom_tip_camera": "TARGET FRAME: RECEIVER RECEPTACLE MOUTH",
            "receptacle_close_camera": "RECEPTACLE CLOSE VIEW: F-16 / RGB-D",
        }.get(camera_name, "FAR VIEW: KC-135 / HARD BOOM / SOFT TIP / F-16"),
        font=_font(17),
        fill=(142, 164, 188),
    )
    status, status_colour = _status(info)
    status_width = draw.textbbox((0, 0), status, font=_font(28))[2]
    draw.rounded_rectangle((1192 - status_width, 28, 1236, 72), radius=10, fill=status_colour)
    draw.text((1206 - status_width, 33), status, font=_font(28), fill=(9, 18, 31))

    depth_maximum_m = 0.40 if camera_name == "boom_tip_camera" else 5.0
    depth = _depth_preview(observation, maximum_m=depth_maximum_m).resize(
        (210, 210), Image.Resampling.LANCZOS
    )
    canvas.paste(depth, (722, 108))
    draw.rounded_rectangle((722, 108, 932, 318), radius=8, outline=(93, 121, 151), width=2)
    draw.text(
        (742, 330),
        f"METRIC DEPTH  0-{depth_maximum_m:.2f} m",
        font=_font(17),
        fill=(142, 164, 188),
    )

    draw.text((962, 112), f"case  {info['case_id']}", font=_font(22), fill=(216, 226, 238))
    draw.text((962, 148), f"seed  {info['episode_seed']}", font=_font(22), fill=(216, 226, 238))
    draw.text((962, 184), f"step  {info['step_count']}", font=_font(22), fill=(216, 226, 238))
    draw.text((962, 220), f"time  {float(info['physics_time_s']):.3f} s", font=_font(22), fill=(216, 226, 238))
    draw.text((962, 256), f"return  {episode_return:.2f}", font=_font(22), fill=(216, 226, 238))
    draw.text((962, 292), "SAC: 82D state -> 4D synergy", font=_font(17), fill=(142, 164, 188))

    _bar(
        draw,
        y=380,
        label="Insertion depth",
        value=float(info["insertion_depth_m"]),
        limit=0.050,
        unit="m",
        lower_is_better=False,
    )
    _bar(
        draw,
        y=458,
        label="Lateral error",
        value=float(info["lateral_error_m"]),
        limit=0.035,
        unit="m",
        lower_is_better=True,
    )
    _bar(
        draw,
        y=536,
        label="Orientation error",
        value=math.degrees(float(info["orientation_error_rad"])),
        limit=12.0,
        unit="deg",
        lower_is_better=True,
    )
    _bar(
        draw,
        y=614,
        label="Peak contact force",
        value=float(info["contact"]["episode_peak_contact_force_n"]),
        limit=float(info["contact"]["safe_limit_n"]),
        unit="N",
        lower_is_better=True,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a frozen state SAC policy with synchronized RGB-D"
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--aircraft-visual-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="0")
    parser.add_argument("--case-seed", type=int, default=20260808)
    parser.add_argument("--episode-seed", type=int, default=80720000)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--camera", default="boom_tip_camera")
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        from stable_baselines3 import SAC
    except ImportError as error:
        raise SystemExit("install the baseline extra: pip install -e .[baseline]") from error

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = AerialRefuelEnv(
        cases=make_aerial_cases(8, seed=args.case_seed),
        default_case=args.case,
        runtime_root=args.runtime_root,
        asset_root=args.asset_root,
        aircraft_visual_root=args.aircraft_visual_root,
        horizon=args.horizon,
        observation_mode="hybrid_rgbd",
        image_size=(args.image_size, args.image_size),
        camera_name=args.camera,
    )
    env = AerialSynergyActionWrapper(base)
    model = SAC.load(args.checkpoint, device=args.device)
    raw_frames: list[Image.Image] = []
    demo_frames: list[Image.Image] = []
    try:
        observation, info = env.reset(
            seed=args.episode_seed,
            options={"case": args.case},
        )
        raw_frames.append(_raw_preview(observation))
        demo_frames.append(
            _demo_frame(
                observation,
                info,
                episode_return=0.0,
                camera_name=args.camera,
            )
        )
        terminated = False
        truncated = False
        episode_return = 0.0
        while not terminated and not truncated:
            action, _ = model.predict(observation["state"], deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            if info["step_count"] % args.frame_stride == 0:
                raw_frames.append(_raw_preview(observation))
                demo_frames.append(
                    _demo_frame(
                        observation,
                        info,
                        episode_return=episode_return,
                        camera_name=args.camera,
                    )
                )
        raw_frames.append(_raw_preview(observation))
        demo_frames.append(
            _demo_frame(
                observation,
                info,
                episode_return=episode_return,
                camera_name=args.camera,
            )
        )

        Image.fromarray(observation["rgb"]).save(output / "final_rgb.png")
        np.save(output / "final_depth_m.npy", observation["depth"])
        raw_frames[-1].save(output / "final_rgb_depth_preview.png")
        demo_frames[-1].save(output / "final_demo.png")
        raw_frames[0].save(
            output / "rollout_rgb_depth.gif",
            save_all=True,
            append_images=raw_frames[1:],
            duration=80,
            loop=0,
        )
        demo_frames[0].save(
            output / "rollout_demo.gif",
            save_all=True,
            append_images=demo_frames[1:],
            duration=80,
            loop=0,
            optimize=False,
        )
        report = {
            "status": "pass",
            "policy_observation": "state_field_only",
            "rendered_observation": "hybrid_rgbd",
            "camera_name": args.camera,
            "aircraft_visuals": args.aircraft_visual_root is not None,
            "case": args.case,
            "episode_seed": args.episode_seed,
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
            "frames": len(raw_frames),
            "demo_anchor": "receiver_receptacle_mouth_frame",
            "demo_artifact": "rollout_demo.gif",
        }
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(report, sort_keys=True))
        if not info["official_success"]:
            raise RuntimeError("frozen policy did not succeed during RGB-D rendering")
    finally:
        env.close()


if __name__ == "__main__":
    main()
