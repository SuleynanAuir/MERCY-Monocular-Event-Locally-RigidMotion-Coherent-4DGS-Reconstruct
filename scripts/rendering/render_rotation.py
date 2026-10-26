"""Render a rotor-only Gaussian PLY with prescribed rigid rotation, no training."""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from gaussian_renderer import render
from scripts.rendering.render_ply import make_camera
from scene.gaussian_model import GaussianModel
from utils.ply_utils import gaussian_degree
from utils.rigid_motion import axis_angle, spin_profile
from utils.rigid_checkpoint import load_rigid_bundle


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gs_path", type=Path)
    parser.add_argument("--deform_path", type=Path,
                        help="Custom analytic deform.pth; resolves its paired PLY and motion parameters")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--center", nargs=3, type=float,
                        help="A point on the shaft axis, in PLY coordinates; not the bounding-box center.")
    parser.add_argument("--axis", nargs=3, type=float)
    parser.add_argument("--rpm", type=float)
    parser.add_argument("--ramp-seconds", type=float)
    parser.add_argument("--duration", type=float, help="Preview duration in seconds (can override checkpoint default)")
    parser.add_argument("--direction", choices=["cw", "ccw"],
                        help="Direction viewed from the positive axis toward the rotor.")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--fovy", type=float, default=60)
    parser.add_argument("--view-angle", type=float, default=0,
                        help="Fixed camera offset from the positive shaft axis, in degrees.")
    parser.add_argument("--radius", type=float)
    parser.add_argument("--shutter-fraction", type=float, default=.5,
                        help="Exposure duration as a fraction of one frame interval.")
    parser.add_argument("--samples", type=int, default=4, help="Temporal samples per exposure.")
    parser.add_argument("--show-axis", action="store_true")
    args = parser.parse_args()
    bundle = None
    if args.deform_path:
        if any(getattr(args, key) is not None for key in ("center", "axis", "rpm", "ramp_seconds", "direction")):
            parser.error("Checkpoint motion is authoritative; do not combine --deform_path with motion flags")
        bundle = load_rigid_bundle(args.deform_path, args.gs_path)
        args.gs_path = bundle.gs_path
        for key in ("center", "axis", "rpm", "ramp_seconds", "direction"):
            setattr(args, key, getattr(bundle.motion, key))
        if args.duration is None:
            args.duration = bundle.motion.duration_seconds
    else:
        if args.gs_path is None or args.center is None:
            parser.error("Provide --deform_path, or both --gs_path and --center")
        for key, default in {"axis": [0, 0, 1], "rpm": 60, "ramp_seconds": 2, "direction": "cw"}.items():
            if getattr(args, key) is None:
                setattr(args, key, default)
    if args.duration is None:
        args.duration = 6
    try:
        spin_profile(0, args.rpm, args.ramp_seconds)
        axis_angle(args.axis, 0)
    except ValueError as error:
        parser.error(str(error))
    if not np.isfinite(args.center).all():
        parser.error("center must be finite")
    if min(args.width, args.height, args.fps, args.samples) < 1 or args.width % 2 or args.height % 2:
        parser.error("dimensions must be positive and even; fps and samples must be positive")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be finite and positive")
    if not 0 < args.fovy < 180 or not math.isfinite(args.view_angle):
        parser.error("invalid camera field of view or angle")
    if args.radius is not None and (not math.isfinite(args.radius) or args.radius <= .01):
        parser.error("camera radius must be finite and greater than .01")
    if not 0 <= args.shutter_fraction <= 1:
        parser.error("shutter fraction must be between 0 and 1")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output must be a new or empty directory")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU and the SC-GS environment are required")

    model = GaussianModel(gaussian_degree(args.gs_path), with_motion_mask=False)
    model.load_ply(str(args.gs_path))
    if not len(model.get_xyz):
        raise ValueError("The model is empty")
    for values in (model.get_xyz, model.get_features, model.get_scaling, model.get_opacity, model.get_rotation):
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite activated Gaussian parameters")
    # This asset stores SH3 but all its directional coefficients are zero.
    # Rotating nonzero SH appearance requires an additional material-frame SH
    # transform; refuse it rather than silently attaching appearance to world axes.
    if torch.count_nonzero(model._features_rest):
        raise ValueError("This rotor renderer currently requires zero higher-order SH coefficients")

    center = torch.tensor(args.center, device="cuda", dtype=torch.float32)
    axis = np.asarray(args.axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    axis_gpu = torch.tensor(axis, device="cuda", dtype=torch.float32)
    original = model.get_xyz.detach()
    offsets = original - center
    extent = float(offsets.norm(dim=1).max())
    fovy = math.radians(args.fovy)
    fovx = 2 * math.atan(math.tan(fovy / 2) * args.width / args.height)
    radius = args.radius or max(extent, .01) / math.sin(min(fovy, fovx) / 2) * 1.15
    up = np.array([0., 1., 0.]) if abs(axis[1]) < .95 else np.array([1., 0., 0.])
    right = np.cross(up, axis)
    right /= np.linalg.norm(right)
    up = np.cross(axis, right)
    view_angle = math.radians(args.view_angle)
    forward = axis * math.cos(view_angle) + right * math.sin(view_angle)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.column_stack((np.cross(up, forward), up, forward))
    pose[:3, 3] = np.asarray(args.center) + radius * forward
    camera = make_camera(pose, args.width, args.height, fovy)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.ones(3, device="cuda")

    # Validate geometric invariants on deterministic samples, on every frame.
    indices = torch.linspace(0, len(original) - 1, min(4096, len(original)), device="cuda").long()
    rest = offsets[indices]
    rest_axial = rest @ axis_gpu
    rest_radial = (rest - rest_axial[:, None] * axis_gpu).norm(dim=1)
    rest_pairs = (rest - rest.roll(37, 0)).norm(dim=1)
    errors = {"axis_distance": 0., "axial_position": 0., "pair_distance": 0.}
    frames = max(1, math.ceil(args.duration * args.fps))
    sign = -1 if args.direction == "cw" else 1
    exposure_offsets = ((np.arange(args.samples) + .5) / args.samples - .5) * args.shutter_fraction / args.fps
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    args.output.mkdir(parents=True, exist_ok=True)
    timeline = []
    checkpoints = {0, min(frames - 1, round(args.ramp_seconds * args.fps / 2)),
                   min(frames - 1, round(args.ramp_seconds * args.fps)), frames - 1}
    with imageio.get_writer(args.output / "preview.mp4", fps=args.fps, codec="libx264",
                            quality=8, macro_block_size=1, pixelformat="yuv420p") as writer:
        for i in range(frames):
            t = i / args.fps
            angle, speed, acceleration = spin_profile(t, args.rpm, args.ramp_seconds)
            matrix, _ = axis_angle(axis, sign * float(angle))
            rotated = rest @ torch.tensor(matrix, device="cuda", dtype=torch.float32).T
            axial = rotated @ axis_gpu
            radial = (rotated - axial[:, None] * axis_gpu).norm(dim=1)
            pairs = (rotated - rotated.roll(37, 0)).norm(dim=1)
            for name, difference in (("axis_distance", radial - rest_radial),
                                     ("axial_position", axial - rest_axial), ("pair_distance", pairs - rest_pairs)):
                errors[name] = max(errors[name], float(difference.abs().max()))
            if max(errors.values()) > max(extent, .01) * 2e-5:
                raise RuntimeError(f"Rigid-body invariant violated: {errors}")
            rgb = torch.zeros((3, args.height, args.width), device="cuda")
            for delta in exposure_offsets:
                if bundle is not None:
                    deformation = bundle.motion.step(original, max(0., t + delta))
                    result = render(camera, model, pipe, background, **deformation)
                else:
                    sample_angle = spin_profile(max(0., t + delta), args.rpm, args.ramp_seconds)[0]
                    rotation, quaternion = axis_angle(axis, sign * float(sample_angle))
                    rotation = torch.tensor(rotation, device="cuda", dtype=torch.float32)
                    quaternion = torch.tensor(quaternion, device="cuda", dtype=torch.float32).expand(len(original), 4)
                    moved = offsets @ rotation.T + center
                    result = render(camera, model, pipe, background, moved - original, 0., 0., d_rotation_bias=quaternion)
                if not torch.isfinite(result["render"]).all():
                    raise RuntimeError(f"Non-finite image at frame {i}")
                rgb += result["render"] / args.samples
            pixels = (rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            picture = Image.fromarray(pixels)
            draw = ImageDraw.Draw(picture)
            current_rpm = float(speed) * 60 / (2 * math.pi)
            draw.rectangle((0, 0, args.width, 62), fill="white")
            draw.text((12, 8), f"RIGID ROTOR | t = {t:.2f} s | {current_rpm:.1f} RPM", font=font, fill="black")
            draw.text((12, 34), f"{args.direction.upper()} from +axis | fixed camera | ramp {args.ramp_seconds:g} s", font=font, fill="black")
            if args.show_axis:
                cx, cy = (args.width - 1) / 2, (args.height - 1) / 2
                draw.line((cx - 5, cy, cx + 5, cy), fill="red", width=1)
                draw.line((cx, cy - 5, cx, cy + 5), fill="red", width=1)
            writer.append_data(np.asarray(picture))
            if i in checkpoints:
                picture.save(args.output / f"frame_{i:04d}.png")
            timeline.append({"time_s": t, "angle_rad": sign * float(angle), "rpm": current_rpm,
                             "angular_velocity_rad_s": sign * float(speed),
                             "angular_acceleration_rad_s2": sign * float(acceleration)})
            if i % args.fps == 0:
                print(f"frame {i}/{frames}: {current_rpm:.1f} RPM", flush=True)

    summary = {"model": str(args.gs_path.resolve()), "gaussians": len(original), "training_steps": 0,
               "deform_checkpoint": str(args.deform_path.resolve()) if args.deform_path else None,
               "motion": "prescribed rigid-body rotation; quintic speed ramp then constant speed",
               "center": args.center, "unit_axis": axis.tolist(), "target_rpm": args.rpm,
               "ramp_seconds": args.ramp_seconds, "direction_viewed_from_positive_axis": args.direction,
               "fps": args.fps, "frames": frames, "duration_seconds": frames / args.fps,
               "width": args.width, "height": args.height, "fovy_degrees": args.fovy,
               "view_angle_degrees": args.view_angle, "show_axis": args.show_axis,
               "fixed_camera_pose": pose.tolist(), "shutter_fraction": args.shutter_fraction,
               "samples_per_exposure": args.samples, "rigid_invariant_max_errors_ply_units": errors,
               "scales_opacity_colors": "unchanged", "gaussian_orientations": "left-composed with rotor quaternion",
               "scope": "Entire PLY assumed to be the rotor; no stationary housing, motor dynamics or airflow simulation",
               "axis_provenance": "User-supplied estimate/configuration, not measured mechanical ground truth",
               "max_gpu_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2}
    (args.output / "motion.json").write_text(json.dumps(timeline, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
