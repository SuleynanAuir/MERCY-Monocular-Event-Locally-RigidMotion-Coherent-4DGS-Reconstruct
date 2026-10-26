"""Render an existing Gaussian PLY with SC-GS, without training or a display."""

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
from PIL import Image
import torch

from utils.cam_utils import orbit_camera
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix
from utils.ply_utils import gaussian_degree


def make_camera(pose, width, height, fovy):
    # OpenGL camera-to-world -> the rasterizer's +z-forward convention.
    w2c = np.linalg.inv(pose)
    w2c[1:3, :] *= -1
    view = torch.as_tensor(w2c, dtype=torch.float32, device="cuda").T.contiguous()
    fovx = 2 * math.atan(math.tan(fovy / 2) * width / height)
    projection = getProjectionMatrix(0.01, 1000.0, fovx, fovy).T.cuda()
    return SimpleNamespace(
        image_width=width, image_height=height, FoVx=fovx, FoVy=fovy,
        world_view_transform=view, full_proj_transform=view @ projection,
        camera_center=torch.as_tensor(pose[:3, 3], dtype=torch.float32, device="cuda"),
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gs_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--fovy", type=float, default=60.0)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--elevation", type=float, default=-15.0)
    parser.add_argument("--camera-origin", action="store_true",
                        help="Use the origin looking along +z (Splatt3r's first camera).")
    args = parser.parse_args()
    if min(args.frames, args.width, args.height, args.fps) < 1:
        parser.error("frames, dimensions and fps must be positive")
    if not 0 < args.fovy < 180 or (args.radius is not None and args.radius <= 0):
        parser.error("fovy must be between 0 and 180; radius must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty; choose a new directory")
    if not torch.cuda.is_available():
        raise RuntimeError("SC-GS rendering requires a CUDA GPU")
    degree = gaussian_degree(args.gs_path)
    model = GaussianModel(degree, with_motion_mask=False)
    model.load_ply(str(args.gs_path))
    if not len(model.get_xyz) or not torch.isfinite(model.get_xyz).all():
        raise ValueError("Model positions are empty or non-finite")
    xyz = model.get_xyz.cpu().numpy()
    lo, hi = np.quantile(xyz, [0.02, 0.98], axis=0)
    center = (lo + hi) / 2
    fovy = math.radians(args.fovy)
    radius = args.radius or max(float(np.linalg.norm(hi - lo)), 0.1) / math.tan(fovy / 2)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.ones(3, device="cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    timings, coverages, frames = [], [], []
    for i in range(args.frames):
        if args.camera_origin:
            pose = np.diag([1., -1., -1., 1.]).astype(np.float32)
        else:
            pose = orbit_camera(args.elevation, 360 * i / args.frames, radius, target=center)
        camera = make_camera(pose, args.width, args.height, fovy)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = render(camera, model, pipe, background, 0., 0., 0.)
        end.record()
        torch.cuda.synchronize()
        if not torch.isfinite(result["render"]).all():
            raise RuntimeError(f"Frame {i} contains non-finite pixels")
        timings.append(start.elapsed_time(end))
        coverages.append(float((result["alpha"] > 0.01).float().mean()))
        rgb = (result["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb).save(args.output / f"{i:04d}.png")
        frames.append(rgb)
    if max(coverages) == 0:
        raise RuntimeError("No visible Gaussians; adjust the camera or radius")
    if args.frames > 1:
        imageio.mimwrite(args.output / "preview.mp4", frames, fps=args.fps, macro_block_size=1)
    summary = {
        "model": str(args.gs_path.resolve()), "gaussians": len(xyz), "sh_degree": degree,
        "gpu": torch.cuda.get_device_name(), "frames": args.frames,
        "resolution": [args.width, args.height], "training_steps": 0,
        "render_ms_mean": float(np.mean(timings)), "alpha_coverage": coverages,
        "max_gpu_allocated_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
