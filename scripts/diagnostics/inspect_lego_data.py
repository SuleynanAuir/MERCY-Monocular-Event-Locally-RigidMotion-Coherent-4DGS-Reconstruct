"""Compare the supplied Lego Gaussian PLYs using its calibrated fixed camera."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch

from gaussian_renderer import render
from scripts.rendering.render_ply import make_camera
from scene.gaussian_model import GaussianModel
from utils.ply_utils import gaussian_degree


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads((args.data / "fixed_camera.json").read_text())
    w2c = np.eye(4)
    w2c[:3, :3] = np.asarray(data["R"]).T
    w2c[:3, 3] = data["T"]
    w2c[1:3] *= -1
    pose = np.linalg.inv(w2c)
    fovy = 2 * math.atan(data["height"] / (2 * data["fy"]))
    camera = make_camera(pose, data["width"], data["height"], fovy)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.tensor(data["background_rgb"], dtype=torch.float32, device="cuda")
    target = np.asarray(Image.open(args.data / "Lego.jpg").convert("RGB"), dtype=np.float32) / 255
    foreground = target.max(-1) > 12 / 255
    report = {"camera": {"pose": pose.tolist(), "width": data["width"], "height": data["height"],
                         "fovy_degrees": math.degrees(fovy),
                         "target": (pose[:3, 3] - data["distance"] * pose[:3, 2]).tolist()},
              "models": []}
    for name, filename in (("original", "Lego.ply"), ("corrected", "fan_contrastive_g0/contrastive_g0.ply")):
        path = args.data / filename
        model = GaussianModel(gaussian_degree(path), with_motion_mask=False)
        model.load_ply(str(path))
        for value in (model.get_xyz, model.get_rotation, model.get_scaling, model.get_opacity, model.get_features):
            if not torch.isfinite(value).all().item():
                raise ValueError(f"Non-finite activated Gaussian attributes: {filename}")
        times = []
        for _ in range(4):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = render(camera, model, pipe, background, 0., 0., 0.)
            end.record(); torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        rgb = result["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        alpha = result["alpha"].squeeze().cpu().numpy()
        Image.fromarray((rgb * 255).astype(np.uint8)).save(args.output / f"{name}.png")
        entry = {"file": filename, "gaussians": len(model.get_xyz), "render_ms": float(np.mean(times[1:])),
                 "mae": float(np.abs(rgb - target).mean()),
                 "foreground_mae": float(np.abs(rgb - target)[foreground].mean()),
                 "mask_iou": float(np.logical_and(alpha > .05, foreground).sum() /
                                   np.logical_or(alpha > .05, foreground).sum()),
                 "camera_source": "fixed_camera.json", "training_steps": 0}
        report["models"].append(entry)
        print(json.dumps(entry), flush=True)
        del model, result, value
        gc.collect(); torch.cuda.empty_cache()
    (args.output / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
