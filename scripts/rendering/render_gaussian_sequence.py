"""Render baked Gaussian frames. No deformation checkpoint or motion evaluation."""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from gaussian_renderer import render
from scripts.rendering.render_ply import make_camera
from scene.gaussian_model import GaussianModel
from utils.gaussian_sequence import GaussianSequence


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence_path", type=Path, required=True, help="Baked NPZ, PLY sequence ZIP or extracted directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    parser.add_argument("--camera-summary", type=Path, help="Override camera using a scripts/rendering/render_rotation.py summary")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    if not torch.cuda.is_available():
        raise RuntimeError("SC-GS rendering requires CUDA")
    with GaussianSequence(args.sequence_path) as sequence:
        camera_data = sequence.metadata.get("preview_camera")
        if args.camera_summary:
            summary = json.loads(args.camera_summary.read_text())
            camera_data = {"pose": summary["fixed_camera_pose"], "width": summary.get("width", 768),
                           "height": summary.get("height", 768), "fovy_degrees": summary.get("fovy_degrees", 60)}
        if camera_data is None:
            parser.error("This sequence has no camera; provide --camera-summary")
        camera = make_camera(np.asarray(camera_data["pose"], dtype=np.float32), camera_data["width"],
                             camera_data["height"], math.radians(camera_data["fovy_degrees"]))
        model = GaussianModel(sequence.metadata["sh_degree"], with_motion_mask=False)
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        background = torch.ones(3, device="cuda")
        fps = sequence.metadata["fps"]
        # An articulated excavator is not one rigid rotor: distances between
        # different parts must change. Retain the legacy fan invariant only
        # when the archive explicitly carries whole-rotor provenance.
        provenance = sequence.metadata.get("motion_provenance_only", {})
        whole_rigid = "axis" in provenance and "center" in provenance
        args.output.mkdir(parents=True, exist_ok=False)
        baseline = None
        invariant_error = 0.0
        maximum_position_change = 0.0
        distinct_frame_hashes = set()
        with imageio.get_writer(args.output / "preview.mp4", fps=fps, codec="libx264", quality=8,
                                macro_block_size=1, pixelformat="yuv420p") as writer:
            for i in range(len(sequence)):
                # Every frame is a full, final Gaussian state; zero deformation.
                with tempfile.TemporaryFile() as stream:
                    stream.write(sequence.frame_bytes(i))
                    stream.seek(0)
                    model.load_ply(stream)
                if len(model.get_xyz) != sequence.metadata["gaussians_per_frame"]:
                    raise ValueError("Gaussian count mismatch")
                for values in (model.get_xyz, model.get_rotation, model.get_scaling, model.get_opacity, model.get_features):
                    if not torch.isfinite(values).all():
                        raise ValueError(f"Non-finite Gaussian parameter in frame {i}")
                entry = sequence.frames[i]
                distinct_frame_hashes.add(entry.get("sha256", entry.get("pose_sha256")))
                if baseline is None:
                    baseline = {key: getattr(model, key).detach().clone()
                                for key in ("_xyz", "_scaling", "_opacity", "_features_dc", "_features_rest")}
                    indices = torch.linspace(0, len(model.get_xyz)-1, min(4096, len(model.get_xyz)), device="cuda").long()
                    original = baseline["_xyz"][indices]
                    rest_distances = (original - original.roll(37, 0)).norm(dim=1)
                    extent = float((original - original.mean(dim=0)).norm(dim=1).max())
                for key in ("_scaling", "_opacity", "_features_dc", "_features_rest"):
                    if not torch.equal(getattr(model, key), baseline[key]):
                        raise ValueError(f"Static Gaussian property {key} changed at frame {i}")
                current = model.get_xyz[indices]
                distances = (current - current.roll(37, 0)).norm(dim=1)
                invariant_error = max(invariant_error, float((distances - rest_distances).abs().max()))
                maximum_position_change = max(maximum_position_change, float((current-original).norm(dim=1).max()))
                if whole_rigid and invariant_error > max(extent, .01) * 2e-5:
                    raise ValueError(f"Stored frames violate rigid pair-distance invariant: {invariant_error}")
                result = render(camera, model, pipe, background, 0., 0., 0.)["render"]
                if not torch.isfinite(result).all():
                    raise ValueError(f"Non-finite render at frame {i}")
                pixels = (result.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                picture = Image.fromarray(pixels)
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, camera_data["width"], 30), fill="white")
                draw.text((10, 8), f"BAKED GAUSSIANS | frame {i:04d} | t={i/fps:.3f}s | no deform.pth", fill="black")
                writer.append_data(np.asarray(picture))
                if i in {0, fps, 2*fps, len(sequence)-1}:
                    picture.save(args.output / f"frame_{i:04d}.png")
                if i % fps == 0:
                    print(f"Rendered stored Gaussian frame {i}/{len(sequence)}", flush=True)
        (args.output / "summary.json").write_text(json.dumps({"sequence": str(args.sequence_path.resolve()),
            "frames": len(sequence), "fps": fps, "gaussians_per_frame": len(model.get_xyz),
            "deform_checkpoint_used": False, "motion_evaluated_during_playback": False,
            "all_frame_sha256_checks_passed": True, "distinct_frame_file_hashes": len(distinct_frame_hashes),
            "all_frames_static_attributes_unchanged": True,
            "whole_object_rigidity_checked": whole_rigid,
            "rigid_pair_distance_max_error_ply_units": invariant_error if whole_rigid else None,
            "maximum_sampled_position_change_ply_units": maximum_position_change,
            "render_deformation": "all zero; positions and orientations loaded directly from frame PLY"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
