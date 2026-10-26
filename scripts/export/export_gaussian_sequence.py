"""Bake analytic motion into a ZIP of complete, already-transformed Gaussian PLYs."""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from plyfile import PlyData, PlyElement
import torch

from utils.gaussian_sequence import FORMAT, VERSION
from utils.rigid_checkpoint import load_rigid_bundle
from utils.rigid_motion import axis_angle


def compose_quaternions(left, right):
    """Scalar-first Hamilton product, same order as the SC-GS renderer."""
    aw, ax, ay, az = left.unbind(-1)
    bw, bx, by, bz = right.unbind(-1)
    result = torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                          aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)
    return torch.where(result[:, :1] < 0, -result, result)


class BakedRotor:
    """Preserve every PLY property; update xyz, quaternion and optional normals."""

    def __init__(self, source, motion, device="cpu"):
        ply = PlyData.read(source)
        if len(ply.elements) != 1 or ply.elements[0].name != "vertex":
            raise ValueError("Only vertex-only Gaussian PLYs are supported")
        self.vertices = ply["vertex"].data.copy()
        if any(dtype[0].kind != "f" for dtype in self.vertices.dtype.fields.values()):
            raise ValueError("Expected floating-point Gaussian vertex properties")
        self.motion = motion
        xyz = np.column_stack([self.vertices[name] for name in ("x", "y", "z")])
        rotations = np.column_stack([self.vertices[f"rot_{i}"] for i in range(4)])
        self.xyz = torch.tensor(xyz.tolist(), dtype=torch.float32, device=device)
        self.rotation = torch.nn.functional.normalize(torch.tensor(rotations.tolist(), dtype=torch.float32, device=device), dim=1)
        self.normals = None
        if all(name in self.vertices.dtype.names for name in ("nx", "ny", "nz")):
            normals = np.column_stack([self.vertices[name] for name in ("nx", "ny", "nz")])
            if np.any(normals):
                self.normals = torch.tensor(normals.tolist(), dtype=torch.float32, device=device)

    @torch.no_grad()
    def frame(self, time_seconds):
        deformation = self.motion.step(self.xyz, time_seconds)
        positions = self.xyz + deformation["d_xyz"]
        orientations = compose_quaternions(deformation["d_rotation_bias"], self.rotation)
        # list conversion keeps CPU operation independent of torch/numpy ABI.
        # Native tensor.numpy is substantially faster on the configured server.
        try:
            xyz, rotation = positions.cpu().numpy(), orientations.cpu().numpy()
        except RuntimeError as error:
            if "Numpy is not available" not in str(error):
                raise
            xyz, rotation = np.asarray(positions.tolist()), np.asarray(orientations.tolist())
        vertices = self.vertices.copy()
        for i, name in enumerate(("x", "y", "z")):
            vertices[name] = xyz[:, i]
        for i in range(4):
            vertices[f"rot_{i}"] = rotation[:, i]
        if self.normals is not None:
            matrix = axis_angle(self.motion.axis, self.motion.state_at(time_seconds)[0])[0]
            normals = self.normals @ self.normals.new_tensor(matrix.tolist()).T
            for i, name in enumerate(("nx", "ny", "nz")):
                vertices[name] = normals[:, i].tolist()
        return PlyData([PlyElement.describe(vertices, "vertex")], text=False, byte_order="<",
                       comments=[f"Baked rigid Gaussian state at t={time_seconds:.12g} seconds; quaternion wxyz"])


def export_sequence(deform_path, output, fps=60, duration=None, device="cpu", camera_summary=None):
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    bundle = load_rigid_bundle(deform_path)
    duration = bundle.motion.duration_seconds if duration is None else duration
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    frame_count = math.ceil(duration * fps)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    raw_bytes = frame_count * (bundle.gs_path.stat().st_size + 256)
    if shutil.disk_usage(parent).free < raw_bytes + 512 * 1024**2:
        raise OSError(f"Need at least {raw_bytes / 1e9:.2f} GB free as a conservative archive-size bound")
    camera = None
    if camera_summary is not None:
        summary = json.loads(Path(camera_summary).read_text())
        camera = {"pose": summary["fixed_camera_pose"], "width": summary.get("width", 768),
                  "height": summary.get("height", 768), "fovy_degrees": summary.get("fovy_degrees", 60)}
    baker = BakedRotor(bundle.gs_path, bundle.motion, device)
    metadata = {"format": FORMAT, "version": VERSION, "frame_count": frame_count, "fps": fps,
                "duration_seconds": frame_count / fps, "gaussians_per_frame": len(baker.vertices),
                "sh_degree": bundle.metadata["gaussians"]["sh_degree"], "time_unit": "seconds",
                "quaternion_order": "wxyz", "stable_gaussian_row_order": True,
                "motion_is_baked": True, "requires_deform_checkpoint_for_playback": False,
                "training_steps": 0, "source_sha256": bundle.metadata["gaussians"]["sha256"],
                "motion_provenance_only": asdict(bundle.motion), "preview_camera": camera,
                "attributes": "Each PLY is complete. xyz/rot and nonzero normals rotate; all other fields unchanged.",
                "scope": "Whole PLY is one rigid rotor; prescribed kinematics, not motor/airflow simulation.",
                "sampling": "Instantaneous states at i/fps, no exposure blur baked into geometry.",
                "frames": []}
    output.mkdir(parents=True, exist_ok=False)
    partial = output / "fan_motion.plyseq.zip.partial"
    started = time.monotonic()
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for i in range(frame_count):
            t = i / fps
            stream = io.BytesIO()
            baker.frame(t).write(stream)
            data = stream.getvalue()
            filename = f"frames/frame_{i:04d}.ply"
            archive.writestr(filename, data)
            angle, speed, acceleration = bundle.motion.state_at(t)
            metadata["frames"].append({"file": filename, "time_seconds": t,
                                       "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                                       "angle_rad": angle, "rpm": abs(speed) * 60 / (2 * math.pi),
                                       "angular_acceleration_rad_s2": acceleration})
            if i % fps == 0 or i == frame_count - 1:
                print(f"Baked {i+1}/{frame_count} frames, t={t:.3f}s, archive={partial.stat().st_size/1e6:.1f} MB, elapsed={time.monotonic()-started:.1f}s", flush=True)
        archive.writestr("sequence.json", json.dumps(metadata, indent=2) + "\n")
        archive.writestr("README.txt", "BAKED MOVING GAUSSIANS\n\n"
                         f"{frame_count} frames, {fps} FPS, {len(baker.vertices)} Gaussians per frame.\n"
                         "Every frames/frame_NNNN.ply contains the actual already-transformed Gaussian state.\n"
                         "No deform.pth, learned network, source PLY or motion formula is needed for playback.\n"
                         "Read sequence.json for timestamps. Gaussian row IDs are consistent across frames.\n"
                         "A single PLY is one snapshot; animate by displaying these files in order.\n"
                         "In the supplied SC-GS repo: python scripts/rendering/render_gaussian_sequence.py --sequence_path THIS.zip --output NEW_DIRECTORY\n"
                         "After extraction, load_sequence.py provides CPU access with only numpy and plyfile.\n"
                         "The first 2 seconds accelerate from rest for this fan; do not loop the entire startup as steady motion.\n")
        archive.write(_PROJECT_ROOT / "utils/gaussian_sequence.py", "load_sequence.py")
    final_path = output / "fan_motion.plyseq.zip"
    partial.rename(final_path)
    (output / "sequence.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Finished: {final_path} ({final_path.stat().st_size/1e9:.3f} GB)", flush=True)
    return final_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deform_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--camera-summary", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    export_sequence(args.deform_path, args.output, args.fps, args.duration, args.device, args.camera_summary)


if __name__ == "__main__":
    main()
