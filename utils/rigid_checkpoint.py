"""Versioned analytic motion checkpoints; NOT SC-GS neural-network weights.

Only PyTorch, NumPy and plyfile are needed to export/load/evaluate on CPU.
Rendering still requires the SC-GS CUDA environment. Times are seconds, not
SC-GS's normalized dataset timestamps. Always evaluate from canonical positions.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from plyfile import PlyData
import torch

from utils.ply_utils import gaussian_degree
from utils.rigid_motion import axis_angle, spin_profile


FORMAT = "scgs_analytic_rigid_rotation"
VERSION = 1
CONVENTIONS = {
    "profile": "quintic_speed_ramp_then_constant",
    "time_unit": "seconds",
    "reference_time_seconds": 0.0,
    "quaternion_order": "wxyz",
    "rotation_composition": "rotor_quaternion_left_multiply_canonical",
    "scope": "entire_ply_is_rotor",
    "appearance": "zero_directional_sh",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_rotor_ply(path):
    """Validate supported appearance/geometry without allocating CUDA tensors."""
    degree = gaussian_degree(path)
    vertex = PlyData.read(path)["vertex"]
    if not len(vertex):
        raise ValueError("The Gaussian PLY is empty")
    names = {p.name for p in vertex.properties}
    for name in names:
        if name in {"x", "y", "z"} or name.startswith(("f_dc_", "f_rest_", "scale_", "rot_")):
            if not np.isfinite(vertex[name]).all():
                raise ValueError(f"Non-finite Gaussian property: {name}")
        if name.startswith("f_rest_") and np.any(vertex[name] != 0):
            raise ValueError("Rigid appearance currently requires zero higher-order SH coefficients")
    # Infinite raw logits are valid: sigmoid(+/-inf) = 1/0.
    if np.isnan(vertex["opacity"]).any():
        raise ValueError("NaN opacity logits")
    with np.errstate(over="ignore"):
        for i in range(3):
            if not np.isfinite(np.exp(vertex[f"scale_{i}"])).all():
                raise ValueError("Non-finite activated Gaussian scale")
    rotations = np.column_stack([vertex[f"rot_{i}"] for i in range(4)]).astype(np.float64)
    if np.any(np.linalg.norm(rotations, axis=1) == 0):
        raise ValueError("Zero Gaussian quaternion")
    return {"count": len(vertex), "sh_degree": degree}


@dataclass(frozen=True)
class RigidRotation:
    center: tuple
    axis: tuple
    rpm: float = 60.0
    ramp_seconds: float = 2.0
    duration_seconds: float = 6.0
    direction: str = "cw"

    def __post_init__(self):
        center = np.asarray(self.center, dtype=np.float64)
        axis = np.asarray(self.axis, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("center must be a finite 3-vector")
        axis_angle(axis, 0)
        if not math.isfinite(float(np.linalg.norm(axis))):
            raise ValueError("axis norm must be finite")
        spin_profile(0, self.rpm, self.ramp_seconds)
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be finite and positive")
        if self.direction not in ("cw", "ccw"):
            raise ValueError("direction must be cw or ccw, viewed from +axis")
        object.__setattr__(self, "center", tuple(float(x) for x in center))
        object.__setattr__(self, "axis", tuple(float(x) for x in axis / np.linalg.norm(axis)))
        for name in ("rpm", "ramp_seconds", "duration_seconds"):
            object.__setattr__(self, name, float(getattr(self, name)))

    def state_at(self, time_seconds):
        """Signed angle, velocity, acceleration; scalar time in seconds.

        duration_seconds is the default preview duration, not a loop period.
        Motion continues at target RPM after the ramp, including beyond duration.
        """
        if np.ndim(time_seconds) != 0:
            raise ValueError("Provide one scalar time in seconds, not per-point timestamps")
        values = spin_profile(float(time_seconds), self.rpm, self.ramp_seconds)
        sign = -1 if self.direction == "cw" else 1
        return tuple(sign * float(value) for value in values)

    def step(self, canonical_xyz, time_seconds):
        """Return keyword arguments for gaussian_renderer.render(..., **motion).

        Use the ORIGINAL Nx3 positions on every call; do not pass last frame's
        moved positions. d_rotation_bias rotates each Gaussian's covariance as
        well as its center. Scales, opacity and appearance remain unchanged.
        This analytic time interface is not intended for training/autograd in t.
        """
        if canonical_xyz.ndim != 2 or canonical_xyz.shape[1] != 3 or not canonical_xyz.is_floating_point():
            raise ValueError("canonical_xyz must be a floating-point Nx3 tensor")
        angle = self.state_at(time_seconds)[0]
        matrix, quaternion = axis_angle(self.axis, angle)
        center = canonical_xyz.new_tensor(self.center)
        rotation = canonical_xyz.new_tensor(matrix.tolist())
        moved = (canonical_xyz - center) @ rotation.T + center
        return {"d_xyz": moved - canonical_xyz, "d_rotation": 0.0, "d_scaling": 0.0,
                "d_opacity": None, "d_color": None,
                "d_rotation_bias": canonical_xyz.new_tensor(quaternion.tolist()).expand(len(canonical_xyz), 4)}


def export_rigid_bundle(gs_path, output, motion):
    """Copy canonical PLY byte-for-byte and save a safe primitive-only checkpoint.

    The destination must not already exist. No source model is modified.
    """
    gs_path, output = Path(gs_path), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {output}")
    metadata = inspect_rotor_ply(gs_path)
    source_hash = sha256_file(gs_path)
    payload = {
        "format": FORMAT, "version": VERSION, "training_steps": 0,
        "conventions": dict(CONVENTIONS), "motion": asdict(motion),
        "gaussians": {"filename": "point_cloud.ply", "sha256": source_hash, **metadata},
        "notes": {
            "compatibility": "Custom analytic parameters, NOT native SC-GS network state_dict",
            "physics": "Prescribed rigid kinematics, not motor torque or airflow simulation",
            "axis_provenance": "Configured estimate, not measured mechanical ground truth",
            "spatial_units": "Same as the source PLY; no physical length calibration assumed",
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "point_cloud.ply"
    with gs_path.open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    if sha256_file(destination) != source_hash:
        raise RuntimeError("Source changed during copy or copied PLY checksum mismatch")
    with (output / "deform.pth").open("xb") as stream:
        torch.save(payload, stream)
    with (output / "manifest.json").open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    return payload


@dataclass(frozen=True)
class RigidBundle:
    gs_path: Path
    motion: RigidRotation
    metadata: dict


def load_rigid_bundle(deform_path, gs_path=None):
    """Load custom checkpoint with restricted unpickling and verify its PLY.

    Both files may be moved together. An explicit gs_path also supports a renamed
    PLY, but it must match the saved SHA256. Never falls back to unsafe pickle.
    """
    deform_path = Path(deform_path)
    payload = torch.load(deform_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT or payload.get("version") != VERSION:
        raise ValueError("Not a supported analytic rigid checkpoint (native SC-GS weights are incompatible)")
    if payload.get("conventions") != CONVENTIONS or payload.get("training_steps") != 0:
        raise ValueError("Unsupported checkpoint conventions")
    metadata = payload.get("gaussians", {})
    if metadata.get("filename") != "point_cloud.ply":
        raise ValueError("Unsupported canonical PLY filename")
    path = Path(gs_path) if gs_path is not None else deform_path.parent / "point_cloud.ply"
    if sha256_file(path) != metadata.get("sha256"):
        raise ValueError("PLY SHA256 mismatch: checkpoint and Gaussians do not belong together")
    actual = inspect_rotor_ply(path)
    if any(actual[key] != metadata.get(key) for key in actual):
        raise ValueError("PLY metadata mismatch")
    motion = RigidRotation(**payload["motion"])
    return RigidBundle(path, motion, payload)
