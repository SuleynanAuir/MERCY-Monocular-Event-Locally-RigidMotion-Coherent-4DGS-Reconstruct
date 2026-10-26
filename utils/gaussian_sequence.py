"""Baked Gaussian NPZ/PLY sequences: positions/orientations are stored, not inferred.

Playback requires only the sequence archive, NumPy and plyfile; it never loads
deform.pth or evaluates a motion function. A conventional single-PLY viewer can
open any extracted frame, but a timeline-aware viewer is needed for animation.
"""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile
import weakref

import numpy as np
from plyfile import PlyData, PlyElement


FORMAT = "baked_gaussian_ply_sequence"
VERSION = 1
SPARSE_VERSION = 2
POSE_FIELDS = ["x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3"]
_SHARED_POSES = weakref.WeakValueDictionary()


class GaussianSequence:
    """Read actual Gaussian states from a baked NPZ, PLY ZIP or extracted directory."""

    def __init__(self, path, *, share_poses=False):
        self.path = Path(path)
        self.archive = None
        self.tensor_archive = None
        self.poses = None
        self.dynamic_indices = None
        try:
            if self.path.suffix == ".npz":
                self.tensor_archive = np.load(self.path, allow_pickle=False)
                data = str(self.tensor_archive["metadata_json"].item())
                # The viewer only reads poses. Its V1/V2 and V1/V3 panels may
                # refer to the same multi-GB V1 archive. Weak sharing avoids
                # decompressing and retaining duplicate copies; ordinary
                # sequence callers keep the original independent semantics.
                stat = self.path.stat()
                identity = (str(self.path.resolve()), stat.st_size, stat.st_mtime_ns)
                self.poses = _SHARED_POSES.get(identity) if share_poses else None
                if self.poses is None:
                    self.poses = self.tensor_archive["poses"]
                    if share_poses:
                        _SHARED_POSES[identity] = self.poses
                self.base_vertices = self.tensor_archive["base_vertices"]
                if "dynamic_indices" in self.tensor_archive:
                    self.dynamic_indices = self.tensor_archive["dynamic_indices"]
            elif self.path.is_dir():
                data = (self.path / "sequence.json").read_bytes()
            else:
                self.archive = zipfile.ZipFile(self.path, "r")
                data = self.archive.read("sequence.json")
            self.metadata = json.loads(data)
            version = self.metadata.get("version")
            if self.metadata.get("format") != FORMAT or version not in (VERSION, SPARSE_VERSION):
                raise ValueError("Not a supported baked Gaussian sequence")
            if version == SPARSE_VERSION:
                if self.poses is None or self.dynamic_indices is None:
                    raise ValueError("Sparse sequence requires NPZ poses and dynamic indices")
                indices = self.dynamic_indices
                n = self.metadata["gaussians_per_frame"]
                if (indices.ndim != 1 or indices.dtype.kind not in "iu" or
                        np.any(indices >= n) or np.any(indices < 0) or
                        np.any(indices[1:] <= indices[:-1])):
                    raise ValueError("Invalid dynamic Gaussian indices")
                if hashlib.sha256(self.base_vertices.tobytes()).hexdigest() != self.metadata.get("base_vertices_sha256"):
                    raise ValueError("Base Gaussian checksum mismatch")
            elif self.dynamic_indices is not None:
                raise ValueError("Dynamic indices require sparse version 2")
            self.frames = self.metadata["frames"]
            if not self.frames or len(self.frames) != self.metadata["frame_count"]:
                raise ValueError("Invalid sequence frame count")
            if not np.isfinite(self.metadata["fps"]) or self.metadata["fps"] <= 0:
                raise ValueError("Invalid sequence FPS")
            if self.poses is not None:
                n = self.metadata["gaussians_per_frame"]
                count = n if self.dynamic_indices is None else len(self.dynamic_indices)
                expected = (count, len(self.frames), len(self.metadata["pose_fields"]))
                if self.poses.shape != expected or len(self.base_vertices) != n or self.poses.dtype != np.float32:
                    raise ValueError("Invalid baked tensor shapes or dtype")
                if self.metadata["pose_fields"] not in (["x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3"],
                        ["x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "nx", "ny", "nz"]):
                    raise ValueError("Unsupported baked pose fields")
                if any(name not in self.base_vertices.dtype.names for name in self.metadata["pose_fields"]):
                    raise ValueError("Missing Gaussian pose fields")
            for index, frame in enumerate(self.frames):
                if frame["file"] != f"frames/frame_{index:04d}.ply":
                    raise ValueError("Unsafe or unsupported frame path")
                if not np.isclose(frame["time_seconds"], index / self.metadata["fps"], rtol=0, atol=1e-9):
                    raise ValueError("Invalid frame timestamp")
            if self.tensor_archive is not None:
                times = self.tensor_archive["times_seconds"]
                expected_times = np.asarray([frame["time_seconds"] for frame in self.frames])
                if times.shape != expected_times.shape or not np.array_equal(times, expected_times):
                    raise ValueError("Baked tensor timestamps disagree with frame metadata")
        except Exception:
            self.close()
            raise

    def __len__(self):
        return len(self.frames)

    def validate_pose_frame(self, index):
        """Sparse archives checksum the shared base and each explicit pose slab."""
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError("Frame index out of range")
        if self.dynamic_indices is not None:
            digest = hashlib.sha256(self.poses[:, index].tobytes()).hexdigest()
            if digest != self.frames[index].get("pose_sha256"):
                raise ValueError(f"Frame {index} pose checksum mismatch")

    def frame_bytes(self, index):
        self.validate_pose_frame(index)
        entry = self.frames[index]
        if self.poses is not None:
            vertices = self.base_vertices.copy()
            for column, name in enumerate(self.metadata["pose_fields"]):
                if self.dynamic_indices is None:
                    vertices[name] = self.poses[:, index, column]
                else:
                    vertices[name][self.dynamic_indices] = self.poses[:, index, column]
            stream = io.BytesIO()
            kind = "Baked rigid Gaussian" if self.dynamic_indices is None else "Baked Gaussian"
            PlyData([PlyElement.describe(vertices, "vertex")], text=False, byte_order="<",
                    comments=[f"{kind} state at t={entry['time_seconds']:.12g} seconds; quaternion wxyz"]).write(stream)
            data = stream.getvalue()
        else:
            data = self.archive.read(entry["file"]) if self.archive else (self.path / entry["file"]).read_bytes()
        if self.dynamic_indices is None and hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Frame {index} checksum mismatch")
        return data

    def frame(self, index):
        """Return a full PLY with stable Gaussian row identities across frames."""
        # plyfile 1.0.3's BytesIO fallback decodes every scalar in Python.
        # A temporary file enables its fast memory-mapped reader. Only one
        # frame is staged; AutoDL's supplied environment puts TMPDIR on data disk.
        with tempfile.TemporaryFile() as stream:
            stream.write(self.frame_bytes(index))
            stream.seek(0)
            ply = PlyData.read(stream)
            for element in ply:
                element.data = element.data.copy()
        if len(ply["vertex"]) != self.metadata["gaussians_per_frame"]:
            raise ValueError("Gaussian count mismatch")
        return ply

    def close(self):
        if self.archive is not None:
            self.archive.close()
        if self.tensor_archive is not None:
            self.tensor_archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
