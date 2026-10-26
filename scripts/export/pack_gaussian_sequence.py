"""Losslessly pack all baked per-Gaussian trajectories into a single NPZ.

No motion parameters are evaluated: this reads the exported PLY frames. Static
attributes are shared once. poses[N,T,7] stores actual XYZ + wxyz at every frame;
nonzero normals add 3 channels. Ordering by Gaussian then time aids compression.
"""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from utils.gaussian_sequence import GaussianSequence


def pack_sequence(source, output):
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("Output must end in .npz")
    partial = output.with_suffix(".npz.partial")
    if output.exists() or partial.exists():
        raise FileExistsError("Refusing to overwrite an existing output or partial file")
    with GaussianSequence(source) as sequence:
        if sequence.dynamic_indices is not None:
            raise ValueError("Sparse NPZ is already packed; use GaussianSequence.frame() to extract full PLYs")
        base = sequence.frame(0)["vertex"].data.copy()
        fields = ["x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3"]
        if all(name in base.dtype.names for name in ("nx", "ny", "nz")) and any(np.any(base[name]) for name in ("nx", "ny", "nz")):
            fields += ["nx", "ny", "nz"]
        poses = np.empty((len(base), len(sequence), len(fields)), dtype=np.float32)
        for index in range(len(sequence)):
            frame = sequence.frame(index)["vertex"].data
            if frame.dtype != base.dtype:
                raise ValueError(f"Gaussian schema changed in frame {index}")
            for column, name in enumerate(fields):
                if frame.dtype[name] != np.dtype("<f4"):
                    raise ValueError("Lossless tensor packing currently requires float32 pose fields")
                poses[:, index, column] = frame[name]
                # Only mutate this disposable in-memory copy, never source PLYs.
                frame[name] = base[name]
            # One contiguous byte comparison checks ALL static fields. Comparing
            # 55 strided fields individually rereads several GB per 64 MB frame.
            if not np.array_equal(frame.view(np.uint8), base.view(np.uint8)):
                raise ValueError(f"Static Gaussian attributes changed in frame {index}")
            if index % sequence.metadata["fps"] == 0:
                print(f"Packed actual Gaussian states {index}/{len(sequence)}", flush=True)
        metadata = copy.deepcopy(sequence.metadata)
        metadata.update(storage="lossless_baked_gaussian_trajectories", pose_fields=fields,
                        poses_layout="N_gaussians,T_frames,channels", motion_is_baked=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Compressing {poses.nbytes/1e9:.3f} GB of explicit trajectories", flush=True)
        with partial.open("xb") as stream:
            np.savez_compressed(stream, poses=poses, base_vertices=base,
                                times_seconds=np.asarray([frame["time_seconds"] for frame in sequence.frames]),
                                metadata_json=np.asarray(json.dumps(metadata)))
        partial.rename(output)
    print(f"Finished: {output} ({output.stat().st_size/1e6:.1f} MB)", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack_sequence(args.sequence_path, args.output)


if __name__ == "__main__":
    main()
