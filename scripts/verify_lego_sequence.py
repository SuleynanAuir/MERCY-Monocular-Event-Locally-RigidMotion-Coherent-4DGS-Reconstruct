"""Verify a baked Lego archive independently of Blender and its exporter."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from plyfile import PlyData
from utils.gaussian_sequence import GaussianSequence, POSE_FIELDS
from utils.rigid_checkpoint import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    with GaussianSequence(args.output / 'lego_motion.npz') as sequence:
        static = np.ones(len(sequence.base_vertices), dtype=bool)
        static[sequence.dynamic_indices] = False
        for i in range(len(sequence)):
            sequence.validate_pose_frame(i)
            pose = sequence.poses[:, i]
            if not np.isfinite(pose).all() or np.any(np.linalg.norm(pose[:, 3:7], axis=-1) < 1e-8):
                raise ValueError(f'Invalid Gaussian pose at frame {i}')
        first = sequence.frame(0)['vertex'].data
        assert first.tobytes() == sequence.base_vertices.tobytes(), 'Rest state changed'
        del first
        moved = sequence.frame(100)['vertex'].data
        expected = PlyData.read(args.output / 'frame_0100.ply')['vertex'].data
        assert moved.tobytes() == expected.tobytes(), 'Independent snapshot differs'
        assert moved[static].tobytes() == sequence.base_vertices[static].tobytes(), 'Static Gaussian moved'
        for name in moved.dtype.names:
            if name not in POSE_FIELDS:
                assert np.array_equal(moved[name], sequence.base_vertices[name]), name
        digest = sha256_file(args.source)
        assert digest == sequence.metadata['source_sha256'], 'Input PLY changed'
        assert digest == sha256_file(args.output / 'point_cloud.ply'), 'Rest PLY copy differs'
        report = {'passed': True, 'frame_count': len(sequence),
                  'gaussians_per_frame': len(sequence.base_vertices),
                  'dynamic_gaussians': len(sequence.dynamic_indices), 'static_gaussians': int(static.sum()),
                  'all_pose_checksums_verified': True, 'all_poses_finite': True,
                  'rest_state_bitwise_equal': True, 'static_rows_bitwise_equal': True,
                  'non_pose_attributes_unchanged': True, 'snapshot_100_matches_archive': True,
                  'source_ply_unchanged_sha256': digest,
                  'archive_bytes': (args.output / 'lego_motion.npz').stat().st_size}
    (args.output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
