import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from utils.gaussian_sequence import GaussianSequence, FORMAT, POSE_FIELDS, SPARSE_VERSION


class SparseSequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'sequence.npz'
        self.base = np.zeros(5, dtype=[(key, '<f4') for key in POSE_FIELDS + ['opacity', 'f_dc_0']])
        self.base['x'] = np.arange(5)
        self.base['rot_0'] = 1
        self.base['opacity'][1] = np.inf
        self.indices = np.array([1, 3], dtype=np.int32)
        self.poses = np.stack([self.base[key][self.indices] for key in POSE_FIELDS], -1)[:, None].repeat(3, axis=1)
        self.poses[:, 1:, 1] = [[2, 4], [3, 6]]
        self.meta = {'format': FORMAT, 'version': SPARSE_VERSION, 'frame_count': 3, 'fps': 60,
                     'gaussians_per_frame': 5, 'pose_fields': POSE_FIELDS,
                     'base_vertices_sha256': hashlib.sha256(self.base.tobytes()).hexdigest(),
                     'frames': [{'file': f'frames/frame_{i:04d}.ply', 'time_seconds': i / 60,
                                 'pose_sha256': hashlib.sha256(self.poses[:, i].tobytes()).hexdigest()} for i in range(3)]}

    def write(self):
        np.savez_compressed(self.path, poses=self.poses, base_vertices=self.base, dynamic_indices=self.indices,
                            metadata_json=np.asarray(json.dumps(self.meta)), times_seconds=np.arange(3) / 60)

    def test_reconstructs_full_independent_frame_with_fixed_row_ids(self):
        self.write()
        with GaussianSequence(self.path) as seq:
            np.testing.assert_array_equal(seq.frame(0)['vertex'].data, self.base)
            current = seq.frame(2)['vertex'].data
            np.testing.assert_array_equal(current[[0, 2, 4]], self.base[[0, 2, 4]])
            np.testing.assert_array_equal(current['y'], [0, 4, 0, 6, 0])
            np.testing.assert_array_equal(current['opacity'], self.base['opacity'])
            for index in (-1, 3, .5):
                with self.assertRaises(IndexError):
                    seq.frame(index)

    def test_pose_corruption_rejected(self):
        self.poses[0, 1, 0] += 1
        self.write()
        with GaussianSequence(self.path) as seq:
            with self.assertRaisesRegex(ValueError, 'checksum'):
                seq.frame(1)

    def test_base_corruption_rejected(self):
        self.base['x'][0] += 1
        self.write()
        with self.assertRaisesRegex(ValueError, 'checksum'):
            GaussianSequence(self.path)

    def test_invalid_dynamic_indices_rejected(self):
        for indices in ([1, 1], [3, 1], [-1, 3], [1, 5]):
            self.indices = np.array(indices, np.int32)
            self.write()
            with self.assertRaisesRegex(ValueError, 'indices'):
                GaussianSequence(self.path)
        self.indices = np.array([1., 3.])
        self.write()
        with self.assertRaisesRegex(ValueError, 'indices'):
            GaussianSequence(self.path)

    def test_bad_shapes_rejected(self):
        self.poses = self.poses[:1]
        self.write()
        with self.assertRaisesRegex(ValueError, 'shapes'):
            GaussianSequence(self.path)


if __name__ == '__main__':
    unittest.main()
