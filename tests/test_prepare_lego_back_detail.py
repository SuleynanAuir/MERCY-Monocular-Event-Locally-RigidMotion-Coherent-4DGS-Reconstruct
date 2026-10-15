import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.prepare_lego_back_detail import (
    assemble_retired, bounded_samples, recheck_inputs, validate_target_parts)
from scripts.diagnose_lego_sparse_points import sha256


class BackDetailProposalTests(unittest.TestCase):
    def motion(self):
        groups = np.zeros(150, np.int32)
        q = np.tile(np.array([1., 0., 0., 0.]), (4, 2, 1))
        t = np.zeros((4, 2, 3))
        return groups, q, t, np.array([-1, 0])

    def test_explicit_static_identity_targets(self):
        actual = validate_target_parts(np.array([0, 11, 15, 141]), *self.motion())
        np.testing.assert_array_equal(actual, [0, 11, 15, 141])
        self.assertEqual(actual.dtype, np.int32)

    def test_no_moving_deforming_platform_or_repeated_targets(self):
        for ids in ([142], [150], [-1], [11, 11], [15, 11], []):
            with self.assertRaises(ValueError):
                validate_target_parts(np.array(ids, np.int32), *self.motion())
        data = list(self.motion()); data[0][11] = 1
        with self.assertRaises(ValueError):
            validate_target_parts(np.array([11]), *data)
        with self.assertRaises(ValueError):
            validate_target_parts(np.array([11]), *self.motion(), excluded_parts=[11])

    def test_motion_checked_at_every_frame_not_only_rest(self):
        data = list(self.motion()); data[2][3, 0, 1] = 1e-4
        with self.assertRaises(ValueError):
            validate_target_parts(np.array([11]), *data)
        data = list(self.motion()); data[1][2, 0] = [0, 1, 0, 0]
        with self.assertRaises(ValueError):
            validate_target_parts(np.array([11]), *data)

    def test_retirement_exact_target_suffix_and_protected_originals(self):
        retired, suffix = assemble_retired(np.array([1, 3]), np.array([1, 3]),
            np.array([11, 180, 140, 434]), 5, 9, np.array([0, 2]), np.array([11, 140]))
        np.testing.assert_array_equal(retired, [1, 3, 5, 7])
        np.testing.assert_array_equal(suffix, [5, 7])
        self.assertEqual(retired.dtype, np.int32)

    def test_retirement_never_touches_dynamic_or_ineligible(self):
        for original, eligible, dynamic in (([1], [1], [1]), ([1], [2], [0]), ([1, 1], [1], [0])):
            with self.assertRaises(ValueError):
                assemble_retired(np.array(original), np.array(eligible), np.array([11]),
                    3, 4, np.array(dynamic), np.array([11]))
        with self.assertRaises(ValueError):
            assemble_retired(np.array([1]), np.array([1]), np.array([11]),
                3, 4, np.array([3]), np.array([11]))

    def geometry(self):
        vertices = np.array([[0., 0, 0], [.003, 0, 0], [0, .003, 0], [0, 0, .003]])
        triangles = np.array([[0, 1, 3], [0, 1, 2]], np.int32)
        return vertices, triangles, np.array([99, 11]), np.array([.5, .8]), np.array([11])

    def test_sampling_keeps_global_triangle_identity_and_actual_plane(self):
        sampled, chosen = bounded_samples(*self.geometry(), .0015)
        np.testing.assert_array_equal(chosen, [1])
        self.assertTrue(np.all(sampled['triangle_ids'] == 1))
        np.testing.assert_allclose(sampled['xyz'][:, 2], 0)
        self.assertTrue(np.all(sampled['part_ids'] == 11))

    def test_sampling_cap_is_fail_closed_and_missing_parts_rejected(self):
        with patch('scripts.prepare_lego_back_detail.MAX_NEW', 1):
            with self.assertRaisesRegex(ValueError, 'Never truncate'):
                bounded_samples(*self.geometry(), .0015)
        args = list(self.geometry()); args[-1] = np.array([11, 12])
        with self.assertRaises(ValueError):
            bounded_samples(*args, .0015)
        with self.assertRaises(ValueError):
            bounded_samples(*self.geometry(), .0001)

    def test_changed_input_hash_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'source'
            path.write_bytes(b'original')
            hashes = {'source': sha256(path)}
            recheck_inputs({'source': path}, hashes)
            path.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError, 'Input changed'):
                recheck_inputs({'source': path}, hashes)


if __name__ == '__main__':
    unittest.main()
