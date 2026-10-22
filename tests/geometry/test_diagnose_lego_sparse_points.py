import unittest

import numpy as np

from scripts.diagnostics.diagnose_lego_sparse_points import classify, decode_attributes, neighborhood_metrics, static_rows, summarize, validate_rules

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


class SparsePointDiagnosisTests(unittest.TestCase):
    def test_low_density_alone_never_selects_near_surface_point(self):
        result = classify(np.array([.001, .03, .03, .03]), np.array([1., .001, .02, .001]),
                          np.array([1., .001, .001, .02]))
        np.testing.assert_array_equal(result['conservative'], [False, False, True, True])

    def test_balanced_and_conservative_are_nested(self):
        result = classify(np.array([.02, .03, .025]), np.array([.006, .009, .02]),
                          np.array([.001, .001, .001]))
        np.testing.assert_array_equal(result['conservative'], [False, True, False])
        np.testing.assert_array_equal(result['balanced'], [True, True, True])

    def test_nonfinite_metrics_and_reversed_rules_rejected(self):
        with self.assertRaises(ValueError):
            classify(np.array([np.nan]), np.array([.1]), np.array([.1]))
        with self.assertRaises(ValueError):
            validate_rules({'mesh_distance': .01, 'knn8_distance': .01, 'maximum_scale': .01},
                           {'mesh_distance': .02, 'knn8_distance': .01, 'maximum_scale': .01})

    def test_motion_elastic_rebuilt_and_suffix_are_protected(self):
        labels = np.array([0, 1, 2, 0, 4, 0, 3, 3], np.int16)
        rows, counts = static_rows(12, 8, 6, np.array([1, 6, 7]), labels, [2], [4])
        np.testing.assert_array_equal(rows, [0, 3, 5])
        self.assertEqual(rows.dtype, np.int32)
        self.assertEqual(counts['compact_suffix_rows_protected'], 4)
        self.assertEqual(counts['rebuilt_rows_protected'], 2)
        self.assertEqual(counts['additional_label_protected_rows'], 2)

    def test_invalid_row_partition_rejected(self):
        with self.assertRaises(ValueError):
            static_rows(12, 8, 9, np.array([1]), np.zeros(8, np.int16), [])
        with self.assertRaises(ValueError):
            static_rows(12, 8, 6, np.array([1, 1]), np.zeros(8, np.int16), [])

    def test_log_attributes_are_decoded_not_used_as_lengths(self):
        dtype = [(name, 'f4') for name in ('scale_0', 'scale_1', 'scale_2', 'opacity')]
        values = np.zeros(3, dtype=dtype)
        for axis in range(3):
            values[f'scale_{axis}'] = np.log([.001, .01, .1])
        values['opacity'] = [-1000, 0, 1000]
        scale, opacity = decode_attributes(values, np.arange(3))
        np.testing.assert_allclose(scale[:, 0], [.001, .01, .1], rtol=1e-6)
        np.testing.assert_array_equal(opacity, [0, .5, 1])

    def test_nonfinite_density_report_is_json_safe(self):
        result = summarize(np.array([1., 2., np.inf]))
        self.assertEqual(result['nonfinite_count'], 1)
        self.assertEqual(result['quantiles']['0.5'], 1.5)
        self.assertEqual(summarize(np.empty(0))['quantiles'], {})

    def test_infinite_opacity_logits_are_finite_saturated_opacity_not_nan(self):
        dtype = [(name, 'f4') for name in ('scale_0', 'scale_1', 'scale_2', 'opacity')]
        values = np.zeros(2, dtype=dtype)
        values['opacity'] = [-np.inf, np.inf]
        _, opacity = decode_attributes(values, np.arange(2))
        np.testing.assert_array_equal(opacity, [0, 1])
        values['opacity'][0] = np.nan
        with self.assertRaises(ValueError):
            decode_attributes(values, np.arange(2))

    @unittest.skipIf(cKDTree is None, 'SciPy binary is unavailable for this local architecture')
    def test_nearest_surface_uses_all_rigid_parts_and_knn_excludes_self(self):
        xyz = np.column_stack((np.arange(9), np.zeros((9, 2))))
        reference = np.array([[0, 0, 0], [8, 0, 0]], dtype=float)
        distance, radius, labels = neighborhood_metrics(xyz, reference, np.array([31, 99]), workers=1, chunk_size=3)
        np.testing.assert_array_equal(distance, [0, 1, 2, 3, 4, 3, 2, 1, 0])
        self.assertEqual(labels[-1], 99)
        self.assertEqual(radius[0], 8)
        self.assertEqual(radius[4], 4)

    @unittest.skipIf(cKDTree is None, 'SciPy binary is unavailable for this local architecture')
    def test_coincident_points_are_dense_not_sparse(self):
        xyz = np.zeros((9, 3), float)
        distance, radius, _ = neighborhood_metrics(xyz, [[1, 0, 0]], np.array([0]), workers=1)
        np.testing.assert_array_equal(radius, np.zeros(9))
        result = classify(distance, radius, np.full(9, .001))
        self.assertFalse(result['conservative'].any())


if __name__ == '__main__':
    unittest.main()
