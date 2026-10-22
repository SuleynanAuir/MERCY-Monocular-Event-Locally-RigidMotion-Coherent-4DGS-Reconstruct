import unittest

import numpy as np

from scripts.appearance.prepare_lego_surface_colors import LIGHT, color_change_selection, luminance, part_color_table, shaded_gray, thin_axis_normals, validate_guard_arrays


class SurfaceColorPreparationTests(unittest.TestCase):
    def test_dark_preservation_blocks_only_brightening_below_strict_boundary(self):
        old = np.array([.05, .10, .12, .50])
        target = np.array([.80, .01, .70, .60])
        before = target.copy()
        chosen, reasons = color_change_selection(old, target, np.zeros(4, bool), .08, .12)
        np.testing.assert_array_equal(chosen, [False, True, True, True])
        np.testing.assert_array_equal(target, before)
        self.assertEqual(reasons['blocked_dark_brightening'], 1)
        self.assertEqual(reasons['selected_dark_rows_darkened'], 1)

    def test_static_reference_rule_rejects_nonroot_without_reassignment(self):
        chosen, reasons = color_change_selection(np.array([.3, .3]), np.array([.8, .8]),
                                                 np.array([False, True]), .08, static_reference_only=True)
        np.testing.assert_array_equal(chosen, [True, False])
        self.assertEqual(reasons['blocked_nonroot_reference'], 1)
        self.assertEqual(reasons['selected_nonroot_reference_rows'], 0)

    def test_safety_rule_overlap_count_is_not_double_counted(self):
        old = np.array([.05, .05, .3, .3])
        target = np.array([.8, .8, .8, .8])
        chosen, reasons = color_change_selection(old, target, np.array([True, False, True, False]), .08, .12, True)
        np.testing.assert_array_equal(chosen, [False, False, False, True])
        self.assertEqual(reasons['blocked_dark_brightening'], 2)
        self.assertEqual(reasons['blocked_nonroot_reference'], 2)
        self.assertEqual(reasons['blocked_by_both_safety_rules'], 1)
        self.assertEqual(reasons['blocked_by_any_safety_rule'], 3)

    def test_optional_safety_rules_preserve_previous_default_selection(self):
        chosen, reasons = color_change_selection(np.array([0., .5, .5]), np.array([.9, .55, .2]),
                                                 np.ones(3, bool), .08)
        np.testing.assert_array_equal(chosen, [True, False, True])
        self.assertEqual(reasons['blocked_by_any_safety_rule'], 0)

    def test_original_gray_uses_renderer_luminance_not_red_only(self):
        rgb = np.array([[.2, .4, .6], [.7, .7, .7]], np.float32)
        result = luminance(rgb)
        np.testing.assert_allclose(result, [.2*.299+.4*.587+.6*.114, .7], atol=1e-7)

    def test_guard_rows_match_actual_full_candidate_decision(self):
        result = validate_guard_arrays(np.array([1, 5]), np.array([1, 3, 5]),
                                       np.array([0., .01, .0001]), .0005)
        np.testing.assert_array_equal(result, [1, 5])
        self.assertEqual(result.dtype, np.int32)
        with self.assertRaises(ValueError):
            validate_guard_arrays(np.array([1, 3]), np.array([1, 3, 5]), np.array([0., .01, .0001]), .0005)

    def test_duplicate_guard_rows_and_nonfinite_maxima_rejected(self):
        with self.assertRaises(ValueError):
            validate_guard_arrays(np.array([1, 1]), np.array([1]), np.array([0.]), .0005)
        with self.assertRaises(ValueError):
            validate_guard_arrays(np.array([1]), np.array([1]), np.array([np.nan]), .0005)

    def test_quaternion_uses_rotation_column_not_row_and_normalizes(self):
        q = np.array([[1., 0, 0, 0], [np.sqrt(2), 0, np.sqrt(2), 0], [0, 1, 0, 0]])
        normal = thin_axis_normals(q, np.tile([-7., -7., -18.], (3, 1)))
        np.testing.assert_allclose(normal, [[0, 0, 1], [1, 0, 0], [0, 0, -1]], atol=1e-7)

    def test_zero_quaternion_and_wrong_thin_axis_rejected(self):
        with self.assertRaises(ValueError):
            thin_axis_normals([[0, 0, 0, 0]], [[-7, -7, -18]])
        with self.assertRaises(ValueError):
            thin_axis_normals([[1, 0, 0, 0]], [[-18, -7, -7]])

    def test_completion_median_filter_strict_bounds_and_minimum_four(self):
        gray = np.array([0., .04, .1, .2, .4, .8, .99, 1.], np.float32)
        table, records = part_color_table(gray, np.zeros(8, np.int32), np.array([0]), np.array([.7]), np.array([0]))
        self.assertAlmostEqual(float(table[0]), .3, places=6)
        self.assertEqual(records['0']['qualifying_parent_rows'], 4)
        self.assertEqual(records['0']['source'], 'same_part_parent_gray_median')

    def test_less_than_four_parent_values_uses_exact_material(self):
        table, records = part_color_table(np.array([.2, .3, .4]), np.array([0, 0, 0]),
                                          np.array([0, 0]), np.array([.75, .75]), np.array([0]))
        self.assertEqual(table[0], .75)
        self.assertEqual(records['0']['source'], 'unobserved_Blender_material_prior')
        with self.assertRaises(ValueError):
            part_color_table(np.array([.2]), np.array([0]), np.array([0, 0]), np.array([.2, .7]), np.array([0]))

    def test_shading_is_two_sided_and_matches_completion_formula(self):
        normal = np.array([[0, 0, 1], [0, 0, -1]], np.float32)
        gray = np.array([.7, .7], np.float32)
        result = shaded_gray(gray, normal)
        np.testing.assert_array_equal(result[:1], result[1:])
        expected = gray.copy(); expected *= .8+.2*np.abs(normal @ LIGHT)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(gray, np.array([.7, .7], np.float32))

    def test_shading_clips_without_modifying_input_arrays(self):
        normal = np.vstack((LIGHT, LIGHT)).astype(np.float32)
        gray = np.array([0., 1.], np.float32)
        before = normal.copy()
        np.testing.assert_allclose(shaded_gray(gray, normal), [.06, .9], atol=1e-7)
        np.testing.assert_array_equal(normal, before)
        np.testing.assert_array_equal(gray, [0, 1])


if __name__ == '__main__':
    unittest.main()
