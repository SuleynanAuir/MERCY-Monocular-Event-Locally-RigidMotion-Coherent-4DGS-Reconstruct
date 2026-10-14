import unittest

import numpy as np

from scripts.restore_lego_occluded_parts import RESTORED_FIELDS, restore_geometry, select_original_support, union_dynamic


class OriginalSupportRestorationTests(unittest.TestCase):
    def setUp(self):
        self.camera = {'pose': np.eye(4).tolist(), 'fovy_degrees': 45., 'width': 3, 'height': 3}
        self.xyz = np.array([[0., 0., -2.], [0., 0., -2.004], [0., 0., -2.1], [10., 0., -2.1], [0., 0., 2.]])
        self.labels = np.array([229, 235, 100, 237, 229])
        self.depth, self.alpha = np.full((3, 3), 2.), np.ones((3, 3))

    def test_only_behind_confident_front_target_parts_selected(self):
        actual, report = select_original_support(self.xyz, self.labels, [229, 235, 237], self.camera, self.depth, self.alpha)
        np.testing.assert_array_equal(actual, [False, True, False, False, False])
        self.assertEqual(report['selected_original_rows'], 1)

    def test_all_ablation_is_explicit_and_includes_unseen_rows(self):
        actual, report = select_original_support(self.xyz, self.labels, [229, 235, 237], self.camera, restore_all=True)
        np.testing.assert_array_equal(actual, [True, True, False, True, True])
        self.assertEqual(report['mode'], 'all_original_support_ablation')

    def test_low_alpha_invalid_depth_and_negative_margin_rejected(self):
        for depth, alpha in ((self.depth, self.alpha*.5), (self.depth*np.nan, self.alpha)):
            selected, _ = select_original_support(self.xyz, self.labels, [235], self.camera, depth, alpha)
            self.assertFalse(selected.any())
        with self.assertRaises(ValueError):
            select_original_support(self.xyz, self.labels, [235], self.camera, self.depth, self.alpha, margin=-.1)

    def test_geometry_reset_preserves_appearance_rotation_and_other_rows(self):
        names = RESTORED_FIELDS+('opacity', 'rot_0', 'f_dc_0')
        original = np.zeros(3, dtype=[(name, 'f4') for name in names])
        current = original.copy()
        for field in names:
            original[field] = [1, 2, 3]
            current[field] = [10, 20, 30]
        prior = current.copy()
        restore_geometry(current, original, np.array([False, True, False]))
        for field in RESTORED_FIELDS:
            np.testing.assert_array_equal(current[field], [10, 2, 30])
        for field in ('opacity', 'rot_0', 'f_dc_0'):
            np.testing.assert_array_equal(current[field], prior[field])

    def test_dynamic_union_never_removes_previous_members(self):
        np.testing.assert_array_equal(union_dynamic(np.array([0, 3]), np.array([False, True, False, False, True])), [0, 1, 3, 4])
        with self.assertRaises(ValueError):
            union_dynamic(np.array([-1]), np.zeros(3, dtype=bool))


if __name__ == '__main__':
    unittest.main()
