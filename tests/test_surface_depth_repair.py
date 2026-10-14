import unittest
import numpy as np
from scripts.repair_lego_surface_depth import sample_part_depth, agreeing_motion_rows, translated_column_depth


class SurfaceDepthRepairTests(unittest.TestCase):
    def test_depth_translation_keeps_column_thickness_and_order(self):
        original = np.array([1.4, 1.7])
        repaired = translated_column_depth(original, np.array([1.6, 1.6]), np.array([1.4, 1.4]))
        np.testing.assert_allclose(repaired, [1.6, 1.9])
        np.testing.assert_allclose(np.diff(repaired), np.diff(original))

    def test_bilinear_never_interpolates_across_part_boundary(self):
        mask = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        depth = np.array([[1., 2., 9.], [3., 4., 9.], [3., 4., 9.]])
        labels, z, valid = sample_part_depth(np.array([[.5, .5], [1.5, .5], [-1, 0]]), mask, depth)
        self.assertAlmostEqual(z[0], 2.5)
        np.testing.assert_array_equal(valid, [True, False, False])
        np.testing.assert_array_equal(labels, [0, 1, -1])

    def test_group_agreement_static_and_elastic_guards(self):
        groups = np.array([0, 1, 1, 2, 0])
        moving = np.array([False, True, True, True, False])
        mask = agreeing_motion_rows(np.array([1, 1, 0, 3, 1]), np.array([2, 3, 1, 3, -1]),
                                    groups, moving, [3], np.ones(5, bool))
        np.testing.assert_array_equal(mask, [True, False, False, False, False])


if __name__ == '__main__':
    unittest.main()
