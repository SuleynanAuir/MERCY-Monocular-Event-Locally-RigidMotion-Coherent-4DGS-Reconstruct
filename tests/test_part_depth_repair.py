import unittest

import numpy as np

from scripts.repair_lego_part_depth import erode_mask, estimate_radial_scale, projection_footprint, radial_repair


class PartDepthRepairTests(unittest.TestCase):
    def test_erodes_one_pixel_and_ignores_low_evidence(self):
        mask = np.ones((5, 5), bool)
        self.assertEqual(int(erode_mask(mask).sum()), 9)
        scale, report = estimate_radial_scale(np.ones((5, 5)) * 1.1, np.ones((5, 5)), erode_mask(mask))
        self.assertEqual(scale, 1)
        self.assertEqual(report['status'], 'insufficient_evidence_kept_unchanged')

    def test_median_scale_and_clamp(self):
        scale, report = estimate_radial_scale(np.ones(20) * 1.11, np.ones(20), np.ones(20, bool))
        self.assertAlmostEqual(scale, 1.11)
        self.assertFalse(report['clamped'])
        scale, report = estimate_radial_scale(np.ones(20) * 1.8, np.ones(20), np.ones(20, bool))
        self.assertEqual(scale, 1.25)
        self.assertTrue(report['clamped'])

    def test_ray_and_footprint_invariant_for_translated_camera(self):
        pose = np.eye(4); pose[:3, 3] = [.2, .3, 2]
        camera = {'pose': pose.tolist(), 'height': 640, 'fovy_degrees': 40}
        xyz = np.array([[.1, .2, .5], [.4, .3, .6]])
        scales = np.log(np.array([[.001, .002, 1e-8], [.002, .001, 1e-8]]))
        repaired_xyz, repaired_scales = radial_repair(xyz, scales, pose[:3, 3], 1.1)
        before = projection_footprint(xyz, scales, camera)
        after = projection_footprint(repaired_xyz, repaired_scales, camera)
        np.testing.assert_allclose(before[1], after[1], atol=1e-11)
        np.testing.assert_allclose(before[2], after[2], atol=1e-11)
        self.assertAlmostEqual(np.linalg.norm(repaired_xyz[1] - repaired_xyz[0]),
                               1.1 * np.linalg.norm(xyz[1] - xyz[0]))


if __name__ == '__main__':
    unittest.main()
