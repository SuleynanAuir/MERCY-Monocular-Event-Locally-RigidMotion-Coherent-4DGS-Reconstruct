import unittest

import numpy as np

from scripts.prepare_lego_roof_rebuild import assemble_retired, gaussian_attributes, geometry_retirement, observed_gray
from scripts.prepare_lego_surface_colors import thin_axis_normals


class RoofPreparationTests(unittest.TestCase):
    def test_retirement_protects_nonroof_geometry_ambiguity_and_footprint(self):
        points = np.zeros((5, 3)); points[4, 0] = 2
        chosen, stats = geometry_retirement(points, np.array([.003]*4+[.001]),
            np.array([.02, .004, .02, .002, .02]), np.array([.001, .001, .01, .0001, .001]),
            np.array([[-1, -1, -1], [1, 1, 1]]))
        np.testing.assert_array_equal(chosen, [True, False, False, False, False])
        self.assertEqual(stats['selected_original_static'], 1)

    def test_retirement_does_not_depend_on_legacy_part_labels(self):
        chosen, _ = geometry_retirement(np.zeros((1, 3)), np.array([.001]), np.array([.02]),
            np.array([.001]), np.array([[-1]*3, [1]*3]))
        self.assertTrue(chosen[0])

    def test_invalid_retirement_metrics_rejected(self):
        with self.assertRaises(ValueError):
            geometry_retirement(np.zeros((1, 3)), np.array([np.nan]), np.array([.02]),
                np.array([.001]), np.array([[-1]*3, [1]*3]))

    def test_suffix_retires_all_roof_only_and_protects_dynamic(self):
        retired, suffix = assemble_retired(np.array([1, 3]), np.array([1, 3, 4]),
            np.array([11, 99, 15, 33]), 5, 9, np.array([0, 2]))
        np.testing.assert_array_equal(retired, [1, 3, 5, 7])
        np.testing.assert_array_equal(suffix, [5, 7])
        self.assertEqual(retired.dtype, np.int32)
        with self.assertRaises(ValueError):
            assemble_retired(np.array([1]), np.array([1]), np.array([11]), 3, 4, np.array([1]))

    def test_noneligible_or_duplicate_retirement_rejected(self):
        with self.assertRaises(ValueError):
            assemble_retired(np.array([1]), np.array([2]), np.array([11]), 3, 4, np.array([0]))
        with self.assertRaises(ValueError):
            assemble_retired(np.array([1, 1]), np.array([1]), np.array([11]), 3, 4, np.array([0]))

    def test_surface_attributes_map_local_z_to_actual_triangle_normal(self):
        normals = np.array([[0., 0, 1], [1., 0, 0], [0., 0, -1]])
        q, scale, opacity = gaussian_attributes(normals)
        np.testing.assert_allclose(thin_axis_normals(q, scale), normals, atol=1e-6)
        np.testing.assert_allclose(np.exp(scale), np.tile([.00105, .00105, .00015], (3, 1)), rtol=1e-6)
        np.testing.assert_allclose(1/(1+np.exp(-opacity)), .95, atol=1e-7)
        self.assertEqual(q.dtype, np.float32)

    def observation_fixture(self):
        camera = {'height': 4, 'width': 4, 'fovy_degrees': 90., 'pose': np.eye(4).tolist()}
        xyz = np.array([[0., 0, -1.]])
        normal = np.array([[0., 0, 1.]])
        parts = np.array([11])
        masks = np.full((3, 4, 4), 11, np.int16)
        depth = np.ones((3, 4, 4), np.float32)
        frames = np.array([0, 1, 2])
        images = np.stack([np.full((4, 4), n, np.uint8) for n in (0, 0, 255)])
        return xyz, normal, parts, masks, depth, frames, images, camera

    def test_true_black_is_preserved_and_temporal_median_not_brightness_gated(self):
        color, support, stats = observed_gray(*self.observation_fixture())
        np.testing.assert_array_equal(color, [0.])
        np.testing.assert_array_equal(support, [3])
        self.assertEqual(stats['observed_points'], 1)

    def test_occlusion_samepart_mask_and_minimum_support_required(self):
        args = list(self.observation_fixture())
        args[4][0] = .5
        args[3][1, 1, 1] = 52
        color, support, _ = observed_gray(*args)
        self.assertTrue(np.isnan(color[0]))
        np.testing.assert_array_equal(support, [1])

    def test_wrong_face_normal_and_elastic_occlusion_rejected(self):
        args = list(self.observation_fixture()); args[1] *= -1
        color, support, _ = observed_gray(*args)
        np.testing.assert_array_equal(support, [0])
        args = self.observation_fixture()
        boxes = [[[[1., 1.], [2., 2.]]]]*3
        color, support, stats = observed_gray(*args, elastic_boxes=boxes)
        np.testing.assert_array_equal(support, [0])
        self.assertEqual(stats['elastic_bbox_rejected_sample_observations'], 3)

    def test_slanted_plane_depth_compared_at_nearest_pixel_ray(self):
        args = list(self.observation_fixture())
        args[1] = np.array([[1., 0, 1.]])/np.sqrt(2)
        # Center is z=-1. Nearest pixel x=2 has ray x=.25, so z depth=4/3.
        args[4][:, 2, 2] = 4/3
        color, support, _ = observed_gray(*args)
        np.testing.assert_array_equal(support, [3])

    def test_invalid_support_and_repeated_frame_ids_rejected(self):
        with self.assertRaises(ValueError):
            observed_gray(*self.observation_fixture(), minimum_support=4)
        args = list(self.observation_fixture()); args[5] = np.array([0, 0, 1])
        with self.assertRaises(ValueError):
            observed_gray(*args)


if __name__ == '__main__':
    unittest.main()
