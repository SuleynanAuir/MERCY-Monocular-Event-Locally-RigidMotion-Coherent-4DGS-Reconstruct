import unittest

import numpy as np

from scripts.motion.rebind_lego_coherent import attachment_limit, initial_registration, project_mask, validate_correspondence


class CoherentBindingTests(unittest.TestCase):
    def test_large_screen_gate_does_not_get_clamped_back_to_geometry_limit(self):
        self.assertEqual(attachment_limit(.1, 'screen', .5), .5)
        self.assertEqual(attachment_limit(.1, 'screen', .065), .1)
        self.assertEqual(attachment_limit(.1, 'geometry', .5), .1)
        self.assertTrue(.35 <= attachment_limit(.1, 'screen', .5))

    def test_requires_camera_registration(self):
        with self.assertRaises(ValueError):
            initial_registration({'registration': np.eye(4).tolist()})
        skew = np.eye(4); skew[0, 0] = 2
        with self.assertRaises(ValueError):
            initial_registration({'initial_registration': skew.tolist()})
        np.testing.assert_array_equal(initial_registration({'initial_registration': np.eye(4).tolist()}), np.eye(4))

    def test_screen_assigns_both_depth_layers_not_behind_camera(self):
        camera = {'pose': np.eye(4).tolist(), 'fovy_degrees': 90}
        mask = np.full((3, 3), -1, dtype=np.int16); mask[1, 1] = 2
        xyz = np.array([[0, 0, -1], [0, 0, -4], [0, 0, 1], [10, 0, -1]])
        np.testing.assert_array_equal(project_mask(xyz, camera, mask), [2, 2, -1, -1])

    def test_elastic_indices_are_global_surface_offsets(self):
        parts = {'point_parts': np.array([0, 0, 1, 1]), 'part_offsets': np.array([0, 2, 4]),
                 'deformed_1': np.zeros((3, 2, 3))}
        meta = {'deforming_objects': [1]}
        validate_correspondence(np.array([0, 1]), np.array([1, 3]), parts, meta)
        with self.assertRaises(ValueError):
            validate_correspondence(np.array([0, 1]), np.array([1, 1]), parts, meta)


if __name__ == '__main__':
    unittest.main()
