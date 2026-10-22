import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    'diagnose_depth', Path(__file__).resolve().parents[2] / 'scripts/diagnostics/diagnose_lego_depth.py')
depth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth)


class DepthDiagnosticTest(unittest.TestCase):
    def test_normalized_ray_distance_is_not_camera_z(self):
        vectors = np.array([[[0., 0., -1.], [1., 0., -1.]]])
        distance = np.array([[2., 2.]])
        z = depth.ray_distance_to_z(distance, vectors, .4)
        np.testing.assert_allclose(z, [[.8, .8 / np.sqrt(2)]])

    def test_ray_to_z_reconstructs_distance(self):
        vectors, _ = depth.camera_vectors(640, 640, 39.597754)
        distances = np.full((640, 640), 4.)
        z = depth.ray_distance_to_z(distances, vectors, .40884)
        np.testing.assert_allclose(np.linalg.norm(vectors * z[..., None], axis=-1),
                                   distances * .40884, atol=1e-12)

    def test_same_ray_depths_have_no_initial_pixel_displacement(self):
        vectors, focal = depth.camera_vectors(20, 20, 40)
        p1, p2 = vectors * 1.5, vectors * 1.8
        uv1 = focal * p1[..., :2] / -p1[..., 2:]
        uv2 = focal * p2[..., :2] / -p2[..., 2:]
        np.testing.assert_allclose(uv1, uv2, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
