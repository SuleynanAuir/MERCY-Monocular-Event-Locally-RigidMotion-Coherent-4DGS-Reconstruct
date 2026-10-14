import unittest

import numpy as np

from utils.mesh_surface_sampling import depth_buffer, normals_to_wxyz, projection, sample_mesh


class MeshSurfaceSamplingTests(unittest.TestCase):
    def mesh(self, spacing=.15, **kwargs):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        return sample_mesh(vertices, np.array([[0, 1, 2]], np.int32),
                           np.array([7], np.int32), np.array([.6], np.float32),
                           [], spacing, **kwargs)

    def test_triangle_samples_stay_on_surface_and_deterministic(self):
        a, b = self.mesh(), self.mesh()
        self.assertGreater(len(a['xyz']), 10)
        np.testing.assert_allclose(a['xyz'][:, 2], 0)
        self.assertTrue(np.all(a['xyz'][:, :2] >= 0))
        self.assertTrue(np.all(a['xyz'][:, :2].sum(axis=1) <= 1.000001))
        np.testing.assert_allclose(a['xyz'][0], [1/3, 1/3, 0])
        np.testing.assert_allclose(a['normals'], np.tile([0, 0, 1], (len(a['xyz']), 1)))
        for name in ('xyz', 'normals', 'part_ids', 'material_gray', 'triangle_ids'):
            np.testing.assert_array_equal(a[name], b[name])
        self.assertEqual(a['stats'], b['stats'])
        self.assertEqual(a['stats']['generated_samples'], 23)

    def test_area_controls_density(self):
        a, b = self.mesh(.2), self.mesh(.1)
        self.assertEqual(a['stats']['generated_samples'], 13)
        self.assertEqual(b['stats']['generated_samples'], 50)
        self.assertGreater(len(b['xyz']), len(a['xyz']) * 2)

    def test_tiny_triangles_get_a_centroid_and_degenerate_gets_none(self):
        vertices = np.array([[0, 0, 0], [1e-9, 0, 0], [0, 1e-9, 0]], np.float32)
        result = sample_mesh(vertices, np.array([[0, 1, 2], [0, 0, 0]]),
                             np.array([0, 0]), np.array([.5, .5]), [], 1.)
        self.assertEqual(len(result['xyz']), 1)
        self.assertEqual(result['stats']['degenerate_triangles'], 1)
        np.testing.assert_allclose(result['xyz'][0], vertices.mean(axis=0))

    def test_excluded_parts_and_cross_part_coincident_points(self):
        vertices = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0]], np.float32)
        triangles = np.tile([0, 1, 2], (3, 1))
        result = sample_mesh(vertices, triangles, np.array([2, 3, 4]), np.array([.1, .2, .3]), [3], 1.)
        np.testing.assert_array_equal(result['part_ids'], [2, 4])
        np.testing.assert_array_equal(result['triangle_ids'], [0, 2])
        self.assertEqual(result['stats']['excluded_triangles'], 1)

    def test_opposite_thin_walls_are_not_merged(self):
        vertices = np.array([[.1, .1, .1], [.2, .1, .1], [.1, .2, .1],
                             [.1, .1, .11], [.2, .1, .11], [.1, .2, .11]], np.float32)
        result = sample_mesh(vertices, np.array([[0, 1, 2], [3, 5, 4]]),
                             np.array([0, 0]), np.array([.2, .8]), [], 1.)
        self.assertEqual(len(result['xyz']), 2)
        np.testing.assert_array_equal(result['normals'][:, 2], [1, -1])
        np.testing.assert_allclose(result['xyz'][:, 2], [.1, .11])

    def test_voxel_selection_never_averages_through_a_hole(self):
        # Two disjoint triangles fit in one voxel. Their midpoint lies in a gap;
        # merging must select a real triangle point, not their mean.
        vertices = np.array([[.1, .1, .1], [.2, .1, .1], [.1, .2, .1],
                             [.7, .7, .1], [.8, .7, .1], [.7, .8, .1]], np.float32)
        result = sample_mesh(vertices, np.array([[0, 1, 2], [3, 4, 5]]),
                             np.array([0, 0]), np.array([.2, .8]), [], 1.)
        self.assertEqual(len(result['xyz']), 1)
        np.testing.assert_allclose(result['xyz'][0], vertices[:3].mean(axis=0))
        self.assertEqual(result['triangle_ids'][0], 0)
        self.assertAlmostEqual(float(result['material_gray'][0]), .2)

    def test_cap_raises_before_sampling_not_truncates(self):
        with self.assertRaisesRegex(ValueError, 'no silent truncation'):
            self.mesh(.01, max_samples=100)
        with self.assertRaises(ValueError):
            self.mesh(.1, max_samples=True)

    def test_empty_mesh(self):
        result = sample_mesh(np.empty((0, 3)), np.empty((0, 3), np.int32),
                             np.empty(0, np.int32), np.empty(0), [], .1)
        self.assertEqual(result['xyz'].shape, (0, 3))
        self.assertEqual(result['triangle_ids'].dtype, np.int32)
        self.assertEqual(result['stats']['merged_samples'], 0)

    def test_invalid_mesh_inputs(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        valid = [vertices, np.array([[0, 1, 2]]), np.array([0]), np.array([.5]), [], .1]
        cases = [(0, np.full((3, 3), np.nan)), (1, np.array([[0., 1., 2.]])),
                 (1, np.array([[0, 1, 3]])), (1, np.array([[-1, 1, 2]])),
                 (2, np.array([-1])), (2, np.array([True])), (3, np.array([1.1])),
                 (3, np.array([np.nan])), (4, [1.2]), (5, 0), (5, np.inf), (5, True)]
        for index, value in cases:
            with self.subTest(index=index, value=value):
                args = valid.copy(); args[index] = value
                with self.assertRaises(ValueError):
                    sample_mesh(*args)

    def test_normals_quaternions_include_both_poles(self):
        normals = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, -2, 0],
                            [1, 2, -3], [1e-8, 0, -1], [1e-8, 0, 1]], np.float64)
        q = normals_to_wxyz(normals)
        self.assertEqual(q.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-7)
        w, x, y, z = q.astype(np.float64).T
        rotated_z = np.stack((2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)), axis=1)
        np.testing.assert_allclose(rotated_z, normals/np.linalg.norm(normals, axis=1, keepdims=True), atol=2e-7)
        self.assertEqual(normals_to_wxyz(np.empty((0, 3))).shape, (0, 4))
        with self.assertRaises(ValueError):
            normals_to_wxyz([[0, 0, 0]])

    def test_projection_matches_pixel_center_rays_and_pose(self):
        width, height, fovy = 7, 5, 63.
        yy, xx = np.mgrid[:height, :width]
        tangent = np.tan(np.deg2rad(fovy)/2)
        rays = np.stack(((2*(xx+.5)/width-1)*tangent*width/height,
                         (1-2*(yy+.5)/height)*tangent, -np.ones_like(xx)), axis=-1).reshape(-1, 3)
        pose = np.array([[0., 0., 1., 3.], [0., 1., 0., -2.], [-1., 0., 0., 5.], [0., 0., 0., 1.]])
        points = (rays*2) @ pose[:3, :3].T + pose[:3, 3]
        xy, depth = projection(points, pose, width, height, fovy)
        np.testing.assert_allclose(xy, np.stack((xx.ravel(), yy.ravel()), axis=1), atol=1e-12)
        np.testing.assert_allclose(depth, 2)
        _, center_depth = projection([[0, 0, 1], [0, 0, 0]], np.eye(4), 3, 3, 90)
        np.testing.assert_array_equal(center_depth, [-1, 0])

    def test_depth_buffer_nearest_positive_and_bounded_footprint(self):
        points = [[0, 0, -3], [0, 0, -2], [0, 0, 1], [0, 0, 0], [100, 0, -1]]
        sharp = depth_buffer(points, np.eye(4), 9, 9, 90, radius=0)
        self.assertEqual(np.isfinite(sharp).sum(), 1)
        self.assertEqual(sharp[4, 4], 2)
        expanded = depth_buffer(points, np.eye(4), 9, 9, 90, radius=1)
        self.assertEqual(np.isfinite(expanded).sum(), 9)
        self.assertTrue(np.isinf(expanded[0, 0]))
        np.testing.assert_allclose(expanded[3:6, 3:6], 2)

    def test_projection_and_depth_input_validation(self):
        for width, height, fovy in ((0, 3, 90), (3, True, 90), (3, 3, 0), (3, 3, 180)):
            with self.assertRaises(ValueError):
                projection([[0, 0, -1]], np.eye(4), width, height, fovy)
        with self.assertRaises(ValueError):
            projection([[0, 0, -1]], np.zeros((4, 4)), 3, 3, 90)
        with self.assertRaises(ValueError):
            depth_buffer([[0, 0, -1]], np.eye(4), 3, 3, 90, radius=2)


if __name__ == '__main__':
    unittest.main()
