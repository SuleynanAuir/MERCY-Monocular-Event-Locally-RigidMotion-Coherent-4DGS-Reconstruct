import unittest
import numpy as np
from utils.mesh_surface import denoise_surface


class SurfaceDenoisingTests(unittest.TestCase):
    def grid(self):
        x, y = np.meshgrid(np.arange(21), np.arange(21))
        vertices = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]).astype(float)
        faces = []
        for j in range(20):
            for i in range(20):
                if 8 <= i < 12 and 8 <= j < 12:
                    continue  # A real through-hole, not a noise patch.
                a = j * 21 + i
                faces.extend([[a, a + 1, a + 21], [a + 1, a + 22, a + 21]])
        return vertices, np.asarray(faces)

    def test_reduces_plane_noise_preserving_hole_and_displacement_limits(self):
        vertices, faces = self.grid()
        vertices[:, 2] = np.random.default_rng(7).normal(0, .2, len(vertices))
        before = faces.copy()
        actual, _, report = denoise_surface(vertices, faces, np.ones_like(vertices), 1)
        used = np.unique(faces)
        self.assertLess(np.std(actual[used, 2]), np.std(vertices[used, 2]) * .7)
        np.testing.assert_array_equal(faces, before)
        self.assertLessEqual(report['max_displacement'], 2 + 1e-6)
        self.assertLessEqual(report['boundary_max_displacement'], 1 + 1e-6)
        self.assertLess(report['laplacian_after'], report['laplacian_before'])

    def test_color_noise_decreases_without_blending_black_white_edges(self):
        vertices, faces = self.grid()
        white = vertices[:, 0] >= 10
        colors = np.full_like(vertices, .05)
        colors[white] = .95
        colors += np.random.default_rng(4).normal(0, .025, colors.shape)
        _, actual, _ = denoise_surface(vertices, faces, colors, 1, iterations=0)
        used = np.unique(faces)
        for mask in (white[used], ~white[used]):
            self.assertLess(actual[used][mask].std(), colors[used][mask].std())
        self.assertGreater(actual[white].min(), .8)
        self.assertLess(actual[~white].max(), .2)

    def test_disconnected_nearby_surfaces_do_not_exchange_colors(self):
        vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        vertices = np.concatenate([vertices, vertices + [0, 0, .001]])
        colors = np.concatenate([np.zeros((3, 3)), np.ones((3, 3))])
        _, actual, _ = denoise_surface(vertices, np.array([[0, 1, 2], [3, 4, 5]]), colors, 1)
        np.testing.assert_allclose(actual, colors)

    def test_irregular_thin_triangles_keep_their_orientation(self):
        vertices, faces = self.grid()
        vertices += np.random.default_rng(5).normal(0, .6, vertices.shape)
        actual, _, report = denoise_surface(vertices, faces, np.ones_like(vertices), 1)
        a, b = vertices[faces], actual[faces]
        before = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
        after = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
        self.assertGreater(report['guarded_vertices'], 0)
        self.assertTrue(np.all(np.sum(before * after, axis=1) >= .019 * np.sum(before ** 2, axis=1)))


if __name__ == '__main__':
    unittest.main()
