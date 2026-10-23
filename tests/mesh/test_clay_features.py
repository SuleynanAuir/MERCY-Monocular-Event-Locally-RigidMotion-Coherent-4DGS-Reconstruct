import unittest
import numpy as np
from utils.clay_features import reconstruct_features, preserve_orientation


def grid(n=51):
    x, y = np.meshgrid(np.linspace(-.25, .25, n), np.linspace(-.25, .25, n))
    faces = []
    for j in range(n-1):
        for i in range(n-1):
            a = j*n+i
            faces.extend([[a, a+1, a+n], [a+1, a+n+1, a+n]])
    return x.ravel(), y.ravel(), np.asarray(faces)


class ClayFeatureTests(unittest.TestCase):
    def test_serialization_cannot_collapse_or_invert_small_triangles(self):
        original = np.array([[1., 0, 0], [1.00000024, 0, 0], [1., .001, 0]], np.float32)
        candidate = original.astype(float)
        candidate[1, 0] = 1.00000003  # Valid in double, collapses in float32.
        out, changed = preserve_orientation(original, candidate, np.array([[0, 1, 2]]))
        self.assertGreater(changed, 0)
        self.assertGreater(np.cross(out[1]-out[0], out[2]-out[0])[2], 0)

    def test_denoises_plane_without_shrinking_tangential_extent(self):
        x, y, f = grid()
        v = np.column_stack([x, y, np.random.default_rng(7).normal(0, .002, len(x))])
        out, normals, report = reconstruct_features(v, f, .045, .008)
        inside = (abs(x) < .2) & (abs(y) < .2)
        self.assertLess(out[inside, 2].std(), v[inside, 2].std()*.3)
        self.assertLess(np.max(abs(np.ptp(out[:, :2], axis=0)-.5)), .001)
        self.assertLessEqual(report['max_displacement'], .008001)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)

    def test_ninety_degree_crease_is_not_rounded_into_a_blur(self):
        x, y, f = grid()
        v = np.column_stack([x, y, abs(x)+np.random.default_rng(8).normal(0, .002, len(x))])
        out, normals, _ = reconstruct_features(v, f, .045, .008)
        inside = (abs(x) < .2) & (abs(y) < .2)
        self.assertLess((out[inside, 2]-abs(out[inside, 0])).std(), .0008)
        # The ideal apex is z=0. A plain spatial smoother raises/rounds it.
        self.assertLess(abs(out[(abs(x) < .001) & (abs(y) < .2), 2].mean()), .002)
        left = normals[(x < -.03) & inside].mean(0)
        right = normals[(x > .03) & inside].mean(0)
        angle = np.degrees(np.arccos(np.clip(left@right/(np.linalg.norm(left)*np.linalg.norm(right)), -1, 1)))
        self.assertGreater(angle, 87)
        self.assertLess(angle, 93)

    def test_oppositely_oriented_thin_sheets_do_not_merge(self):
        x, y, f = grid(31)
        rng = np.random.default_rng(9)
        upper = np.column_stack([x, y, .004+rng.normal(0, .0004, len(x))])
        lower = np.column_stack([x, y, -.004+rng.normal(0, .0004, len(x))])
        vertices = np.concatenate([upper, lower])
        faces = np.concatenate([f, f[:, ::-1]+len(x)])
        out, normals, _ = reconstruct_features(vertices, faces, .045, .003)
        self.assertGreater(out[:len(x), 2].mean()-out[len(x):, 2].mean(), .0078)
        self.assertGreater(normals[:len(x), 2].mean(), .99)
        self.assertLess(normals[len(x):, 2].mean(), -.99)

    def test_invalid_scales_are_rejected(self):
        x, y, f = grid(5)
        v = np.column_stack([x, y, x*0])
        for radius, move in [(0, .01), (.01, .02), (float('nan'), .01)]:
            with self.assertRaises(ValueError): reconstruct_features(v, f, radius, move)
