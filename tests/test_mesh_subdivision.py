import unittest
import numpy as np
from utils.mesh_subdivision import refine_surface, _split_faces


def edge_counts(faces):
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    return np.unique(edges, axis=0, return_counts=True)


class SubdivisionTests(unittest.TestCase):
    def test_every_transition_has_same_area_and_winding(self):
        xyz = np.array([[0., 0, 0], [2, 0, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        for mask in range(8):
            mids = np.array([[3 if mask & 1 else -1, 4 if mask & 2 else -1, 5 if mask & 4 else -1]])
            f, p = _split_faces(np.array([[0, 1, 2]]), mids)
            v = xyz[f]
            area = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])[:, 2]
            self.assertTrue((area > 0).all())
            self.assertAlmostEqual(area.sum(), 4)
            self.assertEqual(len(f), 1 + mask.bit_count())
            self.assertTrue((p == 0).all())

    def test_shared_edge_has_single_midpoint_and_no_interior_crack(self):
        xyz = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        v, f, c, endpoints, _, report = refine_surface(xyz, faces, xyz, 1.1, .1)
        np.testing.assert_array_equal(v[:4], xyz)
        np.testing.assert_array_equal(endpoints, [[0, 2]])
        np.testing.assert_array_equal(c[-1], [.5, .5, 0])
        edges, counts = edge_counts(f)
        self.assertTrue((counts <= 2).all())
        self.assertEqual((counts == 1).sum(), 4)
        self.assertEqual(len(v) - len(edges) + len(f), 1)
        self.assertEqual(report['triangles'], 4)

    def test_sphere_curvature_improves_midpoint_and_keeps_closed_topology(self):
        # Fine convex hull has smooth dihedrals, unlike a coarse octahedron.
        from scipy.spatial import ConvexHull
        n = 600
        y = 1 - 2 * (np.arange(n) + .5) / n
        phi = np.arange(n) * np.pi * (3 - np.sqrt(5))
        r = np.sqrt(1 - y * y)
        xyz = np.column_stack([r * np.cos(phi), y, r * np.sin(phi)]).astype(np.float32)
        faces = ConvexHull(xyz).simplices.copy()
        p = xyz[faces]
        inward = np.sum(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]) * p.mean(axis=1), axis=1) < 0
        faces[inward] = faces[inward][:, [0, 2, 1]]
        v, f, _, endpoints, parents, report = refine_surface(xyz, faces, np.ones_like(xyz), .12, .01, 2.)
        linear = xyz[endpoints].mean(axis=1)
        self.assertGreater(report['curved_vertices'], 0)
        self.assertLess(np.abs(np.linalg.norm(v[n:], axis=1)-1).mean(), np.abs(np.linalg.norm(linear, axis=1)-1).mean() * .5)
        self.assertLessEqual(report['max_midpoint_offset'], .010001)
        np.testing.assert_array_equal(v[:n], xyz)
        edges, counts = edge_counts(f)
        self.assertTrue((counts == 2).all())
        self.assertEqual(len(v) - len(edges) + len(f), 2)
        self.assertLessEqual(len(f), 2 * len(faces))
        a, b = xyz[faces[parents]], v[f]
        self.assertTrue((np.sum(np.cross(a[:, 1]-a[:, 0], a[:, 2]-a[:, 0]) * np.cross(b[:, 1]-b[:, 0], b[:, 2]-b[:, 0]), axis=1) > 0).all())

    def test_crease_stays_linear_and_separate_surfaces_stay_separate(self):
        xyz = np.array([[0., 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])
        faces = np.array([[0, 1, 2], [1, 0, 3]])
        xyz = np.concatenate([xyz, xyz + [0, 0, .001]])
        faces = np.concatenate([faces, faces + 4])
        colors = np.concatenate([np.zeros((4, 3)), np.ones((4, 3))])
        v, f, c, endpoints, _, _ = refine_surface(xyz, faces, colors, 1., .2, 4)
        np.testing.assert_allclose(v[8:], xyz[endpoints].mean(axis=1), atol=1e-7)
        self.assertTrue(((endpoints[:, 0] < 4) == (endpoints[:, 1] < 4)).all())
        self.assertTrue((c[f].std(axis=1) == 0).all())

    def test_thin_faces_remain_safe_and_empty_selection_is_identity(self):
        xyz = np.array([[0., 0, 0], [1, 0, 0], [.5, 1e-9, 0]], np.float32)
        faces = np.array([[0, 1, 2]])
        for target in (.1, 2.):
            v, f, _, _, _, _ = refine_surface(xyz, faces, xyz, target, .1)
            np.testing.assert_array_equal(v, xyz)
            np.testing.assert_array_equal(f, faces)

    def test_duplicate_source_faces_do_not_create_ambiguous_interior_edges(self):
        xyz = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2], [2, 1, 0], [0, 2, 3]])
        v, f, _, _, _, report = refine_surface(xyz, faces, xyz, .5, .1, 4.)
        old_edges, _ = edge_counts(faces)
        new_edges, _ = edge_counts(f)
        self.assertEqual(len(xyz)-len(old_edges)+len(faces), len(v)-len(new_edges)+len(f))
        self.assertEqual(report['protected_duplicate_faces'], 2)
        self.assertTrue(any(np.array_equal(face, [0, 1, 2]) for face in f))
        self.assertTrue(any(np.array_equal(face, [2, 1, 0]) for face in f))


if __name__ == '__main__':
    unittest.main()
