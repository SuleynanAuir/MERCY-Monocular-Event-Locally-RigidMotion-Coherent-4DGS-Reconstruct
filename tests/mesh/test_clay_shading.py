import unittest
import torch
from utils.clay_shading import vertex_normals, studio_clay, posed_clay_normals
from types import SimpleNamespace


class ClayShadingTests(unittest.TestCase):
    def test_filtered_normals_follow_native_then_mlp_rotation(self):
        from utils.event_refinement import qrotate
        n = torch.tensor([[0., 0., 1.], [1., 0., 0.]])
        q = torch.tensor([[.70710678, .70710678, 0., 0.]])
        delta = torch.tensor([[[1., 0., 0., 0.]], [[.70710678, 0., .70710678, 0.]]])
        field = SimpleNamespace(angles=torch.tensor([[0.], [.1]]), delta_q=delta)
        groups = torch.tensor([0, 0])
        expected = qrotate(delta[1][groups], qrotate(q[groups], n))
        torch.testing.assert_close(posed_clay_normals(n, q, groups, torch.eye(3), 1, field), expected)
        torch.testing.assert_close(posed_clay_normals(n, q, groups, torch.eye(3), 0, field), qrotate(q[groups], n))

    def test_planar_normals_and_degenerate_vertices(self):
        v = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [9., 9., 9.]])
        n = vertex_normals(v, torch.tensor([[0, 1, 2], [3, 3, 3]]))
        torch.testing.assert_close(n[:3], torch.tensor([[0., 0., 1.]]).expand(3, 3))
        self.assertTrue(torch.isfinite(n).all())
        torch.testing.assert_close(n[3], torch.zeros(3))

    def test_normals_follow_posed_geometry(self):
        v = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        f = torch.tensor([[0, 1, 2]])
        rotation = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        torch.testing.assert_close(vertex_normals(v @ rotation.T + 7, f),
                                   vertex_normals(v, f) @ rotation.T)

    def test_clay_is_neutral_two_sided_and_reveals_orientation(self):
        n = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        p = torch.tensor([[0., 0., -2.]]).expand(3, 3)
        color = studio_clay(n, p)
        self.assertTrue(torch.isfinite(color).all())
        self.assertTrue(((color >= 0) & (color <= 1)).all())
        torch.testing.assert_close(color[:, 0], color[:, 2])
        self.assertGreater(float(color.max() - color.min()), .1)
        torch.testing.assert_close(studio_clay(n[:1], p[:1]), studio_clay(-n[:1], p[:1]))
