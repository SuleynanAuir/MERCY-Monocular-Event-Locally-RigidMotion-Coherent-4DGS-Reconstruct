"""Real rasterization regression for display-only material coordinates."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA rasterizer required')
class GaussianMaterialTests(unittest.TestCase):
    def test_override_is_used_without_mutating_native_sh_or_coverage(self):
        from gaussian_renderer import render
        from scripts.rendering.render_ply import make_camera
        xyz = torch.zeros(1, 3, device='cuda')
        rotation = torch.tensor([[1., 0, 0, 0]], device='cuda')
        features = torch.zeros(1, 1, 3, device='cuda')
        model = SimpleNamespace(get_xyz=xyz, get_scaling=torch.full_like(xyz, .2),
            get_opacity=torch.ones(1, 1, device='cuda'), get_rotation_bias=lambda _: rotation,
            get_features=features, active_sh_degree=0, max_sh_degree=0)
        pose = np.eye(4, dtype=np.float32); pose[2, 3] = 2
        camera = make_camera(pose, 64, 64, np.deg2rad(45))
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        background = torch.zeros(3, device='cuda')
        with torch.no_grad():
            native = render(camera, model, pipe, background, 0., 0., 0.)
            material = torch.tensor([[.85, .35, .10]], device='cuda')
            chart = render(camera, model, pipe, background, 0., 0., 0., override_color=material)
            again = render(camera, model, pipe, background, 0., 0., 0.)
        torch.testing.assert_close(chart['render'], chart['alpha']*material[0, :, None, None])
        torch.testing.assert_close(chart['alpha'], native['alpha'], rtol=0, atol=0)
        torch.testing.assert_close(again['render'], native['render'], rtol=0, atol=0)
        self.assertGreater(float((chart['render']-native['render']).abs().max()), .1)
        self.assertEqual(float(features.abs().max()), 0.)
