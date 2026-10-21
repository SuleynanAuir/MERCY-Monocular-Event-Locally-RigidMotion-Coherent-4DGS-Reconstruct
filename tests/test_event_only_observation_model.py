"""Synthetic-only checks for event background compositing and its gradients.

No dataset, checkpoint, video, or scene file is opened. CPU cases exercise the
actual trainer helper; optional CUDA cases call the repository rasterizer
directly with three synthetic Gaussians and compare equivalent formulations.
"""

import unittest

try:
    import numpy as np
    import torch
    from PIL import Image  # noqa: F401: trainer import dependency
    from plyfile import PlyData  # noqa: F401: renderer import dependency
except (ImportError, OSError) as error:
    DEPENDENCY_ERROR = str(error)
else:
    DEPENDENCY_ERROR = None
    from scripts.reconstruct_lego_event_only import observation_log
    from utils.event_only_renderer import make_camera


@unittest.skipIf(DEPENDENCY_ERROR is not None, f"Numerical dependencies unavailable: {DEPENDENCY_ERROR}")
class ObservationModelScalarTests(unittest.TestCase):
    def check_coverage_gradient(self, foreground, expected):
        background = 0.04
        coverage = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True)
        black_gray = coverage * foreground
        observed = observation_log(
            {"gray": black_gray, "alpha": coverage.unsqueeze(0)}, background, 1e-4)
        gradient, = torch.autograd.grad(observed.sum(), coverage)
        intensity = 0.1 * foreground + 0.9 * background
        self.assertEqual(tuple(observed.shape), (1, 1))
        self.assertLess(float(observed), 0, "The clamp must precede log, not clip negative log values")
        self.assertAlmostEqual(float(gradient), expected, places=10)
        self.assertAlmostEqual(float(gradient), (foreground - background) / intensity, places=10)
        torch.testing.assert_close(observed.detach(), torch.tensor([[intensity]], dtype=torch.float64).log())

        hard_coverage = coverage.detach().clone().requires_grad_()
        hard_log = (hard_coverage * foreground).clamp_min(background).log()
        hard_gradient, = torch.autograd.grad(hard_log.sum(), hard_coverage)
        self.assertEqual(float(hard_gradient), 0, "Hard intensity flooring removes the coverage gradient")

    def test_bright_foreground_retains_positive_occlusion_gradient(self):
        self.check_coverage_gradient(foreground=0.2, expected=2.857142857142857)

    def test_dark_foreground_retains_negative_occlusion_gradient(self):
        self.check_coverage_gradient(foreground=0.02, expected=-0.5263157894736842)


@unittest.skipIf(DEPENDENCY_ERROR is not None, f"Numerical dependencies unavailable: {DEPENDENCY_ERROR}")
class ObservationModelRasterizerTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        try:
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        except (ImportError, OSError) as error:
            self.skipTest(f"Repository CUDA rasterizer is unavailable: {error}")
        self.Settings = GaussianRasterizationSettings
        self.Rasterizer = GaussianRasterizer
        self.device = torch.device("cuda")
        self.camera = make_camera(
            {"width": 64, "height": 48, "fx": 55., "fy": 54., "cx": 31.6, "cy": 23.8,
             "w2c": np.eye(4).tolist()}, device=self.device)
        self.xyz = torch.tensor([[-.06, .02, 1.8], [.08, -.03, 2.05], [.01, .09, 2.3]],
                                device=self.device, dtype=torch.float32)
        # Non-axis-aligned anisotropic Gaussians yield meaningful q gradients.
        self.quaternion = torch.tensor([[1., .2, .1, .3], [.9, .12, -.25, .14],
                                        [.95, .1, .15, -.2]], device=self.device, dtype=torch.float32)
        self.scales = torch.tensor([[.10, .045, .035], [.055, .12, .04], [.08, .05, .11]],
                                   device=self.device, dtype=torch.float32)
        self.opacity = torch.tensor([[.5], [.45], [.35]], device=self.device, dtype=torch.float32)
        self.colors = torch.tensor([[.20, .30, .25], [.10, .18, .12], [.27, .13, .22]],
                                   device=self.device, dtype=torch.float32)
        self.background = 0.04
        self.luma = torch.tensor([.299, .587, .114], device=self.device, dtype=torch.float32)[:, None, None]
        # Central non-symmetric ROI: avoid translation-invariant whole-image
        # alpha sums and the outer alpha-cutoff pixels in finite differences.
        self.roi = torch.zeros((48, 64), device=self.device, dtype=torch.float32)
        self.roi[22:26, 30:34] = torch.arange(1, 17, device=self.device).reshape(4, 4)
        self.roi /= self.roi.sum()

    def render(self, xyz, quaternion, background):
        camera = self.camera
        settings = self.Settings(
            image_height=camera.image_height, image_width=camera.image_width,
            tanfovx=camera.tanfovx, tanfovy=camera.tanfovy,
            bg=torch.full((3,), background, device=self.device, dtype=torch.float32), scale_modifier=1.,
            viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
            sh_degree=0, campos=camera.camera_center, prefiltered=False, debug=False)
        rotation = quaternion / quaternion.norm(dim=1, keepdim=True)
        rgb, radii, depth, alpha = self.Rasterizer(raster_settings=settings)(
            means3D=xyz, means2D=torch.zeros_like(xyz), opacities=self.opacity,
            shs=None, colors_precomp=self.colors, scales=self.scales,
            rotations=rotation, cov3D_precomp=None)
        return {"rgb": rgb, "gray": (rgb * self.luma).sum(0), "alpha": alpha,
                "depth": depth, "radii": radii}

    def composite_loss(self, xyz, quaternion):
        rendered = self.render(xyz, quaternion, 0.)
        return (observation_log(rendered, self.background, 1e-4) * self.roi).sum()

    def test_composite_matches_native_background_pixels_and_pose_gradients(self):
        xyz_black = self.xyz.clone().requires_grad_()
        q_black = self.quaternion.clone().requires_grad_()
        black = self.render(xyz_black, q_black, 0.)
        rgb_composite = black["rgb"] + (1 - black["alpha"]) * self.background
        log_composite = observation_log(black, self.background, 1e-4)
        grad_xyz_black, grad_q_black = torch.autograd.grad(
            (log_composite * self.roi).sum(), (xyz_black, q_black))

        xyz_native = self.xyz.clone().requires_grad_()
        q_native = self.quaternion.clone().requires_grad_()
        native = self.render(xyz_native, q_native, self.background)
        log_native = observation_log(native, 0., 1e-4)
        grad_xyz_native, grad_q_native = torch.autograd.grad(
            (log_native * self.roi).sum(), (xyz_native, q_native))

        self.assertLess(float(rgb_composite.max()), 1., "Fixture must avoid upper observation clipping")
        self.assertGreater(float(rgb_composite.min()), 1e-4, "Fixture must avoid lower observation clipping")
        self.assertTrue(black["alpha"].requires_grad)
        self.assertTrue(bool((black["radii"] > 0).all()))
        torch.testing.assert_close(rgb_composite, native["rgb"], rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(log_composite, log_native, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(black["alpha"], native["alpha"], rtol=1e-6, atol=1e-7)
        for actual, expected in ((grad_xyz_black, grad_xyz_native), (grad_q_black, grad_q_native)):
            self.assertTrue(bool(torch.isfinite(actual).all()))
            self.assertGreater(float(actual.abs().sum()), 1e-5)
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        for frozen in (self.opacity, self.colors, self.scales):
            self.assertFalse(frozen.requires_grad)
            self.assertIsNone(frozen.grad)

    def test_composite_pose_gradients_match_central_finite_difference(self):
        xyz = self.xyz.clone().requires_grad_()
        quaternion = self.quaternion.clone().requires_grad_()
        gx, gq = torch.autograd.grad(self.composite_loss(xyz, quaternion), (xyz, quaternion))
        # Use nonzero directions to avoid comparing numerical noise around a
        # symmetry. The selection is based only on this synthetic test fixture.
        directions = [("xyz", (0, 0), gx[0, 0]),
                      ("quaternion", divmod(int(gq.abs().argmax()), 4), gq.flatten()[gq.abs().argmax()])]
        epsilon = 5e-4
        for name, index, analytic in directions:
            with self.subTest(parameter=name):
                self.assertGreater(abs(float(analytic)), 1e-4)
                with torch.no_grad():
                    plus_xyz, minus_xyz = self.xyz.clone(), self.xyz.clone()
                    plus_q, minus_q = self.quaternion.clone(), self.quaternion.clone()
                    positive, negative = ((plus_xyz, minus_xyz) if name == "xyz" else (plus_q, minus_q))
                    positive[index] += epsilon
                    negative[index] -= epsilon
                    numerical = ((self.composite_loss(plus_xyz, plus_q) -
                                  self.composite_loss(minus_xyz, minus_q)) / (2 * epsilon))
                self.assertTrue(bool(torch.isfinite(numerical)))
                torch.testing.assert_close(analytic, numerical, rtol=.02, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
