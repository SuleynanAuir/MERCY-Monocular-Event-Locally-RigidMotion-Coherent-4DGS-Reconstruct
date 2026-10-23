import unittest
import torch
from utils.gs_clay import gray_gaussians, filtered_depth, depth_normals, surface_wire, canonical_coordinates, clean_luminance, suppress_isolated_grain


class GaussianClayTests(unittest.TestCase):
    def test_neutral_material_preserves_coverage_and_background_exactly(self):
        alpha = torch.zeros(1, 40, 40)
        alpha[:, 8:32, 8:32] = 1
        alpha[:, 7, 8:32] = .3
        alpha[:, 17:21, 17:21] = 0  # A real small hole must remain open.
        rgb = torch.rand(3, 40, 40)*alpha
        depth = alpha*2
        original = [value.clone() for value in (rgb, depth, alpha)]
        black = gray_gaussians(rgb, depth, alpha, 45, [0, 0, 0])
        white = gray_gaussians(rgb, depth, alpha, 45, [1, 1, 1])
        torch.testing.assert_close(white-black, (1-alpha).expand_as(black))
        torch.testing.assert_close(black[0], black[1])
        torch.testing.assert_close(black[:, alpha[0] == 0], torch.zeros_like(black[:, alpha[0] == 0]))
        for actual, before in zip((rgb, depth, alpha), original): torch.testing.assert_close(actual, before)

    def test_invalid_depth_is_finite_and_has_no_temporal_state(self):
        rgb = torch.full((3, 32, 32), .4)
        alpha = torch.ones(1, 32, 32)
        depth = torch.full_like(alpha, 2.)
        depth[:, 3, 3] = float('nan'); depth[:, 7, 7] = float('inf')
        a = gray_gaussians(rgb, depth, alpha, 45, [0, 0, 0])
        gray_gaussians(rgb*.3, depth*2, alpha*.7, 60, [1, 1, 1])
        b = gray_gaussians(rgb, depth, alpha, 45, [0, 0, 0])
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_lighting_guide_reduces_depth_noise_without_crossing_step(self):
        generator = torch.Generator().manual_seed(7)
        depth = torch.ones(1, 64, 64)*2
        depth[:, :, 32:] = 3
        noisy = depth+torch.randn(depth.shape, generator=generator)*.005
        out, valid = filtered_depth(noisy, torch.ones_like(depth))
        self.assertLess(float((out-depth)[:, 8:-8, 8:24].std()), .002)
        self.assertLess(float((out-depth)[:, 8:-8, 31:33].abs().max()), .01)
        n = depth_normals(out, torch.ones_like(depth), 45)
        torch.testing.assert_close(n.norm(dim=0), torch.ones(64, 64), atol=1e-5, rtol=1e-5)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            gray_gaussians(torch.ones(3, 8, 8), torch.ones(8, 8), torch.ones(1, 8, 8), 45, [0, 0, 0])

    def test_material_coordinates_share_bounds_without_changing_source(self):
        xyz = torch.tensor([[-2., 0, 0], [2., 1, 0]])
        before = xyz.clone()
        colors, bounds = canonical_coordinates(xyz)
        other, _ = canonical_coordinates(xyz.clone(), bounds)
        torch.testing.assert_close(colors, other)
        torch.testing.assert_close(xyz, before)
        self.assertTrue(torch.all((colors >= 0) & (colors <= 1)))

    def test_surface_lines_follow_material_coordinates_and_preserve_holes(self):
        yy, xx = torch.meshgrid(torch.arange(80), torch.arange(80), indexing='ij')
        material = torch.stack((xx*.002+.1, yy*.002+.1, torch.ones(80, 80)*.5))
        alpha = torch.ones(1, 80, 80)
        first = surface_wire(material, alpha)
        moved = surface_wire(material.roll(5, dims=2), alpha)
        torch.testing.assert_close(moved[:, 12:-12, 12:-12], first.roll(5, dims=2)[:, 12:-12, 12:-12])
        self.assertGreater(float(first[:, 12:-12, 12:-12].max()), .5)
        alpha[:, 30:40, 30:40] = 0
        wire = surface_wire(material*alpha, alpha)
        self.assertEqual(float(wire[:, 30:40, 30:40].abs().max()), 0.)
        rgb = torch.full((3, 80, 80), .4)*alpha
        solid = gray_gaussians(rgb, 2*alpha, alpha, 45, [0, 0, 0])
        lined = gray_gaussians(rgb, 2*alpha, alpha, 45, [0, 0, 0], coordinates=material*alpha)
        self.assertGreater(float((solid-lined).max()), .03)
        self.assertLessEqual(float((solid-lined).max()), .14)
        self.assertTrue(torch.all(lined <= solid))
        self.assertEqual(float(lined[:, 30:40, 30:40].abs().max()), 0.)

    def test_unresolved_surface_grid_fades_out(self):
        yy, xx = torch.meshgrid(torch.arange(40), torch.arange(40), indexing='ij')
        material = torch.stack((xx*.03, yy*.03, torch.ones(40, 40)*.5))
        wire = surface_wire(material, torch.ones(1, 40, 40))
        self.assertLess(float(wire[:, 8:-8, 8:-8].abs().max()), .001)

    def test_grain_is_suppressed_without_softening_dark_lines_or_color_steps(self):
        rng = torch.Generator().manual_seed(42)
        luma = .5+torch.randn((1, 64, 64), generator=rng)*.025
        alpha = torch.ones_like(luma); z = torch.full_like(luma, 2.)
        cleaned = clean_luminance(luma, alpha, z)
        self.assertLess(float(cleaned[:, 8:-8, 8:-8].std()), .012)
        edge = torch.full_like(luma, .25); edge[:, :, 32:] = .75
        edge[:, 20, 5:25] = .02
        out = clean_luminance(edge, alpha, z)
        self.assertGreater(float((out[:, 30:50, 32]-out[:, 30:50, 31]).min()), .45)
        self.assertLess(float(out[:, 20, 8:22].max()), .03)

    def test_noisy_depth_does_not_become_strong_surface_relief(self):
        rng = torch.Generator().manual_seed(7)
        alpha = torch.ones(1, 80, 80)
        depth = 2+torch.randn(alpha.shape, generator=rng)*.015
        rgb = torch.full((3, 80, 80), .5)
        result = gray_gaussians(rgb, depth, alpha, 45, [0, 0, 0])
        self.assertLess(float(result[:, 12:-12, 12:-12].std()), .008)

    def test_isolated_impulses_recede_but_thin_diagonal_and_horizontal_lines_remain(self):
        luma = torch.full((1, 64, 64), .5)
        alpha = torch.ones_like(luma); depth = 2*alpha
        luma[:, 10, 10] = .02; luma[:, 10, 30] = .98
        luma[:, 30, 8:24] = .02
        for index in range(36, 55): luma[:, index, index] = .02
        result = clean_luminance(luma, alpha, depth)
        self.assertLess(float((result[:, 10, [10, 30]]-.5).abs().max()), .01)
        self.assertLess(float(result[:, 30, 10:22].max()), .03)
        self.assertLess(max(float(result[:, index, index]) for index in range(38, 53)), .03)

    def test_impulse_filter_preserves_depth_edges_and_translucent_features(self):
        luma = torch.full((1, 32, 32), .5)
        alpha = torch.ones_like(luma); depth = 2*alpha
        luma[:, 8, 8] = .02; alpha[:, 8, 8] = .7
        luma[:, 20, 20] = .02; depth[:, 20, 20] = 3
        before = luma.clone()
        result = suppress_isolated_grain(luma, alpha, depth)
        torch.testing.assert_close(result, before)
        torch.testing.assert_close(luma, before)
