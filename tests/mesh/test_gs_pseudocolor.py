import unittest
import torch
from utils.gs_pseudocolor import palette, pseudocolor_materials, color_gaussians, srgb_to_oklab, oklab_to_srgb, soft_color_finish


class PseudocolorTests(unittest.TestCase):
    def test_perceptual_conversion_matches_white_and_round_trips_srgb(self):
        colors = torch.tensor([[0., 0., 0.], [1., 1., 1.], [1., 0., 0.],
                               [0., 1., 0.], [0., 0., 1.], [.13, .48, .72]])
        lab = srgb_to_oklab(colors)
        torch.testing.assert_close(lab[1], torch.tensor([1., 0., 0.]), atol=1e-6, rtol=0)
        torch.testing.assert_close(oklab_to_srgb(lab), colors, atol=1e-5, rtol=0)

    def test_palette_does_not_flatten_into_bands_at_internal_stops(self):
        for knot in (.2, .4, .6, .8):
            step = 1e-4
            colors = srgb_to_oklab(palette(torch.tensor([knot-step, knot, knot+step])))
            left, right = (colors[1]-colors[0])/step, (colors[2]-colors[1])/step
            self.assertGreater(float(left.norm()), .08)
            self.assertLess(float((left-right).norm()), .01)

    def test_soft_highlights_preserve_order_without_channel_clipping(self):
        shade = torch.linspace(0, 1, 100)[None, None, :]
        color = torch.tensor([.94, .78, .43])[:, None, None]
        result = soft_color_finish(color, shade)
        self.assertTrue(torch.all(result[..., 1:] >= result[..., :-1]))
        self.assertGreater(float((result[..., -1]-result[..., 0]).min()), .3)
        self.assertLess(float(result.max()), .99)

    def test_palette_is_bounded_colored_and_continuous(self):
        colors = palette(torch.linspace(-.1, 1.1, 1001))
        self.assertTrue(torch.isfinite(colors).all())
        self.assertTrue(torch.all((colors >= 0) & (colors <= 1)))
        self.assertGreater(float((colors.max(1).values-colors.min(1).values).min()), .2)
        self.assertLess(float((colors[1:]-colors[:-1]).abs().max()), .01)

    def test_common_points_share_colors_across_differently_sized_sides(self):
        left = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
        right = torch.cat((left, torch.tensor([[2., 1., 0.]])))
        original = [left.clone(), right.clone()]
        colors = pseudocolor_materials([left, right])
        torch.testing.assert_close(colors[0], colors[1][:2], rtol=0, atol=0)
        for after, before in zip((left, right), original): torch.testing.assert_close(after, before)
        # Palette rows are immutable material attributes, reordered with points.
        reordered = pseudocolor_materials([left.flip(0), right])
        torch.testing.assert_close(colors[0].flip(0), reordered[0])

    def test_degenerate_extent_and_invalid_shapes(self):
        self.assertTrue(torch.isfinite(pseudocolor_materials([torch.ones(3, 3)])[0]).all())
        for positions in ([], [torch.zeros(0, 3)], [torch.ones(2, 4)]):
            with self.assertRaises(ValueError): pseudocolor_materials(positions)

    def test_coverage_holes_and_source_buffers_are_preserved(self):
        alpha = torch.zeros(1, 40, 40); alpha[:, 5:35, 5:35] = 1
        alpha[:, 4, 5:35] = .3; alpha[:, 17:21, 17:21] = 0
        rgb = .5*alpha.expand(3, -1, -1); depth = 2*alpha
        color = torch.tensor([.1, .7, .9])[:, None, None]*alpha
        inputs = [rgb, depth, alpha, color]; originals = [x.clone() for x in inputs]
        black = color_gaussians(*inputs, 45, [0, 0, 0])
        white = color_gaussians(*inputs, 45, [1, 1, 1])
        torch.testing.assert_close(white-black, (1-alpha).expand_as(black))
        self.assertEqual(float(black[:, alpha[0] == 0].abs().max()), 0.)
        self.assertGreater(float((black[2]-black[0])[:, 8:16].max()), .3)
        for after, before in zip(inputs, originals): torch.testing.assert_close(after, before)
        again = color_gaussians(*inputs, 45, [0, 0, 0])
        torch.testing.assert_close(black, again, rtol=0, atol=0)
