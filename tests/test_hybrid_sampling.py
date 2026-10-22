import unittest
import torch
import torch.nn.functional as F

from utils.hybrid_sampling import sampling_plan, resolve_hybrid_rgb, smooth_surface
from utils.render_sampling import resolve_rgb


class HybridSamplingTests(unittest.TestCase):
    def test_balanced_is_pixel_identical_to_previous_hybrid(self):
        image = torch.randn((3, 256, 320), generator=torch.Generator().manual_seed(4))
        torch.testing.assert_close(resolve_hybrid_rgb(image, 160, 128, 'balanced'),
                                   resolve_rgb(image, 2), rtol=0, atol=0)

    def test_constant_colors_survive_and_edges_cannot_make_new_extrema(self):
        image = torch.full((3, 384, 384), .4)
        torch.testing.assert_close(resolve_hybrid_rgb(image, 128, 128), torch.full((3, 128, 128), .4))
        image[:, :, 193:] = .8
        result = resolve_hybrid_rgb(image, 128, 128)
        area = F.avg_pool2d(image, 3).unsqueeze(0)
        upper, lower = F.max_pool2d(area, 3, 1, 1)[0], -F.max_pool2d(-area, 3, 1, 1)[0]
        # Fixed and adaptive area averages differ by one float32 rounding ULP.
        eps = torch.finfo(result.dtype).eps
        self.assertTrue(torch.all(result <= upper + eps).item())
        self.assertTrue(torch.all(result >= lower - eps).item())
        self.assertTrue(torch.isfinite(result).all().item())

    def test_large_rectangles_are_bounded_without_changing_output_shape(self):
        for width, height in ((768, 768), (1024, 1024), (1280, 960), (960, 1280)):
            w, h = sampling_plan(width, height, 'detail')
            self.assertLessEqual(max(w, h), 3072)
            self.assertLessEqual(w * h, 3072 ** 2)
            self.assertAlmostEqual(w / h, width / height, delta=.001)
        w, h = sampling_plan(1280, 960, 'detail')
        result = resolve_hybrid_rgb(torch.zeros((3, h, w)), 1280, 960)
        self.assertEqual(result.shape, (3, 960, 1280))

    def test_native_sampling_keeps_source_footprint_at_all_output_sizes(self):
        self.assertEqual(sampling_plan(640, 640, 'surface'), (640, 640))
        self.assertEqual(sampling_plan(1024, 768, 'native'), (1024, 768))
        self.assertEqual(sampling_plan(640, 640, 'surface', native_supersample=2), (1280, 1280))
        image = torch.rand((3, 128, 128), generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(resolve_hybrid_rgb(image, 128, 128, 'native'), image, rtol=0, atol=0)

    def test_surface_reduces_small_interior_noise_without_new_extrema(self):
        noise = torch.randn((1, 64, 64), generator=torch.Generator().manual_seed(3)) * .015
        image = (.5 + noise).expand(3, -1, -1)
        result = smooth_surface(image, torch.full((1, 64, 64), 2.), torch.ones((1, 64, 64)))
        self.assertLess(result[:, 3:-3, 3:-3].std(), image[:, 3:-3, 3:-3].std() * .75)
        self.assertLessEqual((result - image).abs().max(), .035 + 1e-7)
        self.assertGreaterEqual(result.min(), image.min())
        self.assertLessEqual(result.max(), image.max())

    def test_surface_preserves_holes_transparency_and_strong_edges(self):
        image = torch.full((3, 24, 24), .5); image[:, :, 12:] = .9
        alpha = torch.ones((1, 24, 24)); alpha[:, 6:9, 6:9] = 0.
        image[:, 6:9, 6:9] = 0.
        depth = alpha * 2.
        result = smooth_surface(image, depth, alpha)
        torch.testing.assert_close(result[:, :, 11:13], image[:, :, 11:13], rtol=0, atol=0)
        torch.testing.assert_close(result[:, 5:10, 5:10], image[:, 5:10, 5:10], rtol=0, atol=0)
        torch.testing.assert_close(smooth_surface(image, depth, alpha * .8), image, rtol=0, atol=0)
        depth[:, 2:4, 2:4] = float('nan')
        self.assertTrue(torch.isfinite(smooth_surface(image, depth, alpha)).all().item())

    def test_surface_does_not_blend_across_depth_discontinuities(self):
        image = torch.full((3, 24, 24), .5); image[:, :, 12:] = .55
        depth = torch.full((1, 24, 24), 2.); depth[:, :, 12:] = 4.
        result = smooth_surface(image, depth, torch.ones_like(depth))
        torch.testing.assert_close(result[:, :, 11:13], image[:, :, 11:13], rtol=0, atol=0)

    def test_quality_and_shape_errors_are_rejected(self):
        for quality in (None, '', 'ultra'):
            with self.assertRaises(ValueError): sampling_plan(128, 128, quality)
        with self.assertRaises(ValueError): sampling_plan(0, 128)
        with self.assertRaises(ValueError): resolve_hybrid_rgb(torch.zeros((3, 64, 64)), 128, 128)


if __name__ == '__main__': unittest.main()
