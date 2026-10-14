import unittest
import torch

from utils.appearance_refinement import SurfaceAppearance, masked_detail_losses, masked_laplacian_error


class SurfaceAppearanceTests(unittest.TestCase):
    def model(self):
        return SurfaceAppearance(torch.zeros(4, 1, 3), torch.full((4, 3), -7.),
                                 torch.full((4, 1), 2.944439))

    def test_only_color_scalar_size_and_opacity_are_parameters(self):
        model = self.model()
        self.assertEqual(set(dict(model.named_parameters())), {'gray', 'log_scale_ratio', 'opacity'})
        dc, scale, opacity = model()
        torch.testing.assert_close(dc, torch.zeros(4, 1, 3))
        torch.testing.assert_close(scale, torch.full((4, 3), -7.))
        torch.testing.assert_close(opacity, torch.full((4, 1), 2.944439))

    def test_bounds_hold_even_before_projection(self):
        model = self.model()
        with torch.no_grad():
            model.gray[:2] = -10; model.gray[2:] = 10
            model.log_scale_ratio[:2] = -10; model.log_scale_ratio[2:] = 10
            model.opacity[:2] = -10; model.opacity[2:] = 10
        dc, scale, opacity = model()
        gray = dc[:, 0, 0]*.28209479177387814+.5
        self.assertTrue(torch.all((gray >= .00099) & (gray <= .99901)))
        ratio = (scale+7).exp()
        self.assertTrue(torch.all((ratio >= .54999) & (ratio <= 1.10001)))
        self.assertTrue(torch.all((opacity.sigmoid() >= .74999) & (opacity.sigmoid() <= .99501)))
        torch.testing.assert_close(scale[:, 0], scale[:, 2])
        model.project_()
        torch.testing.assert_close(model.opacity, opacity)

    def test_gradient_loss_rejects_noise_on_an_exact_image(self):
        target = torch.zeros(20, 20); target[:, 10:] = .7
        mask = torch.ones_like(target)
        zero = masked_detail_losses(target, target, mask)
        noisy = target + .05*((torch.arange(20)[:, None]+torch.arange(20)[None, :]) % 2)
        losses = masked_detail_losses(noisy, target, mask)
        self.assertEqual(float(zero[1]), 0.)
        self.assertGreater(float(losses[0]), 0.)
        self.assertGreater(float(losses[1]), .04)

    def test_empty_mask_is_finite(self):
        losses = masked_detail_losses(torch.ones(3, 3), torch.zeros(3, 3), torch.zeros(3, 3))
        self.assertTrue(all(torch.isfinite(loss) and loss == 0 for loss in losses))

    def test_anisotropic_axes_are_independent_but_bounded(self):
        model = SurfaceAppearance(torch.zeros(4, 1, 3), torch.full((4, 3), -7.),
                                  torch.full((4, 1), 2.944439), .7, 1.3, anisotropic=True)
        with torch.no_grad():
            model.log_scale_ratio[:, 0] = -10
            model.log_scale_ratio[:, 1] = 10
        _, scales, _ = model()
        torch.testing.assert_close((scales[:, 0]+7).exp(), torch.full((4,), .7))
        torch.testing.assert_close((scales[:, 1]+7).exp(), torch.full((4,), 1.3))
        torch.testing.assert_close(scales[:, 2], torch.full((4,), -7.))
        self.assertEqual(set(dict(model.named_parameters())), {'gray', 'log_scale_ratio', 'opacity'})

    def test_laplacian_loss_does_not_reward_added_noise(self):
        target = torch.zeros(20, 20); target[:, 10:] = .7
        mask = torch.ones_like(target)
        noisy = target + .05*((torch.arange(20)[:, None]+torch.arange(20)[None, :]) % 2)
        self.assertEqual(float(masked_laplacian_error(target, target, mask)), 0.)
        self.assertGreater(float(masked_laplacian_error(noisy, target, mask)), .15)
        self.assertEqual(float(masked_laplacian_error(noisy, target, mask*0)), 0.)


if __name__ == '__main__':
    unittest.main()
