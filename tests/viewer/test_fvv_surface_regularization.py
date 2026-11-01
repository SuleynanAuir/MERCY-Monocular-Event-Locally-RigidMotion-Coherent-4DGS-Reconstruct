"""Analytical checks for the optional normal constraint (requires PyTorch)."""
import unittest
try:
    import torch
except ImportError:
    torch=None
from scripts.fvv.surface_regularization import normal_surface_loss


@unittest.skipIf(torch is None, 'PyTorch is required; run in the Hybrid environment')
class SurfaceRegularization(unittest.TestCase):
    def test_tangent_extent_and_normal_sign_do_not_change_loss(self):
        center=torch.tensor([[.4,.2,.01]],requires_grad=True)
        covariance=torch.tensor([[1.,0,0,2.,0,.0001]],requires_grad=True)
        surface=torch.zeros((1,3));normal=torch.tensor([[0.,0.,1.]])
        nearest=torch.tensor([0])
        loss=normal_surface_loss(center,covariance,surface,normal,nearest)
        self.assertAlmostEqual(loss.item(),.017,places=6)
        self.assertAlmostEqual(loss.item(),normal_surface_loss(center,covariance,surface,-normal,nearest).item(),places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(covariance.grad).all())
        self.assertEqual(center.grad[0,0].item(),0)
        self.assertEqual(covariance.grad[0,0].item(),0)
        self.assertGreater(covariance.grad[0,5].item(),0)

    def test_rotated_covariance_and_thin_surface(self):
        center=torch.zeros((1,3));surface=center.clone();nearest=torch.tensor([0])
        normal=torch.tensor([[2**-.5,2**-.5,0.]])
        covariance=torch.tensor([[.00005,.00005,0,.00005,0,1.]])
        self.assertAlmostEqual(normal_surface_loss(center,covariance,surface,normal,nearest).item(),.007,places=6)
        self.assertEqual(normal_surface_loss(center,covariance*.0001,surface,normal,nearest).item(),0)


if __name__=='__main__':
    unittest.main()
