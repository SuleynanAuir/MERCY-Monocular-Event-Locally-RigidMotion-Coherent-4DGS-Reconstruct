import unittest
import torch
from utils.gaussian_pair_renderer import edge_preserving_rgb


class RGBFilterTests(unittest.TestCase):
    def test_flat_surface_and_transparent_background_preserved(self):
        rgb=torch.full((3,9,9),.3)
        torch.testing.assert_close(edge_preserving_rgb(rgb,torch.ones(1,9,9),{}),rgb)
        textured=torch.rand(3,9,9)
        torch.testing.assert_close(edge_preserving_rgb(textured,torch.zeros(1,9,9),{}),textured)

    def test_small_noise_reduced_without_bleeding_across_silhouette(self):
        rgb=torch.zeros(3,11,11);rgb[:,3:8,3:8]=.5;rgb[:,5,5]=.55
        alpha=torch.zeros(1,11,11);alpha[:,3:8,3:8]=1
        out=edge_preserving_rgb(rgb,alpha,{})
        self.assertLess(float(out[:,5,5].mean()),.53)
        self.assertGreater(float(out[:,3,3].mean()),.49)
        torch.testing.assert_close(out[:,:3],rgb[:,:3])

    def test_unsafe_parameters_rejected(self):
        for setting in [dict(kernel_size=9),dict(color_sigma=0),dict(spatial_sigma=float('nan'))]:
            with self.assertRaises(ValueError):edge_preserving_rgb(torch.zeros(3,5,5),torch.ones(1,5,5),setting)
