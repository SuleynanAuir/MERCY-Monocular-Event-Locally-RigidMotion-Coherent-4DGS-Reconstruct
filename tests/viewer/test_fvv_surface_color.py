import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch=None
from scripts.fvv.surface_color_regularization import build_pairs,surface_color_loss


class SurfaceNeighbors(unittest.TestCase):
    def test_different_parts_times_normals_and_parallel_walls_do_not_mix(self):
        p=np.array([[0,0,0],[.001,0,0],[.0001,0,0],[.0002,0,0],[0,0,.002],[.0003,0,0]],np.float32)
        n=np.tile([0,0,1.],(6,1));n[3]*=-1
        pairs,weights=build_pairs(p,n,np.array([0,0,1,0,0,0]),np.array([0,0,0,0,0,1]))
        np.testing.assert_array_equal(pairs,[[0,1]])
        self.assertTrue(0<weights[0]<=1)

    @unittest.skipIf(torch is None,'Run in Hybrid PyTorch environment')
    def test_neighbor_loss_can_lift_dark_coefficients_and_keeps_equal_colors(self):
        dc=torch.tensor([[[-1.,-1.,-1.]],[[1.,1.,1.]]],requires_grad=True)
        rest=torch.zeros((2,15,3),requires_grad=True);pairs=torch.tensor([[0,1]]);weights=torch.ones(1)
        loss=surface_color_loss(dc,rest,pairs,weights);loss.backward()
        self.assertTrue((dc.grad[0]<0).all());self.assertTrue((dc.grad[1]>0).all())
        self.assertEqual(float(surface_color_loss(torch.ones_like(dc),rest,pairs,weights)),0)


if __name__=='__main__':unittest.main()
