import unittest
import numpy as np
import torch
try:
    from scripts.fvv.motion_coherence import temporal_neighbor_loss,surface_pairs
except ModuleNotFoundError:
    from motion_coherence import temporal_neighbor_loss,surface_pairs


class MotionCoherence(unittest.TestCase):
    def test_static_texture_and_shared_lighting_are_not_penalized(self):
        texture=torch.randn(8,1,3);lighting=torch.randn(48,1,1,3)
        values=(texture[None]+lighting).requires_grad_();pairs=torch.tensor([[0,1],[2,3],[4,5]])
        loss=temporal_neighbor_loss(values,pairs)
        self.assertLess(loss.item(),1e-12)

    def test_independent_flicker_has_finite_corrective_gradient(self):
        values=torch.zeros(5,2,1,3);values[2,0]=1;values.requires_grad_()
        loss=temporal_neighbor_loss(values,torch.tensor([[0,1]]));loss.backward()
        self.assertTrue(torch.isfinite(values.grad).all());self.assertGreater(values.grad[2,0].sum().item(),0)
        self.assertLess(values.grad[2,1].sum().item(),0)

    def test_neighbors_do_not_cross_parts_or_corners(self):
        p=np.array([[0,0,0],[.001,0,0],[.002,0,0],[.003,0,0]],dtype=float)
        normals=np.array([[0,0,1],[0,0,1],[0,0,1],[1,0,0]],dtype=float)
        pairs=surface_pairs(p,normals,np.array([1,1,2,1]))
        self.assertEqual(pairs.tolist(),[[0,1]])


if __name__=='__main__':unittest.main()
