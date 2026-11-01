import unittest
import torch
try:
    from scripts.fvv.temporal_supervision import backward_observed_change
except ModuleNotFoundError:
    from temporal_supervision import backward_observed_change


class TemporalGradient(unittest.TestCase):
    def test_sequential_render_gradients_match_joint_l1_including_zero_residual(self):
        for offset in [0.,.7]:
            p=torch.tensor([[[.2,-.3],[.7,.9]]],requires_grad=True)
            weight=torch.tensor([[1.,4.],[4.,1.]])
            target=(p.sin()-p.square()).detach() if offset==0 else torch.full_like(p,offset)
            loss=8*((p.sin()-p.square()-target).abs()*weight).mean()/weight.mean()
            loss.backward();expected=p.grad.clone();p.grad=None
            actual=backward_observed_change(lambda:p.square(),lambda:p.sin(),target,weight)
            self.assertAlmostEqual(actual,float(loss),places=6)
            torch.testing.assert_close(p.grad,expected)


if __name__=='__main__':unittest.main()
