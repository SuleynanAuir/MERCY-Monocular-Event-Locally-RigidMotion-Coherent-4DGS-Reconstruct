import unittest
import torch
try:
    from scripts.fvv.detail_supervision import detail_loss
except ModuleNotFoundError:
    from detail_supervision import detail_loss


class DetailSupervision(unittest.TestCase):
    def test_exact_match_has_zero_loss_and_gradient(self):
        target=torch.rand(3,80,80)
        image=target.clone().requires_grad_()
        loss=detail_loss(image,target);loss.backward()
        self.assertEqual(loss.item(),0.)
        self.assertEqual(image.grad.abs().sum().item(),0.)

    def test_small_structure_receives_more_per_pixel_attention(self):
        target=torch.zeros(3,80,80)
        target[:,:40,:40]=.5;target[:,40:60,40:60]=.5
        image=(target*.9).requires_grad_()
        detail_loss(image,target).backward()
        large=image.grad[:,10:30,10:30].abs().mean()
        small=image.grad[:,45:55,45:55].abs().mean()
        self.assertGreater(small.item(),large.item()*3)

    def test_empty_foreground_is_finite(self):
        image=torch.rand(3,80,80,requires_grad=True)
        loss=detail_loss(image,torch.zeros_like(image));loss.backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertEqual(loss.item(),0.)

    def test_edge_error_is_penalized_more_than_constant_error(self):
        target=torch.full((3,80,80),.5)
        flat=target+.1
        striped=flat.clone();striped[:,:,::2]=.4
        self.assertGreater(detail_loss(striped,target).item(),detail_loss(flat,target).item())


if __name__=='__main__':unittest.main()
