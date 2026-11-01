"""Inverse-depth supervision must move occluders and preserve original render API."""
import unittest
try:
    import torch
except ImportError:
    torch=None
try:
    from scripts.fvv.ray_supervision import with_inverse_depth,inverse_depth_loss,render_inverse_depth
except ModuleNotFoundError:
    from ray_supervision import with_inverse_depth,inverse_depth_loss,render_inverse_depth


def sample_renderer(x):
    invdepth=x*2
    return dict(render=x) # intentionally unsupported nonliteral schema


def supported_renderer(x):
    invdepth=x*2
    return {'render':x,'depth':x+1,'alpha':1,'flow':0}


class RendererIsolation(unittest.TestCase):
    def test_copy_exposes_existing_channel_without_mutating_original(self):
        before=supported_renderer(3)
        wrapped=with_inverse_depth(supported_renderer)
        self.assertEqual(wrapped(3),dict(before,inverse_depth=6))
        self.assertEqual(supported_renderer(3),before)
        self.assertEqual(len(wrapped.source_sha256),64)

    def test_unknown_upstream_schema_fails_closed(self):
        with self.assertRaises(ValueError):with_inverse_depth(sample_renderer)


@unittest.skipIf(torch is None,'Run in Hybrid PyTorch environment')
class InverseDepthGradient(unittest.TestCase):
    def test_articulated_geometry_uses_depth_not_its_rgb(self):
        from types import SimpleNamespace
        z=torch.tensor(2.,requires_grad=True)
        class Base:
            get_xyz=torch.empty(0,3)
            get_features=torch.empty(0,16,3)
            get_static_features=torch.ones(1,16,3)*20
            @property
            def get_static_xyz(self):return torch.stack((z*0,z*0,z))[None]
            def get_current_covariance_and_mean_offset(self,*args):return None,torch.empty(0,3)
        class Rigid:
            def proxy(self,base,timestamp):
                class Composite:
                    get_static_xyz=torch.cat([base.get_static_xyz,torch.tensor([[0.,0,4.]])])
                    get_static_features=torch.ones(2,16,3)*100
                    def __getattr__(self,key):return getattr(base,key)
                return Composite()
        camera=SimpleNamespace(timestamp=.5,world_view_transform=torch.eye(4))
        def render(cam,model,pipe,bg):return {'render':(model.get_static_features[:,0,:]*.28209479177387814+.5).T[:,None,:],'alpha':torch.ones(1,1,2)}
        image=render_inverse_depth(render,camera,Base(),None,torch.zeros(3),rigid=Rigid())
        torch.testing.assert_close(image,torch.tensor([[[.5,.25]]]))
        second=render_inverse_depth(render,camera,Base(),None,torch.zeros(3),moment=2,rigid=Rigid())
        torch.testing.assert_close(second,torch.tensor([[[.25,.0625]]]))
        image.sum().backward();self.assertAlmostEqual(z.grad.item(),-.25,places=6)

    def test_foreground_gradient_corrects_front_occluders_and_missing_surface(self):
        parts=torch.zeros((8,8),dtype=torch.long);target=torch.ones((8,8));moving=torch.ones_like(parts,dtype=torch.bool)
        for value,sign in [(2,1),(.5,-1)]:
            prediction=torch.full((1,16,16),float(value),requires_grad=True)
            loss=inverse_depth_loss(prediction,target,parts,moving);loss.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertGreater(sign*prediction.grad.sum().item(),0)

    def test_clear_background_removes_even_black_occluders(self):
        prediction=torch.ones((1,16,16),requires_grad=True)
        parts=torch.full((8,8),-1,dtype=torch.long)
        loss=inverse_depth_loss(prediction,torch.zeros((8,8)),parts,torch.zeros_like(parts,dtype=torch.bool))
        loss.backward()
        self.assertGreater(prediction.grad.sum().item(),0)

    def test_coverage_closes_surface_holes(self):
        alpha=torch.full((1,16,16),.3,requires_grad=True)
        parts=torch.zeros((8,8),dtype=torch.long)
        loss=inverse_depth_loss(torch.ones((1,16,16)),torch.ones((8,8)),parts,torch.ones_like(parts,dtype=torch.bool),alpha,.25)
        loss.backward()
        self.assertLess(alpha.grad.sum().item(),0)

    def test_spread_rejects_near_far_cloud_with_correct_mean(self):
        parts=torch.zeros((8,8),dtype=torch.long);target=torch.ones((8,8))
        moving=torch.ones_like(parts,dtype=torch.bool);alpha=torch.ones((1,16,16))
        offset=torch.tensor(.2,requires_grad=True)
        near=1+offset;far=1-offset
        mean=((near+far)/2).expand(1,16,16)
        second=((near.square()+far.square())/2).expand_as(mean)
        loss=inverse_depth_loss(mean,target,parts,moving,alpha,0,second,1)
        self.assertAlmostEqual(loss.item(),.04,places=5)
        loss.backward()
        self.assertGreater(offset.grad.item(),0)


if __name__=='__main__':unittest.main()
