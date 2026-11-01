import unittest
import numpy as np
from scripts.fvv.seed_covariance import factor_surface_motion,rotation4,static_surface_rotation


class SurfaceSeedCovariance(unittest.TestCase):
    def test_static_quaternion_keeps_thin_axis_on_normal_including_half_turns(self):
        rng=np.random.default_rng(11);normal=np.concatenate((np.eye(3),-np.eye(3),rng.normal(size=(100,3))))
        normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        q=static_surface_rotation(normal);w,x,y,z=q.T
        local_z=np.stack((2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)),axis=-1)
        np.testing.assert_allclose(local_z,normal,atol=3e-7)
        np.testing.assert_allclose(np.linalg.norm(q,axis=1),1,atol=1e-7)

    def test_velocity_and_thin_conditional_surface_survive_factorization(self):
        rng=np.random.default_rng(7);velocity=rng.normal(size=(40,3))*.3
        normals=rng.normal(size=(40,3));normals/=np.linalg.norm(normals,axis=1,keepdims=True)
        scales,left,right=factor_surface_motion(velocity,normals)
        rotation=rotation4(left,right);cov=(rotation*np.exp(scales*2)[:,None,:])@np.swapaxes(rotation,1,2)
        inferred_velocity=cov[:,:3,3]/cov[:,3,3,None]
        conditional=cov[:,:3,:3]-cov[:,:3,3,None]*cov[:,None,3,:3]/cov[:,3,3,None,None]
        normal_variance=np.einsum('ni,nij,nj->n',normals,conditional,normals)
        np.testing.assert_allclose(inferred_velocity,velocity,atol=3e-6)
        np.testing.assert_allclose(normal_variance,.0012**2,rtol=1e-3,atol=1e-9)
        np.testing.assert_allclose(cov[:,3,3],.05**2,rtol=1e-5)

    def test_rejects_invalid_normals(self):
        with self.assertRaises(ValueError):factor_surface_motion(np.zeros((1,3)),np.zeros((1,3)))


if __name__=='__main__':unittest.main()
