"""Protect exact surface/part ownership and deterministic resampling."""
import unittest
import numpy as np
from utils.object_surface import resample_surface
from utils.fan_surface import attributes


class ObjectSurfaceTests(unittest.TestCase):
    def setUp(self):
        names=['x','y','z','opacity']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]
        names += [f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
        self.base=np.zeros(100,dtype=[(k,'<f4') for k in names])
        self.base['x']=np.linspace(0,.04,100);self.base['y']=np.tile(np.linspace(0,.04,10),10)
        self.base['opacity']=5;self.base['rot_0']=1
        for i in range(3):self.base[f'scale_{i}']=np.log(.002)
        self.tri=np.array([[[0,0,0],[.04,0,0],[0,.04,0]],[[.08,0,.01],[.12,0,.01],[.08,.04,.01]]])
        self.norm=np.tile([0,0,1.],(2,3,1))

    def test_samples_stay_on_their_source_triangle_and_repeat_exactly(self):
        a=resample_surface(self.tri,self.norm,np.array([0,1]),np.array([0,1]),self.base,spacing=.002)
        b=resample_surface(self.tri,self.norm,np.array([0,1]),np.array([0,1]),self.base,spacing=.002)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        x,r,s,alpha,_=attributes(a[0]);labels=a[1]
        for group,offset in [(0,0),(1,.08)]:
            q=x[labels==group];np.testing.assert_allclose(q[:,2],.01*group,atol=1e-8)
            self.assertTrue(np.all(q[:,0]>=offset-1e-8));self.assertTrue(np.all(q[:,1]>=0))
            self.assertTrue(np.all(q[:,0]-offset+q[:,1]<=.040001))
        world=np.einsum('nij,nj->ni',r,a[2]);np.testing.assert_allclose(world,np.tile([0,0,1.],(len(x),1)),atol=1e-6)
        self.assertTrue(np.all(s[:,2]<s[:,0]*.061));self.assertTrue(np.allclose(alpha,.97))

    def test_missing_appearance_and_invalid_normals_rejected(self):
        for change in ('appearance','normal','spacing'):
            base=self.base.copy();norm=self.norm.copy();spacing=.002
            if change=='appearance':base['opacity']=-20
            if change=='normal':norm[0]=0
            if change=='spacing':spacing=0
            with self.subTest(change=change),self.assertRaises(ValueError):
                resample_surface(self.tri,norm,np.array([0,1]),np.array([0,1]),base,spacing=spacing)


if __name__=='__main__':unittest.main()
