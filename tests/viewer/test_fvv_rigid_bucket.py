import unittest
import torch
import numpy as np
import tempfile,zipfile,io
import gc,weakref
from pathlib import Path
from types import SimpleNamespace
try:
    from scripts.fvv.rigid_bucket import Bucket,pose_at,quaternion_rotate,quaternion_product,read_arrays,bucket_only_model
except ModuleNotFoundError:
    from rigid_bucket import Bucket,pose_at,quaternion_rotate,quaternion_product,read_arrays,bucket_only_model


class RigidBucketMotion(unittest.TestCase):
    def test_proxy_releases_frame_tensors_without_cyclic_garbage_collection(self):
        base=SimpleNamespace(get_static_xyz=torch.empty(0,3),get_static_rotation=torch.empty(0,4),get_static_scaling=torch.empty(0,3),get_static_features=torch.empty(0,16,3),get_static_opacity=torch.empty(0,1))
        b=Bucket.__new__(Bucket);b.values=dict(points=torch.zeros(2,3),rotations=torch.tensor([[1.,0,0,0]]).repeat(2,1),scales=torch.ones(2,3),dc=torch.zeros(2,1,3,requires_grad=True),rest=torch.zeros(2,15,3),opacity=torch.zeros(2,1),temporal_dc=torch.zeros(48,2,1,3),track_rotations=torch.tensor([[1.,0,0,0]]).repeat(399,1),track_translations=torch.zeros(399,3))
        enabled=gc.isenabled();gc.disable()
        try:
            for _ in range(8):
                proxy=b.proxy(base,.5);reference=weakref.ref(proxy.get_static_features);del proxy
                self.assertIsNone(reference(), 'Frame tensors must not wait for cyclic GC')
        finally:
            if enabled:gc.enable()

    def test_whole_body_schema_keeps_body_out_of_arm_diagnostics(self):
        arrays=dict(points=np.zeros((8,3),np.float32),rotations=np.tile([1.,0,0,0],(8,1)).astype(np.float32),scales=np.full((8,3),.002,np.float32),
            dc=np.zeros((8,1,3),np.float32),rest=np.zeros((8,15,3),np.float32),opacity=np.zeros((8,1),np.float32),temporal_dc=np.zeros((48,8,1,3),np.float32),
            point_groups=np.arange(8,dtype=np.float32),track_rotations=np.tile([1.,0,0,0],(399,8,1)).astype(np.float32),track_translations=np.zeros((399,8,3),np.float32))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'body.npz';np.savez_compressed(path,**arrays);valid=read_arrays(path,3)
            b=Bucket.__new__(Bucket);b.values={k:torch.from_numpy(v) for k,v in valid.items()}
            torch.testing.assert_close(b.arm_only().values['point_groups'],torch.arange(1,7,dtype=torch.float32))
            self.assertEqual(b.head_only().values['points'].shape,(1,3))
            arrays['track_translations'][0,7,0]=.1;np.savez_compressed(path,**arrays)
            with self.assertRaises(ValueError):read_arrays(path,3)
            arrays['track_translations'][0,7,0]=0
            arrays['point_groups'][-1]=6;np.savez_compressed(path,**arrays)
            with self.assertRaises(ValueError):read_arrays(path,3)

    def test_articulated_groups_align_quaternion_sign_independently(self):
        q=torch.tensor([[[1.,0,0,0],[1.,0,0,0]],[[-1.,0,0,0],[0.,0,0,1.]]])
        rotation,translation=pose_at(q,torch.zeros(2,2,3),1/120)
        torch.testing.assert_close(rotation[0],torch.tensor([1.,0,0,0]))
        points=torch.tensor([[1.,0,0],[1.,0,0]])
        torch.testing.assert_close(quaternion_rotate(rotation,points),torch.tensor([[1.,0,0],[0.,1,0]]),atol=1e-6,rtol=0)

    def test_halfway_pose_preserves_surface_distances(self):
        q=torch.tensor([[1.,0,0,0],[0.,0,0,1.]])
        t=torch.tensor([[0.,0,0],[2.,0,0]])
        rotation,translation=pose_at(q,t,1/120)
        points=torch.tensor([[1.,0,0],[0.,1,0],[0.,0,0]])
        actual=quaternion_rotate(rotation,points)+translation
        torch.testing.assert_close(actual,torch.tensor([[1.,1,0],[0.,0,0],[1.,0,0]]),atol=1e-6,rtol=0)
        torch.testing.assert_close(torch.cdist(actual,actual),torch.cdist(points,points))

    def test_quaternion_sign_does_not_make_motion_flip(self):
        q=torch.tensor([[1.,0,0,0],[-1.,0,0,0]])
        rotation,translation=pose_at(q,torch.zeros(2,3),1/120)
        torch.testing.assert_close(rotation,torch.tensor([1.,0,0,0]))
        points=torch.tensor([[.2,.4,.8]])
        torch.testing.assert_close(quaternion_rotate(rotation,points),points)

    def test_composition_and_boundaries(self):
        q=torch.tensor([.70710678,0.,0.,.70710678]);p=torch.tensor([[1.,0,0]])
        torch.testing.assert_close(quaternion_rotate(quaternion_product(q,q),p),torch.tensor([[-1.,0,0]]),atol=1e-6,rtol=0)
        tracks=torch.stack((torch.tensor([1.,0,0,0]),q));t=torch.tensor([[0.,0,0],[1.,2,3]])
        torch.testing.assert_close(pose_at(tracks,t,100)[1],t[-1])
        torch.testing.assert_close(pose_at(tracks,t,-100)[1],t[0])
        with self.assertRaises(ValueError):pose_at(tracks,t,float('nan'))

    def test_group_archive_indices_and_bucket_extraction(self):
        arrays=dict(points=np.zeros((2,3),np.float32),rotations=np.tile([1.,0,0,0],(2,1)).astype(np.float32),scales=np.full((2,3),.002,np.float32),
            dc=np.zeros((2,1,3),np.float32),rest=np.zeros((2,15,3),np.float32),opacity=np.zeros((2,1),np.float32),temporal_dc=np.zeros((16,2,1,3),np.float32),
            point_groups=np.array([0,1],np.float32),track_rotations=np.tile([1.,0,0,0],(399,2,1)).astype(np.float32),track_translations=np.zeros((399,2,3),np.float32))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'arm.npz';np.savez_compressed(path,**arrays)
            valid=read_arrays(path,2)
            b=Bucket.__new__(Bucket);b.values={k:torch.from_numpy(v) for k,v in valid.items()}
            head=b.head_only()
            arm=b.arm_only()
            torch.testing.assert_close(arm.values['point_groups'],torch.tensor([1.]))
            self.assertIs(arm.values['track_rotations'],b.values['track_rotations'])
            arm.values['dc'].add_(2)
            torch.testing.assert_close(b.values['dc'],torch.zeros(2,1,3))
            self.assertEqual(head.values['points'].shape,(1,3));self.assertEqual(head.values['track_rotations'].shape,(399,4))
            torch.testing.assert_close(head.values['temporal_dc'],b.values['temporal_dc'][:,:1])
            b.values['track_translations'][:,1,0]=2
            base=SimpleNamespace(get_static_xyz=torch.empty(0,3),get_static_rotation=torch.empty(0,4),get_static_scaling=torch.empty(0,3),get_static_features=torch.empty(0,16,3),get_static_opacity=torch.empty(0,1))
            proxy=b.proxy(base,.123)
            torch.testing.assert_close(proxy.get_static_xyz,torch.tensor([[0.,0,0],[2.,0,0]]))
            for invalid in [.5,-1,2]:
                arrays['point_groups'][1]=invalid;np.savez_compressed(path,**arrays)
                with self.assertRaises(ValueError):read_arrays(path,2)

    def test_proxy_keeps_existing_geometry_and_appearance_exact(self):
        base=SimpleNamespace(get_static_xyz=torch.tensor([[9.,8,7]]),get_static_rotation=torch.tensor([[1.,0,0,0]]),
            get_static_scaling=torch.ones(1,3),get_static_features=torch.ones(1,16,3),get_static_opacity=torch.ones(1,1),sentinel=42)
        b=Bucket.__new__(Bucket)
        b.values=dict(points=torch.tensor([[1.,0,0]]),rotations=torch.tensor([[1.,0,0,0]]),scales=torch.ones(1,3),
            dc=torch.zeros(1,1,3,requires_grad=True),rest=torch.zeros(1,15,3),opacity=torch.zeros(1,1),
            temporal_dc=torch.zeros(16,1,1,3,requires_grad=True),track_rotations=torch.tensor([[1.,0,0,0]]).repeat(399,1),track_translations=torch.zeros(399,3))
        proxy=b.proxy(base,.5)
        for key in vars(base):
            if key!='sentinel':torch.testing.assert_close(getattr(proxy,key)[:1],getattr(base,key),rtol=0,atol=0)
        self.assertEqual(proxy.sentinel,42)
        proxy.get_static_features.sum().backward()
        self.assertTrue((b.values['dc'].grad>0).all())
        self.assertGreater(float(b.values['temporal_dc'].grad.abs().sum()),0)
        torch.testing.assert_close(base.get_static_xyz,torch.tensor([[9.,8,7]]))

    def test_isolated_diagnostic_does_not_mutate_scene_opacity(self):
        base=SimpleNamespace(get_opacity=torch.ones(3,1,requires_grad=True),get_static_opacity=torch.ones(2,1),get_xyz=torch.randn(3,3))
        diagnostic=bucket_only_model(base)
        self.assertEqual(float(diagnostic.get_opacity.sum()),0.)
        self.assertEqual(float(diagnostic.get_static_opacity.sum()),0.)
        self.assertIs(diagnostic.get_xyz,base.get_xyz)
        diagnostic.get_opacity.add_(4.)
        torch.testing.assert_close(base.get_opacity,torch.ones(3,1))
        torch.testing.assert_close(base.get_static_opacity,torch.ones(2,1))

    def test_npz_roundtrip_and_invalid_rotations(self):
        arrays=dict(points=np.zeros((2,3),np.float32),rotations=np.tile([1.,0,0,0],(2,1)).astype(np.float32),scales=np.full((2,3),.002,np.float32),
            dc=np.zeros((2,1,3),np.float32),rest=np.zeros((2,15,3),np.float32),opacity=np.zeros((2,1),np.float32),temporal_dc=np.zeros((16,2,1,3),np.float32),
            track_rotations=np.asfortranarray(np.tile([1.,0,0,0],(399,1)).astype(np.float32)),track_translations=np.zeros((399,3),np.float32))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bucket.npz';np.savez_compressed(path,**arrays)
            np.testing.assert_array_equal(read_arrays(path)['points'],arrays['points'])
            arrays['rotations'][:]=0;np.savez_compressed(path,**arrays)
            with self.assertRaises(ValueError):read_arrays(path)

    def test_npz_rejects_large_declared_shape_before_allocating(self):
        names=['points','rotations','scales','dc','rest','opacity','temporal_dc','track_rotations','track_translations']
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bucket.npz'
            with zipfile.ZipFile(path,'w') as archive:
                for name in names:
                    stream=io.BytesIO();np.lib.format.write_array_header_1_0(stream,dict(descr='<f4',fortran_order=False,shape=(2**30,3)))
                    archive.writestr(name+'.npy',stream.getvalue())
            with self.assertRaises(ValueError):read_arrays(path)


if __name__=='__main__':unittest.main()
