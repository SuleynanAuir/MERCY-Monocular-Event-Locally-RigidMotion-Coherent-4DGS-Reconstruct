import unittest
import numpy as np
from utils.appearance_stage import validate_attribute, temporal_basis, validate_temporal, apply_frame, reset_geometry, apply_geometry


class AppearanceStageTests(unittest.TestCase):
    def test_rotor_response_is_periodic_and_anchored(self):
        from utils.appearance_stage import rotor_basis
        angles=[.3, .9, .9+2*np.pi, .3+2*np.pi]
        basis=rotor_basis(angles,8)
        np.testing.assert_allclose(basis[0],0,atol=1e-6)
        np.testing.assert_allclose(basis[1],basis[2],atol=1e-6)
        np.testing.assert_allclose(basis[3],0,atol=1e-6)
        self.assertLessEqual(np.abs(basis).max(),1.)
        meta=dict(kind='rotor_fourier',degree=8,frame_count=4,angles_rad=angles)
        np.testing.assert_array_equal(validate_temporal(np.zeros((2,8,3),np.float32),meta,2),basis)
        with self.assertRaises(ValueError):rotor_basis([0,np.nan],8)
        with self.assertRaises(ValueError):rotor_basis([0,1],3)
        with self.assertRaises(ValueError):validate_temporal(np.zeros((2,8,3),np.float32),dict(meta,frame_count=5),2)

    def test_geometry_offsets_follow_rotation_and_never_accumulate(self):
        import torch
        from types import SimpleNamespace
        base=torch.tensor([[0.,0.,0.],[2.,0.,0.]])
        model=SimpleNamespace(_xyz=base.clone(),_geometry_base_xyz=base.clone(),
            _geometry_offset_local=torch.tensor([[.01,0.,0.],[.01,0.,0.]]),
            get_rotation=torch.tensor([[1.,0.,0.,0.],[2**-.5,0.,0.,2**-.5]]))
        expected=torch.tensor([[.01,0.,0.],[2.,.01,0.]])
        for _ in range(3):
            reset_geometry(model);apply_geometry(model)
            torch.testing.assert_close(model._xyz,expected,atol=1e-7,rtol=0)
        torch.testing.assert_close(model._geometry_base_xyz,base)

    def test_continuous_basis_and_invalid_parameters(self):
        basis=temporal_basis(5,3)
        np.testing.assert_allclose(basis[0],[-1,1,-1])
        np.testing.assert_allclose(basis[-1],[1,1,1])
        np.testing.assert_allclose(basis[2],[0,-.5,0])
        for count,degree in [(1,3),(5,0),(5,9),(5,True)]:
            with self.assertRaises(ValueError):temporal_basis(count,degree)

    def test_temporal_shape_and_nonfinite_rejected(self):
        meta=dict(kind='legendre',degree=3,frame_count=5)
        self.assertEqual(validate_temporal(np.zeros((2,3,3),np.float32),meta,2).shape,(5,3))
        for x in [np.zeros((2,2,3),np.float32),np.full((2,3,3),np.nan,np.float32)]:
            with self.assertRaises(ValueError):validate_temporal(x,meta,2)

    def test_frame_updates_do_not_accumulate_or_change_geometry(self):
        import torch
        from types import SimpleNamespace
        dc=torch.ones(2,1,3);coefficients=torch.ones(2,3,3)
        basis=torch.tensor(temporal_basis(5,3));xyz=torch.zeros(2,3)
        model=SimpleNamespace(_features_dc=dc,_xyz=xyz,_temporal_appearance=(dc,coefficients,basis))
        apply_frame(model,0);first=model._features_dc.clone()
        apply_frame(model,4);apply_frame(model,0)
        self.assertTrue(torch.equal(first,model._features_dc))
        self.assertIs(model._xyz,xyz)
        self.assertTrue(torch.equal(dc,torch.ones_like(dc)))
        with self.assertRaises(ValueError):apply_frame(model,5)

    def test_preserve_existing_exact_zero_opacity(self):
        parent=np.array([[-np.inf],[np.inf],[0.]],dtype=np.float32)
        validate_attribute('opacity',np.array([[-np.inf],[np.inf],[.5]],dtype=np.float32),parent)

    def test_new_infinity_nan_or_nonopacity_infinity_rejected(self):
        parent=np.zeros((2,1),dtype=np.float32)
        for value in (-np.inf,np.inf,np.nan):
            with self.assertRaises(ValueError):validate_attribute('opacity',np.full_like(parent,value),parent)
        with self.assertRaises(ValueError):validate_attribute('scaling',np.full_like(parent,-np.inf),np.full_like(parent,-np.inf))

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):validate_attribute('opacity',np.zeros((1,2),np.float32),np.zeros((2,1),np.float32))
