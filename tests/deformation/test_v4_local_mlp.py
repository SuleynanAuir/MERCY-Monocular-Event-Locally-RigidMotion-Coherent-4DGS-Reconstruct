import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from tests.deformation.test_v4_joint_trajectory import fixture
from utils.event_refinement import qmul, qrotate
from utils.scgs_v4_field import load_checkpoint
from utils.v4_local_mlp import LocalJointMLP, make_local_basis, save_checkpoint


class LocalMLPTests(unittest.TestCase):
    def make(self):
        data=fixture()
        local,_=make_local_basis(data,[2])
        return LocalJointMLP(data,local)

    def test_zero_network_and_no_scaling_change(self):
        m=self.make().refresh();x=torch.randn(7,3);q=torch.randn(7,4)
        self.assertIs(m.deform(x,q,None,15)[0],x)
        self.assertIs(m.deform_scaling(x,15),x)
        self.assertTrue(torch.equal(m.predict_angles(),torch.zeros_like(m.predict_angles())))

    def nonzero(self):
        m=self.make()
        with torch.no_grad():m.mlp[-1].bias.fill_(.5)
        return m.refresh()

    def test_local_support_anchors_rigidity_and_joint_closure(self):
        m=self.nonzero();self.assertGreater(float(m.angles.abs().max()),0)
        self.assertTrue(torch.all(m.angles[:,:2]==0))
        self.assertTrue(torch.all(m.angles[m.anchors['2'],2]==0))
        before,after=m.trajectories()
        x=torch.cat((torch.tensor([[8.,9.,10.]]),before[20]));q=torch.tensor([[1.,0,0,0.]]).repeat(7,1)
        y,r=m.deform(x,q,None,20)
        self.assertTrue(torch.equal(x[:4],y[:4]));self.assertTrue(torch.equal(q[:4],r[:4]))
        torch.testing.assert_close(torch.cdist(y[4:],y[4:]),torch.cdist(x[4:],x[4:]),atol=3e-7,rtol=1e-6)
        fq=qmul(m.delta_q,m.base_q);ft=qrotate(m.delta_q,m.base_t)+m.delta_t
        pivot=m.pivots[1]
        self.assertLess(float((qrotate(fq[:,1],pivot)+ft[:,1]-qrotate(fq[:,2],pivot)-ft[:,2]).abs().max()),5e-7)

    def test_training_changes_network_not_frozen_motion(self):
        m=self.make();q=m.base_q.clone();basis=m.basis.clone()
        optimizer=torch.optim.Adam(m.mlp.parameters(),lr=.01)
        for _ in range(5):
            optimizer.zero_grad();loss=((m.predict_angles()[:,2]-m.basis[:,0].mean(-1))/1e-5).square().mean();loss.backward();optimizer.step()
        self.assertGreater(float(m.predict_angles().abs().max()),0)
        self.assertTrue(torch.equal(q,m.base_q));self.assertTrue(torch.equal(basis,m.basis))
        self.assertTrue(all(p.grad is not None for p in m.mlp.parameters()))

    def test_checkpoint_recomputes_from_actual_mlp_weights(self):
        m=self.nonzero()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'m.pth';save_checkpoint(path,m,{'test':True})
            other,p=load_checkpoint(path,'cpu')
            self.assertTrue(p['test']);self.assertTrue(torch.equal(m.angles,other.angles))
            for v in other.mlp.parameters():v.data.zero_()
            other.refresh();self.assertTrue(torch.all(other.angles==0))

    def test_protected_group_and_invalid_anchors_rejected(self):
        data=fixture()
        with self.assertRaises(ValueError):make_local_basis(data,[0])
        local,_=make_local_basis(data,[2]);local['basis'][0,0,0]=.001
        with self.assertRaises(ValueError):LocalJointMLP(data,local)

    def test_disabled_local_gate_is_exact_identity(self):
        m=self.nonzero();m.local_settings['group_gains']=[0.];m.refresh()
        self.assertTrue(torch.all(m.angles==0))
        data=fixture();local,_=make_local_basis(data,[2])
        with self.assertRaises(ValueError):LocalJointMLP(data,local,group_gains=[-1.])


if __name__=='__main__':unittest.main()
