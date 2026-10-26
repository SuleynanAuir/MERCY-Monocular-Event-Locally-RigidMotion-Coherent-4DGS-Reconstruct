"""Small mathematical checks, independent of any experiment result."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .protocol import temporal_split,save_json
from .datasets import event_integrity


class ProtocolTests(unittest.TestCase):
    def test_nonblack_background_preserves_dark_foreground(self):
        from .background import estimate,foreground
        frame=np.full((64,64,3),69,np.uint8);frame[24:40,24:40]=0
        record=estimate(frame)
        self.assertEqual(record['rgb'],[69/255]*3)
        mask=foreground(frame.astype(np.float32)/255,record['rgb'])
        self.assertEqual(int(mask.sum()),16*16)
        black=np.zeros_like(frame);black[24:40,24:40]=130
        mask=foreground(black.astype(np.float32)/255,estimate(black)['rgb'])
        self.assertTrue(np.array_equal(mask,(black.astype(np.float32)/255).mean(-1)>8/255))

    def test_guard_and_no_shared_endpoints(self):
        for n,fps in [(400,60),(100,30),(50,30)]:
            split=temporal_split(n,fps)
            used=[]
            for name,ids in split['intervals'].items():
                endpoints={i for a in ids for i in (a,a+1)}
                self.assertTrue(endpoints<=set(split['frames'][name]));used.append(endpoints)
            self.assertFalse(used[0]&used[1] or used[1]&used[2] or used[0]&used[2])

    def test_truncated_events_rejected_before_mmap(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.npy'
            np.save(path,np.array([[1.,2.,0.,1.],[2.,3.,1.,-1.]]))
            self.assertTrue(event_integrity(path,4,4,1.)['valid'])
            with path.open('r+b') as f:f.truncate(path.stat().st_size-4)
            self.assertEqual(event_integrity(path,4,4,1.)['reason'],'truncated_event_array')

    def test_metrics_and_statistical_units(self):
        from .metrics import jerk,holm,paired_effect,image_metrics
        t=np.arange(12)*.1;x=np.zeros((12,1,3));x[:,0,0]=t**3
        self.assertAlmostEqual(jerk(x,t)['value'],6.,places=10)
        identical=np.ones((20,20,3))*.4
        m=image_metrics(identical,identical)
        self.assertEqual(m['rgb_l1'],0);self.assertEqual(m['ssim'],1);self.assertTrue(m['psnr_exact_match'])
        self.assertEqual(paired_effect([1],[.9],-1)['status'],'insufficient_independent_units')
        self.assertAlmostEqual(paired_effect([1]*5,[.8]*5,-1)['p_raw'],.0625)
        self.assertEqual(holm([.01,.04,.03]),[.03,.06,.06])

    def test_zero_mlp_is_exact_baseline_and_has_gradient(self):
        import torch
        from .model import Adapter
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);q=np.tile([1.,0,0,0],(6,2,1)).astype('float32')
            t=np.zeros((6,2,3),np.float32);t[:,1,0]=np.arange(6)*.01
            save_json(path/'prior.json',dict(diagonal=2.,joints=[]))
            np.savez(path/'motion.npz',groups=np.array([0,1]),q=q,t=t,centers=np.zeros((2,3),np.float32),
                envelope=np.tile(np.arange(6)[:,None]/5,(1,2)).astype('float32'),active=np.array([False,True]),sample_indices=np.array([1]))
            a=Adapter(path,'C3');ids=torch.arange(6);r,u=a.transforms(ids)
            self.assertTrue(torch.allclose(r,torch.from_numpy(q)));self.assertTrue(torch.allclose(u,torch.from_numpy(t)))
            u.sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in a.mlp.parameters() if p.grad is not None))
            self.assertGreater(float(a.mlp[-1].bias.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
