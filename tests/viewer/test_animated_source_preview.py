import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.lamp.prepare_deforming_object import bind,basis
from scripts.viewer.prepare_raw_event_references import prepare

class SourceMotion(unittest.TestCase):
    def test_attachment_preserves_offset_and_rigid_rotation(self):
        tri=np.array([[[0.,0,0],[1,0,0],[0,1,0]]]);x=np.array([[.2,.3,.01],[.8,.1,-.02]])
        ids,w,d=bind(x,tri);q=np.einsum('ni,nij->nj',w,tri[ids]);off=x-q
        self.assertTrue(np.allclose(w.sum(1),1))
        r=Rotation.from_euler('z',40,degrees=True).as_matrix();moved=tri@r.T+[1,2,3]
        rotation=basis(moved)@basis(tri).transpose(0,2,1)
        result=np.einsum('ni,nij->nj',w,moved[ids])+np.einsum('nij,nj->ni',rotation[ids],off)
        self.assertTrue(np.allclose(result,x@r.T+[1,2,3]))

    def test_raw_events_use_half_open_rgb_intervals(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'test.mkv').write_bytes(b'video')
            np.save(root/'test_event.npy',np.array([[0,0,.1,1],[1,1,.5,-1],[1,0,.75,1],[0,1,1.,-1]]))
            meta=dict(width=2,height=2,fps=2.,frame_count=3)
            with patch('scripts.viewer.prepare_raw_event_references.probe',return_value=meta),patch('scripts.viewer.prepare_raw_event_references.frames',return_value=iter([np.zeros((2,2,3),np.uint8)]*3)):
                result=prepare(root,'test',root/'output')
            self.assertEqual(result['counts'],[0,2,2])
            self.assertEqual(result['times_seconds'],[0.,.5,1.])
            self.assertEqual(len(list((root/'output').glob('*.png'))),3)

if __name__=='__main__':unittest.main()
