"""Appearance refinement must never silently alter head, geometry or base."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/fvv'))
import check_arm_face_invariants as audit


class WholeArmPreservation(unittest.TestCase):
    def test_only_arm_appearance_is_allowed_to_change(self):
        arrays=dict(points=np.zeros((2,3),np.float32),rotations=np.tile([1.,0,0,0],(2,1)).astype(np.float32),scales=np.full((2,3),.002,np.float32),dc=np.zeros((2,1,3),np.float32),rest=np.zeros((2,15,3),np.float32),opacity=np.zeros((2,1),np.float32),temporal_dc=np.zeros((16,2,1,3),np.float32),point_groups=np.array([0,1],np.float32),track_rotations=np.tile([1.,0,0,0],(399,2,1)).astype(np.float32),track_translations=np.zeros((399,2,3),np.float32))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old=root/'old';new=root/'new';old.mkdir();new.mkdir();out=root/'audit.json'
            for d in [old,new]:
                (d/'model.pth').write_bytes(b'unchanged');(d/'effective.yaml').write_text('unchanged')
                np.savez_compressed(d/'rigid-bucket.npz',**arrays)
            def run():
                with patch.object(sys,'argv',['audit','--previous',str(old),'--model',str(new),'--output',str(out)]):audit.main()
            arrays['dc'][1]=.5;arrays['temporal_dc'][:,1]=.1
            np.savez_compressed(new/'rigid-bucket.npz',**arrays);run()
            self.assertTrue(all(json.loads(out.read_text())['checks'].values()))
            for field in ['points','dc','track_translations']:
                changed={k:v.copy() for k,v in arrays.items()};changed[field].flat[0]=.1
                np.savez_compressed(new/'rigid-bucket.npz',**changed)
                with self.assertRaises(AssertionError):run()
            np.savez_compressed(new/'rigid-bucket.npz',**arrays);(new/'model.pth').write_bytes(b'changed')
            with self.assertRaises(AssertionError):run()


if __name__=='__main__':unittest.main()
