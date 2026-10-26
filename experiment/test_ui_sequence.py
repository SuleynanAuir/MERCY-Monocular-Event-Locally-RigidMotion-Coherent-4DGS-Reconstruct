"""CPU-only checks for shared, immutable viewer pose caches."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from utils.gaussian_sequence import GaussianSequence,FORMAT,SPARSE_VERSION,POSE_FIELDS


class SharedPoseTests(unittest.TestCase):
    def test_opt_in_sharing_preserves_validated_frames_and_lifetime(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'poses.npz'
            base=np.zeros(3,dtype=[(k,'f4') for k in POSE_FIELDS]);base['rot_0']=1
            poses=np.zeros((2,4,7),dtype='f4');poses[:,:,3]=1
            poses[:,:,0]=np.arange(4)
            metadata=dict(format=FORMAT,version=SPARSE_VERSION,frame_count=4,
                gaussians_per_frame=3,fps=20,pose_fields=POSE_FIELDS,
                base_vertices_sha256=hashlib.sha256(base.tobytes()).hexdigest(),
                frames=[dict(file=f'frames/frame_{i:04d}.ply',time_seconds=i/20,
                    pose_sha256=hashlib.sha256(poses[:,i].tobytes()).hexdigest()) for i in range(4)])
            np.savez_compressed(path,base_vertices=base,poses=poses,dynamic_indices=np.array([0,2]),
                metadata_json=np.asarray(json.dumps(metadata)),times_seconds=np.arange(4)/20)
            with GaussianSequence(path,share_poses=True) as first:
                retained=first.poses.transpose(1,0,2)
                with GaussianSequence(path,share_poses=True) as second, GaussianSequence(path) as separate:
                    self.assertIs(first.poses,second.poses)
                    self.assertIsNot(first.poses,separate.poses)
                    for i in range(4):
                        self.assertEqual(first.frame_bytes(i),separate.frame_bytes(i))
                        np.testing.assert_array_equal(retained[i],poses[:,i])
                for i in range(4):first.validate_pose_frame(i)
            del first,second,separate
            np.testing.assert_array_equal(retained,poses.transpose(1,0,2))
            with GaussianSequence(path,share_poses=True) as third:
                self.assertTrue(np.shares_memory(third.poses,retained))


if __name__=='__main__':unittest.main()
