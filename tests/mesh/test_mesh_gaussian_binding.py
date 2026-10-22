import json
import unittest
import numpy as np
from utils.mesh_gaussian_binding import pose_fingerprint, validate_attachment


class BindingTests(unittest.TestCase):
    def fixture(self):
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
        faces=np.array([[0,1,2]],np.int32);groups=np.zeros(3,np.int32)
        xyz=np.array([[.25,.25,.1]],np.float32);q=np.array([[1,0,0,0]],np.float32)
        meta=dict(source_side='left',source_fingerprints={'source':'abc'})
        record=dict(format='mesh_gaussian_binding_v1',mesh_sha256='mesh',gaussian_count=1,
                    canonical_pose_sha256=pose_fingerprint(xyz,q),**meta)
        data=dict(rows=np.array([0]),face_indices=np.array([0]),barycentric=np.array([[.5,.25,.25]],np.float32),
                  offsets=np.array([[0,0,.1]],np.float32),groups=np.array([0]),metadata_json=np.asarray(json.dumps(record)))
        return data,vertices,faces,groups,xyz,q,meta,'mesh'

    def test_volume_offset_is_preserved_instead_of_flattening_appearance(self):
        args=self.fixture();centers,_=validate_attachment(*args)
        np.testing.assert_array_equal(centers,args[4])

    def test_rejects_wrong_mesh_or_gaussian_geometry(self):
        args=list(self.fixture());args[-1]='wrong'
        with self.assertRaisesRegex(ValueError,'does not match'):validate_attachment(*args)
        args=list(self.fixture());args[4][0,0]+=.1
        with self.assertRaisesRegex(ValueError,'does not match'):validate_attachment(*args)

    def test_rejects_cross_group_and_invalid_barycentric_bindings(self):
        args=list(self.fixture());args[3][1]=1
        with self.assertRaisesRegex(ValueError,'motion groups'):validate_attachment(*args)
        args=list(self.fixture());args[0]['barycentric'][0]=[-.1,.6,.5]
        with self.assertRaisesRegex(ValueError,'arrays'):validate_attachment(*args)


if __name__=='__main__':unittest.main()
