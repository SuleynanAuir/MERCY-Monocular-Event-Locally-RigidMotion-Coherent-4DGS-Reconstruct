import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from scipy.spatial import cKDTree
from utils.clay_surface import regularize_part
from utils.viewer_mesh import load_clay_pair
from utils.rigid_checkpoint import sha256_file


class ClaySurfaceTests(unittest.TestCase):
    def test_noise_reduction_small_gap_repair_and_large_opening_preserved(self):
        x,y=np.meshgrid(np.linspace(-.5,.5,41),np.linspace(-.5,.5,41))
        z=np.random.default_rng(7).normal(0,.006,x.size)
        v=np.column_stack([x.ravel(),y.ravel(),z])
        faces=[]
        for j in range(40):
            for i in range(40):
                cx=(x[j,i]+x[j,i+1])/2;cy=(y[j,i]+y[j+1,i])/2
                if (cx-.2)**2+cy**2 < .12**2 or (cx+.2)**2+cy**2 < .026**2: continue
                a=j*41+i;faces.extend([[a,a+1,a+41],[a+1,a+42,a+41]])
        faces=np.asarray(faces);v2,f2,report=regularize_part(v,faces,.02,.045)
        interior=(np.abs(v2[:,:2])<.4).all(1)
        self.assertLess(v2[interior,2].std(),v[np.unique(faces),2].std()*.7)
        self.assertLessEqual(report['actual_max_distance'],.045001)
        self.assertGreater(np.linalg.norm(v2[:,:2]-[.2,0],axis=1).min(),.06)
        self.assertLess(np.linalg.norm(v2[:,:2]-[-.2,0],axis=1).min(),.026)
        self.assertTrue(np.isfinite(v2).all());self.assertTrue((f2>=0).all())
        self.assertLess(f2.max(),len(v2))

    def test_local_patches_fit_planes_and_reject_discontinuous_depth(self):
        from scripts.mesh.patch_clay_surface import local_patch
        x,y=np.meshgrid(np.linspace(-.02,.02,7),np.linspace(-.02,.02,7))
        points=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,-1.)])
        rays=np.array([[0.,0.,-1.],[.01,.01,-1.]])
        result=local_patch(points,rays,.001)
        np.testing.assert_allclose(result[:,2],-1.)
        points[:len(points)//2,2]-=.03
        self.assertIsNone(local_patch(points,rays,.001))

    def test_display_mesh_requires_original_identity_and_exact_motion(self):
        with TemporaryDirectory() as directory:
            p=Path(directory)/'source';p.write_bytes(b'source mesh')
            q=torch.tensor([[[1.,0,0,0]],[[1.,0,0,0]]]);t=torch.zeros(2,1,3)
            source=SimpleNamespace(source_path=p,fingerprints={},correction=None,base_pose=None,target=None,
                motion=(q,t),vertex_groups=torch.tensor([0,0]),deformation=object(),context=None)
            replacement=SimpleNamespace(info={'clay_regularization':{'source_mesh_sha256':sha256_file(p)}},
                motion=(q.clone(),t.clone()),vertex_groups=torch.tensor([0,0,0]),
                vertices=torch.zeros(3,3),faces=torch.tensor([[0,1,2]]),colors=torch.zeros(3,3))
            pair=SimpleNamespace(sides=[replacement,replacement],info={})
            with patch('utils.viewer_mesh.MeshPairRenderer',return_value=pair):
                result=load_clay_pair({},SimpleNamespace(sides=[source,source]))
                self.assertIs(result.sides[0].deformation,source.deformation)
                replacement.motion[1][1,0,0]=.1
                with self.assertRaisesRegex(ValueError,'motion'):load_clay_pair({},SimpleNamespace(sides=[source,source]))
                replacement.motion=(q.clone(),t.clone())
                replacement.info['clay_regularization']['source_mesh_sha256']='wrong'
                with self.assertRaisesRegex(ValueError,'source mesh'):load_clay_pair({},SimpleNamespace(sides=[source,source]))
