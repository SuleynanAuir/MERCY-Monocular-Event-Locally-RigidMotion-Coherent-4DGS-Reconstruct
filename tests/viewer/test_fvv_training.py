import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scripts.fvv import common

class LegoPreparationTests(unittest.TestCase):
    def test_real_frame_split_and_camera_transform(self):
        spec=importlib.util.spec_from_file_location('prepare_lego_test',Path(__file__).resolve().parents[2]/'scripts/fvv/prepare_lego.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'common':common}):spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'rgb').mkdir()
            samples=[]
            for i in range(30):
                path=root/'rgb'/f'{i}.png';Image.new('RGB',(8,8),(i,10,20)).save(path)
                samples.append(dict(frame=i,time_seconds=i/10,rgb=f'rgb/{i}.png'))
            meta=dict(camera=dict(width=8,height=8,K=[[10,0,4],[0,10,4],[0,0,1]],pose_opengl_c2w=np.eye(4).tolist()),
                      samples=samples,sources={'video':{'sha256':'original-video-record'}})
            reference=root/'reference.json';reference.write_text(json.dumps(meta))
            points=np.zeros(4,dtype=[(k,'f4') for k in ['x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity']])
            points['x']=[0,.1,.2,.3];points['z']=-2
            source=root/'original.ply';PlyData([PlyElement.describe(points,'vertex')]).write(source)
            before=common.digest(source)
            with patch('builtins.print'):module.prepare(SimpleNamespace(reference=reference,source=source,output=root/'prepared',points=1000))
            prepared=root/'prepared';p=json.loads((prepared/'provenance.json').read_text())
            self.assertEqual(p['split_counts'],{'train':24,'val':3,'test':3})
            splits=[{x['file_path'] for x in json.loads((prepared/name).read_text())['frames']} for name in ['transforms_train.json','transforms_test.json','transforms_holdout.json']]
            self.assertEqual(len(set.union(*splits)),30)
            self.assertFalse(splits[0]&splits[1] or splits[0]&splits[2] or splits[1]&splits[2])
            self.assertEqual(common.digest(source),before)
            actual=PlyData.read(prepared/'points3d.ply')['vertex'].data
            self.assertTrue(np.allclose(actual['z'],2))
            self.assertTrue((prepared/'images/000000.png').is_symlink())
            self.assertGreater(p['single_view_extent'],0)
