"""V4 must preserve the shared V3 model, including after a render error."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.scgs_v4_viewer import SCGSV4ComparisonRenderer
from utils.viewer_state import validate_request
from viewer import read_scene_catalog


class V4ViewerTests(unittest.TestCase):
    def test_catalog_requires_an_earlier_comparison_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'base.json').write_text('{}')
            (folder/'v4.json').write_text('{}')
            base = dict(id='lego', name='V3', type='comparison', path='base.json')
            v4 = dict(id='v4', name='V4', type='scgs_v4', path='v4.json', base_scene='lego')
            path = folder/'scenes.json'
            path.write_text(json.dumps(dict(scenes=[base, v4], default='v4')))
            scenes, default = read_scene_catalog(path)
            self.assertEqual(default, 'v4')
            self.assertEqual(scenes['v4']['base_scene'], 'lego')
            for invalid in ([v4, base], [{**base, 'type': 'sequence'}, v4]):
                path.write_text(json.dumps(dict(scenes=invalid, default='v4')))
                with self.assertRaisesRegex(ValueError, 'earlier V3'):
                    read_scene_catalog(path)

    def fixture(self):
        model = SimpleNamespace(_xyz=torch.zeros(4, 3), _rotation=torch.zeros(4, 4), _scaling=torch.zeros(4, 3))
        pose = np.eye(4)
        pose[2, 3] = 3
        base = SimpleNamespace(model=model, base_pose=pose, target=np.zeros(3),
                               side_sampling=[{}, {'render_supersample': 1}], pipe=None,
                               backgrounds={'black': torch.zeros(3)},
                               correction=SimpleNamespace(group_transforms=lambda frame: (torch.ones(1, 4), None)),
                               packet=lambda request, jpeg, *args, **kwargs: jpeg)
        def reset(side, frame):
            model._xyz.fill_(0)
            model._rotation.fill_(0)
            model._scaling.fill_(0)
        base.apply_side_pose = reset
        renderer = SCGSV4ComparisonRenderer('unused.json', 'cpu', base)
        renderer.field = SimpleNamespace(deform=lambda x, q, groups, frame: (x+1, q+1),
                                          deform_scaling=lambda scale, frame: scale+2)
        request = validate_request(dict(frame=1, width=256, height=256, background='black'), 400)
        return renderer, request

    def render_case(self, fail=False):
        renderer, request = self.fixture()
        calls = []
        def render(camera, model, *args):
            calls.append((model._xyz.clone(), model._scaling.clone()))
            if fail and len(calls) == 2:
                raise RuntimeError('simulated raster failure')
            return {'render': torch.full((3, 256, 256), .2*len(calls))}
        modules = {'gaussian_renderer': SimpleNamespace(render=render),
                   'render_ply': SimpleNamespace(make_camera=lambda *args: None)}
        with patch.dict(sys.modules, modules):
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'raster failure'):
                    renderer.render(request)
            else:
                self.assertTrue(renderer.render(request).startswith(b'\xff\xd8'))
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(calls[0][0], torch.zeros(4, 3)))
        self.assertTrue(torch.equal(calls[1][0], torch.ones(4, 3)))
        self.assertTrue(torch.equal(calls[1][1], torch.full((4, 3), 2.)))
        self.assertEqual(float(renderer.base.model._xyz.abs().sum()), 0)
        self.assertEqual(float(renderer.base.model._rotation.abs().sum()), 0)
        self.assertEqual(float(renderer.base.model._scaling.abs().sum()), 0)

    def test_both_panels_start_at_v3_and_shared_state_restored(self):
        self.render_case()

    def test_render_failure_also_restores_shared_state(self):
        self.render_case(fail=True)


if __name__ == '__main__':
    unittest.main()
