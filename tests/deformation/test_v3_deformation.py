"""Integration invariants: source identity, one residual, all display modes."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.deformation import test_v4_local_mlp
from utils.comparison_renderer import ComparisonRenderer
from utils.rigid_checkpoint import sha256_file
from utils.v3_deformation import deform_mesh_vertices, load_v3_deformation
from utils.viewer_state import validate_request


class V3DeformationTests(unittest.TestCase):
    def renderer(self):
        r = ComparisonRenderer('unused', 'cpu')
        r.model = SimpleNamespace(_xyz=torch.zeros(7, 3), _rotation=torch.zeros(7, 4),
                                  _scaling=torch.zeros(7, 3))
        r.side_geometry = [(torch.full((7, 3), float(i)), torch.zeros(7, 4), torch.zeros(7, 3))
                           for i in (0, 2)]
        r.indices = [torch.arange(7), torch.arange(7)]
        r.posed = lambda side, frame: torch.cat((torch.full((7, 3), float(side * 2)), torch.zeros(7, 4)), 1)
        r.correction = SimpleNamespace(group_transforms=lambda frame: (None, None))
        r.deformation = SimpleNamespace(deform=lambda x, q, g, f: (x+1, q+1),
                                        deform_scaling=lambda s, f: s)
        return r

    def test_v1_unchanged_v3_applied_once_even_after_repeated_seek(self):
        r = self.renderer()
        for _ in range(3):
            r.apply_side_pose(1, 20)
            self.assertTrue(torch.equal(r.model._xyz, torch.full((7, 3), 3.)))
            r.apply_side_pose(0, 20)
            self.assertTrue(torch.equal(r.model._xyz, torch.zeros(7, 3)))
        r.apply_side_pose(1, 20, auxiliary=False)
        self.assertTrue(torch.equal(r.model._xyz, torch.full((7, 3), 2.)))

    def test_hybrid_deforms_after_binding_exactly_once(self):
        r = self.renderer()
        r.base_pose = np.eye(4); r.base_pose[2, 3] = 3
        r.target = np.zeros(3); r.pipe = None; r.backgrounds = {'black': torch.zeros(3)}
        r.hybrid = [SimpleNamespace(apply=lambda model, frame: model._xyz.fill_(10)) for _ in range(2)]
        r.packet = lambda request, jpeg, *args, **kw: jpeg
        seen = []
        def render(camera, model, *args):
            seen.append(model._xyz.clone())
            return {'render': torch.zeros(3, 256, 256)}
        modules = {'gaussian_renderer': SimpleNamespace(render=render),
                   'scripts.rendering.render_ply': SimpleNamespace(make_camera=lambda *args: None)}
        request = validate_request(dict(frame=20, width=256, height=256, background='black',
                                        render_mode='hybrid', hybrid_quality='native'), 400)
        with patch.dict(sys.modules, modules):
            r.render(request)
        self.assertTrue(torch.equal(seen[0], torch.full((7, 3), 10.)))
        self.assertTrue(torch.equal(seen[1], torch.full((7, 3), 11.)))

    def test_mesh_and_gaussians_share_rigid_residual_and_anchors(self):
        field = test_v4_local_mlp.LocalMLPTests().nonzero()
        before, _ = field.trajectories()
        frame = int(field.angles.abs().amax(1).argmax())
        xyz = before[frame]
        mesh = deform_mesh_vertices(field, xyz, field.groups, frame)
        full = torch.cat((torch.zeros(1, 3), xyz))
        rotation = torch.tensor([[1., 0, 0, 0]]).repeat(len(full), 1)
        gs, _ = field.deform(full, rotation, None, frame)
        self.assertTrue(torch.equal(mesh, gs[1:]))
        protected = field.groups == 1
        self.assertTrue(torch.equal(mesh[protected], xyz[protected]))
        self.assertIs(deform_mesh_vertices(field, xyz, field.groups, 0), xyz)
        selected = field.groups == 2
        torch.testing.assert_close(torch.cdist(mesh[selected], mesh[selected]),
                                   torch.cdist(xyz[selected], xyz[selected]), atol=3e-7, rtol=1e-6)

    def test_loader_rejects_wrong_source_hash_timeline_and_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp).resolve(); (p/'model.pth').write_bytes(b'checkpoint')
            audit = dict(frame_count=3, approved_for_export=False,
                         summary={'all': {}, 'validation_events': {}}, failed_intervals=[],
                         intervals=[dict(interval=i, passed=True, reasons=[], reference_self_check_passed=True)
                                    for i in range(2)])
            (p/'audit.json').write_text(json.dumps(audit))
            settings = dict(format='v3_deformation_v1', checkpoint='model.pth', audit='audit.json',
                            checkpoint_sha256=sha256_file(p/'model.pth'), audit_sha256=sha256_file(p/'audit.json'))
            manifest = p/'manifest.json'; manifest.write_text(json.dumps(settings))
            source = dict(baseline=str(p/'base'), candidate=str(p/'candidate'), artifact_sha256={'source':'hash'})
            provenance = dict(baseline=source['baseline'], candidate=source['candidate'],
                              baseline_artifact_sha256=source['artifact_sha256'], field_kind='local-joint-mlp-v4',
                              enabled_auxiliary_joints=[2], parameters=100, final_geometry={})
            field = SimpleNamespace(settings={'frame_count':3}, rows=torch.tensor([1]))
            r = SimpleNamespace(verified_v3=source, info={'frame_count':3}, side_geometry=[None, [torch.zeros(2,3)]])
            with patch('utils.v3_deformation.load_checkpoint', return_value=(field, provenance)):
                _, info = load_v3_deformation(manifest, r)
                self.assertTrue(info['deformation']['enabled'])
                self.assertFalse(info['event_validation']['approved'])
                provenance['candidate'] = str(p/'wrong')
                with self.assertRaisesRegex(ValueError, 'different V3'): load_v3_deformation(manifest, r)
                provenance['candidate'] = source['candidate']
                field.settings['frame_count'] = 4
                with self.assertRaisesRegex(ValueError, 'timeline'): load_v3_deformation(manifest, r)
                field.settings['frame_count'] = 3
                provenance['field_kind'] = 'old-scale-only'
                with self.assertRaisesRegex(ValueError, 'trained local'): load_v3_deformation(manifest, r)
                (p/'model.pth').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum'): load_v3_deformation(manifest, r)


if __name__ == '__main__':
    unittest.main()
