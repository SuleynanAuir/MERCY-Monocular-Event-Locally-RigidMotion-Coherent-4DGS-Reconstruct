import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.static_surface_rebuild import (FORMAT, PART_IDS, C0, _sha256, static_surface_contract,
    require_no_static_surface_rebuild, validate_report_shape, retirement_eligible_mask,
    validate_extension_arrays, verify_static_surface_rebuild, StaticSurfaceRebuild)


def fixture():
    xyz = np.zeros((7, 3), np.float32)
    opacity = np.zeros(7, np.float32); opacity[0] = np.inf
    mesh = {'vertices': np.asarray([[0., 0., 0.], [.001, 0., 0.], [0., .001, 0.]], np.float32),
            'triangles': np.asarray([[0, 1, 2]]*5, np.int32), 'part_ids': np.asarray(PART_IDS, np.int32)}
    structure = {'parents': [-1, 0], 'source_groups': [{'group': 0, 'objects': list(PART_IDS)},
                                                     {'group': 1, 'objects': [20]}]}
    labels = np.asarray([11, 12, 142, 20, 20], np.int16)
    dynamic = np.asarray([1, 4], np.int32)
    completion = {'part_ids': np.asarray([11, 20], np.int32), 'group_ids': np.asarray([0, 1], np.int32)}
    eligible, required = retirement_eligible_mask(7, 4, labels, structure, dynamic, [142], completion, 5)
    report = {'source_gaussian_count': 7, 'retained_gaussian_count': 4, 'retired_gaussian_count': 2,
              'added_gaussian_count': 1, 'rendered_gaussian_count': 8, 'source_sha256': 'a'*64,
              'retirement_bounds': [[-.01, -.01, -.01], [.011, .011, .01]]}
    arrays = {'retired_indices': np.asarray([0, 5], np.int32), 'new_xyz': np.asarray([[.0002, .0002, 0]], np.float32),
              'new_quaternions': np.asarray([[1., 0., 0., 0.]], np.float32),
              'new_log_scale': np.log(np.asarray([[.001, .001, .0001]], np.float32)),
              'new_opacity_logit': np.asarray([3.], np.float32), 'new_dc_gray': np.asarray([.75], np.float32),
              'part_ids': np.asarray([11], np.int32), 'triangle_ids': np.asarray([0], np.int32),
              'metadata_json': np.asarray(json.dumps({'format': FORMAT, 'source_sha256': report['source_sha256']}))}
    return report, arrays, xyz, opacity, eligible, required, mesh, labels, dynamic, structure, completion


class StaticSurfaceRebuildTests(unittest.TestCase):
    def test_contract_complete_match_legacy_absence_and_export_guard(self):
        contract = {'format': FORMAT, 'report': '/tmp/report.json', 'report_sha256': 'a'*64,
                    'extension': '/tmp/extension.npz', 'extension_sha256': 'b'*64}
        record = {'static_surface_rebuild': contract}
        self.assertIsNone(static_surface_contract({}, {}))
        self.assertIsNone(verify_static_surface_rebuild({}))
        self.assertEqual(static_surface_contract(record, copy.deepcopy(record)), contract)
        for wrong in ({}, {'static_surface_rebuild': {**contract, 'extra': 1}},
                      {'static_surface_rebuild': {**contract, 'extension': 'relative.npz'}}):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                static_surface_contract(record, wrong)
        require_no_static_surface_rebuild({}, 'test')
        with self.assertRaisesRegex(ValueError, 'omit rebuilt geometry'):
            require_no_static_surface_rebuild(record, 'test')

    def test_retirement_excludes_dynamic_bands_new_nonroof_and_requires_root_parts(self):
        report, arrays, xyz, opacity, eligible, required, mesh, labels, dynamic, structure, completion = fixture()
        self.assertEqual(np.flatnonzero(eligible).tolist(), [0, 5])
        self.assertEqual(required.tolist(), [5])
        changed = copy.deepcopy(completion); changed['group_ids'][0] = 1
        with self.assertRaisesRegex(ValueError, 'not static root'):
            retirement_eligible_mask(7, 4, labels, structure, dynamic, [142], changed, 5)
        changed = copy.deepcopy(structure); changed['source_groups'][0]['objects'].remove(15)
        with self.assertRaisesRegex(ValueError, 'Every rebuilt roof part'):
            retirement_eligible_mask(7, 4, labels, changed, dynamic, [142], completion, 5)

    def test_actual_triangle_support_normals_roi_and_strict_arrays(self):
        report, arrays, xyz, opacity, eligible, required, mesh, *_ = fixture()
        validate_extension_arrays(arrays, report, xyz, opacity, eligible, required, mesh)
        variants = [('retired_indices', np.asarray([0, 5], np.int64)),
                    ('retired_indices', np.asarray([5, 0], np.int32)),
                    ('retired_indices', np.asarray([0, 0], np.int32)),
                    ('retired_indices', np.asarray([0, 6], np.int32)),
                    ('new_xyz', arrays['new_xyz'].astype(np.float64)),
                    ('new_xyz', np.asarray([[.0009, .0009, 0]], np.float32)),
                    ('new_xyz', np.asarray([[.0002, .0002, .00001]], np.float32)),
                    ('new_xyz', np.asarray([[np.nan, .0002, 0]], np.float32)),
                    ('new_quaternions', np.asarray([[2., 0., 0., 0.]], np.float32)),
                    ('new_quaternions', np.asarray([[.70710678, .70710678, 0., 0.]], np.float32)),
                    ('new_log_scale', np.log(np.asarray([[.02, .001, .0001]], np.float32))),
                    ('new_log_scale', np.log(np.asarray([[.001, .001, 1e-7]], np.float32))),
                    ('new_opacity_logit', np.asarray([np.inf], np.float32)),
                    ('new_dc_gray', np.asarray([1.1], np.float32)),
                    ('part_ids', np.asarray([12], np.int32)), ('triangle_ids', np.asarray([5], np.int32)),
                    ('metadata_json', np.asarray('{}'))]
        for key, value in variants:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_extension_arrays({**arrays, key: value}, report, xyz, opacity, eligible, required, mesh)
        for wrong in ({**arrays, 'unknown': np.asarray(0)}, {k: v for k, v in arrays.items() if k != 'triangle_ids'}):
            with self.assertRaises(ValueError):
                validate_extension_arrays(wrong, report, xyz, opacity, eligible, required, mesh)
        outside = xyz.copy(); outside[0, 0] = .1
        with self.assertRaisesRegex(ValueError, 'outside'):
            validate_extension_arrays(arrays, report, outside, opacity, eligible, required, mesh)
        with self.assertRaisesRegex(ValueError, 'actual roof mesh'):
            validate_extension_arrays(arrays, {**report, 'retirement_bounds': [[-1., -1., -1.], [1., 1., 1.]]},
                                      xyz, opacity, eligible, required, mesh)
        lower = opacity.copy(); lower[0] = -np.inf
        with self.assertRaisesRegex(ValueError, 'increase'):
            validate_extension_arrays(arrays, report, xyz, lower, eligible, required, mesh)
        only_original = {**arrays, 'retired_indices': np.asarray([0], np.int32)}
        with self.assertRaisesRegex(ValueError, 'every old roof suffix'):
            validate_extension_arrays(only_original, {**report, 'retired_gaussian_count': 1},
                                      xyz, opacity, eligible, required, mesh)

    def test_geometry_tolerance_and_opposite_normal_are_explicitly_bounded(self):
        report, arrays, xyz, opacity, eligible, required, mesh, *_ = fixture()
        near = copy.deepcopy(arrays); near['new_xyz'][0, 2] = 1e-6
        near['new_quaternions'][0] = [0., 1., 0., 0.]
        validate_extension_arrays(near, report, xyz, opacity, eligible, required, mesh)
        edge = copy.deepcopy(arrays); edge['new_xyz'][0] = [-1e-6, .0002, 0]
        validate_extension_arrays(edge, report, xyz, opacity, eligible, required, mesh)
        edge['new_xyz'][0, 0] = -3e-6
        with self.assertRaisesRegex(ValueError, 'actual roof triangle'):
            validate_extension_arrays(edge, report, xyz, opacity, eligible, required, mesh)

    def test_runtime_only_retires_named_rows_and_appends_fixed_geometry_after_motion(self):
        report, arrays, xyz, opacity, *_ = fixture()
        verified = {'report': report, 'arrays': arrays, 'source_xyz': xyz, 'source_opacity': opacity,
                    'source_quaternions': np.tile(np.asarray([1., 0., 0., 0.], np.float32), (7, 1)),
                    'source_log_scale': np.full((7, 3), -7., np.float32), 'source_dc': np.zeros((7, 3), np.float32)}
        original = {'_xyz': torch.from_numpy(xyz.copy()), '_rotation': torch.from_numpy(verified['source_quaternions'].copy()),
                    '_scaling': torch.from_numpy(verified['source_log_scale'].copy()),
                    '_opacity': torch.from_numpy(opacity.copy())[:, None],
                    '_features_dc': torch.from_numpy(verified['source_dc'].copy())[:, None],
                    '_features_rest': torch.zeros(7, 15, 3)}
        runtime = StaticSurfaceRebuild(verified); model = SimpleNamespace(**original)
        runtime.apply_appearance(model)
        self.assertIsNone(runtime.source)
        self.assertEqual(runtime.rendered_count, 8)
        self.assertIs(model._xyz, original['_xyz']); self.assertIs(model._rotation, original['_rotation'])
        self.assertTrue(torch.equal(model._scaling[:7], original['_scaling']))
        self.assertTrue(torch.equal(model._features_dc[:7], original['_features_dc']))
        self.assertTrue(torch.equal(model._opacity[[1, 2, 3, 4, 6]], original['_opacity'][[1, 2, 3, 4, 6]]))
        self.assertTrue(torch.equal(model._opacity[[0, 5]], torch.full((2, 1), -30.)))
        self.assertTrue(torch.isposinf(original['_opacity'][0, 0]))
        self.assertEqual(tuple(model._features_rest.shape), (8, 15, 3)); self.assertEqual(model._features_rest.count_nonzero(), 0)
        torch.testing.assert_close(model._features_dc[7, 0]*C0+.5, torch.full((3,), .75))
        for frame in range(4):
            moved = original['_xyz'].clone(); moved[1, 0] += frame
            result, q = runtime.append_geometry(moved, original['_rotation'])
            self.assertTrue(torch.equal(result[:7], moved)); self.assertTrue(torch.equal(result[7:], runtime.xyz))
            self.assertTrue(torch.equal(q[:7], original['_rotation'])); self.assertTrue(torch.equal(q[7:], runtime.quaternions))
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            runtime.append_geometry(result, q)
        with self.assertRaisesRegex(ValueError, 'once to a fresh'):
            runtime.apply_appearance(model)
        altered = SimpleNamespace(**original); altered._xyz = original['_xyz'].clone(); altered._xyz[3, 0] = 1
        with self.assertRaisesRegex(ValueError, 'immutable source'):
            StaticSurfaceRebuild(verified).apply_appearance(altered)

    def renderer_fixture(self):
        report, arrays, xyz, opacity, *_ = fixture()
        verified = {'report': report, 'arrays': arrays, 'source_xyz': xyz, 'source_opacity': opacity,
                    'source_quaternions': np.tile(np.asarray([1., 0., 0., 0.], np.float32), (7, 1)),
                    'source_log_scale': np.full((7, 3), -7., np.float32), 'source_dc': np.zeros((7, 3), np.float32)}
        model = SimpleNamespace(_xyz=torch.from_numpy(xyz.copy()),
            _rotation=torch.from_numpy(verified['source_quaternions'].copy()),
            _scaling=torch.from_numpy(verified['source_log_scale'].copy()),
            _opacity=torch.from_numpy(opacity.copy())[:, None],
            _features_dc=torch.from_numpy(verified['source_dc'].copy())[:, None], _features_rest=torch.zeros(7, 0, 3))
        runtime = StaticSurfaceRebuild(verified); runtime.apply_appearance(model)

        def compact_pose(frame, correction):
            return torch.tensor([[0., 0., 0., 1., 0., 0., 0.],
                                 [float(frame), float(frame+1), 0., 1., 0., 0., 0.]])
        completion = SimpleNamespace(prefix_count=5, posed=compact_pose)
        return runtime, model, completion

    def test_actual_motion_renderer_resets_parent_then_compact_then_static_once_per_frame(self):
        from scripts.events import refine_lego_events as module
        runtime, model, completion = self.renderer_fixture()
        renderer = module.MotionRenderer.__new__(module.MotionRenderer)
        renderer.model, renderer.base_xyz, renderer.base_rotation = model, model._xyz.clone(), model._rotation.clone()
        renderer.indices = torch.tensor([1, 4]); renderer.completion = completion; renderer.static_surface = runtime
        renderer.camera, renderer.pipe, renderer.background = None, None, torch.zeros(3)
        renderer.supersample = 1; renderer.luma = torch.full((3, 1, 1), 1/3)
        renderer.posed = lambda frame, correction: torch.tensor([
            [float(frame+2), 0., 0., 1., 0., 0., 0.], [float(frame+3), 0., 0., 1., 0., 0., 0.]])
        for frame in (0, 3, 1, 0):
            def fake_render(camera, gaussian, pipe, background, *args):
                for field in ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest'):
                    self.assertEqual(len(getattr(gaussian, field)), 8)
                self.assertTrue(torch.equal(gaussian._xyz[7:], runtime.xyz))
                self.assertTrue(torch.equal(gaussian._rotation[7:], runtime.quaternions))
                self.assertTrue(torch.equal(gaussian._xyz[5:7], completion.posed(frame, None)[:, :3]))
                self.assertEqual(gaussian._xyz[1, 0].item(), frame+2)
                self.assertEqual(gaussian._xyz[4, 0].item(), frame+3)
                self.assertTrue(torch.equal(gaussian._opacity[[0, 5]], torch.full((2, 1), -30.)))
                return {'render': torch.zeros(3, 2, 2)}
            with patch.object(module, 'render', side_effect=fake_render) as render:
                renderer.image(frame, None)
                render.assert_called_once()
        self.assertTrue(torch.equal(renderer.base_xyz, torch.zeros(7, 3)))
        self.assertEqual(len(renderer.base_rotation), 7)

    def test_actual_comparison_pose_switch_keeps_new_tail_and_left_appearance_isolated(self):
        from utils.comparison_renderer import ComparisonRenderer
        runtime, right_model, completion = self.renderer_fixture()
        right_xyz, right_q = runtime.append_geometry(right_model._xyz, right_model._rotation)
        left_xyz, left_q = torch.full((3, 3), -1.), torch.tensor([[1., 0., 0., 0.]]*3)
        left_scale, left_dc, left_opacity = torch.full((3, 3), -5.), torch.full((3, 1, 3), .1), torch.full((3, 1), 1.25)
        renderer = ComparisonRenderer('unused', 'cpu'); renderer.completion = completion; renderer.correction = None
        renderer.side_geometry = [(left_xyz.clone(), left_q.clone(), left_scale.clone()),
                                  (right_xyz.clone(), right_q.clone(), right_model._scaling.clone())]
        renderer.side_appearance = [(left_dc, left_opacity, torch.empty(3, 0, 3)),
            (right_model._features_dc, right_model._opacity, right_model._features_rest)]
        renderer.indices = [torch.tensor([1]), torch.tensor([1, 4])]
        renderer.posed = lambda side, frame: torch.tensor(
            [[float(frame+2), 0., 0., 1., 0., 0., 0.]]*(1 if side == 0 else 2))
        renderer.model = SimpleNamespace(_xyz=left_xyz.clone(), _rotation=left_q.clone(), _scaling=left_scale.clone(),
                                        _features_dc=left_dc.clone(), _opacity=left_opacity.clone(), _features_rest=torch.empty(3, 0, 3))
        for frame, side in enumerate((1, 0, 1, 1, 0)):
            renderer.apply_side_pose(side, frame)
            for key in ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest'):
                self.assertEqual(len(getattr(renderer.model, key)), 8 if side else 3)
            if side:
                self.assertTrue(torch.equal(renderer.model._xyz[7:], runtime.xyz))
                self.assertTrue(torch.equal(renderer.model._rotation[7:], runtime.quaternions))
                self.assertTrue(torch.equal(renderer.model._xyz[5:7], completion.posed(frame, None)[:, :3]))
                self.assertTrue(torch.equal(renderer.model._opacity[[0, 5]], torch.full((2, 1), -30.)))
            else:
                self.assertTrue(torch.equal(renderer.model._opacity, left_opacity))
                self.assertTrue(torch.equal(renderer.model._features_dc, left_dc))
                self.assertTrue(torch.equal(renderer.model._scaling, left_scale))
        self.assertTrue(torch.equal(renderer.side_geometry[1][0], right_xyz))
        self.assertTrue(torch.equal(renderer.side_geometry[0][0], left_xyz))

    def artifact_fixture(self, root):
        from plyfile import PlyData, PlyElement
        report, arrays, xyz, opacity, eligible, required, mesh, labels, dynamic, structure, completion_arrays = fixture()
        root = Path(root); baseline, candidate = root/'baseline', root/'candidate'; baseline.mkdir(); candidate.mkdir()
        names = ['x', 'y', 'z', 'opacity']+[f'rot_{i}' for i in range(4)]+[f'scale_{i}' for i in range(3)]+[f'f_dc_{i}' for i in range(3)]
        vertices = np.zeros(7, dtype=[(name, 'f4') for name in names]); vertices['rot_0'] = 1
        vertices['opacity'] = opacity
        for i in range(3):
            vertices[f'scale_{i}'] = -7.
        PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(root/'source.ply')
        source_hash = _sha256(root/'source.ply')
        arrays['metadata_json'] = np.asarray(json.dumps({'format': FORMAT, 'source_sha256': source_hash}))
        np.savez(root/'mesh.npz', **mesh)
        np.savez(root/'parts.npz', metadata_json=np.asarray(json.dumps({'objects': [f'part{i}' for i in range(143)], 'deforming_objects': [142]})))
        np.savez(root/'binding.npz', labels=labels, dynamic_indices=dynamic)
        np.savez(root/'compact.npz', **completion_arrays)
        np.savez(root/'extension.npz', **arrays); np.savez(root/'selection.npz', **arrays)
        np.save(baseline/'dynamic_indices.npy', dynamic); np.save(baseline/'poses.npy', np.zeros((4, 2, 7), np.float32))
        rebuild = {'retained_gaussian_count': 4, 'binding': str(root/'binding.npz'), 'binding_sha256': _sha256(root/'binding.npz')}
        (root/'rebuild.json').write_text(json.dumps(rebuild))
        completed = {'parent_report': str(root/'rebuild.json'), 'prefix_count': 5,
                     'parts': str(root/'parts.npz'), 'parts_sha256': _sha256(root/'parts.npz'),
                     'mesh_prior': str(root/'mesh.npz'), 'mesh_prior_sha256': _sha256(root/'mesh.npz')}
        (root/'completion.json').write_text(json.dumps(completed))
        completion = {'format': 'lego_multiview_prior_completion_v1', 'report': str(root/'completion.json'),
                      'report_sha256': _sha256(root/'completion.json'), 'extension': str(root/'compact.npz'),
                      'extension_sha256': _sha256(root/'compact.npz')}
        parent_cache = {'source': str(root/'source.ply'), 'source_sha256': source_hash, 'gaussian_count': 7,
                        'dynamic_count': 2, 'rendered_dynamic_count': 3, 'frame_count': 4, 'fps': 60,
                        'camera': {}, 'multiview_completion': completion}
        (baseline/'cache.json').write_text(json.dumps(parent_cache))
        structure['input_sha256'] = {'baseline_cache_metadata': _sha256(baseline/'cache.json')}
        parent_record = {'format': 'event_supervised_hard_joint_alignment_v4', 'frame_count': 4,
                         'joint_structure': structure, 'group_angles': [0., 0.], 'multiview_completion': completion}
        parent_audit = {'multiview_completion': completion}
        (candidate/'correction.json').write_text(json.dumps(parent_record)); (candidate/'event-audit.json').write_text(json.dumps(parent_audit))
        report.update(format=FORMAT, source=str(root/'source.ply'), source_sha256=source_hash,
                      parent_baseline=str(baseline), parent_candidate=str(candidate), allowed_part_ids=list(PART_IDS),
                      unobserved_geometry_is_prior=True, scope='Actual CAD roof prior; not observed hidden geometry.')
        for key, path in (('mesh_prior', root/'mesh.npz'), ('parts', root/'parts.npz'),
                          ('selection', root/'selection.npz'), ('extension', root/'extension.npz')):
            report[key] = str(path); report[key+'_sha256'] = _sha256(path)
        for key, path in (('parent_baseline_cache_sha256', baseline/'cache.json'), ('parent_correction_sha256', candidate/'correction.json'),
                          ('parent_event_audit_sha256', candidate/'event-audit.json'), ('parent_poses_sha256', baseline/'poses.npy'),
                          ('parent_dynamic_indices_sha256', baseline/'dynamic_indices.npy')):
            report[key] = _sha256(path)
        (root/'report.json').write_text(json.dumps(report))
        contract = {'format': FORMAT, 'report': str(root/'report.json'), 'report_sha256': _sha256(root/'report.json'),
                    'extension': report['extension'], 'extension_sha256': report['extension_sha256']}
        cache, record = copy.deepcopy(parent_cache), copy.deepcopy(parent_record)
        cache['static_surface_rebuild'] = contract; cache['rendered_gaussian_count'] = 8
        record['static_surface_rebuild'] = contract
        audit = {'static_surface_rebuild': contract, 'multiview_completion': completion, 'artifact_sha256': {
            'static_surface_report': contract['report_sha256'], 'static_surface_extension': contract['extension_sha256'],
            'source_ply': source_hash, 'baseline_poses': report['parent_poses_sha256'], 'dynamic_indices': report['parent_dynamic_indices_sha256']}}
        return cache, record, audit, report

    def test_report_bounded_scope_and_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, _, report = self.artifact_fixture(root)
            validate_report_shape(report)
            for key, value in (('allowed_part_ids', [11]), ('unobserved_geometry_is_prior', False),
                               ('added_gaussian_count', 100001), ('retired_gaussian_count', 0),
                               ('rendered_gaussian_count', 7), ('source_gaussian_count', True),
                               ('source', 'relative.ply'), ('parent_poses_sha256', 'bad')):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_report_shape({**report, key: value})

    def test_verifier_binds_parent_motion_selection_and_new_audit(self):
        with tempfile.TemporaryDirectory() as root:
            cache, record, audit, report = self.artifact_fixture(root)
            with patch('scripts.events.export_lego_event_preview.verify_candidate') as parent_verify:
                result = verify_static_surface_rebuild(cache, record, audit, baseline=Path(report['parent_baseline']))
                parent_verify.assert_called_once()
                self.assertEqual(result['source_dc'].shape, (7, 3))
                for missing in ('static_surface_report', 'static_surface_extension', 'baseline_poses'):
                    changed = copy.deepcopy(audit); changed['artifact_sha256'].pop(missing)
                    with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, 'audit checksum'):
                        verify_static_surface_rebuild(cache, record, changed)
                changed = copy.deepcopy(record); changed['group_angles'][1] = .001
                with self.assertRaisesRegex(ValueError, 'hard-joint motion'):
                    verify_static_surface_rebuild(cache, changed, audit)
                changed = copy.deepcopy(cache); changed['camera'] = {'changed': 1}
                with self.assertRaisesRegex(ValueError, 'cache identity'):
                    verify_static_surface_rebuild(changed, record, audit)
                changed = copy.deepcopy(cache); changed['gaussian_cleanup'] = {}
                with self.assertRaisesRegex(ValueError, 'combined'):
                    verify_static_surface_rebuild(changed, record, audit)
                with np.load(report['selection'], allow_pickle=False) as data:
                    selected = {key: data[key] for key in data.files}
                selected['new_dc_gray'][0] = .4
                np.savez(report['selection'], **selected)
                report['selection_sha256'] = _sha256(report['selection'])
                Path(cache['static_surface_rebuild']['report']).write_text(json.dumps(report))
                cache['static_surface_rebuild']['report_sha256'] = _sha256(cache['static_surface_rebuild']['report'])
                with self.assertRaisesRegex(ValueError, 'exact selection'):
                    verify_static_surface_rebuild(cache, record, audit)

    def test_child_cache_hash_is_the_only_mutable_motion_input(self):
        with tempfile.TemporaryDirectory() as root:
            cache, record, audit, report = self.artifact_fixture(root)
            baseline = Path(root)/'child'; baseline.mkdir(); (baseline/'cache.json').write_text(json.dumps(cache))
            for filename in ('poses.npy', 'dynamic_indices.npy'):
                (baseline/filename).symlink_to(Path(report['parent_baseline'])/filename)
            record['joint_structure']['input_sha256']['baseline_cache_metadata'] = _sha256(baseline/'cache.json')
            with patch('scripts.events.export_lego_event_preview.verify_candidate'):
                verify_static_surface_rebuild(cache, record, audit, baseline=baseline)
                changed = copy.deepcopy(record); changed['joint_structure']['input_sha256']['baseline_cache_metadata'] = '0'*64
                with self.assertRaisesRegex(ValueError, 'cache fingerprint'):
                    verify_static_surface_rebuild(cache, changed, audit, baseline=baseline)
                changed = copy.deepcopy(record); changed['joint_structure']['input_sha256']['source_ply'] = '0'*64
                with self.assertRaisesRegex(ValueError, 'hard-joint motion'):
                    verify_static_surface_rebuild(cache, changed, audit, baseline=baseline)


if __name__ == '__main__':
    unittest.main()
