import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

import numpy as np
import torch

from utils.multiview_completion import (FORMAT, CompactCompletion, completion_contract,
    require_no_completion, validate_report_shape, validate_extension_arrays, verify_multiview_completion)
from utils.event_refinement import correction_from_record, qmul, qrotate
from utils.comparison_renderer import ComparisonRenderer, validate_comparison_sources, vertex_rows_sha256
from utils.rigid_checkpoint import sha256_file
from test_joint_refinement import example_structure


def record_fixture():
    structure = example_structure()
    structure['source_groups'] = [{'group': i, 'objects': [i], 'object_names': [f'part{i}']} for i in range(5)]
    return {'format': 'event_supervised_hard_joint_alignment_v4', 'frame_count': 5,
            'joint_structure': structure, 'group_angles': [0., 0., 0., .4, -.2], 'max_angle_degrees': 2.}


def array_fixture():
    return {'canonical_xyz': np.asarray([[1., 2., 3.], [.5, .3, .1]], np.float32),
            'canonical_quaternions': np.asarray([[1., 0., 0., 0.], [1., 0., 0., 0.]], np.float32),
            'part_ids': np.asarray([0, 3], np.int32), 'group_ids': np.asarray([0, 3], np.int32),
            'metadata_json': np.asarray(json.dumps({'format': FORMAT, 'unobserved_geometry_is_prior': True}))}


class MultiviewCompletionTests(unittest.TestCase):
    def test_contract_legacy_exact_match_and_explicit_unsupported_export(self):
        contract = {'format': FORMAT, 'report': '/tmp/report.json', 'report_sha256': 'a'*64,
                    'extension': '/tmp/extension.npz', 'extension_sha256': 'b'*64}
        record = {'multiview_completion': contract}
        self.assertIsNone(completion_contract({}, {}))
        self.assertEqual(completion_contract(record, copy.deepcopy(record)), contract)
        for wrong in ({}, {'multiview_completion': {**contract, 'report_sha256': 'c'*64}},
                      {'multiview_completion': {**contract, 'unrecognized': 1}}):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, 'contract mismatch'):
                completion_contract(record, wrong)
        for key, value in (('report', 'relative.json'), ('extension_sha256', 'bogus')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                completion_contract({'multiview_completion': {**contract, key: value}})
        require_no_completion({}, 'test')
        with self.assertRaisesRegex(ValueError, 'omit moving suffix'):
            require_no_completion(record, 'test')

    def test_extension_arrays_require_exact_dtype_shape_and_part_group_identity(self):
        arrays, record = array_fixture(), record_fixture()
        report = {'added_gaussians': 2}
        parts = {'objects': [f'part{i}' for i in range(6)], 'deforming_objects': [5]}
        validate_extension_arrays(arrays, report, record, parts)
        variants = [('canonical_xyz', np.zeros((2, 3), np.float64)),
                    ('canonical_xyz', np.zeros((1, 3), np.float32)),
                    ('canonical_quaternions', np.zeros((2, 4), np.float32)),
                    ('part_ids', np.asarray([0, 5], np.int32)),
                    ('part_ids', np.asarray([0, 3], np.int64)),
                    ('group_ids', np.asarray([0, -1], np.int32)),
                    ('group_ids', np.asarray([0, 2], np.int32)),
                    ('metadata_json', np.asarray('[]'))]
        for key, replacement in variants:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_extension_arrays({**arrays, key: replacement}, report, record, parts)
        broken = arrays['canonical_xyz'].copy(); broken[1, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_extension_arrays({**arrays, 'canonical_xyz': broken}, report, record, parts)

    def test_legacy_baked_export_rejects_native_sampling_extension_before_output(self):
        from scripts import export_lego_event_preview as module
        contract = {'format': FORMAT, 'report': '/tmp/report.json', 'report_sha256': 'a'*64,
                    'extension': '/tmp/extension.npz', 'extension_sha256': 'b'*64}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'output'
            argv = ['test', '--baseline', folder, '--candidate', folder, '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(module, 'verify_candidate',
                    return_value=({'multiview_completion': contract}, {}, {})), patch.object(module.PlyData, 'read') as read:
                with self.assertRaisesRegex(ValueError, 'omit moving suffix'):
                    module.main()
                read.assert_not_called()
                self.assertFalse(output.exists())

    def test_compact_runtime_uses_same_se3_all_frames_with_exact_root_and_rest(self):
        record = record_fixture()
        runtime = CompactCompletion({'report': {'prefix_count': 20}, 'arrays': array_fixture(),
                                     'parent_correction': record})
        model = correction_from_record(record)
        self.assertEqual(runtime.rest.shape, (2, 7))
        self.assertEqual(runtime.moving_count, 1)
        self.assertIs(runtime.posed(0, model), runtime.rest)
        for frame in range(5):
            pose = runtime.posed(frame, model)
            q, t = model.group_transforms(frame)
            expected = torch.cat((qrotate(q[runtime.groups], runtime.rest[:, :3])+t[runtime.groups],
                                  qmul(q[runtime.groups], runtime.rest[:, 3:])), 1)
            torch.testing.assert_close(pose, expected, atol=1e-6, rtol=1e-6)
            self.assertTrue(torch.equal(pose[0], runtime.rest[0]))
        with torch.no_grad():
            model.group_angles[3] += .1
        with self.assertRaisesRegex(ValueError, 'group angles'):
            runtime.posed(1, model)

    def test_runtime_rejects_wrong_model_or_source_constants(self):
        record = record_fixture()
        runtime = CompactCompletion({'report': {'prefix_count': 20}, 'arrays': array_fixture(),
                                     'parent_correction': record})
        with self.assertRaisesRegex(ValueError, 'hard-joint model'):
            runtime.posed(1, object())
        changed = copy.deepcopy(record); changed['joint_structure']['envelopes'][2][3] *= .5
        with self.assertRaisesRegex(ValueError, 'parent motion'):
            runtime.posed(1, correction_from_record(changed))

    def test_comparison_resets_suffix_when_switching_sides_and_keeps_scale(self):
        record = record_fixture()
        runtime = CompactCompletion({'report': {'prefix_count': 3}, 'arrays': array_fixture(),
                                     'parent_correction': record})
        renderer = ComparisonRenderer('unused', 'cpu')
        renderer.completion, renderer.correction = runtime, correction_from_record(record)
        left = torch.zeros((3, 3)); right = torch.cat((left.clone(), runtime.rest[:, :3]))
        leftq = torch.tensor([[1., 0., 0., 0.]]*3); rightq = torch.cat((leftq.clone(), runtime.rest[:, 3:]))
        renderer.side_geometry = [(left, leftq, torch.full_like(left, -3.)),
                                  (right, rightq, torch.full_like(right, -4.))]
        renderer.model = SimpleNamespace(_xyz=left.clone(), _rotation=leftq.clone(), _scaling=torch.zeros_like(left))
        renderer.indices = [torch.tensor([1]), torch.tensor([1])]
        renderer.posed = lambda side, index: torch.tensor([[float(side), 0., 0., 1., 0., 0., 0.]])
        for side in (1, 0, 1, 0):
            renderer.apply_side_pose(side, 3)
            self.assertEqual(len(renderer.model._xyz), 5 if side else 3)
            if side:
                torch.testing.assert_close(renderer.model._xyz[3:], runtime.posed(3, renderer.correction)[:, :3])
            torch.testing.assert_close(renderer.model._scaling, renderer.side_geometry[side][2])
        self.assertTrue(torch.equal(left, torch.zeros_like(left)))

    def artifact_fixture(self, root):
        from plyfile import PlyData, PlyElement
        names = ('x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'opacity', 'f_rest_0')
        original = np.zeros(24, dtype=[(name, 'f4') for name in names])
        original['x'] = np.arange(24); original['rot_0'] = 1
        retained = np.arange(8, dtype=np.int32)
        parent = np.concatenate((original[retained], original[:12]))
        arrays = array_fixture(); tail = np.zeros(2, dtype=original.dtype)
        for axis, name in enumerate(('x', 'y', 'z')):
            tail[name] = arrays['canonical_xyz'][:, axis]
        for axis, name in enumerate(('rot_0', 'rot_1', 'rot_2', 'rot_3')):
            tail[name] = arrays['canonical_quaternions'][:, axis]
        for name, data in (('source', original), ('parent', parent), ('completed', np.concatenate((parent, tail)))):
            PlyData([PlyElement.describe(data, 'vertex')], text=False).write(root/f'{name}.ply')
        np.save(root/'retained.npy', retained)
        parent_report = {'format': 'lego_prior_surface_rebuild_v1',
                         'source_sha256': sha256_file(root/'source.ply'), 'rebuilt_sha256': sha256_file(root/'parent.ply'),
                         'base_repaired_source': str(root/'source.ply'), 'base_repaired_sha256': sha256_file(root/'source.ply'),
                         'retained_indices_path': str(root/'retained.npy'), 'retained_indices_sha256': sha256_file(root/'retained.npy'),
                         'original_gaussian_count': 24, 'retained_gaussian_count': 8, 'retired_gaussian_count': 16,
                         'rebuilt_gaussians': 12, 'gaussian_count': 20, 'frozen_retained_vertex_sha256': vertex_rows_sha256(original[retained])}
        (root/'parent-report.json').write_text(json.dumps(parent_report))
        record = record_fixture(); (root/'parent-correction.json').write_text(json.dumps(record))
        np.savez_compressed(root/'parts.npz', metadata_json=np.asarray(json.dumps({
            'objects': [f'part{i}' for i in range(6)], 'deforming_objects': [5]})))
        (root/'mesh.npz').write_bytes(b'actual mesh fixture provenance')
        np.savez_compressed(root/'extension.npz', **arrays)
        parent_baseline = root/'parent-baseline'; parent_baseline.mkdir()
        indices = np.arange(17, dtype=np.int32)
        pose = np.column_stack([parent[name][indices] for name in ('x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3')])
        np.save(parent_baseline/'poses.npy', np.repeat(pose[None], 5, axis=0))
        np.save(parent_baseline/'dynamic_indices.npy', indices)
        report = {'format': FORMAT, 'original_gaussian_count': 24, 'prefix_count': 20,
                  'gaussian_count': 22, 'added_gaussians': 2, 'scope': 'Unobserved source mesh prior, not image truth.',
                  'unobserved_geometry_is_prior': True, 'parent_baseline': str(parent_baseline),
                  'parent_poses_sha256': sha256_file(parent_baseline/'poses.npy'),
                  'parent_dynamic_indices_sha256': sha256_file(parent_baseline/'dynamic_indices.npy')}
        for key, filename, hash_key in (
            ('source', 'source.ply', 'source_sha256'), ('completed_source', 'completed.ply', 'completed_sha256'),
            ('parent_source', 'parent.ply', 'parent_source_sha256'), ('parent_report', 'parent-report.json', 'parent_report_sha256'),
            ('parent_correction', 'parent-correction.json', 'parent_correction_sha256'), ('parts', 'parts.npz', 'parts_sha256'),
            ('mesh_prior', 'mesh.npz', 'mesh_prior_sha256'), ('extension', 'extension.npz', 'extension_sha256')):
            report[key] = str(root/filename); report[hash_key] = sha256_file(root/filename)
        (root/'report.json').write_text(json.dumps(report))
        contract = {'format': FORMAT, 'report': str(root/'report.json'), 'report_sha256': sha256_file(root/'report.json'),
                    'extension': str(root/'extension.npz'), 'extension_sha256': sha256_file(root/'extension.npz')}
        cache = {'multiview_completion': contract, 'source': str(root/'completed.ply'), 'gaussian_count': 22,
                 'frame_count': 5, 'fps': 60, 'dynamic_count': 17, 'camera': {}}
        record['multiview_completion'] = contract
        audit = {'multiview_completion': contract, 'artifact_sha256': {
            'source_ply': report['completed_sha256'], 'completion_report': contract['report_sha256'],
            'completion_extension': contract['extension_sha256'], 'baseline_poses': report['parent_poses_sha256'],
            'dynamic_indices': report['parent_dynamic_indices_sha256']}}
        return cache, record, audit, report

    def test_artifact_chain_and_comparison_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            cache, record, audit, report = self.artifact_fixture(Path(folder))
            checked = verify_multiview_completion(cache, record, audit)
            self.assertEqual(checked['report'], report)
            left = {**cache, 'gaussian_count': 24}
            self.assertTrue(validate_comparison_sources(left, cache, audit, report['source_sha256'], report))
            missing = copy.deepcopy(audit); missing['artifact_sha256'].pop('completion_extension')
            with self.assertRaisesRegex(ValueError, 'completion_extension'):
                verify_multiview_completion(cache, record, missing)
            changed = copy.deepcopy(record); changed['group_angles'][3] += .1
            with self.assertRaisesRegex(ValueError, 'inherited hard-joint motion'):
                verify_multiview_completion(cache, changed, audit)
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                verify_multiview_completion(cache, record, {})

    def test_immutable_prefix_and_canonical_binding_cannot_be_bypassed_with_new_hashes(self):
        from plyfile import PlyData, PlyElement
        for field, index in (('x', 0), ('x', 20), ('rot_0', 20), ('f_rest_0', 20)):
            with self.subTest(field=field, index=index), tempfile.TemporaryDirectory() as folder:
                cache, record, audit, report = self.artifact_fixture(Path(folder))
                data = PlyData.read(cache['source'])['vertex'].data.copy()
                data[field][index] += .1
                PlyData([PlyElement.describe(data, 'vertex')], text=False).write(cache['source'])
                report['completed_sha256'] = sha256_file(cache['source'])
                Path(cache['multiview_completion']['report']).write_text(json.dumps(report))
                cache['multiview_completion']['report_sha256'] = sha256_file(cache['multiview_completion']['report'])
                with self.assertRaises(ValueError):
                    verify_multiview_completion(cache)

    def test_report_refuses_missing_prior_disclosure_or_unbounded_count(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, report = self.artifact_fixture(Path(folder))
            validate_report_shape(report)
            for key, value in (('added_gaussians', 600001), ('added_gaussians', True),
                               ('unobserved_geometry_is_prior', False), ('scope', ''), ('prefix_count', 0)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_report_shape({**report, key: value})

    def test_parent_baked_motion_cannot_change_after_contract_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            cache, record, audit, report = self.artifact_fixture(Path(folder))
            path = Path(report['parent_baseline'])/'poses.npy'
            poses = np.load(path); poses[-1, -1, 0] += .1
            np.save(path, poses)
            with self.assertRaisesRegex(ValueError, 'parent motion checksum'):
                verify_multiview_completion(cache, record, audit)


if __name__ == '__main__':
    unittest.main()
