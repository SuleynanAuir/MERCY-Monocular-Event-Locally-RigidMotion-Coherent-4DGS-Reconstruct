import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.gaussian_cleanup import (C0, FORMAT, _sha256, cleanup_contract, require_no_cleanup,
    validate_report_shape, cleanup_eligible_mask, validate_overlay_arrays,
    verify_gaussian_cleanup, apply_gaussian_cleanup)


def contract_fixture():
    return {'format': FORMAT, 'report': '/tmp/report.json', 'report_sha256': 'a'*64,
            'overlay': '/tmp/overlay.npz', 'overlay_sha256': 'b'*64}


def arrays_fixture():
    return {'indices': np.asarray([0, 2], np.int32), 'opacity_logit': np.asarray([-30., .5], np.float32),
            'metadata_json': np.asarray(json.dumps({'format': FORMAT, 'source_sha256': 'c'*64}))}


class GaussianCleanupTests(unittest.TestCase):
    def test_contract_legacy_absence_exact_match_and_export_guard(self):
        contract = contract_fixture(); record = {'gaussian_cleanup': contract}
        self.assertIsNone(cleanup_contract({}, {}))
        self.assertIsNone(verify_gaussian_cleanup({}))
        self.assertEqual(cleanup_contract(record, copy.deepcopy(record)), contract)
        for other in ({}, {'gaussian_cleanup': {**contract, 'report_sha256': 'c'*64}},
                      {'gaussian_cleanup': {**contract, 'extra': 0}}):
            with self.subTest(other=other), self.assertRaises(ValueError):
                cleanup_contract(record, other)
        for key, value in (('overlay', 'relative.npz'), ('report_sha256', 'bad'), ('format', 'other')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                cleanup_contract({'gaussian_cleanup': {**contract, key: value}})
        require_no_cleanup({}, 'test')
        with self.assertRaisesRegex(ValueError, 'restore uncleaned'):
            require_no_cleanup(record, 'test')

    def test_eligibility_excludes_baked_roots_bands_moving_and_new_rows(self):
        structure = {'parents': [-1, 0], 'source_groups': [{'group': 0, 'objects': [0, 1]},
                                                         {'group': 1, 'objects': [2]}]}
        result = cleanup_eligible_mask(8, 5, np.asarray([0, 0, 1, 2, 0, 0], np.int16),
                                       structure, np.asarray([1, 5], np.int32), [1])
        self.assertEqual(np.flatnonzero(result).tolist(), [0, 4])
        for bad in (np.asarray([5, 1], np.int32), np.asarray([1, 1], np.int32),
                    np.asarray([6], np.int32), np.asarray([1.], np.float32)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                cleanup_eligible_mask(8, 5, np.zeros(6, np.int16), structure, bad, [1])
        duplicate = copy.deepcopy(structure); duplicate['source_groups'][1]['objects'].append(0)
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            cleanup_eligible_mask(8, 5, np.zeros(6, np.int16), duplicate, np.asarray([1], np.int32), [])

    def test_arrays_allow_positive_infinity_source_but_only_finite_nonincreasing_values(self):
        report = {'modified_gaussian_count': 2, 'gaussian_count': 5, 'source_sha256': 'c'*64,
                  'attributes': ['opacity']}
        source = np.asarray([np.inf, -np.inf, 1., 2., 3.], np.float32)
        eligible = np.asarray([True, False, True, False, False])
        arrays = arrays_fixture()
        validate_overlay_arrays(arrays, report, source, eligible)
        variants = [('indices', np.asarray([2, 0], np.int32)), ('indices', np.asarray([0, 0], np.int32)),
                    ('indices', np.asarray([0, 2], np.uint32)), ('indices', np.asarray([0, 5], np.int32)),
                    ('indices', np.asarray([0, 3], np.int32)), ('opacity_logit', np.asarray([-30., 2.], np.float32)),
                    ('opacity_logit', np.asarray([-31., .5], np.float32)),
                    ('opacity_logit', np.asarray([-np.inf, .5], np.float32)),
                    ('opacity_logit', np.asarray([np.nan, .5], np.float32)),
                    ('opacity_logit', np.asarray([0., .5], np.float64)),
                    ('opacity_logit', np.asarray([[0.], [.5]], np.float32)),
                    ('metadata_json', np.asarray('{}'))]
        for key, value in variants:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_overlay_arrays({**arrays, key: value}, report, source, eligible)
        with self.assertRaises(ValueError):
            validate_overlay_arrays({**arrays, 'log_scaling': np.zeros((2, 3))}, report, source, eligible)
        good64 = {**arrays, 'indices': arrays['indices'].astype(np.int64)}
        validate_overlay_arrays(good64, report, source, eligible)

    def test_apply_preserves_rows_other_attributes_and_unselected_logits_exactly(self):
        opacity = np.asarray([np.inf, -np.inf, 1., 2., 3.], np.float32)
        original = torch.from_numpy(opacity.copy())[:, None]
        xyz, scale = torch.randn(5, 3), torch.randn(5, 3)
        model = SimpleNamespace(_opacity=original, _xyz=xyz, _scaling=scale)
        verified = {'report': {'modified_gaussian_count': 2, 'gaussian_count': 5, 'source_sha256': 'c'*64,
                               'attributes': ['opacity']},
                    'arrays': arrays_fixture(), 'source_opacity': opacity,
                    'eligible': np.asarray([True, False, True, False, False])}
        value = apply_gaussian_cleanup(model, verified)
        self.assertIs(value, model._opacity)
        self.assertIs(model._xyz, xyz); self.assertIs(model._scaling, scale)
        self.assertEqual(tuple(value.shape), (5, 1))
        self.assertTrue(torch.equal(value[[1, 3, 4]], original[[1, 3, 4]]))
        self.assertTrue(torch.equal(original[:, 0], torch.from_numpy(opacity)))
        self.assertTrue(torch.equal(value[[0, 2], 0], torch.tensor([-30., .5])))
        self.assertFalse(value.requires_grad)
        with self.assertRaisesRegex(ValueError, 'immutable source'):
            apply_gaussian_cleanup(model, verified)
        self.assertIsNone(apply_gaussian_cleanup(model, None))

    def test_dc_arrays_require_explicit_single_mode_gray_bounds_and_finite_canonical_color(self):
        report = {'modified_gaussian_count': 2, 'gaussian_count': 5, 'source_sha256': 'c'*64,
                  'attributes': ['dc_gray']}
        arrays = arrays_fixture(); arrays.pop('opacity_logit'); arrays['dc_gray'] = np.asarray([0., 1.], np.float32)
        opacity = np.asarray([np.inf, -np.inf, 1., 2., 3.], np.float32)
        eligible = np.asarray([True, False, True, False, False]); dc = np.zeros((5, 3), np.float32)
        validate_overlay_arrays(arrays, report, opacity, eligible, dc)
        for values in ([-.01, .5], [0., 1.01], [np.nan, .5], [0., np.inf]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_overlay_arrays({**arrays, 'dc_gray': np.asarray(values, np.float32)},
                                        report, opacity, eligible, dc)
        for bad_dc in (None, dc.astype(np.float64), np.zeros((5, 1, 3), np.float32),
                       np.full((5, 3), np.nan, np.float32)):
            with self.subTest(bad_dc=bad_dc), self.assertRaises(ValueError):
                validate_overlay_arrays(arrays, report, opacity, eligible, bad_dc)
        for wrong_arrays in ({**arrays, 'opacity_logit': np.zeros(2, np.float32)}, arrays_fixture()):
            with self.subTest(keys=list(wrong_arrays)), self.assertRaises(ValueError):
                validate_overlay_arrays(wrong_arrays, report, opacity, eligible, dc)
        for attributes in (None, [], ['opacity', 'dc_gray'], ['dc_rgb'], ['opacity']):
            with self.subTest(attributes=attributes), self.assertRaises(ValueError):
                validate_overlay_arrays(arrays, {**report, 'attributes': attributes}, opacity, eligible, dc)
        wrong_opacity = opacity.copy(); wrong_opacity[4] = np.nan
        with self.assertRaisesRegex(ValueError, 'source opacity'):
            validate_overlay_arrays(arrays, report, wrong_opacity, eligible, dc)
        wrong_row = {**arrays, 'indices': np.asarray([0, 3], np.int32)}
        with self.assertRaisesRegex(ValueError, 'static root'):
            validate_overlay_arrays(wrong_row, report, opacity, eligible, dc)

    def test_dc_apply_is_view_independent_sh_dc_and_preserves_opacity_geometry_other_rows(self):
        opacity = np.asarray([np.inf, -np.inf, 1., 2., 3.], np.float32)
        dc = np.arange(15, dtype=np.float32).reshape(5, 3)/10
        arrays = arrays_fixture(); arrays.pop('opacity_logit'); arrays['dc_gray'] = np.asarray([0., 1.], np.float32)
        verified = {'report': {'modified_gaussian_count': 2, 'gaussian_count': 5, 'source_sha256': 'c'*64,
                               'attributes': ['dc_gray']}, 'arrays': arrays, 'source_opacity': opacity,
                    'source_dc': dc, 'eligible': np.asarray([True, False, True, False, False])}
        original = {'_features_dc': torch.from_numpy(dc.copy())[:, None],
                    '_opacity': torch.from_numpy(opacity.copy())[:, None], '_xyz': torch.randn(5, 3),
                    '_rotation': torch.randn(5, 4), '_scaling': torch.randn(5, 3), '_features_rest': torch.empty(5, 0, 3)}
        model = SimpleNamespace(**original)
        result = apply_gaussian_cleanup(model, verified)
        self.assertIs(result, model._features_dc)
        for key, before in original.items():
            if key != '_features_dc':
                self.assertIs(getattr(model, key), before)
        self.assertTrue(torch.equal(original['_features_dc'][:, 0], torch.from_numpy(dc)))
        self.assertTrue(torch.equal(result[[1, 3, 4]], original['_features_dc'][[1, 3, 4]]))
        rendered_rgb = result[[0, 2], 0] * C0 + .5
        torch.testing.assert_close(rendered_rgb, torch.tensor([[0., 0., 0.], [1., 1., 1.]]), rtol=0, atol=1e-7)
        self.assertFalse(result.requires_grad)
        with self.assertRaisesRegex(ValueError, 'runtime DC'):
            apply_gaussian_cleanup(model, verified)
        model = SimpleNamespace(**original); model._opacity = original['_opacity'].clone(); model._opacity[4] -= .1
        with self.assertRaisesRegex(ValueError, 'runtime opacity'):
            apply_gaussian_cleanup(model, verified)

    def artifact_fixture(self, folder, attribute='opacity'):
        from plyfile import PlyData, PlyElement
        root = Path(folder); baseline = root/'parent-baseline'; candidate = root/'parent-candidate'
        baseline.mkdir(); candidate.mkdir()
        vertices = np.zeros(6, dtype=[('x', 'f4'), ('opacity', 'f4'),
                                     ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')])
        vertices['opacity'][0] = np.inf
        source = root/'source.ply'; PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(source)
        np.save(baseline/'poses.npy', np.zeros((4, 2, 7), np.float32))
        np.save(baseline/'dynamic_indices.npy', np.asarray([1, 3], np.int32))
        np.savez(root/'binding.npz', labels=np.asarray([0, 1, 2, 2], np.int16),
                 dynamic_indices=np.asarray([1, 3], np.int32))
        np.savez(root/'parts.npz', metadata_json=np.asarray(json.dumps({'objects': ['root', 'band', 'moving'],
                                                                       'deforming_objects': [1]})))
        rebuild = {'retained_gaussian_count': 3, 'binding': str(root/'binding.npz'),
                   'binding_sha256': _sha256(root/'binding.npz')}
        (root/'rebuild.json').write_text(json.dumps(rebuild))
        completed = {'parent_report': str(root/'rebuild.json'), 'prefix_count': 4, 'parts': str(root/'parts.npz')}
        (root/'completion.json').write_text(json.dumps(completed))
        completion = {'format': 'lego_multiview_prior_completion_v1', 'report': str(root/'completion.json'),
                      'report_sha256': _sha256(root/'completion.json'), 'extension': str(root/'extension.npz'),
                      'extension_sha256': 'd'*64}
        cache = {'source': str(source), 'source_sha256': _sha256(source), 'gaussian_count': 6,
                 'dynamic_count': 2, 'rendered_dynamic_count': 3, 'frame_count': 4, 'fps': 60, 'camera': {},
                 'multiview_completion': completion}
        (baseline/'cache.json').write_text(json.dumps(cache))
        correction = {'format': 'event_supervised_hard_joint_alignment_v4', 'frame_count': 4,
                      'group_angles': [0., 0.], 'multiview_completion': completion, 'scope': 'unchanged parent',
                      'joint_structure': {'parents': [-1, 0], 'input_sha256': {
                          'baseline_cache_metadata': _sha256(baseline/'cache.json')}, 'source_groups': [
                          {'group': 0, 'objects': [0]}, {'group': 1, 'objects': [2]}]}}
        parent_audit = {'multiview_completion': completion}
        (baseline/'cache.json').write_text(json.dumps(cache)); (candidate/'correction.json').write_text(json.dumps(correction))
        (candidate/'event-audit.json').write_text(json.dumps(parent_audit))
        overlay = root/'overlay.npz'
        payload = ({'opacity_logit': np.asarray([-30.], np.float32)} if attribute == 'opacity'
                   else {'dc_gray': np.asarray([.75], np.float32)})
        np.savez(overlay, indices=np.asarray([0], np.int32), **payload,
                 metadata_json=np.asarray(json.dumps({'format': FORMAT, 'source_sha256': cache['source_sha256']})))
        report = {'format': FORMAT, 'source': str(source), 'source_sha256': cache['source_sha256'],
                  'parent_baseline': str(baseline), 'parent_baseline_cache_sha256': _sha256(baseline/'cache.json'),
                  'parent_candidate': str(candidate), 'parent_correction_sha256': _sha256(candidate/'correction.json'),
                  'parent_event_audit_sha256': _sha256(candidate/'event-audit.json'),
                  'parent_poses_sha256': _sha256(baseline/'poses.npy'),
                  'parent_dynamic_indices_sha256': _sha256(baseline/'dynamic_indices.npy'),
                  'overlay': str(overlay), 'overlay_sha256': _sha256(overlay), 'gaussian_count': 6,
                  'retained_gaussian_count': 3, 'modified_gaussian_count': 1, 'attributes': [attribute],
                  'scope': 'Gaussian attenuation only; new audit required.', 'selection_provenance': {'arbitrary': 'allowed'}}
        (root/'report.json').write_text(json.dumps(report))
        contract = {'format': FORMAT, 'report': str(root/'report.json'), 'report_sha256': _sha256(root/'report.json'),
                    'overlay': str(overlay), 'overlay_sha256': _sha256(overlay)}
        cache['gaussian_cleanup'] = contract; correction['gaussian_cleanup'] = contract
        audit = {'gaussian_cleanup': contract, 'multiview_completion': completion, 'artifact_sha256': {
            'cleanup_report': contract['report_sha256'], 'cleanup_overlay': contract['overlay_sha256'],
            'source_ply': report['source_sha256'], 'baseline_poses': report['parent_poses_sha256'],
            'dynamic_indices': report['parent_dynamic_indices_sha256']}}
        return cache, correction, audit, report

    def test_provenance_calls_full_parent_verifier_and_checks_new_audit_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            cache, record, audit, report = self.artifact_fixture(folder)
            record['joint_structure']['input_sha256'] = {
                'baseline_cache_metadata': report['parent_baseline_cache_sha256']}
            with patch('scripts.events.export_lego_event_preview.verify_candidate') as parent_verify:
                checked = verify_gaussian_cleanup(cache, record, audit, baseline=Path(report['parent_baseline']))
                parent_verify.assert_called_once_with(Path(report['parent_baseline']), Path(report['parent_candidate']),
                                                      allow_unapproved=True)
                self.assertEqual(np.flatnonzero(checked['eligible']).tolist(), [0])
                for missing in ('cleanup_report', 'cleanup_overlay', 'baseline_poses'):
                    changed = copy.deepcopy(audit); changed['artifact_sha256'].pop(missing)
                    with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, 'audit checksum'):
                        verify_gaussian_cleanup(cache, record, changed)
                changed = copy.deepcopy(record); changed['scope'] = 'unrecorded correction change'
                with self.assertRaisesRegex(ValueError, 'inherited hard-joint'):
                    verify_gaussian_cleanup(cache, changed)
                changed = copy.deepcopy(cache); changed['camera'] = {'unexpected': 1}
                with self.assertRaisesRegex(ValueError, 'parent cache'):
                    verify_gaussian_cleanup(changed, record)

    def test_dc_artifact_uses_identical_parent_provenance_and_returns_fresh_source_dc(self):
        with tempfile.TemporaryDirectory() as folder:
            cache, record, audit, report = self.artifact_fixture(folder, attribute='dc_gray')
            with patch('scripts.events.export_lego_event_preview.verify_candidate'):
                checked = verify_gaussian_cleanup(cache, record, audit)
                self.assertEqual(checked['source_dc'].shape, (6, 3))
                self.assertEqual(checked['source_dc'].dtype, np.dtype('float32'))
                self.assertTrue(np.isposinf(checked['source_opacity'][0]))
                model = SimpleNamespace(_features_dc=torch.from_numpy(checked['source_dc'].copy())[:, None],
                                        _opacity=torch.from_numpy(checked['source_opacity'].copy())[:, None])
                opacity = model._opacity
                apply_gaussian_cleanup(model, checked)
                self.assertIs(model._opacity, opacity)
                torch.testing.assert_close(model._features_dc[0, 0]*C0+.5, torch.full((3,), .75))
                wrong_audit = copy.deepcopy(audit); wrong_audit['artifact_sha256'].pop('cleanup_overlay')
                with self.assertRaisesRegex(ValueError, 'audit checksum'):
                    verify_gaussian_cleanup(cache, record, wrong_audit)

    def test_report_rejects_geometry_changes_invalid_counts_and_relative_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, report = self.artifact_fixture(folder)
            validate_report_shape(report)
            validate_report_shape({**report, 'attributes': ['dc_gray']})
            for key, value in (('attributes', ['opacity', 'scaling']), ('attributes', ['opacity', 'dc_gray']),
                               ('attributes', []), ('attributes', ['dc_rgb']), ('modified_gaussian_count', 0),
                               ('retained_gaussian_count', 6), ('gaussian_count', True),
                               ('parent_baseline', 'relative'), ('source_sha256', 'notsha')):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_report_shape({**report, key: value})

    def test_only_child_cache_fingerprint_may_change_and_must_match_supplied_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            cache, record, audit, report = self.artifact_fixture(folder)
            child = Path(folder)/'child-baseline'; child.mkdir()
            (child/'cache.json').write_text(json.dumps(cache))
            for filename in ('poses.npy', 'dynamic_indices.npy'):
                (child/filename).symlink_to(Path(report['parent_baseline'])/filename)
            record['joint_structure']['input_sha256']['baseline_cache_metadata'] = _sha256(child/'cache.json')
            with patch('scripts.events.export_lego_event_preview.verify_candidate'):
                verify_gaussian_cleanup(cache, record, audit, baseline=child)
                changed = copy.deepcopy(record)
                changed['joint_structure']['input_sha256']['baseline_cache_metadata'] = '0'*64
                with self.assertRaisesRegex(ValueError, 'cache fingerprint'):
                    verify_gaussian_cleanup(cache, changed, audit, baseline=child)
                changed = copy.deepcopy(record)
                changed['joint_structure']['input_sha256']['source_ply'] = '0'*64
                with self.assertRaisesRegex(ValueError, 'inherited hard-joint'):
                    verify_gaussian_cleanup(cache, changed, audit, baseline=child)
                changed = copy.deepcopy(record); changed['group_angles'][1] = .001
                with self.assertRaisesRegex(ValueError, 'inherited hard-joint'):
                    verify_gaussian_cleanup(cache, changed, audit, baseline=child)

    def test_overlay_binding_and_nested_parent_tampering_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            cache, record, audit, report = self.artifact_fixture(folder)
            with patch('scripts.events.export_lego_event_preview.verify_candidate'):
                binding = Path(folder)/'binding.npz'; binding.write_bytes(b'tamper')
                with self.assertRaisesRegex(ValueError, 'binding checksum'):
                    verify_gaussian_cleanup(cache, record, audit)
        with tempfile.TemporaryDirectory() as folder:
            cache, record, audit, report = self.artifact_fixture(folder)
            Path(report['overlay']).write_bytes(b'tamper')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch: overlay'):
                verify_gaussian_cleanup(cache, record, audit)
        with tempfile.TemporaryDirectory() as folder:
            cache, record, audit, report = self.artifact_fixture(folder)
            for location, key in ((Path(report['parent_baseline'])/'cache.json', 'parent_baseline_cache_sha256'),
                                  (Path(report['parent_candidate'])/'correction.json', 'parent_correction_sha256'),
                                  (Path(report['parent_candidate'])/'event-audit.json', 'parent_event_audit_sha256')):
                value = json.loads(location.read_text()); value['gaussian_cleanup'] = cache['gaussian_cleanup']
                location.write_text(json.dumps(value)); report[key] = _sha256(location)
            report_path = Path(cache['gaussian_cleanup']['report']); report_path.write_text(json.dumps(report))
            for value in (cache, record, audit):
                value['gaussian_cleanup']['report_sha256'] = _sha256(report_path)
            with patch('scripts.events.export_lego_event_preview.verify_candidate') as parent_verify:
                with self.assertRaisesRegex(ValueError, 'must not already'):
                    verify_gaussian_cleanup(cache, record, audit)
                parent_verify.assert_not_called()


if __name__ == '__main__':
    unittest.main()
