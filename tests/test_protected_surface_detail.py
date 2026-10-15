import copy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from utils.protected_surface_detail import (
    FORMAT, C0, SUPPORT_FORMAT, ProtectedSurfaceDetail, protected_surface_contract,
    require_no_protected_surface_detail, target_parts, validate_arrays, additional_guard_views,
    validate_mesh_support, validate_preview, static_support_guard, verify_static_support_guard)


class ProtectedSurfaceTests(unittest.TestCase):
    def test_contract_partial_mismatched_and_legacy(self):
        self.assertIsNone(protected_surface_contract({}, {}))
        contract = {'format': FORMAT, 'report': '/tmp/report.json', 'report_sha256': 'a'*64,
                    'extension': '/tmp/extension.npz', 'extension_sha256': 'b'*64}
        self.assertEqual(protected_surface_contract({'protected_surface_detail': contract}), contract)
        for records in (({'protected_surface_detail': contract}, {}),
                        ({'protected_surface_detail': dict(contract, report='relative')},),
                        ({'protected_surface_detail': dict(contract, extra=1)},)):
            with self.assertRaises(ValueError):
                protected_surface_contract(*records)
        with self.assertRaises(ValueError):
            require_no_protected_surface_detail({'protected_surface_detail': contract}, 'legacy export')

    def test_target_part_whitelist(self):
        np.testing.assert_array_equal(target_parts([0, 11, 141]), [0, 11, 141])
        for value in ([], [1, 1], [2, 1], [True], [142], [-1], [1.0]):
            with self.assertRaises(ValueError):
                target_parts(value)

    def proposal(self):
        selection = {'new_xyz': np.array([[.2, .2, 0], [.4, .2, 0]], np.float32),
            'new_quaternions': np.tile(np.array([1, 0, 0, 0], np.float32), (2, 1)),
            'new_log_scale': np.tile(np.log(np.array([.001, .001, .0001], np.float32)), (2, 1)),
            'new_opacity_logit': np.full(2, 2., np.float32), 'new_dc_gray': np.full(2, .7, np.float32),
            'part_ids': np.full(2, 11, np.int32), 'triangle_ids': np.zeros(2, np.int32),
            'retired_indices': np.array([0, 2], np.int32),
            'metadata_json': np.asarray(json.dumps({'format': 'lego_static_back_detail_proposal_v1', 'target_part_ids': [11]}))}
        preview = {'guarded_remove_add_active': np.array([True, False]),
            'guarded_remove_add_retired_indices': np.array([0], np.int32),
            'guarded_remove_add_maximum_probe_exposure': np.array([0., .02], np.float32),
            'proposed_retired_indices': np.array([0, 2], np.int32),
            'retired_indices': np.array([0, 2], np.int32),
            'retirement_maximum_probe_exposure': np.array([0., 0.], np.float32)}
        plans = [(f, y, 0) for f in (0, 100, 200, 300) for y in (0, 90, 180, 270)]
        plans += [(100, y, 0) for y in (45, 135, 225, 315)]
        plans += [(100, y, p) for p in (-35, 35) for y in range(0, 360, 45)]
        views = [{'view_index': i, 'frame': f, 'yaw': y, 'pitch': p, 'zoom': 1, 'pan_x': 0, 'pan_y': 0}
                 for i, (f, y, p) in enumerate(plans)]
        meta = {'format': 'lego_guarded_roof_detail_preview_v1', 'full_timeline_exposure_scan': True,
            'front_probe_frames': list(range(400)), 'all_motion_frozen': True, 'source_artifacts_changed': False,
            'target_part_ids': [11], 'proposed_new_count': 2, 'threshold': .0001, 'views': views,
            'variants': {'guarded_remove_add': {'probe_converged': True, 'interior_restore_converged': True,
                'interior_restore_history': [{'new_hole_pixels_by_view': [0]*36}], 'active_added_count': 1,
                'retired_count': 1, 'maximum_final_active_exposure': 0.}}}
        return selection, preview, meta

    def test_final_variant_retirements_used_not_global_pre_restore_rows(self):
        arrays, _ = validate_preview(*self.proposal(), 400)
        np.testing.assert_array_equal(arrays['retired_indices'], [0])
        self.assertEqual(len(arrays['new_xyz']), 1)

    def test_empty_final_retirement_allowed(self):
        s, p, m = self.proposal()
        p['guarded_remove_add_retired_indices'] = np.empty(0, np.int32)
        m['variants']['guarded_remove_add']['retired_count'] = 0
        arrays, _ = validate_preview(s, p, m, 400)
        self.assertEqual(len(arrays['retired_indices']), 0)

    def test_sparse_probe_or_nonconverged_or_remaining_holes_rejected(self):
        for change in ('frames', 'scan', 'converge', 'restore', 'holes'):
            s, p, m = self.proposal()
            if change == 'frames': m['front_probe_frames'] = list(range(0, 400, 5))
            if change == 'scan': m['full_timeline_exposure_scan'] = False
            if change == 'converge': m['variants']['guarded_remove_add']['probe_converged'] = False
            if change == 'restore': m['variants']['guarded_remove_add']['interior_restore_converged'] = False
            if change == 'holes': m['variants']['guarded_remove_add']['interior_restore_history'][-1]['new_hole_pixels_by_view'] = [1]
            with self.assertRaises(ValueError, msg=change):
                validate_preview(s, p, m, 400)

    def test_partial_duplicated_or_changed_orbit_guard_rejected(self):
        for change in ('few', 'duplicate', 'frame', 'zoom'):
            s, p, m = self.proposal()
            if change == 'few': m['views'] = m['views'][:5]
            if change == 'duplicate': m['views'][1] = m['views'][0]
            if change == 'frame': m['views'][1]['frame'] = 100
            if change == 'zoom': m['views'][1]['zoom'] = 2
            with self.assertRaisesRegex(ValueError, '36-view'):
                validate_preview(s, p, m, 400)

    def extra_guard(self):
        plans = [(f, y, 0) for f in (50, 150) for y in (0, 90, 180, 270)]
        plans += [(150, y, p) for p in (-60, 60) for y in range(0, 360, 45)]
        return {'additional_guard_views': [{'view_index': i, 'frame': f, 'yaw': y, 'pitch': p,
            'zoom': 1, 'pan_x': 0, 'pan_y': 0} for i, (f, y, p) in enumerate(plans)],
            'additional_guard_masks': '/tmp/exact-masks.npz', 'additional_guard_masks_sha256': 'd'*64,
            'artifact_sha256': {'additional_guard_masks': 'd'*64}}

    def test_additional_guard_requires_all_60_zero_hole_counts(self):
        s, p, m = self.proposal(); m.update(self.extra_guard())
        with self.assertRaises(ValueError): validate_preview(s, p, m, 400)
        m['variants']['guarded_remove_add']['interior_restore_history'][-1]['new_hole_pixels_by_view'] = [0]*60
        validate_preview(s, p, m, 400)
        m['variants']['guarded_remove_add']['interior_restore_history'][-1]['new_hole_pixels_by_view'][-1] = 1
        with self.assertRaises(ValueError): validate_preview(s, p, m, 400)

    def test_additional_guard_exact_plan_and_hashes_required(self):
        self.assertEqual(additional_guard_views({}), [])
        with self.assertRaises(ValueError): additional_guard_views({}, required=True)
        self.assertEqual(len(additional_guard_views(self.extra_guard(), required=True)), 24)
        for change in ('partial', 'hash', 'artifact', 'yaw', 'frame', 'few', 'path'):
            data = self.extra_guard()
            if change == 'partial': del data['additional_guard_masks']
            if change == 'hash': data['additional_guard_masks_sha256'] = 'bad'
            if change == 'artifact': data['artifact_sha256']['additional_guard_masks'] = 'e'*64
            if change == 'yaw': data['additional_guard_views'][21]['yaw'] = 230
            if change == 'frame': data['additional_guard_views'][0]['frame'] = 0
            if change == 'few': data['additional_guard_views'] = data['additional_guard_views'][:23]
            if change == 'path': data['additional_guard_masks'] = 'relative.npz'
            with self.assertRaises(ValueError, msg=change): additional_guard_views(data)

    def test_exposure_actual_maxima_and_retirement_subset_checked(self):
        for change in ('new_exposure', 'old_exposure', 'retired', 'nan', 'count', 'order'):
            s, p, m = self.proposal()
            if change == 'new_exposure': p['guarded_remove_add_maximum_probe_exposure'][0] = .1
            if change == 'old_exposure': p['retirement_maximum_probe_exposure'][0] = .1
            if change == 'retired': p['guarded_remove_add_retired_indices'][0] = 1
            if change == 'nan': p['guarded_remove_add_maximum_probe_exposure'][1] = np.nan
            if change == 'count': m['variants']['guarded_remove_add']['active_added_count'] = 2
            if change == 'order': p['proposed_retired_indices'] = np.array([2, 0], np.int32)
            with self.assertRaises(ValueError, msg=change):
                validate_preview(s, p, m, 400)

    def test_exposure_threshold_cannot_be_relaxed(self):
        for threshold in (1e-4, 5e-5):
            s, p, m = self.proposal()
            m['threshold'] = threshold
            validate_preview(s, p, m, 400)
        for threshold in (float(np.nextafter(1e-4, np.inf)), .001, 0., -1e-4,
                          float('nan'), float('inf'), True):
            s, p, m = self.proposal()
            m['threshold'] = threshold
            with self.assertRaisesRegex(ValueError, 'guard has not converged'):
                validate_preview(s, p, m, 400)

    def support_inputs(self):
        xyz = np.array([[.5, .5, 0], [5.5, .5, 0], [10.5, .5, 0],
                        [-.016, .5, 0], [20, 20, 20], [-.03, .5, 0],
                        [30, 30, 30], [40, 40, 40]], np.float32)
        scales = np.full((8, 3), .0001, np.float32)
        scales[3] = [.0001, .002, .0001]
        quaternion = np.tile(np.array([1, 0, 0, 0], np.float32), (8, 1))
        quaternion[3] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        mesh = {'vertices': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
            [5, 0, 0], [6, 0, 0], [5, 1, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]], np.float32),
            'triangles': np.array([[0, 1, 2]]*5+[[3, 4, 5], [6, 7, 8]], np.int32),
            'part_ids': np.array([11, 12, 13, 14, 15, 63, 64], np.int32)}
        args = (np.arange(8, dtype=np.int32), xyz, np.log(scales), quaternion, mesh,
                np.array([11, 12, 13, 14, 15, 63, 64], np.int32),
                np.array([11, 12, 13, 14, 15, 64], np.int32))
        return args, {'prefix_count': 6, 'suffix_part_ids': np.array([11, 63], np.int32)}

    def test_global_support_preserves_roof_zero_parts_and_exact_suffix(self):
        args, kwargs = self.support_inputs()
        protected, meta = static_support_guard(*args, **kwargs)
        np.testing.assert_array_equal(protected, [0, 1, 3, 6, 7])
        self.assertEqual(meta['zero_added_part_ids'], [63])
        self.assertEqual(meta['protected_prefix_count'], 3)
        self.assertEqual(meta['protected_suffix_count'], 2)
        self.assertFalse(meta['legacy_screen_labels_used'])
        self.assertEqual(meta['sigma'], 4.)
        self.assertEqual(meta['padding'], .01)
        rotated = list(copy.deepcopy(args))
        rotated[3][3] = [1, 0, 0, 0]
        no_rotation, _ = static_support_guard(*rotated, **kwargs)
        self.assertNotIn(3, no_rotation)

    def test_global_support_updates_when_last_active_part_sample_is_removed(self):
        args, kwargs = self.support_inputs()
        changed = list(copy.deepcopy(args))
        changed[-1] = changed[-1][changed[-1] != 64]
        protected, meta = static_support_guard(*changed, **kwargs)
        self.assertEqual(meta['zero_added_part_ids'], [63, 64])
        self.assertIn(2, protected)
        changed[-1] = np.empty(0, np.int32)
        _, empty_meta = static_support_guard(*changed, **kwargs)
        self.assertEqual(empty_meta['zero_added_part_ids'], args[-2].tolist())

    def test_global_support_recomputed_metadata_and_final_retirements(self):
        args, kwargs = self.support_inputs()
        _, meta = static_support_guard(*args, **kwargs)
        permitted = np.array([2, 4, 5], np.int32)
        verify_static_support_guard(meta, args[0], permitted, *args[1:], **kwargs)
        with self.assertRaisesRegex(ValueError, 'globally protected'):
            verify_static_support_guard(meta, args[0], np.array([1, 2], np.int32), *args[1:], **kwargs)
        for key, value in (('sigma', 3.), ('padding', .005), ('protected_count', 0),
                           ('zero_added_part_ids', []), ('protected_sha256', '0'*64),
                           ('legacy_screen_labels_used', True)):
            bad = copy.deepcopy(meta); bad[key] = value
            with self.assertRaisesRegex(ValueError, 'metadata differs'):
                verify_static_support_guard(bad, args[0], permitted, *args[1:], **kwargs)

    def test_preview_support_metadata_uses_final_active_counts(self):
        s, p, m = self.proposal()
        support = {'format': SUPPORT_FORMAT, 'sigma': 4., 'padding': .01,
            'roof_part_ids': [11, 12, 13, 14, 15], 'target_part_ids': [11],
            'zero_added_part_ids': [], 'active_added_counts': {'11': 1},
            'legacy_screen_labels_used': False}
        m['variants']['guarded_remove_add']['global_support_guard'] = support
        validate_preview(s, p, m, 400)
        for key, value in (('sigma', 3.), ('padding', 0.), ('zero_added_part_ids', [11]),
                           ('active_added_counts', {'11': 2}), ('legacy_screen_labels_used', True)):
            changed = copy.deepcopy(m)
            changed['variants']['guarded_remove_add']['global_support_guard'][key] = value
            with self.assertRaisesRegex(ValueError, 'final active part counts'):
                validate_preview(s, p, changed, 400)

    def test_global_support_rejects_unbound_geometry_and_bad_row_schema(self):
        for change in ('order', 'range', 'dtype', 'quaternion', 'finite', 'missing_roof', 'active_part'):
            args, kwargs = self.support_inputs(); args = list(args)
            if change == 'order': args[0][1] = 0
            if change == 'range': args[0][-1] = 8
            if change == 'dtype': args[0] = args[0].astype(np.int64)
            if change == 'quaternion': args[3][0] = 0
            if change == 'finite': args[2][0, 0] = np.nan
            if change == 'missing_roof': args[4]['part_ids'][0] = 64
            if change == 'active_part': args[-1][0] = 99
            with self.assertRaises(ValueError, msg=change):
                static_support_guard(*args, **kwargs)

    def geometry(self):
        arrays, _ = validate_preview(*self.proposal(), 400)
        mesh = {'vertices': np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
                'triangles': np.array([[0, 1, 2]], np.int32), 'part_ids': np.array([11], np.int32)}
        bounds = np.array([[-.01, -.01, -.01], [1.01, 1.01, .01]])
        report = {'source_gaussian_count': 3, 'added_gaussian_count': 1, 'retired_gaussian_count': 1,
            'retirement_bounds': bounds.tolist(), 'allowed_part_ids': [11], 'source_sha256': 'a'*64,
            'input_sha256': {'selection': 'b'*64, 'preview_npz': 'c'*64}}
        arrays['metadata_json'] = np.asarray(json.dumps({'format': FORMAT, 'source_sha256': 'a'*64,
            'selection_sha256': 'b'*64, 'preview_npz_sha256': 'c'*64}))
        return arrays, report, np.zeros((3, 3)), np.array([True, False, False]), mesh

    def test_actual_triangle_normal_and_source_identity_bounds(self):
        args = self.geometry()
        validate_arrays(*args)
        for change in ('off_surface', 'outside_triangle', 'normal', 'part', 'bounds'):
            arrays, report, xyz, eligible, mesh = copy.deepcopy(args)
            if change == 'off_surface': arrays['new_xyz'][0, 2] = .01
            if change == 'outside_triangle': arrays['new_xyz'][0] = [.8, .8, 0]
            if change == 'normal': arrays['new_quaternions'][0] = [0, 1, 0, 0]
            if change == 'part': arrays['part_ids'][0] = 12
            if change == 'bounds': report['retirement_bounds'][0][0] = -.02
            with self.assertRaises(ValueError, msg=change):
                validate_arrays(arrays, report, xyz, eligible, mesh)

    def test_dynamic_dtype_scale_and_gray_rejected(self):
        for change in ('dynamic', 'dtype', 'scale', 'gray', 'unit', 'nan', 'unknown'):
            arrays, report, xyz, eligible, mesh = self.geometry()
            if change == 'dynamic': eligible[0] = False
            if change == 'dtype': arrays['retired_indices'] = arrays['retired_indices'].astype(np.int64)
            if change == 'scale': arrays['new_log_scale'][0] = 1
            if change == 'gray': arrays['new_dc_gray'][0] = 1.1
            if change == 'unit': arrays['new_quaternions'][0] *= 2
            if change == 'nan': arrays['new_opacity_logit'][0] = np.nan
            if change == 'unknown': arrays['unexpected'] = np.zeros(1)
            with self.assertRaises(ValueError, msg=change):
                validate_arrays(arrays, report, xyz, eligible, mesh)

    def runtime(self):
        import torch
        arrays, report, _, _, _ = self.geometry()
        report['rendered_gaussian_count'] = 4
        source = {'report': report, 'arrays': arrays,
            'source_xyz': np.zeros((3, 3), np.float32),
            'source_quaternions': np.tile(np.array([1., 0, 0, 0], np.float32), (3, 1)),
            'source_log_scale': np.full((3, 3), -8., np.float32),
            'source_opacity': np.array([np.inf, .7, -np.inf], np.float32),
            'source_dc': np.full((3, 3), .2, np.float32),
            'residual': {'indices': np.array([1], np.int32), 'delta_gray': np.array([.03], np.float32)}}
        model = SimpleNamespace(_xyz=torch.from_numpy(source['source_xyz']).clone(),
            _rotation=torch.from_numpy(source['source_quaternions']).clone(),
            _scaling=torch.from_numpy(source['source_log_scale']).clone(),
            _opacity=torch.from_numpy(source['source_opacity']).reshape(-1, 1).clone(),
            _features_dc=torch.from_numpy(source['source_dc']).reshape(-1, 1, 3).clone(),
            _features_rest=torch.empty((3, 0, 3)))
        model._features_dc[1] += torch.tensor(.03, dtype=torch.float32)/C0
        return source, model

    def test_runtime_preserves_parent_residual_and_motion_prefix(self):
        import torch
        source, model = self.runtime(); dc = model._features_dc.clone(); scale = model._scaling.clone()
        detail = ProtectedSurfaceDetail(source)
        detail.apply_appearance(model)
        self.assertTrue(torch.equal(model._features_dc[:3], dc))
        self.assertTrue(torch.equal(model._scaling[:3], scale))
        self.assertEqual(model._opacity[0].item(), -30.)
        self.assertEqual(model._opacity[1].item(), source['source_opacity'][1])
        self.assertEqual(model._opacity[2].item(), -float('inf'))
        xyz = torch.ones((3, 3)); q = model._rotation.clone()
        actual, rotation = detail.append_geometry(xyz, q)
        self.assertTrue(torch.equal(actual[:3], xyz)); self.assertTrue(torch.equal(rotation[:3], q))
        self.assertEqual(tuple(actual.shape), (4, 3))
        with self.assertRaises(ValueError): detail.apply_appearance(model)
        with self.assertRaises(ValueError): detail.append_geometry(actual, rotation)

    def test_runtime_rejects_unapplied_or_changed_color_residual(self):
        source, model = self.runtime()
        detail = ProtectedSurfaceDetail(source)
        with self.assertRaises(ValueError): detail.append_geometry(model._xyz, model._rotation)
        model._features_dc[1] = .2
        with self.assertRaisesRegex(ValueError, '_features_dc'):
            detail.apply_appearance(model)


if __name__ == '__main__':
    unittest.main()
