import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.check_hard_joint_correction import (check_candidate, inspect_model, inspect_completion,
                                                source_row_counts, sha256)
from utils.joint_refinement import HardJointEventCorrection
from test_joint_refinement import example_poses, example_structure
from test_multiview_completion import record_fixture
from utils.multiview_completion import CompactCompletion
from utils.event_refinement import correction_from_record


class HardJointCheckTests(unittest.TestCase):
    def setUp(self):
        self.structure = example_structure()
        self.model = HardJointEventCorrection(self.structure)
        _, self.poses = example_poses(self.structure)

    def test_complete_sequence_and_locked_scope(self):
        with torch.no_grad():
            self.model.group_angles.fill_(2)
        report = inspect_model(self.model, self.poses)
        self.assertEqual(report['frames_checked'], 5)
        self.assertEqual(report['dynamic_rows_checked_per_frame'], 17)
        self.assertEqual(report['rigid_rows_checked_per_frame'], 15)
        self.assertEqual(report['unchanged_band_rows_checked_per_frame'], 2)
        self.assertEqual(report['trainable_groups'], [3, 4])
        self.assertEqual(report['locked_groups'], [0, 1, 2])
        self.assertEqual(report['affected_groups'], [3, 4])
        self.assertTrue(report['geometry_invariants_passed'])
        self.assertLess(report['max_joint_gap_scene'], 1e-6)

    def test_missing_frame_rejected(self):
        with self.assertRaisesRegex(ValueError, 'complete pose timeline'):
            inspect_model(self.model, self.poses[:-1])

    def test_band_changes_rejected(self):
        original = self.model.forward
        def broken(rest, baked, index=None):
            result = original(rest, baked, index).clone()
            if index:
                result[-1, 0] += .01
            return result
        self.model.forward = broken
        with self.assertRaisesRegex(ValueError, 'Elastic-band rows changed'):
            inspect_model(self.model, self.poses)

    def test_wrong_rigid_row_mapping_rejected(self):
        original = self.model.forward
        def broken(rest, baked, index=None):
            result = original(rest, baked, index).clone()
            if index:
                result[10, 0] += .01
            return result
        self.model.forward = broken
        with self.assertRaisesRegex(ValueError, 'row-mapping tolerance exceeded'):
            inspect_model(self.model, self.poses)

    def test_canonical_first_frame_change_rejected(self):
        original = self.model.forward
        def broken(rest, baked, index=None):
            result = original(rest, baked, index).clone()
            if index == 0:
                result[10, 0] += .01
            return result
        self.model.forward = broken
        with self.assertRaisesRegex(ValueError, 'Canonical first frame changed'):
            inspect_model(self.model, self.poses)

    def test_artifact_hashes_and_unapproved_event_status_are_preserved(self):
        try:
            from plyfile import PlyData, PlyElement
            from scripts.export_lego_event_preview import verify_candidate
            torch.from_numpy(np.zeros(1, np.float32))
        except (ImportError, RuntimeError):
            self.skipTest('Full artifact check requires the AutoDL NumPy/PyTorch/plyfile environment')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = root/'baseline', root/'candidate'
            baseline.mkdir(); candidate.mkdir()
            source = root/'source.ply'
            names = ('x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3')
            vertex = np.empty(17, dtype=[(name, 'f4') for name in names])
            poses = np.asarray([pose.tolist() for pose in self.poses], dtype=np.float32)
            for column, name in enumerate(names):
                vertex[name] = poses[0, :, column]
            PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(source)
            np.save(baseline/'poses.npy', poses)
            np.save(baseline/'dynamic_indices.npy', np.arange(17))
            cache = {'source': str(source), 'frame_count': 5, 'fps': 60, 'dynamic_count': 17, 'gaussian_count': 17}
            (baseline/'cache.json').write_text(json.dumps(cache))
            np.save(candidate/'rendered_gray.npy', np.zeros((5, 2, 2), dtype=np.float16))
            self.structure['input_sha256'] = {'baseline_cache_metadata': sha256(baseline/'cache.json'),
                                              'dynamic_indices': sha256(baseline/'dynamic_indices.npy')}
            record = {'format': 'event_supervised_hard_joint_alignment_v4', 'frame_count': 5,
                      'joint_structure': self.structure, 'group_angles': [0.]*5, 'max_angle_degrees': 2.,
                      'rgb_appearance_parameters_frozen': True}
            (candidate/'correction.json').write_text(json.dumps(record))
            artifacts = {'source_ply': source, 'baseline_cache_metadata': baseline/'cache.json',
                         'baseline_poses': baseline/'poses.npy', 'dynamic_indices': baseline/'dynamic_indices.npy',
                         'rendered_gray': candidate/'rendered_gray.npy', 'correction': candidate/'correction.json'}
            audit = {'format': 'lego_event_motion_audit_v1', 'event_optimization_performed': True,
                     'approved_for_export': False, 'frame_count': 5, 'fps': 60,
                     'intervals': [{'interval': i} for i in range(4)], 'summary': {'all': {'passed': 0, 'count': 4}},
                     'artifact_sha256': {key: sha256(path) for key, path in artifacts.items()}}
            (candidate/'event-audit.json').write_text(json.dumps(audit))
            checked = check_candidate(baseline, candidate, device='cpu')
            self.assertTrue(checked['candidate_artifact_hashes_verified'])
            self.assertTrue(checked['canonical_dynamic_rows_exactly_match_source_ply'])
            self.assertFalse(checked['event_approved_for_export'])
            source.write_bytes(source.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch: source_ply'):
                check_candidate(baseline, candidate, device='cpu')


class CompactCompletionCheckTests(unittest.TestCase):
    def setUp(self):
        self.record = record_fixture()
        groups = np.asarray([0, 3, 3, 4], np.int32)
        arrays = {'canonical_xyz': np.asarray([[0, 0, 0], [1., .4, 0], [2., .3, .1], [3., .2, 0]], np.float32),
                  'canonical_quaternions': np.asarray([[1., 0., 0., 0.]]*4, np.float32),
                  'group_ids': groups, 'part_ids': groups.copy()}
        self.runtime = CompactCompletion({'report': {'prefix_count': 20}, 'arrays': arrays,
                                          'parent_correction': self.record})
        self.model = correction_from_record(self.record)

    def test_full_timeline_checks_rigid_and_static_suffix_rows(self):
        actual = inspect_completion(self.model, self.runtime)
        self.assertEqual(actual['frames_checked'], 5)
        self.assertEqual(actual['suffix_rows_checked_per_frame'], 4)
        self.assertEqual(actual['moving_suffix_rows'], 3)
        self.assertEqual(actual['fixed_root_suffix_rows'], 1)
        self.assertEqual(actual['pair_samples_per_frame'], 1)
        self.assertTrue(actual['canonical_first_frame_exact'])
        self.assertTrue(actual['static_root_rows_exact'])
        self.assertLess(actual['maximum_intragroup_pair_distance_error_scene'], 1e-6)
        self.assertLess(actual['maximum_unit_quaternion_error'], 1e-6)

    def test_broken_canonical_root_mapping_quaternion_or_row_count_is_detected(self):
        original = self.runtime.posed
        def run_fault(kind):
            def broken(frame, model):
                value = original(frame, model).clone()
                if kind == 'canonical' and frame == 0:
                    value[1, 0] += .1
                elif frame == 1:
                    if kind == 'root':
                        value[0, 0] += .1
                    elif kind == 'mapping':
                        value[1, 0] += .1
                    elif kind == 'quaternion':
                        value[1, 3:] *= 1.1
                    elif kind == 'nan':
                        value[1, 0] = float('nan')
                    elif kind == 'rows':
                        value = value[:-1]
                return value
            self.runtime.posed = broken
            inspect_completion(self.model, self.runtime)
        for kind in ('canonical', 'root', 'mapping', 'quaternion', 'nan', 'rows'):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                run_fault(kind)
        self.runtime.posed = original

    def test_static_count_does_not_include_moving_compact_suffix(self):
        counts = source_row_counts(24, 17, 0, {'suffix_rows_checked_per_frame': 4,
                                              'moving_suffix_rows': 3, 'fixed_root_suffix_rows': 1})
        self.assertEqual(counts['source_rows_without_baked_poses'], 7)
        self.assertEqual(counts['compact_suffix_rows'], 4)
        self.assertEqual(counts['total_moving_source_rows'], 20)
        self.assertEqual(counts['total_static_source_rows'], 4)
        self.assertEqual(counts['non_dynamic_source_rows'], 4)
        no_suffix = source_row_counts(20, 17, 3)
        self.assertEqual(no_suffix['total_moving_source_rows'], 14)
        self.assertEqual(no_suffix['total_static_source_rows'], 6)
        for args in ((24, 17, 18), (10, 17, 3), (24, 17, True)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                source_row_counts(*args)


if __name__ == '__main__':
    unittest.main()
