import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.package_lego_appearance import (apply_attributes, rows_sha256, share_motion_file,
    verify_appearance_only, verify_structure_unchanged, verify_identical_motion)


def vertices():
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
             'f_rest_0', 'scale_0', 'scale_1', 'scale_2', 'opacity', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    result = np.zeros(6, dtype=[(name, '<f4') for name in names])
    result['x'] = np.arange(6)
    result['rot_0'] = 1
    result['opacity'][:2] = [np.inf, -np.inf]
    return result


def attributes():
    return {'dc_gray': np.array([.1, .2, .3], np.float32),
            'log_scale': np.full((3, 3), -4., np.float32),
            'opacity_logit': np.full(3, 2., np.float32)}


class AppearancePackageTests(unittest.TestCase):
    def test_tail_attributes_only_and_frozen_inf_logits(self):
        before = vertices()
        after = apply_attributes(before, 3, attributes())
        self.assertEqual(rows_sha256(before[:3]), rows_sha256(after[:3]))
        for axis in range(3):
            expected = (attributes()['dc_gray'] - np.float32(.5)) / np.float32(.28209479177387814)
            np.testing.assert_array_equal(after[f'f_dc_{axis}'][3:], expected)
            np.testing.assert_array_equal(after[f'scale_{axis}'][3:], -4.)
        np.testing.assert_array_equal(before['x'], after['x'])
        self.assertEqual(float(before['opacity'][-1]), 0.)

    def test_invalid_shapes_types_nonfinite_or_extra_fields_rejected(self):
        for key, value in [('dc_gray', np.zeros(2, np.float32)),
                           ('log_scale', np.zeros((3, 3), np.float64)),
                           ('opacity_logit', np.full(3, np.nan, np.float32)),
                           ('dc_gray', np.full(3, -.2, np.float32)),
                           ('log_scale', np.full((3, 3), 1000., np.float32))]:
            bad = attributes(); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                apply_attributes(vertices(), 3, bad)
        bad = attributes(); bad['xyz'] = np.zeros((3, 3), np.float32)
        with self.assertRaises(ValueError):
            apply_attributes(vertices(), 3, bad)

    def test_protected_seed_fields_and_every_retained_field_rejected(self):
        before = vertices()
        for name, index in [('x', 5), ('rot_0', 4), ('f_rest_0', 3), ('f_dc_0', 1), ('scale_0', 2)]:
            bad = apply_attributes(before, 3, attributes())
            bad[name][index] += 1
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_appearance_only(before, bad, 3)

    def test_only_structure_input_hashes_may_change(self):
        before = {'row_groups': [0, 1], 'input_sha256': {'binding': 'old'}, 'joint_tolerance': 1e-5}
        after = copy.deepcopy(before); after['input_sha256']['binding'] = 'new'
        verify_structure_unchanged(before, after)
        after['row_groups'] = [1, 0]
        with self.assertRaises(ValueError):
            verify_structure_unchanged(before, after)

    def test_motion_storage_does_not_change_source_permissions_or_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / 'source.npy'
            np.save(original, np.arange(5, dtype=np.float32))
            before, mode = original.read_bytes(), original.stat().st_mode
            for storage in ('symlink', 'copy'):
                target = root / (storage + '.npy')
                share_motion_file(original, target, storage)
                self.assertEqual(target.read_bytes(), before)
                self.assertEqual(original.stat().st_mode, mode)
                self.assertEqual(original.read_bytes(), before)
                if storage == 'copy':
                    self.assertEqual(target.stat().st_mode & 0o222, 0)
                with self.assertRaises(FileExistsError):
                    share_motion_file(original, target, storage)

    def test_actual_all_row_all_frame_motion_identity(self):
        from test_joint_refinement import example_structure, example_poses
        structure = example_structure()
        _, poses = example_poses(structure)
        record = {'format': 'event_supervised_hard_joint_alignment_v4',
                  'frame_count': structure['frame_count'], 'joint_structure': structure,
                  'group_angles': [0., 0., 0., .5, -.3], 'max_angle_degrees': 2.}
        old = json.loads(json.dumps(record)); new = copy.deepcopy(old)
        new['joint_structure']['input_sha256'] = {'binding': 'newhash'}
        report = verify_identical_motion(old, new, np.stack([pose.numpy() for pose in poses]))
        self.assertTrue(report['xyz_and_quaternions_byte_identical'])
        self.assertEqual(report['frames_checked'], 5)
        new['group_angles'][-1] += .1
        with self.assertRaises(ValueError):
            verify_identical_motion(old, new, np.stack([pose.numpy() for pose in poses]))


if __name__ == '__main__':
    unittest.main()
