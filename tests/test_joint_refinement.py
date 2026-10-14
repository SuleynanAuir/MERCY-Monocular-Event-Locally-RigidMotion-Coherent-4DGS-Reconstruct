import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from utils.event_refinement import correction_from_record, qmul, qrotate
from utils.joint_refinement import HardJointEventCorrection


def example_structure():
    frames, groups = 5, 5
    angle = torch.linspace(0, 1, frames)[:, None]*torch.tensor([0., .4, .8, 1.1, 1.5])[None]
    zero = torch.zeros_like(angle)
    quaternion = torch.stack(((angle/2).cos(), zero, zero, (angle/2).sin()), -1)
    pairs = [[0, 1], [1, 2], [1, 3], [3, 4], [0, 2]]
    pivots = torch.tensor([[0., 0, 0], [0., 0, 0], [1., 0, 0], [1.5, .2, 0], [0., 0, 0]])
    parents, parent_joints = [-1, 0, 1, 1, 3], [-1, 0, 1, 2, 3]
    translation = torch.zeros(frames, groups, 3)
    for group in range(1, groups):
        parent, pivot = parents[group], pivots[parent_joints[group]]
        translation[:, group] = qrotate(quaternion[:, parent], pivot)+translation[:, parent]-qrotate(quaternion[:, group], pivot)
    envelope = torch.linspace(0, 1, frames).square()[:, None].expand(-1, groups).clone()
    envelope[:, 0] = 0
    return {'format': 'source_hard_joint_structure_v1', 'frame_count': frames, 'joint_tolerance': 1e-5,
            'row_groups': [group for group in range(groups) for _ in range(3)]+[-1, -1],
            'parents': parents, 'parent_joints': parent_joints, 'joint_groups': pairs,
            'joint_pivots': pivots.tolist(), 'joint_axes': [[0., 0, 1]]*len(pairs),
            'source_quaternions': quaternion.tolist(), 'source_translations': translation.tolist(),
            'envelopes': envelope.tolist(), 'locked_groups': [0, 1, 2], 'loop_edges': [4]}


def example_poses(structure):
    torch.manual_seed(40)
    rest = torch.cat((torch.randn(len(structure['row_groups']), 3)*.2,
                      torch.tensor([1., 0., 0., 0.]).expand(len(structure['row_groups']), -1)), -1)
    poses = []
    for frame in range(structure['frame_count']):
        pose = rest.clone()
        for row, group in enumerate(structure['row_groups']):
            if group >= 0:
                q = torch.tensor(structure['source_quaternions'][frame][group])
                t = torch.tensor(structure['source_translations'][frame][group])
                pose[row, :3] = qrotate(q, rest[row, :3])+t
                pose[row, 3:] = qmul(q, rest[row, 3:])
            else:
                pose[row, :3] += .01*frame*torch.tensor([1., -1., 2.])
        poses.append(pose)
    return rest, poses


class HardJointCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.structure = example_structure()
        self.rest, self.poses = example_poses(self.structure)

    def test_zero_residual_matches_source_and_bands_exact(self):
        model = HardJointEventCorrection(self.structure)
        for frame, baked in enumerate(self.poses):
            actual = model(self.rest, baked, frame)
            torch.testing.assert_close(actual, baked, atol=1e-6, rtol=1e-6)
            self.assertTrue(torch.equal(actual[-2:], baked[-2:]))

    def test_fixed_root_tree_and_loop_edges_under_large_random_parameters(self):
        model = HardJointEventCorrection(self.structure)
        with torch.no_grad():
            model.group_angles.copy_(torch.tensor([100., -100., 100., -100., 100.]))
        for frame, baked in enumerate(self.poses):
            actual = model(self.rest, baked, frame)
            self.assertTrue(torch.equal(actual[:3], self.rest[:3]))
            self.assertLess(float(model.joint_errors(frame).max()), 1e-6)
            torch.testing.assert_close(actual[3:9], baked[3:9], atol=1e-6, rtol=1e-6)
        self.assertGreater(float((actual[9:15]-self.poses[-1][9:15]).abs().max()), .001)

    def test_each_rigid_group_retains_pairwise_shape(self):
        model = HardJointEventCorrection(self.structure)
        with torch.no_grad():
            model.group_angles.normal_()
        for frame, baked in enumerate(self.poses):
            actual = model(self.rest, baked, frame)
            for group in range(5):
                rows = slice(group*3, group*3+3)
                torch.testing.assert_close(torch.cdist(actual[rows, :3], actual[rows, :3]),
                                           torch.cdist(self.rest[rows, :3], self.rest[rows, :3]), atol=1e-6, rtol=1e-6)

    def test_gradients_only_on_unlocked_off_cycle_branches(self):
        model = HardJointEventCorrection(self.structure)
        actual = model(self.rest, self.poses[3], 3)
        (actual.square().mean()+model.regularization()).backward()
        self.assertTrue(torch.isfinite(model.group_angles.grad).all())
        self.assertTrue(torch.equal(model.group_angles.grad[:3], torch.zeros(3)))
        self.assertTrue(torch.all(model.group_angles.grad[3:].abs() > 1e-8))

    def test_first_frame_returns_canonical_exactly(self):
        model = HardJointEventCorrection(self.structure)
        with torch.no_grad():
            model.group_angles.normal_()
        self.assertIs(model(self.rest, self.poses[4], 0), self.rest)

    def test_json_structure_and_parameter_round_trip(self):
        model = HardJointEventCorrection(self.structure)
        with torch.no_grad():
            model.group_angles.normal_()
        record = json.loads(json.dumps({'joint_structure': self.structure, 'group_angles': model.group_angles.tolist()}))
        loaded = HardJointEventCorrection(record['joint_structure'])
        with torch.no_grad():
            loaded.group_angles.copy_(torch.tensor(record['group_angles']))
        torch.testing.assert_close(loaded(self.rest, self.poses[4], 4), model(self.rest, self.poses[4], 4))

    def test_unified_loader_round_trip_and_invalid_angles_timeline(self):
        model = HardJointEventCorrection(self.structure)
        with torch.no_grad():
            model.group_angles.normal_()
        record = {'format': 'event_supervised_hard_joint_alignment_v4', 'frame_count': 5,
                  'joint_structure': self.structure, 'group_angles': model.group_angles.tolist(), 'max_angle_degrees': 2.}
        loaded = correction_from_record(json.loads(json.dumps(record)))
        torch.testing.assert_close(loaded(self.rest, self.poses[4], 4), model(self.rest, self.poses[4], 4))
        for key, value in (('group_angles', [0.]*4), ('group_angles', [0., 0., 0., float('nan'), 0.]),
                           ('group_angles', [0., 0., 0., float('inf'), 0.]), ('frame_count', 6)):
            broken = copy.deepcopy(record); broken[key] = value
            with self.assertRaises(ValueError):
                correction_from_record(broken)

    def test_invalid_timeline_rows_and_nonfinite_pose(self):
        model = HardJointEventCorrection(self.structure)
        for frame in (None, True, -1, 5, .5):
            with self.assertRaises(ValueError):
                model(self.rest, self.poses[0], frame)
        with self.assertRaises(ValueError):
            model(self.rest[:-1], self.poses[1], 1)
        invalid = self.rest.clone(); invalid[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            model(invalid, self.poses[1], 1)

    def test_invalid_lock_topology_axis_and_root_are_rejected(self):
        for key, replacement in (('locked_groups', [0]), ('parents', [-1, 2, 1, 1, 3]),
                                  ('joint_tolerance', .001), ('joint_axes', [[0., 0, 0]]*5)):
            broken = copy.deepcopy(self.structure); broken[key] = replacement
            with self.assertRaises(ValueError):
                HardJointEventCorrection(broken)
        broken = copy.deepcopy(self.structure); broken['source_translations'][2][0][0] = .01
        with self.assertRaises(ValueError):
            HardJointEventCorrection(broken)
        with self.assertRaises(ValueError):
            HardJointEventCorrection(self.structure, max_angle_degrees=20)

    def test_unclosed_source_loop_is_refused_not_softly_penalized(self):
        broken = copy.deepcopy(self.structure)
        broken['joint_pivots'][4] = [1., 0., 0.]
        with self.assertRaisesRegex(ValueError, 'closure tolerance exceeded'):
            HardJointEventCorrection(broken)

    def test_builder_registration_bands_and_locked_structure(self):
        try:
            from scipy.spatial.transform import Rotation
            from scripts.lego_joint_structure import build_structure
        except ImportError:
            self.skipTest('Source-archive builder requires working SciPy; checked on AutoDL')
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frames, count = self.structure['frame_count'], 6
            matrix = np.tile(np.eye(4), (frames, count, 1, 1))
            quats = np.asarray(self.structure['source_quaternions'])
            matrix[:, :5, :3, :3] = Rotation.from_quat(quats[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(frames, 5, 3, 3)
            matrix[:, :5, :3, 3] = self.structure['source_translations']
            labels = np.asarray([g if g >= 0 else 5 for g in self.structure['row_groups']])
            np.savez(folder/'parts.npz', object_matrices=matrix, rest_points=self.rest[:, :3].numpy(),
                     point_parts=labels, metadata_json=np.asarray(json.dumps({'objects': [f'p{i}' for i in range(count)],
                                                                              'deforming_objects': [5]})))
            registration = np.eye(4); registration[:3, :3] *= 2; registration[:3, 3] = [.3, -.2, .1]
            indices = np.arange(len(labels))
            np.savez(folder/'binding.npz', labels=labels, dynamic_indices=indices, registration=registration)
            np.save(folder/'dynamic_indices.npy', indices)
            posed = np.stack([pose.numpy() for pose in self.poses])
            posed[:, :, :3] = posed[:, :, :3]*2+registration[:3, 3]
            np.save(folder/'poses.npy', posed)
            (folder/'cache.json').write_text('{}')
            groups = [{'group': i, 'objects': [i], 'representative_object': i, 'object_names': [f'p{i}'], 'static': i == 0}
                      for i in range(5)]
            report = {'format': 'source_rigid_joint_diagnosis_v1', 'frames_checked': frames, 'groups': groups,
                      'source_parts_sha256': hashlib.sha256((folder/'parts.npz').read_bytes()).hexdigest(),
                      'candidate_joints': [{'groups': pair, 'rest_pivot': pivot, 'rest_axis': [0, 0, 1], 'rank': 2}
                                           for pair, pivot in zip(self.structure['joint_groups'], self.structure['joint_pivots'])],
                      'graph': {'spanning_forest_edge_indices': [0, 1, 2, 3], 'loop_closure_edge_indices': [4],
                                'components': [{'suggested_root': 0, 'unanchored': False}]}}
            (folder/'joints.json').write_text(json.dumps(report))
            result = build_structure(folder/'parts.npz', folder/'binding.npz', folder, folder/'joints.json')
            self.assertEqual(result['row_groups'][-2:], [-1, -1])
            self.assertEqual(result['locked_groups'], [0, 1, 2])
            self.assertEqual(result['diagnostics']['trainable_groups'], [3, 4])
            np.testing.assert_allclose(result['joint_pivots'][2], [2.3, -.2, .1])
            model = HardJointEventCorrection(result)
            rest = torch.from_numpy(posed[0])
            for frame in range(frames):
                torch.testing.assert_close(model(rest, torch.from_numpy(posed[frame]), frame),
                                           torch.from_numpy(posed[frame]), atol=2e-6, rtol=1e-6)

    def test_reseeded_source_rows_keep_original_part_ids_and_band_offsets(self):
        """Replacing rigid splats must not turn source-object IDs into row IDs.

        Keep a reordered subset of old static/band rows, discard the old rigid
        rows, and append a different number of rigid rows.  The source mesh
        archive (including its sparse object IDs and global vertex offsets)
        remains unchanged, just as in the moving-surface reconstruction.
        """
        try:
            from scipy.spatial.transform import Rotation
            from scripts.lego_joint_structure import build_structure
            from utils.articulated_motion import ArticulatedBaker, apply_transform
        except ImportError:
            self.skipTest('Source-archive builder requires working SciPy; checked on AutoDL')
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frames, object_count, band = 6, 9, 5
            matrices = np.tile(np.eye(4), (frames, object_count, 1, 1))
            angles = np.linspace(0., .8, frames)
            matrices[:, 3, :3, :3] = Rotation.from_rotvec(np.column_stack((angles*0, angles*0, angles))).as_matrix()
            matrices[:, 7, :3, :3] = Rotation.from_rotvec(np.column_stack((angles*0, angles*0, angles*1.6))).as_matrix()
            pivot = np.array([1., 0., 0.])
            matrices[:, 7, :3, 3] = (matrices[:, 3, :3, :3]-matrices[:, 7, :3, :3]) @ pivot
            tetrahedron = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.], [0., 0., .1]])
            source_points = np.concatenate([tetrahedron+[i*.12, 0., 0.] for i in range(object_count)])
            offsets = np.arange(object_count+1)*len(tetrahedron)
            band_shift = np.column_stack((angles*0, angles*.03, angles*.01))
            band_points = source_points[offsets[band]:offsets[band+1]][None]+band_shift[:, None]
            source_metadata = {'objects': [f'source_object_{i}' for i in range(object_count)],
                               'deforming_objects': [band]}
            np.savez(folder/'parts.npz', object_matrices=matrices, rest_points=source_points,
                     point_parts=np.repeat(np.arange(object_count), len(tetrahedron)),
                     part_offsets=offsets, deformed_5=band_points,
                     metadata_json=np.asarray(json.dumps(source_metadata)))
            registration = np.eye(4)
            registration[:3, :3] = .4*Rotation.from_rotvec([.2, -.1, .3]).as_matrix()
            registration[:3, 3] = [.3, -.2, .1]

            old_labels = np.array([0]*10+[band]*4+[3]*10+[7]*10+[8]*6)
            old_nearest = offsets[old_labels]+np.arange(len(old_labels)) % len(tetrahedron)
            old_xyz = apply_transform(source_points[old_nearest], registration)
            # Static rows are removed/reordered too; retained elastic offsets
            # are global mesh-vertex IDs, never Gaussian-row indices.
            retained = np.array([8, 11, 0, 13, 35])
            new_labels = np.r_[old_labels[retained], np.full(17, 3), np.full(23, 7)]
            new_nearest = np.r_[old_nearest[retained], offsets[3]+np.arange(17) % 4,
                                offsets[7]+np.arange(23) % 4]
            new_xyz = np.concatenate((old_xyz[retained],
                                      apply_transform(source_points[new_nearest[len(retained):]], registration)))
            new_xyz[len(retained):] += np.arange(len(new_labels)-len(retained))[:, None]*[.0001, -.0002, .0003]
            vertices = np.zeros(len(new_labels), dtype=[(name, 'f4') for name in
                                                       ('x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3')])
            raw_quaternions = np.tile([.98, .01, -.02, .03], (len(vertices), 1))
            for column, name in enumerate(('x', 'y', 'z')):
                vertices[name] = new_xyz[:, column]
            for column in range(4):
                vertices[f'rot_{column}'] = raw_quaternions[:, column]
            indices = np.flatnonzero(np.isin(new_labels, [band, 3, 7]))
            np.testing.assert_array_equal(indices[:2], [1, 3])
            self.assertNotEqual(len(new_labels), len(old_labels))
            np.savez(folder/'binding.npz', labels=new_labels, nearest_vertex=new_nearest,
                     dynamic_indices=indices, registration=registration,
                     metadata_json=np.asarray(json.dumps({'gaussian_count': len(vertices)})))
            baker = ArticulatedBaker(vertices, folder/'parts.npz', folder/'binding.npz')
            np.testing.assert_array_equal(baker.bands[band][2], [3, 1])
            self.assertTrue(np.all(baker.bands[band][2] < len(tetrahedron)))
            posed = np.stack([baker.frame(frame) for frame in range(frames)])
            canonical_dynamic = np.column_stack([vertices[name][indices] for name in vertices.dtype.names])
            np.testing.assert_array_equal(posed[0], canonical_dynamic)
            expected_band_delta = band_shift[-1] @ registration[:3, :3].T
            np.testing.assert_allclose(posed[-1, :2, :3]-posed[0, :2, :3],
                                       np.tile(expected_band_delta, (2, 1)), atol=1e-7)
            np.save(folder/'dynamic_indices.npy', indices)
            np.save(folder/'poses.npy', posed)
            (folder/'cache.json').write_text(json.dumps({'gaussian_count': len(vertices)}))
            object_groups = [[0, 1, 2, 4, 6, 8], [3], [7]]
            groups = [{'group': i, 'objects': objects, 'representative_object': objects[0],
                       'object_names': [source_metadata['objects'][label] for label in objects], 'static': i == 0}
                      for i, objects in enumerate(object_groups)]
            report = {'format': 'source_rigid_joint_diagnosis_v1', 'frames_checked': frames, 'groups': groups,
                      'source_parts_sha256': hashlib.sha256((folder/'parts.npz').read_bytes()).hexdigest(),
                      'candidate_joints': [{'groups': pair, 'rest_pivot': point, 'rest_axis': [0., 0., 1.], 'rank': 2}
                                           for pair, point in [([0, 1], [0., 0., 0.]), ([1, 2], pivot.tolist())]],
                      'graph': {'spanning_forest_edge_indices': [0, 1], 'loop_closure_edge_indices': [],
                                'components': [{'suggested_root': 0, 'unanchored': False}]}}
            (folder/'joints.json').write_text(json.dumps(report))
            structure = build_structure(folder/'parts.npz', folder/'binding.npz', folder, folder/'joints.json')
            self.assertEqual(structure['row_groups'], [-1, -1]+[1]*17+[2]*23)
            self.assertEqual(structure['source_groups'][2]['objects'], [7])
            self.assertEqual(structure['locked_groups'], [0])
            self.assertEqual(structure['diagnostics']['rigid_rows'], 40)
            self.assertEqual(structure['diagnostics']['unchanged_deforming_rows'], 2)
            model = HardJointEventCorrection(structure)
            rest = torch.from_numpy(posed[0])
            for frame in range(frames):
                baked = torch.from_numpy(posed[frame])
                torch.testing.assert_close(model(rest, baked, frame), baked, atol=2e-6, rtol=1e-6)
            with torch.no_grad():
                model.group_angles.copy_(torch.tensor([1., -.7, 1.3]))
            for frame in range(frames):
                baked = torch.from_numpy(posed[frame])
                actual = model(rest, baked, frame)
                self.assertTrue(torch.equal(actual[:2], baked[:2]))
                self.assertLess(float(model.joint_errors(frame).max()), 1e-6)
                reconstructed_source = np.column_stack([vertices[name] for name in vertices.dtype.names])
                reconstructed_source[indices] = actual.detach().numpy()
                static_rows = np.flatnonzero(~np.isin(new_labels, [band, 3, 7]))
                np.testing.assert_array_equal(reconstructed_source[static_rows, :3], new_xyz[static_rows].astype(np.float32))
            self.assertIs(model(rest, torch.from_numpy(posed[-1]), 0), rest)
            # An old cache's source indices must be refused even if row counts
            # happen to match after reconstruction.
            np.save(folder/'dynamic_indices.npy', indices[::-1])
            with self.assertRaisesRegex(ValueError, 'timelines disagree'):
                build_structure(folder/'parts.npz', folder/'binding.npz', folder, folder/'joints.json')


if __name__ == '__main__':
    unittest.main()
