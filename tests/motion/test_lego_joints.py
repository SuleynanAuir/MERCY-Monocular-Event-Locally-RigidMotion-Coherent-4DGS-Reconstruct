import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts.diagnostics.diagnose_lego_joints import diagnose, fit_shared_pivot, graph_structure, motion_groups


def rotation_about(pivot, angles, axis='z'):
    result = np.tile(np.eye(4), (len(angles), 1, 1))
    first, second = {'x': (1, 2), 'y': (2, 0), 'z': (0, 1)}[axis]
    result[:, first, first] = result[:, second, second] = np.cos(angles)
    result[:, first, second] = -np.sin(angles)
    result[:, second, first] = np.sin(angles)
    result[:, :3, 3] = pivot-np.einsum('tij,j->ti', result[:, :3, :3], pivot)
    return result


def source_parts(matrices, surfaces):
    return {'object_matrices': matrices, 'rest_points': np.concatenate(surfaces),
            'point_parts': np.concatenate([np.full(len(surface), i) for i, surface in enumerate(surfaces)]),
            'metadata_json': np.asarray(json.dumps({'objects': [str(i) for i in range(len(surfaces))],
                                                    'deforming_objects': []}))}


class SourceJointTests(unittest.TestCase):
    def test_hinge_axis_recovered_from_all_frames(self):
        pivot = np.array([.3, -.2, .8])
        moving = rotation_about(pivot, np.linspace(0, 1.7, 400))
        fixed = np.tile(np.eye(4), (400, 1, 1))
        fit = fit_shared_pivot(fixed, moving)
        self.assertEqual(fit['rank'], 2)
        np.testing.assert_allclose(fit['point'][:2], pivot[:2], atol=1e-12)
        np.testing.assert_allclose(fit['nullspace'][:, 0], [0, 0, 1], atol=1e-12)
        self.assertLess(fit['max_residual'], 1e-12)

    def test_point_joint_and_invalid_translation(self):
        pivot = np.array([.3, -.2, .8])
        angles = np.linspace(0, 1., 31)
        moving = rotation_about(pivot, angles, 'x') @ rotation_about(pivot, angles*.7, 'z')
        fixed = np.tile(np.eye(4), (31, 1, 1))
        fit = fit_shared_pivot(fixed, moving)
        self.assertEqual(fit['rank'], 3)
        np.testing.assert_allclose(fit['point'], pivot, atol=1e-12)
        moving[-1, 2, 3] += .1
        self.assertGreater(fit_shared_pivot(fixed, moving)['max_residual'], .01)

    def test_motion_groups_include_static_and_use_unsampled_frame(self):
        matrices = np.tile(np.eye(4), (400, 3, 1, 1))
        matrices[123, 2, 0, 3] = .01
        self.assertEqual(motion_groups(matrices, range(3), 1e-6, 1e-6), [[0, 1], [2]])

    def test_geometry_chooses_degenerate_axis_location(self):
        pivot = np.array([.3, -.2, .8])
        angles = np.linspace(0, 1, 50)
        matrices = np.stack((np.tile(np.eye(4), (50, 1, 1)), rotation_about(pivot, angles)), 1)
        offsets = np.array([[.001, 0, 0], [-.001, 0, 0], [0, .001, 0], [0, -.001, 0]])
        parts = source_parts(matrices, [pivot+offsets, pivot+offsets+[.0001, 0, 0]])
        result = diagnose(parts, contact_distance=.002, pivot_distance=.003, residual_tolerance=1e-8)
        self.assertEqual(len(result['candidate_joints']), 1)
        edge = result['candidate_joints'][0]
        np.testing.assert_allclose(edge['rest_pivot'], pivot, atol=1e-12)
        self.assertEqual(result['graph']['components'][0]['static_groups'], [0])
        self.assertEqual(result['frames_checked'], 50)

    def test_kinematically_shared_axis_without_contact_is_rejected(self):
        angles = np.linspace(0, 1, 20)
        matrices = np.stack((np.tile(np.eye(4), (20, 1, 1)), rotation_about(np.zeros(3), angles)), 1)
        parts = source_parts(matrices, [np.array([[1., 0, 0], [1.1, 0, 0]]),
                                        np.array([[-1., 0, 0], [-1.1, 0, 0]])])
        result = diagnose(parts, contact_distance=.01, pivot_distance=.1)
        self.assertFalse(result['candidate_joints'])
        self.assertEqual(result['rejected_group_pairs'][0]['rejection_reason'],
                         'no_nearby_sampled_source_surface_contact')

    def test_near_surfaces_far_from_shared_axis_are_rejected(self):
        angles = np.linspace(0, 1, 20)
        matrices = np.stack((np.tile(np.eye(4), (20, 1, 1)), rotation_about(np.zeros(3), angles)), 1)
        parts = source_parts(matrices, [np.array([[1., 0, 0], [1.1, 0, 0]]),
                                        np.array([[1., .001, 0], [1.1, .001, 0]])])
        result = diagnose(parts, contact_distance=.01, pivot_distance=.1)
        self.assertFalse(result['candidate_joints'])
        self.assertEqual(result['rejected_group_pairs'][0]['rejection_reason'],
                         'fitted_pivot_too_far_from_one_or_both_source_surfaces')

    def test_nonrigid_and_elastic_objects_are_not_joint_groups(self):
        matrices = np.tile(np.eye(4), (3, 3, 1, 1))
        matrices[1, 1, 0, 0] = 1.1
        parts = source_parts(matrices, [np.array([[0., 0, 0], [1, 1, 1]])]*3)
        parts['metadata_json'] = np.asarray(json.dumps({'objects': ['a', 'b', 'c'], 'deforming_objects': [2]}))
        result = diagnose(parts)
        self.assertEqual(len(result['groups']), 1)
        self.assertEqual([entry['reason'] for entry in result['excluded_objects']],
                         ['nonrigid_object_delta', 'deforming_source_object'])

    def test_group_member_residual_is_checked_not_only_representative(self):
        pivot = np.array([.3, -.2, .8])
        angles = np.linspace(0, 1, 50)
        first = rotation_about(pivot, angles)
        second = rotation_about(pivot+[.0001, 0, 0], angles)
        matrices = np.stack((np.tile(np.eye(4), (50, 1, 1)), first, second), 1)
        offsets = np.array([[.001, 0, 0], [-.001, 0, 0], [0, .001, 0], [0, -.001, 0]])
        parts = source_parts(matrices, [pivot+offsets]*3)
        result = diagnose(parts, translation_tolerance=.001, contact_distance=.002,
                          pivot_distance=.003, residual_tolerance=1e-8)
        self.assertEqual(len(result['groups']), 2)
        self.assertFalse(result['candidate_joints'])
        self.assertEqual(result['rejected_group_pairs'][0]['rejection_reason'],
                         'all_member_full_timeline_residual_bound_exceeds_tolerance')

    def test_cli_refuses_existing_output_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'existing.json'
            output.write_text('preserve this report')
            script = Path(__file__).resolve().parents[2]/'scripts/diagnostics/diagnose_lego_joints.py'
            process = subprocess.run([sys.executable, str(script), '--parts', str(Path(folder)/'absent.npz'),
                                      '--output', str(output)], capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('FileExistsError', process.stderr)
            self.assertEqual(output.read_text(), 'preserve this report')

    def test_graph_reports_cycles_and_isolated_groups(self):
        edges = [{'groups': pair, 'max_member_residual_bound': 0., 'contact_min_distance': 0.}
                 for pair in ([0, 1], [1, 2], [0, 2])]
        graph = graph_structure(4, edges, [0])
        self.assertEqual(graph['cycle_rank'], 1)
        self.assertEqual(graph['loop_closure_edge_indices'], [2])
        self.assertEqual(graph['components'][1]['groups'], [3])
        self.assertTrue(graph['components'][1]['unanchored'])


if __name__ == '__main__':
    unittest.main()
