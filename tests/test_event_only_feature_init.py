import argparse
import unittest
from unittest.mock import patch

import numpy as np

from utils.event_only_feature_init import (add_binding_arguments, bind_feature_groups,
                                           infer_feature_groups, static_feature_prior,
                                           validate_binding_arguments)


def binding_arguments(**overrides):
    parser = argparse.ArgumentParser()
    add_binding_arguments(parser)
    args = parser.parse_args([])
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


class BindingArgumentTests(unittest.TestCase):
    def test_defaults_are_explicit_bounded_nearest_binding(self):
        args = binding_arguments()
        validate_binding_arguments(args)
        self.assertFalse(args.geodesic_binding)
        self.assertEqual(args.binding_radius_fraction, .08)
        self.assertEqual(args.binding_cell_fraction, .004)
        self.assertEqual(args.binding_path_fraction, .2)
        self.assertEqual(args.binding_max_nodes, 100000)

    def test_explicit_geodesic_cli_arguments(self):
        parser = argparse.ArgumentParser()
        add_binding_arguments(parser)
        args = parser.parse_args(['--geodesic-binding', '--binding-radius-fraction', '.05',
                                  '--binding-cell-fraction', '.003', '--binding-path-fraction', '.1',
                                  '--binding-max-nodes', '5000'])
        validate_binding_arguments(args)
        self.assertTrue(args.geodesic_binding)
        self.assertEqual(args.binding_radius_fraction, .05)
        self.assertEqual(args.binding_cell_fraction, .003)
        self.assertEqual(args.binding_path_fraction, .1)
        self.assertEqual(args.binding_max_nodes, 5000)

    def test_fraction_types_and_bounds_fail_explicitly(self):
        for name in ('binding_radius_fraction', 'binding_cell_fraction', 'binding_path_fraction'):
            for invalid in (0., 1., -.1, 2., float('nan'), float('inf'),
                            float('-inf'), True, False, np.bool_(True), '0.2', None):
                with self.subTest(parameter=name, value=invalid), self.assertRaises(ValueError):
                    validate_binding_arguments(binding_arguments(**{name: invalid}))

    def test_node_cap_is_an_integer_not_a_truthy_or_fractional_value(self):
        for invalid in (0, -1, 100001, True, False, np.bool_(True), 1.5,
                        100000., float('nan'), float('inf'), '50', None):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                validate_binding_arguments(binding_arguments(binding_max_nodes=invalid))

    def test_valid_numpy_scalar_budgets_and_node_limits(self):
        for limit in (1, 100000, np.int64(5000)):
            validate_binding_arguments(binding_arguments(binding_max_nodes=limit,
                                                         binding_radius_fraction=np.float64(.08)))


class BindFeatureGroupsTests(unittest.TestCase):
    def setUp(self):
        self.xyz = np.array([[0., 0., 0.], [3., 4., 0.], [1., 1., 0.]])
        cluster = np.column_stack((np.arange(8) * .01, np.zeros(8), np.zeros(8)))
        self.features = {'points': np.concatenate((cluster, cluster + [1., 1., 0.]))}
        poses = np.tile(np.eye(4), (2, 2, 1, 1))
        poses[:, 1, 0, 3] = [.1, .2]
        self.discovery = {'labels': np.repeat([0, 1], 8).astype(np.int32), 'poses': poses,
                          'pose_valid': np.ones((2, 2), dtype=bool)}
        self.times = np.array([0., .1, .2])
        self.hashes = dict(zip(('g0', 'events', 'intrinsics', 'extrinsics'),
                               ('a' * 64, 'b' * 64, 'c' * 64, 'd' * 64)))

    def call(self, **options):
        return bind_feature_groups(self.xyz, self.features, self.discovery,
                                   self.times, self.hashes, binding_arguments(**options))

    def test_nearest_forwards_same_run_arrays_without_geodesic_prior(self):
        result = {'new_binding': object(), 'diagnostics': {}}
        with patch('utils.event_only_hard_binding.build_hard_binding', return_value=result) as hard, \
                patch('utils.event_only_geodesic_binding.build_geodesic_binding') as geodesic:
            self.assertIs(self.call(), result)
        geodesic.assert_not_called()
        arguments, keywords = hard.call_args
        self.assertIs(arguments[0], self.xyz)
        self.assertIs(arguments[1], self.features['points'])
        np.testing.assert_array_equal(arguments[2], self.discovery['labels'])
        self.assertIs(arguments[3], self.times)
        self.assertIs(arguments[6], self.hashes)
        self.assertEqual(keywords, {'max_binding_distance': .4})
        self.assertEqual(result['diagnostics']['prebinding_unsupported_source_labels'], [])

    def test_prepends_identity_but_never_an_observed_t0_pose(self):
        before_poses = self.discovery['poses'].copy()
        before_valid = self.discovery['pose_valid'].copy()
        # An invalid pose may be NaN; prepending t0 must not turn it into evidence.
        self.discovery['poses'][0, 1] = np.nan
        before_poses[0, 1] = np.nan
        self.discovery['pose_valid'][0, 1] = False
        before_valid[0, 1] = False
        with patch('utils.event_only_hard_binding.build_hard_binding') as hard:
            self.call()
        poses, valid = hard.call_args.args[4:6]
        self.assertEqual(poses.shape, (3, 2, 4, 4))
        self.assertEqual(valid.shape, (3, 2))
        np.testing.assert_array_equal(poses[0], np.tile(np.eye(4), (2, 1, 1)))
        np.testing.assert_array_equal(valid[0], [False, False])
        np.testing.assert_array_equal(poses[1:], before_poses)
        np.testing.assert_array_equal(valid[1:], before_valid)
        np.testing.assert_array_equal(self.discovery['poses'], before_poses)
        np.testing.assert_array_equal(self.discovery['pose_valid'], before_valid)

    def test_geodesic_budgets_use_one_g0_extent_and_forward_exact_prior(self):
        prior = {'source_labels': np.array([0, -1, 1]), 'metadata': {'same_run': True}}
        with patch('utils.event_only_hard_binding.build_hard_binding') as hard, \
                patch('utils.event_only_geodesic_binding.build_geodesic_binding', return_value=prior) as geodesic:
            self.call(geodesic_binding=True, binding_radius_fraction=.1,
                      binding_cell_fraction=.01, binding_path_fraction=.3, binding_max_nodes=50)
        arguments, keywords = geodesic.call_args
        self.assertIs(arguments[0], self.xyz)
        self.assertIs(arguments[1], self.features['points'])
        np.testing.assert_array_equal(arguments[2], self.discovery['labels'])
        self.assertEqual(keywords['cell_size'], .05)
        self.assertEqual(keywords['max_euclidean_distance'], .5)
        self.assertEqual(keywords['max_path_length'], 1.5)
        self.assertAlmostEqual(keywords['max_edge_length'], np.sqrt(3) * .05)
        self.assertEqual(keywords['max_nodes'], 50)
        self.assertIs(hard.call_args.kwargs['assignment_prior'], prior)
        self.assertEqual(hard.call_args.kwargs['max_binding_distance'], .5)
        self.assertIs(hard.call_args.args[2], arguments[2])

    def test_unsupported_static_or_dynamic_group_is_removed_before_both_binders(self):
        for failure in ('insufficient_static_seeds', 'duplicate_dynamic_seeds', 'one_dynamic_pose'):
            with self.subTest(failure=failure):
                self.setUp()
                if failure == 'insufficient_static_seeds':
                    self.discovery['labels'][0] = -1
                    dropped = 0
                elif failure == 'duplicate_dynamic_seeds':
                    self.features['points'][-1] = self.features['points'][-2]
                    dropped = 1
                else:
                    self.discovery['pose_valid'][0, 1] = False
                    dropped = 1
                labels_before = self.discovery['labels'].copy()
                expected = labels_before.copy()
                expected[expected == dropped] = -1
                result = {'diagnostics': {}}
                with patch('utils.event_only_hard_binding.build_hard_binding', return_value=result) as hard, \
                        patch('utils.event_only_geodesic_binding.build_geodesic_binding') as geodesic:
                    self.call(geodesic_binding=True)
                np.testing.assert_array_equal(geodesic.call_args.args[2], expected)
                self.assertIs(geodesic.call_args.args[2], hard.call_args.args[2])
                np.testing.assert_array_equal(self.discovery['labels'], labels_before)
                self.assertEqual(result['diagnostics']['prebinding_unsupported_source_labels'], [dropped])

    def test_geodesic_failure_never_falls_back_to_nearest(self):
        for error in (ValueError('explicit node budget exceeded'), RuntimeError('backend unavailable')):
            with self.subTest(error=type(error)), \
                    patch('utils.event_only_hard_binding.build_hard_binding') as hard, \
                    patch('utils.event_only_geodesic_binding.build_geodesic_binding', side_effect=error):
                with self.assertRaises(type(error)):
                    self.call(geodesic_binding=True)
                hard.assert_not_called()

    def test_bad_parameters_fail_before_any_binding(self):
        with patch('utils.event_only_hard_binding.build_hard_binding') as hard, \
                patch('utils.event_only_geodesic_binding.build_geodesic_binding') as geodesic:
            with self.assertRaises(ValueError):
                self.call(geodesic_binding=True, binding_path_fraction=float('nan'))
        hard.assert_not_called()
        geodesic.assert_not_called()

    def test_degenerate_or_nonfinite_g0_fails_before_binding(self):
        for xyz in (np.zeros((3, 3)), np.array([[0., 0., 0.], [np.nan, 1., 1.]]),
                    np.array([[0., 0., 0.], [np.inf, 1., 1.]])):
            with self.subTest(xyz=xyz), patch('utils.event_only_hard_binding.build_hard_binding') as hard:
                self.xyz = xyz
                with self.assertRaises(ValueError):
                    self.call()
                hard.assert_not_called()

    def test_array_orchestration_does_not_open_files_or_load_cached_arrays(self):
        # Resolve mocked module imports first; any file access made by the helper fails.
        with patch('utils.event_only_hard_binding.build_hard_binding') as hard, \
                patch('utils.event_only_geodesic_binding.build_geodesic_binding') as geodesic, \
                patch('builtins.open', side_effect=AssertionError('Unexpected file input')), \
                patch('numpy.load', side_effect=AssertionError('Unexpected cached array input')):
            self.call(geodesic_binding=True)
        geodesic.assert_called_once()
        hard.assert_called_once()

    def test_real_nearest_binding_keeps_t0_unobserved_and_requires_four_hashes(self):
        # Tiny inputs use the NumPy backend, not the platform-specific SciPy backend.
        self.xyz = np.column_stack((np.arange(8) * .01, np.zeros(8), np.zeros(8)))
        self.features = {'points': self.xyz.copy()}
        self.discovery['labels'] = np.ones(8, dtype=np.int32)
        self.discovery['pose_valid'][:] = True
        result = self.call()
        self.assertEqual(result['diagnostics']['retained_dynamic_group_count'], 1)
        np.testing.assert_array_equal(result['times'], self.times)
        np.testing.assert_allclose(result['translations'][:, 0, 0], [.1, .2], atol=1e-8, rtol=0)
        self.assertEqual(result['diagnostics']['output_groups'][0]['pose_observed_indices'], [1, 2])
        self.assertFalse(result['recommended_frozen_groups'][0])
        for hashes in ({'g0': 'a' * 64}, dict(self.hashes, video='e' * 64)):
            with self.subTest(keys=list(hashes)), self.assertRaises(ValueError):
                bind_feature_groups(self.xyz, self.features, self.discovery, self.times,
                                    hashes, binding_arguments())

    def test_real_binding_rejects_missing_or_misaligned_t0_times(self):
        for times in (np.array([.1, .2, .3]), np.array([0., .1]), np.array([0., .2, .1])):
            with self.subTest(times=times), self.assertRaises(ValueError):
                bind_feature_groups(self.xyz, self.features, self.discovery, times,
                                    self.hashes, binding_arguments())


class InferFeatureGroupsTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.ones((64, 64), dtype=np.float32) * .3
        self.alpha = np.ones_like(self.reference)
        self.depth = np.ones_like(self.reference) * 2
        self.cumulative = np.stack([np.full_like(self.reference, value) for value in (.1, .2, .3, .4)])
        self.camera = {'source': 'synthetic_intrinsics_and_extrinsics'}
        self.times = np.array([0., .025, .05, .075, .1])
        self.indices = [0, 2, 4]
        self.raw_counts = np.zeros((4, 2, 64, 64), dtype=np.uint32)
        # Net-zero events are still activity, excluding this point from the static prior.
        self.raw_counts[1, :, 40, 40] = 1
        self.features = {'points': np.array([[0., 0., 2.], [1., 1., 2.]]),
                         'tracks': np.broadcast_to([[20.5, 20.5], [40.5, 40.5]], (3, 2, 2)).copy(),
                         'tracking_continuous': np.ones((3, 2), dtype=bool),
                         'accepted': np.array([[False, False], [True, False], [True, True]])}

    def call(self, **kwargs):
        return infer_feature_groups(self.reference, self.alpha, self.depth, self.cumulative,
                                    self.camera, self.times, self.indices,
                                    raw_event_counts=iter(self.raw_counts), **kwargs)

    def test_g0_zero_and_correct_cumulative_endpoints_are_forwarded(self):
        sampled = []

        def track(*args, **kwargs):
            sampled.extend(args[3])
            return self.features

        with patch('utils.event_only_feature_tracks.infer_event_feature_tracks', side_effect=track) as tracker, \
                patch('utils.event_only_rigid_discovery.discover_rigid_groups'):
            self.call(event_background=.04, max_corners=128, reference_reacquisition=True, max_groups=4)
        arguments, keywords = tracker.call_args
        for actual, expected in zip(arguments[:3], (self.reference, self.alpha, self.depth)):
            self.assertIs(actual, expected)
        self.assertIs(arguments[4], self.camera)
        np.testing.assert_array_equal(arguments[5], [0., .05, .1])
        np.testing.assert_array_equal(sampled, [np.zeros_like(self.reference),
                                               self.cumulative[1], self.cumulative[3]])
        self.assertEqual(keywords, {'max_corners': 128, 'allow_reference_reacquisition': True,
                                   'background_intensity': .04})

    def test_discovery_excludes_t0_and_static_prior_uses_both_raw_polarities(self):
        discovered = {'same_run_discovery': object()}
        with patch('utils.event_only_feature_tracks.infer_event_feature_tracks', return_value=self.features), \
                patch('utils.event_only_rigid_discovery.discover_rigid_groups', return_value=discovered) as discovery:
            result = self.call(max_groups=4)
        arguments, keywords = discovery.call_args
        self.assertIs(arguments[0], self.features['points'])
        np.testing.assert_array_equal(arguments[1], self.features['tracks'][1:])
        np.testing.assert_array_equal(arguments[2], self.features['accepted'][1:])
        self.assertIs(arguments[3], self.camera)
        np.testing.assert_array_equal(keywords['static_anchor_mask_np'], [True, False])
        self.assertEqual(keywords['max_groups'], 4)
        self.assertIs(result['features'], self.features)
        self.assertIs(result['discovery'], discovered)
        np.testing.assert_array_equal(result['static_prior_mask'], [True, False])
        self.assertIn('not_static_motion_proof', result['static_hypothesis'])

    def test_no_features_fails_before_discovery(self):
        with patch('utils.event_only_feature_tracks.infer_event_feature_tracks',
                   return_value={'points': np.empty((0, 3))}), \
                patch('utils.event_only_rigid_discovery.discover_rigid_groups') as discovery:
            with self.assertRaisesRegex(ValueError, 'no supported feature seeds'):
                self.call()
        discovery.assert_not_called()

    def test_group_inference_does_not_open_files_or_load_old_results(self):
        with patch('utils.event_only_feature_tracks.infer_event_feature_tracks', return_value=self.features), \
                patch('utils.event_only_rigid_discovery.discover_rigid_groups') as discovery, \
                patch('builtins.open', side_effect=AssertionError('Unexpected file input')), \
                patch('numpy.load', side_effect=AssertionError('Unexpected cached array input')):
            self.call()
        discovery.assert_called_once()


class StaticFeaturePriorTests(unittest.TestCase):
    def test_quiet_supported_stationary_feature_is_only_a_static_prior(self):
        tracks = np.broadcast_to([[20.5, 20.5], [40.5, 40.5]], (3, 2, 2)).copy()
        continuous = np.ones((3, 2), dtype=bool)
        events = np.zeros((2, 2, 64, 64), dtype=np.uint32)
        events[0, 0, 40, 40] = 1
        np.testing.assert_array_equal(static_feature_prior(tracks, continuous, events), [True, False])

    def test_motion_or_broken_chain_excludes_static_prior(self):
        tracks = np.broadcast_to([[20.5, 20.5], [40.5, 40.5]], (3, 2, 2)).copy()
        tracks[-1, 0, 0] += 1
        continuous = np.ones((3, 2), dtype=bool)
        continuous[1, 1] = False
        self.assertFalse(static_feature_prior(tracks, continuous, [np.zeros((2, 64, 64), dtype=np.uint32)]).any())

    def test_no_events_cannot_restore_a_broken_prior_chain(self):
        tracks = np.broadcast_to([[20.5, 20.5]], (3, 1, 2)).copy()
        continuous = np.array([[True], [False], [True]])
        self.assertFalse(static_feature_prior(tracks, continuous, [np.zeros((2, 64, 64), dtype=np.uint32)]).any())

    def test_opposite_polarity_cancellation_is_not_event_quiet(self):
        tracks = np.broadcast_to([[20.5, 20.5]], (3, 1, 2)).copy()
        events = np.zeros((1, 2, 64, 64), dtype=np.uint32)
        events[0, :, 20, 20] = 1
        self.assertFalse(static_feature_prior(tracks, np.ones((3, 1), bool), events).any())

    def test_missing_events_and_incorrect_mask_dtype_fail(self):
        tracks = np.broadcast_to([[20.5, 20.5]], (3, 1, 2)).copy()
        for continuous, cumulative in ((np.ones((3, 1), bool), []),
                                       (np.ones((3, 1), int), [np.zeros((64, 64))])):
            with self.assertRaises(ValueError):
                static_feature_prior(tracks, continuous, cumulative)


if __name__ == '__main__':
    unittest.main()
