"""Synthetic arrays only: no dataset, trained model, file or remote access."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from utils.event_only_feature_tracks import (_mutual_sift_matches, _sift_detections,
                                             infer_event_feature_tracks)


class EventOnlyFeatureTrackTests(unittest.TestCase):
    def setUp(self):
        self.width = self.height = 128
        self.camera = {"width": 128, "height": 128, "fx": 100., "fy": 110., "cx": 64., "cy": 61.,
                       "pixel_center": "boundary", "w2c": np.eye(4)}
        rng = np.random.default_rng(7)
        self.reference = cv2.GaussianBlur(rng.uniform(.15, .85, (128, 128)).astype(np.float32), (3, 3), .6)
        self.alpha = np.ones_like(self.reference)
        self.depth = np.full_like(self.reference, 2.)

    def changes(self, shifts):
        for x, y in shifts:
            target = cv2.warpAffine(self.reference, np.array([[1., 0, x], [0, 1., y]], np.float32),
                                   (128, 128), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            yield np.log(target) - np.log(self.reference)

    def run_tracks(self, shifts, **kwargs):
        return infer_event_feature_tracks(self.reference, self.alpha, self.depth, self.changes(shifts),
                                          self.camera, np.arange(len(shifts)) * .05, **kwargs)

    def test_dense_translation_tracks_many_points_and_boundary_pixels(self):
        result = self.run_tracks([(0, 0), (2, -1), (4, -2)])
        self.assertGreater(len(result["points"]), 100)
        self.assertLessEqual(len(result["points"]), 2048)
        supported = result["accepted"][-1]
        self.assertGreater(supported.sum(), 80)
        expected = np.broadcast_to([4, -2], (supported.sum(), 2))
        np.testing.assert_allclose(result["tracks"][-1, supported] - result["tracks"][0, supported], expected, atol=.3)
        self.assertFalse(result["accepted"][0].any())
        self.assertFalse(result["metadata"]["physical_motion_validated"])
        self.assertFalse(result["metadata"]["new_geometry_measurement"])
        json.dumps(result["metadata"], allow_nan=False)

    def test_nonidentity_camera_and_normalized_depth_backprojection(self):
        angle = .45
        self.camera["w2c"] = np.array([[np.cos(angle), -np.sin(angle), 0, .3],
                                        [np.sin(angle), np.cos(angle), 0, -.2], [0, 0, 1, .6], [0, 0, 0, 1]])
        self.alpha[:] = .96
        result = self.run_tracks([(0, 0), (0, 0)])
        world = result["points"]
        camera = world @ self.camera["w2c"][:3, :3].T + self.camera["w2c"][:3, 3]
        np.testing.assert_allclose(camera[:, 2], 2, atol=1e-6)
        projected = np.column_stack((100 * camera[:, 0] / camera[:, 2] + 64, 110 * camera[:, 1] / camera[:, 2] + 61))
        np.testing.assert_allclose(projected, result["tracks"][0], atol=1e-5)

    def test_many_features_follow_small_in_plane_rigid_rotation(self):
        matrices = [cv2.getRotationMatrix2D((63.5, 63.5), degrees, 1.) for degrees in (0, 2, 4)]
        changes = [np.log(cv2.warpAffine(self.reference, transform, (128, 128),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)) - np.log(self.reference)
                   for transform in matrices]
        result = infer_event_feature_tracks(self.reference, self.alpha, self.depth, changes, self.camera, [0, .05, .1])
        supported = result["accepted"][-1]
        self.assertGreater(supported.sum(), 80)
        start = result["tracks"][0, supported] - .5
        expected = start @ matrices[-1][:, :2].T + matrices[-1][:, 2] + .5
        errors = np.linalg.norm(result["tracks"][-1, supported] - expected, axis=1)
        self.assertLess(float(np.mean(errors)), .2)
        self.assertLess(float(np.quantile(errors, .95)), .4)

    def test_quiet_pause_retains_checked_image_prior_but_not_event_observation(self):
        result = self.run_tracks([(0, 0), (0, 0), (2, 0), (2, 0), (4, 0)])
        self.assertGreater(result["tracking_continuous"][1].sum(), 100)
        self.assertFalse(result["accepted"][1].any())
        self.assertFalse(result["event_supported"][1].any())
        self.assertFalse(result["accepted"][3].any())
        self.assertGreater(result["accepted"][-1].sum(), 80)

    def test_bad_forward_backward_pair_breaks_chain_and_local_flow_cannot_revive(self):
        real_lk = cv2.calcOpticalFlowPyrLK
        calls = 0

        def inconsistent(*args, **kwargs):
            nonlocal calls
            calls += 1
            candidate, status, error = real_lk(*args, **kwargs)
            if calls == 4 and candidate is not None:
                candidate = candidate.copy()
                candidate[:, :, 0] += 5
            return candidate, status, error

        with patch("cv2.calcOpticalFlowPyrLK", side_effect=inconsistent):
            result = self.run_tracks([(0, 0), (2, 0), (4, 0), (6, 0)])
        self.assertGreater(result["accepted"][1].sum(), 80)
        self.assertFalse(result["tracking_continuous"][2:].any())
        self.assertFalse(result["accepted"][2:].any())
        np.testing.assert_array_equal(result["tracks"][3], result["tracks"][1])
        self.assertEqual(calls, 4)

    def occluded_maps(self):
        normal = list(self.changes([(0, 0), (2, 0), (4, 0)]))
        return [normal[0], np.log(np.full_like(self.reference, .3)) - np.log(self.reference), normal[-1]]

    def test_occlusion_breaks_chain_and_does_not_local_reassociate(self):
        result = infer_event_feature_tracks(self.reference, self.alpha, self.depth, self.occluded_maps(),
                                          self.camera, [0, .05, .1])
        self.assertFalse(result["accepted"][1:].any())
        self.assertFalse(result["tracking_continuous"][1:].any())
        self.assertFalse(result["reassociated"].any())

    def test_explicit_reference_reassociation_can_recover_without_filling_gap(self):
        result = infer_event_feature_tracks(self.reference, self.alpha, self.depth, self.occluded_maps(),
                                          self.camera, [0, .05, .1], allow_reference_reacquisition=True)
        self.assertFalse(result["accepted"][1].any())
        self.assertGreater(result["reassociated"][2].sum(), 80)
        restored = result["reassociated"][2]
        self.assertFalse(result["tracking_continuous"][2, restored].any())
        np.testing.assert_array_equal(result["tracking_source"][2, restored], 3)
        np.testing.assert_allclose(result["tracks"][2, restored] - result["tracks"][0, restored],
                                   np.broadcast_to([4, 0], (restored.sum(), 2)), atol=.3)

    def test_reassociation_after_movement_starts_from_g0_not_stale_coordinates(self):
        changes = list(self.changes([(0, 0), (2, 0), (6, 0)]))
        changes.insert(2, np.log(np.full_like(self.reference, .3)) - np.log(self.reference))
        reference_u8 = np.rint(self.reference * 255).astype(np.uint8)
        last_u8 = np.rint(np.exp(np.log(self.reference) + changes[-1]) * 255).astype(np.uint8)
        direct_sources = []
        actual_lk = cv2.calcOpticalFlowPyrLK

        def record_source(first, second, coordinates, *args, **kwargs):
            if np.array_equal(first, reference_u8) and np.array_equal(second, last_u8):
                direct_sources.append(coordinates.copy())
            return actual_lk(first, second, coordinates, *args, **kwargs)

        with patch("cv2.calcOpticalFlowPyrLK", side_effect=record_source):
            result = infer_event_feature_tracks(self.reference, self.alpha, self.depth, changes,
                                              self.camera, [0, .05, .1, .15], allow_reference_reacquisition=True)
        self.assertGreater(result["accepted"][1].sum(), 80)
        self.assertFalse(result["accepted"][2].any())
        self.assertGreater(result["reassociated"][3].sum(), 80)
        self.assertEqual(len(direct_sources), 1)
        np.testing.assert_array_equal(direct_sources[0].reshape(-1, 2), result["tracks"][0] - .5)
        restored = result["reassociated"][3]
        np.testing.assert_allclose(result["tracks"][3, restored] - result["tracks"][0, restored],
                                   np.broadcast_to([6, 0], (restored.sum(), 2)), atol=.3)

    def test_patch_mismatch_rejected_even_if_mock_klt_cycle_is_perfect(self):
        def identity_lk(first, second, coordinates, *args, **kwargs):
            return coordinates.copy(), np.ones((len(coordinates), 1), np.uint8), np.zeros((len(coordinates), 1))

        changes = [np.zeros_like(self.reference), np.log(1 - self.reference) - np.log(self.reference)]
        with patch("cv2.calcOpticalFlowPyrLK", side_effect=identity_lk):
            result = infer_event_feature_tracks(self.reference, self.alpha, self.depth, changes,
                                              self.camera, [0, .1], allow_reference_reacquisition=True)
        self.assertFalse(result["accepted"].any())
        self.assertFalse(result["reassociated"].any())

    def test_low_alpha_and_depth_edges_are_not_new_surface_points(self):
        self.alpha[:, :32] = .5
        self.depth[:, 64:] = 4
        result = self.run_tracks([(0, 0), (0, 0)])
        x = result["tracks"][0, :, 0] - .5
        self.assertTrue((x >= 33).all())
        self.assertFalse(((x >= 63) & (x <= 64)).any())
        self.assertGreater(result["diagnostics"]["opaque_pixels_rejected_by_depth_range"], 0)
        self.assertTrue(np.isin(result["points"][:, 2], [2, 4]).all())

    def test_no_corners_is_an_explicit_empty_result_not_fake_tracks(self):
        self.reference[:] = .3
        result = self.run_tracks([(0, 0), (0, 0)])
        self.assertEqual(result["points"].shape, (0, 3))
        self.assertEqual(result["tracks"].shape, (2, 0, 2))
        self.assertEqual(result["diagnostics"]["feature_count"], 0)

    def test_only_array_inputs_and_no_inplace_changes(self):
        original = self.reference.copy()
        with patch("builtins.open", side_effect=AssertionError("No files may be opened")):
            result = self.run_tracks([(0, 0), (1, 0)], max_corners=50)
        self.assertLessEqual(len(result["points"]), 50)
        np.testing.assert_array_equal(self.reference, original)

    @staticmethod
    def identity_lk(first, second, coordinates, *args, **kwargs):
        return coordinates.copy(), np.ones((len(coordinates), 1), np.uint8), np.zeros((len(coordinates), 1))

    def fading_ghost_changes(self, background=.04):
        return [np.log(background + ratio * (self.reference - background)) - np.log(self.reference)
                for ratio in np.linspace(1., .1, 20)]

    def test_gradually_fading_ghost_passes_local_gates_but_not_fixed_g0_survival(self):
        changes = self.fading_ghost_changes()
        with patch("cv2.calcOpticalFlowPyrLK", side_effect=self.identity_lk):
            baseline = infer_event_feature_tracks(self.reference, self.alpha, self.depth, changes,
                                                 self.camera, np.arange(20) * .05)
            protected = infer_event_feature_tracks(self.reference, self.alpha, self.depth, changes,
                                                  self.camera, np.arange(20) * .05, background_intensity=.04)
        self.assertGreater(baseline["accepted"][-1].sum(), 80)
        self.assertGreater(protected["accepted"][1].sum(), 80)
        self.assertFalse(protected["accepted"][-1].any())
        self.assertGreater(protected["diagnostics"]["foreground_survival_rejected_point_steps"], 80)
        self.assertEqual(protected["metadata"]["foreground_survival"]["reference_refresh"], "never_original_G0_only")

    def test_survival_gate_is_also_applied_to_explicit_reference_matches(self):
        changes = [np.zeros_like(self.reference), self.fading_ghost_changes()[-1]]
        with patch("cv2.calcOpticalFlowPyrLK", side_effect=self.identity_lk):
            result = infer_event_feature_tracks(self.reference, self.alpha, self.depth, changes,
                                              self.camera, [0, .1], background_intensity=.04,
                                              patch_max_error=1., allow_reference_reacquisition=True)
        self.assertFalse(result["accepted"][-1].any())
        self.assertFalse(result["reassociated"][-1].any())
        self.assertGreater(result["diagnostics"]["pairs"][-1]["foreground_survival_rejected_reference_matches"], 80)

    def test_normal_translation_survives_known_background_gate(self):
        result = self.run_tracks([(0, 0), (2, 0), (4, 0)], background_intensity=.04)
        self.assertGreater(result["accepted"][-1].sum(), 80)
        self.assertEqual(result["diagnostics"]["foreground_survival_rejected_point_steps"], 0)

    def test_forty_five_degree_rotation_preserves_fixed_disk_statistics(self):
        changes = [np.log(cv2.warpAffine(self.reference, cv2.getRotationMatrix2D((63.5, 63.5), angle, 1.),
                                         (128, 128), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101))
                   - np.log(self.reference) for angle in np.linspace(0, 45, 16)]
        result = infer_event_feature_tracks(self.reference, self.alpha, self.depth, changes, self.camera,
                                          np.arange(16) * .05, background_intensity=.04)
        self.assertGreater(result["accepted"][-1].sum(), 80)
        self.assertEqual(result["diagnostics"]["foreground_survival_rejected_point_steps"], 0)

    def test_dark_material_uses_contrast_without_positive_background_brightness(self):
        self.reference = .1 + .25 * self.reference
        result = self.run_tracks([(0, 0), (2, 0), (4, 0)], background_intensity=.7)
        self.assertEqual(result["diagnostics"]["foreground_survival_brightness_supported_features"], 0)
        self.assertGreater(result["diagnostics"]["foreground_survival_contrast_supported_features"], 80)
        self.assertGreater(result["accepted"][-1].sum(), 80)
        self.assertFalse(result["metadata"]["foreground_survival"]["dark_material_requires_positive_brightness"])

    def test_unknown_background_explicitly_disables_long_term_protection(self):
        result = self.run_tracks([(0, 0), (2, 0)])
        self.assertFalse(result["metadata"]["foreground_survival"]["enabled"])
        self.assertEqual(result["metadata"]["foreground_survival"]["long_term_photometric_protection"], "disabled")
        self.assertEqual(result["diagnostics"]["foreground_survival_unprotected_features"], len(result["points"]))

    def test_invalid_inputs_fail_closed(self):
        changes = list(self.changes([(0, 0), (1, 0)]))
        cases = [(changes[:1], [0, .1]), (changes + [changes[-1]], [0, .1]),
                 ([np.ones_like(self.reference), changes[1]], [0, .1]),
                 ([changes[0], np.full_like(self.reference, np.nan)], [0, .1]),
                 (changes, [1, 2]), (changes, [0, 0])]
        for supplied, times in cases:
            with self.subTest(times=times), self.assertRaises(ValueError):
                infer_event_feature_tracks(self.reference, self.alpha, self.depth, supplied, self.camera, times)
        for options in ({"max_corners": True}, {"max_corners": 2049}, {"minimum_alpha": 1.1}, {"fb_max_error": 0},
                        {"patch_max_error": float("nan")}, {"allow_reference_reacquisition": 1},
                        {"background_intensity": float("nan")}, {"background_intensity": -.1},
                        {"foreground_survival_ratio": 0}, {"foreground_survival_ratio": 1.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_tracks([(0, 0), (1, 0)], **options)


class EventOnlySiftTrackTests(unittest.TestCase):
    setUp = EventOnlyFeatureTrackTests.setUp

    def run_sift(self, changes, **kwargs):
        return infer_event_feature_tracks(self.reference, self.alpha, self.depth, changes,
                                          self.camera, np.arange(len(changes)) * .05,
                                          feature_method='sift', **kwargs)

    def rotated_changes(self, angles):
        matrices = [cv2.getRotationMatrix2D((63.5, 63.5), angle, 1.) for angle in angles]
        changes = [np.log(cv2.warpAffine(self.reference, matrix, (128, 128),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101))
                   - np.log(self.reference) for matrix in matrices]
        return changes, matrices

    def test_optional_backend_preserves_default_corner_outputs(self):
        changes = list(EventOnlyFeatureTrackTests.changes(self, [(0, 0), (2, 0), (4, 0)]))
        implicit = infer_event_feature_tracks(self.reference, self.alpha, self.depth, changes,
                                             self.camera, [0, .05, .1])
        explicit = infer_event_feature_tracks(self.reference, self.alpha, self.depth, changes,
                                             self.camera, [0, .05, .1], feature_method='corners')
        for name in ('points', 'tracks', 'accepted', 'tracking_continuous',
                     'reassociated', 'event_supported', 'tracking_source'):
            np.testing.assert_array_equal(implicit[name], explicit[name])
        self.assertEqual(implicit['metadata'], explicit['metadata'])

    def test_independent_sift_handles_large_in_plane_rotations_without_klt(self):
        changes, matrices = self.rotated_changes([0, 45, 90])
        with patch('cv2.calcOpticalFlowPyrLK', side_effect=AssertionError('SIFT is an independent backend')):
            result = self.run_sift(changes, background_intensity=.04)
        self.assertGreater(len(result['points']), 60)
        self.assertLessEqual(len(result['points']), 2048)
        for index in (1, 2):
            supported = result['accepted'][index]
            self.assertGreater(supported.sum(), 30)
            start = result['tracks'][0, supported] - .5
            expected = start @ matrices[index][:, :2].T + matrices[index][:, 2] + .5
            errors = np.linalg.norm(result['tracks'][index, supported] - expected, axis=1)
            self.assertLess(float(np.mean(errors)), .8)
            self.assertLess(float(np.quantile(errors, .95)), 1.2)
            np.testing.assert_array_equal(result['tracking_source'][index, supported], 4)
        self.assertFalse(result['accepted'][0].any())
        self.assertFalse(result['metadata']['pretrained_models'])
        self.assertFalse(result['metadata']['proxy_is_observed_rgb'])
        self.assertFalse(result['metadata']['physical_motion_validated'])
        self.assertIsNone(result['metadata']['patch_minimum_correlation'])
        self.assertFalse(result['metadata']['sift_matching']['pixelwise_patch_correlation_gate_used'])
        json.dumps(result['metadata'], allow_nan=False)

    def test_quiet_descriptors_continue_image_priors_without_event_evidence(self):
        result = self.run_sift([np.zeros_like(self.reference)] * 3)
        self.assertGreater(result['tracking_continuous'][1:].sum(axis=1).min(), 60)
        self.assertFalse(result['accepted'].any())
        self.assertFalse(result['event_supported'].any())
        self.assertFalse(result['reassociated'].any())
        matched = result['tracking_continuous'][-1]
        np.testing.assert_allclose(result['tracks'][-1, matched], result['tracks'][0, matched], atol=.01)

    def test_occlusion_recovers_original_identity_without_filling_chain_gap(self):
        changes, matrices = self.rotated_changes([0, 45, 90])
        blank = np.log(np.full_like(self.reference, .3)) - np.log(self.reference)
        supplied = [changes[0], changes[1], blank, changes[2], changes[2]]
        recorded_references = []
        original_match = _mutual_sift_matches

        def record_reference(reference, current, *args, **kwargs):
            recorded_references.append(reference.copy())
            return original_match(reference, current, *args, **kwargs)

        with patch('utils.event_only_feature_tracks._mutual_sift_matches', side_effect=record_reference):
            result = self.run_sift(supplied, background_intensity=.04)
        self.assertFalse(result['accepted'][2].any())
        self.assertFalse(result['tracking_continuous'][2:].any())
        self.assertGreater(result['reassociated'][3].sum(), 30)
        restored = result['reassociated'][3]
        self.assertTrue(result['accepted'][3, restored].all())
        np.testing.assert_array_equal(result['tracking_source'][3, restored], 4)
        start = result['tracks'][0, restored] - .5
        expected = start @ matrices[-1][:, :2].T + matrices[-1][:, 2] + .5
        self.assertLess(np.mean(np.linalg.norm(result['tracks'][3, restored] - expected, axis=1)), .8)
        self.assertFalse(result['accepted'][4].any())
        for reference in recorded_references[1:]:
            np.testing.assert_array_equal(reference, recorded_references[0])

    def test_quiet_frame_never_restores_a_failed_descriptor_identity(self):
        original = _sift_detections
        calls = 0

        def missed_detection(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                return np.empty((0, 2), np.float32), np.empty((0, 128), np.float32)
            return original(*args, **kwargs)

        with patch('utils.event_only_feature_tracks._sift_detections', side_effect=missed_detection):
            result = self.run_sift([np.zeros_like(self.reference)] * 3)
        self.assertGreater(len(result['points']), 60)
        self.assertFalse(result['tracking_continuous'][1:].any())
        self.assertFalse(result['reassociated'].any())
        self.assertFalse(result['accepted'].any())
        self.assertFalse(result['tracking_source'][1:].any())

    def test_faded_ghost_fails_fixed_g0_survival_even_with_matching_descriptors(self):
        dark = np.log(.04 + .1 * (self.reference - .04)) - np.log(self.reference)
        detected = []
        original = _sift_detections

        def force_same_descriptors(*args, **kwargs):
            if not detected:
                detected.append(original(*args, **kwargs))
            return detected[0]

        with patch('utils.event_only_feature_tracks._sift_detections', side_effect=force_same_descriptors):
            result = self.run_sift([np.zeros_like(self.reference), dark], background_intensity=.04)
        self.assertGreater(len(result['points']), 60)
        self.assertFalse(result['accepted'][-1].any())
        self.assertFalse(result['tracking_continuous'][-1].any())
        self.assertGreater(result['diagnostics']['foreground_survival_rejected_point_steps'], 60)

    def test_surface_depth_alpha_and_nonidentity_camera_are_respected(self):
        self.alpha[:, :32] = .5
        self.depth[:, 64:] = 4.
        self.camera['w2c'] = np.array([[0., -1., 0., .3], [1., 0., 0., -.2],
                                       [0., 0., 1., .6], [0., 0., 0., 1.]])
        result = self.run_sift([np.zeros_like(self.reference)] * 2)
        self.assertGreater(len(result['points']), 30)
        pixels = result['tracks'][0] - .5
        self.assertTrue((pixels[:, 0] >= 33).all())
        self.assertFalse(((pixels[:, 0] > 62) & (pixels[:, 0] < 65)).any())
        camera_points = result['points'] @ self.camera['w2c'][:3, :3].T + self.camera['w2c'][:3, 3]
        np.testing.assert_allclose(camera_points[:, 2], np.where(pixels[:, 0] < 64, 2., 4.), atol=1e-6)
        projected = np.column_stack((100 * camera_points[:, 0] / camera_points[:, 2] + 64,
                                     110 * camera_points[:, 1] / camera_points[:, 2] + 61))
        np.testing.assert_allclose(projected, result['tracks'][0], atol=1e-5)

    def test_no_descriptors_is_an_explicit_empty_result(self):
        self.reference[:] = .3
        result = self.run_sift([np.zeros_like(self.reference)] * 2)
        self.assertEqual(result['points'].shape, (0, 3))
        self.assertEqual(result['tracks'].shape, (2, 0, 2))
        self.assertFalse(result['accepted'].any())

    def test_only_array_inputs_and_explicit_backend_validation(self):
        original = self.reference.copy()
        with patch('builtins.open', side_effect=AssertionError('No input files')), \
                patch('numpy.load', side_effect=AssertionError('No cached descriptors')):
            result = self.run_sift([np.zeros_like(self.reference)] * 2, max_corners=32)
        self.assertLessEqual(len(result['points']), 32)
        np.testing.assert_array_equal(self.reference, original)
        for method in ('sift_then_old_model', '', None, True):
            with self.subTest(method=method), self.assertRaises(ValueError):
                infer_event_feature_tracks(self.reference, self.alpha, self.depth,
                                           [np.zeros_like(self.reference)] * 2, self.camera, [0, .1],
                                           feature_method=method)


class SiftMatchingGuardTests(unittest.TestCase):
    @staticmethod
    def descriptors(values):
        descriptors = np.zeros((len(values), 128), dtype=np.float32)
        descriptors[:, 0] = values
        return descriptors

    def test_mutual_ratio_rejects_repeated_texture_equal_distance_ambiguity(self):
        for reference, current in (([0, 0, 10], [0, 0, 10]), ([0, 10], [5, 5])):
            matches = _mutual_sift_matches(self.descriptors(reference), self.descriptors(current))
            self.assertFalse(np.isin(matches[:, 0], [0, 1]).any())

    def test_reverse_ratio_is_required_and_target_cannot_have_two_identities(self):
        # Forward query 0 is very confident, but reverse target 0 has two
        # similarly close reference candidates and must abstain.
        reference, current = self.descriptors([0., .1, 20.]), self.descriptors([.045, 20.])
        matches = _mutual_sift_matches(reference, current)
        self.assertNotIn(0, matches[:, 0])
        self.assertNotIn(1, matches[:, 0])
        self.assertEqual(len(np.unique(matches[:, 1])), len(matches))
        np.testing.assert_array_equal(_mutual_sift_matches(self.descriptors([0., 10., 20.]),
                                                         self.descriptors([20., 0., 10.])),
                                      [[0, 1], [1, 2], [2, 0]])

    def test_fewer_than_two_descriptors_cannot_establish_ratio(self):
        self.assertEqual(_mutual_sift_matches(self.descriptors([0.]), self.descriptors([0., 1.])).shape, (0, 2))
        self.assertEqual(_mutual_sift_matches(self.descriptors([]), self.descriptors([])).shape, (0, 2))

    def test_duplicate_orientation_border_and_unsupported_depth_taps_are_removed(self):
        keypoints = [SimpleNamespace(pt=point, size=4., response=response)
                     for point, response in [((20., 20.), 1.), ((20., 20.), 2.), ((4., 20.), 3.),
                                              ((30.3, 30.3), 4.), ((40., 40.), .5)]]
        descriptors = self.descriptors(range(len(keypoints)))
        detector = Mock()
        detector.detectAndCompute.return_value = keypoints, descriptors
        mask = np.ones((64, 64), dtype=bool)
        mask[31, 31] = False
        points, selected = _sift_detections(detector, np.zeros((64, 64), np.uint8), mask, 10, 3., 4)
        np.testing.assert_array_equal(points, [[20., 20.], [40., 40.]])
        np.testing.assert_array_equal(selected[:, 0], [1, 4])
        # Current detections have no G0 surface mask but still deduplicate
        # orientations and reject boundary locations before descriptor matching.
        points, _ = _sift_detections(detector, np.zeros((64, 64), np.uint8), None, 10, 3., 4)
        self.assertEqual(len(points), 3)
        self.assertFalse((points[:, 0] == 4).any())


if __name__ == "__main__":
    unittest.main()
