"""CPU synthetic tests. No video, pretrained network, mesh, or files are used."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from utils.event_only_flow_init import estimate_local_kabsch_rotations, infer_event_flow_initialization


class EventOnlyFlowInitializationTests(unittest.TestCase):
    def setUp(self):
        self.camera = {"width": 96, "height": 96, "fx": 80., "fy": 80.,
                       "cx": 48., "cy": 48., "pixel_center": "boundary", "w2c": np.eye(4)}
        random = np.random.default_rng(11)
        self.reference = cv2.GaussianBlur(random.uniform(.1, .9, (96, 96)).astype(np.float32), (3, 3), .6)
        self.centers = np.array([[-.2, -.2, 2], [.2, -.2, 2], [-.2, .2, 2], [.2, .2, 2]], np.float32)

    def maps(self, shifts):
        output = []
        for x, y in shifts:
            proxy = cv2.warpAffine(self.reference, np.array([[1., 0, x], [0, 1., y]], np.float32),
                                   (96, 96), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            output.append(np.log(np.maximum(proxy, .01)) - np.log(np.maximum(self.reference, .01)))
        return np.asarray(output)

    def run_flow(self, shifts, centers=None, camera=None):
        return infer_event_flow_initialization(self.reference, self.maps(shifts),
                                               self.centers if centers is None else centers,
                                               self.camera if camera is None else camera,
                                               np.arange(len(shifts), dtype=float) * .1)

    def test_texture_translation_tracks_direction_and_constant_depth_magnitude(self):
        result = self.run_flow([(0, 0), (3, -2), (6, -4)])
        self.assertEqual(result["translations"].shape, (2, 4, 3))
        self.assertEqual(result["rotation_vectors"].shape, (2, 4, 3))
        expected = np.array([[3 * 2 / 80, -2 * 2 / 80, 0], [6 * 2 / 80, -4 * 2 / 80, 0]])
        np.testing.assert_allclose(result["translations"].mean(axis=1), expected, atol=.012, rtol=0)
        np.testing.assert_allclose(result["rotation_vectors"], 0, atol=.02, rtol=0)
        self.assertFalse(result["metadata"]["proxy_is_observed_rgb"])
        self.assertFalse(result["metadata"]["physical_uniqueness_guaranteed"])
        self.assertGreater(result["debug"]["total_accepted_node_steps"], 0)

    def test_camera_rotation_and_translation_are_inverted_when_lifting(self):
        camera = dict(self.camera)
        camera["w2c"] = np.array([[0., -1, 0, .3], [1, 0, 0, -.2], [0, 0, 1, .4], [0, 0, 0, 1]])
        world_centers = (self.centers - camera["w2c"][:3, 3]) @ camera["w2c"][:3, :3]
        result = self.run_flow([(0, 0), (4, 0)], world_centers, camera)
        # Positive camera X means negative world Y for this w2c rotation.
        np.testing.assert_allclose(result["translations"][0].mean(axis=0), [0, -.1, 0], atol=.012, rtol=0)

    def test_zero_event_activity_produces_no_spurious_motion(self):
        result = self.run_flow([(0, 0), (0, 0), (0, 0)])
        np.testing.assert_array_equal(result["translations"], np.zeros((2, 4, 3)))
        self.assertEqual(result["debug"]["total_accepted_node_steps"], 0)

    def test_uniform_brightness_changes_without_texture_are_gated(self):
        reference = np.full((96, 96), .3, np.float32)
        changes = np.array([np.zeros_like(reference), np.full_like(reference, .2)])
        result = infer_event_flow_initialization(reference, changes, self.centers, self.camera, [0, 1])
        np.testing.assert_array_equal(result["translations"], np.zeros((1, 4, 3)))
        self.assertEqual(result["debug"]["total_accepted_node_steps"], 0)

    def test_behind_camera_and_offscreen_nodes_do_not_move(self):
        centers = np.array([[0., 0, -2], [200, 0, 2]], np.float32)
        result = self.run_flow([(0, 0), (4, 0)], centers)
        np.testing.assert_array_equal(result["translations"], np.zeros((1, 2, 3)))
        self.assertEqual(result["debug"]["invalid_initial_depth_nodes"], 1)

    def test_iterable_inputs_work_without_materializing_all_event_maps(self):
        changes = self.maps([(0, 0), (2, 0)])
        result = infer_event_flow_initialization(self.reference, (item for item in changes),
                                               self.centers, self.camera, [0, .7])
        self.assertEqual(result["translations"].shape, (1, 4, 3))
        self.assertGreater(result["translations"][0, :, 0].mean(), .02)

    def test_t0_anchor_shape_time_and_map_count_validation(self):
        changes = self.maps([(0, 0), (2, 0)])
        for invalid, times in ((changes + .1, [0, 1]), (changes, [1, 2]), (changes, [0, 0]),
                               (changes[:1], [0, 1]), (changes, [0, 1, 2]),
                               (np.concatenate((changes, changes[:1])), [0, 1]),
                               (changes[:, :90], [0, 1])):
            with self.subTest(times=times, shape=invalid.shape), self.assertRaises(ValueError):
                infer_event_flow_initialization(self.reference, invalid, self.centers, self.camera, times)

    def test_initial_depth_alpha_gate_rejects_hidden_nodes_at_same_projection(self):
        centers = np.array([[0., 0, 2], [0, 0, 3]], dtype=np.float32)
        alpha = np.full_like(self.reference, .5)
        depth_sum = 2 * alpha
        result = infer_event_flow_initialization(self.reference, self.maps([(0, 0), (3, 0)]),
                                               centers, self.camera, [0, 1],
                                               reference_depth_np=depth_sum, reference_alpha_np=alpha)
        self.assertGreater(result["translations"][0, 0, 0], .04)
        np.testing.assert_array_equal(result["translations"][0, 1], [0, 0, 0])
        self.assertEqual(result["debug"]["initial_visible_nodes"], 1)
        self.assertEqual(result["debug"]["initial_occluded_or_low_alpha_nodes"], 1)
        self.assertEqual(result["metadata"]["visibility"], "initial_G0_depth_alpha_checked")
        self.assertFalse(result["metadata"]["future_occlusion_checked"])

    def test_low_alpha_rejects_foreground_flow_and_omission_is_explicit(self):
        alpha = np.full_like(self.reference, .01)
        result = infer_event_flow_initialization(self.reference, self.maps([(0, 0), (3, 0)]),
                                               self.centers, self.camera, [0, 1],
                                               reference_depth_np=2 * alpha, reference_alpha_np=alpha)
        np.testing.assert_array_equal(result["translations"], np.zeros((1, 4, 3)))
        self.assertEqual(result["debug"]["initial_visible_nodes"], 0)
        unchecked = self.run_flow([(0, 0), (3, 0)])
        self.assertEqual(unchecked["metadata"]["visibility"], "visibility_not_occlusion_checked")
        with self.assertRaises(ValueError):
            infer_event_flow_initialization(self.reference, self.maps([(0, 0), (3, 0)]),
                                           self.centers, self.camera, [0, 1], reference_alpha_np=alpha)

    def test_heavy_proxy_saturation_is_not_used_as_motion_evidence(self):
        saturated = np.log(np.ones_like(self.reference)) - np.log(self.reference)
        changes = np.array([np.zeros_like(self.reference), saturated])
        result = infer_event_flow_initialization(self.reference, changes, self.centers, self.camera, [0, 1])
        np.testing.assert_array_equal(result["translations"], np.zeros((1, 4, 3)))
        self.assertEqual(result["debug"]["pairs"][0]["proxy_exposed_informative_ratio"], 0)

    def test_normal_black_background_does_not_count_as_saturation_failure(self):
        reference = np.zeros_like(self.reference)
        reference[25:71, 25:71] = self.reference[25:71, 25:71]
        zeros = np.zeros_like(reference)
        result = infer_event_flow_initialization(reference, [zeros, zeros], self.centers, self.camera, [0, 1])
        self.assertEqual(result["debug"]["pairs"][0]["proxy_exposed_informative_ratio"], 1)

    def constant_flow(self, x):
        result = np.zeros((96, 96, 2), dtype=np.float32)
        result[:, :, 0] = x
        return result

    def test_failed_pair_cannot_be_revived_by_a_later_locally_consistent_pair(self):
        forward, backward = self.constant_flow(3), self.constant_flow(-3)
        # The middle reverse flow is deliberately inconsistent. The last
        # pair is locally good, but its stale coordinates are no longer G0 tracks.
        flows = [forward, backward, forward, forward, forward, backward]
        with patch("cv2.calcOpticalFlowFarneback", side_effect=flows):
            result = self.run_flow([(0, 0), (3, 0), (6, 0), (9, 0)])
        np.testing.assert_array_equal(result["accepted_masks"], [[True] * 4, [False] * 4, [False] * 4])
        np.testing.assert_allclose(result["translations"][:, :, 0], 3 * 2 / 80, atol=1e-6)
        np.testing.assert_array_equal(result["first_chain_break_knot"], 2)
        self.assertFalse(result["final_chain_valid_mask"].any())
        self.assertEqual(result["debug"]["pairs"][2]["blocked_unanchored_pair_candidates"], 4)
        self.assertEqual(result["metadata"]["version"], 3)
        self.assertFalse(result["metadata"]["retained_pose_is_observation"])

    def test_one_broken_node_does_not_invalidate_healthy_node_chains(self):
        forward, backward = self.constant_flow(3), self.constant_flow(-3)
        bad_backward = backward.copy()
        # Node zero projects to (39.5,39.5), then (42.5,39.5), and the
        # second candidate is (45.5,39.5); all other nodes are 16 pixels away.
        bad_backward[37:43, 43:49, 0] = 3
        flows = [forward, backward, forward, bad_backward, forward, backward]
        with patch("cv2.calcOpticalFlowFarneback", side_effect=flows):
            result = self.run_flow([(0, 0), (3, 0), (6, 0), (9, 0)])
        np.testing.assert_array_equal(result["accepted_masks"],
                                      [[True, True, True, True], [False, True, True, True], [False, True, True, True]])
        np.testing.assert_allclose(result["translations"][-1, :, 0], np.array([3, 9, 9, 9]) * 2 / 80, atol=1e-6)
        np.testing.assert_array_equal(result["first_chain_break_knot"], [2, -1, -1, -1])
        self.assertEqual(result["debug"]["pairs"][-1]["chain_valid_nodes"], 3)
        self.assertEqual(result["debug"]["pairs"][-1]["blocked_unanchored_pair_candidates"], 1)

    def test_no_event_interval_breaks_identity_and_retains_pose_only_as_prior(self):
        forward, backward = self.constant_flow(3), self.constant_flow(-3)
        with patch("cv2.calcOpticalFlowFarneback", side_effect=[forward, backward, forward, backward]) as flow:
            result = self.run_flow([(0, 0), (3, 0), (3, 0), (6, 0)])
        self.assertEqual(flow.call_count, 4)
        np.testing.assert_array_equal(result["accepted_masks"], [[True] * 4, [False] * 4, [False] * 4])
        np.testing.assert_allclose(result["translations"][:, :, 0], 3 * 2 / 80, atol=1e-6)
        self.assertEqual(result["debug"]["pairs"][1]["active_fraction"], 0)
        self.assertEqual(result["debug"]["pairs"][1]["newly_broken_chain_nodes"], 4)
        self.assertEqual(result["debug"]["pairs"][2]["accepted_nodes"], 0)

    def test_initial_no_event_gap_cannot_create_a_late_g0_track(self):
        with patch("cv2.calcOpticalFlowFarneback", side_effect=[self.constant_flow(3), self.constant_flow(-3)]):
            result = self.run_flow([(0, 0), (0, 0), (3, 0)])
        self.assertFalse(result["accepted_masks"].any())
        np.testing.assert_array_equal(result["translations"], 0)
        np.testing.assert_array_equal(result["first_chain_break_knot"], 1)
        self.assertEqual(result["debug"]["pairs"][-1]["blocked_unanchored_pair_candidates"], 4)

    def test_kabsch_receives_only_unbroken_chains_and_does_not_claim_new_rotations(self):
        forward, backward = self.constant_flow(3), self.constant_flow(-3)
        flows = [forward, backward, forward, forward, forward, backward]
        with patch("cv2.calcOpticalFlowFarneback", side_effect=flows), patch(
                "utils.event_only_flow_init.estimate_local_kabsch_rotations",
                wraps=estimate_local_kabsch_rotations) as kabsch:
            result = self.run_flow([(0, 0), (3, 0), (6, 0), (9, 0)])
        self.assertEqual(kabsch.call_count, 3)
        for index, call in enumerate(kabsch.call_args_list):
            np.testing.assert_array_equal(call.args[2], result["accepted_masks"][index])
        self.assertTrue(kabsch.call_args_list[0].args[2].all())
        self.assertFalse(kabsch.call_args_list[2].args[2].any())
        self.assertEqual(result["debug"]["pairs"][-1]["reliable_rotation_nodes"], 0)
        np.testing.assert_array_equal(result["rotation_vectors"][-1], result["rotation_vectors"][0])


class LocalKabschInitializationTests(unittest.TestCase):
    def setUp(self):
        self.centers = np.random.default_rng(17).uniform(-1, 1, (48, 3))
        self.tracked = np.ones(len(self.centers), dtype=bool)

    def test_pure_rigid_translation_does_not_create_rotation(self):
        shifts = np.broadcast_to([.5, -.3, 1.2], self.centers.shape).copy()
        result = estimate_local_kabsch_rotations(self.centers, shifts, self.tracked)
        self.assertTrue(result["reliable"].all())
        np.testing.assert_allclose(result["rotation_vectors"], 0, atol=1e-6, rtol=0)

    def test_known_rigid_rotation_and_translation_are_recovered(self):
        vector = np.array([.2, -.35, .45])
        rotation = cv2.Rodrigues(vector)[0]
        shifts = self.centers @ rotation.T + [.3, -.7, .2] - self.centers
        result = estimate_local_kabsch_rotations(self.centers, shifts, self.tracked)
        self.assertTrue(result["reliable"].all())
        np.testing.assert_allclose(result["rotation_vectors"], np.broadcast_to(vector, self.centers.shape),
                                   atol=1e-5, rtol=0)
        self.assertLess(result["fit_relative_rmse"].max(), 1e-6)

    def test_nondegenerate_planar_patch_supports_rotation(self):
        centers = self.centers.copy()
        centers[:, 2] = 0
        vector = np.array([.3, .2, -.4])
        rotation = cv2.Rodrigues(vector)[0]
        shifts = centers @ rotation.T - centers
        result = estimate_local_kabsch_rotations(centers, shifts, self.tracked)
        self.assertTrue(result["reliable"].all())
        np.testing.assert_allclose(result["rotation_vectors"], np.broadcast_to(vector, centers.shape), atol=1e-5, rtol=0)

    def test_collinear_or_insufficiently_tracked_nodes_remain_zero(self):
        line = np.column_stack((np.arange(8), np.zeros((8, 2))))
        for centers, tracked in ((line, np.ones(8, bool)), (self.centers, np.arange(48) < 3)):
            with self.subTest(count=len(centers)):
                result = estimate_local_kabsch_rotations(centers, np.ones_like(centers), tracked)
                self.assertFalse(result["reliable"].any())
                np.testing.assert_array_equal(result["rotation_vectors"], np.zeros_like(centers))

    def test_pause_preserves_previous_reliable_rotation_instead_of_resetting_it(self):
        previous = np.broadcast_to([.2, -.1, .3], self.centers.shape).copy()
        result = estimate_local_kabsch_rotations(self.centers, np.zeros_like(self.centers),
                                                 np.zeros(48, bool), previous)
        self.assertFalse(result["reliable"].any())
        np.testing.assert_allclose(result["rotation_vectors"], previous, atol=1e-7, rtol=0)

    def test_reflection_or_strong_nonrigid_stretch_is_not_certified_as_rotation(self):
        # Symmetric full-rank support makes a reflection irreducibly nonrigid
        # under a proper det(+1) rotation.
        centers = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
        for moved in (centers * [-1, 1, 1], centers * [3, 1, 1]):
            result = estimate_local_kabsch_rotations(centers, moved - centers, np.ones(8, bool))
            self.assertFalse(result["reliable"].any())
            np.testing.assert_array_equal(result["rotation_vectors"], np.zeros_like(centers))

    def test_axis_angle_branch_is_continuous_across_pi(self):
        previous = np.broadcast_to([0, 0, 3.1], self.centers.shape).copy()
        rotation = cv2.Rodrigues(np.array([0., 0., 3.2]))[0]
        shifts = self.centers @ rotation.T - self.centers
        result = estimate_local_kabsch_rotations(self.centers, shifts, self.tracked, previous)
        np.testing.assert_allclose(result["rotation_vectors"], np.broadcast_to([0, 0, 3.2], self.centers.shape),
                                   atol=1e-5, rtol=0)


if __name__ == "__main__":
    unittest.main()
