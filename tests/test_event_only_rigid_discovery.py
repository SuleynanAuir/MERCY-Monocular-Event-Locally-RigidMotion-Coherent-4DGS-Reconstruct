"""Synthetic CPU multi-part SE3 projections, never real data or GPU."""

import unittest
import json
from unittest import mock

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from utils.event_only_rigid_discovery import discover_rigid_groups


class EventOnlyRigidDiscoveryTests(unittest.TestCase):
    def fixture(self, groups=2, rotated_camera=False):
        random = np.random.default_rng(82)
        centers = np.concatenate([random.uniform(-.25, .25, (24, 3)) + [index * .8 - .6, 0, 4.]
                                  for index in range(groups)])
        static_centers = random.uniform(-.2, .2, (10, 3)) + [0, -.8, 4.5]
        centers = np.concatenate((centers, static_centers, [[0, 0, 4.], [0, .1, 4.]]))
        camera = {"width": 640, "height": 480, "fx": 650., "fy": 650.,
                  "cx": 320., "cy": 240., "pixel_center": "boundary", "w2c": np.eye(4)}
        if rotated_camera:
            camera["w2c"][:3, :3] = Rotation.from_rotvec([.05, -.1, .08]).as_matrix()
            camera["w2c"][:3, 3] = [.2, -.15, .3]
        tracks = np.zeros((4, len(centers), 2))
        known_poses = np.tile(np.eye(4), (4, groups, 1, 1))
        for time in range(4):
            moved = centers.copy()
            for group in range(groups):
                pose = np.eye(4)
                if group == 0:
                    pose[:3, :3] = Rotation.from_rotvec([.02, .025, .04 * (time + 1)]).as_matrix()
                    pose[:3, 3] = [.12 * (time + 1), .035 * (time + 1), .01 * (time + 1)]
                elif group == 1:
                    pose[:3, :3] = Rotation.from_rotvec([-.03 * (time + 1), .02, -.025]).as_matrix()
                    pose[:3, 3] = [-.11 * (time + 1), -.04 * (time + 1), -.015 * (time + 1)]
                else:
                    pose[:3, :3] = Rotation.from_rotvec([.025, -.04 * (time + 1), .015]).as_matrix()
                    pose[:3, 3] = [.015 * (time + 1), .14 * (time + 1), .02 * (time + 1)]
                known_poses[time, group] = pose
                ids = slice(group * 24, (group + 1) * 24)
                moved[ids] = centers[ids] @ pose[:3, :3].T + pose[:3, 3]
            projected = moved @ camera["w2c"][:3, :3].T + camera["w2c"][:3, 3]
            tracks[time, :, 0] = camera["fx"] * projected[:, 0] / projected[:, 2] + camera["cx"]
            tracks[time, :, 1] = camera["fy"] * projected[:, 1] / projected[:, 2] + camera["cy"]
        accepted = np.ones(tracks.shape[:2], bool)
        accepted[:, -2:] = False
        tracks[:, -2:] = np.nan
        static = np.zeros(len(centers), bool)
        static[groups * 24:-2] = True
        return centers, tracks, accepted, camera, static, known_poses

    def check_groups(self, groups=2, rotated_camera=False):
        centers, tracks, accepted, camera, static, expected = self.fixture(groups, rotated_camera)
        result = discover_rigid_groups(centers, tracks, accepted, camera, static,
                                       max_groups=groups, reprojection_threshold_px=.8)
        labels = []
        for group in range(groups):
            unique = np.unique(result["labels"][group * 24:(group + 1) * 24])
            self.assertEqual(len(unique), 1)
            self.assertGreater(unique[0], 0)
            label = int(unique[0])
            labels.append(label)
            self.assertTrue(result["pose_valid"][:, label].all())
            np.testing.assert_allclose(result["poses"][:, label], expected[:, group], atol=1e-4, rtol=0)
        self.assertEqual(len(set(labels)), groups)
        np.testing.assert_array_equal(result["labels"][static], np.zeros(static.sum(), dtype=int))
        np.testing.assert_array_equal(result["labels"][-2:], [-1, -1])
        np.testing.assert_allclose(result["poses"][:, 0], np.tile(np.eye(4), (4, 1, 1)))
        self.assertGreaterEqual(len(result["diagnostics"]["selected_high_motion_times"]), 2)
        self.assertFalse(result["metadata"]["labels_are_ground_truth_parts"])

    def test_two_rigid_groups_and_static_identity(self):
        self.check_groups(2)

    def test_three_rigid_groups(self):
        self.check_groups(3)

    def test_world_se3_recovery_inverts_nonidentity_camera(self):
        self.check_groups(2, rotated_camera=True)

    def test_single_timestamp_support_is_not_enough_to_invent_groups(self):
        centers, tracks, accepted, camera, static, _ = self.fixture()
        accepted[1:] = False
        result = discover_rigid_groups(centers, tracks, accepted, camera, static)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 0)
        self.assertTrue((result["labels"][~static] == -1).all())

    def test_missing_time_pose_is_explicitly_invalid_not_interpolated(self):
        centers, tracks, accepted, camera, static, _ = self.fixture()
        accepted[0] = False
        result = discover_rigid_groups(centers, tracks, accepted, camera, static, max_groups=2,
                                       reprojection_threshold_px=.8)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 2)
        self.assertFalse(result["pose_valid"][0, 1:].any())
        self.assertTrue(np.isnan(result["poses"][0, 1:]).all())

    def test_unsupported_or_malformed_inputs_fail_conservatively(self):
        centers, tracks, accepted, camera, static, _ = self.fixture()
        empty = discover_rigid_groups(centers, tracks, np.zeros_like(accepted), camera, static)
        self.assertEqual(empty["diagnostics"]["dynamic_groups"], 0)
        with self.assertRaises(ValueError):
            discover_rigid_groups(centers, tracks[:, :, :1], accepted, camera)
        broken_camera = dict(camera, pixel_center="integer")
        with self.assertRaises(ValueError):
            discover_rigid_groups(centers, tracks, accepted, broken_camera)

    def test_opencv_degenerate_solver_failure_becomes_unknown_not_invented_pose(self):
        centers, tracks, accepted, camera, static, _ = self.fixture()
        with mock.patch.object(cv2, "solvePnPRansac", side_effect=cv2.error("synthetic degenerate solver")):
            result = discover_rigid_groups(centers, tracks, accepted, camera, static)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 0)
        self.assertTrue((result["labels"][~static] == -1).all())
        self.assertTrue(result["diagnostics"]["proposals"])
        for proposal in result["diagnostics"]["proposals"]:
            self.assertEqual(proposal["failure_stage"], "initial_pnp")
            self.assertEqual(proposal["reason"], "pnp_ransac_exception")
            self.assertGreaterEqual(proposal["initial_fit"]["valid_nodes"], 8)

    def test_time_selection_uses_shared_support_not_largest_disjoint_motion(self):
        centers, tracks, accepted, camera, static, expected = self.fixture(groups=1)
        accepted[:, :24] = False
        accepted[0, :8] = True
        accepted[1, 8:16] = True
        accepted[2:, 16:24] = True
        tracks[0, :8] += 300.
        tracks[1, 8:16] += 250.
        result = discover_rigid_groups(centers, tracks, accepted, camera, static,
                                       max_groups=1, reprojection_threshold_px=.8)
        self.assertEqual(result["diagnostics"]["selected_high_motion_times"], [2, 3])
        selection = result["diagnostics"]["time_selections"][0]
        self.assertEqual(selection["common_all_selected_nodes"], 8)
        self.assertEqual(selection["pairwise_common_valid_nodes"], [[8, 8], [8, 8]])
        self.assertTrue((result["labels"][:16] == -1).all())
        self.assertTrue((result["labels"][16:24] == 1).all())
        np.testing.assert_allclose(result["poses"][2:, 1], expected[2:, 0], atol=1e-4, rtol=0)

    def test_equally_supported_time_selection_prefers_temporal_spread(self):
        centers, tracks, accepted, camera, static, _ = self.fixture(groups=1)
        result = discover_rigid_groups(centers, tracks, accepted, camera, static,
                                       consensus_timestamps=2, max_groups=1)
        self.assertEqual(result["diagnostics"]["selected_high_motion_times"], [0, 3])

    def test_three_parts_survive_sparse_missing_observations_without_imputation(self):
        centers, tracks, accepted, camera, static, expected = self.fixture(groups=3)
        for group in range(3):
            for time in range(4):
                begin = group * 24 + time * 4
                accepted[time, begin:begin + 4] = False
                tracks[time, begin:begin + 4] = np.nan
        result = discover_rigid_groups(centers, tracks, accepted, camera, static,
                                       max_groups=3, reprojection_threshold_px=.8)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 3)
        labels = []
        for group in range(3):
            unique = np.unique(result["labels"][group * 24:(group + 1) * 24])
            self.assertEqual(len(unique), 1)
            self.assertGreater(unique[0], 0)
            labels.append(int(unique[0]))
            np.testing.assert_allclose(result["poses"][:, unique[0]], expected[:, group], atol=1e-4, rtol=0)
        self.assertEqual(len(set(labels)), 3)

    def test_local_minority_motion_is_not_drowned_by_unlabelled_static_background(self):
        centers, tracks, accepted, camera, static, expected = self.fixture(groups=1)
        random = np.random.default_rng(183)
        background = random.uniform(-.3, .3, (240, 3)) + [1.2, .9, 4.6]
        background_tracks = np.column_stack((camera["fx"] * background[:, 0] / background[:, 2] + camera["cx"],
                                            camera["fy"] * background[:, 1] / background[:, 2] + camera["cy"]))
        centers = np.concatenate((centers, background))
        tracks = np.concatenate((tracks, np.repeat(background_tracks[None], len(tracks), axis=0)), axis=1)
        accepted = np.concatenate((accepted, np.ones((len(tracks), len(background)), dtype=bool)), axis=1)
        # No privileged static labels: identity-consistent background is not
        # promoted to a dynamic part just because it dominates global RANSAC.
        result = discover_rigid_groups(centers, tracks, accepted, camera, max_groups=1,
                                       reprojection_threshold_px=.8, local_seed_count=8, local_seed_size=32)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 1)
        self.assertTrue((result["labels"][:24] == 1).all())
        self.assertTrue((result["labels"][-240:] == -1).all())
        np.testing.assert_allclose(result["poses"][:, 1], expected[:, 0], atol=1e-4, rtol=0)
        self.assertTrue(all(score < .75 for score in result["diagnostics"]["motion_scores_px"]))
        self.assertEqual(result["diagnostics"]["motion_seed_nodes"], 24)
        self.assertTrue(any(proposal["reason"] == "accepted"
                            for proposal in result["diagnostics"]["proposals"]))
        self.assertTrue(all(proposal["initial_fit"]["candidate_nodes"] <= 24
                            for proposal in result["diagnostics"]["proposals"]))
        self.assertLessEqual(sum(selection["scope"] == "local_geometry"
                                 for selection in result["diagnostics"]["time_selections"]), 8)

    def test_disjoint_time_support_does_not_invent_a_group_or_call_pnp(self):
        centers, tracks, accepted, camera, static, _ = self.fixture(groups=1)
        accepted[:] = False
        accepted[0, :8] = True
        accepted[1, 8:16] = True
        with mock.patch.object(cv2, "solvePnPRansac") as solver:
            result = discover_rigid_groups(centers, tracks, accepted, camera, static)
        solver.assert_not_called()
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 0)
        self.assertEqual(result["diagnostics"]["time_selections"][0]["reason"],
                         "insufficient_cross_time_common_nodes")
        self.assertEqual(result["diagnostics"]["valid_nonstatic_nodes_by_time"], [8, 8, 0, 0])

    def test_only_seven_shared_nodes_remain_unknown(self):
        centers, tracks, accepted, camera, static, _ = self.fixture(groups=1)
        accepted[:, 7:] = False
        result = discover_rigid_groups(centers, tracks, accepted, camera, static)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 0)
        self.assertEqual(result["diagnostics"]["time_selections"][0]["reason"],
                         "insufficient_motion_timestamps")

    def test_collinear_geometry_records_specific_failure_and_stays_unknown(self):
        centers = np.column_stack((np.linspace(-.5, .5, 16), np.zeros(16), np.full(16, 4.)))
        camera = self.fixture(groups=1)[3]
        projected = np.column_stack((camera["fx"] * centers[:, 0] / centers[:, 2] + camera["cx"],
                                     np.full(16, camera["cy"])))
        tracks = np.stack([projected + [20., 0.], projected + [35., 0.], projected + [45., 0.]])
        result = discover_rigid_groups(centers, tracks, np.ones((3, 16), bool), camera)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 0)
        self.assertTrue(result["diagnostics"]["proposals"])
        for proposal in result["diagnostics"]["proposals"]:
            self.assertEqual(proposal["reason"], "degenerate_3d_support")
            self.assertEqual(proposal["failure_stage"], "initial_pnp")
        # Reports can be persisted without NumPy encoders or nonfinite JSON.
        json.dumps(result["diagnostics"], allow_nan=False)

    def test_configuration_cannot_weaken_evidence_floors(self):
        centers, tracks, accepted, camera, static, _ = self.fixture(groups=1)
        for arguments in ({"min_group_nodes": 7}, {"min_consensus_timestamps": 1},
                          {"reprojection_threshold_px": 2.51}, {"local_seed_count": 17}):
            with self.assertRaises(ValueError):
                discover_rigid_groups(centers, tracks, accepted, camera, static, **arguments)

    def test_small_motion_among_static_background_does_not_create_dynamic_group(self):
        random = np.random.default_rng(319)
        centers = random.uniform(-.3, .3, (96, 3)) + [0., 0., 4.]
        camera = self.fixture(groups=1)[3]
        reference = np.column_stack((camera["fx"] * centers[:, 0] / centers[:, 2] + camera["cx"],
                                     camera["fy"] * centers[:, 1] / centers[:, 2] + camera["cy"]))
        tracks = np.repeat(reference[None], 4, axis=0)
        tracks[:, :16, 0] += np.array([1.5, -1.5, 1.5, -1.5])[:, None]
        result = discover_rigid_groups(centers, tracks, np.ones((4, 96), bool), camera)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 0)
        self.assertTrue((result["labels"] == -1).all())
        self.assertEqual(result["diagnostics"]["explicit_static_nodes"], 0)
        self.assertEqual(result["diagnostics"]["motion_seed_nodes"], 0)
        self.assertEqual(result["diagnostics"]["time_selections"][0]["search_rejected_reason"],
                         "insufficient_motion_seed_nodes")
        for proposal in result["diagnostics"]["proposals"]:
            for evidence in proposal.get("fitted_motion_evidence", []):
                self.assertFalse(evidence["dynamic"])
        self.assertEqual(result["metadata"]["subthreshold_motion_policy"],
                         "unresolved_not_automatically_static")

    def test_uniform_coherent_two_pixel_motion_is_unresolved_not_declared_static(self):
        centers, tracks, accepted, camera, static, _ = self.fixture(groups=1)
        centers = centers[:24]
        reference = np.column_stack((camera["fx"] * centers[:, 0] / centers[:, 2] + camera["cx"],
                                     camera["fy"] * centers[:, 1] / centers[:, 2] + camera["cy"]))
        tracks = np.repeat(reference[None], 3, axis=0)
        for time, translation in enumerate((.011, .012, .013)):
            tracks[time, :, 0] += camera["fx"] * translation / centers[:, 2]
        result = discover_rigid_groups(centers, tracks, np.ones((3, 24), bool), camera)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 0)
        self.assertTrue((result["labels"] == -1).all())

    def test_all_time_refinement_splits_parts_matching_only_at_selected_endpoints(self):
        random = np.random.default_rng(917)
        centers = np.concatenate((random.uniform(-.25, .25, (40, 3)) + [-.5, 0., 4.],
                                  random.uniform(-.25, .25, (16, 3)) + [.5, 0., 4.]))
        camera = self.fixture(groups=1)[3]
        tracks = np.zeros((11, 56, 2))
        expected = np.tile(np.eye(4), (11, 2, 1, 1))
        for time in range(11):
            expected[time, :, 0, 3] = .3
            expected[time, 1, 1, 3] = .3 * np.sin(np.pi * time / 10.)
            moved = centers.copy()
            moved[:40] += expected[time, 0, :3, 3]
            moved[40:] += expected[time, 1, :3, 3]
            tracks[time, :, 0] = camera["fx"] * moved[:, 0] / moved[:, 2] + camera["cx"]
            tracks[time, :, 1] = camera["fy"] * moved[:, 1] / moved[:, 2] + camera["cy"]
        result = discover_rigid_groups(centers, tracks, np.ones((11, 56), bool), camera,
                                       max_groups=2, consensus_timestamps=2,
                                       reprojection_threshold_px=.8)
        self.assertEqual(result["diagnostics"]["selected_high_motion_times"], [0, 10])
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 2)
        first_label, second_label = int(result["labels"][0]), int(result["labels"][40])
        self.assertGreater(first_label, 0)
        self.assertGreater(second_label, 0)
        self.assertNotEqual(first_label, second_label)
        self.assertTrue((result["labels"][:40] == first_label).all())
        self.assertTrue((result["labels"][40:] == second_label).all())
        np.testing.assert_allclose(result["poses"][:, first_label], expected[:, 0], atol=1e-4, rtol=0)
        np.testing.assert_allclose(result["poses"][:, second_label], expected[:, 1], atol=1e-4, rtol=0)
        pruning = result["diagnostics"]["groups"][first_label - 1]["timeline_refinement"]
        self.assertEqual(pruning["before_nodes"], 56)
        self.assertEqual(pruning["after_nodes"], 40)
        self.assertEqual(len(pruning["rounds"]), 2)
        self.assertEqual(pruning["final_audit"]["pose_valid_eligible_coverage"], 1.)
        for group in result["diagnostics"]["groups"]:
            audit = group["timeline_refinement"]["final_audit"]
            self.assertEqual(audit["pose_valid_time_count"], 11)
            self.assertGreaterEqual(audit["dynamic_time_count"], 2)
            self.assertTrue(all(count >= 2 for count in audit["inlier_counts"]))
        json.dumps(result["diagnostics"], allow_nan=False)

    def test_solver_failures_cannot_hide_observed_times_from_coverage_denominator(self):
        centers, tracks, accepted, camera, static, _ = self.fixture(groups=1)
        original = cv2.solvePnPRansac
        def fail_middle_frames(object_points, image_points, *args, **kwargs):
            first_id = int(np.argmin(np.linalg.norm(centers - object_points[0], axis=1)))
            distance = np.linalg.norm(tracks[:, first_id] - .5 - image_points[0], axis=1)
            time = int(np.argmin(distance))
            if time in (1, 2):
                raise cv2.error("synthetic unsupported middle frame")
            return original(object_points, image_points, *args, **kwargs)
        with mock.patch.object(cv2, "solvePnPRansac", side_effect=fail_middle_frames):
            result = discover_rigid_groups(centers, tracks, accepted, camera, static,
                                           max_groups=1, consensus_timestamps=2)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 0)
        self.assertTrue((result["labels"][~static] == -1).all())
        rejected = [proposal for proposal in result["diagnostics"]["proposals"]
                    if proposal["reason"] == "insufficient_final_eligible_time_coverage"]
        self.assertTrue(rejected)
        audit = rejected[0]["timeline_refinement"]["final_audit"]
        self.assertEqual(audit["eligible_time_count"], 4)
        self.assertEqual(audit["pose_valid_time_count"], 2)
        self.assertEqual(audit["pose_valid_eligible_coverage"], .5)
        self.assertEqual(result["metadata"]["coverage_rule_scope"],
                         "candidate_initialization_usability_not_event_window_acceptance")

    def test_time_selection_does_not_prefer_unresolved_endpoints_over_real_middle_motion(self):
        centers, _, _, camera, static, _ = self.fixture(groups=1)
        centers = centers[:24]
        tracks = np.zeros((7, 24, 2))
        for time, pixels_at_depth_four in enumerate((1., 10., 20., 40., 20., 10., 1.)):
            moved = centers + [pixels_at_depth_four * 4. / camera["fx"], 0., 0.]
            tracks[time, :, 0] = camera["fx"] * moved[:, 0] / moved[:, 2] + camera["cx"]
            tracks[time, :, 1] = camera["fy"] * moved[:, 1] / moved[:, 2] + camera["cy"]
        result = discover_rigid_groups(centers, tracks, np.ones((7, 24), bool), camera,
                                       consensus_timestamps=2, max_groups=1)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 1)
        selected = result["diagnostics"]["selected_high_motion_times"]
        self.assertEqual(len(selected), 2)
        self.assertNotIn(0, selected)
        self.assertNotIn(6, selected)
        self.assertTrue(result["diagnostics"]["time_selections"][0]["preferred_resolvable_motion_times"])
        self.assertTrue(result["pose_valid"][:, 1].all())

    def test_quiet_majority_cannot_dilute_motion_informative_mismatches(self):
        random = np.random.default_rng(361)
        centers = np.concatenate((random.uniform(-.2, .2, (24, 3)) + [-.4, 0., 4.],
                                  random.uniform(-.2, .2, (8, 3)) + [.4, 0., 4.]))
        camera = self.fixture(groups=1)[3]
        tracks = np.zeros((20, 32, 2))
        expected = np.tile(np.eye(4), (20, 2, 1, 1))
        for time in range(20):
            if time in (0, 19):
                expected[time, :, 0, 3] = .3
            elif time in (5, 10, 15):
                expected[time, 0, 0, 3] = .3
                expected[time, 1, 1, 3] = .3
            moved = centers.copy()
            moved[:24] += expected[time, 0, :3, 3]
            moved[24:] += expected[time, 1, :3, 3]
            tracks[time, :, 0] = camera["fx"] * moved[:, 0] / moved[:, 2] + camera["cx"]
            tracks[time, :, 1] = camera["fy"] * moved[:, 1] / moved[:, 2] + camera["cy"]
        result = discover_rigid_groups(centers, tracks, np.ones((20, 32), bool), camera,
                                       max_groups=2, consensus_timestamps=2)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 2)
        first, second = int(result["labels"][0]), int(result["labels"][24])
        self.assertGreater(first, 0)
        self.assertGreater(second, 0)
        self.assertNotEqual(first, second)
        self.assertTrue((result["labels"][:24] == first).all())
        self.assertTrue((result["labels"][24:] == second).all())
        pruning = result["diagnostics"]["groups"][first - 1]["timeline_refinement"]
        first_round = pruning["rounds"][0]
        self.assertEqual(first_round["before_nodes"], 32)
        self.assertEqual(first_round["all_time_consistent_nodes"], 32)
        self.assertEqual(first_round["motion_informative_consistent_nodes"], 24)
        self.assertEqual(first_round["after_nodes"], 24)
        np.testing.assert_allclose(result["poses"][:, first], expected[:, 0], atol=1e-4, rtol=0)
        np.testing.assert_allclose(result["poses"][:, second], expected[:, 1], atol=1e-4, rtol=0)

    def test_unresolved_pivot_points_are_not_claimed_as_observed_dynamic_members(self):
        random = np.random.default_rng(483)
        moving = random.uniform(-.05, .05, (24, 3)) + [.4, 0., 4.3]
        pivot = np.column_stack((np.zeros(8), np.linspace(-.15, .15, 8), np.full(8, 4.)))
        centers = np.concatenate((moving, pivot))
        center = np.array([0., 0., 4.])
        camera = self.fixture(groups=1)[3]
        tracks = np.zeros((4, 32, 2))
        for time in range(4):
            rotation = Rotation.from_rotvec([0., .2 + .1 * time, 0.]).as_matrix()
            moved = (centers - center) @ rotation.T + center
            tracks[time, :, 0] = camera["fx"] * moved[:, 0] / moved[:, 2] + camera["cx"]
            tracks[time, :, 1] = camera["fy"] * moved[:, 1] / moved[:, 2] + camera["cy"]
        result = discover_rigid_groups(centers, tracks, np.ones((4, 32), bool), camera, max_groups=1)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 1)
        self.assertTrue((result["labels"][:24] == 1).all())
        self.assertTrue((result["labels"][24:] == -1).all())
        self.assertEqual(result["diagnostics"]["motion_seed_nodes"], 24)

    def test_local_search_still_runs_when_global_motion_seed_solver_fails(self):
        centers, tracks, accepted, camera, static, _ = self.fixture(groups=2)
        original = cv2.solvePnPRansac
        def reject_large_global_set(object_points, *args, **kwargs):
            if len(object_points) > 32:
                raise cv2.error("synthetic global hypothesis failure")
            return original(object_points, *args, **kwargs)
        with mock.patch.object(cv2, "solvePnPRansac", side_effect=reject_large_global_set):
            result = discover_rigid_groups(centers, tracks, accepted, camera, static, max_groups=2)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 2)
        self.assertTrue(any(proposal["scope"] == "local_geometry" and proposal["reason"] == "accepted"
                            for proposal in result["diagnostics"]["proposals"]))

    def test_temporal_prior_selects_continuous_branch_only_after_same_pixel_gates(self):
        grid_x, grid_y = np.meshgrid(np.linspace(-.08, .08, 6), np.linspace(-.07, .07, 4))
        centers = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1), np.full(24, 4.)))
        center = np.array([0., 0., 4.])
        camera = self.fixture(groups=1)[3]
        tracks = np.zeros((4, 24, 2))
        expected = np.tile(np.eye(4), (4, 1, 1))
        for time in range(4):
            rotation = Rotation.from_rotvec([0., .25, 0.]).as_matrix()
            translation = center - rotation @ center + [.12 * (time + 1), 0., 0.]
            expected[time, :3, :3] = rotation
            expected[time, :3, 3] = translation
            moved = centers @ rotation.T + translation
            tracks[time, :, 0] = camera["fx"] * moved[:, 0] / moved[:, 2] + camera["cx"]
            tracks[time, :, 1] = camera["fy"] * moved[:, 1] / moved[:, 2] + camera["cy"]
        def time_index(object_points, image_points):
            point = int(np.argmin(np.linalg.norm(centers - object_points[0], axis=1)))
            return int(np.argmin(np.linalg.norm(tracks[:, point] - .5 - image_points[0], axis=1)))
        def ambiguous_ransac(object_points, image_points, *args, **kwargs):
            time = time_index(object_points, image_points)
            rvec = np.array([[0.], [.25 if time % 2 == 0 else -.25], [0.]])
            rotation = cv2.Rodrigues(rvec)[0]
            translation = center - rotation @ center + [.12 * (time + 1), 0., 0.]
            return True, rvec, translation.reshape(3, 1), np.arange(len(object_points), dtype=np.int32)[:, None]
        def previous_branch(object_points, image_points, *args, **kwargs):
            time = time_index(object_points, image_points)
            rvec = kwargs["rvec"].copy()
            rotation = cv2.Rodrigues(rvec)[0]
            translation = center - rotation @ center + [.12 * (time + 1), 0., 0.]
            return True, rvec, translation.reshape(3, 1)
        def retain_injected_branch(*args, **kwargs):
            return args[4].copy(), args[5].copy()
        with mock.patch.object(cv2, "solvePnPRansac", side_effect=ambiguous_ransac), \
                mock.patch.object(cv2, "solvePnPRefineLM", side_effect=retain_injected_branch), \
                mock.patch.object(cv2, "solvePnP", side_effect=previous_branch):
            result = discover_rigid_groups(centers, tracks, np.ones((4, 24), bool), camera, max_groups=1)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 1)
        self.assertTrue(result["pose_valid"][:, 1].all())
        np.testing.assert_allclose(result["poses"][:, 1], expected, atol=1e-6, rtol=0)
        fits = result["diagnostics"]["groups"][0]["times"]
        self.assertTrue(any(fit["temporal_branch_selection"]["selected"] == "previous_pose_initialized_iterative"
                            for fit in fits))
        self.assertTrue(result["metadata"]["temporal_branch_continuity_is_regularization_not_observation"])
        self.assertIsNone(result["metadata"]["temporal_branch_hard_speed_limit"])

    def test_temporal_prior_candidate_failing_observations_cannot_replace_supported_pose(self):
        centers, tracks, accepted, camera, static, expected = self.fixture(groups=1)
        with mock.patch.object(cv2, "solvePnP", return_value=(True, np.zeros((3, 1)), np.full((3, 1), 100.))):
            result = discover_rigid_groups(centers, tracks, accepted, camera, static, max_groups=1)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 1)
        np.testing.assert_allclose(result["poses"][:, 1], expected[:, 0], atol=1e-4, rtol=0)
        alternatives = [candidate for fit in result["diagnostics"]["groups"][0]["times"]
                        for candidate in fit["temporal_branch_selection"]["candidates"]
                        if candidate["source"] == "previous_pose_initialized_iterative"]
        self.assertTrue(alternatives)
        self.assertTrue(all(not candidate["accepted"] for candidate in alternatives))

    def camera_plane_fixture(self, count=32, angles=(.07, .15, .22, .3)):
        random = np.random.default_rng(617)
        camera = {"width": 640, "height": 480, "fx": 650., "fy": 540.,
                  "cx": 320., "cy": 240., "pixel_center": "boundary", "w2c": np.eye(4)}
        camera["w2c"][:3, :3] = Rotation.from_rotvec([.12, -.18, .08]).as_matrix()
        camera["w2c"][:3, 3] = [.2, -.15, .3]
        source_camera = np.column_stack((random.uniform(.2, .7, count),
                                         random.uniform(-.25, .25, count),
                                         random.uniform(3., 5., count)))
        camera_rotation = camera["w2c"][:3, :3]
        camera_translation = camera["w2c"][:3, 3]
        centers = (source_camera - camera_translation) @ camera_rotation
        tracks = np.zeros((len(angles), count, 2))
        poses = np.tile(np.eye(4), (len(angles), 1, 1))
        for time, angle in enumerate(angles):
            rotation = Rotation.from_rotvec([0., 0., angle]).as_matrix()
            translation = np.array([.12 + .06 * time, -.04 - .025 * time, 0.])
            moved = source_camera @ rotation.T + translation
            tracks[time, :, 0] = camera["fx"] * moved[:, 0] / moved[:, 2] + camera["cx"]
            tracks[time, :, 1] = camera["fy"] * moved[:, 1] / moved[:, 2] + camera["cy"]
            poses[time, :3, :3] = camera_rotation.T @ rotation @ camera_rotation
            poses[time, :3, 3] = camera_rotation.T @ (
                rotation @ camera_translation + translation - camera_translation)
        return centers, tracks, np.ones(tracks.shape[:2], bool), camera, poses, source_camera

    def test_camera_plane_recovers_known_pose_with_nonidentity_camera_and_preserves_each_depth(self):
        centers, tracks, accepted, camera, expected, source_camera = self.camera_plane_fixture()
        result = discover_rigid_groups(centers, tracks, accepted, camera, max_groups=1,
                                       motion_model="camera_plane", reprojection_threshold_px=.8)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 1)
        self.assertTrue((result["labels"] == 1).all())
        self.assertTrue(result["pose_valid"][:, 1].all())
        np.testing.assert_allclose(result["poses"][:, 1], expected, atol=1e-10, rtol=0)
        self.assertGreater(float(np.ptp(source_camera[:, 2])), 1.)
        for pose in result["poses"][:, 1]:
            moved_world = centers @ pose[:3, :3].T + pose[:3, 3]
            moved_camera = moved_world @ camera["w2c"][:3, :3].T + camera["w2c"][:3, 3]
            np.testing.assert_allclose(moved_camera[:, 2], source_camera[:, 2], atol=1e-12, rtol=0)
            np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12, rtol=0)
            self.assertAlmostEqual(float(np.linalg.det(pose[:3, :3])), 1.)
        prior = result["metadata"]["camera_plane_prior"]
        self.assertFalse(prior["confirmed"])
        self.assertFalse(prior["extra_observations_used"])
        self.assertTrue(prior["nonplanar_G0_does_not_validate_motion_prior"])
        self.assertFalse(result["metadata"]["physical_motion_validated"])
        self.assertFalse(result["metadata"]["physical_uniqueness_guaranteed"])
        self.assertEqual(result["metadata"]["minimum_group_nodes"], 8)
        self.assertEqual(result["metadata"]["minimum_consensus_fraction"], .8)
        json.dumps(result["metadata"], allow_nan=False)

    def test_camera_plane_rejects_outliers_with_true_perspective_residuals_and_bounded_sampling(self):
        centers, tracks, accepted, camera, expected, _ = self.camera_plane_fixture(count=60)
        random = np.random.default_rng(709)
        tracks[:, :48] += random.normal(0., .08, tracks[:, :48].shape)
        tracks[:, 48:] += random.uniform(40., 120., tracks[:, 48:].shape)
        result = discover_rigid_groups(centers, tracks, accepted, camera, max_groups=1,
                                       motion_model="camera_plane", reprojection_threshold_px=.8,
                                       plane_ransac_iterations=96)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 1)
        self.assertTrue((result["labels"][:48] == 1).all())
        self.assertTrue((result["labels"][48:] == -1).all())
        np.testing.assert_allclose(result["poses"][:, 1], expected, atol=.003, rtol=0)
        for time, pose in enumerate(result["poses"][:, 1]):
            moved = centers[:48] @ pose[:3, :3].T + pose[:3, 3]
            projected = moved @ camera["w2c"][:3, :3].T + camera["w2c"][:3, 3]
            pixels = np.column_stack((camera["fx"] * projected[:, 0] / projected[:, 2] + camera["cx"],
                                      camera["fy"] * projected[:, 1] / projected[:, 2] + camera["cy"]))
            self.assertTrue((np.linalg.norm(pixels - tracks[time, :48], axis=1) <= .8).all())
        fits = [proposal["initial_fit"] for proposal in result["diagnostics"]["proposals"]
                if "initial_fit" in proposal]
        fits += result["diagnostics"]["groups"][0]["times"]
        self.assertTrue(fits)
        for fit in fits:
            self.assertEqual(fit["ransac_sample_size"], 3)
            self.assertLessEqual(fit["ransac_iterations"], 96)

    def test_camera_plane_never_uses_PnP_or_a_PnP_temporal_branch(self):
        centers, tracks, accepted, camera, _, _ = self.camera_plane_fixture()
        with mock.patch.object(cv2, "solvePnPRansac", side_effect=AssertionError("not the plane solver")), \
                mock.patch.object(cv2, "solvePnPRefineLM", side_effect=AssertionError("not the plane solver")), \
                mock.patch.object(cv2, "solvePnP", side_effect=AssertionError("no plane PnP branch")):
            result = discover_rigid_groups(centers, tracks, accepted, camera,
                                           motion_model="camera_plane", max_groups=1)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 1)
        for fit in result["diagnostics"]["groups"][0]["times"]:
            self.assertEqual(fit["temporal_branch_selection"]["selected"], "camera_plane_unique_fit")
            self.assertEqual(fit["temporal_branch_selection"]["candidates"], [])

    def test_camera_plane_continuous_angle_crosses_pi_without_filling_missing_time(self):
        angles = np.array([2.7, 2.95, 3.2, 3.45])
        centers, tracks, accepted, camera, expected, _ = self.camera_plane_fixture(angles=angles)
        accepted[1] = False
        tracks[1] = np.nan
        result = discover_rigid_groups(centers, tracks, accepted, camera,
                                       motion_model="camera_plane", max_groups=1)
        self.assertEqual(result["diagnostics"]["dynamic_groups"], 1)
        observed = np.array([0, 2, 3])
        np.testing.assert_allclose(result["poses"][observed, 1], expected[observed], atol=1e-10, rtol=0)
        np.testing.assert_allclose(result["camera_plane_angles_unwrapped"][observed, 1],
                                   angles[observed], atol=1e-10, rtol=0)
        axis_world = camera["w2c"][:3, :3].T @ [0., 0., 1.]
        np.testing.assert_allclose(result["camera_plane_rotation_vectors"][observed, 1],
                                   angles[observed, None] * axis_world, atol=1e-10, rtol=0)
        self.assertFalse(result["pose_valid"][1, 1])
        self.assertTrue(np.isnan(result["poses"][1, 1]).all())
        self.assertTrue(np.isnan(result["camera_plane_angles_unwrapped"][1, 1]))
        self.assertTrue(np.isnan(result["camera_plane_rotation_vectors"][1, 1]).all())
        np.testing.assert_array_equal(result["camera_plane_rotation_vectors"][:, 0], np.zeros((4, 3)))
        self.assertTrue(result["metadata"]["rotation_vector_unwrap_is_representation_prior_not_observation"])

    def test_camera_plane_does_not_invent_support_from_fewer_than_eight_tracks_or_one_time(self):
        centers, tracks, accepted, camera, _, _ = self.camera_plane_fixture()
        sparse = accepted.copy()
        sparse[:, 7:] = False
        for mask in (sparse, accepted & (np.arange(len(accepted))[:, None] == 0)):
            result = discover_rigid_groups(centers, tracks, mask, camera,
                                           motion_model="camera_plane", max_groups=1)
            self.assertEqual(result["diagnostics"]["dynamic_groups"], 0)
            self.assertTrue((result["labels"] == -1).all())

    def test_default_se3_matches_explicit_mode_and_plane_options_are_validated(self):
        centers, tracks, accepted, camera, static, _ = self.fixture(groups=1)
        implicit = discover_rigid_groups(centers, tracks, accepted, camera, static, max_groups=1)
        explicit = discover_rigid_groups(centers, tracks, accepted, camera, static,
                                         max_groups=1, motion_model="se3")
        np.testing.assert_array_equal(implicit["labels"], explicit["labels"])
        np.testing.assert_array_equal(implicit["pose_valid"], explicit["pose_valid"])
        np.testing.assert_allclose(implicit["poses"], explicit["poses"], atol=0, rtol=0)
        self.assertEqual(implicit["metadata"]["motion_model"], "se3")
        self.assertNotIn("camera_plane_angles_unwrapped", implicit)
        for options in ({"motion_model": "guess"}, {"plane_ransac_iterations": 0},
                        {"plane_ransac_iterations": 513}, {"plane_ransac_iterations": True}):
            with self.assertRaises(ValueError):
                discover_rigid_groups(centers, tracks, accepted, camera, **options)


if __name__ == "__main__":
    unittest.main()
