"""Synthetic hard-binding checks; no input files, video, GPU, or old motion."""

import copy
import itertools
import json
import math
import unittest

import numpy as np

from utils.event_only_hard_binding import build_hard_binding
from utils.event_only_geodesic_binding import (ASSIGNMENT_PRIOR, HASH_POLICY,
                                              geodesic_binding_input_hashes)


HASHES = {name: str(index) * 64 for index, name in
          enumerate(("g0", "events", "intrinsics", "extrinsics"), 1)}


def pose_z(degrees, translation=(0, 0, 0)):
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    result = np.eye(4)
    result[:3, :3] = [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
    result[:3, 3] = translation
    return result


def example():
    offsets = np.array(list(itertools.product((-.1, .1), repeat=3)))
    seeds = np.concatenate((offsets + [-2, 0, 0], offsets + [2, 0, 0]))
    xyz = np.concatenate((seeds, [[2., 0, 0], [0, 0, 3]]))
    labels = np.repeat(np.array([0, 1], dtype=np.int32), 8)
    times = np.arange(5, dtype=np.float64)
    poses = np.tile(np.eye(4), (5, 2, 1, 1))
    valid = np.ones((5, 2), dtype=bool)
    valid[[2, 4], 1] = False
    poses[1, 1] = pose_z(60, (.3, 0, 0))
    poses[3, 1] = pose_z(100, (.6, .1, 0))
    poses[[2, 4], 1] = np.nan
    return dict(g0_xyz=xyz, seed_xyz=seeds, seed_labels=labels, pose_times=times,
                poses=poses, pose_valid=valid, input_sha256=dict(HASHES), max_binding_distance=.3)


def synthetic_assignment_prior(arguments):
    """A declared small prior fixture for validating the receiving contract.

    Graph construction itself is exercised in the separate geodesic suite;
    the cloud-only integration below also uses the real SciPy producer.
    """
    count = len(arguments["g0_xyz"])
    source = np.r_[np.zeros(8, dtype=np.int32), np.ones(9, dtype=np.int32), -1]
    assert count == len(source)
    path = np.full(count, .2)
    path[source < 0] = np.inf
    margin = np.where(source >= 0, 1., 0.)
    metadata = {
        "assignment_prior": ASSIGNMENT_PRIOR, "version": 1, "hash_policy": HASH_POLICY,
        **geodesic_binding_input_hashes(arguments["g0_xyz"], arguments["seed_xyz"], arguments["seed_labels"]),
        "binding_is_spatial_prior_not_parts": True, "automatic_radius_expansion": False,
        "input_G0_row_order_preserved": True, "physical_uniqueness_guaranteed": False,
        "unseen_motion_identifiable": False, "observations_added": 0,
        "cell_size": .125, "max_euclidean_distance": arguments["max_binding_distance"],
        "max_path_length": 1., "max_edge_length": .2, "effective_max_edge_length": .2,
        "max_seed_attachment_distance": .2, "max_nodes": 1000, "neighbors": 8,
        "node_cap_policy": "reject_no_automatic_coarsening",
        "path_cost_includes_seed_and_G0_attachments": True, "min_relative_margin": .15,
    }
    return {"source_labels": source, "nearest_geodesic": path, "margin": margin, "metadata": metadata}


class EventOnlyHardBindingTests(unittest.TestCase):
    def test_optional_prior_none_preserves_default_nearest_behavior_exactly(self):
        arguments = example()
        first = build_hard_binding(**arguments)
        second = build_hard_binding(**arguments, assignment_prior=None)
        for key in ("rotation_vectors", "translations", "initial_world_poses", "pose_observed", "gaussian_group_ids"):
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first["graph"]["metadata"], second["graph"]["metadata"])

    def test_assignment_prior_unknown_stays_unknown_and_does_not_create_observations(self):
        arguments = example()
        prior = synthetic_assignment_prior(arguments)
        prior["source_labels"][8] = -1
        prior["margin"][8] = 0
        before = copy.deepcopy(prior)
        baseline = build_hard_binding(**arguments)
        result = build_hard_binding(**arguments, assignment_prior=prior)
        source = result["group_source_labels"][result["gaussian_group_ids"]]
        np.testing.assert_array_equal(source, prior["source_labels"])
        np.testing.assert_array_equal(result["pose_observed"], baseline["pose_observed"])
        self.assertEqual(result["diagnostics"]["unknown_gaussians"], 2)
        self.assertFalse(result["graph"]["metadata"]["assignment_prior_is_observation"])
        self.assertTrue(result["graph"]["metadata"]["assignment_prior_input_hashes_verified"])
        self.assertEqual(result["graph"]["metadata"]["assignment_prior"], ASSIGNMENT_PRIOR)
        for key in ("source_labels", "nearest_geodesic", "margin"):
            np.testing.assert_array_equal(prior[key], before[key])
        self.assertEqual(prior["metadata"], before["metadata"])
        json.dumps(result["graph"]["metadata"], allow_nan=False)
        for time in (1, 3):
            np.testing.assert_allclose(result["initial_world_poses"][time, 1], arguments["poses"][time, 1], atol=1e-8)

    def test_geometric_prior_may_select_non_nearest_supported_group_within_budget(self):
        arguments = example()
        arguments["g0_xyz"][:8, 0] += 1.8
        arguments["g0_xyz"][8:17, 0] -= 1.8
        arguments["seed_xyz"][:8, 0] += 1.8
        arguments["seed_xyz"][8:, 0] -= 1.8
        arguments["max_binding_distance"] = .6
        prior = synthetic_assignment_prior(arguments)
        prior["source_labels"][0] = 1
        prior["nearest_geodesic"][0] = .5
        result = build_hard_binding(**arguments, assignment_prior=prior)
        self.assertEqual(result["group_source_labels"][result["gaussian_group_ids"][0]], 1)

    def test_assignment_prior_rejects_input_hash_kind_and_observation_claims(self):
        changes = {
            "wrong_G0_hash": lambda value: value["metadata"].update(g0_xyz_sha256="f" * 64),
            "wrong_seed_hash": lambda value: value["metadata"].update(trusted_seed_sha256="f" * 64),
            "wrong_kind": lambda value: value["metadata"].update(assignment_prior="old_semantic_parts"),
            "observations": lambda value: value["metadata"].update(observations_added=1),
            "false_physical_claim": lambda value: value["metadata"].update(physical_uniqueness_guaranteed=True),
            "adaptive_radius": lambda value: value["metadata"].update(automatic_radius_expansion=True),
            "non_json": lambda value: value["metadata"].update(hidden_array=np.ones(1)),
            "nonfinite_json": lambda value: value["metadata"].update(max_path_length=math.inf),
        }
        for name, mutate in changes.items():
            arguments = example()
            prior = synthetic_assignment_prior(arguments)
            mutate(prior)
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_hard_binding(**arguments, assignment_prior=prior)

    def test_assignment_prior_rejects_unsupported_group_and_wrong_row_binding(self):
        arguments = example()
        prior = synthetic_assignment_prior(arguments)
        prior["source_labels"][0] = 2
        with self.assertRaisesRegex(ValueError, "unsupported"):
            build_hard_binding(**arguments, assignment_prior=prior)
        prior = synthetic_assignment_prior(arguments)
        arguments["pose_valid"][3, 1] = False
        with self.assertRaisesRegex(ValueError, "unsupported"):
            build_hard_binding(**arguments, assignment_prior=prior)
        arguments = example()
        prior = synthetic_assignment_prior(arguments)
        prior["source_labels"] = prior["source_labels"][:-1]
        with self.assertRaisesRegex(ValueError, "row order/length"):
            build_hard_binding(**arguments, assignment_prior=prior)

    def test_assignment_prior_requires_finite_in_budget_known_paths_and_valid_margin(self):
        changes = {
            "infinite_known": lambda value: value["nearest_geodesic"].__setitem__(0, np.inf),
            "nan_unknown": lambda value: value["nearest_geodesic"].__setitem__(-1, np.nan),
            "negative_path": lambda value: value["nearest_geodesic"].__setitem__(0, -.1),
            "over_budget": lambda value: value["nearest_geodesic"].__setitem__(0, 2.),
            "below_margin": lambda value: value["margin"].__setitem__(0, .1),
            "nan_margin": lambda value: value["margin"].__setitem__(0, np.nan),
            "unknown_infinite_path_nonzero_margin": lambda value: value["margin"].__setitem__(-1, .5),
            "wrong_edge_budget": lambda value: value["metadata"].update(effective_max_edge_length=.3),
        }
        for name, mutate in changes.items():
            arguments = example()
            prior = synthetic_assignment_prior(arguments)
            mutate(prior)
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_hard_binding(**arguments, assignment_prior=prior)

    def test_assignment_prior_checks_selected_group_distance_not_global_nearest(self):
        arguments = example()
        prior = synthetic_assignment_prior(arguments)
        prior["source_labels"][0] = 1
        with self.assertRaisesRegex(ValueError, "selected-group distance"):
            build_hard_binding(**arguments, assignment_prior=prior)
        prior = synthetic_assignment_prior(arguments)
        prior["nearest_geodesic"][16] = .01
        with self.assertRaisesRegex(ValueError, "Euclidean lower bound"):
            build_hard_binding(**arguments, assignment_prior=prior)

    def test_assignment_prior_respects_explicit_unknown_reject_policy(self):
        arguments = example()
        prior = synthetic_assignment_prior(arguments)
        with self.assertRaisesRegex(ValueError, "bounded geometric assignment prior"):
            build_hard_binding(**arguments, assignment_prior=prior, unknown_policy="reject")

    def test_real_scipy_geodesic_output_is_accepted_without_adding_observations(self):
        from utils import event_only_geodesic_binding as geodesic
        try:
            geodesic._scipy_backend()
        except RuntimeError as error:
            self.skipTest(str(error))
        arguments = example()
        prior = geodesic.build_geodesic_binding(
            arguments["g0_xyz"], arguments["seed_xyz"], arguments["seed_labels"],
            cell_size=.125, max_euclidean_distance=.3, max_path_length=1.,
            max_edge_length=.2, max_nodes=1000)
        result = build_hard_binding(**arguments, assignment_prior=prior)
        np.testing.assert_array_equal(result["group_source_labels"][result["gaussian_group_ids"]], prior["source_labels"])
        self.assertFalse(result["pose_observed"][0].any())
        self.assertFalse(result["pose_observed"][:, 0].any())

    def test_graph_schema_full_row_binding_centers_metadata_and_seed_mapping(self):
        arguments = example()
        result = build_hard_binding(**arguments)
        graph = result["graph"]
        self.assertEqual(set(graph), {"centers", "knn_indices", "knn_weights", "edges", "scene_scale", "metadata"})
        self.assertEqual(graph["knn_indices"].shape, (18, 1))
        self.assertEqual(graph["knn_weights"].shape, (18, 1))
        self.assertEqual(graph["edges"].shape, (0, 2))
        np.testing.assert_array_equal(graph["knn_weights"], np.ones((18, 1)))
        np.testing.assert_array_equal(result["group_source_labels"], [0, 1, -1])
        np.testing.assert_array_equal(result["supervision_seed_groups"], arguments["seed_labels"])
        np.testing.assert_array_equal(result["recommended_frozen_groups"], [True, False, True])
        for group in range(3):
            rows = result["gaussian_group_ids"] == group
            np.testing.assert_allclose(graph["centers"][group], arguments["g0_xyz"][rows].mean(axis=0), atol=1e-7)
        self.assertEqual(graph["metadata"]["input_sha256"], HASHES)
        self.assertEqual(graph["metadata"]["binding_source"], "explicit_G0_raw_events_and_camera")
        self.assertTrue(graph["metadata"]["binding_is_spatial_prior_not_parts"])
        self.assertFalse(graph["metadata"]["unknown_is_observed_static"])
        self.assertFalse(graph["metadata"]["physical_uniqueness_guaranteed"])
        self.assertEqual(result["diagnostics"]["unknown_gaussians"], 1)
        self.assertEqual(result["diagnostics"]["static_prior_gaussians"], 8)
        self.assertEqual(result["diagnostics"]["retained_dynamic_group_count"], 1)
        self.assertEqual(result["diagnostics"]["output_groups"][1]["member_count"], 9)
        self.assertLessEqual(result["diagnostics"]["output_groups"][1]["nearest_seed_distance_quantiles"]["p100"], .3)
        json.dumps(graph["metadata"], allow_nan=False)

    def test_valid_world_poses_convert_to_center_based_translations_without_stretching(self):
        arguments = example()
        result = build_hard_binding(**arguments)
        center = result["graph"]["centers"][1]
        points = arguments["g0_xyz"][result["gaussian_group_ids"] == 1]
        for index in (1, 3):
            pose = arguments["poses"][index, 1]
            expected_translation = pose[:3, 3] + (pose[:3, :3] - np.eye(3)) @ center
            np.testing.assert_allclose(result["translations"][index - 1, 1], expected_translation, atol=2e-7)
            np.testing.assert_allclose(result["initial_world_poses"][index, 1], pose, atol=1e-8)
        for pose in result["initial_world_poses"][:, 1]:
            moved = points @ pose[:3, :3].T + pose[:3, 3]
            np.testing.assert_allclose(np.linalg.norm(moved[:, None] - moved[None, :], axis=-1),
                                       np.linalg.norm(points[:, None] - points[None, :], axis=-1), atol=1e-10)

    def test_missing_times_and_t0_never_become_observations(self):
        result = build_hard_binding(**example())
        np.testing.assert_array_equal(result["pose_observed"][:, 1], [False, True, False, True, False])
        self.assertFalse(result["pose_observed"][:, 0].any())
        self.assertFalse(result["pose_observed"][:, 2].any())
        self.assertTrue(result["graph"]["metadata"]["static_prior_pose_validity"])
        self.assertFalse(result["graph"]["metadata"]["t0_identity_is_observation"])
        np.testing.assert_allclose(result["rotation_vectors"][1, 1],
                                   (result["rotation_vectors"][0, 1] + result["rotation_vectors"][2, 1]) / 2)
        np.testing.assert_allclose(result["translations"][1, 1],
                                   (result["translations"][0, 1] + result["translations"][2, 1]) / 2)
        np.testing.assert_array_equal(result["rotation_vectors"][3, 1], result["rotation_vectors"][2, 1])
        np.testing.assert_array_equal(result["translations"][3, 1], result["translations"][2, 1])
        details = result["diagnostics"]["output_groups"][1]
        self.assertEqual(details["interpolated_initialization_indices"], [2])
        self.assertEqual(details["held_tail_initialization_indices"], [4])
        for key in ("initial_world_poses", "rotation_vectors", "translations"):
            self.assertTrue(np.isfinite(result[key]).all())

    def test_cross_pi_and_full_turn_choose_continuous_equivalent_axis_angle_branch(self):
        arguments = example()
        angles = [0, 170, None, 190, 350, 360, 370]
        arguments["pose_times"] = np.arange(len(angles), dtype=float)
        arguments["poses"] = np.tile(np.eye(4), (len(angles), 2, 1, 1))
        arguments["pose_valid"] = np.ones((len(angles), 2), dtype=bool)
        for index, angle in enumerate(angles):
            if angle is None:
                arguments["poses"][index, 1] = np.nan
                arguments["pose_valid"][index, 1] = False
            else:
                arguments["poses"][index, 1] = pose_z(angle)
        result = build_hard_binding(**arguments)
        expected = np.deg2rad([170, 180, 190, 350, 360, 370])
        np.testing.assert_allclose(result["rotation_vectors"][:, 1, 2], expected, atol=4e-7)
        np.testing.assert_allclose(result["rotation_vectors"][:, 1, :2], 0, atol=1e-8)
        for index, angle in enumerate(angles):
            if angle is not None:
                np.testing.assert_allclose(result["initial_world_poses"][index, 1, :3, :3],
                                           arguments["poses"][index, 1, :3, :3], atol=1e-8)

    def test_propagation_radius_has_boundary_and_far_body_stays_unknown(self):
        arguments = example()
        arguments["g0_xyz"] = np.concatenate((arguments["g0_xyz"], [[2.4, .1, .1], [2.40001, .1, .1]]))
        result = build_hard_binding(**arguments)
        source_labels = result["group_source_labels"][result["gaussian_group_ids"]]
        self.assertEqual(source_labels[-2], 1)
        self.assertEqual(source_labels[-1], -1)
        self.assertEqual(source_labels[17], -1)
        unknown_node = int(np.flatnonzero(result["group_source_labels"] == -1)[0])
        np.testing.assert_array_equal(result["translations"][:, unknown_node], 0)
        np.testing.assert_array_equal(result["rotation_vectors"][:, unknown_node], 0)

    def test_equally_near_competing_groups_remain_unknown(self):
        arguments = example()
        arguments["seed_xyz"][:8, 0] += 1.5
        arguments["seed_xyz"][8:, 0] -= 1.5
        arguments["g0_xyz"] = np.concatenate((arguments["seed_xyz"], [[0, .1, .1]]))
        arguments["max_binding_distance"] = .5
        result = build_hard_binding(**arguments)
        source_labels = result["group_source_labels"][result["gaussian_group_ids"]]
        self.assertEqual(source_labels[-1], -1)
        self.assertEqual(result["diagnostics"]["ambiguous_gaussians"], 1)

    def test_insufficient_dynamic_pose_support_is_unknown_not_static(self):
        arguments = example()
        arguments["pose_valid"][3, 1] = False
        result = build_hard_binding(**arguments)
        np.testing.assert_array_equal(result["group_source_labels"], [0, -1])
        self.assertEqual(result["diagnostics"]["unknown_gaussians"], 10)
        self.assertEqual(result["diagnostics"]["source_groups"][1]["rejection_reason"],
                         "insufficient_non_t0_observed_poses")
        np.testing.assert_array_equal(result["supervision_seed_groups"][8:], -1)

    def test_static_support_does_not_require_pose_valid_but_is_never_observed(self):
        arguments = example()
        arguments["pose_valid"][:, 0] = False
        arguments["poses"][:, 0] = np.nan
        result = build_hard_binding(**arguments)
        self.assertEqual(result["group_source_labels"][0], 0)
        self.assertFalse(result["pose_observed"][:, 0].any())
        np.testing.assert_array_equal(result["initial_world_poses"][:, 0], np.tile(np.eye(4), (5, 1, 1)))

    def test_duplicate_or_low_confidence_seeds_do_not_inflate_support(self):
        for kind in ("duplicate", "low_confidence"):
            arguments = example()
            if kind == "duplicate":
                arguments["seed_xyz"][8:] = arguments["seed_xyz"][8]
            else:
                arguments["seed_confidence"] = np.ones(16)
                arguments["seed_confidence"][-1] = .1
                arguments["min_seed_confidence"] = .5
            result = build_hard_binding(**arguments)
            with self.subTest(kind=kind):
                np.testing.assert_array_equal(result["group_source_labels"], [0, -1])
                self.assertEqual(result["diagnostics"]["source_groups"][1]["rejection_reason"],
                                 "insufficient_distinct_confident_seeds")

    def test_supported_group_with_no_G0_members_is_not_created(self):
        arguments = example()
        arguments["g0_xyz"] = arguments["g0_xyz"][:8]
        result = build_hard_binding(**arguments)
        np.testing.assert_array_equal(result["group_source_labels"], [0])
        self.assertEqual(result["diagnostics"]["source_groups"][1]["rejection_reason"], "no_bound_G0_members")
        np.testing.assert_array_equal(result["supervision_seed_groups"][8:], -1)

    def test_no_trusted_seeds_produces_only_explicit_unknown_identity_prior(self):
        arguments = example()
        arguments.update(seed_xyz=np.empty((0, 3)), seed_labels=np.empty(0, dtype=np.int32),
                         poses=np.empty((5, 0, 4, 4)), pose_valid=np.empty((5, 0), dtype=bool))
        result = build_hard_binding(**arguments)
        np.testing.assert_array_equal(result["group_source_labels"], [-1])
        self.assertEqual(result["diagnostics"]["retained_dynamic_group_count"], 0)
        self.assertEqual(result["diagnostics"]["unknown_gaussians"], 18)
        self.assertEqual(result["diagnostics"]["static_prior_gaussians"], 0)
        self.assertFalse(result["pose_observed"].any())
        self.assertIsNone(result["diagnostics"]["output_groups"][0]["nearest_seed_distance_quantiles"])
        json.dumps(result["graph"]["metadata"], allow_nan=False)

    def test_unknown_reject_policy_and_explicit_finite_radius(self):
        arguments = example()
        with self.assertRaisesRegex(ValueError, "lack an unambiguous"):
            build_hard_binding(**{**arguments, "unknown_policy": "reject"})
        for value in (0, -1, math.inf, math.nan, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_hard_binding(**{**arguments, "max_binding_distance": value})
        arguments.pop("max_binding_distance")
        with self.assertRaises(TypeError):
            build_hard_binding(**arguments)

    def test_invalid_measurement_claims_camera_like_transforms_or_input_schema_fail(self):
        cases = {
            "float_validity": lambda value: value.update(pose_valid=value["pose_valid"].astype(float)),
            "nonidentity_t0": lambda value: value["poses"][0, 1].__setitem__((0, 3), .2),
            "moving_static_prior": lambda value: value["poses"][2, 0].__setitem__((0, 3), .2),
            "invalid_rotation": lambda value: value["poses"][1, 1].__setitem__((0, 0), 5),
            "nonfinite_valid_pose": lambda value: value["poses"][1, 1].__setitem__((0, 3), np.nan),
            "missing_t0": lambda value: value.update(pose_times=np.arange(1, 6)),
            "unordered_times": lambda value: value.update(pose_times=np.array([0, 1, 1, 3, 4])),
            "wrong_seed_label": lambda value: value["seed_labels"].__setitem__(0, 2),
            "bad_hash": lambda value: value["input_sha256"].update(events="bad"),
            "confidence_nan": lambda value: value.update(seed_confidence=np.full(16, np.nan)),
            "infinite_g0": lambda value: value["g0_xyz"].__setitem__((0, 0), np.inf),
        }
        for name, mutation in cases.items():
            arguments = example()
            mutation(arguments)
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_hard_binding(**arguments)

    def test_invalid_nan_t0_is_anchored_without_changing_observed_flags(self):
        arguments = example()
        arguments["pose_valid"][0] = False
        arguments["poses"][0] = np.nan
        result = build_hard_binding(**arguments)
        self.assertFalse(result["pose_observed"][0].any())
        np.testing.assert_array_equal(result["initial_world_poses"][0], np.tile(np.eye(4), (3, 1, 1)))

    def test_inputs_not_mutated_and_output_parameters_are_finite_float32(self):
        arguments = example()
        original = copy.deepcopy(arguments)
        result = build_hard_binding(**arguments)
        for key, value in arguments.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, original[key])
            else:
                self.assertEqual(value, original[key])
        for key in ("rotation_vectors", "translations"):
            self.assertEqual(result[key].dtype, np.float32)
            self.assertEqual(result[key].shape, (4, 3, 3))
            self.assertTrue(np.isfinite(result[key]).all())
        self.assertEqual(result["times"].dtype, np.float64)

    def test_EventOnlyMotion_parameter_copy_preserves_rigid_geometry_and_orientation(self):
        try:
            import torch
            from utils.event_only_motion import EventOnlyMotion
        except (ImportError, OSError) as error:
            self.skipTest(str(error))
        arguments = example()
        result = build_hard_binding(**arguments)
        graph = dict(result["graph"])
        # Lists avoid depending on a local torch/numpy binary bridge. The
        # canonical builder output remains NumPy for the real cloud runtime.
        for key in ("centers", "knn_indices", "knn_weights"):
            graph[key] = torch.tensor(graph[key].tolist(), dtype=torch.int64 if key == "knn_indices" else torch.float32)
        graph["edges"] = torch.empty((0, 2), dtype=torch.int64)
        motion = EventOnlyMotion(graph, result["times"].tolist())
        with torch.no_grad():
            motion.rotation_vectors.copy_(torch.tensor(result["rotation_vectors"].tolist()))
            motion.translations.copy_(torch.tensor(result["translations"].tolist()))
        xyz = torch.tensor(arguments["g0_xyz"].tolist(), dtype=torch.float32)
        quaternions = torch.tensor([[1., 0, 0, 0]]).repeat(len(xyz), 1)
        for time_index in (0, 1, 2, 3, 4):
            moved, rotated = motion.deform(xyz, quaternions, time_index)
            for group in range(3):
                members = result["gaussian_group_ids"] == group
                pose = result["initial_world_poses"][time_index, group]
                expected = arguments["g0_xyz"][members] @ pose[:3, :3].T + pose[:3, 3]
                np.testing.assert_allclose(np.array(moved.detach().tolist())[members], expected, atol=1e-6)
            if time_index == 1:
                expected_q = [math.cos(math.pi / 6), 0, 0, math.sin(math.pi / 6)]
                np.testing.assert_allclose(rotated[8].detach().tolist(), expected_q, atol=1e-6)
        self.assertEqual(motion.metadata["kind"], "event_inferred_hard_rigid_groups_v1")
        self.assertEqual(motion.metadata["input_sha256"], HASHES)


if __name__ == "__main__":
    unittest.main()
