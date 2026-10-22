"""Synthetic CPU-only tests; no dataset, old motion, or renderer dependencies."""

import copy
import io
import math
import unittest

import numpy as np
import torch

from utils.event_only_motion import EventOnlyMotion, HARD_RIGID_GROUP_KIND, build_graph


def graph(centers, count, neighbors=None, weights=None, edges=None):
    centers = np.asarray(centers, dtype=np.float32)
    if neighbors is None:
        neighbors = np.zeros((count, 1), dtype=np.int32)
    neighbors = np.asarray(neighbors, dtype=np.int32)
    if weights is None:
        weights = np.ones(neighbors.shape, dtype=np.float32) / neighbors.shape[1]
    return {"centers": centers, "knn_indices": neighbors,
            "knn_weights": np.asarray(weights, dtype=np.float32),
            "edges": np.asarray(edges if edges is not None else [], dtype=np.int64).reshape(-1, 2),
            "scene_scale": 2.0}


def identity_quaternions(count):
    result = torch.zeros(count, 4)
    result[:, 0] = 1
    return result


def hard_graph():
    result = graph([[2, 0, 0], [0, 3, 0]], 4, neighbors=[[0], [0], [1], [1]])
    result["metadata"] = {
        "kind": HARD_RIGID_GROUP_KIND,
        "binding_source": "explicit_G0_raw_events_and_camera",
        "node_source": "G0_group_member_centers", "blender_parts": False,
        "input_sha256": {name: str(index) * 64 for index, name in
                         enumerate(("g0", "events", "intrinsics", "extrinsics"), 1)},
        "method": "synthetic_test_only",
    }
    return result


class EventOnlyMotionTests(unittest.TestCase):
    def test_hard_groups_are_exact_rigid_with_center_based_translation_and_quaternions(self):
        source = hard_graph()
        motion = EventOnlyMotion(source, [0, 1, 2], chunk_size=1)
        xyz = torch.tensor([[1., 0, 0], [3, 0, 0], [0, 2, 0], [0, 4, 0]])
        source_q = identity_quaternions(4)
        source_q[0] = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0, 0])
        matrices = torch.tensor([[[0., -1, 0], [1, 0, 0], [0, 0, 1]],
                                 [[1., 0, 0], [0, 0, -1], [0, 1, 0]]])
        global_translation = torch.tensor([[.5, 1, 0], [-1., 0, 2]])
        center_translation = (torch.einsum("kij,kj->ki", matrices, motion.centers) -
                              motion.centers + global_translation)
        with torch.no_grad():
            motion.translations[:] = center_translation
            motion.rotation_vectors[:, 0, 2] = math.pi / 2
            motion.rotation_vectors[:, 1, 0] = math.pi / 2
        labels = motion.knn_indices[:, 0]
        expected = torch.einsum("nij,nj->ni", matrices[labels], xyz) + global_translation[labels]
        posed, quaternion = motion.deform(xyz, source_q, 1)
        torch.testing.assert_close(posed, expected, atol=1e-6, rtol=1e-6)
        expected_q = torch.tensor([[.5, .5, .5, .5],
                                   [math.sqrt(.5), 0, 0, math.sqrt(.5)],
                                   [math.sqrt(.5), math.sqrt(.5), 0, 0],
                                   [math.sqrt(.5), math.sqrt(.5), 0, 0]])
        torch.testing.assert_close(quaternion, expected_q, atol=1e-6, rtol=1e-6)
        for time in (0, .25, .7, 1, 1.5, 2):
            moved, _ = motion.deform(xyz, source_q, time)
            for first, second in ((0, 1), (2, 3)):
                torch.testing.assert_close(torch.linalg.vector_norm(moved[first] - moved[second]),
                                           torch.tensor(2.), atol=1e-6, rtol=1e-6)
        arap = motion.spatial_arap(.7)
        self.assertTrue(arap.requires_grad)
        self.assertEqual(arap.item(), 0)
        arap.backward()
        for parameter in (motion.translations, motion.rotation_vectors):
            self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter)))

    def test_hard_group_metadata_records_event_binding_without_identifiability_claims(self):
        source = hard_graph()
        motion = EventOnlyMotion(source, [0, 1])
        self.assertEqual(motion.metadata["kind"], HARD_RIGID_GROUP_KIND)
        self.assertEqual(motion.metadata["binding_source"], "explicit_G0_raw_events_and_camera")
        self.assertEqual(motion.metadata["node_source"], "G0_group_member_centers")
        self.assertFalse(motion.metadata["physical_uniqueness_guaranteed"])
        self.assertFalse(motion.metadata["unseen_motion_identifiable"])
        self.assertEqual(motion.metadata["nodes"], 2)
        self.assertEqual(motion.metadata["source_gaussians"], 4)
        self.assertEqual(motion.metadata["method"], "synthetic_test_only")
        source["metadata"]["input_sha256"]["events"] = "9" * 64
        self.assertEqual(motion.metadata["input_sha256"]["events"], "2" * 64)

    def test_hard_group_contract_rejects_soft_binding_empty_groups_and_false_sources(self):
        cases = {
            "two_neighbors": lambda value: value.update(
                knn_indices=np.tile([[0, 1]], (4, 1)), knn_weights=np.full((4, 2), .5)),
            "near_one": lambda value: value.update(knn_weights=np.full((4, 1), 1. + 1e-9)),
            "cross_group_edge": lambda value: value.update(edges=np.array([[0, 1]], dtype=np.int64)),
            "empty_group": lambda value: value.update(knn_indices=np.zeros((4, 1), dtype=np.int64)),
            "g0_only_binding": lambda value: value["metadata"].update(binding_source="explicit_G0_positions_only"),
            "g0_only_node_claim": lambda value: value["metadata"].update(node_source="explicit_G0_positions_only"),
            "blender_parts": lambda value: value["metadata"].update(blender_parts=True),
            "physical_claim": lambda value: value["metadata"].update(physical_uniqueness_guaranteed=True),
            "hidden_motion_claim": lambda value: value["metadata"].update(unseen_motion_identifiable=True),
            "unknown_version": lambda value: value["metadata"].update(version=2),
            "boolean_version": lambda value: value["metadata"].update(version=True),
            "missing_hash": lambda value: value["metadata"]["input_sha256"].pop("events"),
            "bad_hash": lambda value: value["metadata"]["input_sha256"].update(events="not-a-hash"),
        }
        for name, mutation in cases.items():
            value = copy.deepcopy(hard_graph())
            mutation(value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                EventOnlyMotion(value, [0, 1])

    def test_legacy_metadata_and_state_schema_are_unchanged_by_opt_in_mode(self):
        value = graph([[0, 0, 0]], 2)
        motion = EventOnlyMotion(value, [0, 1, 2])
        self.assertEqual(motion.metadata, {
            "kind": "g0_only_latent_deformation_graph", "version": 1,
            "node_source": "explicit_G0_positions_only", "blender_parts": False,
            "physical_uniqueness_guaranteed": False, "unseen_motion_identifiable": False,
            "quaternion_order": "wxyz", "interpolation": "linear_axis_angle_and_translation",
            "anchor": "fixed_identity_at_time_zero", "nodes": 1, "source_gaussians": 2, "knots": 3,
        })
        self.assertEqual(set(motion.state_dict()),
                         set(EventOnlyMotion(hard_graph(), [0, 1, 2]).state_dict()))

    def test_graph_is_deterministic_bounded_spatial_and_row_preserving(self):
        random = np.random.default_rng(43)
        xyz = random.uniform(-1, 1, (2000, 3)).astype(np.float32)
        first = build_graph(xyz, nodes=32, neighbors=4, seed=7)
        second = build_graph(xyz, nodes=32, neighbors=4, seed=7)
        for key in ("centers", "knn_indices", "knn_weights", "edges"):
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first["knn_indices"].shape, (len(xyz), 4))
        np.testing.assert_allclose(first["knn_weights"].sum(axis=1), 1, atol=1e-6)
        self.assertTrue(np.all(first["knn_weights"] >= 0))
        self.assertLess(np.max(first["knn_indices"]), 32)
        source_rows = {tuple(row) for row in xyz}
        self.assertTrue(all(tuple(center) in source_rows for center in first["centers"]))
        # Nodes include both sides instead of a camera-dependent visible crop.
        self.assertLess(first["centers"][:, 2].min(), -.7)
        self.assertGreater(first["centers"][:, 2].max(), .7)
        self.assertFalse(first["metadata"]["blender_parts"])
        self.assertFalse(first["metadata"]["physical_uniqueness_guaranteed"])

    def test_degenerate_source_still_has_a_valid_single_node_graph(self):
        value = build_graph(np.ones((10, 3)), nodes=192, neighbors=4)
        self.assertEqual(value["centers"].shape, (1, 3))
        self.assertEqual(value["knn_indices"].shape, (10, 1))
        self.assertEqual(value["edges"].shape, (0, 2))
        np.testing.assert_array_equal(value["knn_weights"], np.ones((10, 1)))

    def test_identity_at_every_initial_knot_and_interpolated_time(self):
        xyz = torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1]])
        source_q = identity_quaternions(3)
        motion = EventOnlyMotion(build_graph(xyz.numpy(), nodes=3), [0., .4, 1.7], chunk_size=2)
        self.assertEqual(tuple(motion.rotation_vectors.shape), (2, 3, 3))
        self.assertEqual(tuple(motion.translations.shape), (2, 3, 3))
        for time in (0., .2, .4, .9, 1.7):
            posed, rotated = motion(xyz, source_q, time)
            torch.testing.assert_close(posed, xyz, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(rotated, source_q)
        self.assertEqual(motion.spatial_arap(.8).item(), 0)
        self.assertEqual(motion.temporal_smoothness().item(), 0)

    def test_first_time_is_exact_g0_even_after_parameters_change(self):
        xyz = torch.tensor([[1.1, -.3, 2.4], [0., 1, 0]])
        # Preserve the exact source representation at the anchor, even signs.
        source_q = torch.tensor([[-1., 0, 0, 0], [.9, .1, 0, 0]])
        motion = EventOnlyMotion(graph([[0, 0, 0]], 2), [0, 1, 2])
        with torch.no_grad():
            motion.translations.normal_()
            motion.rotation_vectors.normal_()
        posed, rotated = motion(xyz, source_q, 0)
        self.assertTrue(torch.equal(posed, xyz))
        self.assertTrue(torch.equal(rotated, source_q))
        (posed.sum() + rotated.sum()).backward()
        self.assertTrue(torch.equal(motion.translations.grad, torch.zeros_like(motion.translations)))
        self.assertTrue(motion.rotation_vectors.grad is None or not motion.rotation_vectors.grad.any())

    def test_single_node_rigid_rotation_and_translation(self):
        xyz = torch.tensor([[2., 0, 0], [1, 1, 0]])
        motion = EventOnlyMotion(graph([[1, 0, 0]], 2), [0, 2])
        with torch.no_grad():
            motion.rotation_vectors[0, 0, 2] = math.pi / 2
            motion.translations[0, 0] = torch.tensor([0., 0, 3])
        posed, quaternion = motion(xyz, identity_quaternions(2), 2)
        torch.testing.assert_close(posed, torch.tensor([[1., 1, 3], [0, 0, 3]]), atol=1e-6, rtol=1e-6)
        expected = torch.tensor([math.sqrt(.5), 0, 0, math.sqrt(.5)]).expand(2, -1)
        torch.testing.assert_close(quaternion, expected, atol=1e-6, rtol=1e-6)

    def test_time_interpolation_is_continuous_and_has_no_linear_component_overshoot(self):
        motion = EventOnlyMotion(graph([[0, 0, 0]], 1), [0, .3, 2])
        with torch.no_grad():
            motion.translations[0, 0, 0] = 1
            motion.translations[1, 0, 0] = 2
            motion.rotation_vectors[0, 0, 2] = .4
            motion.rotation_vectors[1, 0, 2] = .8
        before = motion.at_time(.3 - 1e-7)
        after = motion.at_time(.3 + 1e-7)
        for left, right in zip(before, after):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
        for time in np.linspace(.3, 2, 21):
            translation, _ = motion.at_time(float(time))
            self.assertGreaterEqual(translation[0, 0].item(), 1)
            self.assertLessEqual(translation[0, 0].item(), 2)

    def test_zero_initialization_has_finite_nonzero_trainable_knot_gradients(self):
        xyz = torch.tensor([[1., 0, 0], [0, 1, 0]])
        motion = EventOnlyMotion(graph([[0, 0, 0]], 2), [0, 1, 2])
        posed, quaternion = motion(xyz, identity_quaternions(2), .5)
        loss = posed[0, 1] + posed[1, 2] + quaternion[0, 3]
        loss.backward()
        for parameter in (motion.translations, motion.rotation_vectors):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad[0].abs().sum().item(), 0)
            self.assertEqual(parameter.grad[1].abs().sum().item(), 0)

    def test_nonconstant_trajectory_retains_differentiability_in_time(self):
        motion = EventOnlyMotion(graph([[0, 0, 0]], 1), [0, 1, 2])
        with torch.no_grad():
            motion.translations[0, 0, 0] = 2
        time = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
        translation, _ = motion.at_time(time)
        translation.sum().backward()
        self.assertAlmostEqual(time.grad.item(), 2)

    def test_antipodal_node_quaternions_blend_without_cancellation_or_nan(self):
        motion = EventOnlyMotion(graph([[0, 0, 0], [0, 0, 0]], 1,
                                       neighbors=[[0, 1]], weights=[[.5, .5]], edges=[[0, 1]]), [0, 1])
        with torch.no_grad():
            motion.rotation_vectors[0, 0, 2] = math.pi
            motion.rotation_vectors[0, 1, 2] = -math.pi
        xyz, quaternion = motion(torch.tensor([[1., 0, 0]]), identity_quaternions(1), 1)
        self.assertTrue(torch.isfinite(xyz).all())
        self.assertTrue(torch.isfinite(quaternion).all())
        torch.testing.assert_close(xyz, torch.tensor([[-1., 0, 0]]), atol=1e-6, rtol=1e-6)
        self.assertAlmostEqual(torch.linalg.vector_norm(quaternion).item(), 1, places=6)
        self.assertAlmostEqual(abs(quaternion[0, 3].item()), 1, places=6)

    def test_subset_rows_and_small_chunks_preserve_original_order(self):
        xyz = torch.tensor([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        motion = EventOnlyMotion(build_graph(xyz.numpy(), nodes=4), [0, 1], chunk_size=1)
        with torch.no_grad():
            motion.translations[0, :, 2] = torch.arange(4) * .1
        whole = motion(xyz, identity_quaternions(4), .5)
        rows = torch.tensor([3, 0, 2])
        subset = motion(xyz[rows], identity_quaternions(3), .5, indices=rows)
        torch.testing.assert_close(subset[0], whole[0][rows])
        torch.testing.assert_close(subset[1], whole[1][rows])

    def test_arap_is_zero_for_global_rigid_motion_and_positive_for_edge_stretch(self):
        centers = [[0, 0, 0], [1, 0, 0]]
        motion = EventOnlyMotion(graph(centers, 2, neighbors=[[0], [1]], edges=[[0, 1]]), [0, 1])
        with torch.no_grad():
            motion.rotation_vectors[0, :, 2] = math.pi / 2
            motion.translations[0] = torch.tensor([[0., 0, 0], [-1, 1, 0]])
        self.assertLess(motion.spatial_arap(1).item(), 1e-12)
        with torch.no_grad():
            motion.translations[0, 1, 0] += 1
        self.assertGreater(motion.spatial_arap(1).item(), 0)

    def test_temporal_smoothness_uses_nonuniform_knot_spacing(self):
        motion = EventOnlyMotion(graph([[0, 0, 0]], 1), [0, .25, 2])
        with torch.no_grad():
            motion.translations[:, 0, 0] = torch.tensor([.25, 2])
        self.assertEqual(motion.temporal_smoothness().item(), 0)
        with torch.no_grad():
            motion.translations[1, 0, 0] = 4
        loss = motion.temporal_smoothness()
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(motion.translations.grad).all())

    def test_state_dict_round_trip_preserves_motion_and_graph(self):
        source = graph([[0, 0, 0]], 2)
        first = EventOnlyMotion(source, [0, .4, 2])
        with torch.no_grad():
            first.translations.normal_()
            first.rotation_vectors.normal_()
        saved = io.BytesIO()
        torch.save(first.state_dict(), saved)
        saved.seek(0)
        second = EventOnlyMotion(source, [0, .4, 2])
        second.load_state_dict(torch.load(saved, map_location="cpu", weights_only=True))
        xyz, quaternion = torch.tensor([[1., 0, 0], [0, 1, 0]]), identity_quaternions(2)
        for left, right in zip(first(xyz, quaternion, .9), second(xyz, quaternion, .9)):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        self.assertEqual(first.metadata, second.metadata)

    def test_invalid_time_requests_and_knot_grids_fail_closed(self):
        source = graph([[0, 0, 0]], 1)
        motion = EventOnlyMotion(source, [0, 1, 2])
        for value in (-.001, 2.001, math.nan, math.inf, [1]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                motion.at_time(value)
        for times in ([1, 2], [0], [0, 0, 1], [0, 2, 1], [0, math.nan]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                EventOnlyMotion(source, times)


if __name__ == "__main__":
    unittest.main()
