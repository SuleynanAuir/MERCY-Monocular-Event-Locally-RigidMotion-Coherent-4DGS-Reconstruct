"""Synthetic CPU tests; no files, camera projections, real data, or GPU."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from utils.event_only_graph_propagation import propagate_graph_motion


class EventOnlyGraphPropagationTests(unittest.TestCase):
    def setUp(self):
        self.centers = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
        self.visible = self.centers[:, 2] == -1
        self.accepted = self.visible[None].copy()

    def test_pure_translation_recovers_supported_hidden_nodes(self):
        motion = np.zeros((1, 8, 3))
        motion[0, self.visible] = [.2, -.3, .4]
        result = propagate_graph_motion(self.centers, motion, self.visible, self.accepted)
        np.testing.assert_allclose(result["translations"][0], np.tile([.2, -.3, .4], (8, 1)), atol=1e-6)
        np.testing.assert_allclose(result["rotation_vectors"], 0, atol=1e-6)
        self.assertEqual(result["debug"]["knots"][0]["supported_hidden_nodes"], 4)
        self.assertFalse(result["metadata"]["hidden_motion_validated"])

    def test_local_rigid_rotation_recovers_hidden_centers_from_visible_anchors(self):
        vector = np.array([.35, -.2, .25])
        rotation = Rotation.from_rotvec(vector).as_matrix()
        expected = self.centers @ rotation.T + [.2, -.1, .3] - self.centers
        motion = np.zeros((1, 8, 3))
        motion[0, self.visible] = expected[self.visible]
        result = propagate_graph_motion(self.centers, motion, self.visible, self.accepted)
        np.testing.assert_allclose(result["translations"][0], expected, atol=1e-5, rtol=0)
        np.testing.assert_allclose(result["rotation_vectors"][0], np.tile(vector, (8, 1)), atol=1e-5, rtol=0)

    def test_visible_unaccepted_static_anchor_does_not_drift(self):
        visible = self.visible.copy()
        motion = np.zeros((1, 8, 3))
        moving = np.flatnonzero(visible)[1:]
        motion[0, moving, 0] = .3
        accepted = np.zeros((1, 8), bool)
        accepted[0, moving] = True
        static = np.flatnonzero(visible)[0]
        # Untrusted values must not override a static visible anchor.
        motion[0, static] = [90, 90, 90]
        result = propagate_graph_motion(self.centers, motion, visible, accepted)
        np.testing.assert_array_equal(result["translations"][0, static], [0, 0, 0])
        np.testing.assert_array_equal(result["rotation_vectors"][0, static], [0, 0, 0])
        np.testing.assert_allclose(result["translations"][0, moving, 0], .3, atol=1e-7)
        self.assertEqual(result["debug"]["knots"][0]["visible_anchor_max_error"], 0)

    def test_pause_holds_previous_visible_anchors_and_hidden_motion(self):
        motion = np.zeros((2, 8, 3))
        motion[0, self.visible] = [.2, 0, .1]
        motion[1] = 100
        accepted = np.zeros((2, 8), bool)
        accepted[0] = self.visible
        result = propagate_graph_motion(self.centers, motion, self.visible, accepted)
        np.testing.assert_allclose(result["translations"][1], result["translations"][0], atol=1e-7)

    def test_no_trusted_support_means_zero_even_if_hidden_flow_claims_motion(self):
        motion = np.ones((1, 8, 3))
        accepted = (~self.visible)[None]
        result = propagate_graph_motion(self.centers, motion, self.visible, accepted)
        np.testing.assert_array_equal(result["translations"], np.zeros_like(motion))
        np.testing.assert_array_equal(result["rotation_vectors"], np.zeros_like(motion))
        self.assertEqual(result["debug"]["knots"][0]["hidden_accepted_masks_ignored"], 4)

    def test_hop_and_3d_radius_bounds_prevent_remote_propagation(self):
        centers = np.column_stack((np.arange(12), np.zeros((12, 2))))
        edges = np.column_stack((np.arange(11), np.arange(1, 12)))
        visible = np.zeros(12, bool)
        visible[0] = True
        motion = np.zeros((1, 12, 3))
        motion[0, 0, 1] = 1
        result = propagate_graph_motion(centers, motion, visible, visible[None], edges=edges,
                                        max_hops=2, radius_multiplier=3.)
        self.assertGreater(result["translations"][0, 1, 1], 0)
        np.testing.assert_array_equal(result["translations"][0, 3:], np.zeros((9, 3)))
        result = propagate_graph_motion(centers, motion, visible, visible[None], edges=edges,
                                        max_hops=8, radius_multiplier=.4)
        np.testing.assert_array_equal(result["translations"][0, 1:], np.zeros((11, 3)))

    def test_long_bridge_does_not_transfer_foreground_motion_to_remote_back_group(self):
        centers = np.concatenate((self.centers * .2, self.centers * .2 + [5, 0, 0]))
        edges = np.array([(i, j) for group in (range(8), range(8, 16)) for i in group for j in group if i < j] + [(7, 8)])
        visible = np.zeros(16, bool)
        visible[:8] = self.visible
        motion = np.zeros((1, 16, 3))
        motion[0, visible, 1] = .2
        result = propagate_graph_motion(centers, motion, visible, visible[None], edges=edges)
        np.testing.assert_array_equal(result["translations"][0, 8:], np.zeros((8, 3)))
        self.assertGreaterEqual(result["debug"]["rejected_long_or_zero_edges"], 1)

    def test_determinism_and_validation(self):
        motion = np.zeros((1, 8, 3))
        motion[0, self.visible, 0] = .1
        first = propagate_graph_motion(self.centers, motion, self.visible, self.accepted)
        second = propagate_graph_motion(self.centers, motion, self.visible, self.accepted)
        np.testing.assert_array_equal(first["translations"], second["translations"])
        for kwargs in ({"max_hops": 0}, {"radius_multiplier": float("nan")},
                       {"edges": np.array([[0, 99]])}, {"arap_iterations": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                propagate_graph_motion(self.centers, motion, self.visible, self.accepted, **kwargs)

    def test_explicit_static_mask_allows_dynamic_visible_low_texture_nodes_to_propagate(self):
        visible = np.ones(8, bool)
        accepted = self.visible[None]
        motion = np.zeros((1, 8, 3))
        motion[0, self.visible] = [.2, -.1, .3]
        result = propagate_graph_motion(self.centers, motion, visible, accepted,
                                        static_anchor_mask_np=np.zeros(8, bool))
        np.testing.assert_allclose(result["translations"][0], np.tile([.2, -.1, .3], (8, 1)), atol=1e-6)
        self.assertEqual(result["debug"]["knots"][0]["supported_unreliable_visible_nodes"], 4)
        conservative = propagate_graph_motion(self.centers, motion, visible, accepted)
        np.testing.assert_array_equal(conservative["translations"][0, ~self.visible], np.zeros((4, 3)))

    def test_explicit_static_anchor_is_exact_even_when_flow_wrongly_accepts_it(self):
        visible = np.ones(8, bool)
        static = np.zeros(8, bool)
        static[0] = True
        accepted = np.zeros((1, 8), bool)
        accepted[0, :4] = True
        motion = np.zeros((1, 8, 3))
        motion[0, :4] = [.1, 0, 0]
        motion[0, 0] = [99, 99, 99]
        result = propagate_graph_motion(self.centers, motion, visible, accepted,
                                        static_anchor_mask_np=static)
        np.testing.assert_array_equal(result["translations"][0, 0], [0, 0, 0])
        np.testing.assert_array_equal(result["rotation_vectors"][0, 0], [0, 0, 0])
        self.assertEqual(result["debug"]["knots"][0]["static_accepted_masks_ignored"], 1)
        self.assertEqual(result["debug"]["knots"][0]["static_anchor_max_error"], 0)

    def test_explicit_mode_retains_previous_observation_when_current_flow_is_unreliable(self):
        visible = np.ones(8, bool)
        motion = np.zeros((2, 8, 3))
        motion[0, self.visible] = [.2, -.1, .3]
        motion[1] = 99
        accepted = np.zeros((2, 8), bool)
        accepted[0] = self.visible
        result = propagate_graph_motion(self.centers, motion, visible, accepted,
                                        static_anchor_mask_np=np.zeros(8, bool))
        np.testing.assert_allclose(result["translations"][1], result["translations"][0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
