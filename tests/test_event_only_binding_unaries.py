"""Synthetic arrays only: no file, dataset, renderer, SciPy, or GPU use."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from utils.event_only_binding_unaries import score_binding_event_unaries


class BindingUnaryTests(unittest.TestCase):
    def scene(self, count=7, dark=False, directions=None):
        height, width = 48, 80
        background = .04
        random = np.random.default_rng(31)
        texture = random.uniform(.008, .025, (28, 24)) if dark else random.uniform(.22, .75, (28, 24))
        reference = np.full((height, width), background)
        reference[10:38, 15:39] = texture
        alpha = np.zeros_like(reference)
        alpha[10:38, 15:39] = 1
        depth = np.where(alpha > 0, 2., 0.)
        camera = {"width": width, "height": height, "fx": 40., "fy": 40., "cx": 40., "cy": 24.,
                  "pixel_center": "boundary", "w2c": np.eye(4)}
        times = np.arange(count, dtype=float)
        shifts = np.rint(np.linspace(0, 12, count)).astype(int)
        if directions is not None:
            shifts *= np.asarray(directions)
        logs, poses = [], np.broadcast_to(np.eye(4), (count, 2, 4, 4)).copy()
        for index, shift in enumerate(shifts):
            current = np.full_like(reference, background)
            current[10:38, 15 + shift:39 + shift] = texture
            logs.append(np.log(current) - np.log(reference))
            poses[index, 1, 0, 3] = shift * 2 / 40
        valid = np.ones((count, 2), dtype=bool)
        valid[0] = False
        return [np.log(reference), alpha, depth, logs, camera, times, poses, valid]

    def evaluate(self, arguments=None, **options):
        return score_binding_event_unaries(*(self.scene() if arguments is None else arguments),
                                           background=.04, **options)

    def test_translation_wins_with_same_absolute_reference_and_streaming_input(self):
        arguments = self.scene()
        arguments[3] = (frame for frame in arguments[3])
        result = self.evaluate(arguments)
        self.assertGreater(len(result["anchor_points"]), 200)
        self.assertGreater((result["winner"] == 1).mean(), .95)
        self.assertFalse((result["winner"] == 0).any())
        np.testing.assert_allclose(result["cost"][:, 1], 0, atol=1e-20)
        np.testing.assert_array_equal(result["common_time_indices"], np.arange(1, 7))
        self.assertGreater(np.median(result["counterfactual_no_event_cost"][:, 1]), .4)

    def test_departed_bright_original_is_not_rescued_by_patch_mean_subtraction(self):
        result = self.evaluate()
        left = result["anchor_pixels"][:, 0] < 19
        self.assertTrue(left.any())
        self.assertGreater(np.median(result["cost"][left, 0]), 1)
        self.assertTrue((result["winner"][left] == 1).all())
        self.assertIn("no_patch_mean_subtraction", result["metadata"]["loss"])

    def test_dark_material_distinct_from_background_can_supply_evidence(self):
        result = self.evaluate(self.scene(dark=True))
        self.assertGreater(len(result["anchor_points"]), 200)
        self.assertGreater((result["winner"] == 1).mean(), .9)
        self.assertFalse((result["winner"] == 0).any())

    def test_background_like_low_texture_has_no_invented_anchor(self):
        arguments = self.scene()
        arguments[0][:] = np.log(.04)
        arguments[3] = [np.zeros_like(arguments[0]) for _ in arguments[5]]
        result = self.evaluate(arguments)
        self.assertEqual(result["anchor_points"].shape, (0, 3))
        self.assertGreater(result["diagnostics"]["grid_rejected_background_like_low_texture"], 0)
        self.assertEqual(result["diagnostics"]["unknown_anchor_count"], 0)
        json.dumps(result["metadata"], allow_nan=False)
        json.dumps(result["diagnostics"], allow_nan=False)

    def test_no_events_cannot_label_static_merely_from_reference_texture(self):
        arguments = self.scene()
        arguments[3] = [np.zeros_like(arguments[0]) for _ in arguments[5]]
        result = self.evaluate(arguments)
        self.assertTrue((result["winner"] == -1).all())
        self.assertTrue((result["reason"] == "insufficient_net_event_change").all())

    def test_identical_candidates_abstain_even_with_events(self):
        arguments = self.scene()
        arguments[6][:, 0] = arguments[6][:, 1]
        result = self.evaluate(arguments)
        self.assertTrue((result["winner"] == -1).all())
        np.testing.assert_array_equal(result["margin"], 0)
        self.assertTrue((result["reason"] == "ambiguous_candidate_scores").all())

    def test_invalid_nan_poses_are_excluded_from_the_joint_timeline_not_filled(self):
        arguments = self.scene()
        arguments[7][[1, 3], 1] = False
        arguments[6][[1, 3], 1] = np.nan
        original = arguments[6].copy()
        result = self.evaluate(arguments)
        np.testing.assert_array_equal(result["common_time_indices"], [2, 4, 5, 6])
        self.assertTrue((result["winner"] == -1).all())
        self.assertTrue((result["reason"] == "insufficient_timeline_span").all())
        np.testing.assert_array_equal(arguments[6], original)
        self.assertFalse(result["metadata"]["poses_interpolated"])

    def test_each_candidate_own_four_frames_are_not_a_shared_four_frames(self):
        arguments = self.scene(count=9)
        arguments[7][:] = False
        arguments[7][1:5, 0] = True
        arguments[7][5:, 1] = True
        arguments[6][1:][~arguments[7][1:]] = np.nan
        result = self.evaluate(arguments)
        self.assertEqual(len(result["common_times"]), 0)
        self.assertTrue((result["winner"] == -1).all())
        self.assertTrue(np.isinf(result["cost"]).all())
        json.dumps(result["diagnostics"], allow_nan=False)

    def test_four_late_frames_alone_cannot_claim_full_timeline_support(self):
        arguments = self.scene(count=9)
        arguments[7][1:5, 1] = False
        result = self.evaluate(arguments)
        np.testing.assert_array_equal(result["common_time_indices"], [5, 6, 7, 8])
        self.assertTrue((result["winner"] == -1).all())
        self.assertTrue((result["reason"] == "insufficient_timeline_span").all())

    def test_anchor_time_caps_and_determinism(self):
        first = self.evaluate(self.scene(count=33), max_anchors=37)
        second = self.evaluate(self.scene(count=33), max_anchors=37)
        self.assertLessEqual(len(first["anchor_points"]), 37)
        self.assertEqual(len(first["common_times"]), 12)
        np.testing.assert_array_equal(first["common_time_indices"][[0, -1]], [1, 32])
        np.testing.assert_array_equal(first["anchor_pixels"], second["anchor_pixels"])
        np.testing.assert_array_equal(first["cost"], second["cost"])
        self.assertEqual(first["metadata"], second["metadata"])

    def test_nonidentity_camera_backprojection_uses_normalized_depth(self):
        arguments = self.scene()
        angle = .45
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, [.3, -.2, .6]
        arguments[4]["w2c"] = transform
        arguments[1][arguments[1] > 0] = .96
        arguments[6][:, 1, :3, 3] = arguments[6][:, 1, :3, 3] @ rotation
        result = self.evaluate(arguments)
        camera_points = result["anchor_points"] @ rotation.T + transform[:3, 3]
        np.testing.assert_allclose(camera_points[:, 2], 2, atol=1e-12)
        projected = camera_points[:, :2] / camera_points[:, 2:] * [40, 40] + [40, 24]
        np.testing.assert_allclose(projected, result["anchor_pixels"], atol=1e-12)
        self.assertGreater((result["winner"] == 1).mean(), .95)

    def test_low_alpha_and_depth_discontinuity_never_become_surface_anchors(self):
        arguments = self.scene()
        arguments[1][:, :20] = .5
        arguments[2][arguments[1] < .95] = np.nan
        arguments[2][:, 30:] *= 2
        result = self.evaluate(arguments)
        x = result["anchor_pixels"][:, 0] - .5
        self.assertTrue((x >= 21).all())
        self.assertFalse(((x >= 29) & (x <= 30)).any())
        self.assertGreater(result["diagnostics"]["grid_rejected_invalid_or_discontinuous_depth"], 0)

    def test_out_of_frame_candidate_removes_the_same_frames_for_everyone(self):
        arguments = self.scene()
        arguments[6][1:5, 0, 0, 3] = 100
        result = self.evaluate(arguments)
        self.assertFalse(result["shared_time_mask"][:, :4].any())
        self.assertTrue(np.isinf(result["per_time_cost"][:, :4]).all())
        self.assertTrue((result["winner"] == -1).all())
        self.assertTrue((result["common_time_count"] == 2).all())

    def test_proxy_saturation_is_shared_abstention_not_candidate_clipping(self):
        arguments = self.scene()
        arguments[3][1:] = [-arguments[0] for _ in range(6)]
        result = self.evaluate(arguments)
        self.assertFalse(result["shared_time_mask"].any())
        self.assertTrue((result["winner"] == -1).all())
        self.assertGreater(result["diagnostics"]["shared_anchor_times_rejected_proxy_floor_or_saturation"], 0)

    def test_unexplained_darkness_or_occlusion_is_unknown_not_static(self):
        arguments = self.scene()
        arguments[3][1:] = [np.log(.04) - arguments[0] for _ in range(6)]
        result = self.evaluate(arguments)
        self.assertTrue((result["winner"] == -1).all())
        self.assertTrue((result["reason"] == "high_residual_or_occlusion").all())
        self.assertFalse(result["metadata"]["target_occlusion_verified"])

    def test_alternating_motion_support_cannot_win_by_average_alone(self):
        arguments = self.scene(directions=[0, 1, -1, 1, -1, 1, -1])
        translations = np.abs(arguments[6][:, 1, 0, 3])
        arguments[6][:, 0, 0, 3] = -translations
        arguments[6][:, 1, 0, 3] = translations
        result = self.evaluate(arguments, maximum_absolute_log_error=10)
        self.assertTrue((result["winner"] == -1).all())
        self.assertTrue((result["winning_time_fraction"] <= .5 + 1e-12).all())

    def test_one_candidate_is_not_a_comparative_winner(self):
        arguments = self.scene()
        arguments[6], arguments[7] = arguments[6][:, 1:2], arguments[7][:, 1:2]
        result = self.evaluate(arguments)
        self.assertTrue((result["winner"] == -1).all())
        self.assertTrue((result["reason"] == "no_competing_candidate").all())

    def test_array_only_no_mutation_and_json_safe_nonarrays(self):
        arguments = self.scene()
        originals = [array.copy() for array in arguments if isinstance(array, np.ndarray)]
        with patch("builtins.open", side_effect=AssertionError("No file access")):
            result = self.evaluate(arguments)
        actual = [array for array in arguments if isinstance(array, np.ndarray)]
        for before, after in zip(originals, actual):
            np.testing.assert_array_equal(before, after)
        for value in result.values():
            if not isinstance(value, np.ndarray):
                json.dumps(value, allow_nan=False)
        self.assertEqual(result["metadata"]["pose_observations_added"], 0)
        self.assertFalse(result["metadata"]["binding_modified"])
        self.assertFalse(result["metadata"]["approved_for_export"])

    def test_invalid_schemas_are_rejected(self):
        edits = [lambda a: a[4].update(pixel_center="opencv"),
                 lambda a: a[4].update(width=81),
                 lambda a: a[4]["w2c"].__setitem__((0, 0), 2),
                 lambda a: a[7].__setitem__((1, 1), True) or a[6].__setitem__((1, 1, 0, 0), np.nan),
                 lambda a: a[6].__setitem__((0, 1, 0, 3), .1),
                 lambda a: a[5].__setitem__(2, 1),
                 lambda a: a[0].__setitem__((0, 0), .1),
                 lambda a: a[1].__setitem__((0, 0), 2),
                 lambda a: a[3][0].__setitem__((0, 0), .2),
                 lambda a: a[3][1].__setitem__((0, 0), np.nan),
                 lambda a: a.__setitem__(7, a[7].astype(int))]
        for edit in edits:
            with self.subTest(edit=edit):
                arguments = self.scene()
                edit(arguments)
                with self.assertRaises(ValueError):
                    self.evaluate(arguments)
        for key, value in (("max_anchors", 4097), ("max_times", 13), ("min_common_times", 3),
                           ("minimum_alpha", .8), ("minimum_cost_gap", 0),
                           ("minimum_winning_fraction", .5), ("log_floor", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.evaluate(**{key: value})

    def test_stream_length_is_exact_and_unscored_maps_are_still_validated(self):
        for excess in (-1, 1):
            arguments = self.scene()
            arguments[3] = arguments[3][:excess] if excess < 0 else arguments[3] + [arguments[3][-1]]
            with self.assertRaises(ValueError):
                self.evaluate(arguments)
        arguments = self.scene(count=33)
        arguments[3][2][0, 0] = np.inf
        with self.assertRaises(ValueError):
            self.evaluate(arguments)


if __name__ == "__main__":
    unittest.main()
