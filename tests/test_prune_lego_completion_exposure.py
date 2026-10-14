import copy
import json
import unittest

import numpy as np
import torch

from scripts.prune_lego_completion_exposure import (C0, contribution_from_dc_gradient,
    filtered_selection, resume_active_mask, run_pruning, validate_options, validate_selection)


class CompletionExposureTests(unittest.TestCase):
    def test_uniform_dc_color_gradient_is_accumulated_alpha_weight(self):
        # Two pixels, three front-to-back Gaussians, known per-pixel alpha.
        alpha = torch.tensor([[.2, .6, .9], [.7, .1, .4]], dtype=torch.float64)
        transmission = torch.cat((torch.ones((2, 1), dtype=torch.float64),
                                  (1-alpha).cumprod(1)[:, :-1]), 1)
        weights = transmission*alpha
        dc = torch.zeros((3, 1, 3), dtype=torch.float64, requires_grad=True)
        colors = .5+C0*dc[:, 0]
        pixels = weights @ colors
        luma = torch.tensor([.299, .587, .114], dtype=torch.float64)
        (pixels*luma).sum().backward()
        actual = contribution_from_dc_gradient(dc.grad)
        torch.testing.assert_close(actual, weights.sum(0), rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.all(actual > 0))
        self.assertLess(float(pixels.max()), .5)

    def test_hidden_point_has_zero_gradient_until_occluder_is_removed(self):
        for front_alpha, expected in ((1., 0.), (0., .8)):
            dc = torch.zeros((1, 1, 3), requires_grad=True)
            color = .5+C0*dc[0, 0]
            gray = (color*torch.tensor([.299, .587, .114])).sum()
            (gray*((1-front_alpha)*.8)).backward()
            self.assertAlmostEqual(float(contribution_from_dc_gradient(dc.grad)[0]), expected, places=6)

    def test_invalid_gradient_is_not_silently_used(self):
        bad = (None, torch.zeros((3, 3)), torch.ones((3, 1, 3), dtype=torch.int32),
               torch.full((3, 1, 3), float('nan')), torch.full((3, 1, 3), -.01))
        for gradient in bad:
            with self.subTest(gradient=gradient), self.assertRaises(ValueError):
                contribution_from_dc_gradient(gradient)

    def test_final_full_pass_detects_between_probe_exposure_and_repeats_until_stable(self):
        calls = []
        def scan(active, frames, phase, number):
            calls.append((active.copy(), frames, phase, number))
            value = np.zeros(4, np.float32)
            if phase == 'probe':
                if active[0]:
                    value[0] = .5
            elif active[1]:
                value[1] = .5  # Exposed only on a frame absent from probe.
            elif active[2]:
                value[2] = .5  # Uncovered by removing the previous row.
            return value
        keep, stats = run_pruning(scan, 4, max_rounds=4)
        np.testing.assert_array_equal(keep, [False, False, False, True])
        self.assertTrue(stats['converged_full_400'])
        self.assertEqual(stats['history'][-1]['removed'], 0)
        self.assertEqual(stats['history'][-1]['frames_checked'], 400)
        full_calls = [row for row in calls if row[2] == 'full']
        self.assertEqual(len(full_calls), 3)
        for _, frames, _, _ in full_calls:
            self.assertEqual(frames, list(range(400)))
        self.assertIn(399, calls[0][1])

    def test_round_limit_never_claims_convergence_after_last_deletion(self):
        def scan(active, frames, phase, number):
            value = np.zeros(4, np.float32)
            if phase == 'full' and active.any():
                value[np.flatnonzero(active)[0]] = 1.
            return value
        keep, stats = run_pruning(scan, 4, max_rounds=2)
        self.assertEqual(int(keep.sum()), 2)
        self.assertFalse(stats['converged_full_400'])
        self.assertGreater(stats['history'][-1]['removed'], 0)

    def test_threshold_is_per_point_max_contribution_and_inclusive_retention(self):
        def scan(active, frames, phase, number):
            return np.asarray([0., .0005, .0006], np.float64)*active
        keep, stats = run_pruning(scan, 3, max_rounds=2)
        np.testing.assert_array_equal(keep, [True, True, False])
        self.assertTrue(stats['converged_full_400'])

    def test_selection_filter_preserves_geometry_parameters_and_existing_provenance(self):
        meta = {'format': 'lego_unobserved_prior_surface_selection_v1', 'spacing': .0025,
                'guard_tolerance': .0009, 'surface_tolerance': .0024, 'minimum_distance_factor': .4,
                'added_points': 3, 'input_sha256': {'mesh': 'a'*64, 'parent_source': 'b'*64}}
        original = {'xyz': np.arange(9, dtype=np.float32).reshape(3, 3),
                    'normals': np.tile([0., 0., 1.], (3, 1)).astype(np.float32),
                    'part_ids': np.asarray([1, 2, 3], np.int32), 'material_gray': np.asarray([.1, .2, .3], np.float32),
                    'triangle_ids': np.asarray([10, 20, 30], np.int32),
                    'metadata_json': np.asarray(json.dumps(meta))}
        snapshot = copy.deepcopy(original)
        result = filtered_selection(original, np.asarray([True, False, True]), {'converged_full_400': True})
        self.assertEqual(set(result), set(original))
        after = json.loads(str(result['metadata_json']))
        self.assertEqual(after['added_points'], 2)
        for key in ('spacing', 'guard_tolerance', 'surface_tolerance', 'minimum_distance_factor', 'input_sha256'):
            self.assertEqual(after[key], meta[key])
        for key in ('xyz', 'normals', 'part_ids', 'material_gray', 'triangle_ids'):
            np.testing.assert_array_equal(result[key], original[key][[0, 2]])
        for key in original:
            np.testing.assert_array_equal(original[key], snapshot[key])

    def test_bounded_options(self):
        validate_options(4, 10, .0005)
        for values in ((0, 10, .0005), (9, 10, .0005), (4, 0, .0005), (4, 10, 0.),
                       (4, 10, .1), (True, 10, .0005), (4, 10, float('nan'))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_options(*values)
        for cap in (0, 65, True):
            with self.subTest(probe_cap=cap), self.assertRaises(ValueError):
                validate_options(4, 10, .0005, cap)

    def test_longer_probe_peeling_does_not_replace_full_frame_convergence(self):
        def scan(active, frames, phase, number):
            value = np.zeros(8, np.float32)
            exposed = np.flatnonzero(active[:7])
            if len(exposed):
                value[exposed[0]] = .1
            return value
        keep, stats = run_pruning(scan, 8, max_rounds=1, max_probe_rounds=32)
        self.assertEqual(int(keep.sum()), 1)
        self.assertTrue(stats['probe_converged'])
        self.assertTrue(stats['converged_full_400'])
        self.assertEqual(len([row for row in stats['history'] if row['phase'] == 'probe']), 8)
        self.assertEqual(stats['history'][-1]['frames_checked'], 400)
        _, stopped = run_pruning(scan, 8, max_rounds=1, max_probe_rounds=1)
        self.assertFalse(stopped['converged_full_400'])
        self.assertFalse(stopped['probe_converged'])

    def resume_fixture(self):
        meta = {'format': 'lego_unobserved_prior_surface_selection_v1', 'spacing': .0025,
                'guard_tolerance': .0009, 'surface_tolerance': .0024, 'minimum_distance_factor': .4,
                'added_points': 3, 'input_sha256': {'mesh': 'a'*64, 'parent_source': 'b'*64}}
        original = {'xyz': np.arange(9, dtype=np.float32).reshape(3, 3),
                    'normals': np.tile([0., 0., 1.], (3, 1)).astype(np.float32),
                    'part_ids': np.asarray([1, 2, 3], np.int32), 'material_gray': np.asarray([.1, .2, .3], np.float32),
                    'triangle_ids': np.asarray([10, 20, 30], np.int32),
                    'metadata_json': np.asarray(json.dumps(meta))}
        expected = {'source_selection': '/tmp/original.npz', 'source_selection_sha256': 'c'*64,
                    'source_completion_report': '/tmp/report.json', 'source_completion_report_sha256': 'd'*64,
                    'source_ply_sha256': 'e'*64, 'source_correction_sha256': 'f'*64,
                    'parent_source_sha256': '1'*64, 'parent_correction_sha256': '2'*64,
                    'render_model_parameters': {'spacing': .0025, 'tangent_factor': .75, 'normal_factor': .12, 'opacity': .85},
                    'render_sampling': {'render_supersample': 1, 'render_downsample_filter': 'area_after_clamp'},
                    'camera': {'fovy_degrees': 40}, 'threshold_output_pixels': .0005}
        exposure = {'format': 'lego_raster_alpha_exposure_pruning_v1', **expected,
                    'converged_full_400': False, 'original_prefix_unchanged': True, 'source_artifacts_unchanged': True}
        resumed = filtered_selection(original, np.asarray([True, False, True]), exposure)
        return original, resumed, expected

    def test_resume_exact_subset_and_previous_removals_are_preserved(self):
        original, resumed, expected = self.resume_fixture()
        initial = resume_active_mask(original, resumed, expected)
        np.testing.assert_array_equal(initial, [True, False, True])
        def scan(active, frames, phase, number):
            self.assertFalse(active[1])
            return np.zeros(3, np.float32)
        keep, stats = run_pruning(scan, 3, initial_active=initial)
        np.testing.assert_array_equal(keep, initial)
        self.assertEqual(stats['initial_active_count'], 2)
        self.assertEqual(stats['previously_removed_count'], 1)
        self.assertEqual(stats['removed_this_run_count'], 0)
        self.assertTrue(stats['converged_full_400'])
        np.testing.assert_array_equal(initial, [True, False, True])

    def test_resume_refuses_source_threshold_footprint_camera_or_sampling_mismatch(self):
        original, resumed, expected = self.resume_fixture()
        for key in expected:
            wrong = copy.deepcopy(resumed)
            meta = json.loads(str(wrong['metadata_json']))
            meta['exposure_pruning'][key] = 'different'
            wrong['metadata_json'] = np.asarray(json.dumps(meta))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                resume_active_mask(original, wrong, expected)

    def test_resume_refuses_changed_row_bits_duplicates_or_material(self):
        for kind in ('xyz', 'negative_zero', 'duplicate', 'material', 'triangle', 'geometry_metadata'):
            original, resumed, expected = self.resume_fixture()
            if kind == 'xyz':
                resumed['xyz'][0, 0] += .000001
            elif kind == 'negative_zero':
                resumed['xyz'][0, 0] = -0.
            elif kind == 'duplicate':
                for key in ('xyz', 'normals', 'part_ids', 'material_gray', 'triangle_ids'):
                    resumed[key][1] = resumed[key][0]
            elif kind == 'material':
                resumed['material_gray'][0] += .1
            elif kind == 'triangle':
                resumed['triangle_ids'][0] += 1
            else:
                meta = json.loads(str(resumed['metadata_json'])); meta['spacing'] = .004
                resumed['metadata_json'] = np.asarray(json.dumps(meta))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                resume_active_mask(original, resumed, expected)


if __name__ == '__main__':
    unittest.main()
