import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    'lego_detail', Path(__file__).resolve().parents[1] / 'scripts/measure_lego_detail.py')
detail = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detail)


def synthetic_part():
    image = np.zeros((96, 96), dtype=np.float32)
    image[15:80, 20:75] = .75
    yy, xx = np.mgrid[:96, :96]
    for cy in (29, 49, 69):
        image[(xx - 47)**2 + (yy - cy)**2 <= 6**2] = .03
    return image


def blur(image, iterations=5):
    image = image.copy()
    for _ in range(iterations):
        a = np.pad(image, 1, mode='reflect')
        image = (a[:-2, :-2] + 2*a[:-2, 1:-1] + a[:-2, 2:]
                 + 2*a[1:-1, :-2] + 4*a[1:-1, 1:-1] + 2*a[1:-1, 2:]
                 + a[2:, :-2] + 2*a[2:, 1:-1] + a[2:, 2:]) / 16
    return image


class LegoDetailTest(unittest.TestCase):
    def test_identical_has_perfect_edges_and_zero_error(self):
        image = synthetic_part()
        row = detail.measure_detail(image, image)
        for metric in ('rgb_l1', 'sobel_magnitude_l1', 'sobel_vector_l1',
                       'bright_missing_fraction', 'dark_extra_fraction'):
            self.assertEqual(row[metric], 0.)
        for metric in ('edge_precision', 'edge_recall', 'edge_f1'):
            self.assertEqual(row[metric], 1.)

    def test_blur_loses_reference_edges(self):
        reference = synthetic_part()
        row = detail.measure_detail(reference, blur(reference))
        self.assertGreater(row['rgb_l1'], .01)
        self.assertGreater(row['sobel_magnitude_l1'], .01)
        self.assertLess(row['edge_recall'], .5)

    def test_noise_is_not_mistaken_for_genuine_detail(self):
        reference = synthetic_part()
        noisy = np.clip(reference + np.random.default_rng(9).normal(0, .45, reference.shape), 0, 1)
        clean = detail.measure_detail(reference, reference)
        row = detail.measure_detail(reference, noisy)
        self.assertEqual(clean['reference_edge_threshold'], row['reference_edge_threshold'])
        self.assertLess(row['edge_precision'], .75)
        self.assertLess(row['edge_f1'], clean['edge_f1'])
        self.assertGreater(row['sobel_magnitude_l1'], .03)
        self.assertGreater(row['dark_extra_fraction'], .2)

    def test_noise_added_to_blur_increases_mismatch_not_quality(self):
        reference = synthetic_part()
        soft = blur(reference, 2)
        noisy = np.clip(soft + np.random.default_rng(17).normal(0, .2, reference.shape), 0, 1)
        before, after = detail.measure_detail(reference, soft), detail.measure_detail(reference, noisy)
        self.assertGreater(after['sobel_vector_l1'], before['sobel_vector_l1'])
        self.assertGreater(after['rgb_l1'], before['rgb_l1'])
        self.assertGreater(after['dark_extra_fraction'], before['dark_extra_fraction'])

    def test_inverted_edges_fail_signed_direction_matching(self):
        reference = synthetic_part()
        row = detail.measure_detail(reference, 1-reference)
        self.assertLess(row['edge_f1'], .1)

    def test_empty_edges_are_reported_as_unavailable(self):
        row = detail.measure_detail(np.zeros((16, 16)), np.zeros((16, 16)))
        self.assertIsNone(row['edge_f1'])
        self.assertIsNone(row['edge_precision'])
        self.assertIsNone(row['bright_missing_fraction'])
        self.assertIsNone(detail.summarize([row])['edge_f1'])

    def test_sparse_mapping_requires_explicit_valid_original_indices(self):
        self.assertEqual(detail.frame_mapping(3, 400, [0, 50, 100]), {0: 0, 50: 1, 100: 2})
        for indices in (None, [0, 0, 100], [0, 50, 400], [0, 50]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                detail.frame_mapping(3, 400, indices)
        self.assertEqual(detail.frame_mapping(400, 400, [0, 50, 100])[100], 100)

    def test_input_range_and_shape_are_validated(self):
        valid = synthetic_part()
        for value in (float('nan'), float('inf'), -.01, 1.01):
            invalid = valid.copy()
            invalid[0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                detail.measure_detail(valid, invalid)
        with self.assertRaises(ValueError):
            detail.measure_detail(valid, valid[:-1])

    def test_uint8_and_float_reference_use_equivalent_thresholds(self):
        uint8 = np.rint(synthetic_part()*255).astype(np.uint8)
        actual = uint8.astype(np.float32)/255
        row = detail.measure_detail(uint8, actual)
        self.assertEqual(row['rgb_l1'], 0.)
        self.assertEqual(row['edge_f1'], 1.)


if __name__ == '__main__':
    unittest.main()
