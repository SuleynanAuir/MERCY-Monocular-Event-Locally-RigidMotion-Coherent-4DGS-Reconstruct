import copy
import unittest
from unittest.mock import patch

import numpy as np

from scripts.audit_lego_multiview_coverage import (aggregate, compare, measure,
                                                  region_stats, validate_renders)


class MultiviewCoverageTests(unittest.TestCase):
    def test_missing_components_use_eight_connectivity_and_threshold_inclusive(self):
        alpha = np.ones((4, 4), np.float32)
        alpha[0, 0] = alpha[1, 1] = .1
        alpha[3, 3] = .5
        result = region_stats(alpha, np.ones((4, 4), bool), .5)
        self.assertEqual(result['missing_pixels'], 2)
        self.assertEqual(result['largest_missing_component_pixels'], 2)
        self.assertEqual(result['missing_component_count'], 1)

    def test_empty_region_is_unknown_not_complete(self):
        result = region_stats(np.ones((2, 2), np.float32), np.zeros((2, 2), bool), .5)
        self.assertIsNone(result['coverage'])
        self.assertIsNone(result['missing_fraction'])

    def test_view_identity_dtype_and_ranges_are_enforced(self):
        masks = np.zeros((1, 2, 2), np.int16)
        values = np.ones(masks.shape, np.float32)
        metadata = {'views': [{'frame': 0, 'yaw': 180}]}
        validate_renders(values, values, metadata, masks, metadata)
        with self.assertRaisesRegex(ValueError, 'identities'):
            validate_renders(values, values, {'views': [{'frame': 0, 'yaw': 0}]}, masks, metadata)
        with self.assertRaisesRegex(ValueError, 'float32'):
            validate_renders(values.astype(np.float16), values, metadata, masks, metadata)
        bad = values.copy(); bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            validate_renders(bad, values, metadata, masks, metadata)

    def fixture(self):
        masks = np.full((2, 12, 12), -1, np.int16)
        masks[:, 0:8, 0:8] = 0  # Large static base.
        masks[:, 8:12, 8:12] = 1  # Moving part.
        masks[:, 0, 9:12] = 2  # Thin moving rod, erased by erosion.
        metadata = {'objects': ['base', 'arm', 'thin'], 'views': [
            {'frame': 0, 'yaw': 0, 'pitch': 0}, {'frame': 0, 'yaw': 180, 'pitch': 0}]}
        alpha = np.ones(masks.shape, np.float32)
        alpha[1, 8:12, 8:12] = 0
        return masks, metadata, alpha

    def test_moving_backside_is_not_dominated_by_base_and_thin_parts_are_reported(self):
        masks, metadata, alpha = self.fixture()
        with patch('builtins.print'):
            result = measure(alpha, masks, metadata, np.array([0, 1, 1]))
        back = result['summary']['moving_rigid_backside_all_pitches']['0.5']['interior']
        self.assertEqual(back['equal_weight_part_mean_missing_fraction'], 1)
        self.assertEqual(back['zero_interior_or_empty_part_view_count'], 1)
        self.assertEqual(back['eligible_unique_parts'], 1)
        self.assertEqual(back['largest_missing_component_pixels'], 4)
        self.assertGreater(result['views'][0]['thresholds']['0.5']['outside_rigid_prior_pixels'], 0)
        raw = result['summary']['moving_rigid_backside_all_pitches']['0.5']['foreground']
        self.assertEqual(raw['eligible_unique_parts'], 2)
        self.assertEqual(raw['tiny_part_views_under_16_pixels'], 1)

    def test_threshold_sensitivity_is_monotonic(self):
        masks, metadata, alpha = self.fixture()
        alpha[:] = .6
        with patch('builtins.print'):
            result = measure(alpha, masks, metadata, np.array([0, 1, 1]))
        summaries = result['summary']['moving_rigid_all_views']
        self.assertEqual(summaries['0.2']['foreground']['missing_pixels'], 0)
        self.assertEqual(summaries['0.5']['foreground']['missing_pixels'], 0)
        self.assertGreater(summaries['0.8']['foreground']['missing_pixels'], 0)

    def test_comparison_keeps_regressions(self):
        masks, metadata, alpha = self.fixture()
        with patch('builtins.print'):
            before = measure(np.ones_like(alpha), masks, metadata, np.array([0, 1, 1]))
            after = measure(alpha, masks, metadata, np.array([0, 1, 1]))
        changes = compare(before, after)
        self.assertEqual(changes['regressed_part_views'], 1)
        self.assertEqual(changes['regressions'][0]['part_id'], 1)
        self.assertEqual(changes['regressions'][0]['missing_pixel_change'], 4)


if __name__ == '__main__':
    unittest.main()
