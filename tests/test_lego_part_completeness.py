import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    'measure_parts', Path(__file__).resolve().parents[1] / 'scripts/measure_lego_part_completeness.py')
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)


class PartCompletenessTest(unittest.TestCase):
    def test_only_reference_bright_pixels_inside_region_count(self):
        reference = np.array([[.5, .5], [.05, .5]])
        rendered = np.array([[.01, .5], [.0, .0]])
        region = np.array([[True, True], [True, False]])
        result = measure.coverage(reference, rendered, region, min_pixels=1)
        self.assertEqual(result['reference_bright_pixels'], 2)
        self.assertEqual(result['missing_pixels'], 1)
        self.assertEqual(result['missing_fraction'], .5)
        self.assertEqual(result['largest_missing_component_pixels'], 1)

    def test_empty_part_not_counted_as_perfect(self):
        zeros = np.zeros((3, 3))
        result = measure.coverage(zeros, zeros, np.ones((3, 3), dtype=bool))
        self.assertIsNone(result['missing_fraction'])
        self.assertFalse(result['eligible'])
        self.assertEqual(measure.summarize([result]), {'eligible_frames': 0})

    def test_summary_keeps_worst_frame_not_only_global_average(self):
        rows = [dict(frame_index=0, reference_bright_pixels=100, missing_pixels=1,
                     missing_fraction=.01, largest_missing_component_pixels=1, eligible=True),
                dict(frame_index=10, reference_bright_pixels=20, missing_pixels=10,
                     missing_fraction=.5, largest_missing_component_pixels=5, eligible=True)]
        summary = measure.summarize(rows)
        self.assertEqual(summary['worst_frame_index'], 10)
        self.assertEqual(summary['worst_frame_missing_fraction'], .5)
        self.assertAlmostEqual(summary['pixel_weighted_missing_fraction'], 11 / 120)


if __name__ == '__main__':
    unittest.main()
