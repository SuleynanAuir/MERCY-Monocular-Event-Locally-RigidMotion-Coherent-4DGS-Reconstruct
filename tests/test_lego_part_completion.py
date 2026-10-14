import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    'complete_parts', Path(__file__).resolve().parents[1] / 'scripts/complete_lego_parts.py')
completion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(completion)


class PartCompletionTest(unittest.TestCase):
    def aggregate(self, xyz, labels, frames, gray=None):
        xyz = np.asarray(xyz, dtype=float)
        count = len(xyz)
        return completion.aggregate_supported_cells(
            xyz, np.asarray(labels), np.asarray(frames),
            np.asarray(gray if gray is not None else [.5] * count),
            np.full(count, .01), np.tile([0., 0., 1.], (count, 1)), 1., 3)

    def test_requires_distinct_frames_not_repeated_pixels(self):
        rows, _ = self.aggregate([[.1, .1, .1]] * 4, [2] * 4, [10, 10, 10, 20])
        self.assertEqual(rows, [])
        rows, _ = self.aggregate([[.1, .1, .1]] * 4, [2] * 4, [10, 10, 20, 30])
        self.assertEqual(rows[0]['support_frames'], [10, 20, 30])

    def test_different_parts_never_merge(self):
        rows, cells = self.aggregate([[.1, .1, .1]] * 6, [2, 2, 2, 3, 3, 3], [10, 20, 30] * 2)
        self.assertEqual(cells, 2)
        self.assertEqual([row['part_id'] for row in rows], [2, 3])

    def test_position_is_real_hit_not_averaged_across_surface(self):
        xyz = [[.1, .1, .1], [.2, .2, .2], [.4, .4, .4]]
        rows, _ = self.aggregate(xyz, [2] * 3, [10, 20, 30], [.2, .4, .9])
        np.testing.assert_array_equal(rows[0]['xyz'], xyz[1])
        self.assertEqual(rows[0]['gray'], .4)


if __name__ == '__main__':
    unittest.main()
