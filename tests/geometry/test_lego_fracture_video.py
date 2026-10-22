import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    'fracture_video', Path(__file__).resolve().parents[2] / 'scripts/geometry/export_lego_fracture_video.py')
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)


class FractureVideoTest(unittest.TestCase):
    def test_grayscale_validation(self):
        video.validate_render_frame(np.array([[0., .5, 1.]], np.float16))
        for value in (float('nan'), float('inf'), -.01, 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                video.validate_render_frame(np.array([[value]], np.float32))

    def test_worst_frames_are_time_separated_and_chronological(self):
        scores = np.zeros(60)
        scores[[10, 11, 30, 31, 50]] = [1., .9, .8, .7, .6]
        self.assertEqual(video.select_worst_frames(scores, count=3, minimum_gap=12), [10, 30, 50])

    def test_small_sequence_has_valid_selection(self):
        self.assertEqual(video.select_worst_frames([0.1]), [0])


if __name__ == '__main__':
    unittest.main()
