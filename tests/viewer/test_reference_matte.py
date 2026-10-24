import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.viewer.prepare_reference_videos import GeometryMatteSequence, white_rgb_preview


class ReferenceMatteTests(unittest.TestCase):
    def test_enclosed_background_removed_but_black_part_preserved(self):
        rgb = np.zeros((32, 32, 3), np.uint8)
        rgb[6:26, 6:26] = 130
        rgb[10:22, 10:22] = 0
        alpha = np.zeros((32, 32), np.float32)
        alpha[6:26, 6:26] = 1
        alpha[10:22, 10:22] = 0
        alpha[10:22, 15:17] = 1  # Real black rod spanning the hole.
        original = rgb.copy()
        result = white_rgb_preview(rgb, alpha)
        np.testing.assert_array_equal(result[12:20, 11:14], 255)
        np.testing.assert_array_equal(result[12:20, 15:17], 0)
        np.testing.assert_array_equal(result[alpha == 1], rgb[alpha == 1])
        np.testing.assert_array_equal(rgb, original)
        np.testing.assert_array_equal(white_rgb_preview(rgb)[12:20, 11:14], 0)

    def test_partial_coverage_adds_white_without_double_multiplying_rgb(self):
        rgb = np.full((8, 8, 3), 100, np.uint8)
        alpha = np.full((8, 8), .5, np.float32)
        result = white_rgb_preview(rgb, alpha)
        self.assertTrue((result > 200).all())
        for invalid in (np.zeros((2, 2)), np.full((8, 8), np.nan), np.full((8, 8), 2)):
            with self.assertRaises(ValueError):
                white_rgb_preview(rgb, invalid)

    def test_incorrect_camera_matte_is_rejected(self):
        rgb = np.zeros((32, 32, 3), np.uint8)
        rgb[4:12, 4:12] = 200
        alpha = np.zeros((32, 32), np.float32)
        alpha[20:28, 20:28] = 1
        with self.assertRaisesRegex(ValueError, 'aligned'):
            white_rgb_preview(rgb, alpha)

    def test_sequence_requires_correct_source_and_every_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = root / 'source.mkv'; rgb.write_bytes(b'original RGB identity')
            video = dict(width=8, height=8, frame_count=2, fps=60.)
            meta = dict(format='rgb_geometry_matte_v1', **video, rendered_frames=2,
                        rgb_sha256=hashlib.sha256(rgb.read_bytes()).hexdigest(), source_frames=[1, 11])
            path = root / 'matte.json'; path.write_text(json.dumps(meta))
            Image.new('L', (8, 8), 255).save(root / '0000.png')
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                GeometryMatteSequence(root, rgb, video)
            Image.new('L', (8, 8), 0).save(root / '0001.png')
            sequence = GeometryMatteSequence(root, rgb, video)
            np.testing.assert_array_equal(sequence.frame(0), 1)
            np.testing.assert_array_equal(sequence.frame(1), 0)
            rgb.write_bytes(b'different scene')
            with self.assertRaisesRegex(ValueError, 'source or timeline'):
                GeometryMatteSequence(root, rgb, video)


if __name__ == '__main__':
    unittest.main()
