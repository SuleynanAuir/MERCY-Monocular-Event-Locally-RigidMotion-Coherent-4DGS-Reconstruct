import io
import unittest

import numpy as np
from PIL import Image

from experiment.ui_capture import decode_packet, score
from utils.viewer_state import image_packet
from experiment.ui_supplement import paired_blocks
from experiment.metrics import paired_effect


class UIIdentityTests(unittest.TestCase):
    def packet(self):
        pixels = np.zeros((16, 32, 3), np.uint8)
        pixels[:, 16:] = 192
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, 'JPEG', quality=95, subsampling=0)
        jpeg = stream.getvalue()
        packet = image_packet(dict(frame=7, left_frame=7, right_frame=7, panel_width=16), jpeg)
        return packet, jpeg

    def test_keeps_exact_transmitted_jpeg(self):
        packet, jpeg = self.packet()
        _, actual, pixels = decode_packet(packet, 7, 16)
        self.assertEqual(actual, jpeg)
        self.assertEqual(pixels.shape, (16,32,3))
        self.assertLess(pixels[:, :16].mean(), pixels[:, 16:].mean())

    def test_rejects_different_frame_or_camera_size(self):
        packet, _ = self.packet()
        for frame, size in [(8,16), (7,32)]:
            with self.assertRaises(ValueError):
                decode_packet(packet, frame, size)

    def test_metric_identity_and_difference(self):
        a = np.full((16,16,3), 128, np.uint8)
        same = score(a, a)
        different = score(a, np.zeros_like(a))
        self.assertEqual(same['rgb_l1'], 0)
        self.assertEqual(same['ssim'], 1)
        self.assertTrue(same['exact_match'])
        self.assertIsNone(same['psnr'])
        self.assertGreater(different['rgb_l1'], 0)
        self.assertLess(different['ssim'], 1)

    def test_identical_sequences_have_zero_temporal_effect(self):
        result=paired_blocks(np.zeros(79))
        self.assertEqual(result['mean'],0.)
        self.assertEqual(result['ci95'],[0.,0.])
        self.assertEqual(result['p'],1.)

    def test_five_scenes_cannot_reach_exact_two_sided_005(self):
        result=paired_effect(np.zeros(5),np.ones(5),1)
        self.assertEqual(result['p_raw'],.0625)


if __name__ == '__main__':
    unittest.main()
