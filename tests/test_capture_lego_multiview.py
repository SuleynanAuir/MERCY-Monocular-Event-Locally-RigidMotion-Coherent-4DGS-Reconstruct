import io
import json
import struct
import unittest

from PIL import Image

from scripts.capture_lego_multiview import decode_packet, snapshot_plan, verify_scene


class MultiviewCaptureTests(unittest.TestCase):
    def test_plan_has_back_and_tilted_all_around_views(self):
        plan = snapshot_plan()
        self.assertEqual(len(plan), 36)
        self.assertEqual(len(set(plan)), 36)
        for frame in (0, 100, 200, 300):
            self.assertIn((frame, 180, 0), plan)
        for pitch in (-35, 0, 35):
            for yaw in range(0, 360, 45):
                self.assertIn((100, yaw, pitch), plan)

    def test_scene_identity_and_event_decision_are_enforced(self):
        source, correction = 'a' * 64, 'b' * 64
        audit = {'artifact_sha256': {'source_ply': source, 'correction': correction},
                 'summary': {'all': {'passed': 380}}, 'approved_for_export': False}
        scene = {'comparison': True, 'right_source_sha256': source,
                 'candidate_correction_sha256': correction,
                 'event_validation': {'summary': audit['summary']['all'], 'approved': False}}
        verify_scene(scene, audit, source, correction)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            verify_scene(scene, audit, 'c' * 64, correction)
        scene['event_validation']['approved'] = True
        with self.assertRaisesRegex(ValueError, 'decision'):
            verify_scene(scene, audit, source, correction)

    def test_packet_requires_matched_frame_and_dimensions(self):
        request = {'id': 1, 'frame': 100, 'width': 128, 'height': 128}
        metadata = {'id': 1, 'frame': 100, 'left_frame': 100, 'right_frame': 100,
                    'width': 256, 'height': 128}
        jpeg = io.BytesIO()
        Image.new('RGB', (256, 128)).save(jpeg, format='JPEG')

        def packet():
            header = json.dumps(metadata).encode()
            return struct.pack('<I', len(header)) + header + jpeg.getvalue()

        _, _, image = decode_packet(packet(), request)
        self.assertEqual(image.size, (256, 128))
        metadata['right_frame'] = 99
        with self.assertRaisesRegex(ValueError, 'right_frame'):
            decode_packet(packet(), request)
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            decode_packet(b'x', request)


if __name__ == '__main__':
    unittest.main()
