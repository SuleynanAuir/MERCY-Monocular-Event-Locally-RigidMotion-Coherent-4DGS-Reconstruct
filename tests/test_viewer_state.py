import json
import struct
import unittest

import numpy as np

from utils.viewer_state import image_packet, orbit_pose, stream_frame, validate_request, validate_stream


class ViewerStateTests(unittest.TestCase):
    def test_defaults(self):
        state = validate_request({}, 360)
        self.assertEqual(state["frame"], 0)
        self.assertEqual(state["background"], "dark")
        self.assertEqual(state['render_mode'], 'gaussian')

    def test_bad_values(self):
        for data in ([], None, {"frame": -1}, {"frame": 360}, {"frame": 1.5},
                     {"frame": True}, {"width": 10000}, {"zoom": 0},
                     {"yaw": float("nan")}, {"pitch": float("inf")},
                     {"pan_x": "1"}, {"background": []}, {"background": "red"},
                     {"quality": 1}, {"width": 1280, "height": 1280},
                     {'render_mode': 'unknown'}, {'render_mode': []}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_request(data, 360)

    def test_frame_end(self):
        self.assertEqual(validate_request({"frame": 359}, 360)["frame"], 359)

    def test_identity_view(self):
        pose = np.eye(4); pose[2, 3] = 3
        result = orbit_pose(pose, [0, 0, 0], validate_request({}, 360))
        np.testing.assert_allclose(result, pose)

    def test_orbit_basis_and_distance(self):
        pose = np.eye(4); pose[:3, 3] = [1, 2, 6]
        state = validate_request({"yaw": 37, "pitch": -26, "zoom": 2}, 360)
        result = orbit_pose(pose, [1, 2, 3], state)
        np.testing.assert_allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-6)
        self.assertAlmostEqual(np.linalg.det(result[:3, :3]), 1, places=6)
        self.assertAlmostEqual(np.linalg.norm(result[:3, 3] - [1, 2, 3]), 6, places=6)

    def test_pan_and_zoom(self):
        pose = np.eye(4); pose[2, 3] = 2
        result = orbit_pose(pose, [0, 0, 0], validate_request({"pan_x": .5, "pan_y": -.5, "zoom": 3}, 2))
        np.testing.assert_allclose(result[:3, 3], [1, -1, 6])

    def test_degenerate_camera(self):
        with self.assertRaises(ValueError):
            orbit_pose(np.eye(4), [0, 0, 0], validate_request({}, 2))

    def test_binary_packet(self):
        packet = image_packet({"id": 7, "frame": 3}, b"jpeg-data")
        size = struct.unpack("<I", packet[:4])[0]
        self.assertEqual(json.loads(packet[4:4 + size]), {"id": 7, "frame": 3})
        self.assertEqual(packet[4 + size:], b"jpeg-data")

    def test_stream_preserves_source_time_and_speed(self):
        request = validate_stream({"speed": 1, "rate": 60}, 360)
        for i in range(360):
            self.assertEqual(stream_frame(request, i, 60, 360), (i, i == 359))
        request["speed"] = .5
        self.assertEqual(stream_frame(request, 120, 60, 360), (60, False))
        request["speed"] = 2
        self.assertEqual(stream_frame(request, 120, 60, 360), (240, False))

    def test_stream_last_frame_and_loop(self):
        request = validate_stream({"frame": 358, "speed": 2}, 360)
        self.assertEqual(stream_frame(request, 1, 60, 360), (359, True))
        request["loop"] = True
        self.assertEqual(stream_frame(request, 1, 60, 360), (0, False))
        request = validate_stream({"rate": 30}, 360)
        self.assertEqual(stream_frame(request, 15, 60, 360), (30, False))


if __name__ == "__main__":
    unittest.main()
