import copy
import unittest
import numpy as np
from scripts.mesh.refine_viewer_mesh_colors import validate_capture, detail_color_limits


class ColorCaptureTests(unittest.TestCase):
    def test_flat_colors_are_protected_without_locking_existing_details(self):
        colors = np.array([[.8]*3, [.8]*3, [.8]*3, [0.]*3, [1.]*3, [1.]*3], np.float32)
        limits = detail_color_limits(colors, np.array([[0,1,2], [3,4,5]]), .2)
        np.testing.assert_allclose(limits[:3], .01)
        np.testing.assert_allclose(limits[3:], .2)

    def capture(self):
        return dict(format='gaussian_mesh_color_views_v1', scene=dict(frame_count=400, comparison=True),
                    views=[dict(split=split, request=dict(frame=frame, yaw=yaw, render_mode='gaussian', background='black'))
                           for split, frame, yaw in [('train', 0, 0), ('validation', 200, 45)]])

    def test_validation_cannot_reuse_training_camera_even_at_another_time(self):
        capture = self.capture()
        validate_capture(capture, 400)
        capture['views'][1]['request']['yaw'] = 360
        with self.assertRaisesRegex(ValueError, 'separate camera'):
            validate_capture(capture, 400)

    def test_rejects_mesh_images_and_mismatched_timelines(self):
        capture = self.capture()
        with self.assertRaisesRegex(ValueError, 'timeline'):
            validate_capture(capture, 200)
        other = copy.deepcopy(capture)
        other['views'][0]['request']['render_mode'] = 'mesh'
        with self.assertRaisesRegex(ValueError, 'actual Gaussian'):
            validate_capture(other, 400)


if __name__ == '__main__':
    unittest.main()
