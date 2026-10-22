from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.events.refine_lego_events import MotionRenderer
from utils.render_sampling import resolve_rgb


class MotionRendererSamplingTests(unittest.TestCase):
    def test_requested_resolution_scales_native_camera_only(self):
        renderer = MotionRenderer.__new__(MotionRenderer)
        renderer.meta = {'camera': {'pose': np.eye(4).tolist(), 'fovy_degrees': 40.}}
        renderer.supersample = 2
        with patch('scripts.events.refine_lego_events.make_camera', return_value='camera') as make:
            renderer.set_resolution(320)
            self.assertEqual(renderer.width, 320)
            self.assertEqual(make.call_args.args[1:3], (640, 640))
            renderer.set_resolution(640)
            self.assertEqual(make.call_args.args[1:3], (1280, 1280))
            renderer.supersample = 1
            renderer.set_resolution(640)
            self.assertEqual(make.call_args.args[1:3], (640, 640))
        for width in (True, 0, -1, 640.):
            with self.subTest(width=width), self.assertRaises(ValueError):
                renderer.set_resolution(width)

    def test_rgb_resolves_before_gray_without_moving_frozen_rows(self):
        renderer = MotionRenderer.__new__(MotionRenderer)
        renderer.supersample = 2
        renderer.completion = None
        renderer.model = SimpleNamespace()
        renderer.base_xyz = torch.zeros((3, 3))
        renderer.base_rotation = torch.tensor([[1., 0., 0., 0.]] * 3)
        renderer.indices = torch.tensor([1])
        renderer.posed = lambda index, correction: torch.tensor([[2., 3., 4., 1., 0., 0., 0.]])
        renderer.luma = torch.tensor([.299, .587, .114])[:, None, None]
        renderer.camera = renderer.pipe = renderer.background = None
        rgb = torch.linspace(-.1, 1.2, 3 * 8 * 8).reshape(3, 8, 8)
        with patch('scripts.events.refine_lego_events.render', return_value={'render': rgb}):
            gray = renderer.image(1, None)
        expected = (resolve_rgb(rgb, 2) * renderer.luma).sum(0)
        self.assertTrue(torch.equal(gray, expected))
        self.assertEqual(gray.shape, (4, 4))
        self.assertTrue(torch.equal(renderer.base_xyz, torch.zeros((3, 3))))
        self.assertTrue(torch.equal(renderer.model._xyz[[0, 2]], torch.zeros((2, 3))))
        self.assertTrue(torch.equal(renderer.model._xyz[1], torch.tensor([2., 3., 4.])))


if __name__ == '__main__':
    unittest.main()
