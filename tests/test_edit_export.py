"""Export regressions; requires the installed SC-GS CUDA environment, not a GUI."""

import math
from pathlib import Path
import tempfile
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA environment required")
class EditExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from edit_gui import make_edited_gaussians
        from scene.gaussian_model import GaussianModel
        cls.export = staticmethod(make_edited_gaussians)
        cls.model_type = GaussianModel

    def setUp(self):
        self.model = self.model_type(3, with_motion_mask=False)
        self.model.active_sh_degree = 3
        shapes = {"_xyz": (8, 3), "_features_dc": (8, 1, 3),
                  "_features_rest": (8, 15, 3), "_scaling": (8, 3), "_opacity": (8, 1)}
        for name, shape in shapes.items():
            values = torch.arange(math.prod(shape), device="cuda", dtype=torch.float32).reshape(shape) / 100
            setattr(self.model, name, torch.nn.Parameter(values))
        self.model._rotation = torch.nn.Parameter(torch.tensor([1., 0., 0., 0.], device="cuda").repeat(8, 1))

    def test_unedited_export_is_independent_and_unchanged(self):
        exported = self.export(self.model)
        for name in ("_xyz", "_rotation", "_features_dc", "_features_rest"):
            original, copied = getattr(self.model, name), getattr(exported, name)
            self.assertTrue(torch.equal(original, copied))
            self.assertNotEqual(original.data_ptr(), copied.data_ptr())

    def test_translation_does_not_zero_rotation(self):
        exported = self.export(self.model, {"d_xyz": .1, "d_rotation": 0.})
        torch.testing.assert_close(exported.get_xyz, self.model.get_xyz + .1)
        torch.testing.assert_close(exported.get_rotation, self.model.get_rotation)

    def test_rotation_bias_is_composed_in_renderer_order(self):
        half = math.sqrt(.5)
        self.model._rotation = torch.nn.Parameter(torch.tensor([half, half, 0., 0.], device="cuda").repeat(8, 1))
        bias = torch.tensor([half, 0., half, 0.], device="cuda").repeat(8, 1)
        exported = self.export(self.model, {"d_xyz": 0., "d_rotation": 0., "d_rotation_bias": bias})
        expected = torch.tensor([.5, .5, .5, -.5], device="cuda").repeat(8, 1)
        torch.testing.assert_close(exported.get_rotation, expected)

    def test_export_reload_preserves_third_order_colors(self):
        from utils.ply_utils import gaussian_degree
        exported = self.export(self.model, {"d_xyz": .1, "d_rotation": 0.})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edited.ply"
            exported.save_ply(str(path))
            self.assertEqual(gaussian_degree(path), 3)
            reloaded = self.model_type(3, with_motion_mask=False)
            reloaded.load_ply(str(path))
            torch.testing.assert_close(reloaded.get_xyz, exported.get_xyz)
            torch.testing.assert_close(reloaded.get_rotation, exported.get_rotation)
            torch.testing.assert_close(reloaded.get_features, exported.get_features)


if __name__ == "__main__":
    unittest.main()
