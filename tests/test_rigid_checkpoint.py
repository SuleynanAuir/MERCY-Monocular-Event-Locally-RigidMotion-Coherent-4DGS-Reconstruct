"""CPU tests for portable analytic checkpoints and exact rigid deformation."""

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from plyfile import PlyData, PlyElement
import torch

from utils.rigid_checkpoint import RigidRotation, export_rigid_bundle, load_rigid_bundle, sha256_file
from utils.rigid_motion import axis_angle, spin_profile


class RigidCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.ply"
        names = ["x", "y", "z", "opacity"]
        names += [f"{prefix}_{i}" for prefix, count in (("f_dc", 3), ("f_rest", 45), ("scale", 3), ("rot", 4))
                  for i in range(count)]
        self.vertices = np.zeros(8, dtype=[(name, "f4") for name in names])
        self.vertices["x"] = np.arange(8) / 8
        self.vertices["rot_0"] = 1
        self.vertices["opacity"][0] = np.inf  # A valid fully opaque raw logit.
        self.write_source()
        self.motion = RigidRotation([.2, -.3, .4], [1., 2., 3.], 60, 2, 6, "cw")
        self.output = self.root / "bundle"

    def write_source(self):
        PlyData([PlyElement.describe(self.vertices, "vertex")], text=False).write(str(self.source))

    def export(self):
        return export_rigid_bundle(self.source, self.output, self.motion)

    def test_safe_roundtrip_and_lossless_ply(self):
        payload = self.export()
        self.assertEqual(self.source.read_bytes(), (self.output / "point_cloud.ply").read_bytes())
        bundle = load_rigid_bundle(self.output / "deform.pth")
        self.assertEqual(bundle.gs_path, self.output / "point_cloud.ply")
        self.assertEqual(bundle.metadata, payload)
        self.assertEqual(bundle.metadata["training_steps"], 0)
        self.assertEqual(bundle.metadata["gaussians"]["sh_degree"], 3)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["format"], payload["format"])
        for t in (0, .123, 1, 1.999, 2, 3.789, 100):
            np.testing.assert_allclose(bundle.motion.state_at(t), self.motion.state_at(t), rtol=0, atol=0)

    def test_bundle_is_portable_and_can_use_explicit_renamed_ply(self):
        self.export()
        moved = self.root / "moved"
        self.output.rename(moved)
        renamed = self.root / "renamed.ply"
        (moved / "point_cloud.ply").rename(renamed)
        bundle = load_rigid_bundle(moved / "deform.pth", renamed)
        self.assertEqual(bundle.gs_path, renamed)

    def test_reject_mismatched_ply(self):
        self.export()
        with (self.output / "point_cloud.ply").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            load_rigid_bundle(self.output / "deform.pth")

    def test_reject_native_and_unknown_version(self):
        payload = self.export()
        path = self.root / "wrong.pth"
        for data in ({"network.weight": torch.zeros(1)}, dict(payload, version=999)):
            torch.save(data, path)
            with self.assertRaisesRegex(ValueError, "Not a supported analytic"):
                load_rigid_bundle(path)

    def test_reject_wrong_conventions_and_metadata(self):
        payload = self.export()
        path = self.output / "wrong.pth"
        modified = [dict(payload, conventions=dict(payload["conventions"], time_unit="normalized")),
                    dict(payload, gaussians=dict(payload["gaussians"], filename="../source.ply")),
                    dict(payload, gaussians=dict(payload["gaussians"], count=100))]
        for data in modified:
            torch.save(data, path)
            with self.assertRaises(ValueError):
                load_rigid_bundle(path)

    def test_does_not_overwrite_existing_directory(self):
        self.export()
        before = sha256_file(self.output / "deform.pth")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(sha256_file(self.output / "deform.pth"), before)

    def test_reject_directional_sh_before_creating_output(self):
        self.vertices["f_rest_0"][0] = .1
        self.write_source()
        with self.assertRaisesRegex(ValueError, "higher-order SH"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_reject_invalid_geometry_but_accept_infinite_opacity_logits(self):
        self.export()  # +inf opacity was accepted, and is preserved.
        for field, value in (("x", np.nan), ("opacity", np.nan), ("scale_0", 1000)):
            previous = self.vertices[field][0]
            self.vertices[field][0] = value
            self.write_source()
            with self.assertRaises(ValueError):
                export_rigid_bundle(self.source, self.root / f"invalid-{field}", self.motion)
            self.vertices[field][0] = previous

    def test_step_matches_analytic_formula_at_arbitrary_times(self):
        self.export()
        motion = load_rigid_bundle(self.output / "deform.pth").motion
        points = torch.tensor([[.1, .2, .3], [.4, -.5, .6], [-.7, .8, -.9]], dtype=torch.float64)
        original = points.clone()
        for time in (0, .125, 1.17, 2, 5.99, 16):
            result = motion.step(points, time)
            matrix, quaternion = axis_angle(self.motion.axis, -float(spin_profile(time, 60, 2)[0]))
            center = points.new_tensor(self.motion.center)
            expected = (points - center) @ points.new_tensor(matrix.tolist()).T + center
            torch.testing.assert_close(points + result["d_xyz"], expected, rtol=0, atol=1e-14)
            torch.testing.assert_close(result["d_rotation_bias"], points.new_tensor(quaternion.tolist()).expand(3, 4))
            torch.testing.assert_close(torch.cdist(expected, expected), torch.cdist(points, points))
            self.assertEqual(result["d_scaling"], 0)
            self.assertEqual(result["d_rotation"], 0)
            self.assertIsNone(result["d_color"])
            self.assertIsNone(result["d_opacity"])
        self.assertTrue(torch.equal(points, original))

    def test_clockwise_orientation_and_seconds_not_normalized_time(self):
        motion = RigidRotation([0, 0, 0], [0, 0, 1], rpm=60, ramp_seconds=0)
        point = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(point + motion.step(point, .25)["d_xyz"], point.new_tensor([[0., -1., 0.]]))
        self.assertAlmostEqual(motion.state_at(1)[0], -2 * math.pi)

    def test_start_at_rest_and_continue_after_duration(self):
        self.assertEqual(self.motion.state_at(0), (0, 0, 0))
        self.assertAlmostEqual(self.motion.state_at(1)[1], -math.pi)
        self.assertAlmostEqual(self.motion.state_at(2)[1], -2 * math.pi)
        self.assertAlmostEqual(self.motion.state_at(10)[0], -18 * math.pi)

    def test_bad_time_shape_and_parameters(self):
        for time in (-1, float("nan"), [0, 1]):
            with self.assertRaises(ValueError):
                self.motion.state_at(time)
        for kwargs in ({"center": [0, 1]}, {"axis": [0, 0, 0]}, {"rpm": -1},
                       {"ramp_seconds": -1}, {"duration_seconds": 0}, {"direction": "invalid"}):
            args = dict(center=[0, 0, 0], axis=[0, 0, 1])
            args.update(kwargs)
            with self.assertRaises(ValueError):
                RigidRotation(**args)
        with self.assertRaises(ValueError):
            self.motion.step(torch.zeros(3), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA environment required")
    def test_gpu_step_matches_original_render_math_exactly(self):
        self.export()
        motion = load_rigid_bundle(self.output / "deform.pth").motion
        points = torch.rand((100, 3), device="cuda")
        center = torch.tensor(motion.center, device="cuda", dtype=torch.float32)
        for time in (0, .125, 1.17, 2, 5.99):
            rotation, quaternion = axis_angle(motion.axis, -float(spin_profile(time, 60, 2)[0]))
            expected = (points - center) @ torch.tensor(rotation, device="cuda", dtype=torch.float32).T + center
            result = motion.step(points, time)
            self.assertTrue(torch.equal(result["d_xyz"], expected - points))
            self.assertTrue(torch.equal(result["d_rotation_bias"],
                                        torch.tensor(quaternion, device="cuda", dtype=torch.float32).expand(100, 4)))


if __name__ == "__main__":
    unittest.main()
