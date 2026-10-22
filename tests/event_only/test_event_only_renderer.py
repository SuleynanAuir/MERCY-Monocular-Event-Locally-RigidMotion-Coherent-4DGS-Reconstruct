"""CPU schema/projection tests; optional CUDA pose-gradient smoke test."""

from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import torch
    from plyfile import PlyData, PlyElement
except (ImportError, OSError) as error:
    DEPENDENCY_ERROR = str(error)
else:
    DEPENDENCY_ERROR = None
    from utils.event_only_renderer import EventOnlyRenderer, load_g0, make_camera


@unittest.skipIf(DEPENDENCY_ERROR is not None, f"Numerical dependencies unavailable: {DEPENDENCY_ERROR}")
class EventOnlyRendererTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "g0.ply"

    def camera(self):
        return dict(width=640, height=480, fx=510.0, fy=520.0, cx=301.5, cy=229.0,
                    w2c=np.eye(4).tolist())

    def write_g0(self, rest_count=9, mutate=None):
        # Deliberately scrambled header order must not scramble SH coefficients.
        fields = ["x", "y", "z", "f_dc_2", "f_dc_0", "f_dc_1"]
        fields += [f"f_rest_{i}" for i in reversed(range(rest_count))]
        fields += ["opacity", "scale_2", "scale_0", "scale_1", "rot_3", "rot_0", "rot_2", "rot_1"]
        vertices = np.zeros(2, dtype=[(name, "f4") for name in fields] + [("custom_id", "i4")])
        vertices["x"] = [-0.1, 0.1]
        vertices["z"] = 2
        vertices["custom_id"] = [73, 91]
        for i in range(3):
            vertices[f"f_dc_{i}"] = i * 0.1
            vertices[f"scale_{i}"] = np.log((i + 1) * 0.03)
        for i in range(rest_count):
            vertices[f"f_rest_{i}"] = i * 0.01
        vertices["opacity"] = [0, np.log(3)]
        vertices["rot_0"] = [2, 1]
        vertices["rot_1"] = [0, 1]
        if mutate:
            mutate(vertices)
        auxiliary = np.array([(123,)], dtype=[("source_id", "i4")])
        ply = PlyData([PlyElement.describe(vertices, "vertex"),
                       PlyElement.describe(auxiliary, "metadata")],
                      text=False, comments=["preserve original G0 metadata"])
        ply.write(str(self.path))
        return vertices

    def test_complete_attributes_and_standard_sh_order_preserved(self):
        original = self.write_g0()
        g0 = load_g0(self.path)
        self.assertEqual(g0.count, 2)
        self.assertEqual(g0.property_names, original.dtype.names)
        np.testing.assert_array_equal(g0.raw_vertex, original)
        self.assertEqual(g0.ply.comments, ["preserve original G0 metadata"])
        self.assertEqual(int(g0.ply["metadata"].data["source_id"][0]), 123)
        self.assertEqual(g0.sh_degree, 1)
        self.assertEqual(tuple(g0.sh.shape), (2, 4, 3))
        torch.testing.assert_close(g0.sh[0, 1], torch.tensor([0.0, 0.03, 0.06]))
        torch.testing.assert_close(g0.sh[0, 3], torch.tensor([0.02, 0.05, 0.08]))
        torch.testing.assert_close(g0.rotation.norm(dim=1), torch.ones(2))
        torch.testing.assert_close(g0.scaling[0], torch.tensor([0.03, 0.06, 0.09]))
        torch.testing.assert_close(g0.opacity[:, 0], torch.tensor([0.5, 0.75]))
        for name in ("xyz", "rotation", "scaling", "opacity", "sh"):
            self.assertFalse(getattr(g0, name).requires_grad)

    def test_degree_zero_has_only_dc(self):
        self.write_g0(rest_count=0)
        g0 = load_g0(self.path)
        self.assertEqual(g0.sh_degree, 0)
        self.assertEqual(tuple(g0.sh.shape), (2, 1, 3))

    def test_saturated_opacity_logits_preserve_exact_zero_one_and_source(self):
        original = self.write_g0(mutate=lambda data: data['opacity'].__setitem__(slice(None), [np.inf, -np.inf]))
        g0 = load_g0(self.path)
        torch.testing.assert_close(g0.opacity[:, 0], torch.tensor([1., 0.]), atol=0, rtol=0)
        np.testing.assert_array_equal(g0.raw_vertex, original)
        self.write_g0(mutate=lambda data: data['opacity'].__setitem__(0, np.nan))
        with self.assertRaises(ValueError):
            load_g0(self.path)

    def test_invalid_quaternion_rejected_without_identity_substitution(self):
        for value in (0.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.write_g0(mutate=lambda data: data["rot_0"].__setitem__(0, value))
                with self.assertRaises(ValueError):
                    load_g0(self.path)

    def test_huge_finite_quaternion_is_normalized(self):
        self.write_g0(mutate=lambda data: data["rot_0"].__setitem__(0, 1e30))
        torch.testing.assert_close(load_g0(self.path).rotation[0], torch.tensor([1., 0., 0., 0.]))

    def test_incomplete_or_unsupported_sh_rejected(self):
        for count in (1, 6, 72):
            with self.subTest(rest_count=count):
                self.write_g0(rest_count=count)
                with self.assertRaises(ValueError):
                    load_g0(self.path)

    def test_nonfinite_geometry_and_overflowing_scale_rejected(self):
        for field, value in (("x", float("nan")), ("scale_0", 1000), ("scale_1", -1000)):
            with self.subTest(field=field):
                self.write_g0(mutate=lambda data: data[field].__setitem__(0, value))
                with self.assertRaises(ValueError):
                    load_g0(self.path)

    def test_camera_offcentre_boundary_projection_matches_opencv(self):
        spec = self.camera()
        camera = make_camera(spec)
        points = torch.tensor([[0., 0., 2., 1.], [.2, -.1, 2., 1.], [-.5, .4, 3., 1.]])
        clip = points @ camera.full_proj_transform
        ndc = clip[:, :2] / clip[:, 3:4]
        boundary = (ndc + 1) * torch.tensor([camera.width, camera.height]) / 2
        expected = points[:, :2] / points[:, 2:3] * torch.tensor([spec["fx"], spec["fy"]])
        expected += torch.tensor([spec["cx"], spec["cy"]])
        torch.testing.assert_close(boundary, expected)
        # Exact CUDA ndc2Pix convention: ((v + 1) * size - 1) / 2.
        raster_index = ((ndc + 1) * torch.tensor([camera.width, camera.height]) - 1) / 2
        torch.testing.assert_close(raster_index, expected - 0.5)
        self.assertAlmostEqual(float(camera.projection_matrix[0, 2]), 2 * spec["cx"] / 640 - 1)
        self.assertGreater(float(boundary[2, 1]), spec["cy"])

    def test_w2c_translation_rotation_and_camera_center(self):
        spec = self.camera()
        angle = 0.3
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        centre = np.array([0.2, -0.4, 1.0])
        w2c = np.eye(4); w2c[:3, :3] = rotation; w2c[:3, 3] = -rotation @ centre
        spec["w2c"] = w2c.tolist()
        camera = make_camera(spec)
        points = np.array([[0.0, 0.0, 4.0, 1.0], [-0.2, 0.1, 5.0, 1.0]])
        cv = (w2c @ points.T).T
        expected = cv[:, :2] / cv[:, 2:3] * [spec["fx"], spec["fy"]] + [spec["cx"], spec["cy"]]
        clip = torch.tensor(points, dtype=torch.float32) @ camera.full_proj_transform
        boundary = (clip[:, :2] / clip[:, 3:4] + 1) * torch.tensor([640, 480]) / 2
        np.testing.assert_allclose(boundary.numpy(), expected, atol=5e-5)
        np.testing.assert_allclose(camera.camera_center.numpy(), centre, atol=1e-6)

    def test_resize_scales_boundary_intrinsics_without_half_pixel_offset(self):
        spec = self.camera(); spec["height"] = 481
        camera = make_camera(spec, width=321)
        self.assertEqual(camera.height, round(481 * 321 / 640))
        self.assertAlmostEqual(camera.cx, spec["cx"] * 321 / 640)
        self.assertAlmostEqual(camera.cy, spec["cy"] * camera.height / 481)
        self.assertAlmostEqual(camera.fy, spec["fy"] * camera.height / 481)

    def test_camera_rejects_nonrigid_or_invalid_fields(self):
        for field, value in (("width", 0), ("height", 1.5), ("fx", -1), ("cy", float("nan"))):
            spec = self.camera(); spec[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                make_camera(spec)
        spec = self.camera(); spec["w2c"][0][0] = 2
        with self.assertRaises(ValueError):
            make_camera(spec)

    def test_camera_ignores_unrelated_paths_and_cpu_render_is_explicit(self):
        spec = self.camera(); spec["video"] = "/never/read/this.mp4"
        make_camera(spec)
        self.write_g0()
        renderer = EventOnlyRenderer(self.path, spec, device="cpu")
        self.assertNotIn("video", renderer.camera_spec)
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            renderer.render()

    def test_optional_cuda_forward_and_pose_gradients(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        try:
            from diff_gaussian_rasterization import GaussianRasterizer  # noqa: F401
        except ImportError:
            self.skipTest("Repository CUDA rasterizer is unavailable")
        self.write_g0(rest_count=0)
        spec = dict(width=64, height=48, fx=55., fy=56., cx=31.5, cy=23.5,
                    w2c=np.eye(4).tolist())
        renderer = EventOnlyRenderer(self.path, spec)
        xyz = renderer.g0.xyz.clone().requires_grad_()
        rotation = renderer.g0.rotation.clone().requires_grad_()
        result = renderer.render(xyz, rotation)
        self.assertEqual(tuple(result["rgb"].shape), (3, 48, 64))
        self.assertEqual(tuple(result["gray"].shape), (48, 64))
        self.assertGreater(float(result["alpha"].sum()), 0)
        weights = torch.arange(64, device="cuda")[None, :] ** 2
        (result["gray"] * weights).sum().backward()
        self.assertTrue(bool(torch.isfinite(xyz.grad).all()))
        self.assertTrue(bool(torch.isfinite(rotation.grad).all()))
        self.assertGreater(float(xyz.grad.abs().sum()), 0)
        self.assertGreater(float(rotation.grad.abs().sum()), 0)
        self.assertIsNone(renderer.g0.sh.grad)


if __name__ == "__main__":
    unittest.main()
