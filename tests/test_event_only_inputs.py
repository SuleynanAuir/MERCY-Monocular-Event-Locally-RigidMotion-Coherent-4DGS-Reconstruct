"""Synthetic standard-library tests; no project data or prior results are read."""

import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from utils.event_only_inputs import (
    CAMERA_CONVENTION, InputValidationError, inspect_gaussian_ply_header,
    parse_camera, preflight_event_only_inputs, sha256_file,
)


def cameras():
    intrinsics = {
        "width": 640, "height": 640, "fx": 888.8888549804688,
        "fy": 888.8888549804688, "cx": 320, "cy": 320,
        "K": [[888.8888549804688, 0, 320], [0, 888.8888549804688, 320], [0, 0, 1]],
        "coordinate_convention": {
            "image_axes": "u right, v down", "camera_axes": "X right, Y down, Z forward",
            "pixel_origin": "top-left image boundary", "top_left_pixel_center": [0.5, 0.5],
        },
    }
    extrinsics = {key: intrinsics[key] for key in ("width", "height", "fx", "fy", "cx", "cy")}
    extrinsics.update({
        "R": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], "T": [1, 2, 3],
        "camera_convention": CAMERA_CONVENTION, "post_render_transform": "none",
        "image_mirror": False, "mirror_axis": "none",
    })
    return intrinsics, extrinsics


def ply_header(rest=45, file_format="binary_little_endian"):
    names = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{index}" for index in range(rest)]
    return (f"ply\nformat {file_format} 1.0\nelement vertex 1\n" +
            "".join(f"property float {name}\n" for name in names) + "end_header\n").encode("ascii")


class EventOnlyInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.paths = [root / name for name in ("g0.ply", "raw-events.bin", "K.json", "camera.json")]
        # Header tests intentionally do not assert that this artificial body is
        # a complete Gaussian payload; preflight is not a body/semantic audit.
        self.paths[0].write_bytes(ply_header() + b"\xff\x00\xfe\x80")
        self.paths[1].write_bytes(b"synthetic opaque raw events\x00\xff")
        self.intrinsics, self.extrinsics = cameras()
        self.write_camera()

    def write_camera(self):
        self.paths[2].write_text(json.dumps(self.intrinsics), encoding="utf-8")
        self.paths[3].write_text(json.dumps(self.extrinsics), encoding="utf-8")

    def preflight(self, **kwargs):
        return preflight_event_only_inputs(*self.paths, g0_time=kwargs.pop("g0_time", 0),
                                          time_scale=kwargs.pop("time_scale", 1e-6), **kwargs)

    def test_supported_camera_has_explicit_boundary_coordinates_and_transposed_rotation(self):
        result = self.preflight()
        self.assertEqual(result["scope"], "preflight_only")
        self.assertFalse(result["source_authenticity_verified"])
        self.assertEqual(result["allowed_modalities"], ["g0", "events", "camera"])
        self.assertEqual(result["g0_time"], 0)
        self.assertEqual(result["time_scale"], 1e-6)
        camera = result["camera"]
        self.assertEqual(set(camera), {"width", "height", "fx", "fy", "cx", "cy", "pixel_center",
                                       "R_stored_transposed", "T", "w2c"})
        self.assertEqual(camera["pixel_center"], "boundary")
        self.assertEqual(camera["cx"], 320, "Do not silently shift boundary intrinsics by half a pixel")
        self.assertEqual(camera["R_stored_transposed"], self.extrinsics["R"])
        self.assertEqual(camera["w2c"], [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
        self.assertEqual(set(result["input_sha256"]), {"g0", "events", "intrinsics", "extrinsics"})
        for role, path in zip(("g0", "events", "intrinsics", "extrinsics"), self.paths):
            self.assertEqual(result["input_sha256"][role], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_embedded_paths_projections_and_prior_metrics_are_not_followed_or_copied(self):
        malicious = str(Path(self.directory.name) / "MUST_NOT_OPEN-secret-or-blend")
        extras = {"source_blend": malicious, "source_t0_image": malicious,
                  "source_g0": malicious, "projection_matrix": malicious,
                  "full_proj_transform": malicious, "metrics": {"reference_path": malicious}}
        self.intrinsics.update(extras)
        self.extrinsics.update(extras)
        self.write_camera()
        opened = []
        original = Path.open
        def guarded(path, *args, **kwargs):
            self.assertIn(path.resolve(), [value.resolve() for value in self.paths])
            opened.append(path.resolve())
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, "open", guarded):
            result = self.preflight()
        self.assertEqual(set(opened), set(path.resolve() for path in self.paths))
        self.assertNotIn(malicious, json.dumps(result))
        for key in extras:
            self.assertNotIn(key, result["camera"])

    def test_sha256_streams_a_large_explicit_input(self):
        content = b"abcd" * (1024 * 1024)
        self.paths[1].write_bytes(content)
        self.assertEqual(sha256_file(self.paths[1]), hashlib.sha256(content).hexdigest())

    def test_intrinsics_disagreeing_between_the_two_files_fail_closed(self):
        for key, value in (("fx", 900), ("cy", 319.5), ("width", 800)):
            with self.subTest(key=key):
                intrinsics, extrinsics = cameras()
                extrinsics[key] = value
                with self.assertRaisesRegex(InputValidationError, "inconsistent intrinsics"):
                    parse_camera(intrinsics, extrinsics)

    def test_scalar_and_matrix_intrinsics_must_agree_in_each_file(self):
        for role in ("intrinsics", "extrinsics"):
            with self.subTest(role=role):
                intrinsics, extrinsics = cameras()
                target = intrinsics if role == "intrinsics" else extrinsics
                target["K"] = copy.deepcopy(intrinsics["K"])
                target["K"][0][0] = 123
                with self.assertRaisesRegex(InputValidationError, "disagrees"):
                    parse_camera(intrinsics, extrinsics)

    def test_nonorthogonal_reflected_and_nonfinite_rotations_are_rejected(self):
        for rotation in ([[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                          [[1, .01, 0], [0, 1, 0], [0, 0, 1]],
                          [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                          [[math.nan, 0, 0], [0, 1, 0], [0, 0, 1]]]):
            with self.subTest(rotation=rotation):
                self.extrinsics["R"] = rotation
                with self.assertRaises(InputValidationError):
                    parse_camera(self.intrinsics, self.extrinsics)

    def test_skew_distortion_mirroring_and_post_render_transforms_are_rejected(self):
        for changes in ({"skew": 1e-12}, {"distortion": [.1, 0, 0, 0]}, {"k1": .1},
                        {"distortion_model": "opencv"}, {"image_mirror": True},
                        {"mirror_axis": "x"}, {"flip_x": True}, {"post_render_transform": "rotate90"}):
            with self.subTest(changes=changes):
                intrinsics, extrinsics = cameras()
                extrinsics.update(changes)
                with self.assertRaises(InputValidationError):
                    parse_camera(intrinsics, extrinsics)
        self.intrinsics["K"][0][1] = 1e-15
        with self.assertRaisesRegex(InputValidationError, "zero skew"):
            parse_camera(self.intrinsics, self.extrinsics)

    def test_camera_conventions_are_explicit_and_not_guessed(self):
        for key, value in (("camera_convention", "opencv"), ("post_render_transform", None),
                           ("image_mirror", 0), ("mirror_axis", None)):
            with self.subTest(key=key):
                intrinsics, extrinsics = cameras()
                extrinsics[key] = value
                with self.assertRaises(InputValidationError):
                    parse_camera(intrinsics, extrinsics)
        del self.extrinsics["camera_convention"]
        with self.assertRaises(InputValidationError):
            parse_camera(self.intrinsics, self.extrinsics)
        intrinsics, extrinsics = cameras()
        intrinsics["coordinate_convention"]["pixel_origin"] = "top-left pixel center"
        with self.assertRaises(InputValidationError):
            parse_camera(intrinsics, extrinsics)

    def test_nonpositive_focal_lengths_and_bad_dimensions_are_rejected(self):
        for key, value in (("fx", 0), ("fy", -1), ("width", True), ("height", 640.0), ("cx", "320")):
            with self.subTest(key=key):
                intrinsics, extrinsics = cameras()
                intrinsics[key] = value
                with self.assertRaises(InputValidationError):
                    parse_camera(intrinsics, extrinsics)

    def test_only_the_three_allowed_modalities_are_accepted(self):
        for modalities in (("g0", "events", "video"), ("g0", "events", "camera", "blender"),
                           ("g0", "g0", "camera"), "g0,events,camera", None):
            with self.subTest(modalities=modalities):
                with self.assertRaises(InputValidationError):
                    self.preflight(allowed_modalities=modalities)

    def test_time_and_scale_are_explicit_finite_and_scale_is_positive(self):
        with self.assertRaises(TypeError):
            preflight_event_only_inputs(*self.paths)
        for kwargs in ({"g0_time": math.nan}, {"g0_time": math.inf}, {"g0_time": True},
                       {"time_scale": math.nan}, {"time_scale": 0}, {"time_scale": -1},
                       {"time_scale": "1e-6"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(InputValidationError):
                    self.preflight(**kwargs)

    def test_four_files_must_be_regular_resolved_and_distinct(self):
        duplicate = [self.paths[0], self.paths[0], self.paths[2], self.paths[3]]
        with self.assertRaisesRegex(InputValidationError, "distinct"):
            preflight_event_only_inputs(*duplicate, g0_time=0, time_scale=1)
        self.paths[1].unlink()
        self.paths[1].symlink_to(self.paths[0])
        with self.assertRaisesRegex(InputValidationError, "distinct"):
            self.preflight()
        self.paths[1].unlink()
        os.link(self.paths[0], self.paths[1])
        with self.assertRaisesRegex(InputValidationError, "distinct"):
            self.preflight()
        self.paths[1].unlink()
        self.paths[1].mkdir()
        with self.assertRaisesRegex(InputValidationError, "regular"):
            self.preflight()
        self.paths[1].rmdir()
        with self.assertRaisesRegex(InputValidationError, "accessible"):
            self.preflight()

    def test_missing_fields_duplicate_json_and_nonfinite_json_are_rejected(self):
        del self.extrinsics["T"]
        self.write_camera()
        with self.assertRaisesRegex(InputValidationError, "extrinsic"):
            self.preflight()
        self.intrinsics, self.extrinsics = cameras()
        self.write_camera()
        self.paths[2].write_text('{"width":640,"width":640}', encoding="utf-8")
        with self.assertRaisesRegex(InputValidationError, "Duplicate JSON"):
            self.preflight()
        self.paths[2].write_text('{"fx":NaN}', encoding="utf-8")
        with self.assertRaisesRegex(InputValidationError, "Non-finite JSON"):
            self.preflight()

    def test_ply_accepts_complete_rgb_sh_bands_and_all_supported_body_encodings(self):
        for count, degree in ((0, 0), (9, 1), (24, 2), (45, 3), (72, 4)):
            for encoding in ("ascii", "binary_little_endian", "binary_big_endian"):
                with self.subTest(count=count, encoding=encoding):
                    self.paths[0].write_bytes(ply_header(count, encoding) + b"\xff invalid body deliberately ignored")
                    schema = inspect_gaussian_ply_header(self.paths[0])
                    self.assertEqual(schema["sh_degree"], degree)
                    self.assertEqual(schema["format"], encoding)
                    self.assertEqual(schema["vertex_count"], 1)

    def test_ply_header_inspection_never_reads_the_body(self):
        class HeaderOnly(io.BytesIO):
            def read(self, *args):
                raise AssertionError("Header inspection must not read the body")
            def readline(self, *args):
                if self.tell() >= len(ply_header()):
                    raise AssertionError("Header inspection read past end_header")
                return super().readline(*args)
        with mock.patch.object(Path, "open", return_value=HeaderOnly(ply_header() + b"\xff body")):
            self.assertEqual(inspect_gaussian_ply_header(self.paths[0])["sh_degree"], 3)

    def test_ply_missing_nonfloat_duplicate_and_invalid_sh_properties_fail(self):
        variants = [
            ply_header().replace(b"property float opacity\n", b""),
            ply_header().replace(b"property float x\n", b"property int x\n"),
            ply_header().replace(b"property float x\n", b"property float x\nproperty float x\n"),
            ply_header(8), ply_header(12),
            ply_header(9).replace(b"f_rest_8\n", b"f_rest_9\n"),
            ply_header(9).replace(b"f_rest_8\n", b"f_rest_bad\n"),
            ply_header().replace(b"end_header\n", b"property float f_dc_3\nend_header\n"),
            ply_header().replace(b"element vertex 1", b"element vertex 0"),
            ply_header().replace(b"end_header\n", b""),
        ]
        for index, header in enumerate(variants):
            with self.subTest(index=index):
                self.paths[0].write_bytes(header)
                with self.assertRaises(InputValidationError):
                    inspect_gaussian_ply_header(self.paths[0])

    def test_ply_header_has_bounded_size_and_ascii_directives(self):
        for content in (b"ply\ncomment " + b"x" * 9000 + b"\n", b"ply\n\xff\n",
                        b"not-ply\n", b"ply\nformat ascii 2.0\nend_header\n"):
            self.paths[0].write_bytes(content)
            with self.assertRaises(InputValidationError):
                inspect_gaussian_ply_header(self.paths[0])


if __name__ == "__main__":
    unittest.main()
