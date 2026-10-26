import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from plyfile import PlyData, PlyElement

from scripts.export.export_gaussian_sequence import BakedRotor, export_sequence
from utils.gaussian_sequence import GaussianSequence
from utils.rigid_checkpoint import RigidRotation, export_rigid_bundle, sha256_file


class GaussianSequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        names = ["x", "y", "z", "nx", "ny", "nz", "opacity"]
        names += [f"{prefix}_{i}" for prefix, count in (("f_dc", 3), ("scale", 3), ("rot", 4)) for i in range(count)]
        self.vertices = np.zeros(3, dtype=[(name, "f4") for name in names])
        self.vertices["x"] = [1, 0, -1]
        self.vertices["y"] = [0, 1, 0]
        self.vertices["nx"] = 1
        self.vertices["rot_0"] = 1
        self.vertices["opacity"][0] = np.inf
        self.vertices["f_dc_0"] = [.1, .2, .3]
        self.source = self.root / "source.ply"
        PlyData([PlyElement.describe(self.vertices, "vertex")]).write(self.source)
        self.motion = RigidRotation([0, 0, 0], [0, 0, 1], 60, 0, .1, "cw")
        self.bundle = self.root / "bundle"
        export_rigid_bundle(self.source, self.bundle, self.motion)

    def export(self):
        return export_sequence(self.bundle / "deform.pth", self.root / "sequence", fps=20, device="cpu")

    def test_baked_positions_rotations_and_normals(self):
        baker = BakedRotor(self.source, self.motion)
        vertices = baker.frame(.25)["vertex"].data
        np.testing.assert_allclose(vertices["x"], [0, 1, 0], atol=1e-7)
        np.testing.assert_allclose(vertices["y"], [-1, 0, 1], atol=1e-7)
        np.testing.assert_allclose(vertices["nx"], 0, atol=1e-7)
        np.testing.assert_allclose(vertices["ny"], -1, atol=1e-7)
        np.testing.assert_allclose(vertices["rot_0"], np.sqrt(.5), atol=1e-7)
        np.testing.assert_allclose(vertices["rot_3"], -np.sqrt(.5), atol=1e-7)

    def test_quaternion_composition_order(self):
        self.vertices["rot_0"] = np.sqrt(.5)
        self.vertices["rot_1"] = np.sqrt(.5)
        PlyData([PlyElement.describe(self.vertices, "vertex")]).write(self.source)
        frame = BakedRotor(self.source, self.motion).frame(.25)["vertex"]
        actual = np.column_stack([frame[f"rot_{i}"] for i in range(4)])
        np.testing.assert_allclose(actual, np.tile([.5, .5, -.5, -.5], (3, 1)), atol=1e-7)

    def test_all_other_properties_and_source_preserved(self):
        before = sha256_file(self.source)
        frame = BakedRotor(self.source, self.motion).frame(.123)["vertex"].data
        for name in self.vertices.dtype.names:
            if name.startswith(("scale_", "f_dc_")) or name == "opacity":
                np.testing.assert_array_equal(frame[name], self.vertices[name])
        self.assertEqual(before, sha256_file(self.source))

    def test_zip_contains_complete_independently_loadable_plys(self):
        path = self.export()
        with zipfile.ZipFile(path) as archive:
            self.assertIn("load_sequence.py", archive.namelist())
            self.assertNotIn("deform.pth", archive.namelist())
            for i in range(2):
                frame = PlyData.read(io.BytesIO(archive.read(f"frames/frame_{i:04d}.ply")))
                self.assertEqual(len(frame["vertex"]), 3)
                self.assertEqual(frame["vertex"].data.dtype.names, self.vertices.dtype.names)

    def test_playback_does_not_need_original_ply_or_checkpoint(self):
        path = self.export()
        self.source.unlink()
        (self.bundle / "deform.pth").unlink()
        (self.bundle / "point_cloud.ply").unlink()
        with GaussianSequence(path) as sequence:
            self.assertEqual(len(sequence), 2)
            self.assertFalse(sequence.metadata["requires_deform_checkpoint_for_playback"])
            self.assertNotEqual(sequence.frame(0)["vertex"]["y"][0], sequence.frame(1)["vertex"]["y"][0])
            self.assertEqual(sequence.frames[1]["time_seconds"], .05)

    def test_extracted_directory_playback(self):
        path = self.export()
        folder = self.root / "extracted"
        with zipfile.ZipFile(path) as archive:
            archive.extractall(folder)  # Only this test's own generated archive.
        with GaussianSequence(folder) as sequence:
            self.assertEqual(len(sequence.frame(1)["vertex"]), 3)

    def test_bad_index_and_checksum_rejected(self):
        path = self.export()
        with GaussianSequence(path) as sequence:
            for index in (-1, 2, .5):
                with self.assertRaises(IndexError):
                    sequence.frame(index)
            sequence.frames[0]["sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "checksum"):
                sequence.frame(0)

    def test_refuses_overwrite_and_bad_rate(self):
        path = self.export()
        before = sha256_file(path)
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(sha256_file(path), before)
        with self.assertRaises(ValueError):
            export_sequence(self.bundle / "deform.pth", self.root / "bad", fps=0)

    def test_rejects_path_traversal_in_manifest(self):
        path = self.export()
        with GaussianSequence(path) as sequence:
            metadata = sequence.metadata
        metadata["frames"][0]["file"] = "../source.ply"
        bad = self.root / "bad.zip"
        with zipfile.ZipFile(bad, "w") as archive:
            archive.writestr("sequence.json", json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            GaussianSequence(bad)

    def test_compact_tensor_package_recreates_identical_ply_bytes(self):
        from scripts.export.pack_gaussian_sequence import pack_sequence
        source = self.export()
        destination = pack_sequence(source, self.root / "moving.npz")
        with GaussianSequence(source) as reference, GaussianSequence(destination) as packed:
            self.assertEqual(packed.poses.shape, (3, 2, 10))  # Nonzero normals retained too.
            for index in range(len(reference)):
                self.assertEqual(reference.frame_bytes(index), packed.frame_bytes(index))
            np.testing.assert_array_equal(packed.frame(1)["vertex"].data, reference.frame(1)["vertex"].data)
        with self.assertRaises(FileExistsError):
            pack_sequence(source, destination)

    def test_compact_tensor_package_no_normals_needs_only_seven_channels(self):
        from scripts.export.pack_gaussian_sequence import pack_sequence
        self.vertices["nx"] = 0
        PlyData([PlyElement.describe(self.vertices, "vertex")]).write(self.source)
        other = self.root / "zero-normal-bundle"
        export_rigid_bundle(self.source, other, self.motion)
        path = export_sequence(other / "deform.pth", self.root / "zero-normal-sequence", fps=20)
        packed_path = pack_sequence(path, self.root / "seven.npz")
        with GaussianSequence(packed_path) as packed:
            self.assertEqual(packed.poses.shape, (3, 2, 7))
            self.assertEqual(len(packed.frame(1)["vertex"]), 3)


if __name__ == "__main__":
    unittest.main()
