"""Synthetic CPU tests for the event-only checkpoint and sparse exports."""

import copy
import hashlib
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
    from utils.event_only_checkpoint import FORMAT, load_event_only_checkpoint, save_event_only_checkpoint
    from utils.event_only_motion import EventOnlyMotion, HARD_RIGID_GROUP_KIND
    from utils.event_only_renderer import load_g0
    from scripts.events.render_event_only_sequence import requested_ply_times, sequence_times, write_posed_ply


@unittest.skipIf(DEPENDENCY_ERROR is not None, f"Numerical dependencies unavailable: {DEPENDENCY_ERROR}")
class EventOnlyCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.g0 = self.directory / "original.ply"
        self.checkpoint = self.directory / "deform.pth"
        fields = ["x", "y", "z", "opacity"]
        fields += [f"f_dc_{i}" for i in range(3)]
        fields += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
        vertices = np.zeros(3, dtype=[(name, "f4") for name in fields] + [("unchanged_label", "i4")])
        vertices["x"] = [0., 1., .5]; vertices["z"] = 1; vertices["rot_0"] = 1
        vertices["unchanged_label"] = [8, 12, 91]
        for i in range(3):
            vertices[f"scale_{i}"] = -3.0
        PlyData([PlyElement.describe(vertices, "vertex")],
                comments=["retain G0 properties"], obj_info=["source metadata"]).write(str(self.g0))
        self.graph = {
            "centers": torch.tensor([[0., 0., 1.], [1., 0., 1.]]),
            "knn_indices": torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int64),
            "knn_weights": torch.tensor([[1., 0.], [1., 0.], [.5, .5]]),
            "edges": torch.tensor([[0, 1]], dtype=torch.int64), "scene_scale": 1.0,
            "metadata": {"kind": "g0_only_latent_deformation_graph", "version": 1,
                         "node_source": "explicit_G0_positions_only", "blender_parts": False},
        }
        self.motion = EventOnlyMotion(self.graph, [0., .5, 1.])
        with torch.no_grad():
            self.motion.translations[0, :, 0] = .1
            self.motion.rotation_vectors[1, :, 2] = .2
        self.camera = {"width": 64, "height": 48, "fx": 50., "fy": 51., "cx": 32., "cy": 24.,
                       "w2c": np.eye(4).tolist(), "pixel_center": "boundary"}
        self.provenance = {
            "allowed_modalities": ["g0", "events", "camera"],
            "input_sha256": {"g0": hashlib.sha256(self.g0.read_bytes()).hexdigest(),
                             "events": "1" * 64, "intrinsics": "2" * 64, "extrinsics": "3" * 64},
            # An obsolete descriptive path is never followed by the loader.
            "input_files": {"g0": "/nonexistent/descriptive/path.ply"},
        }

    def save(self):
        return save_event_only_checkpoint(self.checkpoint, self.motion, self.graph, self.camera,
                                         self.provenance, {"iterations": 3, "event_passed": False})

    def malformed(self, mutation):
        payload = self.save()
        mutation(payload)
        path = self.directory / "malformed.pth"
        torch.save(payload, path)
        return path

    def use_hard_groups(self):
        self.graph.update(knn_indices=torch.tensor([[0], [1], [0]], dtype=torch.int64),
                          knn_weights=torch.ones(3, 1), edges=torch.empty((0, 2), dtype=torch.int64))
        self.graph["metadata"] = {
            "kind": HARD_RIGID_GROUP_KIND, "version": 1,
            "binding_source": "explicit_G0_raw_events_and_camera",
            "node_source": "G0_group_member_centers", "blender_parts": False,
            "input_sha256": dict(self.provenance["input_sha256"]),
        }
        self.motion = EventOnlyMotion(self.graph, [0., .5, 1.])
        with torch.no_grad():
            self.motion.rotation_vectors[1, 0, 2] = .3
            self.motion.translations[0, 1, 1] = .2

    def test_hard_group_weights_only_roundtrip_retains_binding_and_honest_metadata(self):
        self.use_hard_groups()
        self.save()
        safe = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(safe["format"], FORMAT)
        restored, _, payload = load_event_only_checkpoint(self.checkpoint, self.g0, device="cpu")
        self.assertEqual(restored.metadata, self.motion.metadata)
        self.assertEqual(restored.metadata["kind"], HARD_RIGID_GROUP_KIND)
        self.assertEqual(payload["graph"]["metadata"]["input_sha256"], payload["input_sha256"])
        self.assertEqual(tuple(restored.knn_indices.shape), (3, 1))
        self.assertEqual(tuple(restored.edges.shape), (0, 2))
        for key, expected in self.motion.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], expected, atol=0, rtol=0)
        xyz, q = load_g0(self.g0).xyz, torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)
        for time in (0., .35, 1.):
            for expected, actual in zip(self.motion.deform(xyz, q, time), restored.deform(xyz, q, time)):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_hard_group_load_rejects_each_mismatched_input_hash(self):
        self.use_hard_groups()
        payload = self.save()
        for name in ("g0", "events", "intrinsics", "extrinsics"):
            corrupted = copy.deepcopy(payload)
            corrupted["graph"]["metadata"]["input_sha256"][name] = "a" * 64
            path = self.directory / f"wrong-{name}.pth"
            torch.save(corrupted, path)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "SHA256"):
                load_event_only_checkpoint(path, self.g0, device="cpu")

    def test_hard_group_load_revalidates_structure_and_source(self):
        self.use_hard_groups()
        payload = self.save()
        cases = {
            "missing_members": lambda value: value["state_dict"]["knn_indices"].zero_(),
            "soft_weights": lambda value: value["state_dict"]["knn_weights"].fill_(.999999),
            "wrong_source": lambda value: value["graph"]["metadata"].update(binding_source="G0_only"),
            "g0_only_node_source": lambda value: value["graph"]["metadata"].update(
                node_source="explicit_G0_positions_only"),
            "blender_parts": lambda value: value["graph"]["metadata"].update(blender_parts=True),
        }
        for name, mutation in cases.items():
            corrupted = copy.deepcopy(payload)
            mutation(corrupted)
            path = self.directory / f"invalid-{name}.pth"
            torch.save(corrupted, path)
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_event_only_checkpoint(path, self.g0, device="cpu")

    def test_hard_group_save_rejects_relabeling_as_g0_only(self):
        self.use_hard_groups()
        self.graph["metadata"] = {"kind": "g0_only_latent_deformation_graph", "version": 1,
                                  "node_source": "explicit_G0_positions_only", "blender_parts": False}
        with self.assertRaisesRegex(ValueError, "mode.*provenance"):
            self.save()
        self.assertFalse(self.checkpoint.exists())

    def test_hard_group_save_rejects_relabeling_existing_legacy_motion(self):
        self.use_hard_groups()
        legacy = copy.deepcopy(self.graph)
        legacy.pop("metadata")
        self.motion = EventOnlyMotion(legacy, [0., .5, 1.])
        with self.assertRaisesRegex(ValueError, "mode.*provenance"):
            self.save()

    def test_hard_group_save_rejects_different_provenance_and_near_one_weights(self):
        self.use_hard_groups()
        self.provenance["input_sha256"]["events"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.save()
        self.provenance["input_sha256"]["events"] = "1" * 64
        self.graph["knn_weights"] = torch.full((3, 1), 1. + 1e-9, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.save()
        self.assertFalse(self.checkpoint.exists())

    def test_weights_only_roundtrip_and_motion(self):
        self.save()
        safe = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(safe["format"], FORMAT)
        restored, camera, payload = load_event_only_checkpoint(self.checkpoint, self.g0, device="cpu")
        self.assertEqual(camera, self.camera)
        self.assertEqual(payload["gaussian_count"], 3)
        for name, original in self.motion.state_dict().items():
            torch.testing.assert_close(original, restored.state_dict()[name], rtol=0, atol=0)
        xyz = torch.tensor([[0., 0., 1.], [1., 0., 1.], [.5, 0., 1.]])
        q = torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)
        for time in (0., .35, 1.):
            expected = self.motion.deform(xyz, q, time)
            actual = restored.deform(xyz, q, time)
            for one, two in zip(expected, actual):
                torch.testing.assert_close(one, two)

    def test_save_never_overwrites_checkpoint(self):
        self.save()
        before = self.checkpoint.read_bytes()
        with self.assertRaises(FileExistsError):
            self.save()
        self.assertEqual(before, self.checkpoint.read_bytes())

    def test_g0_hash_mismatch_rejected(self):
        self.save()
        with self.g0.open("ab") as output:
            output.write(b"\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_event_only_checkpoint(self.checkpoint, self.g0, device="cpu")

    def test_wrong_format_rejected(self):
        path = self.malformed(lambda payload: payload.update(format="legacy_scgs"))
        with self.assertRaisesRegex(ValueError, "format"):
            load_event_only_checkpoint(path, self.g0, device="cpu")

    def test_extra_camera_path_rejected(self):
        path = self.malformed(lambda payload: payload["camera"].update(video_path="/not/read.mp4"))
        with self.assertRaisesRegex(ValueError, "camera"):
            load_event_only_checkpoint(path, self.g0, device="cpu")

    def test_nonrigid_camera_rejected(self):
        path = self.malformed(lambda payload: payload["camera"]["w2c"][0].__setitem__(0, 2.))
        with self.assertRaisesRegex(ValueError, "rotation"):
            load_event_only_checkpoint(path, self.g0, device="cpu")

    def test_binding_length_rejected(self):
        path = self.malformed(lambda payload: payload.update(gaussian_count=4))
        with self.assertRaisesRegex(ValueError, "binding length"):
            load_event_only_checkpoint(path, self.g0, device="cpu")

    def test_binding_indices_outside_nodes_rejected(self):
        path = self.malformed(lambda payload: payload["state_dict"]["knn_indices"].__setitem__((0, 0), 5))
        with self.assertRaisesRegex(ValueError, "neighbor"):
            load_event_only_checkpoint(path, self.g0, device="cpu")

    def test_state_graph_disagreement_rejected(self):
        path = self.malformed(lambda payload: payload["graph"].update(centers=payload["graph"]["centers"] + 1))
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            load_event_only_checkpoint(path, self.g0, device="cpu")

    def test_nonincreasing_knots_rejected(self):
        path = self.malformed(lambda payload: payload["times"].__setitem__(1, 0.))
        with self.assertRaisesRegex(ValueError, "Times"):
            load_event_only_checkpoint(path, self.g0, device="cpu")

    def test_disallowed_modalities_rejected(self):
        path = self.malformed(lambda payload: payload["input_provenance"].update(
            allowed_modalities=["g0", "events", "camera", "video"]))
        with self.assertRaisesRegex(ValueError, "modalities"):
            load_event_only_checkpoint(path, self.g0, device="cpu")

    def test_save_rejects_unmatched_graph(self):
        self.graph["centers"] = self.graph["centers"] + 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.save()
        self.assertFalse(self.checkpoint.exists())

    def test_sequence_sampling_includes_exact_last_time(self):
        times = sequence_times(6.65, 30)
        self.assertEqual(times[0], 0)
        self.assertEqual(times[-1], 6.65)
        self.assertEqual(len(times), 201)
        self.assertTrue(all(a < b for a, b in zip(times, times[1:])))
        self.assertEqual(sequence_times(1., 2), [0., .5, 1.])
        with self.assertRaises(ValueError):
            requested_ply_times(",".join(str(i / 10) for i in range(9)), 1.)
        with self.assertRaises(ValueError):
            requested_ply_times("0,2", 1.)

    def test_posed_ply_changes_only_geometry_fields(self):
        g0 = load_g0(self.g0)
        xyz, q = self.motion.deform(g0.xyz, g0.rotation, .5)
        path = self.directory / "posed.ply"
        write_posed_ply(path, g0, xyz, q)
        result = PlyData.read(str(path))
        self.assertEqual(result.comments, g0.ply.comments)
        self.assertEqual(result.obj_info, g0.ply.obj_info)
        self.assertEqual(result["vertex"].data.dtype, g0.raw_vertex.dtype)
        changed = {"x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3"}
        for field in g0.property_names:
            if field not in changed:
                np.testing.assert_array_equal(result["vertex"].data[field], g0.raw_vertex[field])
        self.assertGreater(float(np.abs(result["vertex"].data["x"] - g0.raw_vertex["x"]).sum()), 0)
        with self.assertRaises(FileExistsError):
            write_posed_ply(path, g0, xyz, q)


if __name__ == "__main__":
    unittest.main()
