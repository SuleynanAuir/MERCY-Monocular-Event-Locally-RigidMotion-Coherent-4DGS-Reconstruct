import tempfile
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.geometry.export_lego_multiview_masks import (argument_parser, camera_rays, posed_vertices,
                                                quaternion_matrices, validate_identity, vertex_part_ids, views, write_png)


class MultiviewMaskTests(unittest.TestCase):
    def test_mesh_motion_identity_must_match_not_only_object_count(self):
        mesh = {'format': 'lego_complete_mesh_prior_v1', 'objects': ['a', 'b'],
                'parts_sha256': 'a'*64, 'deforming_objects': [1]}
        motion = {'format': 'lego_corrected_mesh_motion_v1', 'objects': ['a', 'b'],
                  'source_parts_sha256': 'a'*64, 'excluded_deforming_objects': [1]}
        validate_identity(mesh, motion)
        motion['objects'] = ['b', 'a']
        with self.assertRaisesRegex(ValueError, 'identity'):
            validate_identity(mesh, motion)

    def test_plan_matches_readonly_ui_capture(self):
        from scripts.geometry.capture_lego_multiview import snapshot_plan
        self.assertEqual([(row['frame'], row['yaw'], row['pitch']) for row in views()], snapshot_plan())

    def test_front_plan_includes_final_frame_without_duplicate(self):
        plan = views(front_stride=10, frame_count=400)
        self.assertEqual([row['frame'] for row in plan], list(range(0, 400, 10)) + [399])
        self.assertEqual(len(plan), 41)
        self.assertTrue(all(row['yaw'] == row['pitch'] == 0 for row in plan))
        self.assertEqual(len(views(front_stride=10, frame_count=401)), 41)
        with self.assertRaisesRegex(ValueError, 'stride'):
            views(front_stride=0)

    def test_heldout_24_has_no_selection_view_overlap(self):
        selected = {(row['frame'], row['yaw'], row['pitch']) for row in views()}
        extra = {(row['frame'], row['yaw'], row['pitch']) for row in views(heldout=True)}
        self.assertEqual(len(extra), 24)
        self.assertFalse(selected & extra)
        for frame in (50, 150):
            for yaw in (0, 90, 180, 270):
                self.assertIn((frame, yaw, 0), extra)
        for pitch in (-60, 60):
            for yaw in range(0, 360, 45):
                self.assertIn((150, yaw, pitch), extra)
        with self.assertRaisesRegex(ValueError, 'requires'):
            views(heldout=True, frame_count=150)

    def test_roof_back_detail_plan_has_twelve_new_camera_angles(self):
        old = {(r['frame'], r['yaw'], r['pitch']) for r in views()+views(heldout=True)}
        new = {(r['frame'], r['yaw'], r['pitch']) for r in views(detail_check=True)}
        self.assertEqual(len(new), 12)
        self.assertFalse(old & new)
        self.assertEqual({row[2] for row in new}, {50, 70})
        with self.assertRaises(ValueError):
            views(heldout=True, detail_check=True)
        with self.assertRaises(ValueError):
            views(front_stride=10, detail_check=True)

    def test_heldout_and_front_stride_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, 'exclusive'):
            views(front_stride=10, heldout=True)
        argv = ['--mesh', 'mesh.npz', '--motion', 'motion.npz', '--output', 'mask.npz',
                '--heldout', '--front-stride', '10']
        with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit) as raised:
            argument_parser().parse_args(argv)
        self.assertEqual(raised.exception.code, 2)

    def test_second_detail_plan_is_disjoint_from_all_previous_diagnostics(self):
        old = {(r['frame'], r['yaw'], r['pitch']) for r in
               views()+views(heldout=True)+views(detail_check=True)}
        new = {(r['frame'], r['yaw'], r['pitch']) for r in views(detail_check_v2=True)}
        self.assertEqual(len(new), 12)
        self.assertFalse(old & new)
        self.assertEqual({r[0] for r in new}, {250})
        self.assertEqual({r[2] for r in new}, {45, 65})
        with self.assertRaises(ValueError):
            views(detail_check=True, detail_check_v2=True)
        with self.assertRaises(ValueError):
            views(detail_check_v2=True, frame_count=250)

    def test_rays_use_pixel_centers_and_opengl_forward(self):
        origin, directions = camera_rays(np.eye(4), 3, 3, 90)
        np.testing.assert_allclose(origin, 0)
        np.testing.assert_allclose(directions[1, 1], [0, 0, -1])
        self.assertGreater(directions[0, 0, 1], 0)
        self.assertLess(directions[0, 0, 0], 0)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1)

    def test_conflicting_vertex_part_ownership_is_rejected(self):
        triangle = np.array([[0, 1, 2], [0, 2, 3]])
        np.testing.assert_array_equal(vertex_part_ids(5, triangle, np.array([2, 2])), [2, 2, 2, 2, -1])
        with self.assertRaisesRegex(ValueError, 'shared'):
            vertex_part_ids(5, triangle, np.array([1, 2]))

    def test_wxyz_rotation_and_rigid_part_motion(self):
        q = np.array([[np.sqrt(.5), 0, 0, np.sqrt(.5)]])
        np.testing.assert_allclose(quaternion_matrices(q)[0] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        actual = posed_vertices(np.array([[1., 0, 0], [0., 2, 0]]), np.array([0, 1]),
                                np.array([0, -1]), q, np.array([[3., 0, 0]]), {1})
        np.testing.assert_allclose(actual, [[3, 1, 0], [0, 2, 0]], atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'no corrected motion group'):
            posed_vertices(actual, np.array([0, 1]), np.array([0, -1]), q, np.zeros((1, 3)), set())

    def test_png_output_is_exact_and_not_flipped(self):
        pixels = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [10, 20, 30]]], np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'preview.png'
            write_png(path, pixels)
            np.testing.assert_array_equal(np.asarray(Image.open(path)), pixels)


if __name__ == '__main__':
    unittest.main()
