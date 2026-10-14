import copy
import unittest
from unittest.mock import patch

import numpy as np

from scripts.render_lego_multiview import validate_mask_arrays, validate_motion_metadata, validate_view_plan
from utils.viewer_state import orbit_pose


class MultiviewRenderTests(unittest.TestCase):
    def fixture(self):
        pose = np.eye(4); pose[2, 3] = 3.
        camera = {'pose': pose.tolist(), 'target': [0., 0., 0.], 'fovy_degrees': 40.}
        views = []
        for index, yaw in enumerate((0, 180)):
            view = {'view_index': index, 'frame': 100, 'yaw': yaw, 'pitch': 0,
                    'zoom': 1., 'pan_x': 0., 'pan_y': 0.}
            view['camera_pose'] = orbit_pose(camera['pose'], camera['target'], view).tolist()
            views.append(view)
        cache = {'camera': camera, 'frame_count': 400}
        motion = {'format': 'lego_corrected_mesh_motion_v1', 'objects': ['rigid', 'band'],
                  'excluded_deforming_objects': [1], 'baseline_cache_sha256': 'a'*64,
                  'source_correction_sha256': 'b'*64, 'source_parts_sha256': 'c'*64}
        metadata = {'format': 'lego_corrected_rigid_mesh_multiview_masks_v1', 'resolution': [640, 640],
                    'view_count': 2, 'camera': camera, 'views': views, 'objects': ['rigid', 'band'],
                    'excluded_deforming_objects': [1], 'motion_metadata': motion}
        record = {'frame_count': 400, 'joint_structure': {'input_sha256': {'parts': 'c'*64}}}
        return cache, metadata, motion, record

    def test_preserves_exact_mask_views(self):
        cache, metadata, _, _ = self.fixture()
        plan = validate_view_plan(metadata, np.array([100, 100], np.int32), cache)
        self.assertIs(plan, metadata['views'])
        self.assertEqual(plan[1]['yaw'], 180)

    def test_rejects_camera_pose_displacement(self):
        cache, metadata, _, _ = self.fixture()
        metadata['views'][1]['camera_pose'][0][3] += .001
        with self.assertRaisesRegex(ValueError, 'camera pose'):
            validate_view_plan(metadata, np.array([100, 100]), cache)

    def test_rejects_view_frame_and_order_mismatch(self):
        cache, metadata, _, _ = self.fixture()
        with self.assertRaises(ValueError):
            validate_view_plan(metadata, np.array([100, 101]), cache)
        metadata['views'][1]['view_index'] = 0
        with self.assertRaises(ValueError):
            validate_view_plan(metadata, np.array([100, 100]), cache)

    def test_rejects_different_camera_or_resolution(self):
        cache, metadata, _, _ = self.fixture()
        altered = copy.deepcopy(cache); altered['camera']['fovy_degrees'] = 39.
        with self.assertRaises(ValueError):
            validate_view_plan(metadata, np.array([100, 100]), altered)
        metadata['resolution'] = [320, 320]
        with self.assertRaises(ValueError):
            validate_view_plan(metadata, np.array([100, 100]), cache)

    def test_rejects_nonfinite_or_nonpositive_view_parameters(self):
        for field, value in (('zoom', 0), ('pitch', np.nan), ('yaw', True)):
            cache, metadata, _, _ = self.fixture()
            metadata['views'][0][field] = value
            with self.assertRaises(ValueError):
                validate_view_plan(metadata, np.array([100, 100]), cache)

    def test_mask_depth_contract(self):
        _, metadata, _, _ = self.fixture()
        labels = np.full((2, 640, 640), -1, np.int16)
        depth = np.full(labels.shape, np.nan, np.float32)
        labels[0, 10, 20] = 0; depth[0, 10, 20] = 2.
        validate_mask_arrays(labels, depth, metadata)
        depth[0, 10, 20] = 0
        with self.assertRaises(ValueError):
            validate_mask_arrays(labels, depth, metadata)
        depth[0, 10, 20] = 2.; depth[0, 20, 10] = 1.
        with self.assertRaises(ValueError):
            validate_mask_arrays(labels, depth, metadata)

    def test_motion_identity_accepts_exact_parent(self):
        cache, metadata, motion, record = self.fixture()
        validate_motion_metadata(metadata, motion, cache, record, 'a'*64, 'b'*64)

    def test_motion_identity_rejects_unrelated_model(self):
        cache, metadata, motion, record = self.fixture()
        with self.assertRaises(ValueError):
            validate_motion_metadata(metadata, motion, cache, record, 'a'*64, 'd'*64)
        wrong = copy.deepcopy(motion); wrong['source_parts_sha256'] = 'e'*64
        metadata['motion_metadata'] = wrong
        with self.assertRaises(ValueError):
            validate_motion_metadata(metadata, wrong, cache, record, 'a'*64, 'b'*64)

    def test_compact_suffix_requires_exact_parent_motion_metadata(self):
        cache, metadata, motion, record = self.fixture()
        report = {'parent_baseline': '/verified/parent/baseline', 'parent_correction_sha256': 'b'*64}
        with patch('scripts.render_lego_multiview.sha256', return_value='a'*64):
            validate_motion_metadata(metadata, motion, cache, record, 'd'*64, 'e'*64, report)
            report['parent_correction_sha256'] = 'f'*64
            with self.assertRaises(ValueError):
                validate_motion_metadata(metadata, motion, cache, record, 'd'*64, 'e'*64, report)


if __name__ == '__main__':
    unittest.main()
