import json
from pathlib import Path
import tempfile
import unittest
import copy
from types import SimpleNamespace

import numpy as np

from scripts.geometry.build_lego_multiview_completion import visibility, moved, validate_exposure_reuse
from utils.rigid_checkpoint import sha256_file


class CompletionSelectionTests(unittest.TestCase):
    def test_exposure_reuse_requires_convergence_and_exact_footprint(self):
        params = dict(spacing=.0025, tangent_factor=.75, normal_factor=.12, opacity=.85)
        args = SimpleNamespace(**params)
        cache = {'camera': {'pose': 'fixture'}, 'render_supersample': 1}
        exposure = {'format': 'lego_raster_alpha_exposure_pruning_v1', 'converged_full_400': True,
                    'final_pass_frames': 400, 'parent_source_sha256': 'a', 'parent_correction_sha256': 'b',
                    'camera': cache['camera'], 'retained_count': 3, 'render_sampling': {},
                    'render_model_parameters': params}
        metadata = {'added_points': 3, 'exposure_pruning': exposure}
        validate_exposure_reuse(metadata, args, cache, 'a', 'b')
        validate_exposure_reuse({}, args, cache, 'a', 'b')
        for key, value in (('converged_full_400', False), ('final_pass_frames', 399),
                           ('parent_source_sha256', 'wrong'), ('camera', {}), ('retained_count', 2)):
            changed = copy.deepcopy(metadata); changed['exposure_pruning'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_exposure_reuse(changed, args, cache, 'a', 'b')
        for key in params:
            changed = copy.deepcopy(metadata); changed['exposure_pruning']['render_model_parameters'][key] *= .9
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'different Gaussian'):
                validate_exposure_reuse(changed, args, cache, 'a', 'b')

    def fixture(self, root):
        motion_path = root/'motion.npz'
        np.savez(motion_path, fixture=np.asarray([1]))
        motion = {'path': motion_path,
                  'quaternions': np.asarray([[[1., 0., 0., 0.]]], np.float32),
                  'translations': np.zeros((1, 1, 3), np.float32)}
        mask = np.full((1, 5, 5), -1, np.int16)
        depth = np.full((1, 5, 5), np.nan, np.float32)
        mask[0, 2, 2], depth[0, 2, 2] = 4, 2
        meta = {'motion_sha256': sha256_file(motion_path), 'mesh_sha256': 'a'*64,
                'camera': {'fovy_degrees': 90.},
                'views': [{'frame': 0, 'camera_pose': np.eye(4).tolist()}]}
        target = root/'masks.npz'
        np.savez(target, part_ids=mask, depth=depth, metadata_json=np.asarray(json.dumps(meta)))
        return motion, target

    def test_visibility_requires_same_part_and_front_depth(self):
        with tempfile.TemporaryDirectory() as folder:
            motion, masks = self.fixture(Path(folder))
            xyz = np.asarray([[0, 0, -2], [0, 0, -2.01], [0, 0, -2],
                              [0, 0, 2], [100, 0, -2]], np.float32)
            result = visibility(xyz, np.asarray([4, 4, 5, 4, 4]), np.zeros(5, np.int32),
                                motion, masks, .001, progress_label='test', mesh_sha256='a'*64)
            np.testing.assert_array_equal(result, [True, False, False, False, False])

    def test_visibility_rejects_different_mesh_or_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            motion, masks = self.fixture(Path(folder))
            args = (np.asarray([[0, 0, -2]], np.float32), np.asarray([4]), np.asarray([0]), motion, masks, .001)
            with self.assertRaisesRegex(ValueError, 'different source mesh'):
                visibility(*args, progress_label='test', mesh_sha256='b'*64)
            np.savez(motion['path'], fixture=np.asarray([2]))
            with self.assertRaisesRegex(ValueError, 'different corrected motion'):
                visibility(*args, progress_label='test', mesh_sha256='a'*64)

    def test_corrected_group_motion_keeps_part_rigid(self):
        xyz = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        q = np.asarray([[np.sqrt(.5), 0, 0, np.sqrt(.5)]], np.float32)
        result = moved(xyz, np.zeros(3, np.int32), q, np.asarray([[2, 3, 4]], np.float32))
        np.testing.assert_allclose(result, [[2, 3, 4], [2, 4, 4], [1, 3, 4]], atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(result[1]-result[2])), np.sqrt(2), places=6)


if __name__ == '__main__':
    unittest.main()
