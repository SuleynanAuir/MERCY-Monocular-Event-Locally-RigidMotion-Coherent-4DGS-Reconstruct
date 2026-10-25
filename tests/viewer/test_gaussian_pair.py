"""Protect camera/timeline identity and preserve rigid motion when changing frames."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.lamp.prepare_lamp_viewer import bake, camera_pose, rigid_deltas
from utils.gaussian_pair_renderer import validate_pair
from utils.gaussian_sequence import GaussianSequence
from utils.viewer_state import orbit_pose, validate_request


class GaussianPairTests(unittest.TestCase):
    def test_pair_allows_different_point_counts_but_not_camera_or_time(self):
        base = dict(frame_count=2, fps=30, preview_camera={'pose': np.eye(4).tolist()},
                    frames=[{'time_seconds': 0}, {'time_seconds': 1/30}], gaussians_per_frame=10)
        other = copy.deepcopy(base); other['gaussians_per_frame'] = 20
        validate_pair(base, other)
        for key, value in [('fps', 60), ('frame_count', 3), ('preview_camera', {}),
                           ('frames', [{'time_seconds': 0}, {'time_seconds': .1}])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_pair(base, dict(other, **{key: value}))

    def test_front_view_preserves_supplied_extrinsics(self):
        r = Rotation.from_euler('xyz', [12, 20, -3], degrees=True).as_matrix()
        camera = dict(R=r.T.tolist(), T=[.2, -.1, 2], target_center=[0, 0, 0], fov_y_deg=40)
        preview = camera_pose(camera)
        actual = orbit_pose(preview['pose'], preview['target'], validate_request({}, 2))
        view = np.linalg.inv(actual); view[1:3] *= -1
        np.testing.assert_allclose(view[:3, :3], r, atol=2e-7)
        np.testing.assert_allclose(view[:3, 3], camera['T'], atol=2e-7)

    def test_motion_conjugation_preserves_lengths_and_rest(self):
        matrices = np.tile(np.eye(4), (2, 1, 1, 1))
        matrices[1, 0, :3, :3] = Rotation.from_euler('z', 70, degrees=True).as_matrix()
        matrices[1, 0, :3, 3] = [.2, .1, 0]
        registration = np.eye(4); registration[:3, :3] *= .8
        registration[:3, 3] = [1, 2, 3]
        delta = rigid_deltas(matrices, registration)
        np.testing.assert_array_equal(delta[0, 0], np.eye(4))
        x = np.array([[.1, .2, .3], [1, .3, -.5]])
        y = x @ delta[1, 0, :3, :3].T + delta[1, 0, :3, 3]
        self.assertAlmostEqual(np.linalg.norm(x[1]-x[0]), np.linalg.norm(y[1]-y[0]))

    def test_nonrigid_motion_is_rejected(self):
        for bad in (np.diag([1.1, 1, 1, 1]), np.diag([-1., 1, 1, 1])):
            with self.assertRaises(ValueError):
                rigid_deltas(np.stack([np.eye(4), bad])[:, None], np.eye(4))

    def test_offset_shade_rows_follow_their_part_instead_of_freezing(self):
        from plyfile import PlyData, PlyElement
        names = ['x', 'y', 'z', 'opacity'] + [f'f_dc_{i}' for i in range(3)]
        names += [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
        vertices = np.zeros(31, dtype=[(name, '<f4') for name in names])
        vertices['x'][:30] = np.linspace(0, 1, 30)
        vertices['x'][-1], vertices['y'][-1] = .2, .08
        vertices['rot_0'] = 1
        surface = np.column_stack((vertices['x'][:30], np.zeros((30, 2))))
        groups = (surface[:, 0] < .4).astype(np.int16)
        delta = np.tile(np.eye(4), (2, 2, 1, 1)); delta[1, 1, 0, 3] = .1
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'source.ply', Path(directory)/'motion.npz'
            PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(source)
            report = bake(source, target, surface, groups, delta, {}, dict(frame_count=2, fps=30), {})
            self.assertEqual(report['extrapolated_nearest_part_rows'], 1)
            with GaussianSequence(target) as sequence:
                row = int(np.flatnonzero(sequence.dynamic_indices == 30)[0])
                np.testing.assert_allclose(sequence.poses[row, 1, :3], [.3, .08, 0], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
