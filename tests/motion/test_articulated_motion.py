import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from utils.articulated_motion import ArticulatedBaker, quaternion_product, wxyz
from scripts.motion.bind_lego_parts import similarity, transform


class ArticulatedMotionTests(unittest.TestCase):
    def test_similarity_recovers_scale_rotation_translation(self):
        source = np.random.default_rng(4).normal(size=(100, 3))
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler('xyz', [15, 25, 35], degrees=True).as_matrix() * .41
        matrix[:3, 3] = [1, -2, 3]
        np.testing.assert_allclose(similarity(source, transform(source, matrix)), matrix, atol=1e-12)

    def test_quaternion_order(self):
        z = wxyz(Rotation.from_euler('z', 90, degrees=True).as_matrix())
        x = wxyz(Rotation.from_euler('x', 90, degrees=True).as_matrix())
        np.testing.assert_allclose(quaternion_product(z, x), [.5, .5, .5, .5], atol=1e-7)

    def test_rigid_part_updates_positions_and_orientations(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            vertices = np.zeros(3, dtype=[(k, 'f4') for k in ('x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3')])
            vertices['x'] = [1, 2, 9]; vertices['rot_0'] = 1
            matrices = np.tile(np.eye(4), (2, 1, 1, 1))
            matrices[1, 0, :3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
            matrices[1, 0, :3, 3] = [2, 3, 0]
            np.savez(temp / 'parts.npz', object_matrices=matrices,
                     metadata_json=np.asarray(json.dumps({'deforming_objects': []})))
            np.savez(temp / 'binding.npz', dynamic_indices=np.array([0, 1]), labels=np.zeros(3, int),
                     nearest_vertex=np.arange(3), registration=np.eye(4), metadata_json=np.asarray('{}'))
            baker = ArticulatedBaker(vertices, temp / 'parts.npz', temp / 'binding.npz')
            np.testing.assert_allclose(baker.frame(0)[:, :3], [[1, 0, 0], [2, 0, 0]])
            result = baker.frame(1)
            np.testing.assert_allclose(result[:, :3], [[2, 4, 0], [2, 5, 0]], atol=1e-7)
            np.testing.assert_allclose(result[:, 3:], [[2**-.5, 0, 0, 2**-.5]] * 2, atol=1e-7)
            self.assertLess(baker.max_rigidity_error, 1e-7)
            matrices[1, 0, :3, :3] *= 2
            np.savez(temp / 'parts.npz', object_matrices=matrices,
                     metadata_json=np.asarray(json.dumps({'deforming_objects': []})))
            with self.assertRaisesRegex(ValueError, 'not rigid'):
                ArticulatedBaker(vertices, temp / 'parts.npz', temp / 'binding.npz')

    def test_elastic_surface_local_fit_tracks_a_known_rigid_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            rest = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            rotation = Rotation.from_euler('z', 45, degrees=True).as_matrix()
            current = rest @ rotation.T + [1, 2, 3]
            vertices = np.zeros(4, dtype=[(k, 'f4') for k in ('x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3')])
            for i, key in enumerate(('x', 'y', 'z')):
                vertices[key] = rest[:, i]
            vertices['rot_0'] = 1
            np.savez(folder / 'parts.npz', object_matrices=np.tile(np.eye(4), (2, 1, 1, 1)),
                     deformed_0=np.stack((rest, current)), part_offsets=np.array([0, 4]),
                     metadata_json=np.asarray(json.dumps({'deforming_objects': [0]})))
            np.savez(folder / 'binding.npz', dynamic_indices=np.arange(4), labels=np.zeros(4, int),
                     nearest_vertex=np.arange(4), registration=np.eye(4), metadata_json=np.asarray('{}'))
            baker = ArticulatedBaker(vertices, folder / 'parts.npz', folder / 'binding.npz')
            np.testing.assert_allclose(baker.frame(1)[:, :3], current, atol=2e-7)
            np.testing.assert_allclose(baker.frame(1)[:, 3:], np.tile(wxyz(rotation), (4, 1)), atol=2e-7)


if __name__ == '__main__':
    unittest.main()
