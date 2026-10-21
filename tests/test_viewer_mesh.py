import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.reconstruct_viewer_mesh import bilinear, projected_samples, valid_depth
from utils.viewer_mesh import load_mesh_data, projection_matrix
from utils.viewer_state import cached_image_reference, validate_stream


class MeshTests(unittest.TestCase):
    def test_gaussian_depth_is_normalized_and_transparent_pixels_are_excluded(self):
        depth = valid_depth(np.array([1., 2., np.nan, -1]), np.array([.5, .9, 1., 1.]), .65)
        np.testing.assert_allclose(depth, [0, 2 / .9, 0, 0])

    def test_projection_matches_opengl_camera_and_image_aspect(self):
        matrix = projection_matrix(800, 400, 90)
        clip = np.array([[0, 0, -2, 1], [4, 2, -2, 1]], np.float32) @ matrix.T
        np.testing.assert_allclose(clip[:, :2] / clip[:, 3:], [[0, 0], [1, 1]], atol=1e-6)

    def test_color_baking_projection_and_sampling_preserve_edges(self):
        vertices = np.array([[0, 0, -2], [1, 1, -2], [0, 0, 2]], np.float32)
        x, y, z, valid = projected_samples(vertices, np.eye(4), 2., 8)
        np.testing.assert_allclose(x[:2], [3.5, 4.5])
        np.testing.assert_allclose(y[:2], [3.5, 2.5])
        self.assertEqual(valid.tolist(), [True, True, False])
        image = np.array([[0., 1.], [0., 1.]])
        np.testing.assert_allclose(bilinear(image, np.array([.1, .9]), np.array([.5, .5])), [.1, .9])

    def test_source_identity_and_motion_boundaries_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mesh.npz'
            fingerprints = {'candidate_correction_sha256': 'correction', 'right_source_sha256': 'gaussians'}
            data = dict(vertices=np.eye(3, dtype=np.float32), triangles=np.array([[0, 1, 2]], np.int32),
                        colors=np.ones((3, 3), np.float32), vertex_groups=np.zeros(3, np.int32),
                        metadata_json=np.asarray(json.dumps(dict(format='gaussian_tsdf_mesh_v1',
                                                                  source_fingerprints=fingerprints))))
            np.savez(path, **data)
            self.assertEqual(len(load_mesh_data(path, 2, fingerprints)['faces']), 1)
            with self.assertRaisesRegex(ValueError, 'current Gaussian'):
                load_mesh_data(path, 2, {'right_source_sha256': 'different'})
            data['vertex_groups'][2] = 1
            np.savez(path, **data)
            with self.assertRaisesRegex(ValueError, 'independent motion'):
                load_mesh_data(path, 2, fingerprints)
            data['triangles'][0, 2] = 4
            np.savez(path, **data)
            with self.assertRaises(ValueError):
                load_mesh_data(path, 2, fingerprints)

    def test_mesh_cache_reference_preserves_two_synchronized_panels(self):
        import struct
        request = validate_stream({'render_mode': 'mesh', 'width': 512, 'height': 512}, 400)
        request['_stream'] = {'seq': 4}
        packet = cached_image_reference(request, comparison=True)
        length = struct.unpack('<I', packet[:4])[0]
        meta = json.loads(packet[4:4 + length])
        self.assertEqual(meta['width'], 1024)
        self.assertTrue(meta['comparison'])
        self.assertEqual(meta['left_frame'], meta['right_frame'])
        self.assertEqual(meta['render_mode'], 'mesh')


if __name__ == '__main__':
    unittest.main()
