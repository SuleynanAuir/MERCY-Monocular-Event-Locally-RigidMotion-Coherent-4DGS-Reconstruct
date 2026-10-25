import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from utils.gaussian_pair_renderer import load_detail_normals
from utils.lamp_detail import detail_finish, height_weights
from utils.event_refinement import qrotate


class LampDetailTests(unittest.TestCase):
    def test_base_mask_is_fixed_in_canonical_space_with_a_smooth_transition(self):
        xyz = torch.tensor([[0., 0., 0.], [0., .5, 0.], [0., 1., 0.]])
        weights = height_weights(xyz, dict(axis=[0, 1, 0], start=0, end=1))
        torch.testing.assert_close(weights[:, 0], torch.tensor([0., .5, 1.]))
        with self.assertRaises(ValueError):
            height_weights(xyz, dict(axis=[0, 1, 0], start=1, end=0))

    def test_zero_strength_preserves_the_original_surface_shading(self):
        alpha = torch.ones(1, 48, 48)
        rgb = torch.full((3, 48, 48), .5)
        normal = torch.full_like(rgb, .5)
        output = detail_finish(rgb, normal, alpha*2, alpha, 40, [1., 1., 1.], 'clay')
        torch.testing.assert_close(output, rgb, atol=0, rtol=0)

    def test_inconsistent_normals_cannot_add_false_surface_detail(self):
        alpha = torch.ones(1, 48, 48)
        rgb = torch.full((3, 48, 48), .5)
        # The observed plane faces the camera; a sideways transferred normal
        # must not introduce an unsupported illumination change.
        normal = torch.tensor([1., .5, .5])[:, None, None]*alpha
        output = detail_finish(rgb, normal, alpha*2, alpha, 40, [1., 1., 1.], 'gaussian')
        torch.testing.assert_close(output, rgb, atol=0, rtol=0)

    def test_normal_asset_rejects_stale_sequence_and_invalid_directions(self):
        metadata = dict(format='lamp_detail_normals_v1', sequence_sha256='sequence',
                        canonical_sha256='canonical', normal_count=2)
        normals = np.array([[0., 0., 1.], [1., 0., 0.]], np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'normals.npz'
            def save(n):
                np.savez(path, local_normals=n, metadata_json=np.asarray(json.dumps(metadata)))
            save(normals)
            np.testing.assert_array_equal(load_detail_normals(path, 'sequence', 'canonical', 2), normals)
            for sequence, canonical, count in [('stale', 'canonical', 2), ('sequence', 'old', 2), ('sequence', 'canonical', 3)]:
                with self.assertRaises(ValueError):
                    load_detail_normals(path, sequence, canonical, count)
            for value in (0., np.nan):
                broken = normals.copy(); broken[0] = value; save(broken)
                with self.assertRaises(ValueError):
                    load_detail_normals(path, 'sequence', 'canonical', 2)

    def test_normal_directions_follow_rotation_not_translation(self):
        normal = torch.tensor([[1., 0., 0.]])
        q = torch.tensor([[2**-.5, 0., 0., 2**-.5]])
        torch.testing.assert_close(qrotate(q, normal), torch.tensor([[0., 1., 0.]]), atol=1e-6, rtol=0)

    def test_finish_preserves_background_and_finite_antialiased_edges(self):
        alpha = torch.zeros(1, 48, 48)
        alpha[:, 10:38, 10:38] = 1
        alpha[:, 9, 10:38] = .25
        depth = alpha*2
        normal = torch.tensor([.5, .5, 1.])[:, None, None]*alpha
        for background in ([1., 1., 1.], [0., 0., 0.]):
            bg = torch.tensor(background)[:, None, None]
            rgb = .5*alpha+bg*(1-alpha)
            for mode in ('gaussian', 'clay', 'pseudocolor'):
                output = detail_finish(rgb, normal, depth, alpha, 40, background, mode)
                self.assertTrue(torch.isfinite(output).all())
                self.assertGreaterEqual(float(output.min()), 0)
                self.assertLessEqual(float(output.max()), 1)
                mask = (alpha == 0).expand_as(output)
                torch.testing.assert_close(output[mask], rgb[mask], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
