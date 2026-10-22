import unittest
import numpy as np
from utils.surface_detail_metrics import surface_comparison, mesh_structure_edges, edge_agreement


class SurfaceDetailMetricsTests(unittest.TestCase):
    def test_black_solid_is_not_a_hole_and_identity_is_exact(self):
        black = np.zeros((16, 16), np.float32)
        alpha = np.ones_like(black)
        parts = np.ones_like(black, dtype=np.int32)
        score = surface_comparison(black, black, alpha, alpha, parts, [1], black)
        self.assertEqual(score['gray_l1_from_parent'], 0)
        self.assertEqual(score['after_target_missing']['pixels'], 0)
        self.assertEqual(score['reference_mae']['after']['target'], 0)

    def test_holes_include_connected_extent_not_gray_darkness(self):
        old = np.ones((16, 16), np.float32)
        new = old.copy(); new[2:5, 4:8] = 0
        parts = np.ones(old.shape, np.int32)
        score = surface_comparison(old, old, old, new, parts, [1])
        self.assertEqual(score['new_alpha_holes_on_target']['pixels'], 12)
        self.assertEqual(score['new_alpha_holes_on_target']['largest_component'], 12)

    def test_fixed_edges_penalize_noise(self):
        gray = np.zeros((64, 64), np.float32); gray[:, 32:] = .7
        parts = np.zeros(gray.shape, np.int32); parts[:, 32:] = 1
        depth = np.ones_like(gray)
        edges, region = mesh_structure_edges(parts, depth, [0, 1])
        clean = edge_agreement(gray, edges, region)
        rng = np.random.default_rng(7)
        noise = (rng.random(gray.shape)*.3+gray).astype(np.float32)
        noisy = edge_agreement(noise, edges, region)
        self.assertEqual(clean['precision'], 1)
        self.assertEqual(clean['recall'], 1)
        self.assertLess(noisy['precision'], clean['precision'])
        self.assertGreater(noisy['unsupported_edge_pixels'], 0)

    def test_nan_off_surface_safe_but_invalid_hit_rejected(self):
        parts = np.full((8, 8), -1, np.int32); parts[2:6, 2:6] = 1
        depth = np.full((8, 8), np.nan, np.float32); depth[parts == 1] = 1
        edges, region = mesh_structure_edges(parts, depth, [1])
        self.assertTrue(edges.any()); self.assertTrue(region.any())
        depth[3, 3] = np.nan
        with self.assertRaises(ValueError):
            mesh_structure_edges(parts, depth, [1])


if __name__ == '__main__':
    unittest.main()
