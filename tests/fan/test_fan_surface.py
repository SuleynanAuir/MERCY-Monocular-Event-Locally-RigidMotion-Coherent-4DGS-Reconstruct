import unittest
import json
import numpy as np
from utils.fan_surface import regularize, attributes, validate_refinement


def plane(seed=7):
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(-.02, .02, 35), np.linspace(-.02, .02, 35))
    names = ['x', 'y', 'z']+[f'rot_{i}' for i in range(4)]+[f'scale_{i}' for i in range(3)]+['opacity']+[f'f_dc_{i}' for i in range(3)]
    b = np.zeros(x.size, dtype=[(name, '<f4') for name in names])
    b['x'], b['y'], b['z'] = x.ravel(), y.ravel(), rng.normal(0, .00016, x.size)
    b['rot_0'] = 1
    b['scale_0'], b['scale_1'], b['scale_2'] = np.log(.0015), np.log(.001), np.log(1e-8)
    b['opacity'] = 2
    return b


class FanSurfaceTests(unittest.TestCase):
    def test_surface_noise_reduces_without_collapsing_independent_layers(self):
        a = plane(); b = a.copy(); b['z'] += .0004
        source = np.concatenate([a, b]); groups = np.repeat([0, 1], len(a))
        refined, parents, out_groups, report = regularize(source, groups, split_budget=0)
        self.assertTrue(np.array_equal(parents, np.arange(len(source))))
        self.assertTrue(np.array_equal(groups, out_groups))
        self.assertLess(np.std(refined['z'][:len(a)]), np.std(a['z'])*.9)
        np.testing.assert_allclose(refined['z'][len(a):]-refined['z'][:len(a)], .0004, atol=1e-8)
        self.assertLessEqual(report['maximum_center_shift'], .00065)

    def test_split_preserves_center_covariance_and_optical_mass(self):
        base = plane(); groups = np.zeros(len(base), np.int32)
        child, parents, _, report = regularize(base, groups, strength=0, split_budget=20)
        self.assertEqual(len(child), len(base)+20)
        self.assertEqual(report['split_parents'], 20)
        xyz, basis, scales, alpha, _ = attributes(base)
        cx, cb, cs, ca, _ = attributes(child)
        for parent in np.flatnonzero(np.bincount(parents) == 2):
            rows = np.flatnonzero(parents == parent)
            np.testing.assert_allclose(cx[rows].mean(0), xyz[parent], atol=2e-9)
            covariance = basis[parent] @ np.diag(scales[parent]**2) @ basis[parent].T
            mixture = sum(cb[j] @ np.diag(cs[j]**2) @ cb[j].T + np.outer(cx[j]-xyz[parent], cx[j]-xyz[parent]) for j in rows)/2
            np.testing.assert_allclose(mixture, covariance, atol=1e-11)
            optical_mass = (-np.log1p(-ca[rows])*np.prod(np.sort(cs[rows])[:, 1:], axis=1)).sum()
            original_mass = -np.log1p(-alpha[parent])*np.prod(np.sort(scales[parent])[1:])
            self.assertAlmostEqual(optical_mass/original_mass, 1, places=5)

    def test_opaque_isolated_detail_is_not_deleted(self):
        base = plane(); base['x'][-1] = 1
        refined, parents, _, _ = regularize(base, np.zeros(len(base), np.int32), split_budget=0)
        self.assertIn(len(base)-1, parents)
        self.assertEqual(refined['x'][-1], 1)

    def test_thin_line_not_flattened(self):
        base = plane()[:40]; base['x'] = np.arange(40)*.001; base['y'] = 0
        refined, parents, _, report = regularize(base, np.zeros(len(base), np.int32), split_budget=100)
        self.assertEqual(report['regularized'], 0)
        self.assertEqual(report['split_parents'], 0)
        np.testing.assert_array_equal(refined['z'], base['z'])

    def test_invalid_settings_and_attributes_rejected(self):
        b = plane(); groups = np.zeros(len(b), np.int32)
        with self.assertRaises(ValueError): regularize(b, groups, strength=2)
        with self.assertRaises(ValueError): regularize(b, groups[:-1])
        b['x'][0] = np.nan
        with self.assertRaises(ValueError): regularize(b, groups)

    def test_asset_rejects_stale_source_cross_part_and_displaced_children(self):
        base = plane(); groups = np.zeros(len(base), np.int32); groups[len(base)//2:] = 1
        refined, parents, out_groups, report = regularize(base, groups, split_budget=20)
        expected = dict(source_sha256='source', binding_sha256='binding', mesh_sha256='mesh', source_base_sha256='base')
        data = dict(base_vertices=refined, parent_rows=parents, groups=out_groups,
                    metadata_json=np.asarray(json.dumps(dict(format='fan_surface_refinement_v1', report=report, **expected))))
        xyz = attributes(base)[0]
        validate_refinement(data, groups, xyz, expected)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            validate_refinement(data, groups, xyz, dict(expected, source_sha256='stale'))
        data['groups'][0] = 1
        with self.assertRaisesRegex(ValueError, 'motion groups'):
            validate_refinement(data, groups, xyz, expected)
        data['groups'][0] = 0; data['base_vertices']['x'][0] += 1
        with self.assertRaisesRegex(ValueError, 'displacement'):
            validate_refinement(data, groups, xyz, expected)


if __name__ == '__main__': unittest.main()
