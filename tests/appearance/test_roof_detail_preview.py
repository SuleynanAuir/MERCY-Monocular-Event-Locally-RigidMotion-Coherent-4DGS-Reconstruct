import unittest
import numpy as np
from scripts.appearance.preview_lego_roof_detail import (guarded_retirement, next_active,
    validate_target_triangles, validate_additional_guard_metadata)


class RoofDetailPreviewTests(unittest.TestCase):
    def test_additional_guard_keeps_exact_prior_and_separate_plan(self):
        primary = {'mesh': '/exact/mesh.npz', 'mesh_sha256': 'a'*64,
                   'motion': '/exact/motion.npz', 'motion_sha256': 'b'*64,
                   'objects': ['roof'], 'motion_metadata': {'frozen': True},
                   'camera': {'pose': 'exact'}, 'resolution': [640, 640],
                   'views': [{'frame': 100}]}
        additional = {**primary, 'views': [{'frame': 150}]}
        self.assertEqual(validate_additional_guard_metadata(additional, primary), [{'frame': 150}])
        self.assertEqual(primary['views'], [{'frame': 100}])
        for key in ('mesh_sha256', 'motion_sha256', 'objects', 'motion_metadata', 'camera'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_additional_guard_metadata({**additional, key: 'wrong'}, primary)

    def test_additional_guard_plan_must_be_bounded(self):
        for views in ([], [{}]*65, None):
            with self.subTest(views=views), self.assertRaises(ValueError):
                validate_additional_guard_metadata({'views': views}, {})

    def test_visible_retirement_is_protected(self):
        result = guarded_retirement(np.array([1, 3, 5]), np.array([0., .0001, .01]),
                                    np.array([2, 4]), 9, .0001)
        np.testing.assert_array_equal(result, [1, 3])

    def test_dynamic_rows_are_never_retired(self):
        with self.assertRaises(ValueError):
            guarded_retirement(np.array([1, 3]), np.zeros(2), np.array([3]), 9, .0001)

    def test_rejected_additions_cannot_reactivate(self):
        result = next_active(np.array([True, False, True]), np.array([0., 0., .001]), .0001)
        np.testing.assert_array_equal(result, [True, False, False])

    def test_nonfinite_exposure_is_rejected(self):
        with self.assertRaises(ValueError):
            next_active(np.array([True]), np.array([np.nan]), .0001)

    def test_actual_triangle_support_is_required(self):
        mesh = {'vertices': np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                'triangles': np.array([[0, 1, 2]]), 'part_ids': np.array([20])}
        arrays = {'new_xyz': np.array([[.2, .2, 0.]]), 'new_quaternions': np.array([[1., 0., 0., 0.]]),
                  'triangle_ids': np.array([0]), 'part_ids': np.array([20])}
        validate_target_triangles(arrays, mesh, np.array([20]))
        arrays['new_xyz'][0] = [.9, .9, 0.]
        with self.assertRaises(ValueError):
            validate_target_triangles(arrays, mesh, np.array([20]))

    def test_non_target_part_cannot_supply_geometry(self):
        mesh = {'vertices': np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                'triangles': np.array([[0, 1, 2]]), 'part_ids': np.array([20])}
        arrays = {'new_xyz': np.array([[.2, .2, 0.]]), 'new_quaternions': np.array([[1., 0., 0., 0.]]),
                  'triangle_ids': np.array([0]), 'part_ids': np.array([20])}
        with self.assertRaises(ValueError):
            validate_target_triangles(arrays, mesh, np.array([11]))


if __name__ == '__main__':
    unittest.main()
