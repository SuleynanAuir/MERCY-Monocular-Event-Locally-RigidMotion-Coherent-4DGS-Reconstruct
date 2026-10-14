import unittest

import numpy as np

from scripts.preview_lego_sparse_cleanup import protected_selection


class TestProtectedCleanupSelection(unittest.TestCase):
    def test_static_only_preserves_row_identity(self):
        actual = protected_selection(np.array([0, 4, 7], np.int32), 20, 10, np.array([2, 12]))
        np.testing.assert_array_equal(actual, [0, 4, 7])
        self.assertEqual(actual.dtype, np.int64)

    def test_baked_dynamic_and_rebuilt_suffix_are_rejected(self):
        for values in ([2], [10], [19], [-1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                protected_selection(np.array(values, np.int32), 20, 10, np.array([2, 12]))

    def test_ambiguous_row_order_and_float_indices_are_rejected(self):
        for values in (np.array([1, 1]), np.array([4, 1]), np.array([1.0]), np.array([[1]])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                protected_selection(values, 20, 10, np.array([2, 12]))

    def test_empty_diagnostic_selection_is_valid(self):
        result = protected_selection(np.array([], np.int32), 20, 10, np.array([2, 12]))
        self.assertEqual(result.shape, (0,))


if __name__ == '__main__':
    unittest.main()
