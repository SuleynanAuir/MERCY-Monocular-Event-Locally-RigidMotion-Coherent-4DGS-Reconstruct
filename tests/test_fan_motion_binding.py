import unittest
import numpy as np
from utils.fan_motion_binding import stationary_reassignments


class FanBindingTests(unittest.TestCase):
    def test_only_persistent_static_support_changes_membership(self):
        # Static bracket, moving blade, ambiguous hub, unsupported noise,
        # intermittent occlusion, and a small improvement within tolerance.
        result = stationary_reassignments([1, .35, 1, .5, .90, .98], [.5, 1, .98, .1, .3, .85])
        np.testing.assert_array_equal(result, [True, False, False, False, False, False])

    def test_invalid_or_nonfinite_support_cannot_reassign_parts(self):
        for a, b in [([1], [1, 0]), ([float('nan')], [.2]), ([1.1], [.1]),
                     ([1], [-.1]), ([[1]], [[.1]])]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                stationary_reassignments(a, b)

    def test_empty_candidate_set_is_valid(self):
        self.assertEqual(stationary_reassignments([], []).size, 0)


if __name__ == '__main__': unittest.main()
