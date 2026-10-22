import copy
import unittest
import numpy as np
from utils.fan_mesh import comparison_groups, validate_source_pair


class FanMeshTests(unittest.TestCase):
    def test_repaired_support_gets_its_own_mesh_motion_group(self):
        groups = comparison_groups(6, np.array([1, 2, 4]), np.array([1, 4]))
        np.testing.assert_array_equal(groups, [0, 1, 2, 0, 1, 0])
        # In V1 groups 1+2 move, in V3 only group 1 moves.
        np.testing.assert_array_equal(np.flatnonzero(groups > 0), [1, 2, 4])
        np.testing.assert_array_equal(np.flatnonzero(groups == 1), [1, 4])

    def test_different_or_ambiguous_source_rows_are_rejected(self):
        for before, after in [([1, 2], [1, 3]), ([1, 2], [1, 2]),
                              ([2, 1], [1]), ([1, 1, 2], [1]), ([1, 6], [1])]:
            with self.subTest(before=before, after=after), self.assertRaises(ValueError):
                comparison_groups(6, np.array(before), np.array(after))

    def test_only_identical_canonical_geometry_camera_and_timeline_can_be_shared(self):
        source = dict(base_vertices_sha256='g0', frame_count=2, fps=30,
                      preview_camera={'pose': [1, 2, 3]}, input_sha256={'video': 'same'},
                      frames=[{'time_seconds': 0}, {'time_seconds': 1/30}])
        validate_source_pair(source, copy.deepcopy(source))
        for key in ('base_vertices_sha256', 'frame_count', 'fps', 'preview_camera', 'input_sha256'):
            other = copy.deepcopy(source); other[key] = 'different'
            with self.subTest(key=key), self.assertRaises(ValueError): validate_source_pair(source, other)
        other = copy.deepcopy(source); other['frames'][1]['time_seconds'] = .1
        with self.assertRaisesRegex(ValueError, 'timestamps'): validate_source_pair(source, other)


if __name__ == '__main__': unittest.main()
