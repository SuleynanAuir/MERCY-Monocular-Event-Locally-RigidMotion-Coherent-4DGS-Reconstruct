"""CPU-only regression tests for static Gaussian PLY input detection."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from utils.ply_utils import gaussian_degree


class GaussianPlyTests(unittest.TestCase):
    def names(self, degree):
        return (["x", "y", "z", "opacity"]
                + [f"f_dc_{i}" for i in range(3)]
                + [f"scale_{i}" for i in range(3)]
                + [f"rot_{i}" for i in range(4)]
                + [f"f_rest_{i}" for i in range(3 * ((degree + 1) ** 2 - 1))])

    def check_file(self, names, text=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ply"
            vertices = np.zeros(2, dtype=[(name, "f4") for name in names])
            PlyData([PlyElement.describe(vertices, "vertex")], text=text).write(path)
            return gaussian_degree(path)

    def test_supported_degrees_binary_and_ascii(self):
        for degree in range(4):
            for text in (False, True):
                with self.subTest(degree=degree, text=text):
                    self.assertEqual(self.check_file(self.names(degree), text), degree)

    def test_regular_point_cloud_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing properties"):
            self.check_file(["x", "y", "z", "red", "green", "blue"])

    def test_invalid_coefficient_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "coefficient count"):
            self.check_file(self.names(1)[:-1])

    def test_nonconsecutive_coefficients_rejected(self):
        names = self.names(1)
        names[-1] = "f_rest_99"
        with self.assertRaisesRegex(ValueError, "numbered consecutively"):
            self.check_file(names)

    def test_unsupported_degree_rejected(self):
        with self.assertRaisesRegex(ValueError, "coefficient count"):
            self.check_file(self.names(4))


if __name__ == "__main__":
    unittest.main()
