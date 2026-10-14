"""Inspect Gaussian PLY schemas without importing CUDA or training code."""

import math

from plyfile import PlyData


def gaussian_degree(path):
    """Validate the Gaussian properties and infer the stored SH degree (0-3)."""
    vertex = PlyData.read(path)["vertex"]
    names = {p.name for p in vertex.properties}
    required = {"x", "y", "z", "opacity"}
    required.update(f"f_dc_{i}" for i in range(3))
    required.update(f"scale_{i}" for i in range(3))
    required.update(f"rot_{i}" for i in range(4))
    if required - names:
        raise ValueError(f"Not a Gaussian PLY; missing properties: {sorted(required - names)}")
    rest_names = {name for name in names if name.startswith("f_rest_")}
    count = len(rest_names)
    degree = math.isqrt(count // 3 + 1) - 1
    if degree not in range(4) or count != 3 * ((degree + 1) ** 2 - 1):
        raise ValueError(f"Unsupported spherical-harmonic coefficient count: {count}")
    if rest_names != {f"f_rest_{i}" for i in range(count)}:
        raise ValueError("Spherical-harmonic properties must be numbered consecutively from f_rest_0")
    return degree
