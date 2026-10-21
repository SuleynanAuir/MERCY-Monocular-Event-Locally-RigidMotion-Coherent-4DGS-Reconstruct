"""Bounded surface denoising on mesh edges; never connect separate surfaces."""
import numpy as np
from scipy.sparse import coo_matrix, diags


def denoise_surface(vertices, faces, colors, voxel, iterations=16, color_iterations=3):
    """Taubin geometry filtering and bilateral color filtering with fixed limits.

    Open boundaries move at most one voxel, other vertices at most two. No
    triangles are added, so this cannot bridge holes or join nearby components.
    Color weights use the original colors to keep material boundaries stable.
    """
    if not np.isfinite(voxel) or voxel <= 0 or iterations < 0 or color_iterations < 0:
        raise ValueError('Positive voxel size and nonnegative iteration counts required')
    original = np.asarray(vertices, np.float64)
    colors = np.asarray(colors, np.float64)
    faces = np.asarray(faces)
    if (original.ndim != 2 or original.shape[1] != 3 or colors.shape != original.shape
            or not np.isfinite(original).all() or not np.isfinite(colors).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype.kind not in 'iu' or faces.min() < 0 or faces.max() >= len(original)):
        raise ValueError('Invalid mesh arrays')
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = np.zeros(len(original), bool)
    boundary[edges[counts != 2].ravel()] = True
    i, j = edges.T
    length = np.linalg.norm(original[i] - original[j], axis=1)
    # Long triangles must not propagate noise across an under-observed gap.
    weights = np.exp(-.5 * (length / (2 * voxel)) ** 2) / np.maximum(length, voxel * .1)

    def normalized_graph(w, self_weight=0.):
        graph = coo_matrix((np.concatenate([w, w]),
                            (np.concatenate([i, j]), np.concatenate([j, i]))),
                           shape=(len(original), len(original))).tocsr()
        degree = np.asarray(graph.sum(axis=1)).ravel()
        graph += diags(np.where(degree > 0, degree * self_weight, 1.))
        return diags(1 / np.asarray(graph.sum(axis=1)).ravel()) @ graph

    graph = normalized_graph(weights)
    limit = np.where(boundary, voxel, 2 * voxel)

    def bounded(v):
        delta = v - original
        scale = np.minimum(1., limit / np.maximum(np.linalg.norm(delta, axis=1), 1e-15))
        return original + delta * scale[:, None]

    xyz = original.copy()
    before = float(np.linalg.norm(graph @ xyz - xyz, axis=1).mean())
    for _ in range(iterations):
        xyz = bounded(xyz + .5 * (graph @ xyz - xyz))
        xyz = bounded(xyz - .53 * (graph @ xyz - xyz))
    # Thin, irregular TSDF triangles can turn over even with a small movement.
    # Backtrack only affected vertices until every nondegenerate face keeps
    # its orientation and at least 2% of its original projected area.
    def unsafe_vertices():
        affected = np.zeros(len(original), bool)
        for start in range(0, len(faces), 200000):
            chunk = faces[start:start + 200000]
            # Validate the float32 coordinates that will actually be saved and
            # rasterized; near-degenerate faces are sensitive to quantization.
            a = original[chunk]
            b = xyz[chunk].astype(np.float32).astype(np.float64)
            n = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
            m = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
            area_squared = np.sum(n * n, axis=1)
            unsafe = (area_squared > 0) & (np.sum(n * m, axis=1) < .02 * area_squared)
            affected[chunk[unsafe].ravel()] = True
        return affected

    guarded = np.zeros(len(original), bool)
    for _ in range(24):
        unsafe = unsafe_vertices()
        if not unsafe.any():
            break
        guarded |= unsafe
        xyz[unsafe] = .5 * (xyz[unsafe] + original[unsafe])
    else:
        # Pathological geometry should retain its source shape, not fold over.
        xyz = original.copy()
    after = float(np.linalg.norm(graph @ xyz - xyz, axis=1).mean())
    color_delta = np.mean((colors[i] - colors[j]) ** 2, axis=1)
    color_graph = normalized_graph(weights * np.exp(-color_delta / (2 * .10 ** 2)), self_weight=1.)
    rgb = colors.copy()
    for _ in range(color_iterations):
        rgb = color_graph @ rgb
    distance = np.linalg.norm(xyz - original, axis=1)
    report = dict(method='Bounded Taubin geometry and edge-aware bilateral vertex colors',
                  iterations=iterations, color_iterations=color_iterations, voxel_size=voxel,
                  max_displacement=float(distance.max()), mean_displacement=float(distance.mean()),
                  boundary_max_displacement=float(distance[boundary].max()) if boundary.any() else 0.,
                  laplacian_before=before, laplacian_after=after,
                  guarded_vertices=int(guarded.sum()), orientation_preserved=True,
                  topology_preserved=True, color_sigma=.10)
    return xyz.astype(np.float32), np.clip(rgb, 0, 1).astype(np.float32), report
