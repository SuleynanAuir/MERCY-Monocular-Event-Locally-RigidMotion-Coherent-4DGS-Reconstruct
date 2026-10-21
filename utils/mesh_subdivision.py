"""Conforming adaptive refinement of an already reconstructed Gaussian surface."""
import numpy as np


def _split_faces(faces, edge_midpoints):
    """Split each shared edge once, including transition faces (no T junctions)."""
    mask = (edge_midpoints >= 0).astype(np.uint8) @ np.array([1, 2, 4], np.uint8)
    parts, parents = [], []
    # Local slots: a,b,c,ab,bc,ca. All child triangles retain parent winding.
    patterns = {
        0: [(0, 1, 2)],
        1: [(0, 3, 2), (3, 1, 2)],
        2: [(1, 4, 0), (4, 2, 0)],
        4: [(2, 5, 1), (5, 0, 1)],
        3: [(1, 4, 3), (0, 3, 2), (3, 4, 2)],
        6: [(2, 5, 4), (1, 4, 0), (4, 5, 0)],
        5: [(0, 3, 5), (2, 5, 1), (5, 3, 1)],
        7: [(0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)],
    }
    for code, pattern in patterns.items():
        ids = np.flatnonzero(mask == code)
        slots = np.concatenate([faces[ids], edge_midpoints[ids]], axis=1)
        for child in pattern:
            parts.append(slots[:, child])
            parents.append(ids)
    return np.concatenate(parts).astype(np.int32), np.concatenate(parents)


def refine_surface(vertices, faces, colors, target_edge, max_offset, max_growth=2.0):
    """Add vertices on long edges with bounded tangent-plane interpolation.

    Original vertices/colors stay exact. Boundary and sharp-feature edges use
    linear midpoints. Curvature comes only from the existing denoised surface;
    subdivision does not recover observations missing from the Gaussian source.
    """
    xyz = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces)
    rgb = np.asarray(colors, dtype=np.float32)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape
            or not np.isfinite(xyz).all() or not np.isfinite(rgb).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype.kind not in 'iu' or faces.min() < 0 or faces.max() >= len(xyz)
            or not np.isfinite([target_edge, max_offset, max_growth]).all()
            or target_edge <= 0 or max_offset < 0 or max_growth < 1):
        raise ValueError('Valid mesh and positive refinement parameters required')
    points = xyz.astype(np.float64)
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    edges, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    face_edges = inverse.reshape(-1, 3)
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    p = points[faces]
    normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    areas = np.linalg.norm(normals, axis=1)
    unit = normals / np.maximum(areas[:, None], 1e-30)
    vertex_normals = np.column_stack([
        np.bincount(faces.ravel(), weights=np.repeat(normals[:, axis], 3), minlength=len(xyz))
        for axis in range(3)])
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1)[:, None], 1e-30)
    # Protect boundaries and creases, including their endpoint normals.
    summed = np.column_stack([
        np.bincount(inverse, weights=np.repeat(unit[:, axis], 3), minlength=len(edges))
        for axis in range(3)])
    smooth = (counts == 2) & (np.linalg.norm(summed, axis=1) > 2 * np.cos(np.deg2rad(17.5)))
    protected = np.zeros(len(xyz), bool)
    protected[edges[~smooth].ravel()] = True
    # Float32 cannot reliably subdivide arbitrarily thin source triangles.
    longest = lengths[face_edges].max(axis=1)
    unsafe_source = areas <= longest * max(target_edge * 1e-4, np.abs(points).max() * 2e-7)
    # Reconstruction can contain coincident duplicate faces. Subdividing those
    # can introduce shared interior edges with ambiguous incidence; preserve
    # these small regions exactly, including all their incident edge splits.
    _, face_ids, face_counts = np.unique(np.sort(faces, axis=1), axis=0,
                                        return_inverse=True, return_counts=True)
    duplicate_source = face_counts[face_ids] > 1
    unsafe_source |= duplicate_source
    eligible = lengths > target_edge
    eligible[face_edges[unsafe_source].ravel()] = False
    selected = np.flatnonzero(eligible)
    # Every edge split adds exactly one child per incident face.
    budget = int(len(faces) * (max_growth - 1))
    if counts[selected].sum() > budget:
        selected = selected[np.argsort(-lengths[selected], kind='stable')]
        selected = selected[np.cumsum(counts[selected]) <= budget]
    selected.sort()
    rejected = 0
    for _ in range(24):
        endpoints = edges[selected]
        a, b = points[endpoints[:, 0]], points[endpoints[:, 1]]
        mid = (a + b) * .5
        ni, nj = vertex_normals[endpoints[:, 0]], vertex_normals[endpoints[:, 1]]
        # Half the average tangent-plane projection: full projection would
        # overshoot a circular arc's midpoint by approximately a factor of two.
        offset = -.25 * (ni * np.sum((mid - a) * ni, axis=1)[:, None]
                         + nj * np.sum((mid - b) * nj, axis=1)[:, None])
        curved = smooth[selected] & ~protected[endpoints].any(axis=1) & (np.sum(ni * nj, axis=1) > .866)
        offset[~curved] = 0
        bound = np.minimum(max_offset, lengths[selected] * .08)
        offset *= np.minimum(1., bound / np.maximum(np.linalg.norm(offset, axis=1), 1e-30))[:, None]
        edge_mid = np.full(len(edges), -1, np.int32)
        edge_mid[selected] = np.arange(len(selected), dtype=np.int32) + len(xyz)
        children, parents = _split_faces(faces, edge_mid[face_edges])
        result = np.concatenate([xyz, (mid + offset).astype(np.float32)])

        def unsafe_children():
            bad = []
            for start in range(0, len(children), 200000):
                child = children[start:start + 200000]
                parent = parents[start:start + len(child)]
                v = result[child].astype(np.float64)
                n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
                dot = np.sum(n * normals[parent], axis=1)
                bad.append((areas[parent] > 0) & (dot <= .001 * areas[parent] ** 2))
            return np.concatenate(bad)

        # Backtrack only new curved vertices; never move original vertices.
        for attempt in range(9):
            bad = unsafe_children()
            if not bad.any():
                break
            affected = np.unique(children[bad])
            affected = affected[affected >= len(xyz)] - len(xyz)
            offset[affected] *= .5 if attempt < 7 else 0.
            result[len(xyz):] = (mid + offset).astype(np.float32)
        else:
            # Even a linear midpoint can collapse at float32 precision. Keep
            # these parent faces unsplit, propagating consistently to neighbors.
            remove = np.unique(face_edges[np.unique(parents[bad])])
            keep = ~np.isin(selected, remove)
            if keep.all():
                raise ValueError('Source triangles cannot be refined safely')
            rejected += int((~keep).sum())
            selected = selected[keep]
            continue
        break
    else:
        raise ValueError('Unable to preserve float32 triangle orientation')
    new_colors = np.concatenate([rgb, rgb[endpoints].mean(axis=1)])
    report = dict(method='Conforming adaptive edge subdivision with bounded tangent interpolation',
                  target_edge=target_edge, max_offset=max_offset, max_growth=max_growth,
                  original_vertices=len(xyz), vertices=len(result), original_triangles=len(faces),
                  triangles=len(children), split_edges=len(selected), rejected_edges=rejected,
                  protected_duplicate_faces=int(duplicate_source.sum()),
                  curved_vertices=int((np.linalg.norm(offset, axis=1) > 1e-12).sum()),
                  max_midpoint_offset=float(np.linalg.norm(offset, axis=1).max()) if len(offset) else 0.,
                  original_vertices_preserved=True, original_colors_preserved=True,
                  orientation_preserved=True, shared_midpoints=True)
    return result, children, new_colors, endpoints, parents, report
