"""Feature-aware surface reconstruction for display meshes.

Robust local planes supply a static guide, rolling normal filtering supplies
the dynamic guide, and bounded point-to-plane fits update the dense surface.
Inspired by static/dynamic normal filtering (Zhang et al., 2018), not a
reimplementation of its global face-normal optimization. No image filtering.
"""
import numpy as np
from scipy.spatial import cKDTree


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def mesh_normals(vertices, faces):
    result = np.zeros_like(vertices, dtype=np.float64)
    for start in range(0, len(faces), 100000):
        f = faces[start:start+100000]
        p = vertices[f]
        normal = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        for corner in range(3):
            for axis in range(3):
                result[:, axis] += np.bincount(f[:, corner], weights=normal[:, axis], minlength=len(vertices))
    return unit(result)


def preserve_orientation(original, candidate, faces):
    """Backtrack using serialized float32 positions, not just ideal doubles."""
    original = np.asarray(original, dtype=np.float32).astype(np.float64)
    result = np.asarray(candidate, dtype=np.float32).astype(np.float64)
    rejections = np.zeros(len(original), dtype=bool)
    for attempt in range(24):
        bad_ids = []
        for start in range(0, len(faces), 100000):
            f = faces[start:start+100000]
            a, b = original[f], result[f]
            old = np.cross(a[:, 1]-a[:, 0], a[:, 2]-a[:, 0])
            new = np.cross(b[:, 1]-b[:, 0], b[:, 2]-b[:, 0])
            bad = np.sum(old*new, axis=1) < .05*np.sum(old*old, axis=1)
            if np.any(bad): bad_ids.append(f[bad].ravel())
        if not bad_ids:
            return result.astype(np.float32), int(rejections.sum())
        rejected = np.unique(np.concatenate(bad_ids))
        rejections[rejected] = True
        backtracked = original[rejected] + (result[rejected]-original[rejected])*(.5 if attempt < 6 else 0.)
        result[rejected] = backtracked.astype(np.float32).astype(np.float64)
    raise ValueError('Feature reconstruction could not preserve triangle orientation')


def reconstruct_features(vertices, faces, radius, max_move, iterations=4, batch_size=8192):
    """Return bounded geometry and a canonical, edge-aware shading normal field.

    Call separately for each motion group. Oppositely oriented sheets cannot
    contribute to the same fit. The topology, including existing openings, is
    unchanged. Samples set the fitting bandwidth, not output mesh resolution.
    """
    if not np.isfinite([radius, max_move]).all() or not 0 < max_move <= radius or not 1 <= iterations <= 10:
        raise ValueError('Invalid feature reconstruction parameters')
    original = np.asarray(vertices, dtype=np.float64)
    normals = mesh_normals(original, faces)
    valid = np.linalg.norm(normals, axis=1) > .5
    if not valid.any():
        raise ValueError('No nondegenerate surface normals')
    if not valid.all():
        _, nearest = cKDTree(original[valid]).query(original[~valid])
        normals[~valid] = normals[valid][nearest]
    # Keep one actual point per position/orientation cell; averaging opposing
    # blade surfaces in a voxel would destroy thickness before fitting starts.
    keys = np.column_stack((np.floor(original/(radius/4)).astype(np.int64),
                            np.argmax(np.abs(normals), axis=1),
                            (normals[np.arange(len(normals)), np.argmax(np.abs(normals), axis=1)] > 0)))
    _, ids = np.unique(keys, axis=0, return_index=True)
    points, initial = original[ids], normals[ids]
    tree = cKDTree(points)
    k = min(64, len(points))
    dist, neighbors = tree.query(points, k=k, workers=4)
    if k == 1:
        return original.astype(np.float32), normals.astype(np.float32), {'samples': 1, 'max_displacement': 0.}
    centers = points.copy()
    guide = initial.copy()
    confidence = np.zeros(len(points))
    for start in range(0, len(points), batch_size):
        sl = slice(start, start+batch_size)
        p, n = points[sl], initial[sl]
        support = points[neighbors[sl]]
        aligned = np.einsum('bkj,bj->bk', initial[neighbors[sl]], n)
        spatial = np.exp(-.5*(dist[sl]/radius)**2) * (dist[sl] <= 2.5*radius)
        weights = spatial * np.clip((aligned-.15)/.85, 0, 1)**2
        # Iteratively reweighted PCA rejects off-surface points/spikes while
        # estimating a plane. Curved/narrow neighborhoods get lower confidence.
        for _ in range(3):
            w = weights / np.maximum(weights.sum(1, keepdims=True), 1e-12)
            center = np.einsum('bk,bkj->bj', w, support)
            delta = support-center[:, None]
            covariance = np.einsum('bk,bki,bkj->bij', w, delta, delta)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            fitted = eigenvectors[:, :, 0]
            fitted *= np.where(np.sum(fitted*n, axis=1, keepdims=True) < 0, -1, 1)
            residual = np.einsum('bkj,bj->bk', delta, fitted)
            weights = spatial * np.clip((aligned-.15)/.85, 0, 1)**2 / (1+(residual/(radius*.25))**2)
        planar = np.clip(1-eigenvalues[:, 0]/np.maximum(eigenvalues[:, 1]*.45, 1e-15), 0, 1)
        wide = np.clip(eigenvalues[:, 1]/np.maximum(eigenvalues[:, 2]*.15, 1e-15), 0, 1)
        stable = np.clip((np.sum(fitted*n, axis=1)-.25)/.5, 0, 1)
        confidence[sl] = planar*wide*stable
        guide[sl] = unit(fitted*confidence[sl, None]+n*(1-confidence[sl, None]))
        centers[sl] = center
    dynamic = guide.copy()
    # Static guide prevents rolling-filter drift across persistent creases.
    for _ in range(iterations):
        updated = np.empty_like(dynamic)
        for start in range(0, len(points), batch_size):
            sl = slice(start, start+batch_size)
            j = neighbors[sl]
            static_dot = np.einsum('bkj,bj->bk', guide[j], guide[sl])
            dynamic_dot = np.einsum('bkj,bj->bk', dynamic[j], dynamic[sl])
            w = np.exp(-.5*(dist[sl]/radius)**2-(1-static_dot)/.5**2-(1-dynamic_dot)/.35**2)
            w *= (static_dot > .2)*(dist[sl] < 2.5*radius)
            w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
            updated[sl] = unit(.08*guide[sl]+.92*np.einsum('bk,bkj->bj', w, dynamic[j]))
        dynamic = updated
    result, shading = original.copy(), normals.copy()
    for start in range(0, len(original), batch_size):
        sl = slice(start, start+batch_size)
        p, n = original[sl], normals[sl]
        d, j = tree.query(p, k=min(24, len(points)), workers=4)
        normal = dynamic[j]
        aligned = np.einsum('bkj,bj->bk', normal, n)
        w = np.exp(-.5*(d/(radius*.7))**2) * np.clip((aligned-.1)/.9, 0, 1)**2
        w *= (d < 2*radius)
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        # Each nearby plane contributes a normal constraint. The anchor retains
        # tangential detail instead of averaging xyz (which shrinks features).
        fit_weights = w * confidence[j]
        offsets = np.einsum('bkj,bkj->bk', centers[j]-p[:, None], normal)
        fit_weights *= 1/(1+(offsets/max_move)**2)
        matrix = np.einsum('bk,bki,bkj->bij', fit_weights, normal, normal)
        rhs = np.einsum('bk,bk,bkj->bj', fit_weights, offsets, normal)
        displacement = np.linalg.solve(matrix+np.eye(3)*.15, rhs[..., None])[..., 0]
        displacement *= np.minimum(1, max_move/np.maximum(np.linalg.norm(displacement, axis=1, keepdims=True), 1e-12))
        result[sl] += displacement
        filtered = unit(np.einsum('bk,bkj->bj', w, normal))
        shading[sl] = unit(.08*n+.92*filtered)
    result, rejected_count = preserve_orientation(original, result, faces)
    displacement = np.linalg.norm(result-original, axis=1)
    return result.astype(np.float32), shading.astype(np.float32), dict(
        samples=len(points), radius=radius, max_move=max_move,
        max_displacement=float(displacement.max()), mean_displacement=float(displacement.mean()),
        reliable_patch_fraction=float((confidence > .5).mean()),
        foldover_backtracked_vertices=rejected_count, normal_iterations=iterations,
        orientation_guard='serialized-float32')
