"""Part-local regularization of noisy reconstructed surfaces for clay display."""
import numpy as np
from scipy.spatial import cKDTree


def regularize_part(vertices, faces, spacing, max_gap, iterations=2):
    """Fit a smooth surface near the observed mesh, without crossing motion groups.

    Coarse oriented samples suppress sub-voxel ripples before Poisson fitting.
    Extrapolation is limited by distance to the input surface neighborhood.
    This is a display reconstruction, not new measured geometric evidence.
    """
    import open3d as o3d
    if not 0 < spacing <= max_gap or not 0 <= iterations <= 5:
        raise ValueError('Invalid surface regularization scales')
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(faces))
    mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    cloud = o3d.geometry.PointCloud()
    cloud.points, cloud.normals = mesh.vertices, mesh.vertex_normals
    cloud = cloud.voxel_down_sample(spacing)
    points, normals = np.asarray(cloud.points).copy(), np.asarray(cloud.normals).copy()
    valid = np.linalg.norm(normals, axis=1) > .35
    points, normals = points[valid], normals[valid]
    if len(points) < 50:
        return np.asarray(mesh.vertices).astype(np.float32), np.asarray(mesh.triangles).astype(np.int32), {'fallback': 'small part'}
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    original = points.copy()
    # A local plane fit follows the broad surface while retaining edges where
    # normal directions disagree. Bounded displacement prevents part shrinkage.
    for _ in range(iterations):
        dist, neighbors = cKDTree(points).query(points, k=min(24, len(points)), workers=4)
        alignment = np.einsum('nij,nj->ni', normals[neighbors], normals)
        w = np.exp(-.5*(dist/(spacing*1.6))**2) * np.clip((alignment-.25)/.75, 0, 1)**2
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        mean = np.einsum('ni,nij->nj', w, points[neighbors])
        n = np.einsum('ni,nij->nj', w, normals[neighbors])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        projected = points + .65*np.sum((mean-points)*n, axis=1, keepdims=True)*n
        delta = projected-original
        points = original + delta*np.minimum(1., spacing/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12))
        normals = n
    cloud.points, cloud.normals = o3d.utility.Vector3dVector(points), o3d.utility.Vector3dVector(normals)
    depth = int(np.clip(np.ceil(np.log2(np.ptp(points,axis=0).max()*1.05/spacing)), 6, 10))
    fitted, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=depth, scale=1.05, linear_fit=True, n_threads=4)
    # Reject extrapolated vertices before and after smoothing. Large mechanical
    # openings stay open; only gaps within the explicit support radius can fill.
    source_tree = cKDTree(np.asarray(mesh.vertices))
    distance, _ = source_tree.query(np.asarray(fitted.vertices), workers=4)
    fitted.remove_vertices_by_mask(distance > max_gap)
    fitted.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    fitted = fitted.filter_smooth_taubin(number_of_iterations=5,
                                         filter_scope=o3d.geometry.FilterScope.Vertex)
    distance, _ = source_tree.query(np.asarray(fitted.vertices), workers=4)
    fitted.remove_vertices_by_mask(distance > max_gap)
    fitted.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    labels, counts, areas = fitted.cluster_connected_triangles()
    labels, counts, areas = np.asarray(labels), np.asarray(counts), np.asarray(areas)
    # Remove isolated subpixel fragments, never a complete small motion part.
    keep = (areas >= spacing**2*6) | (counts >= 100)
    if len(keep): keep[np.argmax(areas)] = True
    removed = int(np.sum(~keep[labels]))
    fitted.remove_triangles_by_mask(~keep[labels])
    fitted.remove_unreferenced_vertices()
    result = np.asarray(fitted.vertices).astype(np.float32)
    distance, _ = source_tree.query(result, workers=4)
    report = dict(input_vertices=len(vertices), input_triangles=len(faces), sample_count=len(points),
        output_vertices=len(result), output_triangles=len(fitted.triangles), spacing=spacing,
        maximum_support_distance=max_gap, actual_max_distance=float(distance.max()),
        mean_distance=float(distance.mean()), removed_fragment_faces=removed, poisson_depth=depth,
        component_count_before=len(counts), component_count_after=int(keep.sum()))
    return result, np.asarray(fitted.triangles).astype(np.int32), report
