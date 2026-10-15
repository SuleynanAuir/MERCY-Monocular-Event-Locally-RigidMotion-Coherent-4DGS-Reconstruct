"""Native surface-detail diagnostics, NOT ground truth for unseen appearance.

Alpha holes, image changes and mesh structural edges are separate measurements.
High contrast alone is not sharpness: unsupported edges lower edge precision.
"""
import numpy as np
from scipy.ndimage import binary_dilation, label, sobel


def _image(value, name):
    value = np.asarray(value)
    if (value.ndim != 2 or value.dtype.kind != 'f' or not np.isfinite(value).all()
            or np.any(value < 0) or np.any(value > 1)):
        raise ValueError(name+' must be a finite native float image in [0,1]')
    return value


def _components(mask):
    ids, count = label(mask, structure=np.ones((3, 3), bool))
    sizes = np.bincount(ids.ravel())[1:]
    return {'pixels': int(mask.sum()), 'components': int(count),
            'largest_component': int(sizes.max()) if sizes.size else 0}


def surface_comparison(before_gray, after_gray, before_alpha, after_alpha,
                       part_ids, target_parts, reference=None, threshold=.5):
    images = [_image(v, n) for v, n in zip(
        (before_gray, after_gray, before_alpha, after_alpha), ('before', 'after', 'before alpha', 'after alpha'))]
    bg, ag, ba, aa = images
    part_ids, target_parts = np.asarray(part_ids), np.asarray(target_parts)
    if (any(v.shape != bg.shape for v in images) or part_ids.shape != bg.shape
            or part_ids.dtype.kind not in 'iu' or np.any(part_ids < -1)
            or target_parts.ndim != 1 or target_parts.dtype.kind not in 'iu'
            or not len(target_parts) or np.any(target_parts < 0)
            or not np.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError('Invalid shared native shape, part identities or alpha threshold')
    target = np.isin(part_ids, target_parts)
    known = part_ids >= 0
    new_holes = (ba >= threshold) & (aa < threshold)
    delta = np.abs(ag-bg)
    result = {'gray_l1_from_parent': float(delta.mean()), 'gray_max_abs_from_parent': float(delta.max()),
        'target_gray_l1_from_parent': float(delta[target].mean()) if target.any() else None,
        'new_alpha_holes_all': _components(new_holes),
        'new_alpha_holes_on_mesh': _components(new_holes & known),
        'new_alpha_holes_on_target': _components(new_holes & target),
        'target_pixels': int(target.sum()),
        'before_target_missing': _components(target & (ba < threshold)),
        'after_target_missing': _components(target & (aa < threshold)),
        'scope': 'Sampled mesh-prior coverage and native image changes, not unseen texture accuracy.'}
    if reference is not None:
        reference = _image(reference, 'reference')
        if reference.shape != bg.shape:
            raise ValueError('Reference must use the same camera and native pixels')
        result['reference_mae'] = {name: {'full_image': float(np.abs(im-reference).mean()),
            'target': float(np.abs(im-reference)[target].mean()) if target.any() else None}
            for name, im in (('before', bg), ('after', ag))}
    return result


def mesh_structure_edges(part_ids, depth, target_parts, depth_jump=.005):
    """Known mesh part/depth discontinuities; NaN off-surface is intentional."""
    part_ids, depth = np.asarray(part_ids), np.asarray(depth)
    if (part_ids.ndim != 2 or part_ids.dtype.kind not in 'iu' or depth.shape != part_ids.shape
            or np.any(part_ids < -1) or not np.isfinite(depth_jump) or depth_jump <= 0):
        raise ValueError('Invalid prior depth/part images')
    hit = part_ids >= 0
    if not np.isfinite(depth[hit]).all() or np.any(depth[hit] <= 0):
        raise ValueError('Prior hit depth must be finite and positive')
    target = np.isin(part_ids, target_parts)
    edge = np.zeros(part_ids.shape, bool)
    for axis in (0, 1):
        a, b = [slice(None), slice(None)], [slice(None), slice(None)]
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        paired = hit[a] & hit[b]
        gap = np.zeros(paired.shape, bool)
        gap[paired] = np.abs(depth[a][paired]-depth[b][paired]) > depth_jump
        change = ((part_ids[a] != part_ids[b]) | gap) & (target[a] | target[b])
        edge[a] |= change; edge[b] |= change
    return edge, binary_dilation(target, iterations=3)


def edge_agreement(gray, expected_edges, region, gradient_threshold=.04, tolerance=2):
    """Fixed-threshold precision/recall against prior structure, not texture.

    Edges from real paint/texture may be unsupported by a geometric prior, so
    this is an auxiliary diagnostic and must never alone select a model.
    """
    gray = _image(gray, 'gray')
    expected_edges, region = np.asarray(expected_edges), np.asarray(region)
    if (expected_edges.shape != gray.shape or region.shape != gray.shape
            or expected_edges.dtype != bool or region.dtype != bool
            or type(tolerance) is not int or not 0 <= tolerance <= 5
            or not np.isfinite(gradient_threshold) or not 0 < gradient_threshold < 1):
        raise ValueError('Invalid edge masks or fixed comparison thresholds')
    observed = np.hypot(sobel(gray, 0, mode='nearest'), sobel(gray, 1, mode='nearest'))/8
    observed = (observed >= gradient_threshold) & region
    expected = expected_edges & region
    near_expected = binary_dilation(expected, iterations=tolerance) if tolerance else expected
    near_observed = binary_dilation(observed, iterations=tolerance) if tolerance else observed
    precision = float((observed & near_expected).sum()/observed.sum()) if observed.any() else 0.
    recall = float((expected & near_observed).sum()/expected.sum()) if expected.any() else 0.
    return {'precision': precision, 'recall': recall,
        'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.,
        'unsupported_edge_pixels': int((observed & ~near_expected).sum()),
        'expected_edge_pixels': int(expected.sum()), 'observed_edge_pixels': int(observed.sum()),
        'gradient_threshold': gradient_threshold, 'tolerance_pixels': tolerance,
        'scope': 'Agreement with sampled mesh structural edges; not unseen RGB ground truth or proof of detail.'}
