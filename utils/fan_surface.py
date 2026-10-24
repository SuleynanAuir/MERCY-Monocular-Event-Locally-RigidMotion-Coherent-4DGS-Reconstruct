"""Canonical, part-local surface regularization of an existing Fan Gaussian asset.

This is a geometry prior, not new observations or an event-supervised fit.
Motion is inherited from the original rigid groups, including split children.
"""
import numpy as np
import json
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def validate_refinement(data, original_groups, original_xyz, expected):
    """Reject stale sources, cross-part children and unbounded geometry edits."""
    record = json.loads(str(data['metadata_json']))
    if record.get('format') != 'fan_surface_refinement_v1':
        raise ValueError('Unsupported Fan surface refinement')
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('Fan surface source fingerprint mismatch')
    vertices, parents, groups = [data[k] for k in ('base_vertices', 'parent_rows', 'groups')]
    if (parents.shape != (len(vertices),) or parents.dtype.kind not in 'iu'
            or not len(parents) or parents.min() < 0 or parents.max() >= len(original_groups)
            or np.any(np.diff(parents.astype(np.int64)) < 0)
            or groups.shape != parents.shape or groups.dtype.kind not in 'iu'
            or not np.array_equal(groups, np.asarray(original_groups)[parents])):
        raise ValueError('Refinement crosses motion groups or has invalid parent rows')
    xyz, _, scales, _, _ = attributes(vertices)
    report = record['report']
    if (report['gaussians_before'] != len(original_groups) or report['gaussians_after'] != len(vertices)
            or len(vertices) > 2*len(original_groups) or not 0 < report['maximum_shift'] <= .005):
        raise ValueError('Invalid refinement size or displacement bound')
    # A child may move by the regularization bound plus its parent's split
    # offset; largest child sigma is 0.75 of that parent's sigma.
    limit = report['maximum_shift']+scales.max(1)*np.sqrt(1-.75**2)/.75+2e-6
    if np.any(np.linalg.norm(xyz-np.asarray(original_xyz)[parents], axis=1) > limit):
        raise ValueError('Unbounded Fan surface displacement')
    return record


def attributes(vertices):
    xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    quat = np.column_stack([vertices[f'rot_{i}'] for i in range(4)]).astype(np.float64)
    scales = np.exp(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]).astype(np.float64))
    alpha = 1 / (1 + np.exp(-np.clip(vertices['opacity'].astype(np.float64), -40, 40)))
    color = .5 + .28209479177387814 * np.column_stack([vertices[f'f_dc_{i}'] for i in range(3)])
    if (not all(np.isfinite(a).all() for a in (xyz, quat, scales, alpha, color))
            or np.any(np.linalg.norm(quat, axis=1) < 1e-10) or np.any(scales <= 0)):
        raise ValueError('Invalid Gaussian surface attributes')
    basis = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
    return xyz, basis, scales, alpha, color


def regularize(vertices, groups, *, strength=.65, split_budget=300000, neighbors=32,
               maximum_shift=.00065, color_strength=.2, batch_size=4096):
    """Robust bilateral MLS, with edge/line protection and deterministic splitting.

    Neighbors never cross motion groups. Ill-conditioned, thin or boundary
    neighborhoods are left unchanged. Splits conserve center and covariance
    moments and approximately conserve integrated optical density; they are
    not claimed to produce identical alpha-composited images.
    """
    if (not 0 <= strength <= 1 or not 0 <= color_strength <= 1
            or not 0 < maximum_shift <= .005 or split_budget < 0 or neighbors < 8):
        raise ValueError('Invalid surface refinement settings')
    groups = np.asarray(groups)
    if groups.shape != (len(vertices),) or groups.dtype.kind not in 'iu':
        raise ValueError('Each Gaussian needs an integer motion group')
    xyz, basis, scales, alpha, color = attributes(vertices)
    axis = scales.argmin(1)
    normal = basis[np.arange(len(xyz)), :, axis]
    target, new_normal, new_color = xyz.copy(), normal.copy(), color.copy()
    confidence = np.zeros(len(xyz)); spacing = np.zeros(len(xyz))
    support = np.zeros(len(xyz), np.int32)
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        if len(ids) < neighbors: continue
        tree = cKDTree(xyz[ids])
        for start in range(0, len(ids), batch_size):
            rows = ids[start:start + batch_size]
            distances, local = tree.query(xyz[rows], k=neighbors, workers=4)
            near = ids[local]; delta = xyz[near] - xyz[rows, None]
            radius = np.maximum(distances[:, -1], 1e-8)
            spacing[rows] = distances[:, 1:5].mean(1)
            alignment = np.abs(np.einsum('bki,bi->bk', normal[near], normal[rows]))
            dc = np.linalg.norm(color[near] - color[rows, None], axis=2)
            weight = np.exp(-.5*(distances/(radius[:, None]*.6))**2 - .5*(dc/.12)**2)
            weight *= (alignment > .75)*np.clip(alpha[near], .05, 1)
            # Trim parallel layers and distant normal outliers before fitting.
            height = np.einsum('bki,bi->bk', delta, normal[rows])
            median = np.median(height, axis=1)
            mad = np.median(np.abs(height-median[:, None]), axis=1)
            weight *= np.abs(height-median[:, None]) < np.maximum(3*mad, radius*.08)[:, None]
            total = weight.sum(1).clip(1e-10)
            center = np.einsum('bk,bki->bi', weight, delta)/total[:, None]
            centered = delta-center[:, None]
            covariance = np.einsum('bk,bki,bkj->bij', weight, centered, centered)/total[:, None, None]
            eig, vectors = np.linalg.eigh(covariance)
            plane = vectors[:, :, 0]
            plane *= np.where((plane*normal[rows]).sum(1) < 0, -1., 1.)[:, None]
            planarity = eig[:, 0]/eig[:, 1].clip(1e-15)
            line_ratio = eig[:, 1]/eig[:, 2].clip(1e-15)
            tangent_center = center-(center*plane).sum(1)[:, None]*plane
            count = (weight > .1).sum(1); support[rows] = count
            safe = ((planarity < .15) & (line_ratio > .18) & (count >= 10)
                    & (np.linalg.norm(tangent_center, axis=1) < radius*.35)
                    & ((plane*normal[rows]).sum(1) > .8) & (alpha[rows] > .1))
            trust = safe*np.clip(1-planarity/.15, 0, 1)
            confidence[rows] = trust
            shift = np.clip((center*plane).sum(1), -maximum_shift, maximum_shift)*strength*trust
            target[rows] += shift[:, None]*plane
            new_normal[rows] += (plane-normal[rows])*(strength*trust)[:, None]
            # Very small chromatic correction, inside coherent opaque patches.
            mean_color = np.einsum('bk,bki->bi', weight, color[near])/total[:, None]
            change = np.clip(mean_color-color[rows], -.12, .12)
            new_color[rows] += change*(color_strength*strength*trust)[:, None]
    new_normal /= np.linalg.norm(new_normal, axis=1, keepdims=True)
    cross = np.cross(normal, new_normal); sine = np.linalg.norm(cross, axis=1)
    angle = np.arctan2(sine, np.clip((normal*new_normal).sum(1), -1, 1))
    correction = Rotation.from_rotvec(cross*(angle/np.maximum(sine, 1e-15))[:, None])
    refined_basis = correction.as_matrix() @ basis
    result = vertices.copy()
    for i, k in enumerate(('x', 'y', 'z')): result[k] = target[:, i]
    q = Rotation.from_matrix(refined_basis).as_quat()[:, [3, 0, 1, 2]]
    for i in range(4): result[f'rot_{i}'] = q[:, i]
    for i in range(3): result[f'f_dc_{i}'] = (new_color[:, i]-.5)/.28209479177387814
    # Only negligible-opacity primitives are removed; high-opacity detached
    # geometry needs additional observations and is not blindly erased.
    keep = alpha >= .008
    area = np.prod(np.sort(scales, axis=1)[:, 1:], axis=1)
    candidates = np.flatnonzero(keep & (confidence > .4) & (alpha > .4))
    order = np.argsort(-area[candidates], kind='stable')
    split = candidates[order[:split_budget]]
    multiplicity = keep.astype(np.int32); multiplicity[split] = 2
    parents = np.repeat(np.arange(len(vertices)), multiplicity)
    result = result[parents].copy()
    first = np.cumsum(multiplicity)-multiplicity
    largest = scales[split].argmax(1); ratio = .75
    offset = (refined_basis[split, :, largest]
              * (scales[split, largest]*np.sqrt(1-ratio**2))[:, None])
    for sign, indices in ((-1, first[split]), (1, first[split]+1)):
        for i, k in enumerate(('x', 'y', 'z')): result[k][indices] = target[split, i]+sign*offset[:, i]
        for i in range(3):
            chosen = indices[largest == i]
            result[f'scale_{i}'][chosen] += np.log(ratio)
        optical_depth = -np.log1p(-np.minimum(alpha[split], 1-1e-7)) / (2*ratio)
        a = -np.expm1(-optical_depth)
        result['opacity'][indices] = np.log(a)-np.log1p(-a)
    report = dict(gaussians_before=len(vertices), gaussians_after=len(result), split_parents=len(split),
                  pruned_low_opacity=int((~keep).sum()), regularized=int((confidence > 0).sum()),
                  maximum_center_shift=float(np.linalg.norm(target-xyz, axis=1).max()),
                  median_center_shift=float(np.median(np.linalg.norm(target-xyz, axis=1))),
                  strength=strength, maximum_shift=maximum_shift, neighbors=neighbors, color_strength=color_strength,
                  note='Canonical geometry prior; no new image/event supervision. Splits inherit rigid motion.')
    return result, parents, groups[parents], report
