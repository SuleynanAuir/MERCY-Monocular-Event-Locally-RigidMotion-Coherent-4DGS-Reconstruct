"""Reference-surface reconstruction of Lamp V3; not an event-trained model.

The supplied original GS is an explicit geometric prior. Reliable V3 appearance
residuals are transferred within rigid parts. Tangent-plane quadrature increases
surface sampling without random jitter or cross-joint smoothing.
"""
import numpy as np
from scipy.spatial import cKDTree

from utils.fan_surface import attributes


def reconstruct(reference, corrected, reference_groups, corrected_groups, *, split=4, residual_strength=.35):
    if split not in (1, 2, 4) or not 0 <= residual_strength <= 1:
        raise ValueError('Invalid Lamp reconstruction settings')
    if reference.dtype != corrected.dtype:
        raise ValueError('Reference and corrected Gaussian schemas differ')
    groups = []
    for array, vertices in ((reference_groups, reference), (corrected_groups, corrected)):
        array = np.asarray(array)
        if array.shape != (len(vertices),) or array.dtype.kind not in 'iu' or (array < 0).any():
            raise ValueError('Every Gaussian needs a valid rigid-part identity')
        groups.append(array)
    reference_groups, corrected_groups = groups
    xyz, basis, scales, alpha, color = attributes(reference)
    cx, cb, cs, ca, cc = attributes(corrected)
    normal = basis[np.arange(len(xyz)), :, scales.argmin(1)]
    cn = cb[np.arange(len(cx)), :, cs.argmin(1)]
    new_color = color.copy()
    trusted = np.zeros(len(xyz), bool)
    residual = np.zeros_like(color)
    # Normal/scale/opacity gates reject the noisy, nearly transparent additions
    # instead of transferring their dark speckles onto the clean surface prior.
    for group in np.unique(reference_groups):
        rows = np.flatnonzero(reference_groups == group)
        candidates = np.flatnonzero((corrected_groups == group) & (ca >= .5))
        if len(candidates) < 12:
            continue
        tree = cKDTree(cx[candidates])
        for start in range(0, len(rows), 4096):
            ids = rows[start:start+4096]
            distance, nearest = tree.query(xyz[ids], k=12, workers=4)
            near = candidates[nearest]
            radius = scales[ids].max(1).clip(1e-8)
            alignment = np.abs(np.einsum('bki,bi->bk', cn[near], normal[ids]))
            difference = cc[near]-color[ids, None]
            scale_ratio = cs[near].max(2)/radius[:, None]
            valid = ((distance < 2*radius[:, None]) & (alignment > .92)
                     & (scale_ratio > .2) & (scale_ratio < 3)
                     & (np.linalg.norm(difference, axis=2) < .14))
            weight = np.exp(-.5*(distance/radius[:, None])**2)*ca[near]*valid
            total = weight.sum(1)
            supported = (valid.sum(1) >= 4) & (total > 1)
            change = np.einsum('bk,bki->bi', weight, difference)/total.clip(1e-10)[:, None]
            residual[ids] = np.clip(change, -.04, .04)*residual_strength*supported[:, None]
            trusted[ids] = supported
    new_color += residual
    parents = np.repeat(np.arange(len(reference)), split)
    result = reference[parents].copy()
    target = xyz[parents].copy()
    tangent = np.argsort(scales, axis=1)[:, -2:]
    if split > 1:
        ratio = .65 if split == 2 else .6
        axes = tangent[:, -1:] if split == 2 else tangent
        signs = np.array([[-1], [1]]) if split == 2 else np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        for j in range(split):
            selected = np.arange(len(reference))*split+j
            for k in range(axes.shape[1]):
                axis = axes[:, k]
                offset = basis[np.arange(len(xyz)), :, axis]*(scales[np.arange(len(xyz)), axis]*np.sqrt(1-ratio**2))[:, None]
                target[selected] += signs[j, k]*offset
                for a in range(3):
                    result[f'scale_{a}'][selected[axis == a]] += np.log(ratio)
        # Preserve integrated optical depth over the parent tangent footprint.
        tau = -np.log1p(-np.minimum(alpha, .9999))/(split*ratio**axes.shape[1])
        child_alpha = -np.expm1(-tau)
        result['opacity'] = np.repeat(np.log(child_alpha)-np.log1p(-child_alpha), split)
    for i, name in enumerate(('x', 'y', 'z')):
        result[name] = target[:, i]
    for i in range(3):
        result[f'f_dc_{i}'] = np.repeat((new_color[:, i]-.5)/.28209479177387814, split)
    return result, parents, reference_groups[parents], dict(
        method='Original GS geometry prior + reliable part-local V3 appearance residual + deterministic tangent subdivision',
        input_v3_count=len(corrected), reference_count=len(reference), output_count=len(result),
        children_per_reference=split, trusted_residual_sites=int(trusted.sum()),
        low_opacity_v3_rows=int((ca < .5).sum()), maximum_color_change=float(np.abs(residual).max()),
        residual_strength=residual_strength, motion='Each child inherits its parent rigid part; no motion fit',
        limitation='Geometry-prior reconstruction, not new observed detail or event optimization')
