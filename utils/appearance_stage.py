"""Verified appearance and bounded canonical offsets over a fixed motion parent."""
import json
import numpy as np
from utils.experiment_format import sha256

FIELDS = {'features_dc':'_features_dc', 'opacity':'_opacity', 'scaling':'_scaling'}


def reset_geometry(model):
    base = getattr(model, '_geometry_base_xyz', None)
    if base is not None:
        model._xyz = base.clone()


def apply_geometry(model):
    offset = getattr(model, '_geometry_offset_local', None)
    if offset is not None:
        from utils.event_refinement import qrotate
        model._xyz = model._xyz + qrotate(model.get_rotation, offset)


def temporal_basis(frame_count, degree):
    if type(frame_count) is not int or frame_count < 2 or type(degree) is not int or not 1 <= degree <= 8:
        raise ValueError('Invalid temporal appearance basis')
    return np.polynomial.legendre.legvander(np.linspace(-1., 1., frame_count), degree)[:, 1:].astype(np.float32)


def validate_temporal(coefficients, metadata, count):
    basis = temporal_basis(metadata['frame_count'], metadata['degree'])
    if (metadata.get('kind') != 'legendre' or coefficients.shape != (count, metadata['degree'], 3)
            or coefficients.dtype.kind != 'f' or not np.isfinite(coefficients).all()):
        raise ValueError('Invalid temporal appearance coefficients')
    return basis


def apply_frame(model, frame):
    stage = getattr(model, '_temporal_appearance', None)
    if stage is None:
        return
    dc, coefficients, basis = stage
    if not 0 <= frame < len(basis):
        raise ValueError('Temporal appearance frame outside source timeline')
    model._features_dc = dc + (coefficients * basis[frame][None, :, None]).sum(1, keepdim=True)


def validate_attribute(field, values, parent):
    if values.shape != parent.shape or values.dtype.kind != 'f' or np.isnan(values).any():
        raise ValueError('Invalid appearance parameter '+field)
    nonfinite=~np.isfinite(values)
    # Supplied PLYs encode exactly zero/one opacity as logits -inf/+inf.
    # Preserve only the identical parent sentinel; never accept new infinities.
    if nonfinite.any() and (field!='opacity' or not np.all(values[nonfinite]==parent[nonfinite])):
        raise ValueError('Invalid appearance parameter '+field)


def apply_stage(model, manifest_path, entry, parent_sha, frame_count=None):
    import torch
    path = (manifest_path.parent/entry['path']).resolve()
    if sha256(path) != entry['sha256']: raise ValueError('Appearance stage checksum mismatch')
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata_json']))
        if meta.get('format') != 'scgs_appearance_stage_v1' or meta['parent_sha256'] != parent_sha:
            raise ValueError('Appearance stage belongs to a different motion parent')
        with torch.no_grad():
            for field, attr in FIELDS.items():
                values=data[field]
                validate_attribute(field, values, getattr(model, attr).detach().cpu().numpy())
                setattr(model, attr, torch.tensor(values, device='cuda'))
            if meta.get('temporal'):
                if frame_count is None or meta['temporal']['frame_count'] != frame_count:
                    raise ValueError('Temporal appearance timeline does not match parent')
                coefficients = data['temporal_dc']
                basis = validate_temporal(coefficients, meta['temporal'], len(model._xyz))
                model._temporal_appearance = (model._features_dc,
                    torch.tensor(coefficients, device='cuda'), torch.tensor(basis, device='cuda'))
            if meta.get('geometry_refinement'):
                offset = data['geometry_offset_local']
                limit = meta['geometry_refinement']['max_offset_scene_units']
                if (offset.shape != tuple(model._xyz.shape) or offset.dtype.kind != 'f'
                        or not np.isfinite(offset).all() or not np.isfinite(limit) or limit <= 0
                        or np.any(np.linalg.norm(offset, axis=1) > limit*(1+1e-5))):
                    raise ValueError('Invalid bounded geometry offset')
                model._geometry_base_xyz = model._xyz.detach().clone()
                model._geometry_offset_local = torch.tensor(offset, device='cuda')
    return meta
