"""Verified static appearance/coverage parameters over an immutable motion parent."""
import json
import numpy as np
from utils.experiment_format import sha256

FIELDS = {'features_dc':'_features_dc', 'opacity':'_opacity', 'scaling':'_scaling'}


def validate_attribute(field, values, parent):
    if values.shape != parent.shape or values.dtype.kind != 'f' or np.isnan(values).any():
        raise ValueError('Invalid appearance parameter '+field)
    nonfinite=~np.isfinite(values)
    # Supplied PLYs encode exactly zero/one opacity as logits -inf/+inf.
    # Preserve only the identical parent sentinel; never accept new infinities.
    if nonfinite.any() and (field!='opacity' or not np.all(values[nonfinite]==parent[nonfinite])):
        raise ValueError('Invalid appearance parameter '+field)


def apply_stage(model, manifest_path, entry, parent_sha):
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
    return meta
