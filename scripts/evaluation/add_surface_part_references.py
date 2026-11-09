"""Add visible source-part labels to an existing calibrated RGB reference.

Rigid source triangles and motion are required. Existing foreground masks and
image metrics retain their photometric definitions; part labels are geometric.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.experiment_format import sha256


def prepare(reference, surface, motion, output):
    import torch
    import nvdiffrast.torch as dr
    from utils.viewer_mesh import projection_matrix

    source = json.loads(reference.read_text())
    camera = source['camera']
    with np.load(surface, allow_pickle=False) as mesh, np.load(motion, allow_pickle=False) as animation:
        meta = json.loads(str(mesh['metadata_json']))
        timeline = json.loads(str(animation['metadata_json']))
        if meta['parts_sha256'] != sha256(motion) or meta['objects'] != timeline['objects']:
            raise ValueError('Source surface and motion identities differ')
        if timeline.get('maximum_local_geometry_error', 1) > 1e-6:
            raise ValueError('Rigid source geometry required')
        matrices = animation['object_matrices']
        registration = np.asarray(meta['registration'])
        mapped_camera = registration @ np.asarray(timeline['camera_pose'])
        scale = np.linalg.norm(mapped_camera[:3, 0])
        mapped_camera[:3, :3] /= scale
        if not np.allclose(mapped_camera, camera['pose_opengl_c2w'], atol=1e-6):
            raise ValueError('Source and evaluation cameras differ')
        if abs(camera['fovy_degrees'] - timeline['camera_fovy_degrees']) > 1e-4:
            raise ValueError('Source and evaluation fields of view differ')
        groups = mesh['groups'].astype(np.int64)
        if np.any(groups < 0) or np.any(groups >= len(meta['objects'])):
            raise ValueError('Invalid triangle ownership')
        rest = mesh['triangles'].reshape(-1, 3)
        labels = np.repeat(groups, 3)
        inverse = np.linalg.inv(registration @ matrices[0])
        local = np.einsum('nij,nj->ni', inverse[labels, :3, :3], rest) + inverse[labels, :3, 3]
        if int(animation['source_frames'][0]) != meta['source_frame']:
            raise ValueError('Rest frame mismatch')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    # Copy only files referenced by the reference, preserving all existing data.
    for row in source['samples']:
        for key in ('rgb', 'foreground_mask', 'events', 'part_mask'):
            if row.get(key):
                target = output / row[key]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(reference.parent / row[key], target)
    if source.get('source_motion'):
        target = output / source['source_motion']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference.parent / source['source_motion'], target)
    (output / 'parts').mkdir()
    vertices = torch.tensor(local, dtype=torch.float32, device='cuda')
    ownership = torch.tensor(labels, device='cuda')
    faces = torch.arange(len(local), dtype=torch.int32, device='cuda').reshape(-1, 3)
    face_labels = torch.tensor(groups + 1, dtype=torch.int32, device='cuda')
    pose = torch.tensor(camera['pose_opengl_c2w'], dtype=torch.float32, device='cuda')
    projection = torch.tensor(projection_matrix(camera['width'], camera['height'], camera['fovy_degrees']), device='cuda')
    context = dr.RasterizeCudaContext()
    diagnostics = []
    with torch.no_grad():
        for row in source['samples']:
            frame = row['frame']
            if not 0 <= frame < len(matrices) or abs(row['time_seconds'] - frame / timeline['fps']) > 1e-8:
                raise ValueError('Reference/motion timestamp mismatch')
            transforms = torch.tensor(registration @ matrices[frame], dtype=torch.float32, device='cuda')
            xyz = torch.bmm(transforms[ownership, :3, :3], vertices[..., None])[..., 0] + transforms[ownership, :3, 3]
            view = (xyz - pose[:3, 3]) @ pose[:3, :3]
            clip = torch.cat((view, torch.ones_like(view[:, :1])), -1) @ projection.T
            raster, _ = dr.rasterize(context, clip[None].contiguous(), faces, [camera['height'], camera['width']], grad_db=False)
            face = raster[0, ..., 3].long() - 1
            mask = torch.where(face >= 0, face_labels[face.clamp_min(0)], 0).flip(0).cpu().numpy().astype(np.uint16)
            if not mask.any():
                raise ValueError('Source projection is empty')
            name = f'parts/{frame:06d}.png'
            Image.fromarray(mask).save(output / name)
            row['part_mask'] = name
            fg = np.asarray(Image.open(output / row['foreground_mask'])) > 0
            geometric = mask > 0
            diagnostics.append(dict(frame=frame, foreground_iou=float((fg & geometric).sum() / (fg | geometric).sum()), visible_labels=np.unique(mask[geometric]).tolist()))
    source['part_labels'] = {str(i+1): name for i, name in enumerate(meta['objects'])}
    source['mask_definition'] = 'Part masks: nearest visible original source surface at calibrated pixel centers; 0 background. Existing RGB foreground masks unchanged.'
    source['part_mask_provenance'] = dict(surface_sha256=sha256(surface), motion_sha256=sha256(motion), parent_reference_sha256=sha256(reference), definition=source['mask_definition'], diagnostics=diagnostics)
    source.get('unavailable', {}).pop('part_mask', None)
    (output / 'evaluation.json').write_text(json.dumps(source, indent=2, ensure_ascii=False))
    print(json.dumps(diagnostics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'surface', 'motion', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.reference, args.surface, args.motion, args.output)
