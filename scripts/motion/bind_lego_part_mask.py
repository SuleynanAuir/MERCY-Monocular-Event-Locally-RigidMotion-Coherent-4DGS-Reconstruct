"""Bind visible Gaussians to initial-frame part IDs, retaining hidden geometry.

    Changes only ownership of existing splats. Raw events must refine/audit the
    resulting motion; these initial Blender labels aren't event ground truth.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.spatial import cKDTree
import torch
from gaussian_renderer import render
from scripts.rendering.render_ply import make_camera
from scene import GaussianModel
from scripts.motion.rebind_lego_motion import project, sample
from scripts.motion.bind_lego_parts import transform
from utils.rigid_checkpoint import sha256_file


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--parts', type=Path, required=True)
    p.add_argument('--part-mask', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--depth-tolerance', type=float, default=.02)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not np.isfinite(args.depth_tolerance) or args.depth_tolerance <= 0:
        raise ValueError('Invalid visibility tolerance')
    meta = json.loads((args.baseline/'cache.json').read_text())
    with np.load(args.binding, allow_pickle=False) as data:
        binding = {k: data[k] for k in data.files}
    with np.load(args.parts, allow_pickle=False) as data:
        parts = {k: data[k] for k in data.files}
    with np.load(args.part_mask, allow_pickle=False) as data:
        mask = data['part_ids']; mask_meta = json.loads(str(data['metadata_json']))
    part_meta = json.loads(str(parts['metadata_json']))
    if mask.shape != (640,640) or mask_meta['objects'] != part_meta['objects']:
        raise ValueError('Part-mask identity mismatch')
    if abs(mask_meta['fovy_degrees']-meta['camera']['fovy_degrees']) > .1:
        raise ValueError('Part-mask and reconstruction camera FOV disagree')
    model = GaussianModel(3, with_motion_mask=False); model.load_ply(meta['source'])
    if torch.count_nonzero(model._features_rest).item():
        raise ValueError('Expected DC-only source')
    model._features_rest = torch.empty((meta['gaussian_count'],0,3),device='cuda')
    model.active_sh_degree = model.max_sh_degree = 0
    camera = meta['camera']
    view = make_camera(np.asarray(camera['pose']),640,640,np.deg2rad(camera['fovy_degrees']))
    result = render(view, model, SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False),
                    torch.zeros(3,device='cuda'),0.,0.,0.)
    alpha = result['alpha'].squeeze().cpu().numpy()
    depth_image = result['depth'].squeeze().cpu().numpy()/np.maximum(alpha,1e-6)
    xyz = model._xyz.cpu().numpy()
    uv, depth = project(xyz, np.linalg.inv(camera['pose']),320/np.tan(np.deg2rad(camera['fovy_degrees'])/2))
    px = np.round(uv).astype(int)
    inside = (px[:,0]>=0)&(px[:,0]<640)&(px[:,1]>=0)&(px[:,1]<640)&(depth>0)
    proposed = np.full(len(xyz),-1,np.int16)
    proposed[inside] = mask[px[inside,1],px[inside,0]]
    visible = (np.abs(depth-sample(depth_image,uv)) < args.depth_tolerance)&(sample(alpha,uv)>.8)
    change = inside & visible & (proposed>=0) & (proposed!=binding['labels'])
    # Elastic bands retain their local-deformation correspondence; this stage
    # only assigns rigid parts so no band vertex reference can become invalid.
    change &= ~np.isin(binding['labels'],part_meta['deforming_objects'])
    change &= ~np.isin(proposed,part_meta['deforming_objects'])
    surface = transform(parts['rest_points'], binding['registration'])
    prior = json.loads(str(binding['metadata_json']))
    changed = 0
    for label in np.unique(proposed[change]):
        rows = np.flatnonzero(change & (proposed==label))
        points = np.flatnonzero(parts['point_parts']==label)
        distance, nearest = cKDTree(surface[points]).query(xyz[rows],workers=8)
        keep = distance <= prior['max_attachment_distance']
        rows = rows[keep]
        binding['labels'][rows] = label
        binding['nearest_vertex'][rows] = points[nearest[keep]]
        binding['distance'][rows] = distance[keep]
        changed += len(rows)
    matrices = parts['object_matrices'].astype(np.float64)
    delta = matrices @ np.linalg.inv(matrices[0])
    moving = np.max(np.abs(delta-np.eye(4)),axis=(0,2,3))>1e-4
    moving[part_meta['deforming_objects']] = True
    binding['dynamic_indices'] = np.flatnonzero(moving[binding['labels']] &
        (binding['distance']<=prior['max_attachment_distance'])).astype(np.int32)
    prior.update(method='Initial-frame Blender part-ID rays for alpha/depth-visible splats; hidden and elastic correspondences retained.',
                 part_mask_sha256=sha256_file(args.part_mask), changed_gaussians=changed,
                 dynamic_count=len(binding['dynamic_indices']),static_count=len(xyz)-len(binding['dynamic_indices']),
                 depth_tolerance=args.depth_tolerance,event_approved=False)
    binding['metadata_json'] = np.asarray(json.dumps(prior))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**binding)
    args.output.with_suffix('.json').write_text(json.dumps(prior,indent=2)+'\n')
    print(json.dumps({'changed':changed,'dynamic':len(binding['dynamic_indices'])}),flush=True)


if __name__ == '__main__':
    main()
