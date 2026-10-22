"""Do not use foreground RGB tracks to relabel occluded Gaussian surfaces."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from plyfile import PlyData
import torch
from gaussian_renderer import render
from render_ply import make_camera
from scene import GaussianModel
from scripts.motion.rebind_lego_motion import project, sample


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--original', required=True, type=Path)
    parser.add_argument('--proposal', required=True, type=Path)
    parser.add_argument('--parts', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--depth-tolerance', type=float, default=.015)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not np.isfinite(args.depth_tolerance) or args.depth_tolerance <= 0:
        raise ValueError('Invalid depth tolerance')
    meta = json.loads((args.baseline / 'cache.json').read_text())
    with np.load(args.original, allow_pickle=False) as data:
        original = {key: data[key] for key in data.files}
    with np.load(args.proposal, allow_pickle=False) as data:
        proposal = {key: data[key] for key in data.files}
    model = GaussianModel(3, with_motion_mask=False)
    model.load_ply(meta['source'])
    if torch.count_nonzero(model._features_rest).item() != 0:
        raise ValueError('Requires DC-only color')
    model._features_rest = torch.empty((meta['gaussian_count'], 0, 3), device='cuda')
    model.active_sh_degree = model.max_sh_degree = 0
    camera_meta = meta['camera']
    camera = make_camera(np.asarray(camera_meta['pose']), 640, 640, np.deg2rad(camera_meta['fovy_degrees']))
    result = render(camera, model, SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False),
                    torch.zeros(3, device='cuda'), 0., 0., 0.)
    # This rasterizer returns alpha-weighted depth; normalize it before testing visibility.
    alpha = result['alpha'].squeeze().cpu().numpy()
    surface_depth = result['depth'].squeeze().cpu().numpy() / np.maximum(alpha, 1e-6)
    xyz = model._xyz.cpu().numpy()
    uv, depth = project(xyz, np.linalg.inv(camera_meta['pose']),
                        320 / np.tan(np.deg2rad(camera_meta['fovy_degrees']) / 2))
    gap = np.abs(depth - sample(surface_depth, uv))
    visible = (gap <= args.depth_tolerance) & (sample(alpha, uv) > .8)
    changed = proposal['labels'] != original['labels']
    rejected = changed & ~visible
    for key in ('labels', 'nearest_vertex', 'distance'):
        proposal[key][rejected] = original[key][rejected]
    with np.load(args.parts, allow_pickle=False) as parts:
        matrices = parts['object_matrices'].astype(np.float64)
        part_meta = json.loads(str(parts['metadata_json']))
    delta = matrices @ np.linalg.inv(matrices[0])
    moving = np.max(np.abs(delta - np.eye(4)), axis=(0, 2, 3)) > 1e-4
    moving[part_meta['deforming_objects']] = True
    report = json.loads(str(proposal['metadata_json']))
    proposal['dynamic_indices'] = np.flatnonzero(moving[proposal['labels']] &
                                                (proposal['distance'] <= report['max_attachment_distance'])).astype(np.int32)
    report.update(visibility_filter='Only relabel Gaussians near alpha-normalized initial rendered depth; preserve hidden bindings.',
                  depth_tolerance=args.depth_tolerance, occluded_reassignments_reverted=int(rejected.sum()),
                  changed_gaussians=int(np.sum(proposal['labels'] != original['labels'])),
                  dynamic_count=len(proposal['dynamic_indices']), static_count=len(xyz)-len(proposal['dynamic_indices']),
                  proposal_path=str(args.proposal))
    proposal['metadata_json'] = np.asarray(json.dumps(report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **proposal)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('occluded_reassignments_reverted', 'changed_gaussians', 'dynamic_count')}, indent=2))


if __name__ == '__main__':
    main()
