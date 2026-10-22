"""Propose coherent rigid motion binding from RGB tracks; raw events must validate it later.

Does not alter Gaussian appearance, rest geometry, or Blender animation. Hidden
geometry is not established by the single-view tracks. Elastic bands keep their
existing binding. No proposal is called event-approved by this script.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from plyfile import PlyData
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scripts.motion.bind_lego_parts import transform


def sample(image, xy):
    h, w = image.shape[:2]
    x, y = np.clip(xy[:, 0], 0, w - 1.001), np.clip(xy[:, 1], 0, h - 1.001)
    ix, iy = x.astype(int), y.astype(int)
    dx, dy = x - ix, y - iy
    if image.ndim == 3:
        dx, dy = dx[:, None], dy[:, None]
    return ((1-dx)*(1-dy)*image[iy, ix] + dx*(1-dy)*image[iy, ix+1]
            + (1-dx)*dy*image[iy+1, ix] + dx*dy*image[iy+1, ix+1])


def project(xyz, inverse, focal, width=640):
    view = transform(xyz, inverse)
    depth = -view[:, 2]
    uv = focal * view[:, :2] / np.maximum(depth[:, None], .001)
    uv[:, 1] *= -1
    return uv + (width - 1) / 2, depth


def tracks(images, frames):
    cv2.setNumThreads(4)
    h, w = images.shape[1:]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    current = np.stack((xx, yy), -1)
    confidence = np.ones((h, w), np.float32)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    saved, weights = [], []
    for index in range(1, max(frames) + 1):
        forward = flow.calc(images[index-1], images[index], None)
        backward = flow.calc(images[index], images[index-1], None)
        step = cv2.remap(forward, current[..., 0], current[..., 1], cv2.INTER_LINEAR)
        after = current + step
        reverse = cv2.remap(backward, after[..., 0], after[..., 1], cv2.INTER_LINEAR)
        fb = np.linalg.norm(step + reverse, axis=-1)
        inside = (after[..., 0] >= 1) & (after[..., 0] < w-2) & (after[..., 1] >= 1) & (after[..., 1] < h-2)
        confidence *= np.exp(-np.minimum(fb, 5)**2 / 4) * inside
        current = after
        if index in frames:
            saved.append(current.copy()); weights.append(confidence.copy())
            print('Tracked RGB motion to frame', index, 'reliable pixels', int((confidence > .25).sum()), flush=True)
    return np.asarray(saved), np.asarray(weights)


def tracks_raft(images, frames):
    """Four-frame bidirectional tracks avoid multiplying 80 noisy tiny steps.

    Torchvision's published RAFT-small weights are only a correspondence prior;
    they never replace raw-event supervision or the final Gaussian render audit.
    """
    import torch
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    cv2.setNumThreads(4)
    weights = Raft_Small_Weights.DEFAULT
    network = raft_small(weights=weights, progress=True).cuda().eval()
    h, w = images.shape[1:]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    current = np.stack((xx, yy), -1)
    confidence = np.ones((h, w), np.float32)
    saved, certainty, previous = [], [], 0
    with torch.inference_mode():
        for index in sorted(set(range(4, max(frames)+1, 4)) | set(frames)):
            pair = torch.from_numpy(images[[previous, index]].copy()).cuda()[:, None].repeat(1, 3, 1, 1)
            first, second = weights.transforms()(pair, pair.flip(0))
            flow = network(first, second, num_flow_updates=24)[-1].permute(0, 2, 3, 1).cpu().numpy()
            step = cv2.remap(flow[0], current[..., 0], current[..., 1], cv2.INTER_LINEAR)
            after = current + step
            reverse = cv2.remap(flow[1], after[..., 0], after[..., 1], cv2.INTER_LINEAR)
            fb = np.linalg.norm(step+reverse, axis=-1)
            inside = (after[..., 0] >= 1) & (after[..., 0] < w-2) & (after[..., 1] >= 1) & (after[..., 1] < h-2)
            confidence *= np.exp(-np.minimum(fb, 5)**2 / 4) * inside
            current, previous = after, index
            if index in frames:
                saved.append(current.copy()); certainty.append(confidence.copy())
                print('RAFT tracked to', index, 'reliable pixels', int((confidence > .25).sum()), flush=True)
    del network
    torch.cuda.empty_cache()
    return np.asarray(saved), np.asarray(certainty)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--parts', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--supervision', type=Path, required=True)
    parser.add_argument('--correction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometry-weight', type=float, default=4.)
    parser.add_argument('--flow', choices=('dis', 'raft-small'), default='dis')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads((args.baseline / 'cache.json').read_text())
    with np.load(args.parts, allow_pickle=False) as archive:
        parts = {key: archive[key] for key in archive.files}
    with np.load(args.binding, allow_pickle=False) as archive:
        binding = {key: archive[key] for key in archive.files}
    with np.load(args.supervision, allow_pickle=False) as archive:
        images = archive['images']
    prior = json.loads(str(binding['metadata_json']))
    part_meta = json.loads(str(parts['metadata_json']))
    frames = [8, 16, 24, 36, 50, 65, 80]
    trajectory, confidence = (tracks_raft if args.flow == 'raft-small' else tracks)(images, frames)
    vertices = PlyData.read(meta['source'])['vertex'].data
    xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')])
    camera = meta['camera']
    inverse = np.linalg.inv(camera['pose'])
    focal = 320 / np.tan(np.deg2rad(camera['fovy_degrees']) / 2)
    uv, depth = project(xyz, inverse, focal)
    foreground = sample(images[0], uv) > 12
    candidates = np.flatnonzero(foreground & (uv[:, 0] > 1) & (uv[:, 0] < 638)
                               & (uv[:, 1] > 1) & (uv[:, 1] < 638) & (depth > .01)
                               & ~np.isin(binding['labels'], part_meta['deforming_objects']))
    xy, points = uv[candidates], xyz[candidates]
    tracked = np.array([sample(image, xy) for image in trajectory], np.float32)
    weights = np.array([sample(image, xy) for image in confidence], np.float32)
    weights[weights < .1] = 0
    total_weight = weights.sum(0)
    reliable = (weights > .25).sum(0) >= 3
    matrices = parts['object_matrices'].astype(np.float64)
    delta = matrices @ np.linalg.inv(matrices[0])
    features = delta[frames, :, :3, :].transpose(1, 0, 2, 3).reshape(len(matrices[0]), -1)
    group = fclusterdata(features, .002, criterion='distance', method='complete') - 1
    a = binding['registration']
    world_delta = a @ delta @ np.linalg.inv(a)
    correction = json.loads(args.correction.read_text())
    c = np.eye(4)
    q = np.asarray(correction['rotation_wxyz'])
    c[:3, :3] = correction['scale'] * Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    c[:3, 3] = correction['translation']
    world_delta = c @ world_delta @ np.linalg.inv(c)
    aligned_surface = transform(parts['rest_points'], a)
    old_group = group[binding['labels'][candidates]]
    groups = np.unique(group)
    costs = np.empty((len(groups), len(candidates)), np.float32)
    distances = np.empty_like(costs)
    nearest = np.empty(costs.shape, np.int32)
    for g in groups:
        labels = np.flatnonzero(group == g)
        rigid_labels = labels[~np.isin(labels, part_meta['deforming_objects'])]
        if not len(rigid_labels):
            raise ValueError('Motion cluster contains only deforming objects')
        surface_rows = np.flatnonzero(np.isin(parts['point_parts'], rigid_labels))
        distance, neighbor = cKDTree(aligned_surface[surface_rows]).query(points, workers=8)
        nearest[g] = surface_rows[neighbor]
        distances[g] = distance
        reprojection = np.zeros(len(points), np.float32)
        for j, frame in enumerate(frames):
            predicted, _ = project(transform(points, world_delta[frame, rigid_labels[0]]), inverse, focal)
            residual = np.linalg.norm(predicted - tracked[j], axis=-1)
            reprojection += weights[j] * np.minimum(residual, 20)**2
        costs[g] = reprojection / np.maximum(total_weight, .1) + args.geometry_weight * (distance / .03)**2
        print('Motion group', int(g), 'parts', len(labels), flush=True)
    best = np.argmin(costs, axis=0)
    rows = np.arange(len(points))
    gain = costs[old_group, rows] - costs[best, rows]
    change = reliable & (gain > 2.) & (distances[best, rows] < .065) & (costs[best, rows] < 36.)
    selected = candidates[change]
    new_nearest = nearest[best[change], rows[change]]
    old_labels = binding['labels'].copy()
    labels = old_labels.copy()
    labels[selected] = parts['point_parts'][new_nearest]
    binding['nearest_vertex'][selected] = new_nearest
    binding['distance'][selected] = distances[best[change], rows[change]]
    moving = np.max(np.abs(delta - np.eye(4)), axis=(0, 2, 3)) > 1e-4
    moving[part_meta['deforming_objects']] = True
    binding['labels'] = labels
    binding['dynamic_indices'] = np.flatnonzero(moving[labels] & (binding['distance'] <= prior['max_attachment_distance'])).astype(np.int32)
    report = {**prior, 'method': 'RGB forward/backward consistent tracks select rigid motion groups, constrained by surface distance. Must subsequently optimize and audit raw events.',
              'prior_binding': str(args.binding), 'changed_gaussians': int(np.sum(group[labels] != group[old_labels])),
              'tracked_frames': frames, 'geometry_weight': args.geometry_weight, 'flow_backend': args.flow,
              'dynamic_count': len(binding['dynamic_indices']), 'static_count': len(labels) - len(binding['dynamic_indices']),
              'event_approved': False,
              'limitations': 'Single-view observed motion binding proposal, not hidden-geometry truth; elastic band bindings unchanged.'}
    binding['metadata_json'] = np.asarray(json.dumps(report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **binding)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Rebound Gaussians', report['changed_gaussians'], 'dynamic', report['dynamic_count'], flush=True)


if __name__ == '__main__':
    main()
