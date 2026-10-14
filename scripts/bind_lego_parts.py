"""Register a supplied Blender rest surface to Lego Gaussians (no training).

Nearest-surface attachment is an approximation: source PLY has no part IDs.
The output records that uncertainty instead of claiming an exact segmentation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


def similarity(source, target):
    a, b = source.mean(0), target.mean(0)
    x, y = source - a, target - b
    u, d, vt = np.linalg.svd(y.T @ x / len(x))
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = (u * signs) @ vt
    scale = (d * signs).sum() / np.square(x).sum() * len(x)
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = b - scale * rotation @ a
    return matrix


def transform(xyz, matrix):
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--parts', type=Path, required=True)
    parser.add_argument('--inspection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-distance', type=float, default=.1)
    parser.add_argument('--registration', type=Path, help='Reuse a previously verified registration JSON')
    args = parser.parse_args()
    if not np.isfinite(args.max_distance) or args.max_distance <= 0:
        parser.error('--max-distance must be finite and positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    parts = np.load(args.parts, allow_pickle=False)
    metadata = json.loads(str(parts['metadata_json']))
    camera = json.loads(args.inspection.read_text())['camera']
    surface = parts['rest_points'].astype(np.float64)
    vertices = PlyData.read(args.data / 'fan_contrastive_g0/contrastive_g0.ply')['vertex'].data
    xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')])
    # Cameras initialize the otherwise ambiguous coordinate system. Robust
    # similarity ICP refines it using geometry, without altering the Gaussians.
    initial = np.asarray(camera['pose']) @ np.diag([.40884] * 3 + [1]) @ np.linalg.inv(metadata['camera_pose'])
    tree = cKDTree(surface)
    sample = xyz[::max(1, len(xyz) // 40000)].astype(np.float64)
    matrix = (np.asarray(json.loads(args.registration.read_text())['registration'], dtype=np.float64)
              if args.registration else initial.copy())
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ValueError('Invalid registration matrix')
    for iteration in range(0 if args.registration else 250):
        local = transform(sample, np.linalg.inv(matrix))
        distance, index = tree.query(local, workers=8)
        keep = distance < np.percentile(distance, 85)
        update = similarity(surface[index[keep]], sample[keep])
        difference = np.max(np.abs(update - matrix))
        matrix = update
        if iteration % 50 == 0:
            print('Registration', iteration, 'distance', np.percentile(distance, [50, 90]), 'change', difference, flush=True)
        if difference < 1e-7:
            break
    aligned = transform(surface, matrix)
    distance, nearest = cKDTree(aligned).query(xyz, workers=8)
    labels = parts['point_parts'][nearest]
    matrices = parts['object_matrices'].astype(np.float64)
    delta = matrices @ np.linalg.inv(matrices[0])
    moving = np.max(np.abs(delta - np.eye(4)), axis=(0, 2, 3)) > 1e-4
    moving[metadata['deforming_objects']] = True
    # Floaters far from any source surface are not attached to a moving arm.
    attached = distance <= args.max_distance
    indices = np.flatnonzero(moving[labels] & attached)
    report = {'source': str(args.data), 'method': 'camera initialized robust similarity ICP; nearest evaluated mesh vertex attachment',
              'registration': matrix.tolist(), 'initial_registration': initial.tolist(),
              'distance_percentiles': dict(zip(['p50', 'p90', 'p95', 'p99'], np.percentile(distance, [50, 90, 95, 99]).tolist())),
              'max_attachment_distance': args.max_distance, 'unattached': int((~attached).sum()),
              'gaussian_count': len(xyz), 'dynamic_count': len(indices), 'static_count': len(xyz) - len(indices),
              'part_count': len(metadata['objects']), 'moving_part_count': int(moving.sum()),
              'deforming_objects': [metadata['objects'][i] for i in metadata['deforming_objects']],
              'part_gaussians': {name: int(np.sum(labels[indices] == i)) for i, name in enumerate(metadata['objects']) if moving[i]},
              'limitations': 'Geometric attachment is inferred, not ground-truth segmentation. Rubber bands use local surface deformation; no training or physics solver.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, registration=matrix, labels=labels, nearest_vertex=nearest.astype(np.int32),
                        distance=distance.astype(np.float32), dynamic_indices=indices.astype(np.int32),
                        metadata_json=np.asarray(json.dumps(report)))
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'part_gaussians'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
