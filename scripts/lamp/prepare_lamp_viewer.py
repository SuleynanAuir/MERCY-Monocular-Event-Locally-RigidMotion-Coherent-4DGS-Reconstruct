"""Bind supplied Lamp Gaussian versions to the original Blender animation.

No RGB/event fitting is performed. Video metadata only verifies the source
timeline; nearest-surface part attachment is an explicit geometric prior.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from scripts.motion.bind_lego_parts import similarity, transform
from scripts.viewer.prepare_reference_videos import probe
from utils.gaussian_pair_renderer import file_sha
from utils.gaussian_sequence import FORMAT, SPARSE_VERSION, POSE_FIELDS, GaussianSequence
from utils.ply_utils import gaussian_degree


def camera_pose(camera):
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = np.asarray(camera['R']).T
    world_to_camera[:3, 3] = camera['T']
    pose = np.linalg.inv(world_to_camera)
    pose[:3, 1:3] *= -1
    distance = np.linalg.norm(pose[:3, 3]-camera['target_center'])
    return dict(pose=pose.tolist(), target=(pose[:3, 3]-distance*pose[:3, 2]).tolist(),
                fovy_degrees=camera['fov_y_deg'])


def register(surface, xyz, source_camera, target_camera):
    # Spatially uniform samples keep the densely sampled base from dominating
    # the lamp's thin upper arm and shade during geometric registration.
    _, rows = np.unique(np.floor(xyz/.008).astype(np.int32), axis=0, return_index=True)
    sample = xyz[rows].astype(np.float64)
    scale = np.linalg.norm(np.ptp(xyz, axis=0))/np.linalg.norm(np.ptp(surface, axis=0))
    initial = np.asarray(target_camera) @ np.diag([scale, scale, scale, 1]) @ np.linalg.inv(source_camera)
    tree = cKDTree(surface)
    matrix = initial.copy()
    for _ in range(200):
        distance, nearest = tree.query(transform(sample, np.linalg.inv(matrix)), workers=4)
        keep = distance <= np.quantile(distance, .9)
        update = similarity(surface[nearest[keep]], sample[keep])
        change = np.max(np.abs(update-matrix))
        matrix = update
        if change < 1e-7:
            break
    return matrix, initial


def rigid_deltas(matrices, registration):
    delta = registration @ (matrices @ np.linalg.inv(matrices[0])) @ np.linalg.inv(registration)
    linear = delta[..., :3, :3]
    if not np.isfinite(delta).all() or not np.allclose(linear.swapaxes(-1, -2) @ linear, np.eye(3), atol=2e-5):
        raise ValueError('Source motion has scaling/shear; rigid Gaussian baking would be incorrect')
    if np.any(np.linalg.det(linear) < .9999):
        raise ValueError('Source motion contains reflections')
    delta[0] = np.eye(4)
    return delta


def bake(source, destination, surface, groups, delta, preview_camera, timeline, provenance, *, bound_labels=None, name='Lamp · 台灯'):
    vertices = PlyData.read(source)['vertex'].data.copy()
    xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')])
    quat = np.column_stack([vertices[f'rot_{i}'] for i in range(4)])
    if not np.isfinite(xyz).all() or not np.isfinite(quat).all() or np.any(np.linalg.norm(quat, axis=1) < 1e-8):
        raise ValueError('Invalid Gaussian geometry')
    distance, nearest = cKDTree(surface).query(xyz, workers=4)
    labels = groups[nearest]
    if bound_labels is not None:
        labels = np.asarray(bound_labels)
        if (labels.shape != (len(vertices),) or labels.dtype.kind not in 'iu'
                or labels.min() < 0 or labels.max() >= delta.shape[1]):
            raise ValueError('Invalid inherited part binding')
    extent = np.linalg.norm(np.ptp(xyz, axis=0))
    if np.quantile(distance, .95) > extent*.04:
        raise ValueError('Source surface registration is too far from the Gaussians')
    moving = np.max(np.abs(delta-np.eye(4)), axis=(0, 2, 3)) > 1e-5
    close = distance <= extent*.04
    # The supplied GS shade extends beyond its Blender surface, especially its
    # rear cap. Freezing less-confident rows leaves a detached cap at its old
    # location. This is a single isolated object: preserve those rows and let
    # their nearest source part carry them, with an explicit distance guard.
    if distance.max() > extent*.12:
        raise ValueError('Gaussian rows lie too far outside the source object to bind safely')
    indices = np.flatnonzero(moving[labels]).astype(np.int64)
    poses = np.empty((len(indices), timeline['frame_count'], 7), np.float32)
    rotations = Rotation.from_quat(quat[indices][:, [1, 2, 3, 0]])
    entries = []
    for frame in range(timeline['frame_count']):
        for group in np.unique(labels[indices]):
            selected = np.flatnonzero(labels[indices] == group)
            matrix = delta[frame, group]
            poses[selected, frame, :3] = transform(xyz[indices[selected]], matrix)
            poses[selected, frame, 3:] = (Rotation.from_matrix(matrix[:3, :3])*rotations[selected]).as_quat()[:, [3, 0, 1, 2]]
        # Preserve the exact supplied canonical state, including quaternion sign.
        if frame == 0:
            poses[:, 0, :3], poses[:, 0, 3:] = xyz[indices], quat[indices]
        entries.append(dict(file=f'frames/frame_{frame:04d}.ply', time_seconds=frame/timeline['fps'],
                            pose_sha256=hashlib.sha256(poses[:, frame].tobytes()).hexdigest()))
    meta = dict(format=FORMAT, version=SPARSE_VERSION, name=name,
        description='已提供的高斯外观，绑定原始 Blender 部件动画；未使用 RGB 或事件训练。',
        motion_label='Blender 原始部件动画', frame_count=len(entries), fps=timeline['fps'],
        duration_seconds=len(entries)/timeline['fps'], gaussians_per_frame=len(vertices),
        sh_degree=gaussian_degree(source), pose_fields=POSE_FIELDS, frames=entries,
        base_vertices_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),
        preview_camera=preview_camera, default_resolution=1024, default_background='light',
        default_stream_rate=30, default_stream_quality=95, default_still_quality=95,
        event_validation=False, motion_is_baked=True,
        provenance=dict(provenance, source_ply_sha256=file_sha(source)))
    with destination.open('wb') as stream:
        np.savez_compressed(stream, base_vertices=vertices, poses=poses, dynamic_indices=indices,
            times_seconds=np.arange(len(entries))/timeline['fps'], metadata_json=np.asarray(json.dumps(meta)))
    with GaussianSequence(destination) as sequence:
        for i in range(len(sequence)):
            sequence.validate_pose_frame(i)
    return dict(gaussians=len(vertices), moving_gaussians=len(indices), unattached=0,
        extrapolated_nearest_part_rows=int((~close).sum()), maximum_attachment_distance=float(distance.max()),
        distances=dict(zip(('p50', 'p90', 'p95', 'p99'), np.quantile(distance, [.5, .9, .95, .99]).tolist())),
        part_counts=np.bincount(labels, minlength=delta.shape[1]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'parts', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    timeline = probe(args.data/'lamp.mkv')
    with np.load(args.parts, allow_pickle=False) as archive:
        parts = {name: archive[name] for name in archive.files}
    meta = json.loads(str(parts['metadata_json'].item()))
    if (meta.get('format') != 'lamp_blender_parts_v1' or not meta['autoexec_disabled']
            or meta['frame_count'] != timeline['frame_count'] or meta['fps'] != timeline['fps']
            or meta['blend_sha256'] != file_sha(args.data/'lamp.blend')):
        raise ValueError('Blender export identity/timeline does not match Lamp inputs')
    camera = json.loads((args.data/'fixed_camera.json').read_text())
    preview = camera_pose(camera)
    corrected = PlyData.read(args.data/'lamb_corrected_g0.ply')['vertex'].data
    xyz = np.column_stack([corrected[k] for k in ('x', 'y', 'z')])
    registration, initial = register(parts['rest_points'], xyz, meta['camera_pose'], preview['pose'])
    delta = rigid_deltas(parts['object_matrices'], registration)
    surface = transform(parts['rest_points'], registration)
    args.output.mkdir(parents=True)
    provenance = dict(parts_sha256=file_sha(args.parts), blend_sha256=meta['blend_sha256'],
        source_frames=parts['source_frames'].tolist(), registration=registration.tolist(),
        method='Geometric registration and nearest-surface rigid part binding; no training',
        event_fitting=False, rgb_fitting=False)
    report = dict(provenance, initial_registration=initial.tolist(), objects=meta['objects'], sides={})
    config = dict(format='gaussian_pair_v1', name='Lamp · 台灯',
        description='原始高斯 / 已提供修正版，统一绑定 Blender 原始部件运动；未经事件优化验收。')
    for key, source, label in [('left', 'lamp.ply', 'V1 · 原始高斯'),
                               ('right', 'lamb_corrected_g0.ply', 'V3 · 已提供修正版')]:
        path = args.output/f'{key}.npz'
        report['sides'][key] = bake(args.data/source, path, surface, parts['point_parts'], delta, preview, timeline, provenance)
        config[key] = dict(path=path.name, sha256=file_sha(path), label=label)
        print(key, report['sides'][key], flush=True)
    (args.output/'manifest.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    (args.output/'binding-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
