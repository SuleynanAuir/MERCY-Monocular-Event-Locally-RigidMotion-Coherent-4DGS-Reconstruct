"""Prepare supplied Metronome/Microphone GS pairs using their Blender motion.

Reuses the validated rigid-part baker. This is geometric animation attachment,
not event training. Missing camera extrinsics are derived from the registered
source camera; the candidate must be visually reviewed against the RGB video.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from scripts.lamp.prepare_lamp_viewer import bake, camera_pose, register, rigid_deltas
from scripts.motion.bind_lego_parts import similarity, transform
from scripts.viewer.prepare_reference_videos import probe
from utils.gaussian_pair_renderer import file_sha


def spatial_sample(xyz, count=16000):
    step = np.linalg.norm(np.ptp(xyz, axis=0)) / 160
    _, ids = np.unique(np.floor(xyz/step).astype(np.int64), axis=0, return_index=True)
    return xyz[ids[::max(1, int(np.ceil(len(ids)/count)))]].astype(np.float64)


def register_without_camera(surface, xyz):
    """Search proper PCA orientations, then refine with trimmed similarity ICP."""
    source, target = spatial_sample(surface), spatial_sample(xyz)
    a, b = source.mean(0), target.mean(0)
    _, axes_a = np.linalg.eigh(np.cov(source.T))
    _, axes_b = np.linalg.eigh(np.cov(target.T))
    scale = np.sqrt(np.square(target-b).sum()/len(target)
                    / (np.square(source-a).sum()/len(source)))
    tree = cKDTree(source)
    def refine(matrix, steps):
        for _ in range(steps):
            distance, nearest = tree.query(transform(target, np.linalg.inv(matrix)), workers=4)
            keep = distance <= np.quantile(distance, .90)
            update = similarity(source[nearest[keep]], target[keep])
            change = np.max(np.abs(update-matrix)); matrix = update
            if change < 1e-7:
                break
        distance = cKDTree(transform(source, matrix)).query(target, workers=4)[0]
        reverse = cKDTree(target).query(transform(source, matrix), workers=4)[0]
        score = np.mean(np.minimum(distance, np.quantile(distance, .95)))
        score += np.mean(np.minimum(reverse, np.quantile(reverse, .95)))
        return float(score), matrix
    candidates = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            rotation = (axes_b[:, permutation]*signs) @ axes_a.T
            if np.linalg.det(rotation) < .99:
                continue
            matrix = np.eye(4); matrix[:3, :3] = scale*rotation
            matrix[:3, 3] = b-scale*rotation @ a
            candidates.append(refine(matrix, 40))
    candidates.sort(key=lambda value: value[0])
    refined = sorted([refine(matrix, 160) for _, matrix in candidates[:4]], key=lambda v: v[0])
    score, matrix = refined[0]
    return matrix, dict(method='24 proper PCA orientations and trimmed similarity ICP',
                        bidirectional_distance=score, candidate_scores=[v[0] for v in refined])


def registered_camera(source_pose, registration, target, fovy):
    pose = registration @ np.asarray(source_pose)
    scale = np.cbrt(np.linalg.det(registration[:3, :3]))
    pose[:3, :3] /= scale
    # Keep the exact camera position and direction; orbit around its optical
    # axis at the object's depth, not an off-axis bounding-box center.
    distance = np.dot(pose[:3, 3]-target, pose[:3, 2])
    if distance <= 0 or not np.isfinite(pose).all():
        raise ValueError('Registered source camera does not face the object')
    return dict(pose=pose.tolist(), target=(pose[:3, 3]-distance*pose[:3, 2]).tolist(), fovy_degrees=fovy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'parts', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--stem', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    timeline = probe(args.data/(args.stem+'.mkv'))
    with np.load(args.parts, allow_pickle=False) as archive:
        parts = {k: archive[k] for k in archive.files}
    meta = json.loads(str(parts['metadata_json'].item()))
    if (meta.get('format') != 'lamp_blender_parts_v1' or not meta['autoexec_disabled']
            or meta['frame_count'] != timeline['frame_count'] or meta['fps'] != timeline['fps']
            or meta['blend_sha256'] != file_sha(args.data/(args.stem+'.blend'))):
        raise ValueError('Source identity or animation timeline does not match RGB')
    intrinsics = json.loads((args.data/'camera_intrinsics.json').read_text())
    fovy = float(np.rad2deg(2*np.arctan(timeline['height']/(2*intrinsics['fy']))))
    if (intrinsics['width'] != timeline['width'] or intrinsics['height'] != timeline['height']
            or abs(fovy-meta['camera_fovy_degrees']) > .01
            or abs(intrinsics['cx']-timeline['width']/2) > .01
            or abs(intrinsics['cy']-timeline['height']/2) > .01):
        raise ValueError('Source camera intrinsics mismatch or off-axis projection')
    v = PlyData.read(args.data/(args.stem+'.ply'))['vertex'].data
    xyz = np.column_stack([v[k] for k in ('x','y','z')])
    fixed = args.data/'fixed_camera.json'
    if fixed.exists():
        preview = camera_pose(json.loads(fixed.read_text()))
        registration, initial = register(parts['rest_points'], xyz, meta['camera_pose'], preview['pose'])
        registration_info = dict(method='Supplied camera initialized similarity ICP', initial=initial.tolist())
    else:
        registration, registration_info = register_without_camera(parts['rest_points'], xyz)
        preview = registered_camera(meta['camera_pose'], registration, xyz.mean(0), fovy)
    delta = rigid_deltas(parts['object_matrices'], registration)
    surface = transform(parts['rest_points'], registration)
    provenance = dict(parts_sha256=file_sha(args.parts), blend_sha256=meta['blend_sha256'],
        source_frames=parts['source_frames'].tolist(), registration=registration.tolist(),
        registration_info=registration_info, method='Nearest source-surface rigid attachment; no training',
        event_fitting=False, rgb_fitting=False)
    report = dict(provenance, objects=meta['objects'], sides={})
    config = dict(format='gaussian_pair_v1', name=args.name,
        description='原始高斯 / 已提供修正版；绑定原始 Blender 动画，未进行新增事件训练。')
    args.output.mkdir(parents=True)
    for key, source, label in [('left', args.stem+'.ply', 'V1 · 原始高斯'),
                               ('right', 'corrected_g0.ply', 'V3 · 已提供修正版')]:
        path = args.output/f'{key}.npz'
        report['sides'][key] = bake(args.data/source, path, surface, parts['point_parts'],
            delta, preview, timeline, provenance, name=args.name)
        config[key] = dict(path=path.name, sha256=file_sha(path), label=label)
        print(key, json.dumps(report['sides'][key]), flush=True)
    (args.output/'manifest.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    (args.output/'binding-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
