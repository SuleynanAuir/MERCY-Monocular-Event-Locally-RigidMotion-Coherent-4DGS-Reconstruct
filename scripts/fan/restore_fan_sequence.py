"""Re-bake the historical pre-repair Fan motion from its verified input report.

This restores the recorded video-fitted trajectory, not an event-trained model.
The resulting archive has new packaging metadata and is not hash-identical to
the lost archive. Use repair_fan_motion_binding.py for the separate V3 branch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from scripts.fan.prepare_fan_viewer import sha
from utils.gaussian_sequence import FORMAT, SPARSE_VERSION, POSE_FIELDS, GaussianSequence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'motion-report', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.motion_report.read_text())
    for name, digest in report['input_sha256'].items():
        if sha(args.data/name) != digest:
            raise ValueError('Historical input changed: '+name)
    camera = json.loads((args.data/'fixed_camera.json').read_text())
    raw = PlyData.read(args.data/'fan_contrastive_g0.ply')['vertex'].data.copy()
    xyz = np.column_stack([raw[k] for k in ('x', 'y', 'z')])
    quat = np.column_stack([raw[f'rot_{k}'] for k in range(4)])
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    w = np.eye(4); w[:3, :3] = np.asarray(camera['R']).T; w[:3, 3] = camera['T']
    cam = xyz @ w[:3, :3].T + w[:3, 3]
    u = cam[:, 0]/cam[:, 2]*camera['fx']+camera['cx']
    center = np.asarray(report['center'])
    moving = (u > 320) & (xyz[:, 2] > 0) & (xyz[:, 2] < .16)
    moving &= np.linalg.norm(xyz[:, :2]-center[:2], axis=1) < .31
    ids = np.flatnonzero(moving).astype(np.int64)
    if len(ids) != report['moving_gaussians']:
        raise ValueError('Historical segmentation count differs')
    from scripts.lamp.prepare_lamp_viewer import probe
    video = probe(args.data/'Fan.mkv')
    angles = report['angles_rad']
    if len(angles) != video['frame_count']:
        raise ValueError('Historical timeline differs')
    poses = np.empty((len(ids), len(angles), 7), np.float32)
    rest = Rotation.from_quat(quat[ids][:, [1, 2, 3, 0]])
    frames = []
    for i, angle in enumerate(angles):
        rot = Rotation.from_rotvec([0, 0, angle])
        poses[:, i, :3] = (xyz[ids]-center) @ rot.as_matrix().T+center
        poses[:, i, 3:] = (rot*rest).as_quat()[:, [3, 0, 1, 2]]
        if i == 0:
            poses[:, i, :3] = xyz[ids]; poses[:, i, 3:] = quat[ids]
        frames.append(dict(file=f'frames/frame_{i:04d}.ply', time_seconds=i/video['fps'],
                           pose_sha256=hashlib.sha256(poses[:, i].tobytes()).hexdigest()))
    pose = np.linalg.inv(w); pose[:3, 1:3] *= -1
    radius = float(np.linalg.norm(pose[:3, 3]-np.asarray(camera['target_center'])))
    target = pose[:3, 3]-radius*pose[:3, 2]
    meta = dict(format=FORMAT, version=SPARSE_VERSION, name='Fan · 原记录运动恢复',
        description='从已校验输入和原视频拟合运动报告重新打包；不是原归档文件，也未经过事件训练。',
        motion_label='原记录视频拟合旋转', frame_count=len(frames), fps=video['fps'],
        duration_seconds=len(frames)/video['fps'], gaussians_per_frame=len(raw), sh_degree=3,
        pose_fields=POSE_FIELDS, base_vertices_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
        frames=frames, default_resolution=1024, default_background='light',
        default_stream_rate=30, default_stream_quality=95, default_still_quality=95,
        preview_camera=dict(pose=pose.tolist(), target=target.tolist(), fovy_degrees=camera['fov_y_deg']),
        motion_is_baked=True, event_validation=False, input_sha256=report['input_sha256'],
        restoration=dict(motion_report_sha256=sha(args.motion_report), repackaged=True))
    args.output.mkdir(parents=True)
    dest = args.output/'fan-motion.npz'
    with dest.open('wb') as f:
        np.savez_compressed(f, base_vertices=raw, poses=poses, dynamic_indices=ids,
            metadata_json=np.asarray(json.dumps(meta)), times_seconds=np.arange(len(frames))/video['fps'])
    (args.output/'motion-report.json').write_text(json.dumps(report, indent=2))
    with GaussianSequence(dest) as sequence:
        for i in range(len(sequence)):
            sequence.validate_pose_frame(i)
    print(f'Restored {len(frames)} historical frames, {len(ids)} moving Gaussians', flush=True)


if __name__ == '__main__':
    main()
