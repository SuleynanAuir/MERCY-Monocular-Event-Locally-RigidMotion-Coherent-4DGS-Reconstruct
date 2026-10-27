"""Pack current data-folder PLYs losslessly for an explicit static GPU preview.

This does not substitute old motion, checkpoints or metrics for new source data.
Each scene uses its supplied fixed camera; the source hash is preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from plyfile import PlyData
from utils.gaussian_sequence import FORMAT, SPARSE_VERSION, POSE_FIELDS, GaussianSequence


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pack(source, camera_path, output, name):
    camera = json.loads(camera_path.read_text())
    vertices = PlyData.read(source)['vertex'].data.copy()
    for key in POSE_FIELDS:
        if key not in vertices.dtype.names or not np.isfinite(vertices[key]).all():
            raise ValueError(f'Invalid geometry: {source}: {key}')
    rest = [n for n in vertices.dtype.names if n.startswith('f_rest_')]
    degree = int(round(np.sqrt(len(rest) / 3 + 1) - 1))
    if 3 * ((degree + 1) ** 2 - 1) != len(rest):
        raise ValueError('Invalid spherical harmonic field count')
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.asarray(camera['R']).T
    extrinsic[:3, 3] = camera['T']
    pose = np.linalg.inv(extrinsic)
    pose[:3, 1:3] *= -1
    distance = np.linalg.norm(pose[:3, 3] - camera['target_center'])
    fovy = float(camera['fov_y_deg'])
    if not np.isfinite(pose).all() or distance <= 0 or not 0 < fovy < 180:
        raise ValueError('Invalid fixed camera')
    poses = np.empty((0, 1, 7), dtype=np.float32)
    frame = dict(file='frames/frame_0000.ply', time_seconds=0.,
                 pose_sha256=hashlib.sha256(poses[:, 0].tobytes()).hexdigest())
    background = np.asarray(camera.get('background_rgb', [0, 0, 0]))
    meta = dict(format=FORMAT, version=SPARSE_VERSION, name=name,
        description='新 data 文件直接渲染：原始 / 修正高斯与所附固定相机。仅初始模型预览；尚未重建动态 V1/V2/V3 或计算实验指标。',
        motion_label='初始模型 · 无动画', frame_count=1, fps=1., duration_seconds=1.,
        gaussians_per_frame=len(vertices), sh_degree=degree, pose_fields=POSE_FIELDS,
        frames=[frame], base_vertices_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),
        preview_camera=dict(pose=pose.tolist(), target=(pose[:3, 3] - distance * pose[:3, 2]).tolist(), fovy_degrees=fovy),
        default_resolution=640, default_background='dark' if background.mean() < .5 else 'light',
        default_stream_rate=30, default_stream_quality=95, default_still_quality=95,
        motion_is_baked=False, event_validation=False, source_preview=True,
        provenance=dict(source=str(source), source_ply_sha256=digest(source),
                        camera_sha256=digest(camera_path)))
    with output.open('xb') as f:
        np.savez_compressed(f, base_vertices=vertices, poses=poses,
            dynamic_indices=np.empty(0, dtype=np.int64), times_seconds=np.array([0.]),
            metadata_json=np.asarray(json.dumps(meta, ensure_ascii=False)))
    with GaussianSequence(output) as seq:
        seq.validate_pose_frame(0)
        assert np.array_equal(seq.base_vertices.view('u1'), vertices.view('u1'))
    return dict(path=output.name, sha256=digest(output), source_sha256=meta['provenance']['source_ply_sha256'])


def prepare(data, output):
    output.mkdir(parents=True, exist_ok=False)
    scenes = []
    rows = [('fan', 'Fan', 'Fan.ply.ply', 'Fan · 风扇'),
            ('lego', 'Lego', 'Lego.ply', 'Lego · 吊机'),
            ('lamp', 'lamp', 'lamp.ply', 'Lamp · 台灯'),
            ('metronome', 'metronome', 'metronome.ply', 'Metronome · 节拍器'),
            ('microphone', 'micphone', 'micphone.ply', 'Microphone · 麦克风')]
    for key, folder, original, name in rows:
        parts = {}
        for side, file, label in [('left', original, '原始文件'), ('right', 'corrected_g0.ply', '修正文件')]:
            parts[side] = pack(data / folder / file, data / folder / 'fixed_camera.json',
                               output / f'{key}-{side}.npz', name)
            parts[side]['label'] = label
            print(key, side, parts[side]['source_sha256'], flush=True)
        manifest = dict(format='gaussian_pair_v1', name=name + ' · 新 data 模型预览', source_preview=True,
            description='新 data 原始 / 修正文件直接渲染，使用目录内固定相机。仅初始模型，无动态训练或评测结论。', **parts)
        (output / f'{key}.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        scenes.append(dict(id=key, name=name + ' · 新 data', type='gaussian_pair',
                           path=f'{key}.json', cache='cpu'))
    (output / 'scenes.json').write_text(json.dumps(dict(default='lamp', scenes=scenes), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data.resolve(), args.output.resolve())
