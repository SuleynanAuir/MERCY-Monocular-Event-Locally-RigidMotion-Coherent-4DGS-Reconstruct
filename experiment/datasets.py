"""Read-only inventory of complete scenarios and raw event integrity."""
import json
import subprocess
from pathlib import Path
import numpy as np
from .protocol import sha256, save_json, temporal_split

SCENES = {
    'lego': ('Lego_data', 'Lego', 'Lego_event', 'fan_contrastive_g0/contrastive_g0.ply'),
    'fan': ('Fan_data', 'Fan', 'Fan_event.npy', 'fan_contrastive_g0.ply'),
    'lamp': ('lamp_data', 'lamp', 'lamp_event.npy', 'lamb_corrected_g0.ply'),
    'metronome': ('metronome_data', 'metronome', 'metronome_event.npy', 'corrected_g0.ply'),
    'microphone': ('micphone_data', 'micphone', 'micphone_event.npy', 'corrected_g0.ply')}


def video_info(path):
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,avg_frame_rate,nb_read_frames', '-of', 'json', str(path)]))['streams'][0]
    n, d = map(int, data['avg_frame_rate'].split('/'))
    return dict(width=int(data['width']), height=int(data['height']),
                frame_count=int(data['nb_read_frames']), fps=n/d)


def event_integrity(path, width, height, end_seconds):
    try:
        with Path(path).open('rb') as f:
            version = np.lib.format.read_magic(f)
            shape, order, dtype = np.lib.format._read_array_header(f, version)
            expected = f.tell()+int(np.prod(shape))*dtype.itemsize
        actual = Path(path).stat().st_size
        record = dict(shape=list(shape), dtype=str(dtype), expected_bytes=expected, actual_bytes=actual)
        if actual != expected:
            return dict(record, valid=False, reason='truncated_event_array', missing_bytes=expected-actual)
        if len(shape) != 2 or shape[1] != 4 or dtype.kind not in 'fiu' or shape[0] == 0:
            return dict(record, valid=False, reason='unsupported_event_layout')
        data = np.load(path, mmap_mode='r', allow_pickle=False)
        previous = -np.inf
        for start in range(0, len(data), 250000):
            a = np.asarray(data[start:start+250000])
            if (not np.isfinite(a).all() or np.any(np.diff(a[:, 2]) < 0)
                    or a[0, 2] < previous or np.any(a[:, :2] != np.floor(a[:, :2]))
                    or np.any(a[:, :2] < 0) or np.any(a[:, 0] >= width)
                    or np.any(a[:, 1] >= height) or not np.isin(a[:, 3], [-1, 1]).all()):
                return dict(record, valid=False, reason='invalid_event_values')
            previous = float(a[-1, 2])
        if abs(previous-end_seconds) > .05:
            return dict(record, valid=False, reason='timestamp_unit_or_duration_mismatch', end=previous)
        return dict(record, valid=True, count=len(data), start=float(data[0, 2]), end=previous,
                    time_unit='seconds; duration checked', order='x,y,t,p', generation_provenance='not supplied')
    except (OSError, ValueError, EOFError) as error:
        return dict(valid=False, reason=type(error).__name__+': '+str(error))


def ply_header(path):
    with Path(path).open('rb') as f:
        header = []
        for _ in range(200):
            line = f.readline().decode('ascii').strip();header.append(line)
            if line == 'end_header':break
        else:raise ValueError('Unterminated PLY header')
    count = next(int(s.split()[-1]) for s in header if s.startswith('element vertex '))
    return {'gaussians': count, 'format': next(s for s in header if s.startswith('format '))}


def inventory(data_root, output, scenes=None):
    data_root = Path(data_root).resolve();rows = []
    selected=set(SCENES if scenes is None else scenes)
    if not selected or selected-set(SCENES):raise ValueError('Unknown or empty scene selection')
    for name, (folder, stem, events, corrected) in SCENES.items():
        if name not in selected:continue
        base = data_root/folder
        paths = dict(rgb=base/(stem+'.mkv'), events=base/events, g0=base/(stem+'.ply'),
                     corrected=base/corrected, intrinsics=base/'camera_intrinsics.json',
                     camera=base/'fixed_camera.json', blend=base/(stem+'.blend'))
        files = {k: {'path': str(p), 'exists': p.is_file(),
                     **({'bytes': p.stat().st_size, 'sha256': sha256(p)} if p.is_file() else {})}
                 for k, p in paths.items()}
        video = video_info(paths['rgb'])
        event = event_integrity(paths['events'], video['width'], video['height'],
                                (video['frame_count']-1)/video['fps'])
        rows.append(dict(scene=name, files=files, video=video, events=event,
                         gaussian_counts={k: ply_header(paths[k])['gaussians'] for k in ('g0','corrected')},
                         split=temporal_split(video['frame_count'], video['fps']),
                         prior_exposure='Unknown G0 training exposure; Blender animation is an explicit input where available'))
        print('Audited', name, video['frame_count'], 'frames; raw events:', event['valid'], flush=True)
    result = dict(format='mercy_experiment_inventory_v1', scenes=rows,
                  requested_scenes=sorted(selected),missing_from_five_scene_scope=sorted(set(SCENES)-selected),
                  excluded=['data/gaussians: standalone Fan asset, not an independent scene'])
    save_json(output, result)
    return result
