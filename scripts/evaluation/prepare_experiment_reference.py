"""Prepare shared, bounded evaluation references from original observations.

Run once on the server, outside the UI export. No reconstruction output is used
to label masks. The JSON config explicitly selects video, camera, source mesh
and motion, and optional raw x/y/t/p events and response calibration.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.experiment_format import save_json, sha256
from utils.experiment_export import compact_counts


def geometry_masks(mesh_path, parts_path, indices, width, height, fovy):
    """Fixed visible source-object labels, including original elastic motion."""
    import torch
    import nvdiffrast.torch as dr
    from utils.viewer_mesh import projection_matrix
    with np.load(mesh_path, allow_pickle=False) as mesh, np.load(parts_path, allow_pickle=False) as parts:
        meta = json.loads(str(mesh['metadata_json']))
        motion = json.loads(str(parts['metadata_json']))
        if meta['parts_sha256'] != sha256(parts_path) or meta['objects'] != motion['objects']:
            raise ValueError('Source mesh and animation do not match')
        matrices = parts['object_matrices']
        inverse = np.linalg.inv(meta['registration'])
        rest = mesh['vertices'] @ inverse[:3, :3].T + inverse[:3, 3]
        local, labels, sections, offset = [], [], {}, 0
        for i, row in enumerate(meta['parts_detail']):
            if row['part_id'] != i or row['name'] != meta['objects'][i]:
                raise ValueError('Unexpected source vertex ordering')
            section = slice(offset, offset+row['vertices']); sections[i] = section
            transform = np.linalg.inv(matrices[0, i])
            local.append(rest[section] @ transform[:3, :3].T + transform[:3, 3])
            labels.append(np.full(row['vertices'], i, np.int64)); offset += row['vertices']
        if offset != len(rest): raise ValueError('Incomplete source ownership')
        labels = np.concatenate(labels)
        triangles = mesh['triangles']
        if not np.all(labels[triangles] == labels[triangles[:, :1]]):
            raise ValueError('A source triangle crosses parts')
        face_labels = torch.tensor(labels[triangles[:, 0]]+1, dtype=torch.int32, device='cuda')
        vertices = torch.tensor(np.concatenate(local), dtype=torch.float32, device='cuda')
        labels = torch.tensor(labels, device='cuda')
        faces = torch.tensor(np.ascontiguousarray(triangles), dtype=torch.int32, device='cuda')
        elastic = {i: parts[f'deformed_{i}'] for i in meta.get('deforming_objects', [])}
        camera = torch.tensor(motion['camera_pose'], dtype=torch.float32, device='cuda')
        projection = torch.tensor(projection_matrix(width, height, fovy), device='cuda')
        context = dr.RasterizeCudaContext()
        with torch.no_grad():
            for frame in indices:
                transforms = torch.tensor(matrices[frame], dtype=torch.float32, device='cuda')
                xyz = torch.bmm(transforms[labels, :3, :3], vertices[..., None])[..., 0] + transforms[labels, :3, 3]
                for i, values in elastic.items(): xyz[sections[i]] = torch.tensor(values[frame], device='cuda')
                view = (xyz-camera[:3, 3]) @ camera[:3, :3]
                clip = torch.cat((view, torch.ones_like(view[:, :1])), -1) @ projection.T
                raster, _ = dr.rasterize(context, clip[None].contiguous(), faces, [height, width], grad_db=False)
                face = raster[0, ..., 3].long()-1
                mask = torch.where(face >= 0, face_labels[face.clamp_min(0)], 0).flip(0).cpu().numpy()
                yield frame, mask, meta['objects']


def prepare(config, output):
    import cv2
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    try:
        camera, count, fps = config['camera'], config['frame_count'], config['fps']
        width, height = camera['width'], camera['height']
        step = config.get('sample_step', 2)
        if type(step) is not int or not 1 <= step <= count: raise ValueError('Invalid reference sample_step')
        indices = list(range(0, count, step))
        samples = [dict(frame=i, time_seconds=i/fps) for i in indices]
        lookup = {r['frame']: r for r in samples}
        result = dict(format='scgs_experiment_reference_v1', provenance=config['provenance'],
                      camera=camera, background=config['background'], samples=samples,
                      sample_step=step, unavailable=dict(config.get('unavailable', {})), sources={})
        if 'background_rgb' in config: result['background_rgb'] = config['background_rgb']
        for key in ('video', 'mesh', 'parts', 'events', 'calibration'):
            path = config.get(key)
            if path: result['sources'][key] = dict(name=Path(path).name, sha256=sha256(path), bytes=Path(path).stat().st_size)
        (output/'rgb').mkdir()
        video = cv2.VideoCapture(config['video'])
        try:
            if abs(video.get(cv2.CAP_PROP_FPS)-fps) > 1e-6: raise ValueError('Video FPS mismatch')
            frame = 0
            while True:
                ok, bgr = video.read()
                if not ok: break
                if bgr.shape != (height, width, 3): raise ValueError('Native video/camera size mismatch')
                if frame in lookup:
                    name = f'rgb/{frame:06d}.png'
                    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(output/name, compress_level=3)
                    lookup[frame]['rgb'] = name
                frame += 1
            if frame != count: raise ValueError(f'Video count mismatch: {frame} != {count}')
        finally: video.release()
        print(f'RGB {len(samples)} sampled frames', flush=True)
        if config.get('rgb_foreground'):
            if config.get('mesh'): raise ValueError('Choose geometric or RGB-support masks explicitly')
            annotation = config['rgb_foreground']
            threshold = annotation['threshold_u8']
            background = np.asarray(annotation['background_rgb_u8'], dtype=float)
            if (type(threshold) is not int or not 1 <= threshold <= 64
                    or background.shape != (3,) or not np.isfinite(background).all()
                    or np.any((background < 0) | (background > 255))):
                raise ValueError('Invalid fixed RGB foreground definition')
            (output/'masks').mkdir()
            audit = []
            for row in samples:
                rgb = np.asarray(Image.open(output/row['rgb']).convert('RGB'), dtype=float)
                distance = np.max(np.abs(rgb-background), axis=-1)
                name = f'masks/{row["frame"]:06d}.png'
                Image.fromarray(distance > threshold).save(output/name, compress_level=3)
                row['foreground_mask'] = name
                audit.append(dict(frame=row['frame'], pixels={str(t):int((distance>t).sum())
                                  for t in sorted({max(1, threshold//2), threshold, threshold*2})}))
            result['mask_definition'] = ('Fixed original-RGB background-distance support mask; '
                'max channel distance > '+str(threshold)+'/255. Shared by all methods; '
                'photometric foreground proxy, not geometric or manual ground truth.')
            result['mask_quality'] = 'rgb_background_proxy'
            result['rgb_foreground'] = annotation
            result['unavailable'].pop('foreground_mask', None)
            save_json(output/'mask-threshold-audit.json', audit)
        if config.get('mesh') and config.get('parts'):
            (output/'masks').mkdir(); (output/'parts').mkdir()
            fovy = float(np.rad2deg(2*np.arctan(height/(2*camera['K'][1][1]))))
            for frame, labels, objects in geometry_masks(config['mesh'], config['parts'], indices, width, height, fovy):
                for key, image, folder in [('foreground_mask', Image.fromarray(labels > 0), 'masks'),
                        ('part_mask', Image.fromarray(labels.astype(np.uint8 if len(objects) < 256 else np.uint16)), 'parts')]:
                    name = f'{folder}/{frame:06d}.png'; image.save(output/name, compress_level=3)
                    lookup[frame][key] = name
                if frame % 40 == 0: print(f'Source masks {frame}/{count}', flush=True)
            result['part_labels'] = {str(i+1): name for i, name in enumerate(objects)}
            result['mask_definition'] = 'Uncorrected source Blender geometry at pixel centers; nearest visible surface, 0 background. Shared geometric annotation, not transparency/photometric opacity.'
            with np.load(config['parts'], allow_pickle=False) as parts:
                motion = json.loads(str(parts['metadata_json']))
                np.savez_compressed(output/'source-motion.npz',
                    frames=np.asarray(indices), times_seconds=np.asarray(indices)/fps,
                    object_matrices=parts['object_matrices'][indices],
                    source_camera_pose=np.asarray(motion['camera_pose']),
                    part_ids=np.arange(1,len(objects)+1))
            result['source_motion'] = 'source-motion.npz'
            result['source_motion_scope'] = 'Original Blender object matrices in source world coordinates; part IDs match shared masks. Elastic vertex deformation is not represented by object matrices. Not method predictions.'
        if config.get('events'):
            from utils.event_supervision import EventStream
            stream = EventStream(config['events'], width, height)
            if abs(stream.times[-1]-(count-1)/fps) > 2/fps:
                raise ValueError('Raw event duration and RGB timeline differ')
            calibration = None
            if config.get('calibration'):
                original = json.loads(Path(config['calibration']).read_text())
                c = original['calibration']
                if c['offset_frames'] != 0: raise ValueError('Nonzero event offset needs explicit source clock mapping')
                calibration = dict(space='rgb_image', transfer='encoded', luminance='opencv_bt601_u8',
                    log_floor=c['log_floor'], contrast_positive=c['thresholds'][0], contrast_negative=c['thresholds'][1],
                    provenance='Fixed empirical response fitted from original RGB and raw events, not an independent sensor calibration',
                    calibration_ids=original['calibration_ids'], time_offset_seconds=0,
                    coordinate_mapping='identity x/y pixels; timestamp seconds; interval (t0,t1]',
                    source_calibration_sha256=sha256(config['calibration']))
            (output/'events').mkdir()
            for previous, row in zip(samples, samples[1:]):
                counts = stream.counts(previous['time_seconds'], row['time_seconds'])
                name = f'events/{row["frame"]:06d}.npz'
                np.savez_compressed(output/name, positive=compact_counts(counts[0]), negative=compact_counts(counts[1]),
                                    t0=previous['time_seconds'], t1=row['time_seconds'])
                row['events'] = name
            if calibration:
                result['event_calibration'] = calibration
                for previous, row in zip(samples, samples[1:]):
                    row['event_split'] = ('overlaps_response_calibration' if any(
                        previous['frame'] <= i < row['frame'] for i in calibration['calibration_ids']) else 'response_validation')
            else:
                result['event_calibration'] = dict(space='rgb_image', transfer='encoded', luminance='opencv_bt601_u8',
                    log_floor=.03, coordinate_mapping='identity x/y pixels; seconds; (t0,t1]',
                    provenance='Raw source coordinates and RGB timing; contrast response not supplied')
                result['unavailable']['event_response'] = 'Raw counts saved; no verified positive/negative contrast thresholds'
        result['storage_bytes'] = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if result['storage_bytes'] > 512*1024**2: raise ValueError('Reference cache exceeds 512 MiB')
        save_json(output/'reference.json', result)
        print(json.dumps(dict(path=str(output/'reference.json'), bytes=result['storage_bytes'], frames=len(samples))), flush=True)
    except BaseException:
        shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.output)


if __name__ == '__main__': main()
