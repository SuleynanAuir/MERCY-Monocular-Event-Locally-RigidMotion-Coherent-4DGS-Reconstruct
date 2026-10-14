"""Render native Gaussian RGB/alpha at verified mesh-prior diagnostic cameras.

This is an explicitly UNAPPROVED predeployment diagnostic. Neither a prior
silhouette match nor an attractive unseen view grants event/3D acceptance.
No reference image is composited and no smoothing or sharpening is applied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from utils.render_sampling import require_native_sampling, require_same_sampling
from utils.viewer_state import orbit_pose


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_view_plan(metadata, frame_indices, cache):
    """CPU-only exact camera/timeline checks; retain the input view records."""
    views = metadata.get('views')
    if (metadata.get('format') != 'lego_corrected_rigid_mesh_multiview_masks_v1'
            or metadata.get('resolution') != [640, 640]
            or not isinstance(views, list) or not 1 <= len(views) <= 400
            or metadata.get('view_count') != len(views)
            or frame_indices.shape != (len(views),) or frame_indices.dtype.kind not in 'iu'
            or metadata.get('camera') != cache.get('camera')):
        raise ValueError('Mask camera, native resolution, view count or timeline metadata differs')
    camera = cache['camera']
    if not 0 < camera.get('fovy_degrees', 0) < 180:
        raise ValueError('Invalid diagnostic field of view')
    for index, view in enumerate(views):
        frame = view.get('frame')
        if (view.get('view_index') != index or type(frame) is not int
                or not 0 <= frame < cache['frame_count'] or frame_indices[index] != frame):
            raise ValueError('Mask view ordering or frame index differs')
        for field in ('yaw', 'pitch', 'zoom', 'pan_x', 'pan_y'):
            value = view.get(field)
            if type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError(f'Invalid mask camera parameter: {field}')
        if view['zoom'] <= 0:
            raise ValueError('Mask camera zoom must be positive')
        pose = np.asarray(view.get('camera_pose'), np.float64)
        expected = orbit_pose(camera['pose'], camera['target'], view)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose, expected, rtol=0, atol=1e-7)):
            raise ValueError(f'Mask camera pose does not match UI orbit at view {index}')
    return views


def validate_mask_arrays(part_ids, depth, metadata):
    expected = (metadata['view_count'], 640, 640)
    objects = metadata.get('objects')
    if (not isinstance(objects, list) or not objects or part_ids.shape != expected
            or part_ids.dtype.kind not in 'iu' or depth.shape != expected
            or depth.dtype.kind != 'f' or np.any(part_ids < -1) or np.any(part_ids >= len(objects))):
        raise ValueError('Invalid rigid prior mask/depth arrays')
    hit = part_ids >= 0
    if not np.isfinite(depth[hit]).all() or np.any(depth[hit] <= 0) or not np.isnan(depth[~hit]).all():
        raise ValueError('Prior mask depths must be positive at hits and NaN off-surface')


def validate_motion_metadata(mask_meta, motion_meta, cache, record,
                             cache_hash, correction_hash, completion_report=None):
    """A suffix may inherit parent motion, never unrelated mask/correction data."""
    expected_cache, expected_correction = cache_hash, correction_hash
    if completion_report is not None:
        expected_cache = sha256(Path(completion_report['parent_baseline']) / 'cache.json')
        expected_correction = completion_report['parent_correction_sha256']
    if (motion_meta.get('format') != 'lego_corrected_mesh_motion_v1'
            or mask_meta.get('motion_metadata') != motion_meta
            or mask_meta.get('objects') != motion_meta.get('objects')
            or mask_meta.get('excluded_deforming_objects') != motion_meta.get('excluded_deforming_objects')
            or motion_meta.get('baseline_cache_sha256') != expected_cache
            or motion_meta.get('source_correction_sha256') != expected_correction
            or motion_meta.get('source_parts_sha256') != record['joint_structure']['input_sha256'].get('parts')
            or record.get('frame_count') != cache.get('frame_count')):
        raise ValueError('Prior mask motion provenance differs from the rendered candidate')


def verify_inputs(baseline, candidate, masks):
    """CPU checks only, including compact suffix provenance and actual group SE3."""
    import torch
    from utils.event_refinement import correction_from_record
    from utils.multiview_completion import verify_multiview_completion

    cache_path, correction_path = baseline/'cache.json', candidate/'correction.json'
    cache, record = json.loads(cache_path.read_text()), json.loads(correction_path.read_text())
    require_native_sampling(cache, 'Multiview RGB/alpha diagnostic')
    sampling = require_same_sampling(cache, record)
    if (record.get('format') != 'event_supervised_hard_joint_alignment_v4'
            or record.get('frame_count') != cache.get('frame_count')
            or len(record['joint_structure']['row_groups']) != cache.get('dynamic_count')):
        raise ValueError('Expected the preserved hard-joint candidate and matching baked rows')
    inputs = {'source_ply': Path(cache['source']), 'correction': correction_path,
              'baseline_cache_metadata': cache_path, 'baseline_poses': baseline/'poses.npy',
              'dynamic_indices': baseline/'dynamic_indices.npy', 'masks': masks}
    hashes = {key: sha256(path) for key, path in inputs.items()}
    if cache.get('source_sha256') != hashes['source_ply']:
        raise ValueError('Source PLY checksum differs from cache')
    for key in ('baseline_cache_metadata', 'dynamic_indices'):
        if record['joint_structure']['input_sha256'].get(key) != hashes[key]:
            raise ValueError(f'Correction input checksum mismatch: {key}')
    completed = verify_multiview_completion(cache, record, baseline=baseline)
    from utils.gaussian_cleanup import verify_gaussian_cleanup
    cleaned = verify_gaussian_cleanup(cache, record, baseline=baseline)
    if cleaned is not None:
        for field, artifact in (('report', 'cleanup_report'), ('overlay', 'cleanup_overlay')):
            inputs[artifact] = Path(cache['gaussian_cleanup'][field])
            hashes[artifact] = sha256(inputs[artifact])
    from utils.static_surface_rebuild import verify_static_surface_rebuild, static_surface_contract
    static_surface = verify_static_surface_rebuild(cache, record, baseline=baseline)
    if static_surface is not None:
        for field, artifact in (('report', 'static_surface_report'), ('extension', 'static_surface_extension')):
            inputs[artifact] = Path(cache['static_surface_rebuild'][field])
            hashes[artifact] = sha256(inputs[artifact])
    report = completed['report'] if completed is not None else None
    if completed is not None:
        del completed
    with np.load(masks, allow_pickle=False) as archive:
        mask_meta = json.loads(str(archive['metadata_json']))
        views = validate_view_plan(mask_meta, archive['frame_indices'], cache)
        validate_mask_arrays(archive['part_ids'], archive['depth'], mask_meta)
    for key in ('motion', 'mesh'):
        path = Path(mask_meta[key])
        if not path.is_absolute() or sha256(path) != mask_meta.get(key+'_sha256'):
            raise ValueError(f'Prior mask {key} checksum mismatch')
        inputs[key] = path
        hashes[key] = mask_meta[key+'_sha256']
    with np.load(inputs['motion'], allow_pickle=False) as archive:
        motion_meta = json.loads(str(archive['metadata_json']))
        motion_camera = json.loads(str(archive['camera_json']))
        quaternions, translations, group_ids = archive['quaternions'], archive['translations'], archive['group_ids']
    validate_motion_metadata(mask_meta, motion_meta, cache, record,
                             hashes['baseline_cache_metadata'], hashes['correction'], report)
    if motion_camera != cache['camera']:
        raise ValueError('Mask group-motion camera differs from Gaussian camera')
    group_count = len(record['joint_structure']['parents'])
    if (quaternions.shape != (cache['frame_count'], group_count, 4)
            or translations.shape != (cache['frame_count'], group_count, 3)
            or group_ids.shape != (len(mask_meta['objects']),) or group_ids.dtype.kind not in 'iu'
            or not np.isfinite(quaternions).all() or not np.isfinite(translations).all()):
        raise ValueError('Invalid prior group-motion arrays')
    expected_groups = np.full(len(group_ids), -1, np.int32)
    for group in record['joint_structure']['source_groups']:
        expected_groups[group['objects']] = group['group']
    if not np.array_equal(group_ids, expected_groups):
        raise ValueError('Mask group/part identity differs from candidate hard-joint groups')
    poses = np.load(baseline/'poses.npy', mmap_mode='r', allow_pickle=False)
    indices = np.load(baseline/'dynamic_indices.npy', allow_pickle=False)
    if (poses.dtype != np.float32 or poses.shape != (cache['frame_count'], cache['dynamic_count'], 7)
            or indices.shape != (cache['dynamic_count'],) or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= cache['gaussian_count'])
            or np.any(indices[1:] <= indices[:-1])):
        raise ValueError('Invalid baked Gaussian motion schema')
    torch.set_num_threads(2)
    model = correction_from_record(record, 'cpu')
    maximum_pose_error = 0.
    with torch.no_grad():
        for frame in sorted({view['frame'] for view in views}):
            q, t = model.group_transforms(frame)
            error = max(float(np.max(np.abs(q.numpy()-quaternions[frame]))),
                        float(np.max(np.abs(t.numpy()-translations[frame]))))
            maximum_pose_error = max(maximum_pose_error, error)
            if error > 1e-6:
                raise ValueError(f'Prior mask group motion differs at frame {frame}: {error}')
    # Existing audits may be unapproved, but any provided artifact identity must
    # be exact. A fresh suffix need not have an event audit yet.
    audit_path = candidate/'event-audit.json'
    audit_decision = None
    if audit_path.exists():
        event_audit = json.loads(audit_path.read_text())
        require_same_sampling(cache, record, event_audit)
        extra = [key for key in ('cleanup_report', 'cleanup_overlay', 'static_surface_report', 'static_surface_extension') if key in hashes]
        for key in ['source_ply', 'correction', 'baseline_cache_metadata', 'baseline_poses', 'dynamic_indices', *extra]:
            if event_audit.get('artifact_sha256', {}).get(key) != hashes[key]:
                raise ValueError(f'Existing event audit artifact differs: {key}')
        static_surface_contract(cache, record, event_audit)
        audit_decision = event_audit.get('approved_for_export')
        inputs['existing_event_audit'] = audit_path
        hashes['existing_event_audit'] = sha256(audit_path)
    return cache, record, mask_meta, sampling, inputs, hashes, maximum_pose_error, audit_decision


def contact_sheet(output, views, frame=None):
    from PIL import Image, ImageDraw
    chosen = [view for view in views if frame is None or view['frame'] == frame]
    columns, tile, title = (6 if frame is None else 8), 256, 26
    rows = (len(chosen)+columns-1)//columns
    sheet = Image.new('RGB', (columns*tile, rows*(tile+title)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for index, view in enumerate(chosen):
        x, y = index % columns*tile, index//columns*(tile+title)
        image_path = output/f'view-{view["view_index"]:02d}.png'
        with Image.open(image_path) as image:
            # Contact sheet only, never used for quantitative/visual pixel QA.
            sheet.paste(image.resize((tile, tile), Image.Resampling.NEAREST), (x, y+title))
        draw.text((x+3, y+6), f'f{view["frame"]} yaw {view["yaw"]} pitch {view["pitch"]}', fill='white')
    name = 'contact-all.png' if frame is None else f'contact-frame-{frame:04d}.png'
    sheet.save(output/name)
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'masks', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.baseline, args.candidate, args.masks = (path.resolve(strict=True)
                                              for path in (args.baseline, args.candidate, args.masks))
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    cache, record, mask_meta, sampling, inputs, hashes, pose_error, audit_decision = verify_inputs(
        args.baseline, args.candidate, args.masks)

    # Imports requiring the CUDA environment happen only after CPU provenance checks.
    import torch
    from PIL import Image
    from gaussian_renderer import render
    from render_ply import make_camera
    from scripts.refine_lego_events import MotionRenderer
    from utils.event_refinement import correction_from_record
    renderer = MotionRenderer(args.baseline, 640)
    require_same_sampling(renderer.meta, cache)
    correction = correction_from_record(record, 'cuda')
    views = mask_meta['views']
    gray = np.empty((len(views), 640, 640), np.float32)
    alpha = np.empty_like(gray)
    args.output.mkdir(parents=True, exist_ok=False)
    last_frame, timings = None, []
    with torch.no_grad():
        for index, view in enumerate(views):
            start = time.monotonic()
            frame = view['frame']
            if frame != last_frame:
                # The shared path applies both baked parent motion and compact
                # suffix motion; do not invent a separate posing implementation.
                renderer.image(frame, correction)
                last_frame = frame
            camera = make_camera(np.asarray(view['camera_pose']), 640, 640,
                                 np.deg2rad(cache['camera']['fovy_degrees']))
            result = render(camera, renderer.model, renderer.pipe, renderer.background, 0., 0., 0.)
            rgb = result['render'].clamp(0, 1)
            gray[index] = (rgb*renderer.luma).sum(0).cpu().numpy()
            alpha[index] = result['alpha'].reshape(640, 640).cpu().numpy()
            if (not np.isfinite(gray[index]).all() or not np.isfinite(alpha[index]).all()
                    or gray[index].min() < 0 or gray[index].max() > 1
                    or alpha[index].min() < 0 or alpha[index].max() > 1):
                raise ValueError(f'Invalid native Gaussian gray/alpha render at view {index}')
            image = np.rint(rgb.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
            Image.fromarray(image).save(args.output/f'view-{index:02d}.png')
            Image.fromarray(np.rint(alpha[index]*255).astype(np.uint8)).save(args.output/f'view-{index:02d}-alpha.png')
            seconds = time.monotonic()-start
            timings.append(seconds)
            print(json.dumps({'view_index': index, 'frame': frame, 'yaw': view['yaw'],
                              'pitch': view['pitch'], 'seconds': seconds}), flush=True)
    for key, path in inputs.items():
        if sha256(path) != hashes[key]:
            raise ValueError(f'Input changed during rendering: {key}')
    contacts = [contact_sheet(args.output, views)]
    if any(view['frame'] == 100 for view in views):
        contacts.append(contact_sheet(args.output, views, 100))
    metadata = {
        'format': 'lego_multiview_gaussian_render_v1', 'baseline': str(args.baseline),
        'candidate': str(args.candidate), 'masks': str(args.masks), 'masks_sha256': hashes['masks'],
        'motion': str(inputs['motion']), 'motion_sha256': hashes['motion'], 'mesh_sha256': hashes['mesh'],
        'artifact_sha256': hashes, 'views': views, 'view_count': len(views), 'resolution': [640, 640],
        'camera': cache['camera'], **sampling, 'background': 'black',
        'group_pose_maximum_absolute_difference': pose_error,
        'compact_suffix_present': cache.get('multiview_completion') is not None,
        'multiview_completion': cache.get('multiview_completion'),
        'gaussian_cleanup': cache.get('gaussian_cleanup'),
        'static_surface_rebuild': cache.get('static_surface_rebuild'),
        'diagnostic_only': True, 'approved_for_export': False, 'existing_event_audit_approved': audit_decision,
        'inputs_rechecked_after_render': True, 'native_pngs': [f'view-{i:02d}.png' for i in range(len(views))],
        'contact_sheets': contacts, 'view_seconds': timings,
        'array_encoding': 'gray and alpha are native 640x640 float32 Gaussian renders; compressed losslessly.',
        'image_encoding': 'Native RGB/alpha PNGs are rounded to uint8 without image filters. '
                          'Only contact overviews use nearest-neighbor resizing, for navigation not pixel QA.',
        'scope': 'UNAPPROVED diagnostic of actual Gaussian renders at exact rigid mesh-prior cameras and '
                 'event-corrected group motion. No event approval is inferred. The back and other unobserved '
                 'views have no monocular ground truth. Finite-view prior coverage is neither continuous '
                 'all-view completeness nor recovered unseen texture. Elastic bands remain in Gaussian '
                 'renders but are absent from rigid mesh-prior masks.',
    }
    np.savez_compressed(args.output/'render.npz', alpha=alpha, gray=gray,
                        metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))
    (args.output/'render.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output/'render.npz'), 'views': len(views),
                      'approved_for_export': False, 'source_ply_sha256': hashes['source_ply']}), flush=True)


if __name__ == '__main__':
    main()
