"""Render serialized detail-refined Gaussians and audit them without changing motion."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch

from scripts.events.refine_lego_events import MotionRenderer
from scripts.events.audit_lego_event_motion import audit
from utils.event_refinement import correction_from_record
from utils.rigid_checkpoint import sha256_file
from utils.render_sampling import require_same_sampling


def verify_appearance_record(cache, record, optimization_path, attributes_path):
    """Verify the exact packaged fitting evidence before any GPU work/output."""
    appearance = cache.get('appearance_refinement')
    if (not isinstance(appearance, dict) or appearance != record.get('appearance_refinement')
            or appearance.get('format') != 'lego_rigid_seed_appearance_refinement_v1'):
        raise ValueError('Cache/correction appearance_refinement provenance mismatch')
    optimization_path, attributes_path = Path(optimization_path), Path(attributes_path)
    if sha256_file(optimization_path) != appearance.get('optimization_sha256'):
        raise ValueError('Packaged optimization checksum mismatch')
    optimization = json.loads(optimization_path.read_text())
    attributes_hash = sha256_file(attributes_path)
    if (attributes_hash != appearance.get('attributes_sha256')
            or attributes_hash != optimization.get('attributes_sha256')):
        raise ValueError('Packaged appearance attributes checksum mismatch')
    inputs = optimization.get('input_sha256', {})
    if (any(not isinstance(appearance.get(key), str) or len(appearance[key]) != 64
            for key in ('parent_source_sha256', 'parent_rebuild_report_sha256'))
            or any(type(appearance.get(key)) is not int or appearance[key] <= 0
                   for key in ('optimized_rows', 'row_start'))
            or inputs.get('source_ply') != appearance.get('parent_source_sha256')
            or inputs.get('rebuild_report') != appearance.get('parent_rebuild_report_sha256')
            or optimization.get('seed_count') != appearance.get('optimized_rows')
            or optimization.get('retained_count') != appearance.get('row_start')):
        raise ValueError('Packaged appearance optimization parent or row counts mismatch')
    sampling = require_same_sampling(cache, record, optimization, appearance)
    return optimization, sampling


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'appearance-run', 'supervision'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    for name in ('rendered_gray.npy', 'event-audit.json', 'serialized-render-check.json'):
        if (args.candidate/name).exists():
            raise FileExistsError(args.candidate/name)
    cache = json.loads((args.baseline/'cache.json').read_text())
    record = json.loads((args.candidate/'correction.json').read_text())
    optimization, sampling = verify_appearance_record(
        cache, record, args.appearance_run/'optimization.json', args.appearance_run/'attributes.npz')
    if (optimization['format'] != 'lego_frozen_motion_surface_appearance_v1'
            or not optimization['positions_rotations_motion_frozen']
            or optimization['input_sha256']['supervision'] != sha256_file(args.supervision)
            or optimization['weights']['event'] <= 0):
        raise ValueError('Missing event-guided, frozen-motion appearance provenance')
    if record['format'] != 'event_supervised_hard_joint_alignment_v4':
        raise ValueError('Expected preserved hard-joint motion')
    renderer = MotionRenderer(args.baseline, 640)
    require_same_sampling(renderer.meta, cache)
    correction = correction_from_record(record, 'cuda')
    expected = np.load(args.appearance_run/'rendered_gray.npy', mmap_mode='r', allow_pickle=False)
    shape = (len(renderer.poses), 640, 640)
    if expected.shape != shape or not np.issubdtype(expected.dtype, np.floating):
        raise ValueError('Training render and packaged source timeline differ')
    rendered = np.lib.format.open_memmap(args.candidate/'rendered_gray.npy', mode='w+', dtype=np.float16, shape=shape)
    errors = []
    for index in range(len(rendered)):
        frame = renderer.image(index, correction).cpu().numpy()
        if not np.isfinite(frame).all() or frame.min() < 0 or frame.max() > 1:
            raise ValueError('Invalid serialized Gaussian render')
        rendered[index] = frame
        delta = np.abs(rendered[index].astype(np.float32)-expected[index].astype(np.float32))
        mae, maximum = float(delta.mean()), float(delta.max())
        if mae > 2e-5 or maximum > .002:
            raise ValueError(f'Serialized source differs from optimized model at frame {index}: {mae}, {maximum}')
        errors.append({'frame': index, 'mae': mae, 'maximum': maximum})
        if index in (0, 50, 100, 150, 200, 300, 399):
            Image.fromarray(np.rint(frame*255).astype(np.uint8)).save(args.candidate/f'frame_{index:04d}.png')
        if index % 100 == 0:
            print('Verified serialized detail render', index, flush=True)
    rendered.flush()
    check = {'frames': len(rendered), 'passed': True,
             **sampling,
             'maximum_frame_mae': max(row['mae'] for row in errors),
             'maximum_pixel_difference': max(row['maximum'] for row in errors),
             'training_render_sha256': sha256_file(args.appearance_run/'rendered_gray.npy'),
             'serialized_source_sha256': sha256_file(renderer.meta['source']), 'per_frame': errors}
    (args.candidate/'serialized-render-check.json').write_text(json.dumps(check, indent=2)+'\n')
    result = audit(rendered, args.supervision)
    result.update(event_optimization_performed=True, appearance_optimization_performed=True,
                  **sampling,
                  motion_optimization_inherited=True,
                  appearance_optimization={'record': str(args.appearance_run/'optimization.json'),
                      'sha256': sha256_file(args.appearance_run/'optimization.json'),
                      'scope': 'New rigid surface color, bounded size and opacity fitted to RGB, gradients and TRAIN events. Geometry and motion unchanged.'})
    result['artifact_sha256'] = {key: sha256_file(path) for key, path in {
        'correction': args.candidate/'correction.json', 'rendered_gray': args.candidate/'rendered_gray.npy',
        'supervision': args.supervision, 'baseline_cache_metadata': args.baseline/'cache.json',
        'baseline_poses': args.baseline/'poses.npy', 'dynamic_indices': args.baseline/'dynamic_indices.npy',
        'source_ply': renderer.meta['source']}.items()}
    result['correction'] = {key: value for key, value in record.items() if key != 'history'}
    (args.candidate/'event-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'summary': result['summary'], 'approved_for_export': result['approved_for_export'],
                      'serialized_render_maximum_mae': check['maximum_frame_mae']}, indent=2), flush=True)
    if not result['approved_for_export']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
