"""Render all 400 original-camera frames and independently audit compact priors.

No motion or appearance is optimized here. The inherited event-corrected parent
is explicitly distinguished from newly added unobserved mesh-prior surfaces.
Every original event interval still participates in the unchanged acceptance
gate. A completed but unapproved run exits with code 2, not a fake pass.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch

from scripts.events.audit_lego_event_motion import audit
from scripts.events.export_lego_event_preview import verify_candidate
from utils.event_refinement import correction_from_record
from utils.multiview_completion import completion_contract, verify_multiview_completion
from utils.render_sampling import require_same_sampling
from utils.rigid_checkpoint import sha256_file
from utils.gaussian_cleanup import cleanup_contract, verify_gaussian_cleanup
from utils.static_surface_rebuild import static_surface_contract, verify_static_surface_rebuild
from utils.color_residual import color_residual_contract, verify_color_residual
from utils.protected_surface_detail import protected_surface_contract, verify_protected_surface_detail


FRAME_COUNT = 400
WIDTH = 640
SNAPSHOT_FRAMES = (0, 50, 100, 150, 200, 300, 399)


def output_paths(candidate):
    candidate = Path(candidate)
    return [candidate / name for name in ('rendered_gray.npy', 'rendered_gray.npy.partial',
                                         'event-audit.json', 'completion-render-check.json')] + [
        candidate / f'frame_{index:04d}.png' for index in SNAPSHOT_FRAMES]


def validate_supervision_contract(meta, image_shape, count_shape, cache):
    if (type(cache.get('frame_count')) is not int or cache['frame_count'] != FRAME_COUNT
            or meta.get('frame_count') != FRAME_COUNT or image_shape != (FRAME_COUNT, WIDTH, WIDTH)
            or count_shape != (FRAME_COUNT-1, 2, WIDTH, WIDTH)
            or not isinstance(meta.get('fps'), (int, float)) or not np.isfinite(meta['fps'])
            or meta['fps'] <= 0 or meta['fps'] != cache.get('fps')):
        raise ValueError('Completion audit requires the complete 400-frame 640x640 original supervision timeline')


def prepare_inputs(baseline, candidate, supervision):
    """Fail bad contracts and overwrites before GPU initialization or output."""
    baseline, candidate, supervision = map(Path, (baseline, candidate, supervision))
    for path in output_paths(candidate):
        if path.exists():
            raise FileExistsError(path)
    cache = json.loads((baseline / 'cache.json').read_text())
    record = json.loads((candidate / 'correction.json').read_text())
    if completion_contract(cache, record) is None:
        raise ValueError('This audit requires an explicit compact multiview completion')
    sampling = require_same_sampling(cache, record)
    verified = verify_multiview_completion(cache, record, baseline=baseline)
    cleaned = verify_gaussian_cleanup(cache, record, baseline=baseline)
    static_surface = verify_static_surface_rebuild(cache, record, baseline=baseline)
    residual = verify_color_residual(cache, record, baseline=baseline)
    detail = verify_protected_surface_detail(cache, record, baseline=baseline)
    report = verified['report']
    parent_candidate = Path(report['parent_correction']).parent
    parent_cache, parent_record, parent_audit = verify_candidate(
        Path(report['parent_baseline']), parent_candidate, allow_unapproved=True)
    supervision_sha256 = sha256_file(supervision)
    if (parent_audit.get('artifact_sha256', {}).get('supervision') != supervision_sha256
            or parent_cache['source'] != report['parent_source']
            or parent_cache['camera'] != cache['camera']
            or parent_audit['artifact_sha256']['correction'] != report['parent_correction_sha256']):
        raise ValueError('Completion audit source supervision/camera differs from the audited parent')
    require_same_sampling(cache, parent_cache, parent_record, parent_audit)
    with np.load(supervision, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata_json']))
        validate_supervision_contract(meta, archive['images'].shape, archive['counts'].shape, cache)
    comparison_candidate = Path(cleaned['report']['parent_candidate']) if cleaned is not None else parent_candidate
    if static_surface is not None:
        comparison_candidate = Path(static_surface['report']['parent_candidate'])
    if residual is not None:
        comparison_candidate = Path(residual['report']['parent_candidate'])
    if detail is not None:
        comparison_candidate = Path(detail['report']['parent_candidate'])
    expected = np.load(comparison_candidate / 'rendered_gray.npy', mmap_mode='r', allow_pickle=False)
    if expected.shape != (FRAME_COUNT, WIDTH, WIDTH) or expected.dtype not in (np.float16, np.float32):
        raise ValueError('Audited parent rendering has an incompatible frame shape or dtype')
    needed = FRAME_COUNT * WIDTH * WIDTH * 2 + 64 * 1024**2
    if shutil.disk_usage(candidate).free < needed:
        raise OSError('Insufficient space for full completion render audit and reserve')
    return cache, record, verified, sampling, expected


def comparison_metrics(frame, parent):
    frame, parent = np.asarray(frame, np.float32), np.asarray(parent, np.float32)
    if (frame.shape != (WIDTH, WIDTH) or parent.shape != frame.shape
            or not np.isfinite(frame).all() or not np.isfinite(parent).all()
            or frame.min() < 0 or frame.max() > 1 or parent.min() < 0 or parent.max() > 1):
        raise ValueError('Completion/parent render must be finite 640x640 linear grayscale in [0,1]')
    delta = np.abs(frame-parent)
    foreground = parent > .08
    return {'mean_absolute_change': float(delta.mean()), 'maximum_pixel_change': float(delta.max()),
            'parent_foreground_mean_absolute_change': float(delta[foreground].mean()) if foreground.any() else 0.,
            'newly_bright_background_fraction': float(np.mean((parent < .03) & (frame > .08))),
            'parent_bright_pixels_below_coverage_threshold': int(np.sum((parent > .12) & (frame < .08)))}


def attach_provenance(result, cache, record, baseline, candidate, supervision):
    """Attach exact source identities without altering any event gate decision."""
    baseline, candidate = Path(baseline), Path(candidate)
    contract = completion_contract(cache, record)
    if contract is None:
        raise ValueError('Missing compact multiview completion audit contract')
    result = dict(result)
    result.update(**require_same_sampling(cache, record), multiview_completion=contract,
        event_optimization_performed=True, event_optimization_inherited=True,
        motion_optimization_inherited=True, geometry_prior_completion_performed=True,
        completion_motion_reoptimized=False, completion_appearance_event_optimized=False,
        unobserved_geometry_is_prior=True,
        completion_audit_scope='All 399 original single-camera raw-event intervals, using actual newly serialized '
          'Gaussian PLY plus compact inherited hard-joint motion. Existing parent event fitting is inherited; '
          'newly appended surfaces are NOT newly event-trained. Back/side appearance and geometry remain '
          'known-mesh priors, not event-validated unseen truth.')
    paths = {'correction': candidate / 'correction.json', 'rendered_gray': candidate / 'rendered_gray.npy',
             'supervision': Path(supervision), 'baseline_cache_metadata': baseline / 'cache.json',
             'baseline_poses': baseline / 'poses.npy', 'dynamic_indices': baseline / 'dynamic_indices.npy',
             'source_ply': Path(cache['source']), 'completion_report': Path(contract['report']),
             'completion_extension': Path(contract['extension'])}
    cleanup = cleanup_contract(cache, record)
    if cleanup is not None:
        attributes = json.loads(Path(cleanup['report']).read_text())['attributes']
        result.update(gaussian_cleanup=cleanup, gaussian_attribute_cleanup_performed=True,
                      gaussian_opacity_cleanup_performed='opacity' in attributes,
                      gaussian_dc_prior_adjustment_performed='dc_gray' in attributes,
                      cleanup_motion_reoptimized=False,
                      cleanup_scope='Actual static-Gaussian attribute overlay over the unchanged serialized source. '
                        'This fresh audit includes the cleanup; old event decisions are not reused.')
        paths.update(cleanup_report=Path(cleanup['report']), cleanup_overlay=Path(cleanup['overlay']))
    surface = static_surface_contract(cache, record)
    if surface is not None:
        result.update(static_surface_rebuild=surface, static_roof_surface_rebuilt=True,
                      static_rebuild_motion_reoptimized=False,
                      static_rebuild_scope='Explicit appended static roof Gaussian surfaces plus retired source opacities. '
                        'Source PLY and inherited event-corrected motion are unchanged; this fresh render includes the replacement.')
        paths.update(static_surface_report=Path(surface['report']), static_surface_extension=Path(surface['extension']))
    residual = color_residual_contract(cache, record)
    if residual is not None:
        result.update(color_residual=residual, observed_color_residual_fitted=True,
            residual_motion_reoptimized=False,
            color_residual_scope='Bounded same-chroma source Gaussian color increments; geometry, opacity, '
                'covariance, source rows and corrected motion are unchanged. Actual residual colors are '
                'included in this fresh full-event audit. Unseen views are not newly observed.',
            validation_note='Existing event-heldout intervals and every fifth RGB timestamp were withheld '
                'from this residual stage only. Parent-stage RGB independence is not claimed.')
        paths.update(color_residual_report=Path(residual['report']), color_residual=Path(residual['residual']))
    detail = protected_surface_contract(cache, record)
    if detail is not None:
        result.update(protected_surface_detail=detail, protected_static_surface_refinement=True,
            surface_detail_motion_reoptimized=False,
            surface_detail_scope='Guarded static Gaussian retirement and exact-triangle thin surface additions, '
                'after the inherited color residual. All source motion is unchanged; this fresh full-timeline '
                'render audit includes both layers. Unseen appearance and geometry remain explicit priors.')
        paths.update(surface_detail_report=Path(detail['report']), surface_detail_extension=Path(detail['extension']))
    result['artifact_sha256'] = {key: sha256_file(path) for key, path in paths.items()}
    if (result['artifact_sha256']['completion_report'] != contract['report_sha256']
            or result['artifact_sha256']['completion_extension'] != contract['extension_sha256']):
        raise ValueError('Completion artifacts changed while rendering')
    if detail is not None and (
            result['artifact_sha256']['surface_detail_report'] != detail['report_sha256']
            or result['artifact_sha256']['surface_detail_extension'] != detail['extension_sha256']):
        raise ValueError('Protected surface detail artifacts changed while rendering')
    return result


def write_new_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'supervision'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    cache, record, verified, sampling, parent_frames = prepare_inputs(args.baseline, args.candidate, args.supervision)
    # Delay the CUDA renderer import/initialization until all CPU-only preflight
    # checks have passed; malformed contracts never trigger a partial GPU job.
    from scripts.events.refine_lego_events import MotionRenderer
    renderer = MotionRenderer(args.baseline, WIDTH)
    require_same_sampling(renderer.meta, cache)
    if renderer.completion is None or len(renderer.poses) != FRAME_COUNT:
        raise ValueError('Renderer did not load the complete compact-motion timeline')
    correction = correction_from_record(record, 'cuda')
    temporary = args.candidate / 'rendered_gray.npy.partial'
    target = args.candidate / 'rendered_gray.npy'
    rendered = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16,
                                         shape=(FRAME_COUNT, WIDTH, WIDTH))
    comparisons = []
    for index in range(FRAME_COUNT):
        frame = renderer.image(index, correction).cpu().numpy()
        metrics = comparison_metrics(frame, parent_frames[index])
        rendered[index] = frame
        comparisons.append({'frame': index, **metrics})
        if index in SNAPSHOT_FRAMES:
            Image.fromarray(np.rint(frame * 255).astype(np.uint8)).save(args.candidate / f'frame_{index:04d}.png')
        if (index+1) % 40 == 0 or index == 0:
            rendered.flush()
            print(json.dumps({'stage': 'original_camera_render', 'frame': index+1, 'frames': FRAME_COUNT,
                              'mean_absolute_change_from_parent': metrics['mean_absolute_change']}), flush=True)
    rendered.flush()
    del rendered
    temporary.rename(target)
    rendered = np.load(target, mmap_mode='r', allow_pickle=False)
    print('Auditing all 399 unchanged raw-event intervals', flush=True)
    result = attach_provenance(audit(rendered, args.supervision), cache, record,
                               args.baseline, args.candidate, args.supervision)
    parent_report = verified['report']
    for key, expected in (('baseline_poses', parent_report['parent_poses_sha256']),
                          ('dynamic_indices', parent_report['parent_dynamic_indices_sha256'])):
        if result['artifact_sha256'][key] != expected:
            raise ValueError('Completion render changed inherited motion: ' + key)
    check = {'format': 'lego_compact_completion_render_check_v1', 'frames': FRAME_COUNT,
             'resolution': [WIDTH, WIDTH], **sampling,
             'all_render_frames_finite_in_unit_range': True, 'original_camera_used': True,
             'parent_baked_motion_hashes_unchanged': True, 'gaussian_prefix_unchanged': True,
             'new_suffix_uses_same_corrected_group_se3': True,
             'completion_report_sha256': result['artifact_sha256']['completion_report'],
             'completion_extension_sha256': result['artifact_sha256']['completion_extension'],
             'serialized_source_sha256': result['artifact_sha256']['source_ply'],
             'rendered_gray_sha256': result['artifact_sha256']['rendered_gray'],
             'mean_frame_change_from_parent': float(np.mean([row['mean_absolute_change'] for row in comparisons])),
             'maximum_frame_change_from_parent': max(row['mean_absolute_change'] for row in comparisons),
             'worst_change_frame': max(comparisons, key=lambda row: row['mean_absolute_change'])['frame'],
             'parent_comparison_per_frame': comparisons,
             'event_approved_for_export': result['approved_for_export'],
             'scope': 'Serialized rendering/provenance check, not visual completeness acceptance. '
                      'Original-view differences are reported, not hidden or threshold-relaxed. '
                      'Novel viewpoints require a separate mesh-prior consistency and visual check.'}
    if cleanup_contract(cache, record) is not None:
        check.update(gaussian_cleanup=cache['gaussian_cleanup'],
                     gaussian_prefix_unchanged_scope='Serialized source only; explicitly audited static attribute overlay is applied at render time.',
                     comparison_reference='The cleanup report parent candidate, before applying this overlay.')
    if static_surface_contract(cache, record) is not None:
        check.update(static_surface_rebuild=cache['static_surface_rebuild'],
                     gaussian_prefix_unchanged_scope='Serialized source only; roof retirement opacity and appended static Gaussians are separately audited.',
                     comparison_reference='The static surface report parent candidate, before replacing the roof.')
    if color_residual_contract(cache, record) is not None:
        check.update(color_residual=cache['color_residual'],
                     gaussian_prefix_unchanged_scope='Serialized source and geometry only; audited RGB residual is applied at render time.',
                     comparison_reference='The residual report parent candidate, before applying measured color residuals.')
    if protected_surface_contract(cache, record) is not None:
        check.update(protected_surface_detail=cache['protected_surface_detail'],
                     gaussian_prefix_unchanged_scope='Serialized source only; guarded retirement and added static surfaces are applied after color residuals.',
                     comparison_reference='The protected surface detail parent candidate, including its measured color residual.')
    write_new_json(args.candidate / 'completion-render-check.json', check)
    write_new_json(args.candidate / 'event-audit.json', result)
    verify_candidate(args.baseline, args.candidate, allow_unapproved=True)
    print(json.dumps({'summary': result['summary'], 'approved_for_export': result['approved_for_export'],
                      'mean_frame_change_from_parent': check['mean_frame_change_from_parent'],
                      'maximum_frame_change_from_parent': check['maximum_frame_change_from_parent'],
                      'worst_change_frame': check['worst_change_frame']}, indent=2), flush=True)
    if not result['approved_for_export']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
