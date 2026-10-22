"""Remove appended priors exposed by the actual original-camera rasterizer.

Uniform DC=0 gives RGB=.5, preventing color clamping. For a DC-only Gaussian,
sum_RGB(d sum_pixels(gray) / d DC) / C0 is its accumulated alpha contribution
in output-pixel units. Removing exposed points can reveal other points, so a
complete 400-frame zero-removal pass is required before declaring convergence.
Only a NEW filtered selection is written; the parent and completed PLYs remain
unchanged. This is visibility filtering, not event training or event approval.
The CUDA early-termination trigger may have zero DC gradient while truncating
later contributions. Zero exposure is therefore NOT exact RGB invariance;
actual rebuilt 400-frame RGB/event parity remains mandatory.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from scripts.events.export_lego_event_preview import verify_candidate
from utils.event_refinement import correction_from_record
from utils.mesh_surface_sampling import normals_to_wxyz
from utils.multiview_completion import verify_multiview_completion
from utils.render_sampling import render_sampling_contract
from utils.rigid_checkpoint import sha256_file


C0 = .28209479177387814
SELECTION_KEYS = ('xyz', 'normals', 'part_ids', 'material_gray', 'triangle_ids')


def contribution_from_dc_gradient(gradient):
    if (gradient is None or gradient.ndim != 3 or gradient.shape[1:] != (1, 3)
            or not torch.is_floating_point(gradient) or not torch.isfinite(gradient).all()
            or torch.any(gradient < -1e-7)):
        raise ValueError('Invalid/nonpositive raster DC gradient; alpha exposure cannot be inferred')
    value = gradient.sum((1, 2)) / C0
    if not torch.isfinite(value).all() or torch.any(value < -1e-6):
        raise ValueError('Nonfinite or negative alpha contribution')
    return value.clamp_min(0)


def validate_options(max_rounds, probe_stride, threshold, max_probe_rounds=32):
    if (type(max_rounds) is not int or not 1 <= max_rounds <= 8
            or type(probe_stride) is not int or not 1 <= probe_stride <= 100
            or type(max_probe_rounds) is not int or not 1 <= max_probe_rounds <= 64
            or not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
            or not np.isfinite(threshold) or not 0 < threshold <= .01):
        raise ValueError('Require max-rounds 1..8, max-probe-rounds 1..64, probe-stride 1..100, '
                         'and threshold in (0,.01] pixels')


def run_pruning(scan, count, max_rounds=4, probe_stride=10, threshold=.0005, *,
                max_probe_rounds=32, initial_active=None):
    """Independent full passes: never hide a nonconverged last deletion round."""
    validate_options(max_rounds, probe_stride, threshold, max_probe_rounds)
    if type(count) is not int or not 1 <= count <= 600000:
        raise ValueError('Invalid initial compact suffix count')
    if initial_active is None:
        active = np.ones(count, bool)
    else:
        if not isinstance(initial_active, np.ndarray) or initial_active.dtype != np.bool_ or initial_active.shape != (count,):
            raise ValueError('Initial exposure mask must be an exact boolean source-row mask')
        active = initial_active.copy()
    initial_active_count = int(active.sum())
    maximum_seen = np.zeros(count, np.float32)
    history, converged = [], False
    probe = sorted(set(range(0, 400, probe_stride)) | {399})
    last = np.zeros(count, np.float32)
    for phase, frames, limit in (('probe', probe, max_probe_rounds), ('full', list(range(400)), max_rounds)):
        for number in range(1, limit+1):
            before = int(active.sum())
            value = np.asarray(scan(active.copy(), frames, phase, number))
            if value.shape != (count,) or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError('Invalid per-Gaussian exposure scan result')
            maximum_seen = np.maximum(maximum_seen, value)
            reject = active & (value > threshold)
            active[reject] = False
            last = value
            row = {'phase': phase, 'round': number, 'frames_checked': len(frames), 'active_before': before,
                   'removed': int(reject.sum()), 'retained': int(active.sum()),
                   'maximum_active_contribution_before': float(value[active | reject].max()) if before else 0.,
                   'maximum_retained_contribution_after': float(value[active].max()) if active.any() else 0.}
            history.append(row)
            print(json.dumps({'stage': 'exposure_pruning_round', **row}), flush=True)
            if phase == 'full' and not reject.any():
                converged = True
            if not reject.any():
                break
    return active, {'converged_full_400': converged, 'initial_count': count, 'retained_count': int(active.sum()),
                    'initial_active_count': initial_active_count,
                    'previously_removed_count': count-initial_active_count,
                    'removed_this_run_count': initial_active_count-int(active.sum()),
                    'removed_count': int((~active).sum()), 'retained_fraction': float(active.mean()),
                    'threshold_output_pixels': threshold, 'max_full_rounds': max_rounds,
                    'probe_round_cap': max_probe_rounds, 'probe_stride': probe_stride, 'probe_frames': probe,
                    'probe_converged': [row for row in history if row['phase'] == 'probe'][-1]['removed'] == 0,
                    'history': history, 'final_pass_frames': history[-1]['frames_checked'],
                    'maximum_retained_contribution': float(last[active].max()) if active.any() else 0.,
                    'sum_retained_per_frame_max_contribution': float(last[active].sum(dtype=np.float64)),
                    'maximum_seen_contribution_quantiles': np.quantile(maximum_seen, [0, .5, .9, .99, 1]).tolist(),
                    'scope': 'Finite original-camera raster alpha contribution at all 400 supplied times; '
                             'no continuous-time, arbitrary-view completeness or event acceptance claim.'}


def validate_selection(selection, verified):
    arrays, report = verified['arrays'], verified['report']
    count = report['added_gaussians']
    for key in SELECTION_KEYS:
        expected = (count, 3) if key in ('xyz', 'normals') else (count,)
        if key not in selection or selection[key].shape != expected or not np.isfinite(selection[key]).all():
            raise ValueError('Invalid source selection array: '+key)
    if (not np.array_equal(selection['xyz'], arrays['canonical_xyz'])
            or not np.array_equal(selection['part_ids'], arrays['part_ids'])
            or not np.array_equal(normals_to_wxyz(selection['normals']), arrays['canonical_quaternions'])):
        raise ValueError('Source selection does not match the exact completed Gaussian suffix')
    meta = json.loads(str(selection['metadata_json']))
    if meta != report['selection'] or meta.get('added_points') != count:
        raise ValueError('Selection metadata differs from the audited completion report')
    return meta


def filtered_selection(selection, active, exposure):
    if active.dtype != np.bool_ or active.shape != (len(selection['xyz']),):
        raise ValueError('Invalid exposure keep mask')
    metadata = copy.deepcopy(json.loads(str(selection['metadata_json'])))
    metadata['added_points'] = int(active.sum())
    metadata['exposure_pruning'] = copy.deepcopy(exposure)
    result = {key: selection[key][active] for key in SELECTION_KEYS}
    result['metadata_json'] = np.asarray(json.dumps(metadata, allow_nan=False))
    return result


def exposure_identity(cache, record_report, audit, selection_path, threshold):
    params = record_report['parameters']
    return {'source_selection': str(Path(selection_path).resolve()),
        'source_selection_sha256': sha256_file(selection_path),
        'source_completion_report': cache['multiview_completion']['report'],
        'source_completion_report_sha256': cache['multiview_completion']['report_sha256'],
        'source_ply_sha256': audit['artifact_sha256']['source_ply'],
        'source_correction_sha256': audit['artifact_sha256']['correction'],
        'parent_source_sha256': record_report['parent_source_sha256'],
        'parent_correction_sha256': record_report['parent_correction_sha256'],
        'render_model_parameters': {key: params[key] for key in ('spacing', 'tangent_factor', 'normal_factor', 'opacity')},
        'render_sampling': render_sampling_contract(cache), 'camera': cache['camera'],
        'threshold_output_pixels': threshold}


def selection_identity_keys(selection):
    """Lossless float32-bit/part/triangle keys; no rounded spatial matching."""
    xyz, parts, triangles = (selection[key] for key in ('xyz', 'part_ids', 'triangle_ids'))
    if (xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
            or parts.dtype != np.int32 or parts.shape != (len(xyz),)
            or triangles.dtype != np.int32 or triangles.shape != (len(xyz),)):
        raise ValueError('Resume selection requires exact float32 XYZ and int32 source part/triangle identities')
    keys = np.empty(len(xyz), dtype=[('x', '<u4'), ('y', '<u4'), ('z', '<u4'), ('part', '<i4'), ('triangle', '<i4')])
    bits = np.ascontiguousarray(xyz).view(np.uint32)
    for axis, name in enumerate(('x', 'y', 'z')):
        keys[name] = bits[:, axis]
    keys['part'], keys['triangle'] = parts, triangles
    return keys


def resume_active_mask(original, resumed, expected_identity):
    """Restore only an exact, provenance-matched subset of the original v1 rows."""
    meta = json.loads(str(resumed['metadata_json']))
    exposure = meta.get('exposure_pruning')
    if (not isinstance(exposure, dict) or exposure.get('format') != 'lego_raster_alpha_exposure_pruning_v1'
            or exposure.get('original_prefix_unchanged') is not True
            or exposure.get('source_artifacts_unchanged') is not True
            or type(exposure.get('converged_full_400')) is not bool):
        raise ValueError('Resume selection lacks an explicit exposure pruning provenance record')
    for key, expected in expected_identity.items():
        if exposure.get(key) != expected:
            raise ValueError('Resume exposure provenance mismatch: '+key)
    old_meta = json.loads(str(original['metadata_json']))
    compare = {key: value for key, value in meta.items() if key not in ('added_points', 'exposure_pruning')}
    old_compare = {key: value for key, value in old_meta.items() if key not in ('added_points', 'exposure_pruning')}
    if compare != old_compare or meta.get('added_points') != len(resumed['xyz']):
        raise ValueError('Resume geometric selection parameters or point count changed')
    original_keys, selected_keys = selection_identity_keys(original), selection_identity_keys(resumed)
    order = np.argsort(original_keys)
    sorted_keys = original_keys[order]
    if len(sorted_keys) > 1 and np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError('Original selection contains ambiguous duplicate source identities')
    positions = np.searchsorted(sorted_keys, selected_keys)
    if (np.any(positions >= len(sorted_keys)) or not np.array_equal(sorted_keys[positions], selected_keys)
            or len(np.unique(positions)) != len(positions)):
        raise ValueError('Resume selection is not an exact unique subset of the original source rows')
    rows = order[positions]
    for key in SELECTION_KEYS:
        if key not in resumed or resumed[key].dtype != original[key].dtype or not np.array_equal(resumed[key], original[key][rows]):
            raise ValueError('Resume selection changed source Gaussian geometry/material attributes: '+key)
    active = np.zeros(len(original_keys), bool)
    active[rows] = True
    return active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--selection', type=Path,
                        help='Defaults to the audited completion model/selection.npz')
    parser.add_argument('--resume-selection', type=Path,
                        help='Continue an exact exposure-filtered subset of the original audited v1 selection')
    parser.add_argument('--max-rounds', type=int, default=4,
                        help='At most this many full 400-frame passes; never marks an unfinished pass converged')
    parser.add_argument('--max-probe-rounds', type=int, default=32,
                        help='Bounded cheap probe peeling passes before full-frame verification (1..64)')
    parser.add_argument('--probe-stride', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=.0005,
                        help='Maximum alpha contribution per appended Gaussian in output-pixel units')
    args = parser.parse_args()
    validate_options(args.max_rounds, args.probe_stride, args.threshold, args.max_probe_rounds)
    if args.output.suffix != '.npz':
        raise ValueError('Output must name a new .npz selection')
    report_path = args.output.with_suffix('.json')
    if args.output.exists() or report_path.exists():
        raise FileExistsError('Selection or exposure report output already exists')
    cache, record, audit = verify_candidate(args.baseline, args.candidate, allow_unapproved=True)
    verified = verify_multiview_completion(cache, record, audit, baseline=args.baseline)
    if verified is None or cache['frame_count'] != 400:
        raise ValueError('Requires an audited compact completion with all 400 source frames')
    report = verified['report']
    selection_path = args.selection or Path(report['completed_source']).parent/'selection.npz'
    extension_meta = json.loads(str(verified['arrays']['metadata_json']))
    if sha256_file(selection_path) != extension_meta.get('selection_sha256'):
        raise ValueError('Selection is not the exact provenance-bound completed selection')
    with np.load(selection_path, allow_pickle=False) as archive:
        selection = {key: archive[key] for key in archive.files}
    validate_selection(selection, verified)
    identity = exposure_identity(cache, report, audit, selection_path, args.threshold)
    initial_active, resume_provenance = None, None
    if args.resume_selection is not None:
        with np.load(args.resume_selection, allow_pickle=False) as archive:
            resumed = {key: archive[key] for key in archive.files}
        initial_active = resume_active_mask(selection, resumed, identity)
        resume_provenance = {'selection': str(args.resume_selection.resolve()),
                             'sha256': sha256_file(args.resume_selection),
                             'initial_active_count': int(initial_active.sum())}
        print(json.dumps({'stage': 'resume_verified', **resume_provenance}), flush=True)
        del resumed
    from scripts.events.refine_lego_events import MotionRenderer
    torch.set_num_threads(2)
    renderer = MotionRenderer(args.baseline, 640)
    correction = correction_from_record(record, 'cuda').eval()
    for parameter in correction.parameters():
        parameter.requires_grad_(False)
    prefix, count = report['prefix_count'], report['added_gaussians']
    base_opacity = renderer.model._opacity.detach().clone()
    prefix_dc = torch.zeros_like(renderer.model._features_dc[:prefix])
    suffix_dc = torch.zeros_like(renderer.model._features_dc[prefix:], requires_grad=True)
    if suffix_dc.shape != (count, 1, 3):
        raise ValueError('Unexpected DC suffix shape')

    def scan(active, frames, phase, number):
        with torch.no_grad():
            renderer.model._opacity = base_opacity.clone()
            inactive = torch.from_numpy(~active).to(device=base_opacity.device)
            renderer.model._opacity[prefix:][inactive] = -30.
        maxima = torch.zeros(count, dtype=torch.float32, device='cuda')
        for position, frame_index in enumerate(frames):
            suffix_dc.grad = None
            # Recreate concatenation each backward pass; no old autograd graph
            # is retained across frames, and parent DC values are constants.
            renderer.model._features_dc = torch.cat((prefix_dc, suffix_dc), 0)
            frame = renderer.image(frame_index, correction)
            if (not torch.isfinite(frame).all() or torch.any(frame < -1e-7) or torch.any(frame > .50001)):
                raise ValueError('Uniform .5 colors produced a clamped/nonfinite image; derivative would be invalid')
            frame.sum().backward()
            contribution = contribution_from_dc_gradient(suffix_dc.grad)
            maxima = torch.maximum(maxima, contribution.detach())
            if position == 0 or (position+1) % 40 == 0 or position+1 == len(frames):
                print(json.dumps({'stage': 'alpha_gradient_scan', 'phase': phase, 'round': number,
                    'frame': frame_index, 'progress': position+1, 'frames': len(frames),
                    'above_threshold_so_far': int(torch.count_nonzero(maxima > args.threshold)),
                    'maximum_alpha_contribution': float(maxima.max())}), flush=True)
            del frame, contribution
        return maxima.cpu().numpy()

    active, stats = run_pruning(scan, count, args.max_rounds, args.probe_stride, args.threshold,
                                max_probe_rounds=args.max_probe_rounds, initial_active=initial_active)
    exposure = {'format': 'lego_raster_alpha_exposure_pruning_v1', **stats,
        **identity, 'resumed_from': resume_provenance,
        'formula': 'sum_RGB(d sum_output_pixels(grayscale) / d f_dc_RGB) / SH_C0',
        'uniform_rgb': .5, 'SH_C0': C0, 'removed_opacity_logit': -30.,
        'original_prefix_unchanged': True, 'source_artifacts_unchanged': True,
        'event_optimized': False, 'event_approved': False,
        'limitations': [
            'DC gradients measure differentiable color contributions, not the discrete CUDA early-termination trigger. '
            'A zero-gradient Gaussian may still stop later compositing; zero exposure does not guarantee exact RGB parity.',
            'The threshold applies to each Gaussian, not the whole image; summed retained maxima provide only a contribution reference.',
            'The rebuilt compact model must pass independent full-400 RGB comparison and unchanged event checks before deployment.'],
        'reuse_constraint': 'Same parent, inherited correction, scales, opacity, camera and sampling operator; '
                           'new complete event/render audits required. Nonconverged selections must not be deployed.'}
    output = filtered_selection(selection, active, exposure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, **output)
    result = {**exposure, 'output_selection': str(args.output.resolve()), 'output_sha256': sha256_file(args.output),
              'status': 'CONVERGED_FULL_400_REQUIRES_REBUILD_AND_EVENT_AUDIT'
                        if stats['converged_full_400'] and active.any() else 'NOT_USABLE_REQUIRES_REVIEW'}
    with report_path.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    if not stats['converged_full_400'] or not active.any():
        raise SystemExit(2)


if __name__ == '__main__':
    main()
