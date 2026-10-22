"""Measured global/local Gaussian-color residual fitting, with no geometry edits.

Only training-camera-visible, nonelastic parent rows are optimized. A small
linear-RGB residual preserves chroma. Pixel/derivative losses use measured video;
training events constrain temporal changes; existing novel views are trust
references, not new observations. Held-out timestamps/events never select the
checkpoint. This produces a diagnostic candidate, not automatic UI deployment.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation

from scripts.events.export_lego_event_preview import verify_candidate
from scripts.geometry.prune_lego_completion_exposure import contribution_from_dc_gradient
from scripts.diagnostics.measure_lego_detail import measure_detail, summarize
from utils.event_refinement import correction_from_record
from utils.residual_appearance import (BoundedColorResidual, split_timeline,
    multiscale_detail_terms, local_detail_loss, normalized_event_loss, temporal_nonregression_loss, DETAIL_ROIS)
from utils.rigid_checkpoint import sha256_file


def write_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'supervision', 'multiview', 'output', 'render-output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--iterations', type=int, default=240)
    p.add_argument('--maximum-delta', type=float, default=.08)
    p.add_argument('--visibility-threshold', type=float, default=.001)
    p.add_argument('--lr', type=float, default=.004)
    p.add_argument('--event-weight', type=float, default=.25)
    p.add_argument('--trust-weight', type=float, default=2.)
    p.add_argument('--temporal-guard-weight', type=float, default=.8)
    args = p.parse_args()
    if args.output.exists() or args.render_output.exists():
        raise FileExistsError('Use new diagnostic output and render paths')
    if (not 1 <= args.iterations <= 2000 or not 0 < args.maximum_delta <= .15
            or not 0 < args.visibility_threshold <= .1 or not 0 < args.lr <= .02
            or not 0 <= args.event_weight <= 2 or not 0 <= args.trust_weight <= 10
            or not 0 <= args.temporal_guard_weight <= 5):
        raise ValueError('Invalid bounded optimization settings')
    for name in ('baseline', 'candidate', 'supervision', 'multiview'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    torch.set_num_threads(4)
    torch.manual_seed(31)
    rng = np.random.default_rng(31)
    print('Verifying immutable parent candidate and supervision', flush=True)
    cache, record, parent_audit = verify_candidate(args.baseline, args.candidate, allow_unapproved=True)
    if (cache['frame_count'] != 400 or cache.get('render_supersample', 1) != 1
            or any(cache.get(k) for k in ('gaussian_cleanup', 'static_surface_rebuild', 'color_residual'))):
        raise ValueError('Require unmodified native 400-frame compact-completion parent')
    input_paths = {'source_ply': Path(cache['source']), 'correction': args.candidate/'correction.json',
        'baseline_cache_metadata': args.baseline/'cache.json', 'baseline_poses': args.baseline/'poses.npy',
        'dynamic_indices': args.baseline/'dynamic_indices.npy', 'event_audit': args.candidate/'event-audit.json',
        'supervision': args.supervision, 'multiview_render': args.multiview}
    hashes = {k: sha256_file(path) for k, path in input_paths.items()}
    if hashes['supervision'] != parent_audit['artifact_sha256']['supervision']:
        raise ValueError('Supervision differs from audited parent')
    completed = json.loads(Path(cache['multiview_completion']['report']).read_text())
    rebuild = json.loads(Path(completed['parent_report']).read_text())
    with np.load(rebuild['binding'], allow_pickle=False) as archive:
        labels = archive['labels']
    with np.load(completed['parts'], allow_pickle=False) as archive:
        part_meta = json.loads(str(archive['metadata_json']))
    if len(labels) != completed['prefix_count']:
        raise ValueError('Parent row binding does not match compact prefix')
    allowed = np.zeros(cache['gaussian_count'], bool)
    allowed[:len(labels)] = ~np.isin(labels, part_meta['deforming_objects'])
    with np.load(args.multiview, allow_pickle=False) as archive:
        view_meta = json.loads(str(archive['metadata_json']))
        view_reference = archive['gray']
    if (view_meta['artifact_sha256']['correction'] != hashes['correction']
            or view_meta['artifact_sha256']['source_ply'] != hashes['source_ply']
            or view_reference.shape != (36, 640, 640)):
        raise ValueError('Novel-view trust renders do not identify the exact parent')
    with np.load(args.supervision, allow_pickle=False) as archive:
        images, counts = archive['images'], archive['counts']
        supervision_meta = json.loads(str(archive['metadata_json']))
    if images.shape != (400, 640, 640) or images.dtype != np.uint8 or counts.shape != (399, 2, 640, 640):
        raise ValueError('Invalid native supervision timeline')
    train, held_events, held_rgb = split_timeline(400)
    floor, thresholds = (supervision_meta['calibration'][key] for key in ('log_floor', 'thresholds'))
    event_arrays = {factor: np.empty((399, 640//factor, 640//factor), np.float32) for factor in (4, 8)}
    for index, count in enumerate(counts):
        signed = count[0].astype(np.float32)*thresholds[0]-count[1].astype(np.float32)*thresholds[1]
        for factor, target in event_arrays.items():
            side = 640//factor
            target[index] = signed.reshape(side, factor, side, factor).mean((1, 3))
    del counts
    train = train[np.abs(event_arrays[8][train]).sum((1, 2)) > 1]
    if len(train) < 32:
        raise ValueError('Insufficient observable training intervals')
    training_frames = np.unique(np.r_[train, train+1])
    hard_intervals = np.array([int(i) for i in train if (
        not parent_audit['intervals'][i]['passed']
        or parent_audit['intervals'][i]['metrics']['relative_l1'] > parent_audit['intervals'][i]['thresholds']['relative_l1']-.2
        or parent_audit['intervals'][i]['metrics']['polarity_agreement'] < parent_audit['intervals'][i]['thresholds']['polarity_agreement']+.05)], dtype=np.int32)
    parent_frames = np.load(args.candidate/'rendered_gray.npy', mmap_mode='r', allow_pickle=False)
    parent_change = np.zeros_like(event_arrays[8])
    for index in train:
        log_delta = np.log(np.maximum(parent_frames[index+1].astype(np.float32), floor))-np.log(np.maximum(parent_frames[index].astype(np.float32), floor))
        parent_change[index] = log_delta.reshape(80, 8, 80, 8).mean((1, 3))
    region = binary_dilation((images[training_frames] > 8).any(0), iterations=2)
    args.output.mkdir(parents=True, exist_ok=False)
    args.render_output.parent.mkdir(parents=True, exist_ok=True)
    from scripts.events.refine_lego_events import MotionRenderer
    from render_ply import make_camera
    renderer = MotionRenderer(args.baseline, 640)
    correction = correction_from_record(record, 'cuda').eval()
    for parameter in correction.parameters():
        parameter.requires_grad_(False)
    original_camera = renderer.camera
    original_dc = renderer.model._features_dc.detach().clone()
    protected = {k: getattr(renderer.model, k).detach().clone() for k in ('_scaling', '_opacity')}
    reference = torch.from_numpy(images).cuda()
    events = {factor: torch.from_numpy(value).cuda() for factor, value in event_arrays.items()}
    parent_events = torch.from_numpy(parent_change).cuda()
    weight = torch.from_numpy(np.where(region, 1., .1).astype(np.float32)).cuda()

    # Uniform-color gradients measure actual raster visibility, not screen
    # nearest-part labels. No RGB/event-heldout target enters this selection.
    probe_frames = training_frames[np.linspace(0, len(training_frames)-1, 40).astype(int)]
    maximum = torch.zeros(cache['gaussian_count'], device='cuda')
    probe_dc = torch.zeros_like(original_dc, requires_grad=True)
    for frame in probe_frames:
        probe_dc.grad = None
        renderer.model._features_dc = probe_dc
        image = renderer.image(int(frame), correction)
        image.sum().backward()
        maximum = torch.maximum(maximum, contribution_from_dc_gradient(probe_dc.grad).detach())
    eligible = torch.from_numpy(allowed).cuda() & (maximum >= args.visibility_threshold)
    indices = torch.nonzero(eligible).flatten()
    visibility = maximum[indices].cpu().numpy()
    del probe_dc, maximum
    if not len(indices):
        raise ValueError('No confidently visible nonelastic source rows')
    appearance = BoundedColorResidual(original_dc, indices, args.maximum_delta).cuda()
    optimizer = torch.optim.Adam([appearance.delta], lr=args.lr)
    trust_views = [1, 2, 3, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    trust_target = torch.from_numpy(view_reference).cuda()
    cameras = {i: make_camera(np.asarray(view_meta['views'][i]['camera_pose']), 640, 640,
                            np.deg2rad(cache['camera']['fovy_degrees'])) for i in trust_views}
    probe_intervals = np.unique(np.r_[train[np.linspace(0, len(train)-1, 12).astype(int)], hard_intervals])
    print(json.dumps({'stage': 'confidence_selection', 'optimized_rows': len(indices),
        'training_intervals': len(train), 'rgb_heldout_frames': len(held_rgb),
        'event_heldout_intervals': len(held_events)}), flush=True)

    def apply():
        renderer.model._features_dc = appearance()

    def loss(index):
        renderer.camera = original_camera
        apply()
        first, second = renderer.image(index, correction), renderer.image(index+1, correction)
        target_a, target_b = reference[index].float()/255, reference[index+1].float()/255
        a, b = multiscale_detail_terms(first, target_a, weight), multiscale_detail_terms(second, target_b, weight)
        rgb, gradient, laplacian = [(x+y)*.5 for x, y in zip(a, b)]
        local = .5*(local_detail_loss(first, target_a)+local_detail_loss(second, target_b))
        log_change = second.clamp_min(floor).log()-first.clamp_min(floor).log()
        pooled = {factor: F.avg_pool2d(log_change[None, None], factor)[0, 0] for factor in (4, 8)}
        event = sum(normalized_event_loss(pooled[factor], events[factor][index]) for factor in (4, 8))*.5
        guard = temporal_nonregression_loss(pooled[8], events[8][index], parent_events[index])
        total = (8*rgb+6*gradient+.6*laplacian+3*local+args.event_weight*event
                 + args.temporal_guard_weight*guard+.01*appearance.regularization())
        return total, (rgb, gradient, laplacian, local, event, guard)

    def trust_loss(view):
        renderer.camera = cameras[view]
        apply()
        current = renderer.image(view_meta['views'][view]['frame'], correction)
        # No smoothing target or synthetic hidden observation is introduced.
        return (current-trust_target[view]).abs().mean()

    @torch.no_grad()
    def evaluate_training_probe():
        values = []
        for index in probe_intervals:
            total, components = loss(int(index))
            values.append([total.item()]+[item.item() for item in components])
        trust = np.mean([trust_loss(view).item() for view in trust_views])
        row = dict(zip(('loss', 'rgb_l1', 'gradient_l1', 'laplacian_l1', 'local_loss', 'event_loss', 'temporal_guard'),
                       np.mean(values, axis=0).tolist()))
        row['novel_view_trust_l1'] = float(trust)
        row['selection_score'] = row['loss']+args.trust_weight*float(trust)
        return row

    initial = evaluate_training_probe()
    best_score, best = initial['selection_score'], appearance.delta.detach().clone()
    best_iteration, history = 0, [{'iteration': 0, **initial}]
    print(json.dumps(history[-1]), flush=True)
    started = time.monotonic()
    shuffled = rng.permutation(train).tolist()
    difficult = rng.permutation(hard_intervals).tolist()
    for iteration in range(1, args.iterations+1):
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]['lr'] = args.lr*(.15+.85*.5*(1+math.cos(math.pi*iteration/args.iterations)))
        if iteration % 4 == 0 and len(hard_intervals):
            if not difficult:
                difficult = rng.permutation(hard_intervals).tolist()
            index = difficult.pop()
        else:
            if not shuffled:
                shuffled = rng.permutation(train).tolist()
            index = shuffled.pop()
        total, _ = loss(int(index))
        if not torch.isfinite(total):
            raise ValueError('Nonfinite measured fitting loss')
        total.backward()
        if iteration % 4 == 0 and args.trust_weight:
            trust = trust_loss(trust_views[(iteration//4-1) % len(trust_views)])
            (4*args.trust_weight*trust).backward()
        torch.nn.utils.clip_grad_norm_([appearance.delta], 1.)
        optimizer.step(); appearance.project_()
        if iteration % 60 == 0 or iteration == args.iterations:
            row = evaluate_training_probe()
            history.append({'iteration': iteration, 'seconds': time.monotonic()-started, **row})
            if row['selection_score'] < best_score:
                best_score, best_iteration = row['selection_score'], iteration
                best = appearance.delta.detach().clone()
            print(json.dumps(history[-1]), flush=True)
    with torch.no_grad():
        appearance.delta.copy_(best)
        apply()
        for key, value in protected.items():
            if not torch.equal(value, getattr(renderer.model, key)):
                raise ValueError('Color fitting changed protected Gaussian '+key)
        delta = appearance.delta.cpu().numpy().astype(np.float32)
        rows = indices.cpu().numpy().astype(np.int32)
        np.savez_compressed(args.output/'residual.npz', indices=rows, delta_gray=delta,
                            maximum_training_visibility=visibility)
        report = {'format': 'lego_observed_color_residual_fit_v1',
            'parent_baseline': str(args.baseline), 'parent_candidate': str(args.candidate),
            'source': cache['source'], 'gaussian_count': cache['gaussian_count'],
            'optimized_count': len(rows), 'prefix_count': completed['prefix_count'],
            'iterations': args.iterations, 'selected_iteration': best_iteration, 'history': history,
            'lr': args.lr, 'weights': {'rgb': 8., 'gradient': 6., 'laplacian': .6, 'local': 3.,
                'event': args.event_weight, 'novel_view_trust': args.trust_weight,
                'temporal_nonregression': args.temporal_guard_weight, 'residual_l2': .01},
            'hard_training_intervals': hard_intervals.tolist(),
            'sampling': 'Shuffled full-training cycles; every fourth step replays a parent-audit hard TRAIN interval.',
            'maximum_delta': args.maximum_delta, 'visibility_threshold': args.visibility_threshold,
            'visibility_probe_frames': probe_frames.tolist(), 'training_intervals': train.tolist(),
            'training_probe_intervals': probe_intervals.tolist(), 'rgb_heldout_frames': held_rgb.tolist(),
            'event_heldout_intervals': held_events.tolist(), 'trust_view_indices': trust_views,
            'detail_rois': DETAIL_ROIS, 'input_paths': {k: str(v) for k, v in input_paths.items()},
            'input_sha256': hashes, 'residual_sha256': sha256_file(args.output/'residual.npz'),
            'positions_rotations_scales_opacity_motion_frozen': True,
            'elastic_and_compact_completion_rows_frozen': True,
            'source_chroma_preserved': True, 'heldout_data_not_used_for_fitting_or_selection': True,
            'scope': 'Bounded source Gaussian color refinement, not image enhancement or geometry completion. '
                     'Temporal heldout frames are withheld from this residual stage only; the inherited parent '
                     'may have seen them. They share the original camera. Novel-view trust images are previous '
                     'Gaussian renders, not back-side truth. Full event and visual acceptance remains separate.'}
        write_new(args.output/'optimization.json', report)
        renderer.camera = original_camera
        rendered = np.lib.format.open_memmap(args.render_output, mode='w+', dtype=np.float16,
                                            shape=(400, 640, 640))
        for frame in range(400):
            actual = renderer.image(frame, correction).cpu().numpy()
            if not np.isfinite(actual).all():
                raise ValueError('Nonfinite candidate image')
            rendered[frame] = actual
            if frame in (0, 50, 100, 150, 200, 300, 399):
                Image.fromarray(np.rint(actual*255).astype(np.uint8)).save(args.output/f'frame_{frame:04d}.png')
            if frame % 80 == 0:
                print(json.dumps({'stage': 'render_full_timeline', 'frame': frame}), flush=True)
        rendered.flush()
        parent = np.load(args.candidate/'rendered_gray.npy', mmap_mode='r', allow_pickle=False)
        detail = {'split': '80 timestamps withheld from this residual stage only, same camera; no checkpoint selection. Parent-stage RGB independence is not claimed.',
                  'rois': {}, 'full_image': {}}
        for name, (x0, y0, x1, y1) in DETAIL_ROIS.items():
            detail['rois'][name] = {label: summarize([measure_detail(images[i, y0:y1, x0:x1],
                value[i, y0:y1, x0:x1]) for i in held_rgb]) for label, value in [('parent', parent), ('candidate', rendered)]}
        for label, value in [('parent', parent), ('candidate', rendered)]:
            detail['full_image'][label] = float(np.mean([np.abs(value[i].astype(np.float32)-images[i]/255).mean() for i in held_rgb]))
        write_new(args.output/'heldout-detail.json', detail)
        # Preserve native multiview RGB and verify exact alpha invariance.
        from gaussian_renderer import render
        novel = args.output/'multiview'; novel.mkdir()
        gray = np.empty_like(view_reference); alpha_errors = []
        with np.load(args.multiview, allow_pickle=False) as archive:
            parent_alpha = archive['alpha']
        for vi, view in enumerate(view_meta['views']):
            renderer.camera = make_camera(np.asarray(view['camera_pose']), 640, 640,
                                          np.deg2rad(cache['camera']['fovy_degrees']))
            gray[vi] = renderer.image(view['frame'], correction).cpu().numpy()
            result = render(renderer.camera, renderer.model, renderer.pipe, renderer.background, 0., 0., 0.)
            actual_alpha = result['alpha'].reshape(640, 640).cpu().numpy()
            alpha_errors.append(float(np.abs(actual_alpha-parent_alpha[vi]).max()))
            rgb = result['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(novel/f'view-{vi:02d}.png')
        if max(alpha_errors) > 1e-6:
            raise ValueError('Color-only fitting changed alpha coverage')
        np.savez_compressed(novel/'gray.npz', gray=gray)
        write_new(novel/'check.json', {'views': view_meta['views'], 'alpha_max_absolute_error': max(alpha_errors),
            'alpha_unchanged': True, 'rgb_mean_absolute_changes': np.abs(gray-view_reference).mean((1, 2)).tolist(),
            'diagnostic_only': True, 'unobserved_reference_is_parent_not_ground_truth': True})
    from scripts.events.audit_lego_event_motion import audit
    assessment = audit(rendered, args.supervision)
    assessment['validation_note'] = 'Every fifth RGB timestamp and existing validation-event interval are excluded from residual fitting and model selection. Same camera, not independent viewpoints.'
    write_new(args.output/'residual-event-audit.json', assessment)
    for key, path in input_paths.items():
        if sha256_file(path) != hashes[key]:
            raise ValueError('Input changed during fitting: '+key)
    write_new(args.output/'completion.json', {'source_inputs_rechecked': True,
        'rendered_gray': str(args.render_output.resolve()), 'rendered_gray_sha256': sha256_file(args.render_output),
        'selected_iteration': best_iteration, 'parent_event_summary': parent_audit['summary'],
        'candidate_event_summary': assessment['summary'], 'event_approved': assessment['approved_for_export'],
        'heldout_global_mae': detail['full_image'], 'alpha_unchanged': True, 'deployed': False})
    print(json.dumps({'stage': 'completed_diagnostic', 'rgb_mae': detail['full_image'],
        'event_summary': assessment['summary'], 'approved': assessment['approved_for_export']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
