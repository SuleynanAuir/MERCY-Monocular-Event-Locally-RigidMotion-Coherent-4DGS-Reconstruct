"""Fit a bounded residual MLP using raw events and an event-only rigid baseline.

All appearance and baseline parameters are frozen. A reserved window is not an
independent holdout if the baseline already used it; that is recorded explicitly.
No RGB, video, Blender geometry, old viewer motion, or appearance fitting is read.
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
from PIL import Image

from utils.event_only_checkpoint import load_event_only_checkpoint
from utils.event_only_inputs import sha256_file
from utils.event_only_renderer import EventOnlyRenderer
from utils.event_only_supervision import build_event_windows, audit_event_prediction
from utils.event_mlp import build_residual_graph, EventResidualMLP, save_mlp_checkpoint
from scripts.events.reconstruct_lego_event_only import difference_loss, observation_log, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'g0', 'events', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--nodes-per-group', type=int, default=32)
    p.add_argument('--translation-fraction', type=float, default=.01)
    p.add_argument('--rotation-radians', type=float, default=.08)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--magnitude-weight', type=float, default=.01)
    p.add_argument('--rigidity-weight', type=float, default=200.)
    p.add_argument('--acceleration-weight', type=float, default=.0001)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.steps < 1 or args.seed < 0 or not 0 < args.learning_rate < .1 or
            not all(math.isfinite(v) and v >= 0 for v in
                    (args.magnitude_weight, args.rigidity_weight, args.acceleration_weight))):
        p.error('Invalid optimizer settings')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    base, camera, payload = load_event_only_checkpoint(args.base, args.g0, device='cpu')
    provenance, base_summary = payload['input_provenance'], payload['training_summary']
    if sha256_file(args.events) != payload['input_sha256']['events']:
        raise ValueError('Raw events do not match the event-only baseline')
    # Rebuild the same raw windows, never inherit cached targets or RGB proxies.
    windows = build_event_windows(args.events, camera['width'], camera['height'],
                                  g0_time=provenance['g0_time'], time_scale=provenance['time_scale'],
                                  step_seconds=.05, substeps=2)
    times, counts, train = windows['times'], windows['counts'], windows['train_mask']
    if abs(float(base.times[-1]) - float(times[-1])) > 1e-8:
        raise ValueError('Baseline and raw events have different time extents')
    if not train.any() or train.all():
        raise ValueError('Need training and reserved event intervals')
    positive, negative = base_summary['thresholds']
    background = float(base_summary.get('event_background', 0.))
    floor = float(base_summary.get('log_intensity_floor', .01))
    renderer = EventOnlyRenderer(args.g0, camera)
    graph = build_residual_graph(renderer.g0.xyz.cpu().numpy(), base, args.nodes_per_group)
    motion = EventResidualMLP(base, graph, translation_fraction=args.translation_fraction,
                              rotation_radians=args.rotation_radians).cuda()
    if not motion.active_nodes.any():
        raise ValueError('The baseline contains no nonstatic motion to refine')
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    expected = counts[:, 0].astype(np.float32) * positive - counts[:, 1].astype(np.float32) * negative
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(args.output / 'settings.json', settings)
    write_json(args.output / 'input-provenance.json', {
        'allowed_modalities': ['g0', 'events', 'camera'], 'base_checkpoint_sha256': sha256_file(args.base),
        'input_sha256': payload['input_sha256'], 'video_supervision_used': False,
        'base_fit_split': base_summary['fit_split'], 'residual_fit_split': 'train',
        'independent_holdout': False,
        'scope': 'Reserved windows were not used by residual gradients; base may already contain those events.'})
    root = Path(__file__).resolve().parents[2]
    write_json(args.output / 'execution-code-sha256.json', {
        str(path.relative_to(root)): sha256_file(path)
        for path in [Path(__file__).resolve(), root/'utils/event_mlp.py',
                     root/'utils/event_only_renderer.py', root/'utils/event_only_motion.py',
                     root/'utils/event_only_supervision.py', root/'scripts/events/reconstruct_lego_event_only.py']})

    def render(model, t):
        xyz, rotation = model.deform(renderer.g0.xyz, renderer.g0.rotation, float(t))
        return renderer.render(xyz, rotation)

    def log_render(model, t):
        return observation_log(render(model, t), background, floor)

    def audit(model, label):
        predicted = np.empty((len(counts), camera['height'], camera['width']), np.float32)
        with torch.no_grad():
            previous = log_render(model, times[0])
            for i, t in enumerate(times[1:]):
                current = log_render(model, t)
                predicted[i] = (current - previous).cpu().numpy()
                previous = current
                if i % 60 == 0:
                    print(json.dumps({'phase': label + '_audit', 'window': i, 'total': len(counts)}), flush=True)
        result = audit_event_prediction(predicted, counts, train_mask=train, fit_split='train',
                                        threshold_positive=positive, threshold_negative=negative,
                                        thresholds_confirmed=base_summary.get('thresholds_confirmed', False))
        # Never relabel a baseline trained on every event as independent validation.
        result.update(approved_for_export=False, status='candidate_unapproved',
                      evaluation_scope='residual_stage_reserved_windows_only_not_independent_holdout')
        result['approval_blockers'].append('baseline_independent_holdout_not_established')
        write_json(args.output / (label + '-event-audit.json'), result)
        del predicted
        return result

    baseline_audit = audit(motion.base, 'baseline')
    # Zero-initial residual must reproduce the base and preserve static rows.
    with torch.no_grad():
        sample = torch.arange(0, renderer.g0.count, 17, device='cuda')
        x, q = renderer.g0.xyz[sample], renderer.g0.rotation[sample]
        bx, bq = motion.base.deform(x, q, float(times[-1]), sample)
        mx, mq = motion.deform(x, q, float(times[-1]), sample)
        initial_error = max(float((bx-mx).abs().max()), float((bq-mq).abs().max()))
        if initial_error > 2e-6:
            raise ValueError('Zero MLP does not reproduce base motion')
    optimizer = torch.optim.Adam(motion.mlp.parameters(), lr=args.learning_rate)
    order = np.flatnonzero(train)
    rng = np.random.default_rng(args.seed)
    history, seen = [], set()
    for iteration in range(args.steps):
        if iteration % len(order) == 0:
            rng.shuffle(order)
        i = int(order[iteration % len(order)])
        optimizer.zero_grad(set_to_none=True)
        first, second = log_render(motion, times[i]), log_render(motion, times[i+1])
        target = torch.as_tensor(expected[i], device='cuda')
        event = difference_loss(second-first, target)
        magnitude, arap, acceleration = motion.regularization(float(times[i+1]), min(.025, float(times[-1])/2))
        loss = event + args.magnitude_weight*magnitude + args.rigidity_weight*arap + args.acceleration_weight*acceleration
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite residual loss')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(motion.mlp.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        seen.add(i)
        if iteration % 25 == 0 or iteration == args.steps-1:
            record = {'step': iteration+1, 'event_loss': float(event.detach()), 'loss': float(loss.detach()),
                      'gradient_norm': float(gradient), 'magnitude': float(magnitude.detach()),
                      'arap': float(arap.detach()), 'acceleration': float(acceleration.detach()),
                      'elapsed_seconds': time.perf_counter()-started}
            history.append(record)
            print(json.dumps(record), flush=True)
            write_json(args.output/'training-progress.json', {'history': history})
        del first, second, loss, event, magnitude, arap, acceleration
    summary = {'format': 'event_residual_mlp_training_v1', 'training_performed': True,
               'steps': args.steps, 'fit_split': 'train', 'base_fit_split': base_summary['fit_split'],
               'independent_holdout': False, 'g0_appearance_frozen': True, 'base_motion_frozen': True,
               'video_supervision_used': False, 'static_gaussians': int(motion.static_rows.sum()),
               'active_control_nodes': int(motion.active_nodes.sum()),
               'mlp_parameters': sum(v.numel() for v in motion.mlp.parameters()),
               'training_windows_seen': sorted(seen), 'initial_base_equivalence_max_error': initial_error,
               'thresholds': [positive, negative], 'event_background': background,
               'log_intensity_floor': floor, 'history': history,
               'settings': settings, 'approved_for_export': False}
    save_mlp_checkpoint(args.output/'deform-mlp.pth', motion, graph, payload, summary)
    candidate_audit = audit(motion, 'mlp')
    # Comparison and rendered examples are outputs, never training targets.
    comparisons = {}
    for split in ('all', 'train_events', 'validation_events'):
        a, b = baseline_audit['summary'][split], candidate_audit['summary'][split]
        comparisons[split] = {'baseline': a, 'mlp': b,
                             'relative_l1_reduction_fraction': (a['relative_l1']-b['relative_l1'])/a['relative_l1']
                             if a['relative_l1'] else None}
    with torch.no_grad():
        max_shift, max_turn, static_error = 0., 0., 0.
        for t in times:
            dt, dr = motion.residual(float(t))
            max_shift = max(max_shift, float(dt.norm(dim=1).max()))
            max_turn = max(max_turn, float(dr.norm(dim=1).max()))
        for fraction in (0., .25, .5, .75, 1.):
            t = float(times[-1])*fraction
            images = [render(model, t)['rgb'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                      for model in (motion.base, motion)]
            Image.fromarray(np.rint(np.concatenate(images, axis=1)*255).astype(np.uint8)).save(
                args.output / f'comparison-{fraction:.2f}.png')
            x, q = renderer.g0.xyz[sample], renderer.g0.rotation[sample]
            mx, mq = motion.deform(x, q, t, sample)
            mask = motion.static_rows[sample]
            if mask.any():
                static_error = max(static_error, float((mx[mask]-x[mask]).abs().max()),
                                   float((mq[mask]-q[mask]).abs().max()))
    summary.update(comparison=comparisons, max_node_residual_translation=max_shift,
                   max_node_residual_rotation_radians=max_turn, sampled_static_pose_max_error=static_error,
                   elapsed_seconds=time.perf_counter()-started,
                   peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2)
    write_json(args.output/'training-summary.json', summary)
    print(json.dumps({'phase': 'finished', 'comparison': comparisons, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
