"""Optimize only a bounded auxiliary MLP on the exact existing Blender/V3.

Training reads event counts, never RGB targets. Inherited V3 appearance,
motion, event calibration and train/validation split are kept unchanged.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from scripts.events.export_lego_event_preview import verify_candidate
from scripts.events.audit_lego_event_motion import audit
from utils.rigid_checkpoint import sha256_file
from utils.v3_auxiliary_mlp import V3AuxiliaryMLP, save_checkpoint, load_checkpoint, event_objective
from utils.v3_auxiliary_renderer import FrozenV3Renderer


def main(field_kind='v3-auxiliary'):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('baseline', 'candidate', 'supervision', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--probe-every', type=int, default=40)
    parser.add_argument('--learning-rate', type=float, default=.0001)
    parser.add_argument('--optimizer', choices=('adam', 'guarded-sgd'), default='adam',
                        help='guarded-sgd uses all training events and accepts only measured improvements')
    parser.add_argument('--translation-fraction', type=float, default=.0005 if field_kind == 'scgs-v4' else .005)
    parser.add_argument('--rotation-radians', type=float, default=.005 if field_kind == 'scgs-v4' else .025)
    parser.add_argument('--scale-fraction', type=float, default=.02,
                        help='V4 only: bounded local Gaussian shape deformation')
    parser.add_argument('--shape-warmup-steps', type=int, default=0,
                        help='V4 guarded optimizer: initially keep the V3 positions/rotations exact')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.probe_every) < 1 or args.shape_warmup_steps < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('Positive steps and finite learning rate required')
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    torch.manual_seed(20261022)
    np.random.seed(20261022)
    print('Verifying the complete current V3 and its existing supervision...', flush=True)
    cache, record, baseline_audit = verify_candidate(args.baseline, args.candidate, allow_unapproved=True)
    if sha256_file(args.supervision) != baseline_audit['artifact_sha256']['supervision']:
        raise ValueError('Supervision differs from the current V3 audit')
    # Explicitly do not load archive[images] here. RGB is evaluation-only in audit().
    with np.load(args.supervision, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json']))
        counts = archive['counts']
    if counts.shape != (cache['frame_count']-1, 2, 640, 640):
        raise ValueError('Expected the complete native-resolution V3 event timeline')
    if metadata['calibration'] != baseline_audit['calibration']:
        raise ValueError('Calibration must stay identical to V3')
    floor = float(metadata['calibration']['log_floor'])
    positive, negative = metadata['calibration']['thresholds']
    targets = np.empty((len(counts), 80, 80), np.float32)
    for i, count in enumerate(counts):
        signed = count[0].astype(np.float32)*positive-count[1].astype(np.float32)*negative
        targets[i] = signed.reshape(80, 8, 80, 8).mean((1, 3))
    del counts
    targets = torch.from_numpy(targets).cuda()
    train = np.asarray([i for i in range(len(targets)) if i % 5 != 4], np.int64)
    validation = [i for i in range(len(targets)) if i % 5 == 4]
    # Use every training interval for selection, so rare difficult windows
    # cannot be missed by a small probe. No validation data enter selection.
    probe = train.copy()
    renderer = FrozenV3Renderer(args.baseline, record)
    save_field, load_field = save_checkpoint, load_checkpoint
    if field_kind == 'scgs-v4':
        from utils.scgs_v4_field import SCGSV4Field, make_scgs_graph
        from utils.scgs_v4_field import save_checkpoint as save_field, load_checkpoint as load_field
        graph = make_scgs_graph(renderer)
        model = SCGSV4Field(graph, cache['frame_count'], args.translation_fraction,
                           args.rotation_radians, args.scale_fraction).cuda()
    else:
        graph = renderer.graph()
        model = V3AuxiliaryMLP(graph, cache['frame_count'], args.translation_fraction, args.rotation_radians).cuda()
    baseline_frames = np.load(args.candidate/'rendered_gray.npy', mmap_mode='r')
    frozen_state = {k: v.detach().clone() for k, v in renderer.correction.state_dict().items()}
    identity_checks = []
    with torch.no_grad():
        for frame in (0, 99, 199, 299, 399):
            base = renderer.gray(frame)
            zero = renderer.gray(frame, model)
            if not torch.equal(base, zero):
                raise ValueError('Zero auxiliary must render exactly the original V3')
            stored = torch.from_numpy(np.asarray(baseline_frames[frame], np.float32)).cuda()
            error = float((base-stored).abs().max())
            if error > .0005:
                raise ValueError(f'Original V3 baseline render mismatch at {frame}: {error}')
            identity_checks.append(dict(frame=frame, zero_mlp_exact=True, saved_v3_max_abs_error=error))
    args.output.mkdir(parents=True)
    sources = args.output/'source-snapshot'
    sources.mkdir()
    source_paths = ['utils/v3_auxiliary_mlp.py', 'utils/v3_auxiliary_renderer.py',
                 'scripts/deformation/train_v3_auxiliary_mlp.py', 'utils/event_mlp.py',
                 'scripts/events/refine_lego_events.py', 'scripts/events/audit_lego_event_motion.py']
    if field_kind == 'scgs-v4':
        source_paths += ['utils/scgs_v4_field.py', 'scripts/deformation/train_scgs_v4.py', 'utils/time_utils.py']
    for path in source_paths:
        dest = sources/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve().parents[2]/path, dest)
    provenance = dict(field_kind=field_kind, baseline=str(args.baseline), candidate=str(args.candidate),
                      supervision=str(args.supervision), baseline_artifact_sha256=baseline_audit['artifact_sha256'],
                      training_supervision='events only; existing V3 and its calibration are frozen inherited priors',
                      validation_note='79 intervals excluded from auxiliary gradients and selection; V3 and inherited calibration may have prior exposure to video/events.',
                      train_intervals=train.tolist(), validation_intervals=validation,
                      selection_probe_intervals=probe.tolist(), calibration=metadata['calibration'],
                      identity_checks=identity_checks, batch_size=args.batch_size,
                      learning_rate=args.learning_rate, optimizer=args.optimizer,
                      shape_warmup_steps=args.shape_warmup_steps)
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    parameter_groups = model.parameter_groups(args.learning_rate) if hasattr(model, 'parameter_groups') else model.mlp.parameters()
    optimizer = torch.optim.Adam(parameter_groups, lr=args.learning_rate)

    def event_loss(index):
        first, second = renderer.gray(index, model), renderer.gray(index+1, model)
        return event_objective(first, second, targets[index], floor)

    @torch.no_grad()
    def probe_loss():
        return float(np.mean([float(event_loss(int(i))) for i in probe]))

    best = probe_loss()
    initial = best
    best_step = 0
    best_state = copy.deepcopy(model.mlp.state_dict())
    rng = np.random.default_rng(20261022)
    order = rng.permutation(train).tolist()
    log = [dict(step=0, train_probe=best)]
    print(json.dumps(dict(stage='training', baseline_probe=best, active_rows=len(model.rows),
                          control_nodes=len(model.centers), parameters=sum(p.numel() for p in model.parameters()),
                          scene_scale=float(model.scene_scale), translation_bound=float(model.scene_scale)*args.translation_fraction)), flush=True)
    training_start = time.monotonic()
    for step in range(1, args.steps+1):
        if args.optimizer == 'guarded-sgd':
            optimizer.zero_grad(set_to_none=True)
            for index in train:
                index = int(index)
                magnitude, smooth, acceleration = model.regularization(index+.5)
                loss = event_loss(index)+.002*magnitude+.02*smooth+10*acceleration
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite auxiliary objective')
                (loss/len(train)).backward()
            parameters = list(model.mlp.parameters())
            previous = [p.detach().clone() for p in parameters]
            rates = {id(p): (0. if step <= args.shape_warmup_steps and group.get('lr_factor', 1.) < 1
                              else group.get('lr_factor', 1.))
                     for group in optimizer.param_groups for p in group['params']}
            gradients = [p.grad.detach().clone()*rates[id(p)] for p in parameters]
            scale = max(float(g.abs().max()) for g in gradients)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError('Invalid auxiliary descent direction')
            trials = []
            accepted = False
            for multiplier in (1., .3, .1, .03, .01, .003, .001, .0003):
                rate = args.learning_rate*multiplier
                with torch.no_grad():
                    for p, old, gradient in zip(parameters, previous, gradients):
                        p.copy_(old-gradient*(rate/scale))
                value = probe_loss()
                trials.append(dict(max_parameter_step=rate, train_probe=value))
                print(json.dumps(dict(stage='backtracking', step=step, **trials[-1])), flush=True)
                if value < best-1e-7:
                    best, best_step = value, step
                    best_state = copy.deepcopy(model.mlp.state_dict())
                    accepted = True
                    save_field(args.output/'accepted-progress.pth', model, {**provenance, 'best_step': best_step})
                    break
            if not accepted:
                model.mlp.load_state_dict(best_state)
            entry = dict(step=step, train_probe=best, best_probe=best, best_step=best_step,
                         accepted=accepted, trials=trials, elapsed_seconds=round(time.monotonic()-training_start, 1))
            log.append(entry)
            (args.output/'training-log.json').write_text(json.dumps(log, indent=2)+'\n')
            print(json.dumps(entry), flush=True)
            if not accepted:
                break
            continue
        optimizer.zero_grad(set_to_none=True)
        data_values = []
        for _ in range(args.batch_size):
            if not order:
                order = rng.permutation(train).tolist()
            index = int(order.pop())
            data_loss = event_loss(index)
            magnitude, smooth, acceleration = model.regularization(index+.5)
            loss = data_loss+.002*magnitude+.02*smooth+10*acceleration
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite auxiliary objective')
            (loss/args.batch_size).backward()
            data_values.append(float(data_loss.detach()))
        if any(p.grad is not None for p in renderer.correction.parameters()):
            raise RuntimeError('V3 must remain frozen')
        torch.nn.utils.clip_grad_norm_(model.mlp.parameters(), 1.)
        optimizer.step()
        progress = step/args.steps
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate*group.get('lr_factor', 1.)*(.1+.9*(1+math.cos(math.pi*progress))/2)
        if step % args.probe_every == 0 or step == args.steps:
            value = probe_loss()
            if value < best:
                best, best_step = value, step
                best_state = copy.deepcopy(model.mlp.state_dict())
            entry = dict(step=step, event_objective=float(np.mean(data_values)), train_probe=value,
                         best_probe=best, best_step=best_step, elapsed_seconds=round(time.monotonic()-training_start, 1))
            log.append(entry)
            (args.output/'training-log.json').write_text(json.dumps(log, indent=2)+'\n')
            print(json.dumps(entry), flush=True)
    model.mlp.load_state_dict(best_state)
    for key, value in renderer.correction.state_dict().items():
        if not torch.equal(value, frozen_state[key]):
            raise RuntimeError('The V3 correction was changed')
    provenance.update(best_step=best_step, initial_train_probe=initial, best_train_probe=best)
    save_field(args.output/'auxiliary-mlp.pth', model, provenance)
    # Evaluate the actual reloaded checkpoint, not just the in-memory network.
    del model
    model, loaded_provenance = load_field(args.output/'auxiliary-mlp.pth')
    if loaded_provenance != provenance:
        raise RuntimeError('Checkpoint provenance mismatch')
    rendered = np.lib.format.open_memmap(args.output/'rendered_gray.npy', mode='w+', dtype=np.float16,
                                         shape=baseline_frames.shape)
    max_displacement, static_error, max_scale_change = 0., 0., 0.
    with torch.no_grad():
        for frame in range(cache['frame_count']):
            rendered[frame] = renderer.gray(frame, model).cpu().numpy()
            if frame % 50 == 0 or frame == cache['frame_count']-1:
                xyz, rotation, group_q = renderer.base_pose(frame)
                moved, oriented = model.deform(xyz, rotation, group_q, frame)
                mask = torch.ones(len(xyz), dtype=torch.bool, device='cuda')
                mask[model.rows] = False
                error = max(float((moved[mask]-xyz[mask]).abs().max()), float((oriented[mask]-rotation[mask]).abs().max()))
                static_error = max(static_error, error)
                max_displacement = max(max_displacement, float((moved-xyz).norm(dim=-1).max()))
                if hasattr(model, 'deform_scaling'):
                    scale = model.deform_scaling(renderer.base_scaling, frame)
                    scale_error = float((scale[mask]-renderer.base_scaling[mask]).abs().max())
                    static_error = max(static_error, scale_error)
                    max_scale_change = max(max_scale_change, float(((scale-renderer.base_scaling).exp()-1).abs().max()))
                    if scale_error != 0 or (frame == 0 and not torch.equal(scale, renderer.base_scaling)):
                        raise RuntimeError('Protected scales or t=0 changed')
                if error != 0 or (frame == 0 and (not torch.equal(moved, xyz) or not torch.equal(oriented, rotation))):
                    raise RuntimeError('Protected rows or t=0 changed')
                print(json.dumps(dict(stage='evaluation_render', frame=frame)), flush=True)
    rendered.flush()
    print('Auditing 399 intervals with the unchanged V3 metric operator...', flush=True)
    result = audit(rendered, args.supervision)
    result['validation_note'] = provenance['validation_note']
    result['training_supervision'] = provenance['training_supervision']
    (args.output/'event-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output/'baseline-event-audit.json').write_text(json.dumps(baseline_audit, indent=2)+'\n')
    summary = dict(field_kind=field_kind, baseline=baseline_audit['summary'], auxiliary=result['summary'],
                   best_step=best_step, total_steps=args.steps, completed_steps=step,
                   batch_size=len(train) if args.optimizer == 'guarded-sgd' else args.batch_size,
                   optimizer=args.optimizer,
                   frozen_v3_motion=True, frozen_v3_appearance=True, static_pose_max_error=static_error,
                   sampled_max_residual_translation=max_displacement,
                   sampled_max_relative_scale_change=max_scale_change,
                   translation_bound=float(model.scene_scale)*args.translation_fraction,
                   rotation_bound_radians=args.rotation_radians,
                   identity_checks=identity_checks, baseline_approved=baseline_audit['approved_for_export'],
                   auxiliary_approved=result['approved_for_export'],
                   elapsed_seconds=time.monotonic()-started,
                   peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/1024**2,
                   training_supervision=provenance['training_supervision'], validation_note=provenance['validation_note'])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
