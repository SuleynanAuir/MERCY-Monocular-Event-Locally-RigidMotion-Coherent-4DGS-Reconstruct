"""Smooth time-dependent rigid event alignment, with frozen Gaussian appearance.

Uses only training event intervals. Validation event targets are never used for
gradients, hard sampling, or checkpoint selection. Audit policy is unchanged.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
import torch
import torch.nn.functional as F

from scripts.refine_lego_events import MotionRenderer
from scripts.audit_lego_event_motion import audit
from utils.event_refinement import TemporalRigidMotionCorrection, ArticulatedEventCorrection, correction_from_record
from utils.rigid_checkpoint import sha256_file
from utils.render_sampling import require_native_sampling


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--supervision', type=Path, required=True)
    p.add_argument('--initial-correction', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=800)
    p.add_argument('--knots', type=int, default=17)
    p.add_argument('--lr', type=float, default=.02)
    p.add_argument('--foreground-weight', type=float, default=8.)
    p.add_argument('--parts', type=Path, help='Enable contact-regularized articulated residuals')
    p.add_argument('--binding', type=Path)
    p.add_argument('--joint-report', type=Path, help='Use hard joint FK, never the old shared conjugation')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    require_native_sampling(json.loads((args.baseline / 'cache.json').read_text()),
                            'Legacy temporal/articulated motion training')
    if args.iterations < 1 or not np.isfinite(args.lr) or not 0 < args.lr <= .1:
        raise ValueError('Invalid optimizer configuration')
    if bool(args.parts) != bool(args.binding) or not np.isfinite(args.foreground_weight) or args.foreground_weight < 0:
        raise ValueError('Parts and binding must be supplied together; coverage weight must be finite/nonnegative')
    if args.joint_report and (not args.parts or args.initial_correction):
        raise ValueError('Hard-joint training requires parts/binding and starts without a legacy correction')
    if not args.joint_report and not args.initial_correction:
        raise ValueError('Legacy temporal refinement requires --initial-correction')
    with np.load(args.supervision, allow_pickle=False) as data:
        images, counts = data['images'], data['counts']
        meta = json.loads(str(data['metadata_json']))
    train_ids = np.flatnonzero(np.arange(len(counts)) % 5 != 4)
    active = binary_dilation(np.any(counts[train_ids] > 0, axis=(0, 1)), iterations=3)
    positive, negative = meta['calibration']['thresholds']
    targets = {}
    for size in (40, 80, 160):
        pooled = np.empty((len(counts), size, size), np.float32)
        factor = 640 // size
        for i, count in enumerate(counts):
            signed = count[0].astype(np.float32)*positive - count[1].astype(np.float32)*negative
            pooled[i] = signed.reshape(size, factor, size, factor).mean((1, 3))
        targets[size] = torch.from_numpy(pooled).cuda()
    del counts
    renderer = MotionRenderer(args.baseline, 640)
    if len(renderer.poses) != len(images) or images.shape[1:] != (640, 640):
        raise ValueError('Baseline and supervision timelines/resolutions disagree')
    rgb = torch.from_numpy(images.astype(np.float32) / 255).cuda()
    region = torch.from_numpy(active).cuda()
    floor = meta['calibration']['log_floor']
    structure = None
    if args.joint_report:
        from scripts.lego_joint_structure import build_structure
        from utils.joint_refinement import HardJointEventCorrection
        structure = build_structure(args.parts, args.binding, args.baseline, args.joint_report)
        correction = HardJointEventCorrection(structure).cuda()
    elif args.parts:
        from scripts.lego_motion_groups import group_structure
        structure = group_structure(args.parts, args.binding, args.baseline)
        correction = ArticulatedEventCorrection(len(images), args.knots, **structure).cuda()
    else:
        correction = TemporalRigidMotionCorrection(len(images), args.knots).cuda()
    if args.initial_correction:
        initial_record = json.loads(args.initial_correction.read_text())
        initial_model = correction_from_record(initial_record, 'cuda')
        with torch.no_grad():
            correction.raw.copy_(initial_model.raw)
            if isinstance(initial_model, TemporalRigidMotionCorrection):
                if correction.offsets.shape != initial_model.offsets.shape:
                    raise ValueError('Continuation requires identical knot count')
                correction.offsets.copy_(initial_model.offsets)
            if isinstance(initial_model, ArticulatedEventCorrection):
                if not isinstance(correction, ArticulatedEventCorrection) or not torch.equal(correction.groups, initial_model.groups):
                    raise ValueError('Articulated continuation requires identical groups')
                correction.group_raw.copy_(initial_model.group_raw)
        del initial_model
    mass = targets[80].abs().sum((1, 2)).cpu().numpy()
    train_ids = train_ids[mass[train_ids] > 1]
    if len(train_ids) < 32:
        raise ValueError('Insufficient observable training intervals')
    probe = train_ids[np.linspace(0, len(train_ids)-1, 32).astype(int)]
    if args.joint_report:
        optimizer = torch.optim.Adam(correction.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam([{'params': [correction.raw], 'lr': args.lr * .25},
                                      {'params': [correction.offsets], 'lr': args.lr}])
        if structure:
            optimizer.add_param_group({'params': [correction.group_raw], 'lr': args.lr})
    rng = np.random.default_rng(11)
    args.output.mkdir(parents=True)

    def loss(index):
        first, second = renderer.image(index, correction), renderer.image(index+1, correction)
        delta = second.clamp_min(floor).log() - first.clamp_min(floor).log()
        event_loss = delta.new_zeros(())
        for size, weight in ((40, .15), (80, 1.), (160, .2)):
            predicted = F.avg_pool2d(delta[None, None], 640 // size)[0, 0]
            expected = targets[size][index]
            l1 = (predicted-expected).abs().sum() / expected.abs().sum().clamp_min(1.)
            cosine = 1 - (predicted*expected).sum() / (predicted.norm()*expected.norm()).clamp_min(1e-6)
            event_loss = event_loss + weight * (l1 + .5*cosine)
        expected = targets[80][index]
        predicted = F.avg_pool2d(delta[None, None], 8)[0, 0]
        strong = expected.abs() >= .01
        polarity = (F.relu(.01 - predicted*expected.sign()) * expected.abs() * strong).sum() / (expected.abs().sum().clamp_min(1.) * .01)
        foreground = (rgb[index+1] > .08) & region
        coverage = (F.relu(.08-second)*foreground).sum() / (.08*foreground.sum().clamp_min(1))
        image = ((second-rgb[index+1]).abs()*region).sum() / region.sum().clamp_min(1)
        penalty = correction.regularization()
        if not args.joint_report:
            penalty = penalty + .001*correction.raw.square().mean()
        total = event_loss + .05*polarity + 2*image + args.foreground_weight*coverage + penalty
        return total

    @torch.no_grad()
    def probe_loss():
        return float(np.mean([loss(int(i)).item() for i in probe]))

    best_score = initial_score = probe_loss()
    best = {k: v.detach().clone() for k, v in correction.state_dict().items()}
    history = [{'iteration': 0, 'training_probe_loss': best_score}]
    print(json.dumps(history[-1]), flush=True)
    # Two time-separated event pairs reduce noisy single-pair updates. A
    # coverage/polarity term prevents a high cosine obtained by shrinking motion.
    for iteration in range(1, args.iterations+1):
        optimizer.zero_grad(set_to_none=True)
        selected = rng.choice(train_ids, size=2, replace=False)
        score = 0.
        for index in selected:
            total = loss(int(index)) / len(selected)
            if not torch.isfinite(total):
                raise ValueError('Non-finite training loss')
            total.backward(); score += total.item()
        torch.nn.utils.clip_grad_norm_(correction.parameters(), 5.)
        optimizer.step()
        if iteration % 100 == 0 or iteration == args.iterations:
            tested = probe_loss()
            if tested < best_score:
                best_score, best = tested, {k: v.detach().clone() for k,v in correction.state_dict().items()}
            row = {'iteration': iteration, 'training_probe_loss': tested, 'batch_loss': score}
            history.append(row); print(json.dumps(row), flush=True)
    correction.load_state_dict(best)
    result = {'format': 'event_supervised_temporal_rigid_alignment_v2', 'frame_count': len(images),
              'initial_probe_loss': initial_score, 'best_probe_loss': best_score,
              'iterations': args.iterations, 'history': history, 'calibration': meta['calibration'],
              'training_render_resolution': 640, 'foreground_weight': args.foreground_weight,
              'initial_correction': str(args.initial_correction) if args.initial_correction else None,
              'optimizer_lr': args.lr, 'rgb_appearance_parameters_frozen': True,
              'event_validation_intervals_not_trained': True, 'probe_intervals': probe.tolist(),
              'scope': 'Shared smooth conjugation; per-part distances preserved. Monocular hidden geometry and exact articulation attachment are not guaranteed.'}
    if args.joint_report:
        result.update(format='event_supervised_hard_joint_alignment_v4', joint_structure=structure,
                      group_angles=correction.group_angles.detach().cpu().tolist(), max_angle_degrees=2.,
                      scope='Whole rigid-part FK; fixed root, coincident tree joints and locked closed-loop residuals. Hidden surface shape is not established by monocular evidence.',
                      group_input_sha256={'parts': sha256_file(args.parts), 'binding': sha256_file(args.binding),
                                          'joint_report': sha256_file(args.joint_report)})
    else:
        result.update(knot_count=args.knots, raw=correction.raw.detach().cpu().tolist(),
                      temporal_offsets=correction.offsets.detach().cpu().tolist())
    if structure and not args.joint_report:
        result.update(format='event_supervised_articulated_alignment_v3', group_structure=structure,
                      group_raw=correction.group_raw.detach().cpu().tolist(),
                      scope='Bounded per-motion-group rigid pose refinement with smooth global alignment and prior-contact protection; no exact-joint/unique-depth claim.',
                      group_input_sha256={'parts': sha256_file(args.parts), 'binding': sha256_file(args.binding)})
    (args.output / 'correction.json').write_text(json.dumps(result, indent=2)+'\n')
    with torch.no_grad():
        rendered = np.lib.format.open_memmap(args.output / 'rendered_gray.npy', mode='w+', dtype=np.float16,
                                            shape=(len(images), 640, 640))
        for i in range(len(images)):
            image = renderer.image(i, correction).cpu().numpy()
            rendered[i] = image
            if i in (0, 50, 100, 200, 300, 399):
                Image.fromarray((image*255).astype(np.uint8)).save(args.output / f'frame_{i:04d}.png')
            if i % 100 == 0:
                print('Render temporal', i, flush=True)
        rendered.flush()
    validation = audit(rendered, args.supervision)
    validation['event_optimization_performed'] = True
    validation['artifact_sha256'] = {key: sha256_file(path) for key, path in {
        'correction': args.output / 'correction.json', 'rendered_gray': args.output / 'rendered_gray.npy',
        'supervision': args.supervision, 'baseline_cache_metadata': args.baseline / 'cache.json',
        'baseline_poses': args.baseline / 'poses.npy', 'dynamic_indices': args.baseline / 'dynamic_indices.npy',
        'source_ply': renderer.meta['source']}.items()}
    validation['correction'] = {key: value for key, value in result.items() if key != 'history'}
    (args.output / 'event-audit.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps({'summary': validation['summary'], 'failure_reason_counts': validation['failure_reason_counts'],
                      'approved_for_export': validation['approved_for_export']}, indent=2), flush=True)
    if not validation['approved_for_export']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
