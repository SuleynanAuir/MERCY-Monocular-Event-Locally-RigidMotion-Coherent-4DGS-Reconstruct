"""Event-supervised, constrained temporal residual search on the frozen V3."""
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
from utils.event_refinement import qmul, qrotate
from utils.rigid_checkpoint import sha256_file
from utils.scgs_v4_field import load_checkpoint
from utils.v3_auxiliary_mlp import event_objective
from utils.v3_auxiliary_renderer import FrozenV3Renderer
from utils.v4_joint_trajectory import JointTrajectoryField, hinge_angles, smooth_proposal, save_checkpoint
from utils.trajectory_stability import trajectory_metrics


@torch.no_grad()
def geometry_report(model, base_angles, anchors, fps, joint_groups):
    # Equal representation per moving part, including small gear components.
    indices = []
    for group in model.groups.unique():
        rows = torch.where(model.groups == group)[0]
        indices.append(rows[torch.linspace(0, len(rows)-1, min(32, len(rows)), device=rows.device).round().long()])
    indices = torch.cat(indices)
    before, after = model.trajectories(indices)
    before, after = before.cpu().numpy(), after.cpu().numpy()
    delta = model.angles.cpu().double().numpy()
    final_angles = base_angles+delta
    moving = [g for g in range(base_angles.shape[1]) if g not in model.locked and np.ptp(base_angles[:, g]) > 1e-6]
    ranges = np.ptp(base_angles[:, moving], axis=0)
    def angular(value):
        return float(np.sqrt(np.mean((np.diff(value[:, moving], n=3, axis=0)*fps**3/ranges)**2)))
    base_q, base_t = model.base_q, model.base_t
    final_q = qmul(model.delta_q, base_q)
    final_t = qrotate(model.delta_q, base_t)+model.delta_t
    pairs = torch.as_tensor(joint_groups, device=base_q.device)
    first = qrotate(final_q[:, pairs[:, 0]], model.pivots[None])+final_t[:, pairs[:, 0]]
    second = qrotate(final_q[:, pairs[:, 1]], model.pivots[None])+final_t[:, pairs[:, 1]]
    max_displacement = 0.
    # Bound every affected Gaussian at every frame, not just metric probes.
    for start in range(0, len(model.rows), 4096):
        a, b = model.trajectories(torch.arange(start, min(start+4096, len(model.rows)), device=model.rows.device))
        max_displacement = max(max_displacement, float((a-b).norm(dim=-1).max()))
    lag_by_group = {}
    for g in moving:
        reference, value = base_angles[:, g], final_angles[:, g]
        def score(lag):
            a, b = (reference[-lag:], value[:lag]) if lag < 0 else (reference[:-lag], value[lag:]) if lag > 0 else (reference, value)
            return float(np.mean((a-b)**2))
        lag_by_group[str(g)] = min(range(-3, 4), key=score)
    anchor_error = max([float(np.abs(delta[indices, int(group)]).max()) for group, indices in anchors.items()] or [0.])
    return dict(v3=trajectory_metrics(before, fps, float(model.scene_scale)),
                v4=trajectory_metrics(after, fps, float(model.scene_scale)),
                angular_jerk_normalized_v3=angular(base_angles), angular_jerk_normalized_v4=angular(final_angles),
                max_position_residual=max_displacement, max_joint_angle_degrees=float(np.abs(delta).max()*180/np.pi),
                max_joint_closure_error=float((first-second).norm(dim=-1).max()),
                max_relative_motion_range_change=float(np.max(np.abs(np.ptp(final_angles[:, moving], axis=0)/ranges-1))),
                turning_anchor_max_error_radians=anchor_error, phase_lag_frames=lag_by_group,
                first_frame_exact=bool(torch.all(model.angles[0] == 0)),
                locked_joint_residual_exact_zero=bool(torch.all(model.angles[:, model.locked] == 0)),
                rigid_parts=True, appearance_and_scales_frozen=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-field', type=Path, required=True, help='Verified old V4 binding; its learned residual is not inherited')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-angle-degrees', type=float, default=.1)
    parser.add_argument('--max-position-fraction', type=float, default=.0005)
    parser.add_argument('--event-budget-fraction', type=float, default=.001,
                        help='Maximum full-training-event objective increase, 0.001 = 0.1%%; report actual tradeoff')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not (0 < args.max_angle_degrees <= .5 and 0 < args.max_position_fraction <= .002 and 0 <= args.event_budget_fraction <= .001):
        parser.error('Expected conservative finite residual and event budgets')
    torch.set_num_threads(4)
    started = time.monotonic()
    old, inherited = load_checkpoint(args.base_field, 'cpu')
    baseline, candidate = Path(inherited['baseline']), Path(inherited['candidate'])
    print('Verifying exact V3 inputs...', flush=True)
    cache, record, baseline_audit = verify_candidate(baseline, candidate, allow_unapproved=True)
    if baseline_audit['artifact_sha256'] != inherited['baseline_artifact_sha256']:
        raise ValueError('Old field is not bound to the current V3')
    renderer = FrozenV3Renderer(baseline, record)
    graph = renderer.graph()
    if not torch.equal(torch.as_tensor(graph['rows']), old.rows) or not torch.equal(torch.as_tensor(graph['groups']), old.groups):
        raise ValueError('Moving V3 binding differs')
    with torch.no_grad():
        canonical = renderer.base_pose(0)[0][old.rows.cuda()].cpu()
        if not torch.equal(canonical, old.canonical): raise ValueError('Canonical V3 geometry differs')
        transformations = [renderer.correction.group_transforms(i) for i in range(cache['frame_count'])]
        base_q, base_t = [torch.stack([value[k] for value in transformations]).cpu() for k in (0, 1)]
    c = renderer.correction
    data = dict(rows=old.rows, groups=old.groups, canonical=canonical, base_q=base_q, base_t=base_t,
                parents=c.parents, parent_joints=c.parent_joints, order=c.order, locked=c.locked_groups,
                pivots=c.joint_pivots.cpu(), axes=c.joint_axes.cpu(), scene_scale=old.scene_scale,
                angles=torch.zeros(base_q.shape[:2]))
    del old
    angles = hinge_angles(base_q, data['parents'], data['parent_joints'], data['axes'])
    # These four V3 root-mounted gears share their measured driver trajectory.
    # Infer coupling only when source curves have matching normalized shape.
    coupled = []
    gear_groups = [9, 10, 11, 12]
    if max(gear_groups) < angles.shape[1] and all(g not in data['locked'] and data['parents'][g] == 0 for g in gear_groups):
        normalized = [(angles[:, g]-angles[0, g])/np.ptp(angles[:, g]) for g in gear_groups]
        if max(np.max(np.abs(a-normalized[0])) for a in normalized) < .001:
            coupled = [gear_groups]
    supervision = Path(inherited['supervision'])
    if sha256_file(supervision) != baseline_audit['artifact_sha256']['supervision']:
        raise ValueError('Event source changed')
    with np.load(supervision, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json'])); counts = archive['counts']
    if metadata['calibration'] != baseline_audit['calibration']:
        raise ValueError('Event calibration changed')
    positive, negative = metadata['calibration']['thresholds']; floor = metadata['calibration']['log_floor']
    targets = np.empty((len(counts), 80, 80), np.float32)
    for i, count in enumerate(counts):
        target = count[0].astype(np.float32)*positive-count[1].astype(np.float32)*negative
        targets[i] = target.reshape(80, 8, 80, 8).mean((1, 3))
    del counts
    targets = torch.from_numpy(targets).cuda()
    train = [i for i in range(len(targets)) if i % 5 != 4]
    validation = [i for i in range(len(targets)) if i % 5 == 4]
    args.output.mkdir(parents=True)
    sources = ['utils/v4_joint_trajectory.py', 'utils/trajectory_stability.py', 'scripts/deformation/train_v4_joint_trajectory.py',
               'utils/v3_auxiliary_renderer.py', 'utils/v3_auxiliary_mlp.py', 'utils/joint_refinement.py',
               'scripts/events/refine_lego_events.py', 'scripts/events/audit_lego_event_motion.py']
    for name in sources:
        target = args.output/'source-snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve().parents[2]/name, target)
    provenance = dict(field_kind='joint-trajectory-v4', method='bounded joint trajectory residual; constrained least squares and event-supervised selection; no MLP training',
                      baseline=str(baseline), candidate=str(candidate), supervision=str(supervision),
                      baseline_artifact_sha256=baseline_audit['artifact_sha256'], binding_checkpoint_sha256=sha256_file(args.base_field),
                      training_supervision='training events only; V3 trajectory is the frozen geometric prior',
                      validation_note='79 event intervals excluded from selection; inherited V3/calibration may have prior exposure.',
                      train_intervals=train, validation_intervals=validation, coupled_groups=coupled,
                      max_angle_degrees=args.max_angle_degrees, max_position_fraction=args.max_position_fraction,
                      event_budget_fraction=args.event_budget_fraction)
    limit = math.radians(args.max_angle_degrees)
    model = JointTrajectoryField(data, limit).cuda()
    model.validate_renderer(renderer)
    with torch.no_grad():
        for frame in (0, 50, 99, 150, 199, 299, 399):
            if not torch.equal(renderer.gray(frame), renderer.gray(frame, model)):
                raise ValueError('Zero joint field must preserve V3 renders exactly')
    @torch.no_grad()
    def evaluate(field):
        return float(np.mean([float(event_objective(renderer.gray(i, field), renderer.gray(i+1, field), targets[i], floor)) for i in train]))
    initial = evaluate(model)
    initial_geometry = geometry_report(model, angles, {}, cache['fps'], c.joint_groups.cpu().tolist())
    best_score, best_data, best_geometry, best_event, best_index = 1., copy.deepcopy(data), initial_geometry, initial, 0
    history = [dict(index=0, strength=0., fraction=0., event_objective=initial, score=1., accepted=True, geometry=initial_geometry)]
    print(json.dumps(dict(stage='baseline', event=initial, geometry=initial_geometry)), flush=True)
    index = 0
    # Symmetric offline solves introduce no causal filter delay. Hard anchors
    # preserve extrema timing/amplitude. Full training events guard each trial.
    for strength in (1., 16., 256.):
        proposal, anchors = smooth_proposal(angles, data['locked'], strength, limit, coupled)
        for fraction in (1., .25):
            index += 1
            trial = dict(data); trial['angles'] = torch.from_numpy((proposal*fraction).astype(np.float32))
            field = JointTrajectoryField(trial, limit).cuda()
            geometry = geometry_report(field, angles, anchors, cache['fps'], c.joint_groups.cpu().tolist())
            angular_ratio = geometry['angular_jerk_normalized_v4']/initial_geometry['angular_jerk_normalized_v3']
            world_ratio = geometry['v4']['jerk_rms_scene_per_s3']/initial_geometry['v3']['jerk_rms_scene_per_s3']
            feasible = (geometry['max_position_residual'] <= float(field.scene_scale)*args.max_position_fraction
                        and geometry['max_joint_closure_error'] <= c.joint_tolerance
                        and geometry['max_relative_motion_range_change'] <= 1e-6
                        and geometry['turning_anchor_max_error_radians'] == 0
                        and all(lag == 0 for lag in geometry['phase_lag_frames'].values())
                        and angular_ratio < 1 and world_ratio <= 1.001)
            value = evaluate(field) if feasible else None
            # Final composed 3D and angular jerk enter candidate selection,
            # with a separately enforced event fidelity budget.
            score = .5*angular_ratio+.5*world_ratio
            event_ok = value is not None and value <= initial*(1+args.event_budget_fraction)
            accepted = event_ok and score < best_score-1e-7
            if accepted:
                best_score, best_data, best_geometry, best_event, best_index = score, copy.deepcopy(trial), geometry, value, index
            row = dict(index=index, strength=strength, fraction=fraction, geometry_feasible=bool(feasible),
                       event_objective=value, event_guard_passed=bool(event_ok), score=score, accepted=bool(accepted), geometry=geometry)
            history.append(row)
            (args.output/'search-log.json').write_text(json.dumps(history, indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k != 'geometry'}), flush=True)
            del field
    selected = JointTrajectoryField(best_data, limit)
    provenance.update(best_step=best_index, initial_train_objective=initial, selected_train_objective=best_event,
                      selected_score=best_score, final_geometry=best_geometry)
    save_checkpoint(args.output/'auxiliary-mlp.pth', selected, provenance)
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    model, loaded = load_checkpoint(args.output/'auxiliary-mlp.pth')
    if loaded != provenance: raise RuntimeError('Checkpoint metadata changed')
    with torch.no_grad():
        rendered = np.lib.format.open_memmap(args.output/'rendered_gray.npy', mode='w+', dtype=np.float16,
                                             shape=(cache['frame_count'], 640, 640))
        for frame in range(cache['frame_count']):
            rendered[frame] = renderer.gray(frame, model).cpu().numpy()
            if frame % 100 == 0: print(json.dumps(dict(stage='render', frame=frame)), flush=True)
        rendered.flush()
    result = audit(rendered, supervision)
    result['validation_note'] = provenance['validation_note']
    (args.output/'event-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output/'baseline-event-audit.json').write_text(json.dumps(baseline_audit, indent=2)+'\n')
    summary = dict(method=provenance['method'], best_candidate=best_index, geometry=best_geometry,
                   baseline=baseline_audit['summary'], v4=result['summary'], approved=result['approved_for_export'],
                   training_event_objective_baseline=initial, training_event_objective_v4=best_event,
                   event_budget_fraction=args.event_budget_fraction, validation_note=provenance['validation_note'],
                   elapsed_seconds=time.monotonic()-started)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
