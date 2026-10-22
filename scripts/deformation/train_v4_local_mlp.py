"""Train a local joint MLP with event supervision and final-motion constraints.

Native render finite differences provide a small-motion event surrogate.
Selection always re-renders the complete training split at native resolution.
Held-out events never enter gradients, candidate selection or basis features.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.nn.functional as F
from scripts.events.export_lego_event_preview import verify_candidate
from scripts.events.audit_lego_event_motion import audit
from scripts.deformation.train_v4_joint_trajectory import geometry_report
from scripts.deformation.report_v4_joint_stability import crossing_shifts
from utils.event_refinement import qmul
from utils.rigid_checkpoint import sha256_file
from utils.scgs_v4_field import load_checkpoint
from utils.v3_auxiliary_renderer import FrozenV3Renderer
from utils.v4_joint_trajectory import hinge_angles
from utils.v4_local_mlp import LocalJointMLP, make_local_basis, save_checkpoint, SOURCE_KEYS


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-field', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--reuse-linearization', type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.steps < 1: parser.error('Positive training steps required')
    torch.set_num_threads(4); torch.manual_seed(20261022)
    started = time.monotonic()
    old, inherited = load_checkpoint(args.base_field, 'cpu')
    baseline, candidate = Path(inherited['baseline']), Path(inherited['candidate'])
    print('Verifying frozen V3 and events...', flush=True)
    cache, record, baseline_audit = verify_candidate(baseline, candidate, allow_unapproved=True)
    if baseline_audit['artifact_sha256'] != inherited['baseline_artifact_sha256']:
        raise ValueError('V3 changed since the binding checkpoint')
    renderer = FrozenV3Renderer(baseline, record)
    data = {k: getattr(old, k).detach().cpu() for k in SOURCE_KEYS}
    data.update(parents=old.parents, parent_joints=old.parent_joints, order=old.order, locked=old.locked)
    active = [6, 7, 8]  # Small articulated branches; main transmission stays V3.
    local, base_angles = make_local_basis(data, active)
    model = LocalJointMLP(data, local).cuda().refresh()
    model.validate_renderer(renderer)
    # Confirm that inherited motion tables themselves are the exact V3 FK.
    with torch.no_grad():
        for frame in range(cache['frame_count']):
            q, t = renderer.correction.group_transforms(frame)
            if not torch.equal(q.cpu(), model.base_q[frame].cpu()) or not torch.equal(t.cpu(), model.base_t[frame].cpu()):
                raise ValueError('Frozen V3 transforms differ')
        for frame in (0, 50, 99, 150, 199, 299, 399):
            if not torch.equal(renderer.gray(frame), renderer.gray(frame, model)):
                raise ValueError('Zero MLP must render exactly V3')
    supervision = Path(inherited['supervision'])
    if sha256_file(supervision) != baseline_audit['artifact_sha256']['supervision']:
        raise ValueError('Event source changed')
    with np.load(supervision, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata_json'])); counts = z['counts']
    if meta['calibration'] != baseline_audit['calibration']:
        raise ValueError('Event calibration changed')
    positive, negative = meta['calibration']['thresholds']; floor = meta['calibration']['log_floor']
    targets = np.stack([(c[0].astype(np.float32)*positive-c[1].astype(np.float32)*negative).reshape(80,8,80,8).mean((1,3)) for c in counts])
    del counts
    target = torch.from_numpy(targets).cuda()
    train = torch.tensor([i for i in range(len(target)) if i % 5 != 4], device='cuda')
    validation = [i for i in range(len(target)) if i % 5 == 4]
    args.output.mkdir(parents=True)
    for name in ('utils/v4_local_mlp.py', 'utils/v4_joint_trajectory.py', 'utils/scgs_v4_field.py',
                 'scripts/deformation/train_v4_local_mlp.py', 'scripts/deformation/train_v4_joint_trajectory.py',
                 'utils/v3_auxiliary_renderer.py', 'scripts/events/audit_lego_event_motion.py'):
        dest = args.output/'source-snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve().parents[2]/name, dest)
    provenance = dict(field_kind='local-joint-mlp-v4', method='pose-conditioned MLP mixing bounded local joint temporal residuals',
                      baseline=str(baseline), candidate=str(candidate), supervision=str(supervision),
                      baseline_artifact_sha256=baseline_audit['artifact_sha256'],
                      binding_checkpoint_sha256=sha256_file(args.base_field),
                      training_supervision='training event counts only; frozen V3 motion supplies geometric regularization and temporal bases; no RGB targets',
                      validation_note='79 intervals excluded from new MLP gradients and selection; inherited V3 and calibration may have prior exposure.',
                      train_intervals=train.cpu().tolist(), validation_intervals=validation,
                      active_joint_groups=active, protected_joint_groups=[g for g in range(model.base_q.shape[1]) if g not in active],
                      event_budget_fraction=.0002, local_event_guard='Must improve previous V4 on identical native training-event support',
                      parameters=sum(p.numel() for p in model.mlp.parameters()), mlp_trained=True)
    dump(args.output/'provenance.json', provenance)
    def pooled(gray):
        return F.avg_pool2d(gray.clamp_min(floor).log()[None,None], 8)[0,0]
    # Finite differences avoid relying on unstable subpixel rasterizer gradients.
    # Only source renders are used here; no event validation labels are read.
    linearization = args.output/'event-linearization.npz'
    if args.reuse_linearization:
        with np.load(args.reuse_linearization, allow_pickle=False) as z:
            if str(z['source_sha256']) != sha256_file(args.base_field) or z['active'].tolist() != active:
                raise ValueError('Linearization is for a different source or local support')
            base_log = torch.from_numpy(z['base_log']).cuda(); jac = torch.from_numpy(z['jac']).cuda()
        shutil.copy2(args.reuse_linearization, linearization)
    else:
        base_log = torch.empty((cache['frame_count'],80,80), device='cuda')
        jac = torch.empty((cache['frame_count'],len(active),80,80), device='cuda')
        epsilon = 1e-4
        with torch.no_grad():
            for frame in range(cache['frame_count']):
                base_log[frame] = pooled(renderer.gray(frame))
                if frame == 0:
                    jac[frame] = 0
                    continue
                for j, group in enumerate(active):
                    values=[]
                    for sign in (-1,1):
                        model.angles.zero_(); model.angles[frame,group] = sign*epsilon
                        q,t = model.delta_transforms(); model.delta_q.copy_(q); model.delta_t.copy_(t)
                        values.append(pooled(renderer.gray(frame, model)))
                    jac[frame,j] = (values[1]-values[0])/(2*epsilon)
                if frame % 50 == 0: print(json.dumps(dict(stage='native_event_linearization',frame=frame)),flush=True)
        model.refresh()
        np.savez_compressed(linearization, base_log=base_log.cpu().numpy(), jac=jac.cpu().numpy(),
                            source_sha256=sha256_file(args.base_field), active=active)
    # Fixed, source-only image support for local fidelity, dilated one 8px cell.
    support = jac.abs().amax(1)
    support = support > (support.amax((1,2),keepdim=True)*.005).clamp_min(.001)
    support = F.max_pool2d(support.float()[:,None],3,1,1)[:,0].bool()
    mask = (support[1:] | support[:-1]).float()
    np.save(args.output/'local-event-mask.npy',mask.cpu().numpy().astype(np.uint8))
    reference = base_log[1:]-base_log[:-1]
    def losses(prediction):
        error = (prediction-target).abs()
        full = error.sum((1,2))/target.abs().sum((1,2)).clamp_min(1e-12)
        local_error = (error*mask).sum((1,2))/(target.abs()*mask).sum((1,2)).clamp_min(1e-6)
        cosine = 1-(prediction*target).sum((1,2))/(prediction.square().sum((1,2)).sqrt()*target.square().sum((1,2)).sqrt()).clamp_min(1e-12)
        return full[train].mean(), local_error[train].mean(), cosine[train].mean()
    baseline_loss = tuple(float(v) for v in losses(reference))
    angles = torch.tensor(base_angles, device='cuda', dtype=torch.float64)
    ranges = torch.tensor(np.ptp(base_angles,axis=0), device='cuda').clamp_min(1e-8)
    moving = list(range(6,13)); fps=cache['fps']
    def jerk(a, groups):
        return (torch.diff(a[:,groups],n=3,dim=0)*fps**3/ranges[groups]).square().mean()
    base_jerk = jerk(angles,active).detach()
    @torch.no_grad()
    def native_losses(field=None):
        field=model if field is None else field
        logs=torch.stack([pooled(renderer.gray(i,field)) for i in range(cache['frame_count'])])
        return tuple(float(v) for v in losses(logs[1:]-logs[:-1]))
    old=old.cuda()
    previous_loss=native_losses(old)
    previous_angles=hinge_angles(qmul(old.delta_q,old.base_q).cpu(),old.parents,old.parent_joints,old.axes.cpu())
    previous_ratio=float(np.sqrt(np.mean((np.diff(previous_angles[:,moving],3,axis=0)*fps**3/np.ptp(base_angles[:,moving],axis=0))**2))/np.sqrt(np.mean((np.diff(base_angles[:,moving],3,axis=0)*fps**3/np.ptp(base_angles[:,moving],axis=0))**2)))
    provenance.update(previous_native_train_l1=previous_loss[0],previous_native_local_train_l1=previous_loss[1],
                      selection_goal='Improve previous V4 angular jerk and both full/local training-event L1; cap full training L1 increase over V3 at 0.02 percent')
    history=[]; best=None; best_score=float('inf')
    def probe(step):
        nonlocal best,best_score
        model.refresh()
        with torch.no_grad():
            final=hinge_angles(qmul(model.delta_q,model.base_q).cpu(),model.parents,model.parent_joints,model.axes.cpu())
            ratio=float(np.sqrt(np.mean((np.diff(final[:,moving],3,axis=0)*fps**3/np.ptp(base_angles[:,moving],axis=0))**2))/np.sqrt(np.mean((np.diff(base_angles[:,moving],3,axis=0)*fps**3/np.ptp(base_angles[:,moving],axis=0))**2)))
            local_ratio=float(np.sqrt(np.mean((np.diff(final[:,active],3,axis=0)*fps**3/np.ptp(base_angles[:,active],axis=0))**2))/np.sqrt(float(base_jerk)))
            motion_range=float(np.max(np.abs(np.ptp(final[:,active],axis=0)/np.ptp(base_angles[:,active],axis=0)-1)))
            shifts=crossing_shifts(base_angles,final)
            phase=max([abs(x['shift_frames']) for x in shifts] or [0.])
        measured=native_losses()
        event_ok=(measured[0] <= min(previous_loss[0],baseline_loss[0]*(1+.0002)) and measured[1] <= previous_loss[1])
        per_group_ok=all(np.mean(np.diff(final[:,g],3)**2) <= np.mean(np.diff(base_angles[:,g],3)**2)*1.00001 for g in active)
        feasible=ratio<previous_ratio and local_ratio<1 and motion_range<1e-6 and phase<.25 and per_group_ok
        accepted=event_ok and feasible and ratio<best_score
        entry=dict(step=step,gain=model.local_settings['gain'],group_gains=model.local_settings['group_gains'],native_train_l1=measured[0],native_local_train_l1=measured[1],
                   native_train_cosine_loss=measured[2],angular_jerk_ratio=ratio,local_angular_jerk_ratio=local_ratio,
                   range_change=motion_range,max_crossing_shift_frames=phase,event_guard_passed=event_ok,feasible=feasible,accepted=accepted)
        history.append(entry);dump(args.output/'training-log.json',history);print(json.dumps(entry),flush=True)
        if accepted:
            best_score=ratio;best=(copy.deepcopy(model.mlp.state_dict()),copy.deepcopy(model.local_settings),entry)
            save_checkpoint(args.output/'accepted-progress.pth',model,{**provenance,'best_step':step})
    print(json.dumps(dict(stage='training',baseline_train_l1=baseline_loss[0],baseline_local_train_l1=baseline_loss[1],parameters=provenance['parameters'])),flush=True)
    optimizer=torch.optim.Adam(model.mlp.parameters(),lr=.003)
    for step in range(1,args.steps+1):
        optimizer.zero_grad(set_to_none=True)
        delta=model.predict_angles()
        delta_log=(jac*delta[:,active,None,None].float()).sum(1)
        predicted=reference+delta_log[1:]-delta_log[:-1]
        full,local_l1,cos=losses(predicted)
        smooth=jerk(angles+delta,active)/base_jerk
        # Absolute event objectives encourage fidelity, one-sided barriers
        # discourage exchanging event quality for geometric smoothness.
        event_penalty=200*(F.relu(full/baseline_loss[0]-1)+F.relu(local_l1/baseline_loss[1]-1))
        loss=smooth+20*(full/baseline_loss[0]+local_l1/baseline_loss[1])+event_penalty+.01*(delta/np.deg2rad(.05)).square().mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite local MLP training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.mlp.parameters(),1.);optimizer.step()
        if step in (100,300,600,args.steps):
            probe(step)
    final_state=copy.deepcopy(model.mlp.state_dict())
    # Per-branch gates are calibrated only against native training events.
    # These are local auxiliary gains; V3 motion amplitudes are never scaled.
    for group_gains in ([1.,1.,1.],[0.,1.,1.],[1.,1.,0.],[0.,1.,0.],[1.,0.,1.]):
        for gain in (.35,.5,.75,1.):
            model.mlp.load_state_dict(final_state)
            model.local_settings.update(gain=gain,group_gains=list(group_gains));probe(args.steps)
    save_checkpoint(args.output/'last-trial.pth',model,{**provenance,'best_step':args.steps,'unapproved_trial':True})
    if best is None:
        dump(args.output/'rejected-summary.json',dict(reason='No trained local MLP satisfied native event and geometry guards',baseline_train_metrics=baseline_loss,candidates=history))
        raise RuntimeError('No acceptable local MLP; existing deployed V4 remains the fallback')
    model.mlp.load_state_dict(best[0]);model.local_settings.update(best[1]);model.refresh()
    geometry=geometry_report(model,base_angles,model.anchors,fps,renderer.correction.joint_groups.cpu().tolist())
    actual=hinge_angles(qmul(model.delta_q,model.base_q).cpu(),model.parents,model.parent_joints,model.axes.cpu())
    def measured_jerk(values, groups):
        return float(np.sqrt(np.mean((np.diff(values[:,groups],n=3,axis=0)*fps**3/np.ptp(base_angles[:,groups],axis=0))**2)))
    geometry.update(angular_jerk_normalized_v4=measured_jerk(actual,moving),
                    local_angular_jerk_v3=measured_jerk(base_angles,active),local_angular_jerk_v4=measured_jerk(actual,active),
                    active_joint_groups=active,local_mlp=True,metric_source='Actual composed quaternions, not intended additive angles',
                    per_group=[dict(group=g,v3=measured_jerk(base_angles,[g]),v4=measured_jerk(actual,[g])) for g in moving])
    if geometry['max_joint_closure_error']>renderer.correction.joint_tolerance or geometry['max_position_residual']>float(model.scene_scale)*.0005:
        raise ValueError('Selected local MLP violated a final spatial constraint')
    if geometry['v4']['jerk_rms_scene_per_s3'] > geometry['v3']['jerk_rms_scene_per_s3']*1.001:
        raise ValueError('Selected local MLP exceeded the composed 3D jerk budget')
    provenance.update(best_step=best[2]['step'],selected=best[2],final_geometry=geometry,
                      baseline_train_l1=baseline_loss[0],baseline_local_train_l1=baseline_loss[1])
    save_checkpoint(args.output/'auxiliary-mlp.pth',model,provenance)
    dump(args.output/'provenance.json',provenance)
    model,loaded=load_checkpoint(args.output/'auxiliary-mlp.pth')
    if loaded != provenance:raise RuntimeError('Reload changed MLP provenance')
    with torch.no_grad():
        rendered=np.lib.format.open_memmap(args.output/'rendered_gray.npy',mode='w+',dtype=np.float16,shape=(cache['frame_count'],640,640))
        protected_error=0.
        for frame in range(cache['frame_count']):
            rendered[frame]=renderer.gray(frame,model).cpu().numpy()
            xyz,rotation,q=renderer.base_pose(frame);new,orient=model.deform(xyz,rotation,q,frame)
            protected=torch.ones(len(xyz),dtype=torch.bool,device='cuda')
            for group in active:protected[model.rows[model.groups==group]]=False
            protected_error=max(protected_error,float((xyz[protected]-new[protected]).abs().max()),float((rotation[protected]-orient[protected]).abs().max()))
            if frame%100==0:print(json.dumps(dict(stage='reload_render',frame=frame)),flush=True)
        rendered.flush()
    if protected_error != 0:raise ValueError('Protected global motion changed')
    result=audit(rendered,supervision);result['validation_note']=provenance['validation_note']
    dump(args.output/'event-audit.json',result);dump(args.output/'baseline-event-audit.json',baseline_audit)
    summary=dict(method=provenance['method'],geometry=geometry,baseline=baseline_audit['summary'],v4=result['summary'],
                 selected=best[2],baseline_native_train_l1=baseline_loss[0],baseline_native_local_train_l1=baseline_loss[1],
                 affected_gaussians=sum(int((model.groups==g).sum()) for g in active),active_joint_groups=active,
                 protected_pose_max_error=protected_error,approved=result['approved_for_export'],
                 validation_note=provenance['validation_note'],elapsed_seconds=time.monotonic()-started)
    dump(args.output/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
