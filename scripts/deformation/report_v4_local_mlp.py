"""Audit actual local MLP outputs, with the previous V4 as a separate baseline."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from utils.scgs_v4_field import load_checkpoint
from utils.v4_joint_trajectory import hinge_angles
from utils.v4_local_mlp import save_checkpoint
from utils.event_refinement import qmul
from utils.event_supervision import log_intensity,signed_event_change
from utils.trajectory_stability import trajectory_metrics
from scripts.deformation.report_v4_joint_stability import crossing_shifts
from scripts.events.audit_lego_event_motion import pool


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    m,p=load_checkpoint(args.run/'auxiliary-mlp.pth','cuda')
    old,_=load_checkpoint(args.previous/'auxiliary-mlp.pth','cuda')
    # Measure the CUDA execution used by the UI before moving diagnostics to CPU.
    final_quaternion=qmul(m.delta_q,m.base_q).cpu()
    old_quaternion=qmul(old.delta_q,old.base_q).cpu()
    m=m.cpu();old=old.cpu()
    summary=json.loads((args.run/'summary.json').read_text());previous=json.loads((args.previous/'summary.json').read_text())
    fps=summary['geometry']['v3']['fps'];active=m.active_groups;moving=list(range(6,13))
    a=hinge_angles(m.base_q,m.parents,m.parent_joints,m.axes)
    b=hinge_angles(final_quaternion,m.parents,m.parent_joints,m.axes)
    o=hinge_angles(old_quaternion,old.parents,old.parent_joints,old.axes)
    ranges=np.ptp(a,axis=0)
    def angular(v,groups):return float(np.sqrt(np.mean((np.diff(v[:,groups],n=3,axis=0)*fps**3/ranges[groups])**2)))
    groups=[dict(group=g,v3=angular(a,[g]),previous_v4=angular(o,[g]),local_mlp=angular(b,[g]),
                 relative_range_change=float(np.ptp(b[:,g])/ranges[g]-1),
                 max_residual_degrees=float(m.angles[:,g].abs().max()*180/np.pi)) for g in moving]
    geometry=summary['geometry']
    enabled=[g for g in active if torch.any(m.angles[:,g] != 0)]
    affected=set(enabled)
    for g in m.order:
        if m.parents[g] in affected:affected.add(g)
    affected_count=sum(int((m.groups==g).sum()) for g in affected)
    summary.update(affected_gaussians=affected_count,enabled_auxiliary_joints=enabled,affected_rigid_groups=sorted(affected))
    p.update(enabled_auxiliary_joints=enabled,affected_rigid_groups=sorted(affected),affected_gaussians=affected_count)
    geometry.update(angular_jerk_normalized_v4=angular(b,moving),local_angular_jerk_v3=angular(a,active),
                    local_angular_jerk_v4=angular(b,active),active_joint_groups=active,local_mlp=True,
                    metric_source='Actual composed quaternions after reloading trained MLP weights',per_group=groups)
    # Update metadata only; checkpoint weights and computed motion remain exact.
    original_angles=m.angles.clone();p['final_geometry']=geometry
    save_checkpoint(args.run/'auxiliary-mlp.pth',m,p)
    reloaded,_=load_checkpoint(args.run/'auxiliary-mlp.pth','cuda')
    if not torch.equal(original_angles,reloaded.angles.cpu()):raise ValueError('Metadata update changed motion')
    (args.run/'provenance.json').write_text(json.dumps(p,indent=2)+'\n')
    summary['geometry']=geometry
    (args.run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    mask=np.load(args.run/'local-event-mask.npy').astype(bool)
    with np.load(p['supervision'],allow_pickle=False) as z:
        counts=z['counts'];meta=json.loads(str(z['metadata_json']))
    floor=meta['calibration']['log_floor'];thresholds=meta['calibration']['thresholds']
    rendered=[np.load(Path(p['candidate'])/'rendered_gray.npy',mmap_mode='r'),
              np.load(args.previous/'rendered_gray.npy',mmap_mode='r'),np.load(args.run/'rendered_gray.npy',mmap_mode='r')]
    local_errors=[]
    for values in rendered:
        errors=[]
        for i,c in enumerate(counts):
            expected=pool(signed_event_change(c.astype(np.float32),thresholds),8)
            predicted=pool(log_intensity(values[i+1],floor)-log_intensity(values[i],floor),8)
            errors.append(float((np.abs(predicted-expected)*mask[i]).sum()/max((np.abs(expected)*mask[i]).sum(),1e-6)))
        local_errors.append(errors)
    splits=dict(all=list(range(len(mask))),train=[i for i in range(len(mask)) if i%5!=4],held_out=[i for i in range(len(mask)) if i%5==4])
    local_events={split:{label:float(np.mean(np.asarray(errors)[indices])) for label,errors in zip(('v3','previous_v4','local_mlp'),local_errors)} for split,indices in splits.items()}
    checks=dict(angular_jerk={label:angular(v,moving) for label,v in (('v3',a),('previous_v4',o),('local_mlp',b))},
                local_angular_jerk={label:angular(v,active) for label,v in (('v3',a),('previous_v4',o),('local_mlp',b))},
                per_group=groups,local_event_l1=local_events,local_event_scope='Fixed source-render sensitivity mask, 80x80 cells; no held-out labels in mask construction or selection',
                affected_gaussians=summary['affected_gaussians'],active_joint_groups=active,
                protected_pose_max_error=summary['protected_pose_max_error'],
                max_crossing_shift_frames=max(abs(r['shift_frames']) for r in crossing_shifts(a,b)),
                max_joint_angle_degrees=geometry['max_joint_angle_degrees'],
                max_position_residual=geometry['max_position_residual'],max_joint_closure_error=geometry['max_joint_closure_error'],
                trained_mlp_parameters=sum(x.numel() for x in m.mlp.parameters()),
                nonzero_mlp_parameters=sum(int(torch.count_nonzero(x)) for x in m.mlp.parameters()),
                full_event_l1={label:s['all']['relative_l1'] for label,s in (('v3',summary['baseline']),('previous_v4',previous['v4']),('local_mlp',summary['v4']))},
                passed_intervals={label:s['all']['passed'] for label,s in (('v3',summary['baseline']),('previous_v4',previous['v4']),('local_mlp',summary['v4']))},
                limitation='Geometric and event improvements do not prove perceptually obvious de-jittering; inherited V3/calibration are not independent of the full source data.')
    indices=[]
    for g in active:
        rows=torch.where(m.groups==g)[0];indices.append(rows[torch.linspace(0,len(rows)-1,min(128,len(rows))).round().long()])
    before,after=m.trajectories(torch.cat(indices))
    checks['local_world_trajectory']={label:trajectory_metrics(v.numpy(),fps,float(m.scene_scale)) for label,v in (('v3',before),('local_mlp',after))}
    checks['local_world_jerk_relative_change']=checks['local_world_trajectory']['local_mlp']['jerk_rms_scene_per_s3']/checks['local_world_trajectory']['v3']['jerk_rms_scene_per_s3']-1
    checks['local_world_jerk_budget_fraction']=.001
    if checks['local_world_jerk_relative_change'] > .001:
        raise ValueError('Local composed 3D trajectory exceeded the 0.1 percent jerk budget')
    (args.run/'local-evaluation.json').write_text(json.dumps(checks,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    labels=['V3','Previous V4','Local MLP'];colors=['#61778f','#b59a74','#249d94']
    x=np.arange(len(active));width=.24
    for k,(values,label,color) in enumerate(zip((a,o,b),labels,colors)):
        axes[0,0].bar(x+(k-1)*width,[angular(values,[g]) for g in active],width,label=label,color=color)
    axes[0,0].set(xticks=x,xticklabels=[f'Joint {g}' for g in active],title='Local joint acceleration change',ylabel='Normalized angular jerk (1/s³)');axes[0,0].legend(fontsize=8)
    for g in active:axes[0,1].plot(np.arange(len(a))/fps,np.rad2deg(m.angles[:,g]),label=f'Joint {g}')
    axes[0,1].set(title='MLP correction, actual size',xlabel='Time (s)',ylabel='Residual angle (degrees)');axes[0,1].legend()
    x=np.arange(2)
    for k,(label,key,color) in enumerate(zip(labels,('v3','previous_v4','local_mlp'),colors)):
        values=[local_events[s][key] for s in ('train','held_out')]
        bars=axes[1,0].bar(x+(k-1)*width,values,width,label=label,color=color);axes[1,0].bar_label(bars,fmt='%.5f',fontsize=8,padding=3)
    axes[1,0].set(xticks=x,xticklabels=['Training events','Held-out events'],title='Local event fidelity (fixed support)',ylabel='Relative L1');axes[1,0].legend(loc='lower right',fontsize=8)
    for g in active:
        axes[1,1].plot(np.arange(len(a))/fps,np.rad2deg(a[:,g]),label=f'V3 joint {g}',alpha=.65)
        axes[1,1].plot(np.arange(len(a))/fps,np.rad2deg(b[:,g]),'--',color=axes[1,1].lines[-1].get_color())
    axes[1,1].set(title='Preserved local motion range (V4 dashed)',xlabel='Time (s)',ylabel='Joint angle (degrees)');axes[1,1].legend(fontsize=8)
    fig.suptitle('V4: trained LOCAL joint MLP | V3 main drive and rigid parts retained\nMeasured numerical gains are not proof of visually obvious improvement.',fontsize=12)
    fig.savefig(args.run/'local-mlp-evaluation.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in checks.items() if k not in ('local_world_trajectory','per_group')},indent=2))


if __name__=='__main__':main()
