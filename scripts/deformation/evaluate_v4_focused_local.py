"""Re-evaluate a frozen V3/V4 pair on the deployed MLP's local support.

No retraining, checkpoint edits, best-window search or held-out selection.
Old all-scene results remain reference metrics, with their original scope.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from utils.scgs_v4_field import load_checkpoint
from utils.event_refinement import qmul
from utils.v4_joint_trajectory import hinge_angles
from utils.event_supervision import log_intensity,signed_event_change
from utils.rigid_checkpoint import sha256_file
from utils.local_motion_metrics import FORMAT,comparison,angular_metrics,local_sensitivity_mask,event_terms,summarize_events
from scripts.events.audit_lego_event_motion import pool


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    checkpoint=args.run/'auxiliary-mlp.pth';checkpoint_hash=sha256_file(checkpoint)
    model,p=load_checkpoint(checkpoint,'cuda')
    composed=qmul(model.delta_q,model.base_q).cpu();model=model.cpu()
    groups=list(p['enabled_auxiliary_joints'])
    if groups!=[g for g in model.active_groups if torch.any(model.angles[:,g]!=0)]:
        raise ValueError('Declared active support differs from saved MLP')
    affected=set(groups)
    for g in model.order:
        if model.parents[g] in affected:affected.add(g)
    affected=sorted(affected)
    frames=len(model.base_q)
    prior=json.loads((args.run/'summary.json').read_text());fps=prior['geometry']['v3']['fps']
    baseline_angles=hinge_angles(model.base_q,model.parents,model.parent_joints,model.axes)
    v4_angles=hinge_angles(composed,model.parents,model.parent_joints,model.axes)
    angular=angular_metrics(baseline_angles,v4_angles,groups,fps)
    args.output.mkdir(parents=True)
    linearization=args.run/'event-linearization.npz'
    with np.load(linearization,allow_pickle=False) as z:
        if str(z['source_sha256'])!=p['binding_checkpoint_sha256']:
            raise ValueError('Source-render sensitivity belongs to a different V3 binding')
        mask=local_sensitivity_mask(z['jac'],z['active'].tolist(),groups)
    np.save(args.output/'focused-event-mask.npy',mask.astype(np.uint8))
    world=[]
    for group in affected:
        indices=torch.where(model.groups==group)[0]
        canonical=model.canonical[indices].numpy().astype(np.float64)
        scale=float(np.linalg.norm(np.ptp(canonical,axis=0)))
        if scale<=1e-8:raise ValueError('Degenerate local part size')
        a,b=model.trajectories(indices)
        a,b=a.numpy().astype(np.float64),b.numpy().astype(np.float64)
        def jerk(x):return float(np.sqrt(np.mean(np.sum((np.diff(x,3,axis=0)*fps**3)**2,axis=-1))))
        ja,jb=jerk(a),jerk(b)
        world.append(dict(group=group,gaussians=len(indices),normalization_v3_local_length=scale,
                          raw_jerk=comparison(ja,jb),normalized_jerk=comparison(ja/scale,jb/scale)))
    world_score=comparison(np.sqrt(np.mean([g['normalized_jerk']['v3']**2 for g in world])),
                           np.sqrt(np.mean([g['normalized_jerk']['v4']**2 for g in world])))
    supervision=Path(p['supervision']);v3_path=Path(p['candidate'])/'rendered_gray.npy';v4_path=args.run/'rendered_gray.npy'
    hashes=dict(checkpoint=checkpoint_hash,supervision=sha256_file(supervision),v3_render=sha256_file(v3_path),
                v4_render=sha256_file(v4_path),source_sensitivity=sha256_file(linearization))
    if hashes['supervision']!=p['baseline_artifact_sha256']['supervision'] or hashes['v3_render']!=p['baseline_artifact_sha256']['rendered_gray']:
        raise ValueError('Frozen V3 render or event data changed')
    with np.load(supervision,allow_pickle=False) as z:
        counts=z['counts'];meta=json.loads(str(z['metadata_json']))
    floor=meta['calibration']['log_floor'];threshold=meta['calibration']['thresholds']
    v3=np.load(v3_path,mmap_mode='r');v4=np.load(v4_path,mmap_mode='r')
    if v3.shape!=v4.shape or len(v3)!=frames or len(mask)!=frames-1 or len(counts)!=frames-1:
        raise ValueError('Metric timelines differ')
    errors_a,errors_b,denominators,global_denominators=[],[],[],[]
    for i,count in enumerate(counts):
        observed=pool(signed_event_change(count.astype(np.float32),threshold),8)
        first=pool(log_intensity(v3[i+1],floor)-log_intensity(v3[i],floor),8)
        second=pool(log_intensity(v4[i+1],floor)-log_intensity(v4[i],floor),8)
        ea,da=event_terms(first,observed,mask[i]);eb,db=event_terms(second,observed,mask[i])
        if da!=db:raise ValueError('V3/V4 denominators differ')
        errors_a.append(ea);errors_b.append(eb);denominators.append(da);global_denominators.append(float(np.abs(observed).sum()))
    splits=dict(all=list(range(frames-1)),train=[i for i in range(frames-1) if i%5!=4],held_out=[i for i in range(frames-1) if i%5==4])
    events={name:summarize_events(errors_a,errors_b,denominators,indices) for name,indices in splits.items()}
    np.savez_compressed(args.output/'focused-event-terms.npz',v3_error=errors_a,v4_error=errors_b,local_denominator=denominators,global_denominator=global_denominators)
    scope=dict(joint_groups=groups,affected_rigid_groups=affected,affected_gaussians=sum(x['gaussians'] for x in world),
               frame_count=frames,interval_count=frames-1,train_interval_count=len(splits['train']),held_out_interval_count=len(splits['held_out']),
               fps=fps,mean_image_coverage_percent=float(mask.mean()*100),
               local_event_mass_percent=float(sum(denominators)/sum(global_denominators)*100),
               pixel_comparison_resolution=[80,80],source_render_resolution=list(v3.shape[1:]),
               selection='Enabled joints from the already frozen deployed checkpoint, before calculating this report; all time intervals retained.',
               mask='V3 source-render sensitivity of selected joints; fixed 0.5% relative / 0.001 absolute threshold, one-cell dilation, union of consecutive frames; identical mask for V3 and V4.',
               scope_note='Post-training focused diagnostic, not a newly trained improvement or an independently preregistered benchmark. Coarse image support may include occluded or neighboring parts.')
    global_reference=dict(angular_jerk=comparison(prior['geometry']['angular_jerk_normalized_v3'],prior['geometry']['angular_jerk_normalized_v4']),
                          event_l1=comparison(prior['baseline']['all']['relative_l1'],prior['v4']['all']['relative_l1']),
                          passed=dict(v3=prior['baseline']['all']['passed'],v4=prior['v4']['all']['passed'],intervals=frames-1))
    ui=dict(scope=scope,angular=angular['normalized_jerk'],per_joint=angular['per_group'],world_jerk=world_score,
            event_l1=events['all']['mean_interval_relative_l1'],held_out_event_l1=events['held_out']['mean_interval_relative_l1'],
            summed_event_l1=events['all']['summed_relative_l1'],observable_intervals=events['all']['observable_intervals'],
            zero_event_intervals=events['all']['zero_event_intervals'],motion_range_change=angular['max_relative_range_change'],
            global_reference=global_reference,checkpoint_unchanged=True)
    report=dict(format=FORMAT,checkpoint_sha256=checkpoint_hash,input_sha256=hashes,scope=scope,angular=angular,
                world=dict(normalized_jerk=world_score,per_part=world,denominator='Frozen V3 local-part bounding-box diagonal; all affected Gaussians, equal weighting per local part.'),
                events=events,event_denominator='For each interval, sum(abs(observed events)) only inside the fixed local mask; numerator is sum(abs(V3/V4 predicted - observed)) inside the SAME mask.',
                improvement_denominator='The corresponding V3 LOCAL score: 100*(V3_local-V4_local)/V3_local. Positive means lower error/jerk.',
                global_reference=global_reference,ui_summary=ui)
    if sha256_file(checkpoint)!=checkpoint_hash:raise ValueError('Evaluation must not alter the MLP checkpoint')
    output=args.output/'focused-metrics.json';output.write_text(json.dumps(report,indent=2)+'\n')
    source_manifest=json.loads((args.run/'viewer-manifest.json').read_text())
    manifest={**source_manifest,'checkpoint':str(checkpoint.resolve()),'audit':str((args.run/'event-audit.json').resolve()),
              'local_metrics':output.name,'local_metrics_sha256':sha256_file(output)}
    (args.output/'viewer-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ('utils/local_motion_metrics.py','scripts/deformation/evaluate_v4_focused_local.py'):
        dest=args.output/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__).resolve().parents[2]/name,dest)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    x=np.arange(len(groups));width=.35
    for offset,label,color in ((-width/2,'v3','#61778f'),(width/2,'v4','#249d94')):
        bars=axes[0,0].bar(x+offset,[r['normalized_jerk'][label] for r in angular['per_group']],width,label=label.upper(),color=color)
        axes[0,0].bar_label(bars,fmt='%.5f',fontsize=9)
    axes[0,0].set(xticks=x,xticklabels=[f'Joint {g}' for g in groups],title='Only enabled local joints (lower is better)',ylabel='Jerk / frozen V3 local angular range');axes[0,0].legend()
    x=np.arange(3)
    for offset,label,color in ((-width/2,'v3','#61778f'),(width/2,'v4','#249d94')):
        vals=[events[s]['mean_interval_relative_l1'][label] for s in ('all','train','held_out')]
        bars=axes[0,1].bar(x+offset,vals,width,label=label.upper(),color=color);axes[0,1].bar_label(bars,fmt='%.5f',fontsize=9)
    axes[0,1].set(xticks=x,xticklabels=['All intervals','Training','Held-out'],title='Same local mask in numerator and denominator',ylabel='Local event L1 (mean interval ratio)');axes[0,1].legend(loc='lower right')
    axes[1,0].imshow(np.asarray(v3[150]),cmap='gray',vmin=0,vmax=1)
    axes[1,0].contour(np.repeat(np.repeat(mask[150],8,0),8,1),levels=[.5],colors=['#43e6b5'],linewidths=.8)
    axes[1,0].set(title=f'Fixed V3 source sensitivity ROI | mean coverage {scope["mean_image_coverage_percent"]:.2f}%');axes[1,0].axis('off')
    items=[('Local angular jerk',ui['angular']),('Local event L1',ui['event_l1']),('Local held-out L1',ui['held_out_event_l1']),('Local 3D jerk',ui['world_jerk'])]
    vals=[v['improvement_percent'] for _,v in items];bars=axes[1,1].barh([n for n,_ in items],vals,color=['#249d94' if x>=0 else '#ba745a' for x in vals])
    axes[1,1].bar_label(bars,fmt='%+.3f%%',padding=3,fontsize=9);axes[1,1].axvline(0,color='#9aa4ab',linewidth=1)
    axes[1,1].set(title='V3 local score is the improvement denominator',xlabel='Improvement (%) — negative means worse');axes[1,1].margins(x=.25)
    fig.suptitle('Same frozen V4 checkpoint | focused LOCAL evaluation, all frames\nChanged measurement scope, not a new training gain.',fontsize=12)
    fig.savefig(args.output/'focused-local-comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(ui,indent=2),flush=True)


if __name__=='__main__':main()
