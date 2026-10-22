"""Export honest trajectory/phase evidence for the constrained V4 candidate."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from utils.scgs_v4_field import load_checkpoint
from utils.v4_joint_trajectory import hinge_angles
from utils.event_refinement import qmul


def crossing_shifts(before, after):
    result=[]
    for g in range(before.shape[1]):
        if np.ptp(before[:,g])<1e-6:continue
        a,b=before[:,g],after[:,g]
        d=np.diff(a);threshold=max(np.ptp(a)*1e-7,1e-10)
        sign=np.sign(np.where(abs(d)>threshold,d,0));turns=[0];previous=0
        for i,s in enumerate(sign):
            if s and previous and s!=previous:turns.append(i)
            if s:previous=s
        turns.append(len(a)-1)
        for start,end in zip(turns[:-1],turns[1:]):
            if end-start<8 or abs(a[end]-a[start])<1e-6:continue
            x,y=a[start:end+1],b[start:end+1];times=np.arange(start,end+1,dtype=float)
            if x[-1]<x[0]:x,y,times=x[::-1],y[::-1],times[::-1]
            for fraction in (.25,.5,.75):
                value=x[0]+fraction*(x[-1]-x[0])
                t0=float(np.interp(value,x,times));t1=float(np.interp(value,y,times))
                result.append(dict(group=g,start=start,end=end,level_fraction=fraction,shift_frames=t1-t0))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');args=parser.parse_args()
    torch.set_num_threads(1)
    model,p=load_checkpoint(args.run/'auxiliary-mlp.pth',args.device)
    final_quaternion=qmul(model.delta_q,model.base_q).cpu()
    model=model.cpu()
    summary=json.loads((args.run/'summary.json').read_text());fps=summary['geometry']['v3']['fps']
    a=hinge_angles(model.base_q,model.parents,model.parent_joints,model.axes)
    delta=model.angles.numpy().astype(np.float64)
    b=hinge_angles(final_quaternion,model.parents,model.parent_joints,model.axes)
    crossings=crossing_shifts(a,b);largest=max([abs(r['shift_frames']) for r in crossings] or [0.])
    checks=dict(crossings=crossings,max_crossing_shift_frames=largest,max_crossing_shift_ms=largest/fps*1000,
                measurement='Quarter, half and three-quarter angle crossings in each monotonic source segment; not a continuous all-time phase bound.',
                phase_guard_quarter_frame_passed=largest<=.25)
    moving=[g for g in range(a.shape[1]) if g not in model.locked and np.ptp(a[:,g])>1e-6]
    ranges=np.ptp(a[:,moving],axis=0)
    def angular_jerk(value):return float(np.sqrt(np.mean((np.diff(value[:,moving],3,axis=0)*fps**3/ranges)**2)))
    checks.update(base_angular_jerk=angular_jerk(a),planned_angular_jerk=angular_jerk(a+delta),
                  reloaded_composed_angular_jerk=angular_jerk(b),
                  max_composition_angle_roundoff=float(np.abs(b-a-delta).max()))
    indices=[]
    for group in model.groups.unique():
        rows=torch.where(model.groups==group)[0];indices.append(rows[torch.linspace(0,len(rows)-1,min(32,len(rows))).round().long()])
    indices=torch.cat(indices)
    before,after=model.trajectories(indices)
    rigidity=0.
    for group in model.groups.unique():
        mask=model.groups[indices]==group
        x,y=before[:,mask],after[:,mask]
        distance_before=(x[:,1:]-x[:,:1]).norm(dim=-1)
        distance_after=(y[:,1:]-y[:,:1]).norm(dim=-1)
        if distance_before.numel():rigidity=max(rigidity,float((distance_before-distance_after).abs().max()))
    checks['sampled_pair_distance_max_error']=rigidity
    (args.run/'phase-rigidity-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    if not checks['phase_guard_quarter_frame_passed']:raise ValueError('Subframe phase guard failed')
    np.savez_compressed(args.run/'trajectory-evidence.npz',base_joint_angles=a,residual_joint_angles=delta,
                        v3_positions=before.numpy(),v4_positions=after.numpy(),fps=fps,groups=model.groups[indices].numpy())
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    time=np.arange(len(a))/fps;g=9 if a.shape[1]>9 else 1
    axes[0,0].plot(time,np.rad2deg(a[:,g]),label='V3',color='#496884')
    axes[0,0].plot(time,np.rad2deg(b[:,g]),'--',label='V4',color='#d67c36')
    axes[0,0].set(title=f'Preserved main motion: joint {g}',xlabel='Time (s)',ylabel='Joint angle (degrees)');axes[0,0].legend()
    for group in range(delta.shape[1]):
        if np.any(delta[:,group]):axes[0,1].plot(time,np.rad2deg(delta[:,group]),label=str(group),linewidth=1)
    axes[0,1].set(title='Small auxiliary correction (actual angle units)',xlabel='Time (s)',ylabel='Residual angle (degrees)')
    if np.any(delta):axes[0,1].legend(title='Joint',ncol=4,fontsize=8)
    for value,label,color in ((a,'V3','#496884'),(b,'V4','#d67c36')):
        jerk=np.sqrt(np.mean((np.diff(value[:,moving],3,axis=0)*fps**3/ranges)**2,axis=1))
        axes[1,0].plot((np.arange(len(jerk))+1.5)/fps,jerk,label=label,color=color,linewidth=1)
    axes[1,0].set(title='Final joint acceleration change (normalized jerk)',xlabel='Time (s)',ylabel='RMS normalized jerk (1/s³)');axes[1,0].legend()
    x=np.arange(2);width=.34
    base=[summary['baseline'][s]['relative_l1'] for s in ('train_events','validation_events')]
    final=[summary['v4'][s]['relative_l1'] for s in ('train_events','validation_events')]
    for offset,values,label,color in ((-width/2,base,'V3','#496884'),(width/2,final,'V4','#d67c36')):
        bars=axes[1,1].bar(x+offset,values,width,label=label,color=color)
        axes[1,1].bar_label(bars,fmt='%.6f',fontsize=8,padding=3)
    axes[1,1].set(xticks=x,xticklabels=['Training events','Held-out for this residual'],ylabel='Relative L1 (lower is better)',title='Event fidelity at native 640 px');axes[1,1].legend(loc='lower right')
    reduction=100*(1-checks['reloaded_composed_angular_jerk']/checks['base_angular_jerk'])
    fig.suptitle(f'V4 constrained temporal residual | angular jerk change: {-reduction:+.3f}%\n'
                 f'Max measured crossing shift: {checks["max_crossing_shift_ms"]:.4f} ms | rigid-distance error: {rigidity:.2g}\n'
                 'Geometric smoothness is not proof of perceptual improvement; event holdout excludes only the new residual.',fontsize=11)
    fig.savefig(args.run/'stability-comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in checks.items() if k!='crossings'},indent=2))


if __name__=='__main__':main()
