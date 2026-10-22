"""Clearly labeled synthetic micro-jitter stress test; never a deployed candidate."""
import argparse,json,math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from utils.scgs_v4_field import load_checkpoint
from utils.v4_joint_trajectory import JointTrajectoryField, hinge_angles, anchor_frames, smooth_proposal
from utils.event_refinement import qmul, qrotate
from scripts.deformation.report_v4_joint_stability import crossing_shifts


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    model,p=load_checkpoint(args.run/'auxiliary-mlp.pth','cpu')
    a=hinge_angles(model.base_q,model.parents,model.parent_joints,model.axes)
    fps=60.;t=np.arange(len(a))/fps;noise=np.zeros_like(a);amplitude=math.radians(.02)
    coupled=p['coupled_groups'];covered={g for cluster in coupled for g in cluster}
    clusters=coupled+[[g] for g in range(a.shape[1]) if g not in model.locked and g not in covered]
    for cluster in clusters:
        driver=max(cluster,key=lambda g:np.ptp(a[:,g]));reference=a[:,driver]
        anchors=sorted({i for g in cluster for i in anchor_frames(a[:,g])})
        distance=np.abs(np.arange(len(a))[:,None]-np.array(anchors)[None]).min(1)
        u=np.clip(distance/5,0,1);envelope=u*u*(3-2*u)
        wave=amplitude*np.sin(2*np.pi*15*t)*envelope
        centered=reference-reference.mean()
        for g in cluster:
            ratio=np.dot(centered,a[:,g]-a[:,g].mean())/np.dot(centered,centered)
            noise[:,g]=wave*ratio
    noisy=a+noise
    correction,anchors=smooth_proposal(noisy,model.locked,16.,math.radians(.1),coupled)
    recovered=noisy+correction
    moving=[g for g in range(a.shape[1]) if g not in model.locked and np.ptp(a[:,g])>1e-6]
    ranges=np.ptp(a[:,moving],axis=0)
    def jerk(x):return float(np.sqrt(np.mean((np.diff(x[:,moving],3,axis=0)*fps**3/ranges)**2)))
    rms_before=float(np.sqrt(np.mean(noise[:,moving]**2)))
    rms_after=float(np.sqrt(np.mean((recovered[:,moving]-a[:,moving])**2)))
    result=dict(synthetic=True,not_a_real_v3_improvement_measurement=True,event_supervision_evaluated=False,
                purpose='Geometry-only recovery test with deliberately injected 15 Hz joint noise; does not replace original V3 or deployed V4.',
                injected_max_angle_degrees=float(np.abs(noise).max()*180/np.pi),smoothing_strength=16.,
                source_angular_jerk=jerk(a),injected_angular_jerk=jerk(noisy),recovered_angular_jerk=jerk(recovered),
                injected_angle_error_rms_radians=rms_before,recovered_angle_error_rms_radians=rms_after,
                injected_error_reduction_fraction=1-rms_after/rms_before,
                max_relative_motion_range_change=float(np.max(np.abs(np.ptp(recovered[:,moving],axis=0)/ranges-1))),
                protected_joint_residual_exact_zero=bool(np.all((recovered-a)[:,model.locked]==0)))
    shifts=crossing_shifts(a,recovered)
    result['max_measured_crossing_shift_frames']=max([abs(row['shift_frames']) for row in shifts] or [0.])
    data={k:v.clone() for k,v in model.named_buffers() if k not in ('centers','delta_q','delta_t')}
    data.update(parents=model.parents,parent_joints=model.parent_joints,order=model.order,locked=model.locked,
                angles=torch.from_numpy((recovered-a).astype(np.float32)))
    checked=JointTrajectoryField(data,model.settings['max_angle_radians'])
    record=json.loads((Path(p['candidate'])/'correction.json').read_text())
    pairs=torch.as_tensor(record['joint_structure']['joint_groups'])
    q=qmul(checked.delta_q,checked.base_q);translation=qrotate(checked.delta_q,checked.base_t)+checked.delta_t
    first=qrotate(q[:,pairs[:,0]],checked.pivots[None])+translation[:,pairs[:,0]]
    second=qrotate(q[:,pairs[:,1]],checked.pivots[None])+translation[:,pairs[:,1]]
    result['max_joint_closure_error']=float((first-second).norm(dim=-1).max())
    (args.run/'synthetic-stress-test.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,6),constrained_layout=True)
    g=6;window=(t>=.35)&(t<=.85)
    axes[0].plot(t[window],np.rad2deg(noise[window,g]),label='Injected error',color='#c2693e')
    axes[0].plot(t[window],np.rad2deg((recovered-a)[window,g]),label='After constrained smoothing',color='#31897f')
    axes[0].axhline(0,color='#6a7580',linewidth=.6);axes[0].set(ylabel='Error from V3 (degrees)',xlabel='Time (s)',title='Joint 6: recovery of deliberately injected noise');axes[0].legend()
    values=[result['source_angular_jerk'],result['injected_angular_jerk'],result['recovered_angular_jerk']]
    bars=axes[1].bar(['Original V3','Synthetic noisy input','Recovered test'],values,color=['#496884','#c2693e','#31897f'])
    axes[1].bar_label(bars,fmt='%.3f',padding=3);axes[1].set(ylabel='Normalized angular jerk (1/s³)')
    fig.suptitle('SYNTHETIC STRESS TEST — not the real V3 improvement\n15 Hz / up to 0.02° injected micro-jitter; geometry-only test, no event-quality claim',fontsize=12)
    fig.savefig(args.run/'synthetic-stress-test.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
