"""Analytic constant-speed Fan control for finite-difference roundoff."""
import json
import numpy as np
from .protocol import ROOT,save_json,sha256
from .metrics import jerk
from .visualize import style,save,COLORS


def main(root=ROOT):
    import matplotlib.pyplot as plt
    prepared=root/'01_main_comparison/prepared/fan'
    prior=json.loads((prepared/'prior.json').read_text())
    sup=json.loads((prepared/'supervision/metadata.json').read_text())
    evaluation=root/'01_main_comparison/runs/fan_C1_s17_full/evaluation.json'
    baseline=json.loads(evaluation.read_text())
    times=np.asarray(sup['split']['frames']['test'])/prior['video']['fps']
    angle=times*prior['details']['angular_velocity_rad_s']
    rounded=angle.astype(np.float32).astype(np.float64)
    high=jerk(angle[:,None],times,angular=True)['value']
    low=jerk(rounded[:,None],times,angular=True)['value']
    actual=baseline['metrics']['angular_jerk']['value']
    record=dict(scene='fan',control='Analytic constant angular velocity from the frozen training-only prior',
        theoretical_angular_jerk=0.,float64_sampled_angular_jerk=high,float32_sampled_angular_jerk=low,
        actual_native_C1_angular_jerk=actual,unit='rad/s^3',frame_ids=sup['split']['frames']['test'],
        evaluation_sha256=sha256(evaluation),
        interpretation='The nonzero C1 angular jerk comes from numerical sampling/rotation arithmetic; C1 has constant angular velocity by construction. Near-floor changes must not be claimed as physical jitter improvements. 3D jerk additionally contains the legitimate centripetal motion contribution and is not itself a pure jitter metric.')
    save_json(root/'05_significance/precision_control.json',record)
    style();fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].plot(times,angle,color=COLORS[0],lw=2,label='Analytic angle (float64)')
    axes[0].plot(times,rounded,'o',color=COLORS[2],ms=3,label='Rounded float32 samples')
    axes[0].set(title='Same constant-speed trajectory',xlabel='Test time (s)',ylabel='Joint angle (rad)')
    axes[0].legend(frameon=False,fontsize=8)
    for data,color,label in [(angle,COLORS[0],'float64'),(rounded,COLORS[2],'float32')]:
        derivative=np.diff(data,n=3)/(times[1]-times[0])**3
        axes[1].plot(times[3:],derivative,'o-',color=color,ms=3,label=label)
    axes[1].set(title='Third differences amplify rounding',xlabel='Test time (s)',ylabel='Angular jerk (rad/s³)')
    axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Numerical precision control · theoretical angular jerk = 0',fontsize=15,fontweight='bold')
    fig.supxlabel(f'Native C1 measured RMS = {actual:.5g} rad/s³ · does not establish physical jitter',fontsize=10)
    save(fig,root/'05_significance/visualization/precision_control')
    print('Precision control:',record,flush=True)


if __name__=='__main__':main()
