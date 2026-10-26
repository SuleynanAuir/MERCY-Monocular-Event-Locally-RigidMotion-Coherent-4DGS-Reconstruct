"""Describe a frozen prior's training fit without selecting a new model."""
import json
from pathlib import Path
import numpy as np
from .protocol import ROOT, save_json, sha256


def fan_training_fit(root=ROOT):
    root = Path(root)
    path = root/'01_main_comparison/diagnostics/fan_phase_prior/prior.json'
    if not path.exists():
        return
    prior = json.loads(path.read_text())
    active_path = root/'01_main_comparison/prepared/fan/prior.json'
    active = json.loads(active_path.read_text())
    if any(prior[key] != active[key] for key in ('source_sha256', 'motion_sha256')):
        raise ValueError('Diagnostic is not the same frozen Fan input')
    details = prior['details']
    times = np.asarray(details['training_frames'])/prior['video']['fps']
    observed = np.asarray(details['observed_training_angles_rad'])
    fitted = times*details['angular_velocity_rad_s']
    error = observed-fitted
    rmse = float(np.sqrt(np.mean(error**2)))
    if not np.isclose(rmse, active['details']['training_angle_rmse']):
        raise ValueError('Diagnostic does not reproduce the frozen training fit')
    record = dict(scene='fan', scope='Post-run diagnostic of training frames only; no model selection or score replacement',
        prior_sha256=sha256(active_path), diagnostic_sha256=sha256(path),
        source_sha256=prior['source_sha256'], motion_sha256=prior['motion_sha256'],
        training_frames=details['training_frames'], observed_training_angles_rad=observed.tolist(),
        constant_speed_angles_rad=fitted.tolist(), training_rmse_rad=rmse,
        training_rmse_degrees=float(np.rad2deg(rmse)),
        local_rotation_vector_norm_cap_degrees=float(np.sqrt(3)*.5),
        reference='Polar-correlation estimates from training RGB, not independent motion ground truth',
        interpretation='The constant-speed prior underfits varying and reversing training motion. This is not a clean test of sub-degree residual correction. Preserve the registered benchmark; any improved prior requires a separately declared revision applied equally to all methods.')
    save_json(path.parent/'training_fit.json', record)
    from .visualize import style, save, COLORS
    import matplotlib.pyplot as plt
    style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    axes[0].plot(times, np.rad2deg(observed), 'o-', ms=3, color=COLORS[0], label='Training RGB polar estimate')
    axes[0].plot(times, np.rad2deg(fitted), color=COLORS[1], lw=2, label='Frozen constant-speed prior')
    axes[0].set(xlabel='Training time (s)', ylabel='Rotation from initial frame (degrees)', title='Motion prior fit')
    axes[0].legend(frameon=False, fontsize=8)
    cap = record['local_rotation_vector_norm_cap_degrees']
    axes[1].axhspan(-cap, cap, color=COLORS[2], alpha=.2, label='Local rotation norm cap (±0.87°)')
    axes[1].plot(times, np.rad2deg(error), color=COLORS[0], lw=2, label='Training estimate − prior')
    axes[1].axhline(0, color='#AAB6BF', lw=.7)
    axes[1].set(xlabel='Training time (s)', ylabel='Angular difference (degrees)', title=f'Training fit RMSE = {np.rad2deg(rmse):.2f}°')
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('Fan failure analysis · frozen prior cannot explain training motion', fontsize=15, fontweight='bold')
    fig.supxlabel('Image-derived estimates, not motion ground truth · diagnostic only · all registered scores retained', fontsize=9)
    save(fig, root/'01_main_comparison/visualization/fan_prior_diagnostic')
    return record


if __name__ == '__main__':
    fan_training_fit()
