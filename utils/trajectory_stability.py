"""Trajectory diagnostics: high-frequency motion is evidence, not proof of jitter."""
import numpy as np


def trajectory_metrics(positions, fps, scene_scale, cutoff_hz=10.):
    x = np.asarray(positions, dtype=np.float64)
    if (x.ndim != 3 or x.shape[0] < 9 or x.shape[1] < 1 or x.shape[2] != 3
            or not np.isfinite(x).all() or not np.isfinite(fps) or fps <= 0
            or not np.isfinite(scene_scale) or scene_scale <= 0
            or not 0 < cutoff_hz < fps/2):
        raise ValueError('Expected finite [frames>=9, probes, 3] positions and valid units')
    x = x/scene_scale
    velocity = np.diff(x, axis=0)*fps
    acceleration = np.diff(velocity, axis=0)*fps
    jerk = np.diff(acceleration, axis=0)*fps
    # Window the velocity to reduce endpoint leakage in the spectral diagnostic.
    centered = velocity-velocity.mean(0, keepdims=True)
    spectrum = np.fft.rfft(centered*np.hanning(len(centered))[:, None, None], axis=0)
    power = np.abs(spectrum)**2
    power[1:-1 if len(centered) % 2 == 0 else None] *= 2
    frequencies = np.fft.rfftfreq(len(centered), d=1/fps)
    total = float(power.sum())
    # Interior cubic-fit residual: constant-speed and smooth cubic motion vanish.
    offsets = np.arange(-4, 5, dtype=np.float64)
    weights = np.linalg.pinv(np.vander(offsets, 4, increasing=True))[0]
    smooth = sum(weight*x[i:i+len(x)-8] for i, weight in enumerate(weights))
    highpass = x[4:-4]-smooth
    def rms(value):
        return float(np.sqrt(np.mean(np.sum(value*value, axis=-1))))
    per_frame_jerk = np.sqrt(np.mean(np.sum(jerk*jerk, axis=-1), axis=1))
    return dict(probes=x.shape[1], frames=x.shape[0], fps=float(fps),
                velocity_rms_scene_per_s=rms(velocity),
                acceleration_rms_scene_per_s2=rms(acceleration),
                jerk_rms_scene_per_s3=rms(jerk),
                cubic_highpass_rms_scene=rms(highpass),
                velocity_power_above_cutoff_fraction=float(power[frequencies >= cutoff_hz].sum()/total) if total > 1e-24 else 0.,
                cutoff_hz=float(cutoff_hz),
                largest_jerk_centers_frames=(np.argsort(per_frame_jerk)[-8:][::-1]+1.5).tolist())
