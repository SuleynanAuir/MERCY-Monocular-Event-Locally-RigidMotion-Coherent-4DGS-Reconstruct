"""Metrics keep per-frame denominators and never replace missing evidence with zero."""
import numpy as np
from skimage.metrics import structural_similarity


def image_metrics(reference, predicted, foreground=None):
    a, b = np.asarray(reference, np.float64), np.asarray(predicted, np.float64)
    if (a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3 or min(a.shape[:2]) < 11
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or min(a.min(), b.min()) < 0 or max(a.max(), b.max()) > 1):
        raise ValueError('Expected aligned RGB [H,W,3] in [0,1]')
    difference = a-b;mse = float(np.mean(difference**2))
    score, field = structural_similarity(a, b, data_range=1., channel_axis=-1,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False, full=True)
    row = dict(rgb_l1=float(np.abs(difference).mean()), mse=mse,
               psnr=float(-10*np.log10(mse)) if mse > 0 else None,
               psnr_exact_match=mse == 0, ssim=float(score))
    if foreground is not None:
        mask = np.asarray(foreground, bool)
        if mask.shape != a.shape[:2] or not mask.any():
            raise ValueError('Foreground must be a nonempty external fixed mask')
        m = float(np.mean(difference[mask]**2))
        row['foreground'] = dict(pixels=int(mask.sum()), rgb_l1=float(np.abs(difference[mask]).mean()),
            psnr=float(-10*np.log10(m)) if m else None, psnr_exact_match=m == 0,
            ssim=float(field[mask].mean()))
    return row


def missing_fraction(reference_mask, alpha, threshold=.5):
    mask = np.asarray(reference_mask, bool);alpha = np.asarray(alpha)
    if mask.shape != alpha.shape or not np.isfinite(alpha).all() or not 0 < threshold < 1:
        raise ValueError('Invalid coverage inputs')
    denominator = int(mask.sum());numerator = int((mask & (alpha < threshold)).sum())
    return dict(numerator=numerator, denominator=denominator,
                value=numerator/denominator if denominator else None, threshold=threshold)


def jerk(trajectory, times, angular=False):
    """RMS Euclidean third derivative; angles are declared unwrapped joint radians."""
    values = np.asarray(trajectory, np.float64);times = np.asarray(times, np.float64)
    if len(times) < 4 or values.shape[0] != len(times) or not np.isfinite(values).all():
        raise ValueError('Need at least four finite trajectory samples')
    dt = np.diff(times)
    if (not np.isfinite(times).all() or np.any(dt <= 0)
            or not np.allclose(dt, dt[0], rtol=1e-5, atol=1e-9)):
        raise ValueError('Jerk requires uniformly sampled increasing timestamps')
    if angular:
        values = np.unwrap(values, axis=0)
    derivative = np.diff(values, n=3, axis=0)/dt[0]**3
    squares = derivative**2
    if not angular:
        if values.ndim != 3 or values.shape[-1] != 3:
            raise ValueError('3D jerk requires [time, point, xyz]')
        squares = squares.sum(-1)
    return dict(value=float(np.sqrt(squares.mean())), unit='rad/s^3' if angular else 'scene_unit/s^3',
                samples=len(derivative), aggregation='RMS over times and fixed active IDs',
                amplitude=np.ptp(values, axis=0).tolist())


def holm(pvalues):
    values = np.asarray(pvalues, float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('Invalid p values')
    order = np.argsort(values);out = np.empty_like(values)
    out[order] = np.minimum(1, np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return out.tolist()


def paired_effect(baseline, candidate, direction, seed=20261026, resamples=10000):
    """Independent scene/run units only; frame-level pseudoreplication is forbidden."""
    a,b = np.asarray(baseline,float),np.asarray(candidate,float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid paired observations')
    diff=(b-a)*direction
    if len(diff)<2:return dict(status='insufficient_independent_units',n=len(diff))
    rng=np.random.default_rng(seed)
    boot=np.mean(diff[rng.integers(0,len(diff),(resamples,len(diff)))],axis=1)
    # Exact paired sign-flip test is valid only for independent exchangeable
    # pairs under the null; this assumption is part of the result metadata.
    if len(diff)<=18:
        bits=(np.arange(2**len(diff))[:,None]>>np.arange(len(diff)))&1
        null=np.mean((2*bits-1)*diff,axis=1)
        p=float(np.mean(np.abs(null)>=abs(diff.mean())-1e-14))
    else:
        null=np.mean(rng.choice([-1,1],size=(resamples,len(diff)))*diff,axis=1)
        p=float((1+np.count_nonzero(np.abs(null)>=abs(diff.mean())))/(resamples+1))
    return dict(status='measured',n=len(diff),effect=float(diff.mean()),
        ci95=np.quantile(boot,[.025,.975]).tolist(),p_raw=p,seed=seed,resamples=resamples,
        bootstrap='paired percentile over independent scene/run units',
        permutation='two-sided paired sign-flip; assumes exchangeable signs under null')
