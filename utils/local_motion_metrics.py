"""Comparable local numerators AND denominators for frozen V3/V4 evaluation."""
import numpy as np

FORMAT = 'v4_focused_local_metrics_v1'


def comparison(before, after):
    before, after = float(before), float(after)
    if not np.isfinite([before, after]).all() or min(before, after) < 0:
        raise ValueError('Expected finite nonnegative scores')
    return dict(v3=before, v4=after, absolute_change=after-before,
                improvement_percent=100*(before-after)/before if before > 0 else None)


def angular_metrics(before, after, groups, fps):
    a,b=np.asarray(before,dtype=np.float64),np.asarray(after,dtype=np.float64)
    groups=list(groups)
    if (a.shape!=b.shape or a.ndim!=2 or len(a)<4 or not groups or len(set(groups))!=len(groups)
            or any(not 0<=g<a.shape[1] for g in groups) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.isfinite(fps) or fps<=0):
        raise ValueError('Invalid local joint trajectories')
    ranges=np.ptp(a[:,groups],axis=0)
    if np.any(ranges<=1e-8):raise ValueError('Cannot normalize by a static joint range')
    def values(x):
        raw=np.diff(x[:,groups],n=3,axis=0)*fps**3
        return np.sqrt(np.mean(raw**2,axis=0)),np.sqrt(np.mean((raw/ranges)**2,axis=0))
    raw_a,na=values(a);raw_b,nb=values(b)
    per_group=[dict(group=g,normalization_v3_local_range_radians=float(ranges[i]),
                    normalized_jerk=comparison(na[i],nb[i]),raw_jerk=comparison(raw_a[i],raw_b[i]),
                    v3_range_radians=float(ranges[i]),v4_range_radians=float(np.ptp(b[:,g]))) for i,g in enumerate(groups)]
    return dict(normalized_jerk=comparison(np.sqrt(np.mean(na**2)),np.sqrt(np.mean(nb**2))),
                raw_jerk=comparison(np.sqrt(np.mean(raw_a**2)),np.sqrt(np.mean(raw_b**2))),
                groups=groups,per_group=per_group,frames=len(a),difference_samples=len(a)-3,
                max_relative_range_change=float(np.max(np.abs(np.ptp(b[:,groups],axis=0)/ranges-1))),
                denominator='Each selected joint uses its frozen V3 range; RMS averages only the selected joints and all valid third differences.')


def local_sensitivity_mask(jacobian, source_groups, selected_groups):
    jac=np.asarray(jacobian)
    selected=list(selected_groups);source=list(source_groups)
    if (jac.ndim!=4 or jac.shape[1]!=len(source) or not selected or len(set(selected))!=len(selected)
            or any(g not in source for g in selected) or not np.isfinite(jac).all()):
        raise ValueError('Invalid source-only local sensitivity')
    support=np.max(np.abs(jac[:,[source.index(g) for g in selected]]),axis=1)
    frames=support>np.maximum(support.max((1,2),keepdims=True)*.005,.001)
    padded=np.pad(frames,((0,0),(1,1),(1,1)))
    frames=np.lib.stride_tricks.sliding_window_view(padded,(3,3),axis=(1,2)).any((-1,-2))
    return frames[1:]|frames[:-1]


def event_terms(predicted, observed, mask):
    predicted,observed=np.asarray(predicted,dtype=np.float64),np.asarray(observed,dtype=np.float64)
    mask=np.asarray(mask,dtype=bool)
    if predicted.shape!=observed.shape or mask.shape!=observed.shape or not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError('Both event fields must use the same finite spatial support')
    return float(np.abs(predicted-observed)[mask].sum()),float(np.abs(observed)[mask].sum())


def summarize_events(errors_v3,errors_v4,denominators,indices):
    a,b,d=[np.asarray(v,dtype=np.float64)[indices] for v in (errors_v3,errors_v4,denominators)]
    if not len(d) or not np.isfinite([a,b,d]).all() or np.any(np.array([a,b,d])<0):
        raise ValueError('Invalid accumulated local events')
    valid=d>1e-12
    # Zero-event windows are explicit. Their false predicted changes remain in
    # the summed loss; undefined per-window ratios are never assigned a huge epsilon ratio.
    mean=comparison(np.mean(a[valid]/d[valid]),np.mean(b[valid]/d[valid])) if valid.any() else None
    total=comparison(a.sum()/d.sum(),b.sum()/d.sum()) if d.sum()>1e-12 else None
    return dict(intervals=len(d),observable_intervals=int(valid.sum()),zero_event_intervals=int((~valid).sum()),
                mean_interval_relative_l1=mean,summed_relative_l1=total,
                local_observed_abs_sum=float(d.sum()),v3_error_abs_sum=float(a.sum()),v4_error_abs_sum=float(b.sum()),
                zero_event_false_change_v3=float(a[~valid].sum()),zero_event_false_change_v4=float(b[~valid].sum()))


def load_viewer_metrics(path,expected_hash,checkpoint_hash,groups,frames):
    import hashlib,json
    from pathlib import Path
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected_hash:raise ValueError('Local metric checksum mismatch')
    report=json.loads(data)
    if (report.get('format')!=FORMAT or report.get('checkpoint_sha256')!=checkpoint_hash
            or report['scope']['joint_groups']!=list(groups) or report['scope']['frame_count']!=frames):
        raise ValueError('Local metrics belong to a different model or scope')
    summary=dict(report['ui_summary'])
    if 'events' in report:
        summary['summed_held_out_event_l1']=report['events']['held_out']['summed_relative_l1']
    return summary
