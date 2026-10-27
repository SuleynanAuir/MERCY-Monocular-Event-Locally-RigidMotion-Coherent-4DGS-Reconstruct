"""Analyze immutable source exports using the explicit current display mapping."""
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label

from scripts.evaluation.evaluate_experiment import verify, third_derivative, rotation_jerk
from utils.version_presentation import export_source_order, REVISION, SOURCE_TO_DISPLAY, NOTICE

VERSIONS = ('0', '1', '2')
NAMES = ('V1', 'V2', 'V3')
SEED = 20261027
METRICS = {
    'foreground_psnr': ('Foreground PSNR', 'dB', 1),
    'psnr': ('Full-frame PSNR', 'dB', 1),
    'ssim': ('Full-frame SSIM', '', 1),
    'foreground_ssim': ('Foreground SSIM', '', 1),
    'lpips': ('LPIPS (AlexNet)', '', -1),
    'iou': ('Silhouette IoU', '', 1),
    'missing': ('Foreground missing', '%', -1),
    'part_missing': ('Visible-part missing (macro)', '%', -1),
    'event_l1': ('Event relative L1', '', -1),
    'event_cosine': ('Event cosine', '', 1),
    'polarity': ('Polarity agreement', '%', 1),
    'silent': ('No-event log-change', '', -1),
}
PRIMARY = ('foreground_psnr', 'lpips', 'iou', 'event_l1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def stationary_indices(n, length, count=5000, seed=SEED):
    """Circular stationary bootstrap; paired versions share every sampled index."""
    rng = np.random.default_rng(seed)
    indices = np.empty((count, n), dtype=np.int32)
    indices[:, 0] = rng.integers(n, size=count)
    for i in range(1, n):
        restart = rng.random(count) < 1/length
        indices[:, i] = np.where(restart, rng.integers(n, size=count), (indices[:, i-1]+1) % n)
    return indices


def paired_interval(a, b, direction, indices):
    delta = (np.asarray(b)-np.asarray(a))*direction
    finite = np.isfinite(delta)
    assert finite.any()
    estimates = np.nanmean(delta[indices], axis=1)
    return dict(oriented_delta=float(np.nanmean(delta)), low=float(np.nanquantile(estimates, .025)),
                high=float(np.nanquantile(estimates, .975)), n=int(finite.sum()),
                win_fraction=float(np.mean(delta[finite]>0)),
                raw_delta=float(np.nanmean(np.asarray(b)-np.asarray(a))))


def analyze(root, cache, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    data = output/'data'; data.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'manifest.json').read_text())
    verify(root, manifest)
    with gzip.open(cache, 'rt') as f: base = json.load(f)
    assert base['manifest_sha256'] == sha(root/'manifest.json'), 'Cached metrics belong to another export'
    lpips = json.loads((data/'lpips.json').read_text())
    assert lpips['manifest_sha256'] == base['manifest_sha256']
    assert set(base['versions']) == set(VERSIONS)
    source_ids = export_source_order(manifest)
    source_positions = [next(i for i,row in enumerate(manifest['versions']) if row['id']==v) for v in source_ids]
    samples, references = manifest['samples'], manifest['reference']['samples']
    n = len(samples); times = np.array([s['time_seconds'] for s in samples])
    assert n == 200 and [s['frame'] for s in samples] == list(range(0, 400, 2))
    assert all((s['width'],s['height'],s['render_mode']) == (640,640,'gaussian') for s in samples)
    arrays = {key: [] for key in METRICS}
    frame_rows, event_rows = [], []
    for display_index,v in enumerate(source_ids):
        version = base['versions'][v]; rows = version['frames']
        assert len(rows) == n and all(r['available'] for r in rows)
        assert len(lpips['versions'][v]) == n
        values = {
            'foreground_psnr': [r['foreground']['psnr_db'] for r in rows],
            'psnr': [r['psnr_db'] for r in rows], 'ssim': [r['ssim'] for r in rows],
            'foreground_ssim': [r['foreground']['ssim'] for r in rows],
            'lpips': lpips['versions'][v], 'iou': [r['silhouette_iou'] for r in rows],
            'missing': [r['foreground_missing_fraction']*100 for r in rows],
            'part_missing': [np.mean([p['alpha_missing_fraction'] for p in r['parts'].values()])*100 for r in rows],
        }
        for key, field, scale in [('event_l1','relative_l1',1),('event_cosine','cosine',1),
                                   ('polarity','polarity_agreement',100),('silent','no_event_change_mean',1)]:
            series = np.full(n-1, np.nan)
            for e in version['events']:
                assert e['available']
                if e['split'] == 'response_validation' and e[field] is not None:
                    series[e['sample']-1] = e[field]*scale
            values[key] = series
        for key, value in values.items(): arrays[key].append(value)
        for i, row in enumerate(rows):
            frame_rows.append(dict(version=NAMES[display_index],source_version_id=v,frame=row['frame'],time_seconds=row['time_seconds'],
                                   **{key: float(value[i]) for key,value in values.items() if not key.startswith('event_') and key not in ('polarity','silent')}))
        for e in version['events']:
            event_rows.append(dict(version=NAMES[display_index],source_version_id=v,sample=e['sample'],split=e['split'],
                t0=e['t0'],t1=e['t1'],relative_l1=e['relative_l1'],cosine=e['cosine'],
                polarity_agreement=e['polarity_agreement'],no_event_change_mean=e['no_event_change_mean'],
                error_l1_sum=e['error_l1_sum'],observed_abs_sum=e['observed_abs_sum']))
    arrays = {k: np.asarray(v, dtype=np.float64) for k,v in arrays.items()}
    assert all(np.isfinite(a).sum(axis=1).tolist()==([180]*3 if a.shape[1]==199 else [200]*3) for a in arrays.values())
    table(output/'tables/frame_metrics.csv',frame_rows)
    table(output/'tables/event_metrics.csv',event_rows)
    blocks = { (m,l): stationary_indices(m,l) for m in (n,n-1) for l in (5,10,20) }
    comparisons = []
    for key, values in arrays.items():
        title, unit, direction = METRICS[key]
        for a,b in ((0,1),(1,2),(0,2)):
            for length in (5,10,20):
                result = paired_interval(values[a], values[b], direction, blocks[(values.shape[1],length)])
                comparisons.append(dict(metric=key,label=title,unit=unit,pair=f'V{a+1} -> V{b+1}',
                    expected_block_length=length,primary=key in PRIMARY,**result))
    table(output/'tables/paired_differences.csv',comparisons)
    # Structure sensitivity is an evaluation-threshold experiment, not a
    # training ablation. Every version sees the same fixed geometric masks.
    thresholds = [.25,.5,.75]
    structure = np.zeros((3,3,n,6), dtype=np.float64)
    area_counts = np.zeros((3,3,3), dtype=np.int64)  # TP, GT, prediction
    gt_bbox = [640,640,0,0]
    for i, reference in enumerate(references):
        foreground = np.asarray(Image.open(root/reference['foreground_mask'])) > 0
        ys,xs = np.nonzero(foreground)
        gt_bbox=[min(gt_bbox[0],int(xs.min())),min(gt_bbox[1],int(ys.min())),
                 max(gt_bbox[2],int(xs.max())+1),max(gt_bbox[3],int(ys.max())+1)]
        for v in range(3):
            alpha = np.asarray(Image.open(root/f'versions/{source_ids[v]}/alpha/{i:06d}.png'))/65535
            for t,threshold in enumerate(thresholds):
                predicted = alpha >= threshold
                tp=int((predicted&foreground).sum());gt=int(foreground.sum());pred=int(predicted.sum())
                iou=tp/(gt+pred-tp); missing=(gt-tp)/gt
                largest=np.nan
                if threshold==.5:
                    labels,count=label(foreground&~predicted)
                    largest=int(np.bincount(labels.ravel())[1:].max(initial=0))/gt
                    assert abs(iou-arrays['iou'][v,i])<1e-12
                structure[v,t,i] = [iou,missing,tp/gt,tp/pred if pred else 0,largest,gt]
                area_counts[v,t] += [tp,gt,pred]
        if (i+1)%50==0: print('Structure audit',i+1,'/',n,flush=True)
    sensitivity=[]
    for v in range(3):
        for t,threshold in enumerate(thresholds):
            tp,gt,pred = area_counts[v,t]
            sensitivity.append(dict(version=NAMES[v],alpha_threshold=threshold,
                iou_macro=float(structure[v,t,:,0].mean()),missing_macro_percent=float(structure[v,t,:,1].mean()*100),
                iou_pooled=float(tp/(gt+pred-tp)),missing_pooled_percent=float((gt-tp)/gt*100),
                precision_pooled=float(tp/pred),recall_pooled=float(tp/gt)))
    table(output/'tables/alpha_sensitivity.csv',sensitivity)
    part_ids = sorted({int(p) for r in base['versions']['0']['frames'] for p in r['parts']})
    part_lookup={p:i for i,p in enumerate(part_ids)}
    visibility=np.zeros((n,len(part_ids)),np.int32)
    missing=np.full((3,n,len(part_ids)),np.nan,np.float32)
    for v in range(3):
        for i,frame in enumerate(base['versions'][source_ids[v]]['frames']):
            for key,part in frame['parts'].items():
                j=part_lookup[int(key)]
                if v==0:visibility[i,j]=part['pixels']
                else:assert visibility[i,j]==part['pixels']
                missing[v,i,j]=part['alpha_missing_fraction']
    part_rows=[]
    for j,pid in enumerate(part_ids):
        support=visibility[:,j]>0
        entry=dict(part_id=pid,label=manifest['reference'].get('part_labels',{}).get(str(pid),str(pid)),
            visible_frames=int(support.sum()),visible_pixel_frames=int(visibility[:,j].sum()))
        for v in range(3):
            entry[f'V{v+1}_missing_percent']=float(np.nansum(missing[v,:,j]*visibility[:,j])/visibility[:,j].sum()*100)
        part_rows.append(entry)
    table(output/'tables/part_coverage.csv',part_rows)
    # Common groups still have different point samples. Do not promote the
    # large raw 3D-jerk difference to evidence of improved physical motion.
    motion_rows, sample_rows, group_rows = [], [], []
    motion_curves={}
    for v in range(3):
        with np.load(root/f'versions/{source_ids[v]}/motion.npz',allow_pickle=False) as state:
            motion=dict(base['versions'][source_ids[v]]['motion'])
            motion_rows.append(dict(version=NAMES[v],points=state['points'].shape[1],
                unassigned_points=int((state['point_groups']<0).sum()),
                position_jerk_rms=motion['position_jerk_rms'],rotation_jerk_rms=motion['rotation_jerk_rms'],
                angular_jerk_rms=motion['angular_jerk_rms'],joint_gap_max=motion['joint_gap_max'],
                rigidity_error=motion['rigidity_error'],motion_amplitude=motion['motion_amplitude'],
                swing_max_rad=motion['joint_swing_max_rad']))
            points=state['points'].astype(float);groups=state['point_groups']
            jerk=third_derivative(points,times)
            for group in np.unique(groups):
                keep=groups==group
                group_rows.append(dict(version=NAMES[v],group=int(group),points=int(keep.sum()),
                    position_jerk_rms=float(np.sqrt(np.square(jerk[:,keep]).sum(-1).mean())),
                    scope='Within-version samples; negative group is unassigned'))
            motion_curves[f'position_{v}']=np.sqrt(np.square(jerk).sum(-1).mean(-1))
            rotations=state['group_quaternions'].astype(float)
            rotations/=np.linalg.norm(rotations,axis=-1,keepdims=True)
            increments=2*np.arccos(np.clip(np.abs((rotations[:-1]*rotations[1:]).sum(-1)),0,1))
            for stride in (1,2,4):
                idx=np.arange(0,n,stride);pt=points[idx];tm=times[idx]
                accumulated=np.array([increments[a:b].sum(0) for a,b in zip(idx[:-1],idx[1:])])
                sample_rows.append(dict(version=NAMES[v],source_frame_step=2*stride,sample_count=len(idx),
                    rotation_path_gt_pi_count=int((accumulated>np.pi).sum()),
                    max_accumulated_rotation_rad=float(accumulated.max()),
                    rotation_alias_risk=bool((accumulated>np.pi).any()),
                    position_jerk_rms=float(np.sqrt(np.square(third_derivative(pt,tm)).sum(-1).mean())),
                    rotation_jerk_rms=float(np.sqrt(np.square(rotation_jerk(state['group_quaternions'][idx],tm)).sum(-1).mean()))))
    table(output/'tables/motion_diagnostics.csv',motion_rows)
    table(output/'tables/motion_sampling_sensitivity.csv',sample_rows)
    table(output/'tables/motion_group_sampling.csv',group_rows)
    timing=json.loads((root/manifest['performance']['timings']).read_text())
    assert len(timing['samples'])==1000 and all(not s['includes_state_capture'] for s in timing['samples'])
    latencies=np.array([r['panel_ms'] for r in timing['samples']]).reshape(5,n,3)[:,:,source_positions]
    efficiency=[]
    for v in range(3):
        efficiency.append(dict(version=NAMES[v],points=manifest['versions'][source_positions[v]]['point_count'],
            mean_ms=float(latencies[:,:,v].mean()),p50_ms=float(np.median(latencies[:,:,v])),
            p95_ms=float(np.percentile(latencies[:,:,v],95)),
            throughput_fps=1000/float(latencies[:,:,v].mean()),
            repeat_mean_min_ms=float(latencies[:,:,v].mean(1).min()),repeat_mean_max_ms=float(latencies[:,:,v].mean(1).max()),
            memory_bytes=None,memory_scope='Only process-wide memory recorded; per-version memory unavailable'))
    table(output/'tables/efficiency.csv',efficiency)
    summary=[]
    for v in range(3):
        entry=dict(version=NAMES[v],**{key:float(np.nanmean(values[v])) for key,values in arrays.items()})
        eligible=[e for e in base['versions'][source_ids[v]]['events'] if e['split']=='response_validation']
        entry['event_l1_pooled']=sum(e['error_l1_sum'] for e in eligible)/sum(e['observed_abs_sum'] for e in eligible)
        entry['latency_ms']=efficiency[v]['mean_ms'];entry['point_count']=efficiency[v]['points']
        entry['largest_missing_fraction_percent']=float(structure[v,1,:,4].mean()*100)
        summary.append(entry)
    table(output/'tables/main_results.csv',summary)
    passed = {f'V{a+1} -> V{b+1}': all((summary[b][k]-summary[a][k])*METRICS[k][2]>0 for k in PRIMARY)
              for a,b in ((0,1),(1,2),(0,2))}
    descriptor=next(s for s in manifest['sources'] if s['source'].endswith('/middle-v2.json'))
    branch=json.loads((root/descriptor['file']).read_text())
    protocol=dict(type='Retrospective frozen-artifact audit; no training or test selection was performed',
        source=str(root),manifest_sha256=base['manifest_sha256'],sample_count=n,version_order=list(NAMES),
        source_version_ids=list(source_ids),version_presentation=dict(revision=REVISION,source_to_display=SOURCE_TO_DISPLAY,disclosure=NOTICE),
        resolution=[640,640],source_frame_step=2,camera_count=1,source_fps=60,
        primary_metrics=list(PRIMARY),secondary_metrics=[k for k in METRICS if k not in PRIMARY],
        aggregation='Frame/window macro means; pixel/event pooled results separately labeled',
        event_primary_windows=180,event_calibration_overlap_windows=19,
        alpha_thresholds=thresholds,default_alpha=.5,qualitative_source_frames=[0,200,398],
        part_selection='All common visible parts contribute to metrics; current figure grouping is recorded in data/visualization_protocol.json',
        bootstrap=dict(method='paired circular stationary bootstrap',replicates=5000,seed=SEED,
            expected_block_lengths=[5,10,20],default_expected_block_length=10,
            interpretation='Descriptive within-sequence 95% resampling intervals; no scene/seed generalization or significance claim. Stationarity is only approximate.'),
        monotonic_primary_means=passed,strict_nested_provenance_verified=False,
        archived_branch=branch['branch'],training_seed_repeats=0,
        limitations=['Single scene and camera; no independent held-out training claim',
            'Current V1 (original V2) is an archived branch, not a verified direct parent of V3',
            'Shared masks are source geometry coverage, not independent semantic labels',
            'Event thresholds are fixed empirical RGB/event calibration',
            'Current V1 pure-hinge angle jerk unavailable; use labeled SO(3)/twist diagnostics',
            'Point IDs and sampling populations differ across versions; raw 3D jerk is diagnostic only',
            'At source-frame step 8, accumulated group rotation exceeds pi in 24 intervals per version; shortest-increment rotation jerk may alias',
            'Memory is process-wide; runtime repeats are not independent training seeds'])
    save(data/'protocol.json',protocol)
    save(data/'analysis.json',dict(summary=summary,comparisons=comparisons,motion=motion_rows,
        efficiency=efficiency,sensitivity=sensitivity,parts=part_rows,sampling=sample_rows,
        source_bbox=gt_bbox,provenance=dict(archived_descriptor=branch,manifest_sha256=base['manifest_sha256']),
        performance={**base['performance'],'versions':{str(i):base['performance']['versions'][v] for i,v in enumerate(source_ids)}},protocol=protocol))
    save(data/'version_mapping.json',dict(revision=REVISION,disclosure=NOTICE,source_manifest_sha256=base['manifest_sha256'],
        rows=[dict(display_version=NAMES[i],source_version_id=v,source_label=base['versions'][v]['label']) for i,v in enumerate(source_ids)],
        immutable_evidence_files=['base_metrics.json.gz','input_manifest.json.gz','lpips.json']))
    np.savez_compressed(data/'series.npz',times=times,event_times=times[1:],structure=structure,
        part_ids=part_ids,part_visibility=visibility,part_missing_detail=missing,latencies=latencies,
        **arrays,**motion_curves)
    if Path(cache).resolve() != (data/'base_metrics.json.gz').resolve():
        shutil.copyfile(cache,data/'base_metrics.json.gz')
    with gzip.open(data/'input_manifest.json.gz','wt',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False)
    print('Primary progression verdict:',passed,flush=True)
    return json.loads((data/'analysis.json').read_text())
