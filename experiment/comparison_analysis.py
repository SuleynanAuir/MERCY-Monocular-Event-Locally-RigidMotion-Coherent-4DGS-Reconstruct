"""Capability-aware analysis for arbitrary aligned SC-GS comparison exports."""
import gzip
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from experiment.export_report_analysis import METRICS, PRIMARY, stationary_indices, paired_interval, save, table, sha
from scripts.evaluation.evaluate_experiment import artifact, third_derivative
from utils.version_presentation import export_source_order, mapping, REVISION


def number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def mean(values):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def build(root, output, base, lpips=None):
    root,output=Path(root),Path(output);data=output/'data';data.mkdir(parents=True,exist_ok=True)
    m=json.loads((root/'manifest.json').read_text());ids=export_source_order(m);nv=len(ids);samples=m['samples'];n=len(samples)
    if not 2<=nv<=6:raise ValueError('对照试验需要 2–6 个版本')
    positions=[next(i for i,v in enumerate(m['versions']) if v['id']==sid) for sid in ids]
    if m.get('version_presentation',{}).get('revision')==REVISION:
        names=[m['versions'][i]['label'].split(' · ')[0] for i in positions]
    else:names=[r['display_version'] for r in mapping([v['label'] for v in m['versions']])]
    times=np.array([s['time_seconds'] for s in samples],float)
    arrays={key:np.full((nv,n-1 if key in ('event_l1','event_cosine','polarity','silent') else n),np.nan) for key in METRICS}
    part_ids=sorted({int(k) for sid in ids for row in base['versions'][sid]['frames'] for k in row.get('parts',{})});lookup={p:i for i,p in enumerate(part_ids)}
    visibility=np.zeros((n,len(part_ids)),np.int32);missing=np.full((nv,n,len(part_ids)),np.nan)
    frame_rows=[];event_rows=[];motion=[];groups=[];curves={};availability=[]
    for v,sid in enumerate(ids):
        b=base['versions'][sid]
        for i,row in enumerate(b['frames']):
            fg=row.get('foreground',{});parts=row.get('parts',{})
            values=dict(psnr=row.get('psnr_db'),ssim=row.get('ssim'),foreground_psnr=fg.get('psnr_db'),foreground_ssim=fg.get('ssim'),lpips=row.get('lpips'),iou=row.get('silhouette_iou'),missing=None if row.get('foreground_missing_fraction') is None else row['foreground_missing_fraction']*100,part_missing=mean([p.get('alpha_missing_fraction') for p in parts.values()]))
            if values['part_missing'] is not None:values['part_missing']*=100
            if lpips and sid in lpips['versions']:values['lpips']=lpips['versions'][sid][i]
            for k,value in values.items():arrays[k][v,i]=np.nan if value is None else value
            frame_rows.append(dict(version=names[v],source_version_id=sid,frame=row['frame'],time_seconds=row['time_seconds'],psnr_infinite=row.get('psnr_infinite',False),foreground_psnr_infinite=fg.get('psnr_infinite',False),**values))
            for label,part in parts.items():
                j=lookup[int(label)];count=part.get('pixels',0)
                if v and visibility[i,j]!=count:raise ValueError('各版本公共部件掩码像素数不一致')
                visibility[i,j]=count;missing[v,i,j]=part.get('alpha_missing_fraction',np.nan)
        for e in b['events']:
            eligible=e.get('available') and e.get('split')=='response_validation' and e['sample']>0
            event_rows.append(dict(version=names[v],source_version_id=sid,primary=bool(eligible),**e))
            if eligible:
                for k,field,scale in [('event_l1','relative_l1',1),('event_cosine','cosine',1),('polarity','polarity_agreement',100),('silent','no_event_change_mean',1)]:
                    arrays[k][v,e['sample']-1]=np.nan if e.get(field) is None else e[field]*scale
        mr=dict(version=names[v],points=None,unassigned_points=None,**{k:b.get('motion',{}).get(k) for k in ('motion_amplitude','position_jerk_rms','rotation_jerk_rms','angular_jerk_rms','joint_gap_max','rigidity_error')},swing_max_rad=b.get('motion',{}).get('joint_swing_max_rad'))
        path=f'versions/{sid}/motion.npz'
        if path in m['files']:
            with np.load(artifact(root,path),allow_pickle=False) as state:
                points=state['points'];mr['points']=points.shape[1]
                if not np.array_equal(state['times_seconds'],times):raise ValueError('轨迹时间与渲染时间表不一致')
                if len(times)>=4 and len(points)==len(times):
                    jerk=third_derivative(points,times);curves[f'position_{v}']=np.sqrt(np.square(jerk).sum(-1).mean(-1))
                if 'point_groups' in state:
                    g=state['point_groups'];mr['unassigned_points']=int((g<0).sum())
                    for gid in np.unique(g):groups.append(dict(version=names[v],group=int(gid),points=int((g==gid).sum())))
        motion.append(mr)
        availability.append(dict(version=names[v],total_frames=n,aligned_rgb=sum(r.get('available',False) for r in b['frames']),infinite_psnr_frames=sum(r.get('psnr_infinite',False) for r in b['frames']),metrics={k:int(np.isfinite(a[v]).sum()) for k,a in arrays.items()},motion_unavailable=b.get('motion',{}).get('unavailable',{}),unavailable_reasons={k: ('No eligible response-validation event windows with calibrated counts' if k in ('event_l1','event_cosine','polarity','silent') else 'No finite aligned measurements; check RGB, masks, alpha, LPIPS availability and infinity flags') for k,values in arrays.items() if not np.isfinite(values[v]).any()}))
    comparisons=[];indices={}
    for k,values in arrays.items():
        for x,y in combinations(range(nv),2):
            if not np.isfinite(values[x]-values[y]).any():continue
            for block in (5,10,20):
                size=values.shape[1];key=(size,block)
                if key not in indices:indices[key]=stationary_indices(size,block)
                with np.errstate(invalid='ignore'):
                    result=paired_interval(values[x],values[y],METRICS[k][2],indices[key])
                comparisons.append(dict(metric=k,pair=f'{names[x]} -> {names[y]}',expected_block_length=block,primary=k in PRIMARY,**result))
    t=json.loads(artifact(root,m.get('performance',{}).get('timings','timings.json')).read_text());samples_t=t['samples'];repeats=max((r.get('repeat',0) for r in samples_t),default=0)+1
    lat=np.full((repeats,n,nv),np.nan)
    for r in samples_t:
        for v,pos in enumerate(positions):
            if len(r.get('panel_ms',[]))>pos:lat[r.get('repeat',0),r['sample'],v]=r['panel_ms'][pos]
    efficiency=[];summary=[]
    for v,sid in enumerate(ids):
        vals=lat[:,:,v];valid=vals[np.isfinite(vals)];row=dict(version=names[v],points=m['versions'][positions[v]].get('point_count'),mean_ms=mean(vals),p50_ms=float(np.median(valid)) if len(valid) else None,p95_ms=float(np.percentile(valid,95)) if len(valid) else None)
        efficiency.append(row);summary.append(dict(version=names[v],**{k:mean(values[v]) for k,values in arrays.items()},latency_ms=row['mean_ms'],point_count=row['points']))
        missing_refs=m.get('reference',{}).get('unavailable',{})
        reasons={}
        for k in ('foreground_psnr','iou','event_l1','lpips'):
            if summary[-1][k] is not None:continue
            if k=='event_l1':
                reason=missing_refs.get('events') or missing_refs.get('event_response')
                if reason and 'truncated' in reason:reason='原始事件文件被截断；需要完整 x/y/t/极性数据'
                elif reason and 'preview' in reason:reason='缺少原始事件；彩色预览视频不能还原事件计数'
                elif reason:reason='缺少可靠事件标定或完整时间窗：'+reason
                else:reason='没有通过标定且与拟合时间窗分离的事件验证样本'
            elif k in ('foreground_psnr','iou'):
                reason=('缺少各版本共用的前景参考掩码' if missing_refs.get('foreground_mask')
                        else 'RGB、相机、采样时间、背景或 alpha 覆盖未对齐')
            else:reason='缺少对齐 RGB 或 LPIPS 模型权重'
            reasons[k]=reason
        summary[-1]['metric_unavailable_reasons']=reasons
        # Perfect matches are infinity, never a finite zero or silently omitted mean.
        for key,field in [('psnr','psnr_infinite'),('foreground_psnr','foreground_psnr_infinite')]:
            if any(r['version']==names[v] and r[field] for r in frame_rows):summary[-1][key]=None;summary[-1][key+'_infinite']=True
    protocol=dict(manifest_sha256=base['manifest_sha256'],source_version_ids=list(ids),version_order=names,sample_count=n,resolutions=sorted({(s['width'],s['height']) for s in samples}),source_frames=[s['frame'] for s in samples],event_primary_rule='response_validation only; calibration-overlap and unspecified windows retained in raw metrics but excluded',bootstrap='5000 paired stationary resamples, blocks 5/10/20; descriptive within sequence',scope='Comparison only; no ablations',version_presentation=m.get('version_presentation',{'revision':REVISION,'legacy_display_swap':True}))
    save(data/'analysis.json',dict(summary=summary,comparisons=comparisons,motion=motion,motion_groups=groups,efficiency=efficiency,protocol=protocol,availability=availability,performance={**base.get('performance',{}),'versions':{names[i]:base.get('performance',{}).get('versions',{}).get(sid) for i,sid in enumerate(ids)}}))
    save(data/'protocol.json',protocol);save(data/'availability.json',availability);save(data/'version_mapping.json',dict(rows=[dict(display_version=names[v],source_version_id=sid,source_label=m['versions'][positions[v]]['label']) for v,sid in enumerate(ids)],revision=REVISION))
    np.savez_compressed(data/'series.npz',times=times,event_times=times[1:],part_ids=part_ids,part_visibility=visibility,part_missing_detail=missing,latencies=lat,**arrays,**curves)
    def csvfile(name,rows):
        if not rows:return
        fields=list(dict.fromkeys(k for r in rows for k in r));table(output/'tables'/name,[{k:r.get(k) for k in fields} for r in rows])
    csvfile('main_results.csv',summary);csvfile('frame_metrics.csv',frame_rows);csvfile('event_metrics.csv',event_rows);csvfile('paired_differences.csv',comparisons);csvfile('motion_diagnostics.csv',motion);csvfile('motion_group_sampling.csv',groups);csvfile('efficiency.csv',efficiency)
    csvfile('part_coverage.csv',[dict(part_id=pid,visible_frames=int((visibility[:,j]>0).sum()),visible_pixel_frames=int(visibility[:,j].sum()),**{names[v]+'_missing_percent':number(np.nansum(missing[v,:,j]*visibility[:,j])/max(visibility[:,j].sum(),1)*100) if np.isfinite(missing[v,:,j]).any() else None for v in range(nv)}) for j,pid in enumerate(part_ids)])
    with gzip.open(data/'input_manifest.json.gz','wt') as f:json.dump(m,f)
    return json.loads((data/'analysis.json').read_text())
