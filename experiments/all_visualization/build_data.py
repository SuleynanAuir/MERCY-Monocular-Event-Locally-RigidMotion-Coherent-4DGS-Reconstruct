"""Build the plotting dataset from verified archives; never change source data."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.build_summary import verify
BASE=ROOT/'experiments';OUT=Path(__file__).resolve().parent

def read(p,exact=False):
 return json.loads(p.read_text(),**(dict(parse_float=str,parse_int=str) if exact else {}))
def finite(values):
 return [float(v) if np.isfinite(v) else None for v in values]
def build():
 integrity=verify();src=read(BASE/'summary.json');runs=read(BASE/'all_runs.json')['runs'];objects=[]
 for item in src['objects']:
  path=BASE/item['source_snapshot'];raw=read(path,True);parsed=read(path)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==item['source_sha256']
  special=item['object']=='robot'
  rows=([dict(version='旧 V3',**raw['summary']['original_all']['baseline']),dict(version='推荐 V3',**raw['summary']['original_all']['candidate'])] if special else raw['summary'])
  series={};times=[];series_source=None
  if special:
   records={v:sorted([r for r in parsed['samples'] if r['version']==v and r['kind']=='original-video'],key=lambda r:r['frame']) for v in ('baseline','candidate')}
   times=[r['frame'] for r in records['candidate']]
   assert times==[r['frame'] for r in records['baseline']]
   for v,label in [('baseline','旧 V3'),('candidate','推荐 V3')]:
    series[label]={k:[r.get(k) for r in records[v]] for k in ('psnr','foreground_psnr','ssim','lpips','iou','missing')}
   series_source=item['source_snapshot']
  else:
   archives=[a for a in item['attachments'] if a['original'].endswith('/series.npz')]
   if archives:
    series_source=archives[0]['snapshot']
    with np.load(BASE/series_source,allow_pickle=False) as z:
     frame_count=len(z['times']);times=parsed.get('protocol',{}).get('source_frames',list(range(frame_count)))
     for i,row in enumerate(rows):
      series[row['version']]={k:finite(z[k][i]) for k in ('psnr','foreground_psnr','ssim','foreground_ssim','lpips','iou','missing') if k in z}
  if item['object']=='arm' and not series:
   supplemental=OUT/'supplemental'
   provenance=read(supplemental/'provenance.json')
   for entry in provenance['files']:
    assert hashlib.sha256((supplemental/entry['file']).read_bytes()).hexdigest()==entry['sha256']
   protocol=read(supplemental/'arm-protocol.json')
   assert protocol['version_order']==[r['version'] for r in rows]
   times=protocol['source_frames']
   with np.load(supplemental/'arm-series.npz',allow_pickle=False) as z:
    assert len(times)==len(z['times'])==item['sample_count']
    for i,row in enumerate(rows):
     for key in ('psnr','foreground_psnr','ssim','foreground_ssim','lpips','iou','missing'):
      assert np.isclose(np.mean(z[key][i]),float(row[key]),rtol=1e-10,atol=1e-12), (key,row['version'])
     series[row['version']]={key:finite(z[key][i]) for key in ('psnr','foreground_psnr','ssim','foreground_ssim','lpips','iou','missing')}
   series_source='visualization/supplemental/arm-series.npz'
  history=[dict(source=r['source'],summary=r['summary'],protocol=r.get('protocol')) for r in runs if r['object']==item['object']]
  objects.append(dict(id=item['object'],version=item['recommended_version'],samples=item['sample_count'],metrics=item['metrics_exact_decimal_strings'],
   versions=rows,recommended='推荐 V3' if special else 'V3',reason=item['selection_reason'],snapshot=item['source_snapshot'],sha256=item['source_sha256'],
   pointer=item['json_pointer'],original=item['source_original'],special=special,series=series,times=times,seriesSource=series_source,history=history,
   protocol=raw.get('protocol'),warnings=raw.get('warnings',[]),performance=raw.get('performance'),
   supplemental=raw['summary'] if special else None))
 data=dict(snapshot='2026-11-02',scope=src['scope'],objects=objects,reportCount=len(runs),integrity=integrity,
  provenance=dict(summary_sha256=hashlib.sha256((BASE/'summary.json').read_bytes()).hexdigest(),
  note='Original metric strings preserved; chart labels are rounded for display. Deltas are derived within each report; no cross-object ranking.'))
 text=json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))
 (OUT/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(objects=len(objects),reports=len(runs),series_objects=sum(bool(o['series']) for o in objects),verified=integrity['passed'])))
if __name__=='__main__':build()
