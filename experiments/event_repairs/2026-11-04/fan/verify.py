import gzip,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
j=json.loads(Path('/tmp/fan-events-repair-job.json').read_text());out=Path(j['output']);root=Path(j['source'])
a=json.loads((out/'data/analysis.json').read_text());old=json.loads(Path('experiment/output/vis_fan_20261103_235954_689777/data/analysis.json').read_text());m=json.loads((root/'manifest.json').read_text());b=json.loads(gzip.decompress((out/'data/base_metrics.json.gz').read_bytes()))
for sid,new,previous in zip(['0','1','2'],a['summary'],old['summary']):
 for key in ('psnr','ssim','foreground_psnr','foreground_ssim','lpips','iou','missing'):
  assert np.isclose(new[key],previous[key],rtol=1e-10,atol=1e-12),(sid,key,new[key],previous[key])
 values=[]
 for row in m['reference']['samples']:
  part=np.asarray(Image.open(root/row['part_mask']));alpha=np.asarray(Image.open(root/f"versions/{sid}/alpha/{row['sample']:06d}.png"))/65535
  values.append(np.mean([np.mean(alpha[part==label]<.5) for label in np.unique(part) if label>0]))
 assert np.isclose(np.mean(values)*100,new['part_missing'],rtol=1e-10)
 events=[e for e in b['versions'][sid]['events'] if e.get('available') and e.get('split')=='response_validation']
 assert len(events)==39
 for key,field,scale in [('event_l1','relative_l1',1),('event_cosine','cosine',1),('polarity','polarity_agreement',100),('silent','no_event_change_mean',1)]:
  val=np.mean([e[field]*scale for e in events if e.get(field) is not None]);assert np.isfinite(val) and np.isclose(val,new[key],rtol=1e-12)
 print(new)
artifact=json.loads((out/'artifact_manifest.json').read_text());print('ARTIFACT_KEYS',artifact.keys())
print('Verified: unchanged RGB metrics; independent alpha/part macro; 39 eligible event-window means per version')
