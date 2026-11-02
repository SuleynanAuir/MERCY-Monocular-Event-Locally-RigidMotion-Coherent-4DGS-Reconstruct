"""Combine explicitly labeled source-asset renders and unchanged captured-video references."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from utils.experiment_format import sha256
p=argparse.ArgumentParser();p.add_argument('--renders',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
s=json.loads((a.renders/'source.json').read_text());original=json.loads(a.original.read_text())
if len(s['samples'])!=66:raise ValueError('Incomplete 66-view/time reference set')
train={r['view'] for r in s['samples'] if r['split']=='train'};valid={r['view'] for r in s['samples'] if r['split']=='validation'}
if train&valid or len(train)!=18 or len(valid)!=6:raise ValueError('Camera split overlap or missing views')
a.output.mkdir(parents=True,exist_ok=False);rows=[]
for i,r in enumerate(s['samples']):
 rgba=np.asarray(Image.open(a.renders/r['rgb']).convert('RGBA'));alpha=rgba[:,:,3]/255
 rgb=np.rint(rgba[:,:,:3]*alpha[:,:,None]).astype(np.uint8)
 # Keep the existing monochrome Robot presentation; only synthetic refs change.
 gray=np.rint(rgb@np.array([.2126,.7152,.0722])).astype(np.uint8);rgb=np.repeat(gray[:,:,None],3,axis=2)
 mask=(alpha>=.5).astype(np.uint8)*255
 if not mask.any():raise ValueError('Empty source render')
 name=f'synthetic-{i:03d}';Image.fromarray(rgb).save(a.output/(name+'.png'));Image.fromarray(mask).save(a.output/(name+'-mask.png'))
 rows.append(dict(r,rgb=name+'.png',foreground_mask=name+'-mask.png',reference_kind='synthetic-source-asset'))
for i,r in enumerate(original['samples']):
 row=dict(r,view='recorded-front',split='validation' if i%5==4 else 'train',camera=r.get('camera',original['camera']),reference_kind='original-video')
 for k in ['rgb','foreground_mask']:
  name=f'original-{i:03d}-{k}.png';shutil.copyfile(a.original.parent/r[k],a.output/name);row[k]=name
 rows.append(row)
record=dict(format='scgs_experiment_reference_v1',camera=original['camera'],background='dark',background_rgb=[0,0,0],samples=rows,provenance=dict(synthetic_color_conversion='Rec.709 luminance replicated to RGB; original video unchanged',multiview=s['provenance'],original_reference_sha256=sha256(a.original),scope='Synthetic source asset views plus unchanged original RGB; validation camera angles excluded from training'))
(a.output/'reference.json').write_text(json.dumps(record,indent=2));print(len(rows),'samples;',len(train),'train and',len(valid),'held-out synthetic views')
