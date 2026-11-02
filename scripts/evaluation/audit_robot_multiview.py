"""Compare two verified appearance stages on fixed multi-camera references."""
import argparse,json,sys,io,base64,html
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from utils.experiment_format import sha256
from utils.appearance_stage import apply_stage,apply_frame,apply_geometry
from utils.render_sampling import resolve_rgb
from experiment.metrics import image_metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import torch,lpips
 from viewer import Renderer
 from gaussian_renderer import render
 from scripts.rendering.render_ply import make_camera
 from unittest.mock import patch
 torch.hub.set_dir(str(Path.home()/'.cache/scgs-comparison/weights'));load=torch.load
 with patch('torch.load',lambda *x,**kw:load(*x,**dict(kw,weights_only=True))):net=lpips.LPIPS(net='alex',version='0.1',verbose=False).cuda().eval()
 a.output.mkdir(parents=True,exist_ok=False);src=json.loads(a.reference.read_text());r=Renderer(a.sequence,'cpu');r.load();parent=sha256(a.sequence);bg=torch.zeros(3,device='cuda');results=[];gallery={};cameras={}
 with torch.no_grad():
  for label,stage in [('baseline',a.baseline),('candidate',a.candidate)]:
   apply_stage(r.model,stage.parent/'manifest.json',dict(path=stage.name,sha256=sha256(stage)),parent,len(r.poses))
   for i,row in enumerate(src['samples']):
    camera=row.get('camera',src['camera']);key=json.dumps(camera,sort_keys=True);h,w=camera['height'],camera['width']
    if key not in cameras:cameras[key]=make_camera(np.asarray(camera['pose_opengl_c2w']),w*2,h*2,2*np.arctan(h/(2*camera['K'][1][1])))
    state=r.poses[row['frame']].to('cuda');r.model._xyz=state[:,:3].contiguous();r.model._rotation=state[:,3:7].contiguous();apply_frame(r.model,row['frame']);apply_geometry(r.model)
    output=render(cameras[key],r.model,r.pipe,bg,0.,0.,0.);rgb=resolve_rgb(output['render'],2);alpha=torch.nn.functional.avg_pool2d(output['alpha'],2)
    predicted=(rgb.permute(1,2,0)*255).byte().cpu().numpy();truth=np.asarray(Image.open(a.reference.parent/row['rgb']).convert('RGB'));mask=np.asarray(Image.open(a.reference.parent/row['foreground_mask']))>0
    m=image_metrics(truth/255.,predicted/255.,mask);cover=alpha.squeeze().cpu().numpy()>=.5
    ref=torch.tensor(truth.copy(),device='cuda').permute(2,0,1).float()/255;pred=torch.tensor(predicted.copy(),device='cuda').permute(2,0,1).float()/255
    record=dict(version=label,view=row.get('view','source'),frame=row['frame'],split=row.get('split'),kind=row.get('reference_kind'),psnr=m['psnr'],ssim=m['ssim'],foreground_psnr=m['foreground']['psnr'],lpips=float(net(pred[None]*2-1,ref[None]*2-1)),iou=float((cover&mask).sum()/max((cover|mask).sum(),1)),missing=float((~cover&mask).sum()/max(mask.sum(),1)*100));results.append(record)
    if row.get('split') in ('validation','test') or (row.get('reference_kind')=='synthetic-source-asset' and row['frame']==12):
     gallery.setdefault(i,{'row':row,'reference':truth})[label]=predicted
   print('AUDIT',label,'complete',flush=True)
 keys=['psnr','ssim','foreground_psnr','lpips','iou','missing'];summary={}
 for group in ['synthetic_train','synthetic_validation','synthetic_test','original_all','original_validation']:
  summary[group]={}
  for version in ['baseline','candidate']:
   subset=[x for x in results if x['version']==version and ((group.startswith('synthetic') and x['kind']=='synthetic-source-asset' and x['split']==group.split('_')[1]) or (group.startswith('original') and x['kind']=='original-video' and (group.endswith('all') or x['split']=='validation')))]
   summary[group][version]={k:float(np.mean([x[k] for x in subset])) for k in keys}
 views={}
 for view in sorted({x['view'] for x in results if x['kind']=='synthetic-source-asset'}):
  views[view]={v:{k:float(np.mean([x[k] for x in results if x['view']==view and x['version']==v])) for k in keys} for v in ['baseline','candidate']}
 report=dict(reference_sha256=sha256(a.reference),baseline_sha256=sha256(a.baseline),candidate_sha256=sha256(a.candidate),summary=summary,per_view=views,samples=results,scope='Synthetic source-asset views; separate unchanged original-video evaluation. Validation views used for checkpoint selection; synthetic_test directions excluded from training and checkpoint selection. Same asset, not unseen-scene generalization.')
 (a.output/'audit.json').write_text(json.dumps(report,indent=2));parts=['<meta charset="utf-8"><title>Robot 多视角验证</title><h1>Robot 多视角验证</h1><p>源资产合成参考与原始视频分别评估；列：参考／旧版／候选。留出视角用于选点，非独立测试。</p><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>']
 for i,entry in gallery.items():
  pics=[Image.fromarray(entry[k]).resize((320,320)) for k in ['reference','baseline','candidate']];canvas=Image.new('RGB',(960,320))
  for x,pic in enumerate(pics):canvas.paste(pic,(x*320,0))
  buf=io.BytesIO();canvas.save(buf,format='JPEG',quality=90);row=entry['row'];parts.append(f'<h3>{html.escape(row.get("view","source"))} · frame {row["frame"]} · {row.get("split")}</h3><img width="960" src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}">')
 (a.output/'report.html').write_text('\n'.join(parts));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
