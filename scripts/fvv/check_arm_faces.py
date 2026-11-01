"""Compare every whole-arm direction; report worst views and source part IDs.

Reference segmentation is downsampled source geometry, not captured truth.
No good-view selection or automatic model activation.
"""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image,ImageDraw
import hybrid
from refine_lego import view_camera
from rigid_bucket import bucket_only_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['repo','previous','model','views','masks','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    meta=json.loads((a.views/'views.json').read_text());maskmeta=json.loads((a.masks/'masks.json').read_text())
    if len(meta['views'])!=meta['expected_views'] or len(maskmeta['views'])!=len(meta['views']):raise ValueError('Incomplete references')
    masks={r['image']:r['mask'] for r in maskmeta['views']};ids=[p['id'] for p in maskmeta['arm_parts']];report={}
    validation=[v for v in meta['views'] if v['split']=='validation'];tile=240
    panels={e:Image.new('RGB',(8*tile,3*(tile+24))) for e in sorted(set(v['elevation'] for v in validation))}
    for name,model,row in [('previous',a.previous,1),('candidate',a.model,2)]:
        r=hybrid.Renderer(SimpleNamespace(repo=a.repo,model=model));rows=[]
        if r.bucket is None:raise ValueError('Comparison requires two rigid arm models')
        isolated_render=r.bucket.arm_only().wrap(r.base_render_fn)
        arm_only=bucket_only_model(r.model)
        with torch.no_grad():
            for view in meta['views']:
                target=torch.from_numpy(np.array(Image.open(a.views/view['file']).convert('RGB'))).permute(2,0,1).float()/255
                image=isolated_render(view_camera(view),arm_only,r.pipe,r.bg)['render'].clamp(0,1).cpu()
                with np.load(a.masks/masks[view['file']],allow_pickle=False) as d:parts=d['parts'];visible_parts=d['visible_parts']
                target_small=torch.nn.functional.interpolate(target[None],size=(160,160),mode='area')[0]
                image_small=torch.nn.functional.interpolate(image[None],size=(160,160),mode='area')[0]
                error=image_small-target_small;mask=torch.from_numpy(np.isin(parts,ids))
                def metrics(m):
                    count=int(m.sum())
                    if count<4:return dict(pixels=count,psnr=None,dark_fraction=None)
                    return dict(pixels=count,psnr=float(-10*torch.log10(error[:,m].square().mean().clamp_min(1e-12))),dark_fraction=float(((image_small.mean(0)-target_small.mean(0))[m]<-.15).float().mean()))
                record={k:view[k] for k in ['file','time','angle','elevation','split']};record.update(metrics(mask))
                full_image=r.render_fn(view_camera(view),r.model,r.pipe,r.bg)['render'].clamp(0,1).cpu()
                full_small=torch.nn.functional.interpolate(full_image[None],size=(160,160),mode='area')[0]
                full_mask=torch.from_numpy(np.isin(visible_parts,ids));count=int(full_mask.sum())
                record['full_scene_visible_arm']=dict(pixels=count,psnr=float(-10*torch.log10((full_small-target_small)[:,full_mask].square().mean().clamp_min(1e-12))) if count>=64 else None,dark_fraction=float(((full_small.mean(0)-target_small.mean(0))[full_mask]<-.15).float().mean()) if count>=64 else None)
                record['parts']={str(i):metrics(torch.from_numpy(parts==i)) for i in ids};rows.append(record)
                if view['split']=='validation':
                    group=[v for v in validation if v['elevation']==view['elevation']];col=group.index(view);panel=panels[view['elevation']]
                    for rgb,rr in [(target,0),(image,row)]:
                        panel.paste(Image.fromarray((rgb.permute(1,2,0).numpy()*255).astype('uint8')).resize((tile,tile)),(col*tile,rr*(tile+24)+24))
                        ImageDraw.Draw(panel).text((col*tile+3,rr*(tile+24)+3),f'{["Source","Previous","Candidate"][rr]} az={view["angle"]:g}',fill='white')
        summary={}
        for split in ['train','validation']:
            subset=[x for x in rows if x['split']==split and x['psnr'] is not None]
            full_subset=[x['full_scene_visible_arm'] for x in subset if x['full_scene_visible_arm']['psnr'] is not None]
            summary[split]=dict(full_scene_visible_arm=dict(views=len(full_subset),mean_psnr=float(np.mean([x['psnr'] for x in full_subset])),worst_psnr=min(x['psnr'] for x in full_subset),mean_dark_fraction=float(np.mean([x['dark_fraction'] for x in full_subset]))),mean_psnr=float(np.mean([x['psnr'] for x in subset])),worst_psnr=min(x['psnr'] for x in subset),mean_dark_fraction=float(np.mean([x['dark_fraction'] for x in subset])),worst_views=sorted(subset,key=lambda x:x['psnr'])[:5],by_elevation={str(e):float(np.mean([x['psnr'] for x in subset if x['elevation']==e])) for e in panels})
        report[name]=dict(sha256=r.meta['sha256'],arm_sha256=r.meta.get('rigid_bucket',{}).get('sha256'),summary=summary,rows=rows)
        del r;torch.cuda.empty_cache()
    for e,panel in panels.items():panel.save(a.output/f'elevation-{e:+g}.jpg',quality=95)
    report['scope']=f'All {len(meta["views"])} source RGB views; {len(validation)} middle-time intermediate-angle images excluded from this refinement. Geometry and original video are production-fitted. Isolated arm and full-scene visible-arm metrics use 160-square source-part masks; full-scene views require at least 64 visible arm pixels. Not arbitrary-view guarantee.'
    (a.output/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v['summary'] for k,v in report.items() if isinstance(v,dict)},indent=2))


if __name__=='__main__':main()
