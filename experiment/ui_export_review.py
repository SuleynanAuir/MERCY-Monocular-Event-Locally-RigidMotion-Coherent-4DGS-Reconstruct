"""Review original UI video exports, separately from live/raw-model metrics.

No re-rendering, replacement geometry, sharpening or frame selection. UI labels
and box borders are masked identically in V1, V3 and RGB; SSIM windows touching
those masks are excluded. Lossy video measurements are an archival appendix.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity

from .protocol import save_json, sha256

SOURCES = {
    'lego': 'outputs/video-export/lego-disk/videos',
    'fan': 'outputs/video-export/ui/fan-videos',
    'lamp': 'outputs/lamp-detail-20261025/videos',
}


def layout(meta):
    panel = meta['width']//4
    regions = meta['regions']
    image_height = math.floor(panel*(meta['overview']['bottom']-meta['overview']['top'])+.5)
    if not regions:
        raise ValueError('This archival layout expects the exported detail row')
    dw = meta['width']/(2*len(regions))
    pad = min(12, max(4, dw*.025))
    natural = max(r['h']/r['w'] for r in regions)*(dw-2*pad)+2*pad
    detail = math.ceil(max(2*pad+24, min(natural, dw*1.6, image_height*1.15)))
    header = (meta['height']-image_height-1-detail)//2
    if not 10 <= header <= 120:
        raise ValueError('Unexpected UI export header; inspect source layout')
    return panel, image_height, header


def valid_mask(meta):
    panel, height, header = layout(meta)
    mask = np.ones((height, panel), bool)
    unit = header/28
    # Twelve pixels also exclude codec ringing around colored UI marks. This
    # fixed mask uses only the export metadata, never either model's pixels.
    margin = 12
    def exclude(x0,y0,x1,y1):
        x0,y0 = max(0,math.floor(x0)),max(0,math.floor(y0))
        x1,y1 = min(panel,math.ceil(x1)),min(height,math.ceil(y1))
        if x1>x0 and y1>y0:mask[y0:y1,x0:x1]=False
    for r in meta['regions']:
        x,y = r['x']*panel,(r['y']-meta['overview']['top'])*panel
        w,h = r['w']*panel,r['h']*panel
        for a,b,c,d in ((x,y,x+w,y),(x,y+h,x+w,y+h),(x,y,x,y+h),(x+w,y,x+w,y+h)):
            exclude(a-margin,b-margin,c+margin,d+margin)
        label_y=max(0,y-20*unit)
        exclude(x-margin,label_y-margin,x+24*unit+margin,label_y+20*unit+margin)
    mask[:margin]=False;mask[-margin:]=False
    mask[:,:margin]=False;mask[:,-margin:]=False
    return mask, binary_erosion(mask, structure=np.ones((11,11),bool), border_value=0)


def masked_scores(ref, pred, mask, ssim_mask):
    a,b=ref.astype(np.float32)/255,pred.astype(np.float32)/255
    diff=a-b;mse=float(np.square(diff)[mask].mean())
    _,field=structural_similarity(a,b,channel_axis=2,data_range=1,gaussian_weights=True,
        sigma=1.5,use_sample_covariance=False,full=True)
    return dict(rgb_l1=float(np.abs(diff)[mask].mean()),
                psnr=float(-10*np.log10(mse)) if mse else None,
                ssim=float(field.mean(axis=2)[ssim_mask].mean()))


def run(repo, output):
    output.mkdir(parents=True,exist_ok=False)
    summary=[]
    for scene,relative in SOURCES.items():
        source=repo/relative;meta=json.loads((source/'manifest.json').read_text())
        if any(meta['camera'][k]!=v for k,v in dict(yaw=0,pitch=0,zoom=1,pan_x=0,pan_y=0).items()):
            raise ValueError('Free-view export cannot be scored against the fixed RGB panel')
        panel,height,header=layout(meta);mask,ssim_mask=valid_mask(meta)
        dest=output/scene;dest.mkdir()
        Image.fromarray(mask.astype('uint8')*255).save(dest/'metric_mask.png')
        Image.fromarray(ssim_mask.astype('uint8')*255).save(dest/'ssim_window_mask.png')
        for mode in ('gaussian','clay'):
            videos=list(source.glob('*_'+mode+'.webm'))
            if len(videos)!=1:raise ValueError('Ambiguous original UI export')
            video=videos[0];frames=[];nbytes=meta['width']*meta['height']*3
            process=subprocess.Popen(['ffmpeg','-v','error','-i',str(video),'-f','rawvideo',
                                      '-pix_fmt','rgb24','-vsync','0','pipe:1'],stdout=subprocess.PIPE)
            for i in range(meta['frame_count']):
                data=process.stdout.read(nbytes)
                if len(data)!=nbytes:raise ValueError('Truncated UI export')
                pixels=np.frombuffer(data,np.uint8).reshape(meta['height'],meta['width'],3)
                row=pixels[header:header+height]
                panels=[row[:,k*panel:(k+1)*panel] for k in range(3)]
                frames.append(dict(frame=i,time_seconds=meta['times_seconds'][i],metrics={
                    v:masked_scores(panels[2],panels[j],mask,ssim_mask)
                    for j,v in enumerate(('V1','V3'))}))
                if i in (0,meta['frame_count']//2,meta['frame_count']-1):
                    # Preserve the entire original frame, including detail boxes.
                    Image.fromarray(pixels).save(dest/f'{mode}-original-ui-{i:04d}.png')
                if (i+1)%50==0:print(scene,mode,i+1,flush=True)
            if process.stdout.read(1) or process.wait()!=0:raise ValueError('Unexpected frame count/decoder exit')
            record=dict(scene=scene,mode=mode,source=str(video),source_sha256=sha256(video),
                manifest_sha256=sha256(source/'manifest.json'),source_kind='archived_lossy_ui_export',
                current_deployment_identity_verified=False,frame_count=len(frames),
                layout=dict(panel_width=panel,image_height=height,header=header),
                valid_pixels=int(mask.sum()),ssim_valid_centers=int(ssim_mask.sum()),
                scope='Fixed-front original UI export versus its own RGB panel; same predeclared overlay exclusion mask; no held-out training claim',
                means={v:{key:float(np.mean([f['metrics'][v][key] for f in frames]))
                          for key in ('rgb_l1','psnr','ssim')} for v in ('V1','V3')},frames=frames)
            save_json(dest/(mode+'.json'),record)
            summary.append({k:v for k,v in record.items() if k!='frames'})
    save_json(output/'results.json',dict(scope='Archival UI pixel review only, not a fresh five-scene model experiment',rows=summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.repo.resolve(),args.output)
