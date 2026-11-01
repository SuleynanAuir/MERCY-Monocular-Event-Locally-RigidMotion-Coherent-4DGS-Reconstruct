"""Fixed-camera, multi-angle old/new motion comparison without frame caches."""
import argparse,json,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image,ImageDraw
import hybrid
from refine_lego import view_camera


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['repo','previous','model','views','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
    views=json.loads((a.views/'views.json').read_text())['views']
    available=[v for v in views if v['split']=='validation']
    chosen=[min(available,key=lambda v:abs(v.get('angle',0)-angle)+abs(v.get('elevation',0))) for angle in [-112.5,-22.5,67.5]]
    models=[hybrid.Renderer(SimpleNamespace(repo=a.repo,model=m,temporal_sigma=1/60)) for m in [a.previous,a.model]]
    summary=dict(scope='Fixed source cameras, continuously advancing model time. Old top/new bottom; no frame interpolation or temporal RGB blending.',views=chosen,models=[dict(core=r.meta['sha256'],sidecar=r.meta['rigid_bucket']['sha256']) for r in models],fps=60,frames=399)
    tile=384;w=3*tile;h=2*(tile+24);tmp=a.output/'comparison.partial.mp4';start=time.monotonic()
    cmd=['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','60','-i','pipe:0','-an','-c:v','libx264','-crf','16','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)]
    with (a.output/'encoder.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        try:
            with torch.no_grad():
                for frame in range(399):
                    canvas=Image.new('RGB',(w,h));draw=ImageDraw.Draw(canvas)
                    for row,r in enumerate(models):
                        for col,view in enumerate(chosen):
                            cam=view_camera(dict(view,time=frame/60));rgb=r.render_fn(cam,r.model,r.pipe,r.bg)['render'].clamp(0,1)
                            rgb=(rgb.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                            canvas.paste(Image.fromarray(rgb).resize((tile,tile),Image.Resampling.LANCZOS),(col*tile,row*(tile+24)+24))
                            draw.text((col*tile+5,row*(tile+24)+4),f'{["Previous","Candidate"][row]} angle={view.get("angle",0)} t={frame/60:.2f}',fill='white')
                    proc.stdin.write(canvas.tobytes())
                    if frame in [0,199,398]:canvas.save(a.output/f'preview-{frame}.jpg',quality=95)
                    if time.monotonic()-start>600 or tmp.exists() and tmp.stat().st_size>128*1024**2:raise RuntimeError('Comparison budget exceeded')
            proc.stdin.close()
            if proc.wait(timeout=60):raise RuntimeError('Encoder failed')
        except BaseException:
            if proc.poll() is None:proc.kill();proc.wait()
            tmp.unlink(missing_ok=True);raise
    tmp.rename(a.output/'comparison.mp4');summary['elapsed']=time.monotonic()-start
    (a.output/'parameters.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
