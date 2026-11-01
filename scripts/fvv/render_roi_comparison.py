"""Small, bounded A/B clip from actual model renders at off-grid timestamps."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import subprocess
import math
import json
import numpy as np
import torch
from PIL import Image,ImageDraw
import hybrid


def run(a):
    torch.set_num_threads(4)
    renderers=[hybrid.Renderer(SimpleNamespace(repo=a.repo.resolve(),model=m.resolve(),temporal_sigma=1/60)) for m in [a.previous,a.model]]
    plan=None
    if a.orbit or a.showcase or a.overview:
        from orbit_export import trajectory
        plan=trajectory(dict(path='overview' if a.overview else 'showcase' if a.showcase else 'full',seconds=24 if a.overview else 12,fps=30,width=640,radius=1.8,elevation=15,
                             position=[0,0,-.3],yaw=0,pitch=0,time=.1,temporal_samples=1),renderers[0].meta)
    panel_width=a.panel_width
    image_height=round(panel_width*9/16) if plan else 400
    height=image_height+20
    proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{panel_width*2}x{height+height%2}','-r','30' if plan else '60','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(a.output)],stdin=subprocess.PIPE)
    try:
        # 3-second clip; offset by half a training frame to test interpolation.
        for i in range(plan['frames'] if plan else 360 if a.dolly else 180):
            timestamp=(.2 if a.dolly else 2)+(i+.5)/60
            distance=1-math.cos(i*2*math.pi/360) if a.dolly else 0
            panel=Image.new('RGB',(panel_width*2,height+height%2))
            for j,r in enumerate(renderers):
                p=plan['poses'][i] if plan else dict(position=[0,0,-distance],yaw=0,pitch=0,fov=r.meta['fov'],width=640,height=640,time=timestamp)
                out=r.frame(p,raw=True).clamp(0,1)
                im=Image.fromarray((out.permute(1,2,0).numpy()*255).astype(np.uint8))
                panel.paste(im.resize((panel_width,image_height)) if plan else im.resize((panel_width,400)) if a.dolly else im.crop((150,100,550,500) if a.arm else (40,40,440,440)).resize((panel_width,400)),(j*panel_width,20))
            draw=ImageDraw.Draw(panel)
            draw.text((5,4),'Previous FVV',fill='white');draw.text((panel_width+5,4),f'Refined  t={p["time"]:.3f}s',fill='white')
            proc.stdin.write(panel.tobytes())
        proc.stdin.close()
        if proc.wait(timeout=30):raise RuntimeError('Comparison encoding failed')
        if plan:
            a.output.with_suffix('.json').write_text(json.dumps(dict(trajectory=plan,
                previous_sha256=renderers[0].meta['sha256'],candidate_sha256=renderers[1].meta['sha256'],
                candidate_rigid_bucket=renderers[1].meta.get('rigid_bucket'),
                temporal_sigma=1/60),indent=2))
    finally:
        if proc.poll() is None:proc.kill();proc.wait()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['repo','previous','model','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--panel-width',type=int,choices=[400,640],default=400)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--arm',action='store_true',help='Articulated arm close-up at interpolated timestamps')
    mode.add_argument('--dolly',action='store_true',help='Six-second full-frame actual camera pullback, distance 0..2')
    mode.add_argument('--orbit',action='store_true',help='Twelve-second full 360-degree orbit with elevation, paired poses and parameter sidecar')
    mode.add_argument('--showcase',action='store_true',help='Twelve-second calibrated shallow Lego arc, paired poses and parameter sidecar')
    mode.add_argument('--overview',action='store_true',help='Same 24-second full overview route as the UI export')
    run(p.parse_args())
