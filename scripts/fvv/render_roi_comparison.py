"""Small, bounded A/B clip from actual model renders at off-grid timestamps."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import subprocess
import math
import numpy as np
import torch
from PIL import Image,ImageDraw
import hybrid


def run(a):
    torch.set_num_threads(4)
    renderers=[hybrid.Renderer(SimpleNamespace(repo=a.repo.resolve(),model=m.resolve(),temporal_sigma=1/60)) for m in [a.previous,a.model]]
    proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
        '-s','800x420','-r','60','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(a.output)],stdin=subprocess.PIPE)
    try:
        # 3-second clip; offset by half a training frame to test interpolation.
        for i in range(360 if a.dolly else 180):
            timestamp=(.2 if a.dolly else 2)+(i+.5)/60
            distance=1-math.cos(i*2*math.pi/360) if a.dolly else 0
            panel=Image.new('RGB',(800,420))
            for j,r in enumerate(renderers):
                p=dict(position=[0,0,-distance],yaw=0,pitch=0,fov=r.meta['fov'],width=640,height=640,time=timestamp)
                out=r.frame(p,raw=True).clamp(0,1)
                im=Image.fromarray((out.permute(1,2,0).numpy()*255).astype(np.uint8))
                panel.paste(im.resize((400,400)) if a.dolly else im.crop((40,40,440,440)),(j*400,20))
            draw=ImageDraw.Draw(panel)
            draw.text((5,4),'Previous FVV',fill='white');draw.text((405,4),f'Refined  t={timestamp:.3f}s  dolly={distance:.2f}',fill='white')
            proc.stdin.write(panel.tobytes())
        proc.stdin.close()
        if proc.wait(timeout=30):raise RuntimeError('Comparison encoding failed')
    finally:
        if proc.poll() is None:proc.kill();proc.wait()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['repo','previous','model','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--dolly',action='store_true',help='Six-second full-frame actual camera pullback, distance 0..2')
    run(p.parse_args())
