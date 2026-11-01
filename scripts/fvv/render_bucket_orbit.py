"""Bounded isolated whole-bucket 360-degree diagnostic comparison video."""
import argparse,json,math,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image,ImageDraw
import hybrid
from rigid_bucket import bucket_only_model,pose_at,quaternion_rotate
from refine_lego import view_camera


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['repo','previous','model','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    models=[hybrid.Renderer(SimpleNamespace(repo=a.repo,model=m)) for m in [a.previous,a.model]]
    if any(r.bucket is None for r in models):raise ValueError('Two rigid bucket models required')
    for r in models:
        r.bucket=r.bucket.head_only();r.render_fn=r.bucket.wrap(r.base_render_fn)
    v=models[-1].bucket.values;center=(v['points'].min(0).values+v['points'].max(0).values)/2
    radius=float((v['points']-center).norm(dim=-1).max());fov=float(models[-1].meta['fov']);distance=radius/math.sin(math.radians(fov)/2)*1.25
    a.output.parent.mkdir(parents=True,exist_ok=True)
    cmd=['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','768x408','-r','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(a.output)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE);trajectory=[]
    try:
        with torch.no_grad():
            for frame in range(540):
                band=frame//180;phase=(frame%180)/179;timestamp=phase*398/60;elevation=[-65.,0.,65.][band];angle=-180+360*phase
                q,t=pose_at(v['track_rotations'],v['track_translations'],timestamp);target=quaternion_rotate(q,center[None])[0]+t
                theta,phi=map(math.radians,[angle,elevation]);offset=torch.tensor([[math.sin(theta)*math.cos(phi),-math.sin(phi),-math.cos(theta)*math.cos(phi)]],device='cuda')
                position=target+quaternion_rotate(q,offset)[0]*distance;forward=(target-position)/distance;down=quaternion_rotate(q,torch.tensor([[0.,1.,0.]],device='cuda'))[0]
                right=torch.nn.functional.normalize(torch.cross(down,forward,dim=-1),dim=-1);down=torch.cross(forward,right,dim=-1)
                pose=torch.eye(4);pose[:3,:3]=torch.stack([right,down,forward],dim=1).cpu();pose[:3,3]=position.cpu()
                view=dict(c2w=pose.tolist(),time=timestamp,fov=fov);trajectory.append(dict(**view,azimuth=angle,elevation=elevation))
                panel=Image.new('RGB',(768,408));draw=ImageDraw.Draw(panel)
                for col,r in enumerate(models):
                    rgb=r.render_fn(view_camera(view),bucket_only_model(r.model),r.pipe,r.bg)['render'].clamp(0,1).permute(1,2,0).cpu().numpy()
                    panel.paste(Image.fromarray((rgb*255).astype('uint8')).resize((384,384)),(col*384,24))
                    draw.text((col*384+4,4),f'{["Previous","Candidate"][col]} isolated bucket | elevation {elevation:g}',fill='white')
                process.stdin.write(np.array(panel).tobytes())
                if frame%60==0:print('FRAME',frame,flush=True)
        process.stdin.close()
        if process.wait(timeout=60):raise RuntimeError('Video encoding failed')
    except BaseException:
        process.kill();process.wait();raise
    if a.output.stat().st_size>64*1024**2:raise OSError('Video budget exceeded')
    a.output.with_suffix('.json').write_text(json.dumps(dict(scope='Isolated whole-bucket diagnostic, scene occluders disabled ONLY for this video. Not full-scene output or browser FPS.',fps=30,frames=540,duration_seconds=18,models=[dict(sha256=r.meta['sha256'],bucket_sha256=r.meta['rigid_bucket']['sha256']) for r in models],trajectory=trajectory),indent=2))


if __name__=='__main__':main()
