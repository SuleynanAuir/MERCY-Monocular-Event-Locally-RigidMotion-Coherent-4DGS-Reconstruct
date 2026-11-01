"""Offline supersampled full-orbit video; never modifies model or serving code."""
import argparse,json,os,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
import hybrid
from orbit_export import trajectory
from common import digest,usage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['repo','model','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seconds',type=int,choices=[6,12,24],default=24)
    p.add_argument('--width',type=int,choices=[1280,1920],default=1920)
    p.add_argument('--scale',type=float,choices=[1.5,2.],default=2.)
    p.add_argument('--probe',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if usage(a.output.parent)>2*1024**3:raise OSError('Export experiment exceeds 2 GiB')
    torch.set_num_threads(4)
    r=hybrid.Renderer(SimpleNamespace(repo=a.repo,model=a.model,temporal_sigma=1/60))
    plan=trajectory(dict(path='overview',seconds=a.seconds,fps=60,width=1280,temporal_samples=1),r.meta)
    height=a.width*9//16;render_width=int(a.width*a.scale);render_height=int(height*a.scale)
    def render(pose):
        high=r.frame(dict(pose,width=render_width,height=render_height,temporal_samples=1),raw=True)
        low=torch.nn.functional.interpolate(high[None],size=(height,a.width),mode='area')[0]
        return (low.clamp(0,1).permute(1,2,0)*255).round().byte().numpy()
    summary=dict(model_sha256=r.meta['sha256'],sidecar_sha256=r.meta['rigid_bucket']['sha256'],trained_at=r.meta['trained_at'],
        model_directory=str(a.model.resolve()),render_resolution=[render_width,render_height],output_resolution=[a.width,height],
        spatial_sampling='Render actual Gaussian projection at higher resolution then area downsample; no AI upscaling or sharpening',
        temporal_samples=1,temporal_sigma=1/60,fps=60,seconds=a.seconds,codec='H.264',crf=16,
        source_code_sha256=digest(Path(__file__)),trajectory=dict({k:v for k,v in plan.items() if k!='poses'},width=a.width,height=height))
    start=time.monotonic()
    first=render(plan['poses'][0]);Image.fromarray(first).save(a.output/'preview-0000.png')
    summary['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    if a.probe:
        summary['probe_only']=True;(a.output/'parameters.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));return
    tmp=a.output/'orbit.partial.mp4';proc=None
    try:
        command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{a.width}x{height}',
            '-r','60','-i','pipe:0','-an','-c:v','libx264','-preset','slow','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',
            '-metadata','comment='+json.dumps(summary),str(tmp)]
        with (a.output/'encoder.log').open('w') as log:
            proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
            for i,pose in enumerate(plan['poses']):
                if time.monotonic()-start>1200:raise TimeoutError('Export exceeds 20 minutes')
                pixels=first if i==0 else render(pose)
                proc.stdin.write(pixels.tobytes())
                if i in [len(plan['poses'])//4,len(plan['poses'])//2,3*len(plan['poses'])//4]:Image.fromarray(pixels).save(a.output/f'preview-{i:04d}.png')
                if tmp.exists() and tmp.stat().st_size>256*1024**2:raise OSError('Video exceeds 256 MiB')
                if i%60==0:
                    status=dict(frame=i+1,frames=len(plan['poses']),elapsed=time.monotonic()-start)
                    (a.output/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
            proc.stdin.close()
            if proc.wait(timeout=120):raise RuntimeError('Encoder failed')
        if tmp.stat().st_size>256*1024**2:raise OSError('Video exceeds 256 MiB')
        if digest(a.model/'model.pth')!=summary['model_sha256'] or digest(a.model/'rigid-bucket.npz')!=summary['sidecar_sha256']:raise RuntimeError('Source model changed')
        summary.update(elapsed_seconds=time.monotonic()-start,frames=[dict(index=i,video_time=i/60,**dict(pose,width=a.width,height=height)) for i,pose in enumerate(plan['poses'])],gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        (a.output/'parameters.json').write_text(json.dumps(summary,indent=2))
        tmp.rename(a.output/'orbit.mp4');(a.output/'status.json').write_text(json.dumps(dict(state='complete',frames=len(plan['poses']))))
    except BaseException:
        if proc is not None and proc.poll() is None:proc.kill();proc.wait()
        tmp.unlink(missing_ok=True);raise


if __name__=='__main__':main()
