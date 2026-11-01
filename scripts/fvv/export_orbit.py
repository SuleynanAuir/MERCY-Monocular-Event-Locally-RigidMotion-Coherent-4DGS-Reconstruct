"""CPU-only client for the UI's bounded offline orbit export service.

python -m scripts.fvv.export_orbit --scene lego --output /tmp/lego-orbit
"""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlencode,urlsplit
from urllib.request import Request,urlopen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:20896/fvv')
    p.add_argument('--scene',default='lego');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--path',choices=['overview','showcase','full'],help='Lego defaults to a calibrated all-around overview; full preserves the 360-degree route')
    p.add_argument('--seconds',type=int,choices=[6,12,24],default=12)
    p.add_argument('--fps',type=int,choices=[30,60],default=30)
    p.add_argument('--width',type=int,choices=[640,960,1280],default=960)
    p.add_argument('--radius',type=float,default=1.5);p.add_argument('--elevation',type=float,default=15)
    p.add_argument('--pose-json',type=Path,help='Optional position/yaw/pitch/time/fov/temporal_samples JSON')
    a=p.parse_args();u=urlsplit(a.url)
    if u.scheme!='http' or u.hostname not in ['127.0.0.1','localhost','::1']:p.error('Use a local UI URL through your SSH tunnel')
    a.output.mkdir(parents=True,exist_ok=False)
    base=a.url.rstrip('/')+'/api/orbit-export';query='?'+urlencode({'scene':a.scene})
    def api(url,data=None):
        req=Request(url,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=30) as r:return json.load(r)
    data=json.loads(a.pose_json.read_text()) if a.pose_json else {}
    data.update(seconds=a.seconds,fps=a.fps,width=a.width,radius=a.radius,elevation=a.elevation)
    if a.path:data['path']=a.path
    job=api(base+query,data);ident=job['id'];deadline=time.monotonic()+1300
    while job['state']=='running':
        if time.monotonic()>deadline:raise TimeoutError('Export polling timed out; inspect the UI job before retrying')
        print(f"{job['frame']}/{job['frames']} frames",flush=True);time.sleep(1)
        job=api(base+query)['job']
        if not job or job['id']!=ident:raise RuntimeError('Export job changed; inspect the UI history')
    if job['state']!='complete':raise RuntimeError(job.get('error','Export failed'))
    for kind,name in [('video','orbit.mp4'),('parameters','parameters.json')]:
        partial=a.output/(name+'.partial')
        with urlopen(base+'/'+ident+'/'+kind+query,timeout=60) as r,partial.open('xb') as f:
            while chunk:=r.read(256*1024):f.write(chunk)
        partial.rename(a.output/name)
    print(a.output.resolve())


if __name__=='__main__':main()
