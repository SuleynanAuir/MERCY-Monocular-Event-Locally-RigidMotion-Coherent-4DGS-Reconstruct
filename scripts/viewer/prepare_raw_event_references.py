"""Frame-address supplied RGB with original x,y,time,polarity events (seconds)."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.viewer.prepare_reference_videos import probe,frames

def prepare(data,stem,output):
    rgb=data/(stem+'.mkv');events=data/(stem+'_event.npy');meta=probe(rgb)
    a=np.load(events,mmap_mode='r',allow_pickle=False)
    if a.ndim!=2 or a.shape[1]!=4 or not np.isfinite(a).all():raise ValueError('Expected finite x,y,t,p events')
    if np.any(np.diff(a[:,2])<0) or not np.isin(a[:,3],[-1,1]).all():raise ValueError('Invalid event time/polarity')
    w,h=meta['width'],meta['height'];xy=a[:,:2]
    if not np.equal(xy,np.floor(xy)).all() or (xy<0).any() or (xy[:,0]>=w).any() or (xy[:,1]>=h).any():raise ValueError('Event pixel bounds mismatch')
    end=(meta['frame_count']-1)/meta['fps']
    if a[0,2]<0 or a[-1,2]>end+1e-6:raise ValueError('Event/RGB duration mismatch')
    output.mkdir(parents=True,exist_ok=False);counts=[]
    for i,im in enumerate(frames(rgb,w,h)):
        t=i/meta['fps'];lo=0 if i==0 else np.searchsorted(a[:,2],(i-1)/meta['fps'],side='right');hi=np.searchsorted(a[:,2],t,side='right')
        ev=a[lo:hi];canvas=np.zeros((h,w,3),np.uint8)
        for sign,color in [(1,[255,80,60]),(-1,[50,130,255])]:
            v=ev[ev[:,3]==sign];canvas[v[:,1].astype(int),v[:,0].astype(int)]=color
        Image.fromarray(np.concatenate([im,canvas],axis=1)).save(output/f'{i:04d}.png');counts.append(len(ev))
    def sha(p):
        s=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024**2),b''):s.update(b)
        return s.hexdigest()
    record=dict(format='rgb_event_reference_v1',**meta,times_seconds=[i/meta['fps'] for i in range(meta['frame_count'])],frame_extension='.png',alignment=dict(space='rgb_image',interval='(previous RGB timestamp, current RGB timestamp]',event_time_unit='seconds',event_frame_offset=0,method='Supplied pixel coordinates and timestamps, no synthesized events',limitation='Shared pixel/time convention; no independent calibration claim'),counts=counts,sources=dict(rgb=dict(sha256=sha(rgb)),event=dict(sha256=sha(events))))
    (output/'reference.json').write_text(json.dumps(record,indent=2));return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--stem',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.data,a.stem,a.output)
