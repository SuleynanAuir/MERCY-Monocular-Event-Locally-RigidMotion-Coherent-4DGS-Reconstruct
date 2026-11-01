"""Candidate-only bounded temporal color regularization; never deploys itself."""
import argparse,json,os
from pathlib import Path
import numpy as np
from rigid_bucket import read_arrays
from common import digest,model_timestamp,usage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);meta=json.loads((a.model/'manifest.json').read_text());v=read_arrays(a.model/'rigid-bucket.npz')
    if meta['rigid_bucket'].get('allface_train_views')!=96:raise ValueError('Requires four-time all-face supervision')
    temporal=v['temporal_dc'];knots=len(temporal)
    anchors=np.array([0,99,299,398])/398*(knots-1)
    def at(t):
        i=min(int(t),knots-2);f=t-i;return (1-f)*temporal[i]+f*temporal[i+1]
    values=np.stack([at(t) for t in anchors]);smooth=[]
    for t in range(knots):
        i=min(max(np.searchsorted(anchors,t,side='right')-1,0),2);f=(t-anchors[i])/(anchors[i+1]-anchors[i]);smooth.append((1-f)*values[i]+f*values[i+1])
    v['temporal_dc']=(.5*temporal+.5*np.stack(smooth)).astype(np.float32)
    np.savez_compressed(a.output/'rigid-bucket.npz',**v);read_arrays(a.output/'rigid-bucket.npz')
    for name in ['model.pth','effective.yaml']:os.link(a.model/name,a.output/name)
    source_sha=meta['rigid_bucket']['sha256'];meta['rigid_bucket']['sha256']=digest(a.output/'rigid-bucket.npz')
    info=dict(kind='temporal_color_regularization',strength=.5,anchor_frames=[0,99,299,398],source_bucket_sha256=source_sha,source_trained_at=meta['trained_at'],scope='Uniform temporal color interpolation candidate; geometry and base model unchanged; image and temporal validation required')
    meta['rigid_bucket']['color_regularization']=info;meta['trained_at']=model_timestamp();meta['bytes']=usage(a.output)
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2));(a.output/'color-regularization.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))


if __name__=='__main__':main()
