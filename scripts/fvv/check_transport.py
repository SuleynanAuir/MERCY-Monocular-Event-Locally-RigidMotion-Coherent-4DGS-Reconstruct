"""Compare actual GPU renders through JPEG quality and lossless entropy coding."""
import argparse
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import hybrid


def run(args):
    torch.set_num_threads(4)
    renderer=hybrid.Renderer(SimpleNamespace(repo=args.repo.resolve(),model=args.model.resolve(),temporal_sigma=1/60))
    rows=[]
    for timestamp in [1.,2.5,4.,5.5]:
        for distance in [0.,.75]:
            pose=dict(position=[0,0,-distance],yaw=0,pitch=0,fov=renderer.meta['fov'],width=960,height=640,time=timestamp)
            raw=renderer.frame(pose,raw=True).clamp(0,1)
            target=(raw*255).byte().permute(1,2,0).contiguous().numpy()
            foreground=target.mean(2)>8
            for quality in [92,95,98]:
                decoded=[]
                for optimized in [False,True]:
                    stream=io.BytesIO();start=time.perf_counter()
                    Image.fromarray(target).save(stream,format='JPEG',quality=quality,subsampling=0,optimize=optimized)
                    elapsed=(time.perf_counter()-start)*1000
                    with Image.open(io.BytesIO(stream.getvalue())) as image:pixels=np.array(image)
                    decoded.append(pixels)
                    error=(pixels.astype(np.float32)-target.astype(np.float32))/255
                    rows.append(dict(time=timestamp,distance=distance,quality=quality,optimize=optimized,
                        bytes=len(stream.getvalue()),encode_ms=elapsed,
                        foreground_psnr=float(-10*np.log10(np.square(error[foreground]).mean()))))
                if not np.array_equal(*decoded):raise AssertionError('Entropy optimization changed decoded pixels')
    report=dict(model_sha256=renderer.meta['sha256'],scope='Eight actual GPU renders; compression fidelity relative to identical raw render, not scene reconstruction quality',
        optimized_decoded_pixels_identical=True,rows=rows,means={})
    for quality in [92,95,98]:
        for optimized in [False,True]:
            subset=[r for r in rows if r['quality']==quality and r['optimize']==optimized]
            report['means'][f'{quality}-optimized-{optimized}']={key:float(np.mean([r[key] for r in subset])) for key in ['bytes','encode_ms','foreground_psnr']}
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report['means'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','model','output']:parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
