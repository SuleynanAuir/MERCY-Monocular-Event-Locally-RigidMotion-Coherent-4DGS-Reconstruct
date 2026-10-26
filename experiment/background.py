"""Freeze a scene's constant background from the first training frame only."""
import json
from pathlib import Path
import numpy as np


def estimate(frame):
    frame=np.asarray(frame)
    if frame.ndim!=3 or frame.shape[-1]!=3 or frame.dtype!=np.uint8:
        raise ValueError('Background estimation expects decoded uint8 RGB')
    border=np.concatenate([frame[:8].reshape(-1,3),frame[-8:].reshape(-1,3),
                           frame[:,:8].reshape(-1,3),frame[:,-8:].reshape(-1,3)])
    color=np.median(border,axis=0)
    deviation=float(np.quantile(np.max(abs(border.astype(float)-color),axis=1),.999))
    if deviation>4:raise ValueError('Nonuniform image border: constant background protocol not applicable')
    return dict(rgb=(color/255).tolist(),source='median of 8-pixel border of first training RGB frame',
                border_q999_deviation_uint8=deviation,used_validation_or_test=False)


def training_background(prepared):
    directory=Path(prepared)/'supervision'
    meta=json.loads((directory/'metadata.json').read_text())
    frame=meta['split']['frames']['train'][0]
    record=estimate(np.load(directory/'rgb.npy',mmap_mode='r')[frame])
    record['source_frame']=frame
    return record


def foreground(rgb,background):
    return np.abs(np.asarray(rgb)-np.asarray(background,dtype=np.float32)).mean(-1)>8/255
