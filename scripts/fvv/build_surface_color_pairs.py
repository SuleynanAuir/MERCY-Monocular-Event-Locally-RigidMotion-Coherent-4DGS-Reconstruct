"""Prepare bounded same-part, same-time source-surface appearance neighbors."""
import argparse,json,time
from pathlib import Path
import numpy as np
from common import digest
from surface_color_regularization import build_pairs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    start=time.monotonic()
    with np.load(a.source,allow_pickle=False) as data:
        points=data['points'];normals=data['normals'];parts=data['parts'];times=data['times']
    count=len(points)
    if not 1<=count<=1048576 or points.shape!=(count,3) or normals.shape!=points.shape or parts.shape!=(count,) or times.shape!=(count,1):raise ValueError('Invalid bounded source')
    if not all(np.isfinite(x).all() for x in [points,normals,times]) or not np.allclose(np.linalg.norm(normals,axis=1),1,atol=1e-4):raise ValueError('Invalid source geometry')
    pairs,weights=build_pairs(points,normals,parts,times)
    if not 1<=len(pairs)<=count:raise ValueError('Unexpected neighbor count')
    if time.monotonic()-start>300:raise TimeoutError('Appearance neighbor preparation budget')
    metadata=dict(source_sha256=digest(a.source),dynamic_points=count,pairs=len(pairs),radius=.008,normal_cosine_min=.98,plane_tolerance=.0008,
        scope='Source surface appearance neighbors at the same time and part; no geometry or image-space filtering',elapsed_seconds=time.monotonic()-start)
    temporary=a.output.with_suffix('.partial.npz')
    try:
        np.savez_compressed(temporary,pairs=pairs,weights=weights,metadata_json=np.asarray(json.dumps(metadata)))
        if temporary.stat().st_size>32*1024**2:raise OSError('Appearance neighbors exceed 32 MiB')
        temporary.replace(a.output)
    finally:temporary.unlink(missing_ok=True)
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
