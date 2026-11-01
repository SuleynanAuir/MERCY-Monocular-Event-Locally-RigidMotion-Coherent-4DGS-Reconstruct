"""Local appearance consistency on the same explicit moving source surface."""
import numpy as np


def build_pairs(points,normals,parts,times):
    """Never couple different times/parts, opposite normals or parallel walls."""
    pairs=[];weights=[];times=np.asarray(times).reshape(-1)
    for timestamp in np.unique(times):
        at_time=np.flatnonzero(times==timestamp)
        for part in np.unique(parts[at_time]):
            rows=at_time[parts[at_time]==part]
            if len(rows)<2:continue
            p=points[rows];n=normals[rows]
            for start in range(0,len(rows),256):
                end=min(start+256,len(rows));delta=p[start:end,None]-p[None]
                squared=(delta*delta).sum(-1)
                valid=(n[start:end]@n.T)>.98
                valid &= np.abs((delta*n[start:end,None]).sum(-1))<.0008
                valid &= np.abs((delta*n[None]).sum(-1))<.0008
                valid &= squared<.008**2
                valid &= rows[start:end,None]!=rows[None]
                distance=np.where(valid,squared,np.inf);nearest=distance.argmin(1)
                best=distance[np.arange(end-start),nearest];keep=np.isfinite(best)
                pairs.extend(zip(rows[start:end][keep],rows[nearest[keep]]))
                weights.extend(np.exp(-best[keep]/(2*.004**2)).tolist())
    if not pairs:return np.empty((0,2),np.uint32),np.empty(0,np.float32)
    pairs=np.sort(np.asarray(pairs,np.uint32),axis=1);weights=np.asarray(weights,np.float32)
    _,indices=np.unique(pairs,axis=0,return_index=True)
    return pairs[indices],weights[indices]


def surface_color_loss(dc,rest,pairs,weights):
    """Regularize source-neighbor coefficients, without image-space smoothing."""
    a,b=pairs.unbind(-1)
    dc_error=(dc[a]-dc[b]).square().sum((1,2))
    rest_error=(rest[a]-rest[b]).square().sum((1,2))
    # Keep view-dependent variation more flexible than the base color.
    return (.28209479177387814**2/3)*((dc_error+.1*rest_error)*weights).sum()/weights.sum().clamp_min(1e-8)
