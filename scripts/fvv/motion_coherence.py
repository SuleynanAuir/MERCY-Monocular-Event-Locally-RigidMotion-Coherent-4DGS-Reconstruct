"""Same-surface temporal appearance coherence; does not constrain rigid motion.

Only differences of temporal color changes are regularized. Static texture and
shared lighting changes are not forced to be constant across space or time.
"""
import numpy as np


def surface_pairs(points,normals,parts):
    from scipy.spatial import cKDTree
    distance,near=cKDTree(points).query(points,k=min(9,len(points)),workers=4)
    left=np.repeat(np.arange(len(points)),near.shape[1]-1);right=near[:,1:].reshape(-1)
    offset=points[left]-points[right]
    valid=(left<right)&(distance[:,1:].reshape(-1)<.007)&(parts[left]==parts[right])
    valid&=((normals[left]*normals[right]).sum(-1)>.95)
    valid&=(np.abs((offset*normals[left]).sum(-1))<.0007)&(np.abs((offset*normals[right]).sum(-1))<.0007)
    return np.stack([left[valid],right[valid]],axis=1).astype(np.int64)


def temporal_neighbor_loss(temporal,pairs):
    changes=temporal[1:]-temporal[:-1]
    left,right=pairs.unbind(1)
    return (changes[:,left]-changes[:,right]).square().mean()
