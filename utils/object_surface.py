"""Deterministic surface-constrained GS resampling from an explicit mesh prior.

Samples are Gaussians, not a substituted image or mesh render. Every sample has
one source rigid part. Appearance is robustly estimated per material/planar patch
to avoid baking a moving part's canonical silhouette into a static surface.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from utils.fan_surface import attributes


def resample_surface(triangles, normals, groups, materials, corrected, *, spacing=.0015):
    tri=np.asarray(triangles,dtype=np.float64);normals=np.asarray(normals,dtype=np.float64)
    groups=np.asarray(groups);materials=np.asarray(materials)
    if (tri.ndim!=3 or tri.shape[1:]!=(3,3) or normals.shape!=tri.shape
            or groups.shape!=(len(tri),) or materials.shape!=groups.shape
            or groups.dtype.kind not in 'iu' or materials.dtype.kind not in 'iu'
            or not np.isfinite(tri).all() or not np.isfinite(normals).all()
            or np.any(np.linalg.norm(normals,axis=2)<.9) or not .0003<=spacing<=.01):
        raise ValueError('Invalid source surface')
    edge1=tri[:,1]-tri[:,0];edge2=tri[:,2]-tri[:,0]
    cross=np.cross(edge1,edge2);area=np.linalg.norm(cross,axis=1)/2
    face_normal=cross/np.maximum(2*area[:,None],1e-20)
    ids=np.flatnonzero(area>1e-15)
    # Low-discrepancy samples avoid random temporal/surface noise. Covariance is
    # tangent to the actual source face; normal thickness is tightly bounded.
    positions=[];directions=[];faces=[]
    for face in ids:
        count=max(4,int(np.ceil(area[face]/spacing**2)))
        k=np.arange(count,dtype=np.float64)+.5
        uv=np.column_stack((k/count,np.mod(k*.6180339887498949,1)))
        root=np.sqrt(uv[:,0]);weights=np.column_stack((1-root,root*(1-uv[:,1]),root*uv[:,1]))
        positions.append(weights@tri[face]);n=weights@normals[face]
        n/=np.linalg.norm(n,axis=1,keepdims=True);directions.append(n);faces.append(np.full(count,face,np.int32))
    if not positions:raise ValueError('Source surface has no area')
    xyz=np.concatenate(positions);normal=np.concatenate(directions);face_ids=np.concatenate(faces)
    labels=groups[face_ids]
    cx,cb,cs,ca,cc=attributes(corrected)
    trusted=(ca>.5)&(cs.max(1)>spacing*.12)
    if trusted.sum()<12:raise ValueError('Not enough supported appearance samples')
    reference_tree=cKDTree(cx[trusted]);reference_colors=cc[trusted]
    # Patch identity includes part, material, face normal and plane offset.
    # Parallel surfaces on different planes and different articulated parts
    # must never blend together.
    planes=np.einsum('ni,ni->n',face_normal,tri.mean(1))
    key=np.column_stack((groups,materials,np.rint(face_normal*200),np.rint(planes/spacing)))
    _,patches=np.unique(key,axis=0,return_inverse=True)
    face_colors=np.zeros((len(tri),3));support=np.zeros(len(tri),bool)
    # Stable grouping preserves the former ascending face/sample order exactly,
    # but avoids scanning every Gaussian for every small planar/material patch.
    # Dense microphone grilles otherwise spend minutes in repeated np.isin.
    face_order=np.argsort(patches,kind='stable')
    face_keys=patches[face_order]
    point_patch=patches[face_ids]
    point_order=np.argsort(point_patch,kind='stable')
    point_keys=point_patch[point_order]
    for patch in np.unique(patches[ids]):
        lo,hi=np.searchsorted(face_keys,[patch,patch+1])
        f=face_order[lo:hi]
        lo,hi=np.searchsorted(point_keys,[patch,patch+1])
        pool=point_order[lo:hi];sample=pool[::max(1,len(pool)//512)]
        # Each patch has at most about 512 query points; repeatedly spawning
        # worker pools costs more than this small nearest-neighbor lookup.
        distance,nearest=reference_tree.query(xyz[sample],k=4,workers=1)
        valid=distance<.04
        colors=reference_colors[nearest][valid]
        if len(colors)>12:
            color=np.median(colors,axis=0);support[f]=True
        else:
            color=np.median(reference_colors[nearest],axis=(0,1))
        face_colors[f]=color
    # Broad canonical illumination makes geometry legible while remaining
    # attached to each part. Additional view lighting is declared separately.
    key_light=np.array([-.4,.8,.7]);key_light/=np.linalg.norm(key_light)
    fill=np.array([.7,.3,-.5]);fill/=np.linalg.norm(fill)
    lighting=.73+.28*np.clip(normal@key_light,0,1)+.10*np.clip(normal@fill,0,1)
    color=np.clip(face_colors[face_ids]*lighting[:,None],0,1)
    dtype=corrected.dtype;result=np.zeros(len(xyz),dtype=dtype)
    for i,k in enumerate(('x','y','z')):result[k]=xyz[:,i]
    # Orient the smallest axis along the smooth source normal.
    helper=np.tile([1.,0.,0.],(len(xyz),1));helper[np.abs(normal[:,0])>.8]=[0,1,0]
    tangent=np.cross(helper,normal);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    bitangent=np.cross(normal,tangent)
    basis=np.stack((tangent,bitangent,normal),axis=2)
    quat=Rotation.from_matrix(basis).as_quat()[:,[3,0,1,2]]
    for i in range(4):result[f'rot_{i}']=quat[:,i]
    # Small triangles use their own footprint, rather than giant splats across
    # a thin gap or neighboring part.
    footprint=np.sqrt(area[face_ids]/np.bincount(face_ids,minlength=len(tri))[face_ids])
    sigma=np.minimum(spacing,footprint)*.90
    for i in range(2):result[f'scale_{i}']=np.log(sigma)
    result['scale_2']=np.log(np.maximum(sigma*.06,1e-7))
    result['opacity']=np.log(.97/.03)
    for i in range(3):result[f'f_dc_{i}']=(color[:,i]-.5)/.28209479177387814
    local_normal=np.einsum('nij,ni->nj',basis,normal).astype(np.float32)
    report=dict(method='Source-surface tangent Gaussians with robust material/planar-patch V3 appearance',
        input_gaussians=len(corrected),output_gaussians=len(result),spacing=spacing,
        planar_patches=int(len(np.unique(patches[ids]))),supported_triangles=int(support.sum()),
        source_triangles=len(tri),motion='Exact source rigid-part ownership; no temporal filtering',
        limitation='Uses supplied Blender geometry as a prior; not new event training or independent reconstruction accuracy')
    return result,labels,local_normal,report
