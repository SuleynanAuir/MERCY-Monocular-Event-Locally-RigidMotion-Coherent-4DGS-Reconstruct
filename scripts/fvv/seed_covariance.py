"""Exact factorization of surface-and-velocity covariance into Hybrid 4D factors."""
import numpy as np


def static_surface_rotation(normals):
    """Scalar-first 3D quaternions whose local z axis follows each normal."""
    normals=np.asarray(normals,np.float64)
    if normals.ndim!=2 or normals.shape[1]!=3 or not np.allclose(np.linalg.norm(normals,axis=1),1,atol=1e-4):raise ValueError('Invalid static normals')
    axis=np.zeros_like(normals);axis[:,0]=1;axis[np.abs(normals[:,0])>.9]=[0,1,0]
    tangent=np.cross(normals,axis);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    m=np.stack((tangent,np.cross(normals,tangent),normals),axis=-1)
    a,b,c=m[:,0,0],m[:,1,1],m[:,2,2]
    squared=np.stack((1+a+b+c,1+a-b-c,1-a+b-c,1-a-b+c),axis=-1).clip(0)
    candidates=np.stack((
        np.stack((squared[:,0],m[:,2,1]-m[:,1,2],m[:,0,2]-m[:,2,0],m[:,1,0]-m[:,0,1]),axis=-1),
        np.stack((m[:,2,1]-m[:,1,2],squared[:,1],m[:,1,0]+m[:,0,1],m[:,0,2]+m[:,2,0]),axis=-1),
        np.stack((m[:,0,2]-m[:,2,0],m[:,1,0]+m[:,0,1],squared[:,2],m[:,1,2]+m[:,2,1]),axis=-1),
        np.stack((m[:,1,0]-m[:,0,1],m[:,0,2]+m[:,2,0],m[:,1,2]+m[:,2,1],squared[:,3]),axis=-1)),axis=1)
    choice=squared.argmax(-1);rows=np.arange(len(normals))
    return (candidates[rows,choice]/(2*np.sqrt(squared[rows,choice,None]))).astype(np.float32)


def rotation4(left,right):
    a,b,c,d=np.moveaxis(left,-1,0);p,q,r,s=np.moveaxis(right,-1,0)
    l=np.stack((a,-b,-c,-d,b,a,-d,c,c,d,a,-b,d,-c,b,a),axis=-1).reshape(left.shape[:-1]+(4,4))
    rmat=np.stack((p,q,r,s,-q,p,-s,r,-r,s,p,-q,-s,-r,q,p),axis=-1).reshape(right.shape[:-1]+(4,4))
    return (l@rmat)[...,::-1,::-1]


def factor_surface_motion(velocity,normals,tangent=.006,thickness=.0012,temporal=.05):
    velocity=np.asarray(velocity,np.float64);normals=np.asarray(normals,np.float64)
    if velocity.ndim!=2 or velocity.shape[1]!=3 or normals.shape!=velocity.shape or not np.isfinite(velocity).all() or not np.isfinite(normals).all():
        raise ValueError('Invalid surface motion arrays')
    if min(tangent,thickness,temporal)<=0 or not np.allclose(np.linalg.norm(normals,axis=1),1,atol=1e-4):
        raise ValueError('Invalid surface scales or normals')
    spatial=tangent**2*np.eye(3)[None]+(thickness**2-tangent**2)*normals[:,:,None]*normals[:,None,:]
    covariance=np.zeros((len(velocity),4,4),np.float64)
    covariance[:,:3,:3]=spatial+temporal**2*velocity[:,:,None]*velocity[:,None,:]
    covariance[:,:3,3]=velocity*temporal**2;covariance[:,3,:3]=velocity*temporal**2;covariance[:,3,3]=temporal**2
    values,rotation=np.linalg.eigh(covariance)
    if (values<=0).any():raise ValueError('Non-positive seed covariance')
    rotation[:,:,0]*=np.linalg.det(rotation)[:,None]
    basis=np.empty((4,4,4,4))
    for i in range(4):
        for j in range(4):basis[i,j]=rotation4(np.eye(4)[i],np.eye(4)[j])
    coefficients=np.einsum('nab,ijab->nij',rotation,basis)/4
    u,s,vh=np.linalg.svd(coefficients);left=u[:,:,0];right=vh[:,0,:]
    reconstructed=rotation4(left,right)
    if not np.allclose(reconstructed,rotation,atol=1e-7):raise ValueError('Quaternion factorization mismatch')
    recovered=(reconstructed*values[:,None,:])@np.swapaxes(reconstructed,1,2)
    if not np.allclose(recovered,covariance,atol=1e-10):raise ValueError('Covariance factorization mismatch')
    return np.log(np.sqrt(values)).astype(np.float32),left.astype(np.float32),right.astype(np.float32)
