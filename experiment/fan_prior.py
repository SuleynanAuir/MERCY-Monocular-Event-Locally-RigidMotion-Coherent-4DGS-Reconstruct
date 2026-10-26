"""Training-only rotor identification; constant angular velocity extrapolation."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from .supervision import read_rgb
from .prepare import save_prior


def build(scene,destination,vertices):
    from scripts.lamp.prepare_lamp_viewer import camera_pose
    from utils.fan_motion_binding import refine_static_membership
    camera=json.loads(Path(scene['files']['camera']['path']).read_text())
    train=scene['split']['frames']['train'];end=train[-1]
    # Decode stops at the training boundary; validation/test frames never enter
    # rotor center, segmentation, or angular velocity identification.
    frames=[]
    for i,rgb in enumerate(read_rgb(scene['files']['rgb']['path'])):
        if i>end:break
        frames.append(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255)
    xyz=np.column_stack([vertices[k] for k in ('x','y','z')])
    w=np.eye(4);w[:3,:3]=np.asarray(camera['R']).T;w[:3,3]=camera['T']
    k=np.array([[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1]])
    def project(points):
        p=points@w[:3,:3].T+w[:3,3];uv=p@k.T
        return (uv[:,:2]/uv[:,2:]).astype(np.float32)
    front=(project(xyz)[:,0]>320)&(xyz[:,2]>0)&(xyz[:,2]<.16)
    plane=float(np.median(xyz[front,2]))
    angles=np.arange(1440)*2*np.pi/1440;radii=np.linspace(.10,.285,80)
    xx=np.cos(angles)[:,None]*radii;yy=np.sin(angles)[:,None]*radii
    def polar(frame,center):
        p=np.stack([xx+center[0],yy+center[1],np.full_like(xx,plane)],-1).reshape(-1,3)
        uv=project(p).reshape(1440,80,2)
        a=cv2.remap(frame,uv[:,:,0],uv[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        return a-a.mean(0,keepdims=True)
    def fit(center,subset=None):
        base=polar(frames[0],center);fft=np.fft.rfft(base,axis=0);theta=[];errors=[]
        for frame in frames if subset is None else subset:
            current=polar(frame,center)
            corr=np.fft.irfft(np.conj(fft)*np.fft.rfft(current,axis=0),n=1440,axis=0).sum(1)
            if subset is None and theta:
                ids=(int(round(theta[-1]*1440/(2*np.pi)))+np.arange(-120,121))%1440
                pos=int(np.argmax(corr[ids]))
                if pos in (0,len(ids)-1):raise ValueError('Rotor motion ambiguity at training frame')
                j=int(ids[pos])
            else:j=int(corr.argmax())
            a,b=corr[(j-1)%1440],corr[(j+1)%1440]
            sub=.5*(a-b)/(a-2*corr[j]+b+1e-12)
            theta.append((j+sub)*2*np.pi/1440)
            errors.append(float(((base*base).sum()+(current*current).sum()-2*corr[j])/base.size))
        return np.unwrap(theta),float(np.mean(errors))
    subset=[frames[i] for i in np.linspace(0,end,8).astype(int)]
    optimum=minimize(lambda c:fit(c,subset)[1],[.23,.308],method='Nelder-Mead',
        options={'maxiter':70,'xatol':1e-5,'fatol':1e-7})
    observed,fit_error=fit(optimum.x);observed-=observed[0]
    # Through-origin constant velocity is fixed a priori, not selected by test.
    times=np.arange(len(observed))/scene['video']['fps']
    velocity=float(times@observed/(times@times))
    theta=np.arange(scene['video']['frame_count'])/scene['video']['fps']*velocity
    center=np.array([*optimum.x,plane]);moving=front&(np.linalg.norm(xyz[:,:2]-center[:2],axis=1)<.31)
    moving,_,repair=refine_static_membership(xyz,moving,center,theta[:len(frames)],frames,camera)
    repair['scope']='All membership selection/check frames are inside the training block only.'
    delta=np.broadcast_to(np.eye(4),(len(theta),2,4,4)).copy()
    delta[:,1,:3,:3]=Rotation.from_rotvec(np.column_stack([np.zeros_like(theta),np.zeros_like(theta),theta])).as_matrix()
    delta[:,1,:3,3]=center-np.einsum('tij,j->ti',delta[:,1,:3,:3],center)
    return save_prior(scene,destination,scene['files']['corrected']['path'],xyz,moving.astype(np.int32),
        delta,np.stack([xyz[~moving].mean(0),center]),camera_pose(camera),
        [dict(groups=[0,1],pivot=center.tolist(),axis=[0.,0.,1.])],np.array([],dtype=np.int64),
        dict(prior='training RGB polar correlation + constant velocity extrapolation; no Blender available',
             training_frames=train,training_angle_rmse=float(np.sqrt(np.mean((theta[:len(observed)]-observed)**2))),
             observed_training_angles_rad=observed.tolist(),
             angular_velocity_rad_s=velocity,polar_fit_mse=fit_error,center_fit_converged=bool(optimum.success),
             membership=repair,joint='assumed world-Z rotor shaft, not independently measured',
             event_supervision=False))
