"""Unified, bounded motion adapters. Canonical appearance stays immutable."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from utils.event_refinement import RigidMotionCorrection,qmul,qconj,qrotate


def rotvec_quat(v):
    theta=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    return torch.cat([torch.cos(theta/2),.5*torch.sinc(theta/(2*torch.pi))*v],-1)


class SceneAlignment(RigidMotionCorrection):
    def __init__(self,diagonal):
        super().__init__();self.diagonal=diagonal

    def parameters_physical(self,index=None):
        q,t,s=super().parameters_physical(index)
        return q,t*(self.diagonal/3),s


class Adapter(nn.Module):
    def __init__(self,prepared,method,seed=17):
        super().__init__();prepared=Path(prepared)
        self.metadata=json.loads((prepared/'prior.json').read_text())
        self.method=method;self.diagonal=self.metadata['diagonal']
        torch.manual_seed(seed)
        with np.load(prepared/'motion.npz',allow_pickle=False) as z:
            for key in ('groups','q','t','centers','envelope','active','sample_indices'):
                value=z[key];value=value.astype(np.int64) if key in ('groups','sample_indices') else value
                self.register_buffer(key,torch.from_numpy(value.copy()))
        self.alignment=SceneAlignment(self.diagonal)
        self.mlp=nn.Sequential(nn.Linear(13,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,6))
        nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)
        joints=self.metadata['joints']
        self.register_buffer('joint_groups',torch.tensor([x['groups'] for x in joints],dtype=torch.long).reshape(-1,2))
        self.register_buffer('pivots',torch.tensor([x['pivot'] for x in joints],dtype=torch.float32).reshape(-1,3))
        self.register_buffer('axes',torch.tensor([x['axis'] for x in joints],dtype=torch.float32).reshape(-1,3))
        self.gain=1.
        self.alignment.requires_grad_(method!='C1');self.mlp.requires_grad_(method=='C3')

    def transforms(self,frames):
        q,t=self.q[frames],self.t[frames]
        if self.method=='C1':return q,t
        a,b,s=self.alignment.parameters_physical()
        r=qmul(qmul(a,q),qconj(a))
        u=s*qrotate(a,t)+b-qrotate(r,b)
        if self.method!='C3':return r,u
        center=(self.centers-self.centers.mean(0))/self.diagonal
        center=center.expand(*q.shape[:-1],3)
        time=frames.to(q.dtype)/(len(self.q)-1)
        time=time[...,None,None].expand(*q.shape[:-1],1)
        features=torch.cat([center,q,t/self.diagonal,time,
                            torch.sin(time*torch.pi),torch.cos(time*torch.pi)],-1)
        residual=torch.tanh(self.mlp(features))*self.envelope[frames,...,None]*self.gain
        local=rotvec_quat(residual[...,:3]*(.5*torch.pi/180))
        offset=residual[...,3:]*(.001*self.diagonal)
        world_center=qrotate(r,self.centers)+u
        return qmul(local,r),qrotate(local,u-world_center)+world_center+offset

    def joint_coordinates(self,q,t):
        if not len(self.pivots):return q.new_empty((*q.shape[:-2],0)),q.new_empty((*q.shape[:-2],0))
        left,right=self.joint_groups.unbind(-1)
        gap=torch.linalg.vector_norm(qrotate(q[...,left,:],self.pivots)+t[...,left,:]
             -qrotate(q[...,right,:],self.pivots)-t[...,right,:],dim=-1)
        relative=qmul(qconj(q[...,left,:]),q[...,right,:])
        # Twist angle around the source-rest revolute axis; unwrap at evaluation.
        angle=2*torch.atan2((relative[...,1:]*self.axes).sum(-1),relative[...,0])
        return gap,angle


class SceneRenderer:
    def __init__(self,prepared,adapter):
        from plyfile import PlyData
        from types import SimpleNamespace
        from scripts.fan.evaluate_fan_surface import model_from_vertices
        from scripts.rendering.render_ply import make_camera
        self.prepared=Path(prepared);self.adapter=adapter;meta=adapter.metadata
        base=PlyData.read(meta['source'])['vertex'].data
        self.model=model_from_vertices(base)
        self.xyz=self.model._xyz.detach().clone();self.rotation=self.model._rotation.detach().clone()
        self.rotation=torch.nn.functional.normalize(self.rotation,dim=-1)
        self.pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
        from .background import training_background
        self.background_record=training_background(self.prepared)
        self.background=torch.tensor(self.background_record['rgb'],dtype=torch.float32,device='cuda')
        cam=meta['camera'];video=meta['video']
        self.camera=make_camera(np.array(cam['pose']),video['width'],video['height'],np.deg2rad(cam['fovy_degrees']))
        with np.load(self.prepared/'motion.npz') as z:self.bands=torch.from_numpy(z['band_indices'].copy()).long().cuda()
        self.band_poses=np.load(self.prepared/'bands.npy',mmap_mode='r') if len(self.bands) else None

    def render(self,frame):
        from gaussian_renderer import render
        a=self.adapter;idx=torch.as_tensor(frame,device=a.q.device)
        q,t=a.transforms(idx);q=q[a.groups];t=t[a.groups]
        xyz=qrotate(q,self.xyz)+t;rotation=qmul(q,self.rotation)
        if self.band_poses is not None:
            band=torch.tensor(np.asarray(self.band_poses[frame]),device='cuda')
            # Elastic source bands retain the original baked local-rigid fits.
            # No learned group residual is applied to these nonrigid rows.
            if a.method!='C1':
                rest=torch.cat([self.xyz[self.bands],self.rotation[self.bands]],-1)
                band=a.alignment(rest,band,frame)
            xyz=xyz.index_copy(0,self.bands,band[:,:3]);rotation=rotation.index_copy(0,self.bands,band[:,3:])
        self.model._xyz=xyz;self.model._rotation=rotation
        return render(self.camera,self.model,self.pipe,self.background,0.,0.,0.)

    def trajectories(self,frames):
        a=self.adapter;q,t=a.transforms(frames)
        ids=a.sample_indices;g=a.groups[ids]
        xyz=qrotate(q[:,g],self.xyz[ids])+t[:,g]
        gap,angle=a.joint_coordinates(q,t)
        return xyz,gap,angle
