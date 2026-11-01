"""Optional explicit-source rigid Gaussian bucket, isolated from legacy models.

One canonical surface and a bounded pose track replace independent bucket 4D
trajectories. This is prior-assisted rendering, not monocular motion recovery.
"""
import json,math
import zipfile
import numpy as np


class _FieldProxy:
    """One reusable type: per-frame classes retain CUDA tensors through MRO cycles."""
    def __init__(self,base,fields):self.base=base;self.fields=fields
    def __getattr__(self,key):return self.fields[key] if key in self.fields else getattr(self.base,key)


class _OpacityMaskedModel:
    def __init__(self,base):self.base=base
    def __getattr__(self,key):
        import torch
        if key in ['get_opacity','get_static_opacity']:return torch.zeros_like(getattr(self.base,key))
        return getattr(self.base,key)


def quaternion_product(a,b):
    import torch
    w,x,y,z=a.unbind(-1);v,i,j,k=b.unbind(-1)
    return torch.stack((w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v),-1)


def quaternion_rotate(q,points):
    import torch
    vector=q[...,1:].expand_as(points)
    uv=torch.cross(vector,points,dim=-1)
    return points+2*(q[...,:1]*uv+torch.cross(vector,uv,dim=-1))


def pose_at(rotations,translations,timestamp):
    """Sign-aligned normalized quaternion interpolation; never blend matrices."""
    import torch
    if not math.isfinite(float(timestamp)):raise ValueError('Nonfinite bucket time')
    value=min(max(float(timestamp)*60,0),len(rotations)-1)
    index=min(int(value),len(rotations)-2);fraction=value-index
    q0,q1=rotations[index],rotations[index+1]
    q1=torch.where((q0*q1).sum(dim=-1,keepdim=True)<0,-q1,q1)
    q=torch.nn.functional.normalize((1-fraction)*q0+fraction*q1,dim=-1)
    return q,(1-fraction)*translations[index]+fraction*translations[index+1]


def read_arrays(path,schema=1):
    """Bound nested NPZ expansion and array headers before NumPy allocates."""
    if schema not in (1,2,3):raise ValueError('Unsupported rigid schema')
    limit={1:32,2:64,3:128}[schema]*1024**2
    expected={'points','rotations','scales','dc','rest','opacity','track_rotations','track_translations','temporal_dc'}
    if schema>=2:expected.add('point_groups')
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist()
        if len(entries)!=len(expected) or {e.filename for e in entries}!={k+'.npy' for k in expected} or sum(e.file_size for e in entries)>limit:
            raise ValueError('Unexpected or oversized rigid bucket archive')
        shapes={}
        for entry in entries:
            with archive.open(entry) as stream:
                version=np.lib.format.read_magic(stream)
                if version!=(1,0):raise ValueError('Unsupported rigid array header')
                shape,order,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=1024)
                if dtype!=np.dtype('float32') or any(type(x)!=int or x<0 for x in shape) or math.prod(shape)>limit//4 or stream.tell()+math.prod(shape)*4!=entry.file_size:
                    raise ValueError('Invalid rigid array layout')
                shapes[entry.filename[:-4]]=shape
        n=shapes['points'][0] if shapes['points'] else 0
        knots=shapes['temporal_dc'][0] if shapes['temporal_dc'] else 0
        wanted={'points':(n,3),'rotations':(n,4),'scales':(n,3),'dc':(n,1,3),'rest':(n,15,3),'opacity':(n,1),'track_rotations':(399,4),'track_translations':(399,3),'temporal_dc':(knots,n,1,3)}
        if schema>=2:
            groups=shapes['track_rotations'][1] if len(shapes['track_rotations'])==3 else 0
            if not 2<=groups<=8:raise ValueError('Rigid group budget exceeded')
            if schema==3 and groups!=8:raise ValueError('Whole-scene schema requires seven articulated groups plus fixed body')
            wanted.update(point_groups=(n,),track_rotations=(399,groups,4),track_translations=(399,groups,3))
        if not 1<=n<=(163840 if schema==3 else 65536) or knots not in [16,48] or shapes!=wanted:raise ValueError('Invalid rigid array shapes')
    with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in expected}
    if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite rigid parameters')
    if schema>=2:
        ids=arrays['point_groups']
        if not np.array_equal(ids,np.rint(ids)) or not np.array_equal(np.unique(ids),np.arange(groups)):raise ValueError('Invalid articulated group indices')
    if schema==3 and (not np.allclose(arrays['track_rotations'][:,7],[1,0,0,0],atol=1e-7) or not np.array_equal(arrays['track_translations'][:,7],np.zeros((399,3)))):
        raise ValueError('Fixed body must have an identity track')
    for key in ['rotations','track_rotations']:
        if not np.allclose(np.linalg.norm(arrays[key],axis=-1),1,atol=2e-4):raise ValueError('Nonunit rigid quaternion')
    if not ((arrays['scales']>0)&(arrays['scales']<=.02)).all():raise ValueError('Rigid surface scale budget exceeded')
    return arrays


def bucket_only_model(base):
    """Offline diagnostics only: mask base opacity without changing its tensors.

    Bucket.proxy appends the visible bucket after this wrapper. Production
    rendering never calls this helper, so ordinary scene occlusion is retained.
    """
    return _OpacityMaskedModel(base)


class Bucket:
    def __init__(self,arrays):
        import torch
        self.values={k:torch.as_tensor(v,dtype=torch.float32,device='cuda') for k,v in arrays.items()}

    def head_only(self):
        """Offline bucket diagnostics: select original group zero in schema 2."""
        if 'point_groups' not in self.values:return self
        selected=self.values['point_groups']==0
        result=Bucket.__new__(Bucket);result.values={}
        for key,value in self.values.items():
            if key=='point_groups':continue
            result.values[key]=value[:,0] if key.startswith('track_') else value[:,selected] if key=='temporal_dc' else value[selected]
        return result

    def arm_only(self):
        """Offline whole moving-arm view; preserve all six relative trajectories."""
        if 'point_groups' not in self.values:raise ValueError('Articulated arm required')
        selected=(self.values['point_groups']>0)&(self.values['point_groups']<=6)
        result=Bucket.__new__(Bucket);result.values={}
        for key,value in self.values.items():
            result.values[key]=value if key.startswith('track_') else value[:,selected] if key=='temporal_dc' else value[selected]
        return result

    @classmethod
    def load(cls,root,meta):
        from common import digest
        path=root/'rigid-bucket.npz'
        schema=meta.get('schema')
        if schema not in (1,2,3) or meta.get('trained') is not True or path.stat().st_size>{1:32,2:64,3:128}[schema]*1024**2 or digest(path)!=meta.get('sha256'):
            raise ValueError('Invalid/untrained rigid bucket artifact')
        return cls(read_arrays(path,schema))

    def proxy(self,base,timestamp):
        import torch
        v=self.values;q,translation=pose_at(v['track_rotations'],v['track_translations'],timestamp)
        if 'point_groups' in v:
            groups=v['point_groups'].long();q=q[groups];translation=translation[groups]
        knots=len(v['temporal_dc']);progress=min(max(float(timestamp)*60/398*(knots-1),0),knots-1);index=min(int(progress),knots-2);weight=progress-index
        dc=v['dc']+(1-weight)*v['temporal_dc'][index]+weight*v['temporal_dc'][index+1]
        fields=dict(get_static_xyz=torch.cat((base.get_static_xyz,quaternion_rotate(q,v['points'])+translation)),
            get_static_rotation=torch.cat((base.get_static_rotation,quaternion_product(q,v['rotations']))),
            get_static_scaling=torch.cat((base.get_static_scaling,v['scales'])),
            get_static_features=torch.cat((base.get_static_features,torch.cat((dc,v['rest']),dim=1))),
            get_static_opacity=torch.cat((base.get_static_opacity,v['opacity'].sigmoid())))
        return _FieldProxy(base,fields)

    def wrap(self,render):
        def rigid_render(camera,model,pipe,bg,*args,**kwargs):
            return render(camera,self.proxy(model,camera.timestamp),pipe,bg,*args,**kwargs)
        return rigid_render
