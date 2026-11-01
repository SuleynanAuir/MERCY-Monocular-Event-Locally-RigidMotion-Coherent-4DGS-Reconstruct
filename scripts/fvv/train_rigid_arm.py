"""Train six articulated arm surfaces; freeze existing bucket and remaining base."""
import argparse,json,time,signal
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
import hybrid
from common import digest,usage,model_timestamp
from rigid_bucket import Bucket,read_arrays
from seed_covariance import static_surface_rotation
from refine_lego import camera,view_camera,decode
from training_resources import pause,resume


def fit(a):
    torch.set_num_threads(4);torch.manual_seed(1102);rng=np.random.default_rng(1102)
    a.output.mkdir(exist_ok=False)
    if usage(a.output.parent)>2*1024**3:raise OSError('Arm experiment exceeds 2 GiB')
    if digest(a.video)!='5cbf3042a577359a420d694e0117e2de715e2c5671c7a9fb7f5519917ed0a3bd':raise ValueError('Unexpected video')
    r=hybrid.Renderer(SimpleNamespace(repo=a.repo,model=a.model));head=r.bucket
    if not head or r.meta['rigid_bucket']['schema']!=1:raise ValueError('Requires the existing single bucket parent')
    with np.load(a.source,allow_pickle=False) as d:meta=json.loads(str(d['metadata_json']));ids=d['indices'];points=d['points'];normals=d['normals'];labels=d['parts'];groups=d['groups'];tracks=d['tracks']
    if meta['parent_sha256']!=r.meta['sha256'] or meta['parent_bucket_sha256']!=r.meta['rigid_bucket']['sha256']:raise ValueError('Arm source parent mismatch')
    cells=np.concatenate([np.floor(points/.0015).astype(np.int64),labels[:,None],np.rint(normals*8).astype(np.int64)],axis=1)
    _,unique=np.unique(cells,axis=0,return_index=True);selected=np.sort(rng.choice(unique,min(32768,len(unique)),replace=False));points=points[selected];normals=normals[selected];normals/=np.linalg.norm(normals,axis=1,keepdims=True);point_groups=groups[selected]
    if len(np.unique(point_groups))!=6:raise ValueError('A motion group has no surface support')
    n=len(points);rotations=Rotation.from_matrix(tracks[:,:,:3,:3].reshape(-1,3,3)).as_quat()[:,[3,0,1,2]].reshape(399,6,4).astype(np.float32)
    arrays=dict(points=points,rotations=static_surface_rotation(normals),scales=np.tile([.0025,.0025,.00045],(n,1)).astype(np.float32),dc=np.full((n,1,3),(.55-.5)/.28209479177387814,np.float32),rest=np.zeros((n,15,3),np.float32),opacity=np.full((n,1),2.2,np.float32),temporal_dc=np.zeros((48,n,1,3),np.float32),point_groups=point_groups.astype(np.float32),track_rotations=rotations,track_translations=tracks[:,:,:3,3])
    arm=Bucket(arrays)
    from training_kernel import select
    kernel=select(a.repo)
    optimizer_groups=[]
    for key,lr in [('dc',.01),('rest',.0005),('opacity',.003),('temporal_dc',.002)]:
        value=torch.nn.Parameter(arm.values[key]);arm.values[key]=value;optimizer_groups.append(dict(params=[value],lr=lr))
    optimizer=torch.optim.Adam(optimizer_groups,eps=1e-15)
    state,iteration=hybrid.load_checkpoint(a.model/'model.pth','cpu');state=list(state);keep=torch.ones(len(state[1]),dtype=torch.bool)
    if ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids) or ids.min()<0 or ids.max()>=len(keep):raise ValueError('Invalid arm retirement indices')
    keep[torch.from_numpy(ids)]=False
    for field in [1,2,3,4,5,6,13,14,15]:state[field]=state[field][keep]
    retained=tuple(state);r.model.restore(tuple(x.cuda() if torch.is_tensor(x) else x for x in retained),None)
    for name in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','_t','_scaling_t','_rotation_r','static_xyz','static_features_dc','static_features_rest','static_scaling','static_rotation','static_opacity']:getattr(r.model,name).requires_grad_(False)
    r.render_fn=head.wrap(arm.wrap(r.base_render_fn));frames=decode(a.video);cams=[camera(r,i/60) for i in range(399)]
    views=[]
    for v in json.loads((a.views/'views.json').read_text())['views']:
        if v['split']=='train':views.append((view_camera(v),torch.from_numpy(np.array(Image.open(a.views/v['file']).convert('RGB')))))
    from utils.loss_utils import ssim
    def loss_at(cam,pixels,real=False):
        target=pixels.cuda().permute(2,0,1).float()/255;image=r.render_fn(cam,r.model,r.pipe,r.bg)['render'];weight=1+3*(target.mean(0)>.03).float()
        if real:weight[120:470,210:470]*=3
        l1=((image-target).abs()*weight).mean()/weight.mean();edge=((image[:,:,1:]-image[:,:,:-1])-(target[:,:,1:]-target[:,:,:-1])).abs().mean()+((image[:,1:,:]-image[:,:-1,:])-(target[:,1:,:]-target[:,:-1,:])).abs().mean()
        return .8*l1+.2*(1-ssim(image,target))+.2*edge
    started=time.monotonic();order=[]
    for step in range(1,a.steps+1):
        if time.monotonic()-started>3600:raise TimeoutError('Arm training exceeded one hour')
        optimizer.zero_grad(set_to_none=True);index=int(rng.integers(399));loss=8*loss_at(cams[index],torch.from_numpy(frames[index].copy()),True);loss.backward();logged=float(loss.detach())
        for _ in range(2):
            if not order:order=rng.permutation(len(views)).tolist()
            cam,pixels=views[order.pop()];loss=2*loss_at(cam,pixels);loss.backward();logged+=float(loss.detach())
        index=int(rng.integers(398));first=r.render_fn(cams[index],r.model,r.pipe,r.bg)['render'];second=r.render_fn(cams[index+1],r.model,r.pipe,r.bg)['render'];observed=torch.from_numpy((frames[index+1].astype(np.float32)-frames[index].astype(np.float32))/255).cuda().permute(2,0,1)
        loss=8*(second-first-observed)[:,120:470,210:470].abs().mean();loss.backward();logged+=float(loss.detach())
        temporal=arm.values['temporal_dc'];regularization=.005*temporal.square().mean()+.02*(temporal[2:]-2*temporal[1:-1]+temporal[:-2]).square().mean()+.002*(1-arm.values['opacity'].sigmoid()).square().mean()
        regularization.backward();optimizer.step()
        with torch.no_grad():
            arm.values['dc'].clamp_(-1.7,1.77);arm.values['temporal_dc'].clamp_(-2,2);arm.values['opacity'].clamp_(-4,8)
        if not np.isfinite(logged):raise ValueError('Nonfinite arm loss')
        if step==1 or step%100==0:
            row=dict(step=step,loss=logged,elapsed=time.monotonic()-started)
            with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    for field,name in [(1,'_xyz'),(2,'_features_dc'),(3,'_features_rest'),(4,'_scaling'),(5,'_rotation'),(6,'_opacity'),(13,'_t'),(14,'_scaling_t'),(15,'_rotation_r'),(19,'static_xyz'),(20,'static_features_dc'),(21,'static_features_rest'),(22,'static_scaling'),(23,'static_rotation'),(24,'static_opacity')]:
        if not torch.equal(getattr(r.model,name).detach().cpu(),retained[field]):raise AssertionError('Non-arm base changed')
    original=read_arrays(a.model/'rigid-bucket.npz');out={}
    for key,value in original.items():
        if key.startswith('track_'):out[key]=np.concatenate([value[:,None],arm.values[key].detach().cpu().numpy()],axis=1)
        else:out[key]=np.concatenate([value,arm.values[key].detach().cpu().numpy()],axis=1 if key=='temporal_dc' else 0)
    out['point_groups']=np.concatenate([np.zeros(len(original['points']),np.float32),point_groups.astype(np.float32)+1])
    for key,value in original.items():
        saved=out[key][:,0] if key.startswith('track_') else out[key][:,:len(value[0])] if key=='temporal_dc' else out[key][:len(value)]
        if not np.array_equal(saved,value):raise AssertionError('Original bucket changed: '+key)
    np.savez_compressed(a.output/'rigid-bucket.npz',**out);read_arrays(a.output/'rigid-bucket.npz',2)
    hybrid.save_bounded(hybrid.compact((retained,iteration+a.steps)),a.output/'model.pth',512*1024**2);(a.output/'effective.yaml').write_bytes((a.model/'effective.yaml').read_bytes())
    evidence=dict(parent_sha256=r.meta['sha256'],parent_bucket_sha256=r.meta['rigid_bucket']['sha256'],source_sha256=digest(a.source),steps=a.steps,points=n,retired_dynamic=len(ids),groups=meta['groups'],bucket_preserved_exact=True,nonarm_base_preserved_exact=True,elapsed_seconds=time.monotonic()-started,training_kernel=kernel,video_sha256=digest(a.video),views_sha256=digest(a.views/'views.json'),scope=meta['scope'])
    result=dict(r.meta);result.update(sha256=digest(a.output/'model.pth'),trained_at=model_timestamp(),training_iteration=iteration+a.steps,arm_refinement=evidence)
    result['rigid_bucket']=dict(r.meta['rigid_bucket'],schema=2,sha256=digest(a.output/'rigid-bucket.npz'),points=len(out['points']),original_bucket_points=len(original['points']),motion_groups=7,arm_points=n,source='Existing bucket plus six explicit supplied rigid arm groups; not monocular recovery')
    (a.output/'arm-training.json').write_text(json.dumps(evidence,indent=2));result['bytes']=usage(a.output);(a.output/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(evidence))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['repo','model','source','video','views','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,choices=[1000,6000,12000],default=6000);p.add_argument('--registry',type=Path,action='append',default=[])
    a=p.parse_args();records=[];lock=a.model.parent/'.fvv-writer.lock';signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(KeyboardInterrupt()))
    with lock.open('x') as f:f.write('articulated arm training')
    try:pause(a.registry,records);fit(a)
    finally:
        try:resume(records)
        finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
