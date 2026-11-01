"""Refine every supplied moving-arm surface from bounded multi-angle RGB references.

Preserve geometry, articulation, original bucket and complete base checkpoint.
Isolation is an offline training target only; production retains scene occlusion.
"""
import argparse,json,time,signal,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
import hybrid
from common import digest,usage,model_timestamp
from rigid_bucket import read_arrays,bucket_only_model
from refine_lego import camera,view_camera,decode
from training_resources import pause,resume


def fit(a):
    torch.set_num_threads(4);torch.manual_seed(1103);rng=np.random.default_rng(1103)
    a.output.mkdir(exist_ok=False)
    if usage(a.output.parent)>2*1024**3:raise OSError('Arm experiment exceeds 2 GiB')
    if digest(a.video)!='5cbf3042a577359a420d694e0117e2de715e2c5671c7a9fb7f5519917ed0a3bd':raise ValueError('Unexpected source video')
    original=read_arrays(a.model/'rigid-bucket.npz',2);selected=original['point_groups']>0
    refs=json.loads((a.arm_views/'views.json').read_text())
    if refs.get('isolated_arm') is not True or len(refs['views'])!=refs['expected_views'] or not 1<=len(refs['views'])<=128:raise ValueError('Incomplete bounded arm references')
    r=hybrid.Renderer(SimpleNamespace(repo=a.repo,model=a.model));head=r.bucket.head_only();arm=r.bucket.arm_only()
    surface_pairs=None
    if a.surface_color_weight:
        if not a.surface_source:raise ValueError('Explicit arm source required for surface neighbors')
        from scipy.spatial import cKDTree
        with np.load(a.surface_source,allow_pickle=False) as d:points=d['points'];normals=d['normals'];labels=d['parts'];groups=d['groups']
        cells=np.concatenate([np.floor(points/.0015).astype(np.int64),labels[:,None],np.rint(normals*8).astype(np.int64)],axis=1)
        _,unique=np.unique(cells,axis=0,return_index=True);chosen=np.sort(np.random.default_rng(1102).choice(unique,min(32768,len(unique)),replace=False))
        points=points[chosen];normals=normals[chosen];normals/=np.linalg.norm(normals,axis=1,keepdims=True);labels=labels[chosen]
        if not np.array_equal(points,original['points'][selected]) or not np.array_equal(groups[chosen]+1,original['point_groups'][selected]):raise ValueError('Source surface correspondence changed')
        distance,near=cKDTree(points).query(points,k=5);left=np.repeat(np.arange(len(points)),4);right=near[:,1:].reshape(-1);offset=points[left]-points[right]
        valid=(left<right)&(distance[:,1:].reshape(-1)<.008)&(labels[left]==labels[right])&((normals[left]*normals[right]).sum(1)>.98)
        valid&=(np.abs((offset*normals[left]).sum(1))<.0008)&(np.abs((offset*normals[right]).sum(1))<.0008)
        pairs=np.stack([left[valid],right[valid]],axis=1)
        if not 1000<=len(pairs)<=131072:raise ValueError('Unexpected coplanar neighbor budget')
        surface_pairs=torch.from_numpy(pairs).cuda()
    from training_kernel import select
    kernel=select(a.repo)
    for name in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','_t','_scaling_t','_rotation_r','static_xyz','static_features_dc','static_features_rest','static_scaling','static_rotation','static_opacity']:getattr(r.model,name).requires_grad_(False)
    optimizer_groups=[]
    for key,lr in [('dc',.004),('rest',.0003),('opacity',.002),('temporal_dc',.001)]:
        value=torch.nn.Parameter(arm.values[key].detach().clone());arm.values[key]=value;optimizer_groups.append(dict(params=[value],lr=lr*(.5 if a.surface_color_weight else 1)))
    optimizer=torch.optim.Adam(optimizer_groups,eps=1e-15)
    full=head.wrap(arm.wrap(r.base_render_fn));isolated=arm.wrap(r.base_render_fn);masked=bucket_only_model(r.model)
    def load(folder):
        result=[]
        for view in json.loads((folder/'views.json').read_text())['views']:
            if view['split']=='train':result.append((view_camera(view),torch.from_numpy(np.array(Image.open(folder/view['file']).convert('RGB')))))
        if not result:raise ValueError('No training views')
        return result
    focused=load(a.arm_views);wide=load(a.views);frames=decode(a.video);cams=[camera(r,i/60) for i in range(399)]
    from utils.loss_utils import ssim
    def loss_at(cam,pixels,isolate=False,real=False):
        target=pixels.cuda().permute(2,0,1).float()/255
        image=(isolated(cam,masked,r.pipe,r.bg) if isolate else full(cam,r.model,r.pipe,r.bg))['render']
        weight=1+3*(target.mean(0)>.03).float()
        if real:weight[120:470,210:470]*=3
        l1=((image-target).abs()*weight).mean()/weight.mean()
        edge=((image[:,:,1:]-image[:,:,:-1])-(target[:,:,1:]-target[:,:,:-1])).abs().mean()+((image[:,1:,:]-image[:,:-1,:])-(target[:,1:,:]-target[:,:-1,:])).abs().mean()
        return .8*l1+.2*(1-ssim(image,target))+.2*edge
    order=[];wide_order=[];start=time.monotonic()
    for step in range(1,a.steps+1):
        if time.monotonic()-start>3600:raise TimeoutError('Arm refinement exceeded one hour')
        optimizer.zero_grad(set_to_none=True);i=int(rng.integers(399));loss=8*loss_at(cams[i],torch.from_numpy(frames[i].copy()),real=True);loss.backward();logged=float(loss.detach())
        for _ in range(2):
            if not order:order=rng.permutation(len(focused)).tolist()
            cam,pixels=focused[order.pop()];loss=4*loss_at(cam,pixels,isolate=True);loss.backward();logged+=float(loss.detach())
        if not wide_order:wide_order=rng.permutation(len(wide)).tolist()
        cam,pixels=wide[wide_order.pop()];loss=2*loss_at(cam,pixels);loss.backward();logged+=float(loss.detach())
        i=int(rng.integers(398));first=full(cams[i],r.model,r.pipe,r.bg)['render'];second=full(cams[i+1],r.model,r.pipe,r.bg)['render'];observed=torch.from_numpy((frames[i+1].astype(np.float32)-frames[i].astype(np.float32))/255).cuda().permute(2,0,1)
        loss=8*(second-first-observed)[:,120:470,210:470].abs().mean();loss.backward()
        temporal=arm.values['temporal_dc'];reg=.005*temporal.square().mean()+.02*(temporal[2:]-2*temporal[1:-1]+temporal[:-2]).square().mean()
        if surface_pairs is not None:
            pairs=surface_pairs[torch.randint(len(surface_pairs),(min(8192,len(surface_pairs)),),device='cuda')];left,right=pairs.unbind(1)
            reg+=a.surface_color_weight*((arm.values['dc'][left]-arm.values['dc'][right]).square().mean()+.1*(arm.values['rest'][left]-arm.values['rest'][right]).square().mean()+.25*(temporal[:,left]-temporal[:,right]).square().mean())
        reg.backward();optimizer.step()
        with torch.no_grad():arm.values['dc'].clamp_(-1.7,1.77);arm.values['temporal_dc'].clamp_(-2,2);arm.values['opacity'].clamp_(-4,8)
        if not np.isfinite(logged):raise ValueError('Nonfinite loss')
        if step==1 or step%100==0:
            row=dict(step=step,loss=logged,elapsed=time.monotonic()-start)
            with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    out={k:v.copy() for k,v in original.items()}
    for key in ['dc','rest','opacity','temporal_dc']:
        value=arm.values[key].detach().cpu().numpy()
        if key=='temporal_dc':out[key][:,selected]=value
        else:out[key][selected]=value
    for key,value in original.items():
        same=np.array_equal(out[key][:,~selected],value[:,~selected]) if key=='temporal_dc' else np.array_equal(out[key][~selected],value[~selected]) if key in ['dc','rest','opacity'] else np.array_equal(out[key],value)
        if not same:raise AssertionError('Frozen parameters changed: '+key)
    np.savez_compressed(a.output/'rigid-bucket.npz',**out);read_arrays(a.output/'rigid-bucket.npz',2)
    shutil.copyfile(a.model/'model.pth',a.output/'model.pth');shutil.copyfile(a.model/'effective.yaml',a.output/'effective.yaml')
    result=dict(r.meta);result.update(trained_at=model_timestamp(),training_iteration=r.meta['training_iteration']+a.steps)
    evidence=dict(parent_core_sha256=r.meta['sha256'],parent_sidecar_sha256=r.meta['rigid_bucket']['sha256'],steps=a.steps,arm_views_sha256=digest(a.arm_views/'views.json'),training_images=len(focused),validation_images=len(refs['views'])-len(focused),base_and_bucket_exact=True,geometry_exact=True,kernel=kernel,elapsed=time.monotonic()-start,scope=refs['scope'])
    evidence.update(surface_color_weight=a.surface_color_weight,surface_pairs=len(surface_pairs) if surface_pairs is not None else 0,surface_source_sha256=digest(a.surface_source) if a.surface_source else None)
    result['arm_allfaces_refinement']=evidence;result['rigid_bucket']=dict(r.meta['rigid_bucket'],sha256=digest(a.output/'rigid-bucket.npz'));result['bytes']=usage(a.output)
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(evidence),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['repo','model','video','views','arm-views','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,choices=[3000,6000,12000],default=6000);p.add_argument('--registry',type=Path,action='append',default=[])
    p.add_argument('--surface-source',type=Path);p.add_argument('--surface-color-weight',type=float,choices=[0.,.05],default=0.)
    a=p.parse_args();records=[];lock=a.model.parent/'.fvv-writer.lock';signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(KeyboardInterrupt()))
    with lock.open('x') as f:f.write('whole-arm appearance refinement')
    try:pause(a.registry,records);fit(a)
    finally:
        try:resume(records)
        finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
