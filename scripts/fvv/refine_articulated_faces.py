"""Refine complete articulated bucket AND arm from all-face source references.

Preserve all geometry, articulation and the complete base checkpoint.
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
    head_refs=json.loads((a.bucket_views/'views.json').read_text())
    if len(head_refs['views'])!=head_refs['expected_views'] or not 1<=len(head_refs['views'])<=128:raise ValueError('Incomplete bounded bucket references')
    r=hybrid.Renderer(SimpleNamespace(repo=a.repo,model=a.model));head=r.bucket.head_only();arm=r.bucket.arm_only()
    from training_kernel import select
    kernel=select(a.repo)
    for name in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','_t','_scaling_t','_rotation_r','static_xyz','static_features_dc','static_features_rest','static_scaling','static_rotation','static_opacity']:getattr(r.model,name).requires_grad_(False)
    motion_pairs={}
    if a.motion_coherence:
        if not a.motion_neighbors:raise ValueError('Coherence requires validated source neighbors')
        with np.load(a.motion_neighbors,allow_pickle=False) as data:
            evidence=json.loads(str(data['metadata_json']))
            if evidence['parent_sidecar_sha256']!=r.meta['rigid_bucket']['sha256']:raise ValueError('Motion neighbors differ from model')
            for name,component in [('bucket',head),('arm',arm)]:
                pairs=data[name]
                if pairs.ndim!=2 or pairs.shape[1]!=2 or not 1000<len(pairs)<300000 or pairs.dtype.kind not in 'iu' or pairs.min()<0 or pairs.max()>=len(component.values['points']):raise ValueError('Invalid motion neighbor pairs')
                motion_pairs[name]=torch.from_numpy(pairs).cuda()
    optimizer_groups=[]
    for component in ([arm] if a.arm_only else [head,arm]):
        for key,lr in [('dc',.001),('rest',.00005),('opacity',.0005),('temporal_dc',.00025)]:
            if key=='opacity' and a.freeze_opacity:continue
            if key=='temporal_dc' and a.motion_coherence:lr=.001
            value=torch.nn.Parameter(component.values[key].detach().clone())
            component.values[key]=value;optimizer_groups.append(dict(params=[value],lr=lr))
    optimizer=torch.optim.Adam(optimizer_groups,eps=1e-15)
    full=head.wrap(arm.wrap(r.base_render_fn))
    isolated={'arm':arm.wrap(r.base_render_fn),'bucket':head.wrap(r.base_render_fn)}
    masked=bucket_only_model(r.model)
    def load(folder):
        result=[]
        for view in json.loads((folder/'views.json').read_text())['views']:
            if view['split']=='train':result.append((view_camera(view),torch.from_numpy(np.array(Image.open(folder/view['file']).convert('RGB')))))
        if not result:raise ValueError('No training views')
        return result
    focused={'arm':load(a.arm_views),'bucket':load(a.bucket_views)};wide=load(a.views);frames=decode(a.video);cams=[camera(r,i/60) for i in range(399)]
    from utils.loss_utils import ssim
    def loss_at(cam,pixels,isolate=False,real=False):
        target=pixels.cuda().permute(2,0,1).float()/255
        image=(isolated[isolate](cam,masked,r.pipe,r.bg) if isolate else full(cam,r.model,r.pipe,r.bg))['render']
        weight=1+3*(target.mean(0)>.03).float()
        l1=((image-target).abs()*weight).mean()/weight.mean()
        edge=((image[:,:,1:]-image[:,:,:-1])-(target[:,:,1:]-target[:,:,:-1])).abs().mean()+((image[:,1:,:]-image[:,:-1,:])-(target[:,1:,:]-target[:,:-1,:])).abs().mean()
        return .8*l1+.2*(1-ssim(image,target))+.2*edge
    order={'arm':[],'bucket':[]};wide_order=[];start=time.monotonic()
    for step in range(1,a.steps+1):
        if time.monotonic()-start>3600:raise TimeoutError('Arm refinement exceeded one hour')
        optimizer.zero_grad(set_to_none=True);i=int(rng.integers(399));loss=(16 if a.motion_fit else 8)*loss_at(cams[i],torch.from_numpy(frames[i].copy()),real=True);loss.backward();logged=float(loss.detach())
        for name in (['arm'] if a.arm_only else ['arm','bucket']):
            if not order[name]:order[name]=rng.permutation(len(focused[name])).tolist()
            cam,pixels=focused[name][order[name].pop()]
            loss=(12 if a.arm_only else 6)*loss_at(cam,pixels,isolate=name);loss.backward();logged+=float(loss.detach())
        if not wide_order:wide_order=rng.permutation(len(wide)).tolist()
        cam,pixels=wide[wide_order.pop()];loss=2*loss_at(cam,pixels);loss.backward();logged+=float(loss.detach())
        i=int(rng.integers(398));first=full(cams[i],r.model,r.pipe,r.bg)['render'];second=full(cams[i+1],r.model,r.pipe,r.bg)['render'];observed=torch.from_numpy((frames[i+1].astype(np.float32)-frames[i].astype(np.float32))/255).cuda().permute(2,0,1)
        if a.motion_fit:
            with torch.no_grad():
                alpha0=full(cams[i],masked,r.pipe,r.bg)['alpha']
                alpha1=full(cams[i+1],masked,r.pipe,r.bg)['alpha']
                roi=torch.nn.functional.max_pool2d(((alpha0>.05)|(alpha1>.05)).float()[None],5,1,2)[0]
            loss=16*((second-first-observed).abs()*roi).sum()/(3*roi.sum().clamp_min(1))
        else:loss=8*(second-first-observed).abs().mean()
        loss.backward()
        reg=0
        for component in [head,arm]:
            temporal=component.values['temporal_dc']
            selected_knots=temporal[:,torch.randint(temporal.shape[1],(4096,),device='cuda')]
            reg=reg+.005*selected_knots.square().mean()+.02*(selected_knots[2:]-2*selected_knots[1:-1]+selected_knots[:-2]).square().mean()
        coherence=0
        if motion_pairs:
            from motion_coherence import temporal_neighbor_loss
            for name,component in [('bucket',head),('arm',arm)]:
                pairs=motion_pairs[name];sample=pairs[torch.randint(len(pairs),(4096,),device='cuda')]
                coherence=coherence+temporal_neighbor_loss(component.values['temporal_dc'],sample)
            reg=reg+a.motion_coherence*coherence
        reg.backward();optimizer.step()
        with torch.no_grad():
            for component in [head,arm]:
                component.values['dc'].clamp_(-1.7,1.77)
                component.values['temporal_dc'].clamp_(-2,2)
                if not a.freeze_opacity:component.values['opacity'].clamp_(-4,8)
        if not np.isfinite(logged):raise ValueError('Nonfinite loss')
        if step==1 or step%100==0:
            row=dict(step=step,loss=logged,coherence=float(coherence),elapsed=time.monotonic()-start)
            with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    out={k:v.copy() for k,v in original.items()}
    for component,mask in [(head,~selected),(arm,selected)]:
        for key in ['dc','rest','opacity','temporal_dc']:
            value=component.values[key].detach().cpu().numpy()
            if key=='temporal_dc':out[key][:,mask]=value
            else:out[key][mask]=value
    for key in original:
        if key not in ['dc','rest','opacity','temporal_dc'] and not np.array_equal(out[key],original[key]):
            raise AssertionError('Frozen geometry changed: '+key)
    if a.arm_only:
        for key in ['dc','rest','opacity','temporal_dc']:
            value=out[key][:,~selected] if key=='temporal_dc' else out[key][~selected]
            before=original[key][:,~selected] if key=='temporal_dc' else original[key][~selected]
            if not np.array_equal(value,before):raise AssertionError('Protected head changed: '+key)
    if a.freeze_opacity and not np.array_equal(out['opacity'],original['opacity']):raise AssertionError('Frozen opacity changed')
    np.savez_compressed(a.output/'rigid-bucket.npz',**out);read_arrays(a.output/'rigid-bucket.npz',2)
    shutil.copyfile(a.model/'model.pth',a.output/'model.pth');shutil.copyfile(a.model/'effective.yaml',a.output/'effective.yaml')
    result=dict(r.meta);result.update(trained_at=model_timestamp(),training_iteration=r.meta['training_iteration']+a.steps)
    evidence=dict(parent_core_sha256=r.meta['sha256'],parent_sidecar_sha256=r.meta['rigid_bucket']['sha256'],steps=a.steps,
        arm_views_sha256=digest(a.arm_views/'views.json'),bucket_views_sha256=digest(a.bucket_views/'views.json'),
        training_images={k:len(v) for k,v in focused.items()},motion_coherence=a.motion_coherence,motion_roi_fit=a.motion_fit,head_exact=a.arm_only,opacity_exact=a.freeze_opacity,motion_neighbors_sha256=digest(a.motion_neighbors) if a.motion_neighbors else None,base_exact=True,geometry_exact=True,kernel=kernel,elapsed=time.monotonic()-start,
        scope='Complete bucket and six arm groups; isolated all-face source references plus full scene and original RGB temporal supervision')
    result['articulated_allfaces_refinement']=evidence;result['rigid_bucket']=dict(r.meta['rigid_bucket'],sha256=digest(a.output/'rigid-bucket.npz'));result['bytes']=usage(a.output)
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(evidence),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['repo','model','video','views','arm-views','bucket-views','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,choices=[3000,6000,12000],default=6000);p.add_argument('--registry',type=Path,action='append',default=[])
    p.add_argument('--motion-neighbors',type=Path)
    p.add_argument('--motion-coherence',type=float,choices=[0.,.25,.5,1.,2.,5.],default=0.)
    p.add_argument('--freeze-opacity',action='store_true')
    p.add_argument('--motion-fit',action='store_true',help='Prioritize observed RGB temporal changes within the complete articulated silhouette')
    p.add_argument('--arm-only',action='store_true',help='Preserve bucket appearance and focus on all arm directions')
    a=p.parse_args();records=[];lock=a.model.parent/'.fvv-writer.lock';signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(KeyboardInterrupt()))
    with lock.open('x') as f:f.write('articulated all-face appearance refinement')
    try:pause(a.registry,records);fit(a)
    finally:
        try:resume(records)
        finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
