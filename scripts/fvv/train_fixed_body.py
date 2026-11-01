"""Source-assisted fixed-body candidate; preserve all seven articulated groups.

Only provided static geometry is fixed. Other moving components remain in the
original 4D model. Candidate creation never changes the active viewer model.
"""
import argparse,json,signal,time,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:128,garbage_collection_threshold:0.8')
import torch
from PIL import Image
from scipy.spatial import cKDTree
import hybrid
from common import digest,usage,model_timestamp
from rigid_bucket import Bucket,read_arrays
from seed_covariance import static_surface_rotation
from refine_lego import camera,view_camera,decode
from training_resources import pause,resume
from temporal_supervision import backward_observed_change


FIELDS=[(1,'_xyz'),(2,'_features_dc'),(3,'_features_rest'),(4,'_scaling'),(5,'_rotation'),(6,'_opacity'),(13,'_t'),(14,'_scaling_t'),(15,'_rotation_r'),(19,'static_xyz'),(20,'static_features_dc'),(21,'static_features_rest'),(22,'static_scaling'),(23,'static_rotation'),(24,'static_opacity')]


def fit(a):
    torch.set_num_threads(4);torch.manual_seed(1104);rng=np.random.default_rng(1104)
    a.output.mkdir(exist_ok=False);started=time.monotonic()
    if usage(a.output.parent)>2*1024**3:raise OSError('Experiment storage budget exceeded')
    if digest(a.video)!='5cbf3042a577359a420d694e0117e2de715e2c5671c7a9fb7f5519917ed0a3bd':raise ValueError('Unexpected video')
    r=hybrid.Renderer(SimpleNamespace(repo=a.repo,model=a.model));protected=r.bucket
    if r.meta.get('rigid_bucket',{}).get('schema')!=2:raise ValueError('Requires seven-group parent')
    original=read_arrays(a.model/'rigid-bucket.npz',2)
    if original['track_rotations'].shape[1]!=7:raise ValueError('Requires bucket plus six arm groups')
    with np.load(a.source,allow_pickle=False) as d:
        points=d['static_points'];normals=d['static_normals'];labels=d['static_parts']
        moving=d['points'];moving_labels=d['parts'];times=d['times'].reshape(-1)
    if len(points)>200000 or len(moving)>1200000 or not np.isfinite(points).all():raise ValueError('Source budget exceeded')
    # Infer which old 4D rows describe static source parts at their peak time.
    peaks=r.model.get_t.detach().cpu().numpy().reshape(-1);stamps=np.unique(times)
    if len(stamps)>64 or len(peaks)>450000:raise ValueError('Binding budget exceeded')
    assigned=np.abs(peaks[:,None]-stamps).argmin(1);nearest_labels=np.full(len(peaks),-1,np.int32);distances=np.zeros(len(peaks),np.float32)
    with torch.no_grad():
        for k,t in enumerate(stamps):
            rows=np.flatnonzero(assigned==k)
            if not len(rows):continue
            mask=times==t;surface=np.concatenate([moving[mask],points]);part=np.concatenate([moving_labels[mask],labels])
            _,offset=r.model.get_current_covariance_and_mean_offset(1.,float(t))
            distance,near=cKDTree(surface).query((r.model.get_xyz+offset)[rows].cpu().numpy(),workers=4)
            nearest_labels[rows]=part[near];distances[rows]=distance
            if time.monotonic()-started>600:raise TimeoutError('Body binding exceeded ten minutes')
    retired=np.isin(nearest_labels,np.unique(labels))
    if not 1000<retired.sum()<len(peaks):raise ValueError('Unexpected static membership')
    # Preserve all supplied static samples: decimation lost thin/curved support.
    if len(points)>98304:raise ValueError('Static source exceeds fixed body budget')
    chosen=np.arange(len(points))
    points=points[chosen].astype(np.float32);normals=normals[chosen].astype(np.float32);sampled_labels=labels[chosen]
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    n=len(points)
    arrays=dict(points=points,rotations=static_surface_rotation(normals),scales=np.tile([.0035,.0035,.00055],(n,1)).astype(np.float32),dc=np.full((n,1,3),(.55-.5)/.28209479177387814,np.float32),rest=np.zeros((n,15,3),np.float32),opacity=np.full((n,1),2.2,np.float32),temporal_dc=np.zeros((48,n,1,3),np.float32),track_rotations=np.tile([1.,0,0,0],(399,1)).astype(np.float32),track_translations=np.zeros((399,3),np.float32))
    projected_bytes=sum(v.nbytes for v in original.values())+sum(v.nbytes for v in arrays.values())+n*4+16384
    if len(original['points'])+n>163840 or projected_bytes>128*1024**2:raise OSError('Packed body sidecar would exceed budget')
    body=Bucket(arrays)
    state,iteration=hybrid.load_checkpoint(a.model/'model.pth','cpu');state=list(state);keep=torch.from_numpy(~retired);old_static=len(state[19])
    for field,_ in FIELDS:state[field]=state[field][keep] if field<19 else state[field][:0]
    retained=tuple(state);r.model.restore(tuple(x.cuda() if torch.is_tensor(x) else x for x in retained),None)
    for _,name in FIELDS:getattr(r.model,name).requires_grad_(False)
    from training_kernel import select
    kernel=select(a.repo);groups=[]
    for key,lr in [('dc',.008),('rest',.0005),('opacity',.003),('temporal_dc',.002)]:
        value=torch.nn.Parameter(body.values[key]);body.values[key]=value;groups.append(dict(params=[value],lr=lr))
    optimizer=torch.optim.Adam(groups,eps=1e-15)
    # Match the saved sidecar's protected/body order, including depth ties.
    render=protected.wrap(body.wrap(r.base_render_fn))
    frames=decode(a.video);cams=[camera(r,i/60) for i in range(399)];views=[];validation=[]
    for v in json.loads((a.views/'views.json').read_text())['views']:
        item=(view_camera(v),torch.from_numpy(np.array(Image.open(a.views/v['file']).convert('RGB'))))
        (views if v['split']=='train' else validation).append(item)
    if not views or not validation:raise ValueError('Training/validation views required')
    evidence=dict(parent_sha256=r.meta['sha256'],parent_sidecar_sha256=r.meta['rigid_bucket']['sha256'],source_sha256=digest(a.source),steps=a.steps,points=n,static_parts=len(np.unique(labels)),sampled_parts=len(np.unique(sampled_labels)),retired_dynamic=int(retired.sum()),retained_dynamic=int(keep.sum()),retired_static=old_static,membership_distance_percentiles=np.percentile(distances[retired],[50,90,95,99]).tolist(),geometry_fixed=True,protected_exact=True,kernel=kernel,scope='Provided static source surfaces; nearest-surface membership is inferred, not segmentation truth. Remaining moving parts retained.')
    evidence.update(seed=1104,surface_scales=[.0035,.0035,.00055],temporal_color_knots=48,learning_rates={key:lr for key,lr in [('dc',.008),('rest',.0005),('opacity',.003),('temporal_dc',.002)]},loss_weights=dict(original_rgb=8,multiview=3,multiview_batch=2,observed_temporal_change=8,edge=.3),training_views=len(views),validation_views=len(validation),decoded_sidecar_budget=128*1024**2)
    (a.output/'binding.json').write_text(json.dumps(evidence,indent=2))
    from utils.loss_utils import ssim
    def loss_at(cam,pixels):
        target=pixels.cuda().permute(2,0,1).float()/255;image=render(cam,r.model,r.pipe,r.bg)['render'];weight=1+3*(target.mean(0)>.03).float()
        l1=((image-target).abs()*weight).mean()/weight.mean()
        edge=((image[:,:,1:]-image[:,:,:-1])-(target[:,:,1:]-target[:,:,:-1])).abs().mean()+((image[:,1:,:]-image[:,:-1,:])-(target[:,1:,:]-target[:,:-1,:])).abs().mean()
        return .8*l1+.2*(1-ssim(image,target))+.3*edge
    order=[]
    for step in range(1,a.steps+1):
        if time.monotonic()-started>7200:raise TimeoutError('Body training exceeded two hours')
        optimizer.zero_grad(set_to_none=True);i=int(rng.integers(399));loss=8*loss_at(cams[i],torch.from_numpy(frames[i].copy()));loss.backward();logged=float(loss.detach())
        for _ in range(2):
            if not order:order=rng.permutation(len(views)).tolist()
            cam,pixels=views[order.pop()];loss=3*loss_at(cam,pixels);loss.backward();logged+=float(loss.detach())
        i=int(rng.integers(398))
        observed=torch.from_numpy((frames[i+1].astype(np.float32)-frames[i].astype(np.float32))/255).cuda().permute(2,0,1)
        weight=1+3*(torch.from_numpy(frames[i].mean(2)>.03*255).cuda()).float()
        # Exact L1 subgradient, but only one rasterizer backward graph at once.
        # No optimizer update occurs between the two deterministic evaluations.
        backward_observed_change(lambda:render(cams[i],r.model,r.pipe,r.bg)['render'],lambda:render(cams[i+1],r.model,r.pipe,r.bg)['render'],observed,weight)
        # Uniform point subsampling is an unbiased temporal regularizer estimate.
        temporal=body.values['temporal_dc'][:,torch.randint(n,(min(n,8192),),device='cuda')]
        reg=.005*temporal.square().mean()+.02*(temporal[2:]-2*temporal[1:-1]+temporal[:-2]).square().mean();reg.backward()
        if not np.isfinite(logged) or any(v.grad is not None and not torch.isfinite(v.grad).all() for g in groups for v in g['params']):raise ValueError('Nonfinite body training')
        optimizer.step()
        with torch.no_grad():body.values['dc'].clamp_(-1.7,1.77);body.values['temporal_dc'].clamp_(-2,2);body.values['opacity'].clamp_(-4,8)
        if step==1 or step%100==0:
            row=dict(step=step,loss=logged,elapsed=time.monotonic()-started,cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            if step%1000==0 or step==a.steps:
                scores=[]
                with torch.no_grad():
                    for cam,pixels in validation:
                        target=pixels.cuda().permute(2,0,1).float()/255;image=render(cam,r.model,r.pipe,r.bg)['render'];mask=target.mean(0)>.03
                        scores.append(float(-10*torch.log10((image[:,mask]-target[:,mask]).square().mean().clamp_min(1e-12))))
                row['validation_fg_psnr']=float(np.mean(scores))
            with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    for field,name in FIELDS:
        if not torch.equal(getattr(r.model,name).detach().cpu(),retained[field]):raise AssertionError('Retained core changed')
    for key in ['points','rotations','scales','track_rotations','track_translations']:
        if not np.array_equal(body.values[key].detach().cpu().numpy(),arrays[key]):raise AssertionError('Fixed surface geometry changed')
    out={}
    for key,value in original.items():
        if key=='point_groups':out[key]=np.concatenate([value,np.full(n,7,np.float32)])
        elif key.startswith('track_'):out[key]=np.concatenate([value,body.values[key].detach().cpu().numpy()[:,None]],axis=1)
        else:out[key]=np.concatenate([value,body.values[key].detach().cpu().numpy()],axis=1 if key=='temporal_dc' else 0)
        recovered=out[key][:,:value.shape[1]] if key.startswith('track_') or key=='temporal_dc' else out[key][:len(value)]
        if not np.array_equal(recovered,value):raise AssertionError('Protected array changed: '+key)
    np.savez_compressed(a.output/'rigid-bucket.npz',**out);packed=Bucket(read_arrays(a.output/'rigid-bucket.npz',3))
    with torch.no_grad():
        for timestamp in [0.,1.67,398/60]:
            trained=body.proxy(protected.proxy(r.model,timestamp),timestamp);loaded=packed.proxy(r.model,timestamp)
            for field in ['get_static_xyz','get_static_rotation','get_static_scaling','get_static_features','get_static_opacity']:
                torch.testing.assert_close(getattr(trained,field),getattr(loaded,field),rtol=0,atol=0)
    del packed
    total_iteration=max(iteration,int(r.meta.get('training_iteration',0)))+a.steps
    hybrid.save_bounded(hybrid.compact((retained,total_iteration)),a.output/'model.pth',512*1024**2)
    (a.output/'effective.yaml').write_bytes((a.model/'effective.yaml').read_bytes())
    evidence.update(elapsed=time.monotonic()-started,video_sha256=digest(a.video),views_sha256=digest(a.views/'views.json'),packed_proxy_exact=True,training_script_sha256=digest(Path(__file__)),proxy_script_sha256=digest(Path(__file__).with_name('rigid_bucket.py')))
    result=dict(r.meta);result.update(sha256=digest(a.output/'model.pth'),trained_at=model_timestamp(),training_iteration=total_iteration,body_refinement=evidence)
    result['rigid_bucket']=dict(r.meta['rigid_bucket'],schema=3,sha256=digest(a.output/'rigid-bucket.npz'),points=len(out['points']),body_points=n,motion_groups=8,source='Seven preserved articulated groups plus explicitly supplied fixed source surfaces; not monocular reconstruction')
    result['bytes']=usage(a.output);(a.output/'manifest.json').write_text(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['repo','model','source','video','views','output','models-root']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,choices=[10,6000,12000],default=6000);p.add_argument('--registry',type=Path,action='append',default=[])
    a=p.parse_args();records=[];lock=a.models_root/'.fvv-writer.lock';signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(KeyboardInterrupt()))
    with lock.open('x') as f:f.write('fixed body training')
    try:pause(a.registry,records);fit(a)
    finally:
        try:resume(records)
        finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
