"""Fit a bounded canonical Gaussian bucket; preserve every remaining base row.

Explicit supplied mesh motion is a prior. Never automatically deploy a candidate.
"""
import argparse,json,os,signal,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import hybrid
from common import digest,usage,model_timestamp
from rigid_bucket import Bucket,bucket_only_model
from seed_covariance import static_surface_rotation
from refine_lego import decode,camera,view_camera
from training_resources import pause,resume


def fit(a):
    torch.set_num_threads(4);torch.manual_seed(1101);rng=np.random.default_rng(1101)
    if a.resume and a.points>32768:raise ValueError('Dense temporal colors require at most 32768 points to stay within 32 MiB decoded storage')
    a.output.mkdir(exist_ok=False)
    if usage(a.output.parent)>8*1024**3:raise OSError('Experiment budget exceeds 8 GiB')
    if digest(a.video)!='5cbf3042a577359a420d694e0117e2de715e2c5671c7a9fb7f5519917ed0a3bd':raise ValueError('Unexpected real Lego video')
    r=hybrid.Renderer(SimpleNamespace(repo=a.repo,model=a.model))
    if r.bucket is not None:raise ValueError('Initial bucket fit requires a plain parent')
    from training_kernel import select
    kernel=select(a.repo)
    with np.load(a.source,allow_pickle=False) as data:
        source_meta=json.loads(str(data['metadata_json']));ids=data['indices'];points=data['points'];normals=data['normals'];labels=data['parts'];tracks=data['tracks']
    if source_meta['parent_sha256']!=r.meta['sha256']:raise ValueError('Bucket diagnosis belongs to another parent')
    if not np.isfinite(points).all() or len(points)>1048576 or tracks.shape!=(399,4,4):raise ValueError('Invalid source budget')
    # Bounded spatial stratification avoids independent time-window duplicates.
    cells=np.concatenate((np.floor(points/.0015).astype(np.int64),labels[:,None],np.rint(normals*8).astype(np.int64)),axis=1)
    _,unique=np.unique(cells,axis=0,return_index=True)
    chosen=np.sort(rng.choice(unique,min(a.points,len(unique)),replace=False))
    points=points[chosen];normals=normals[chosen];normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    surface_pairs=None
    if a.surface_color_weight:
        if not a.bucket_views or not a.resume:raise ValueError('Surface color fitting requires all-face resume')
        chosen_labels=labels[chosen];dist,near=cKDTree(points).query(points,k=5)
        first=np.repeat(np.arange(len(points)),4);second=near[:,1:].reshape(-1);offset=points[first]-points[second]
        keep=(first<second)&(dist[:,1:].reshape(-1)<.008)&(chosen_labels[first]==chosen_labels[second])
        keep&=(normals[first]*normals[second]).sum(1)>.98
        keep&=(np.abs((offset*normals[first]).sum(1))<.0008)&(np.abs((offset*normals[second]).sum(1))<.0008)
        pairs=np.stack([first[keep],second[keep]],axis=1)
        if not 1000<=len(pairs)<=131072:raise ValueError('Unexpected surface pair budget')
        surface_pairs=torch.from_numpy(pairs).cuda()
    n=len(points);rotation=Rotation.from_matrix(tracks[:,:3,:3]).as_quat()[:,[3,0,1,2]].astype(np.float32)
    arrays=dict(points=points,rotations=static_surface_rotation(normals),scales=np.tile([.0028,.0028,.0005],(n,1)).astype(np.float32),
        dc=np.full((n,1,3),(.55-.5)/.28209479177387814,np.float32),rest=np.zeros((n,15,3),np.float32),opacity=np.full((n,1),2.2,np.float32),
        track_rotations=rotation,track_translations=tracks[:,:3,3],temporal_dc=np.zeros((16,n,1,3),np.float32))
    prior_steps=0;prior_sha=None
    bucket=Bucket(arrays);groups=[]
    if a.resume:
        prior_meta=json.loads((a.resume/'manifest.json').read_text())['rigid_bucket']
        if prior_meta['parent_sha256']!=r.meta['sha256'] or prior_meta['source_sha256']!=digest(a.source):raise ValueError('Bucket resume parent/source mismatch')
        previous=Bucket.load(a.resume,prior_meta)
        for key in ['points','rotations','scales','track_rotations','track_translations']:
            if not torch.equal(bucket.values[key],previous.values[key]):raise ValueError('Bucket resume geometry mismatch')
        for key in ['dc','rest','opacity']:bucket.values[key]=previous.values[key]
        temporal=previous.values['temporal_dc']
        bucket.values['temporal_dc']=torch.nn.functional.interpolate(temporal.permute(1,2,3,0).reshape(-1,1,len(temporal)),size=48,mode='linear',align_corners=True).reshape(n,1,3,48).permute(3,0,1,2).contiguous()
        prior_steps=prior_meta['steps'];prior_sha=prior_meta['sha256'];del previous
    rates=[('dc',.003),('rest',.00015),('opacity',.001),('temporal_dc',.002)] if a.resume else [('dc',.01),('rest',.0005),('opacity',.003),('temporal_dc',.003)]
    for key,lr in rates:
        value=torch.nn.Parameter(bucket.values[key]);bucket.values[key]=value;groups.append(dict(params=[value],lr=lr))
    optimizer=torch.optim.Adam(groups,eps=1e-15)
    state,iteration=hybrid.load_checkpoint(a.model/'model.pth','cpu');state=list(state)
    keep=torch.ones(len(state[1]),dtype=torch.bool)
    if ids.dtype.kind not in 'iu' or ids.min()<0 or ids.max()>=len(keep) or len(np.unique(ids))!=len(ids):raise ValueError('Invalid retirement indices')
    keep[torch.from_numpy(ids)]=False
    for field in [1,2,3,4,5,6,13,14,15]:state[field]=state[field][keep]
    # The kept tensors are never optimizer parameters. Save this exact subset.
    retained=tuple(state)
    r.model.restore(tuple(v.cuda() if torch.is_tensor(v) else v for v in retained),None)
    for name in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','_t','_scaling_t','_rotation_r','static_xyz','static_features_dc','static_features_rest','static_scaling','static_rotation','static_opacity']:
        getattr(r.model,name).requires_grad_(False)
    r.render_fn=bucket.wrap(r.base_render_fn)
    frames=decode(a.video);views=[]
    for view in json.loads((a.views/'views.json').read_text())['views']:
        if view['split']=='train':views.append((view_camera(view),torch.from_numpy(np.array(Image.open(a.views/view['file']).convert('RGB')))))
    if not views:raise ValueError('No source appearance views')
    bucket_views=[]
    if a.bucket_views:
        if not a.resume:raise ValueError('All-face refinement requires an existing rigid bucket')
        manifest=json.loads((a.bucket_views/'views.json').read_text())
        if manifest.get('isolated_bucket') is not True:raise ValueError('Expected isolated bucket RGB references')
        if len(manifest['views'])!=manifest['expected_views'] or manifest['expected_views']>128:raise ValueError('Incomplete or oversized bucket references')
        for view in manifest['views']:
            if view['split']=='train':bucket_views.append((view_camera(view),torch.from_numpy(np.array(Image.open(a.bucket_views/view['file']).convert('RGB')))))
        if len(bucket_views)<24:raise ValueError('Insufficient all-face references')
    bucket_order=[]
    bucket_only=bucket_only_model(r.model)
    from utils.loss_utils import ssim
    cams=[camera(r,i/60) for i in range(399)]
    started=time.monotonic();order=[]
    def loss_at(cam,pixels,isolated=False):
        target=pixels.cuda().permute(2,0,1).float()/255
        image=r.render_fn(cam,bucket_only if isolated else r.model,r.pipe,r.bg)['render']
        weight=1+3*(target.mean(0)>.03).float()
        l1=((image-target).abs()*weight).mean()/weight.mean()
        edge=((image[:,:,1:]-image[:,:,:-1])-(target[:,:,1:]-target[:,:,:-1])).abs().mean()+((image[:,1:,:]-image[:,:-1,:])-(target[:,1:,:]-target[:,:-1,:])).abs().mean()
        return .8*l1+.2*(1-ssim(image,target))+.2*edge
    for step in range(1,a.steps+1):
        if time.monotonic()-started>3600:raise TimeoutError('Bucket fit exceeded one hour')
        optimizer.zero_grad(set_to_none=True)
        frame=int(rng.integers(399));loss=8*loss_at(cams[frame],torch.from_numpy(frames[frame].copy()));loss.backward();logged=float(loss.detach())
        for _ in range(2):
            if not order:order=rng.permutation(len(views)).tolist()
            cam,pixels=views[order.pop()];loss=1.5*loss_at(cam,pixels);loss.backward();logged+=float(loss.detach())
        for _ in range(2 if bucket_views else 0):
            if not bucket_order:bucket_order=rng.permutation(len(bucket_views)).tolist()
            cam,pixels=bucket_views[bucket_order.pop()];loss=4*loss_at(cam,pixels,isolated=True);loss.backward();logged+=float(loss.detach())
        temporal_logged=0.
        if a.resume:
            index=int(rng.integers(398));first=r.render_fn(cams[index],r.model,r.pipe,r.bg)['render'];second=r.render_fn(cams[index+1],r.model,r.pipe,r.bg)['render']
            observed=torch.from_numpy((frames[index+1].astype(np.float32)-frames[index].astype(np.float32))/255).cuda().permute(2,0,1)
            temporal_loss=8*((second-first-observed)[:,40:470,40:470]).abs().mean()
            temporal_loss.backward();temporal_logged=float(temporal_loss.detach());logged+=temporal_logged
        temporal=bucket.values['temporal_dc'];regularization=.005*temporal.square().mean()+.02*(temporal[2:]-2*temporal[1:-1]+temporal[:-2]).square().mean()
        regularization+=.002*(1-bucket.values['opacity'].sigmoid()).square().mean()
        if surface_pairs is not None:
            selected=surface_pairs[torch.randint(len(surface_pairs),(min(8192,len(surface_pairs)),),device='cuda')];left,right=selected.unbind(1)
            dc=bucket.values['dc'];rest=bucket.values['rest']
            regularization+=a.surface_color_weight*((dc[left]-dc[right]).square().mean()+.1*(rest[left]-rest[right]).square().mean()+.25*(temporal[:,left]-temporal[:,right]).square().mean())
        regularization.backward();optimizer.step()
        with torch.no_grad():
            bucket.values['dc'].clamp_(-1.7,1.77);bucket.values['temporal_dc'].clamp_(-2,2);bucket.values['opacity'].clamp_(-4,8)
        if not np.isfinite(logged):raise ValueError('Nonfinite bucket loss')
        if step==1 or step%100==0:
            row=dict(step=step,loss=logged,temporal=temporal_logged,regularization=float(regularization.detach()),elapsed=time.monotonic()-started)
            with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if step%500==0 or step==a.steps:
            panel=Image.new('RGB',(4*320,2*320))
            preview=json.loads((a.views/'views.json').read_text())['views'];preview=[v for v in preview if v['split']=='validation'][:8]
            with torch.no_grad():
                for i,view in enumerate(preview):
                    image=r.render_fn(view_camera(view),r.model,r.pipe,r.bg)['render'].clamp(0,1)
                    panel.paste(Image.fromarray((image.permute(1,2,0)*255).byte().cpu().numpy()).resize((320,320)),(i%4*320,i//4*320))
            panel.save(a.output/'preview.jpg',quality=95)
    # Check all retained base parameters before writing a deployment-ready artifact.
    for field,name in [(1,'_xyz'),(2,'_features_dc'),(3,'_features_rest'),(4,'_scaling'),(5,'_rotation'),(6,'_opacity'),(13,'_t'),(14,'_scaling_t'),(15,'_rotation_r'),(19,'static_xyz'),(20,'static_features_dc'),(21,'static_features_rest'),(22,'static_scaling'),(23,'static_rotation'),(24,'static_opacity')]:
        if not torch.equal(getattr(r.model,name).detach().cpu(),retained[field]):raise AssertionError('A retained parent parameter changed')
    hybrid.save_bounded(hybrid.compact((retained,iteration+prior_steps+a.steps)),a.output/'model.pth',512*1024**2)
    (a.output/'effective.yaml').write_bytes((a.model/'effective.yaml').read_bytes())
    out={k:v.detach().cpu().numpy().astype(np.float32) for k,v in bucket.values.items()}
    temporary=a.output/'rigid-bucket.partial.npz'
    try:
        np.savez_compressed(temporary,**out)
        if temporary.stat().st_size>32*1024**2:raise OSError('Bucket sidecar exceeds 32 MiB')
        temporary.replace(a.output/'rigid-bucket.npz')
    finally:temporary.unlink(missing_ok=True)
    meta=dict(r.meta);meta.pop('seed_initialization',None)
    meta.update(sha256=digest(a.output/'model.pth'),trained_at=model_timestamp(),training_iteration=iteration+prior_steps+a.steps,
        refinement=dict(kind='rigid_bucket',parent_sha256=r.meta['sha256'],steps=a.steps,all_frames=True),
        rigid_bucket=dict(schema=1,trained=True,sha256=digest(a.output/'rigid-bucket.npz'),points=n,retired_dynamic=len(ids),steps=prior_steps+a.steps,
            parent_sha256=r.meta['sha256'],source_sha256=digest(a.source),source='Explicit supplied Lego mesh and rigid motion; not monocular reconstruction',
            allface_views_sha256=digest(a.bucket_views/'views.json') if a.bucket_views else None,allface_train_views=len(bucket_views),
            preserved_nonbucket_exact=True,temporal_color_knots=len(bucket.values['temporal_dc']),geometry_trainable=False))
    evidence=dict(**meta['rigid_bucket'],training_kernel=kernel,video_sha256=digest(a.video),views_sha256=digest(a.views/'views.json'),
        frames=399,production_fit=True,validation_scope='Extra source RGB views; supplied geometry is not held out',elapsed_seconds=time.monotonic()-started,
        allface_weight=4 if bucket_views else 0,surface_color_weight=a.surface_color_weight,surface_color_pairs=len(surface_pairs) if surface_pairs is not None else 0,
        tangent_sigma=.0028,normal_sigma=.0005,learning_rates=dict(rates),previous_bucket_sha256=prior_sha,adjacent_rgb_weight=8 if a.resume else 0)
    (a.output/'bucket-training.json').write_text(json.dumps(evidence,indent=2));meta['bytes']=usage(a.output)
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(evidence),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['repo','model','source','video','views','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--points',type=int,choices=[16384,32768,65536],default=32768)
    p.add_argument('--steps',type=int,choices=[1000,3000,6000],default=3000)
    p.add_argument('--resume',type=Path,help='Continue only an identical source bucket with denser smooth color tracks and adjacent-RGB supervision')
    p.add_argument('--bucket-views',type=Path,help='Additional complete bucket-centered angular RGB references; validation split excluded')
    p.add_argument('--surface-color-weight',type=float,choices=[0.,.05],default=0.,help='Same-part coplanar neighbor color regularization; never crosses sharp normal boundaries')
    p.add_argument('--registry',type=Path,action='append',default=[])
    a=p.parse_args();records=[];lock=a.model.parent/'.fvv-writer.lock'
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(KeyboardInterrupt()))
    with lock.open('x') as f:f.write('isolated rigid bucket training')
    try:pause(a.registry,records);fit(a)
    finally:
        try:resume(records)
        finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
