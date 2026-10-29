"""Bounded real-video refinement; fixed point count and motion-aware ROI loss.

Use only the native training rasterizer. Never use the inference-only kernel
for backward. The original model and video are read-only inputs.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
import hybrid
from common import digest, usage


def camera(r, timestamp):
    from utils.graphics_utils import getProjectionMatrix
    world=torch.eye(4,device='cuda')
    fov=math.radians(r.meta['fov'])
    return SimpleNamespace(FoVx=fov,FoVy=fov,image_width=640,image_height=640,
        timestamp=timestamp,world_view_transform=world,
        full_proj_transform=getProjectionMatrix(znear=.01,zfar=10000.,fovX=fov,fovY=fov).T.cuda(),
        camera_center=torch.zeros(3,device='cuda'))


def view_camera(view):
    from utils.graphics_utils import getProjectionMatrix
    c2w=torch.tensor(view['c2w'],device='cuda',dtype=torch.float32)
    world=torch.linalg.inv(c2w).T.contiguous()
    fov=math.radians(view['fov'])
    projection=getProjectionMatrix(znear=.01,zfar=10000.,fovX=fov,fovY=fov).T.cuda()
    return SimpleNamespace(FoVx=fov,FoVy=fov,image_width=640,image_height=640,
        timestamp=view['time'],world_view_transform=world,full_proj_transform=world@projection,
        camera_center=c2w[:3,3].contiguous())


def decode(video):
    # Hard output bound: at most 400 real 640x640 RGB frames in RAM, no frame cache.
    proc=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(video),'-frames:v','400',
        '-f','rawvideo','-pix_fmt','rgb24','-'],check=True,stdout=subprocess.PIPE)
    pixels=np.frombuffer(proc.stdout,dtype=np.uint8)
    if pixels.size != 400*640*640*3:raise ValueError('Expected 400 real 640x640 frames')
    return pixels.reshape(400,640,640,3)


def run(args):
    torch.set_num_threads(4);torch.manual_seed(6666);np.random.seed(6666)
    if not 1<=args.steps<=10000:raise ValueError('Steps must be 1..10000')
    if digest(args.video) != '5cbf3042a577359a420d694e0117e2de715e2c5671c7a9fb7f5519917ed0a3bd':
        raise ValueError('Video differs from verified real Lego source')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    if usage(root.parent)>8*1024**3:raise OSError('Model root exceeds 8 GiB')
    r=hybrid.Renderer(SimpleNamespace(repo=args.repo.resolve(),model=args.model.resolve()))
    frames=decode(args.video)
    # Verify decoder/color/time alignment against existing real references.
    for idx in [0,100,200,398]:
        target=np.array(Image.open(args.dataset/'images'/f'{idx:06d}.png').convert('RGB'))
        if not np.array_equal(target,frames[idx]):raise ValueError('Decoded RGB does not match reference')
    # Development mode preserves old holdouts. Production mode fits every
    # real frame inside the existing [0, 398/60] playback interval.
    excluded=set() if args.all_frames else set(range(0,399,10))|set(range(11,399,20))
    training=[i for i in range(399) if i not in excluded]
    pairs=[i for i in training if i+1 in training]
    removed_pathological=0
    if args.multiview:
        with torch.no_grad():
            covariance,_=r.model.get_current_covariance_and_mean_offset(1.,0.)
            keep=covariance[:,[0,3,5]].clamp_min(0).sqrt().amax(1)<=.25
            removed_pathological=int((~keep).sum())
            state,_=hybrid.load_checkpoint(args.model/'model.pth','cuda');state=list(state)
            for field in [1,2,3,4,5,6,13,14,15]:state[field]=state[field][keep]
            r.model.restore(tuple(state),None)
            # Saturated sigmoid opacities have zero float32 gradient. Keep
            # them trainable during multiview correction (alpha <= .999665).
            r.model._opacity.clamp_(max=8.)
    names={'_xyz':args.position_lr,'_features_dc':.001,'_features_rest':.00005,
           '_opacity':args.opacity_lr if args.opacity_lr is not None else (.02 if args.multiview else .005),'_scaling':.0005,'_rotation':.0001,
           '_t':.0001,'_scaling_t':.0005,'_rotation_r':.0001}
    if args.multiview and args.train_static:
        with torch.no_grad():r.model.static_opacity.clamp_(max=8.)
        names.update({'static'+name:lr for name,lr in list(names.items())
                      if name in ['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation']})
    groups=[]
    for name,lr in names.items():
        param=torch.nn.Parameter(getattr(r.model,name).detach())
        setattr(r.model,name,param);groups.append(dict(params=[param],lr=lr))
    # Static geometry is opt-in for source-scene multiview correction only.
    # Point counts stay bounded; no densification or optimizer checkpoint.
    optimizer=torch.optim.Adam(groups,eps=1e-15)
    from utils.loss_utils import ssim
    weight=torch.ones((1,640,640),device='cuda')
    weight[:,40:470,40:470]=4
    weight/=weight.mean()
    cams=[camera(r,i/60) for i in range(399)]
    novel=[]
    if args.multiview:
        manifest=json.loads((args.multiview/'views.json').read_text());views=manifest['views']
        if len(views)!=manifest['expected_views']:raise ValueError('Source-scene export is incomplete')
        for view in views:
            if view['split']=='train':
                with Image.open(args.multiview/view['file']) as image:
                    if image.size!=(640,640):raise ValueError('Expected 640x640 source-scene views')
                    pixels=torch.from_numpy(np.array(image.convert('RGB')))
                novel.append((view_camera(view),pixels))
        if not novel:raise ValueError('Multiview export contains no training views')
    started=time.monotonic()
    for step in range(1,args.steps+1):
        if time.monotonic()-started>20*60:raise TimeoutError('Refinement exceeded 20 minute budget')
        i=int(np.random.choice(pairs));optimizer.zero_grad(set_to_none=True)
        images=[];targets=[]
        for j in [i,i+1]:
            target=torch.tensor(frames[j],device='cuda',dtype=torch.float32).permute(2,0,1)/255
            package=r.render_fn(cams[j],r.model,r.pipe,r.bg)
            out=package['render']
            images.append(out);targets.append(target)
        appearance=sum(.8*((im-gt).abs()*weight).mean()+.2*(1-ssim(im,gt)) for im,gt in zip(images,targets))/2
        # Match actual observed brightness/motion change, not image constancy.
        temporal=(((images[1]-images[0])-(targets[1]-targets[0])).abs()*weight).mean()
        loss=appearance+.5*temporal
        if args.edge_weight:
            edge_loss=0
            for image,target in zip(images,targets):
                error=image-target
                edge_loss=edge_loss+(error[:,:,1:]-error[:,:,:-1]).abs().mean()+(error[:,1:,:]-error[:,:-1,:]).abs().mean()
            loss=loss+args.edge_weight*edge_loss/4
        if args.midpoint_weight:
            # Temporal regularization only on stable interiors. Exclude moving
            # edges and their 5x5 neighborhood to avoid double edges/blur.
            changed=(targets[1]-targets[0]).abs().amax(0,keepdim=True)>.04
            excluded_edge=torch.nn.functional.max_pool2d(changed.float().unsqueeze(0),5,1,2)[0]
            stable=(1-excluded_edge)*weight
            middle=r.render_fn(camera(r,(i+.5)/60),r.model,r.pipe,r.bg)['render']
            midpoint=((middle-(targets[0]+targets[1])/2).abs()*stable).sum()/(3*stable.sum().clamp_min(1))
            loss=loss+args.midpoint_weight*midpoint
        if novel:
            cam,pixels=novel[int(np.random.randint(len(novel)))]
            target=pixels.cuda().permute(2,0,1).float()/255
            image=r.render_fn(cam,r.model,r.pipe,r.bg)['render']
            # Source-scene views supervise geometry and occlusion; retain full
            # original-video appearance and temporal losses every iteration.
            novel_weight=1+3*(target.mean(0)>8/255).float()
            novel_weight/=novel_weight.mean()
            loss=loss+args.multiview_weight*(.8*((image-target).abs()*novel_weight).mean()+.2*(1-ssim(image,target)))
        if not torch.isfinite(loss):raise ValueError('Non-finite refinement loss')
        loss.backward();optimizer.step()
        if args.multiview:
            with torch.no_grad():
                r.model._opacity.clamp_(max=8.)
                if args.train_static:r.model.static_opacity.clamp_(max=8.)
        if step%100==0 or step==1:
            row=dict(step=step,loss=float(loss),temporal=float(temporal),elapsed=time.monotonic()-started)
            with (root/'refinement.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    # capture() expects optimizer, but compact() discards its states immediately.
    r.model.optimizer=optimizer
    parent_iteration=int(hybrid.load_checkpoint(args.model/'model.pth','cpu')[1])
    artifact=hybrid.compact((r.model.capture(),parent_iteration+args.steps))
    hybrid.save_bounded(artifact,root/'model.pth',512*1024**2)
    (root/'effective.yaml').write_bytes((args.model/'effective.yaml').read_bytes())
    meta=dict(r.meta);meta.update(sha256=digest(root/'model.pth'),config_sha256=digest(root/'effective.yaml'),
        bytes=usage(root),budget_bytes=512*1024**2,training_iteration=parent_iteration+args.steps,
        refinement=dict(parent_sha256=r.meta['sha256'],steps=args.steps,all_frames=args.all_frames,
                        midpoint_weight=args.midpoint_weight))
    (root/'refinement-provenance.json').write_text(json.dumps(dict(
        parent_sha256=r.meta['sha256'],video_sha256=digest(args.video),steps=args.steps,
        evaluation_scope='in-sequence production fit; no held-out quality claim' if args.all_frames else 'preserved development holdouts',
        training_frames=training,excluded_frames=sorted(excluded),time_step=1/60,
        roi_xyxy=[40,40,470,470],roi_weight=4,temporal_loss_weight=.5,
        multiview_manifest_sha256=digest(args.multiview/'views.json') if args.multiview else None,
        multiview_training_count=len(novel),multiview_foreground_weight=4,position_lr=args.position_lr,
        train_static=bool(args.multiview and args.train_static),parameter_learning_rates=names,
        multiview_loss_weight=args.multiview_weight,
        removed_pathological_splats=removed_pathological,opacity_logit_cap=8 if args.multiview else None,
        edge_weight=args.edge_weight,
        midpoint_weight=args.midpoint_weight,midpoint_scope='stable-interior interpolation regularizer, not measured midpoint ground truth',
        temporal_loss='L1 difference from observed real adjacent RGB differences',
        dynamic_points=len(r.model.get_xyz),static_points=len(r.model.get_static_xyz),
        elapsed_seconds=time.monotonic()-started),indent=2))
    meta['bytes']=usage(root)
    # Publish readiness only after both the artifact and provenance are complete.
    (root/'manifest.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['repo','model','video','dataset','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--multiview',type=Path)
    p.add_argument('--multiview-weight',type=float,default=1.5,choices=[.25,.5,1.,1.5])
    p.add_argument('--opacity-lr',type=float,choices=[.001,.003,.005,.02])
    p.add_argument('--position-lr',type=float,default=1e-6,choices=[1e-6,3e-6,1e-5,3e-5,1e-4])
    p.add_argument('--train-static',action='store_true')
    p.add_argument('--edge-weight',type=float,default=0,choices=[0,0.25,0.5,1])
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--midpoint-weight',type=float,default=0,choices=[0,0.5,1,2])
    p.add_argument('--all-frames',action='store_true',help='Production fit: use all observed timestamps; no holdout claim')
    args=p.parse_args()
    # Coordinate with UI training/import jobs; only remove a lock we created.
    lock=args.output.resolve().parent/'.fvv-writer.lock'
    with lock.open('x') as stream:
        stream.write('real Lego ROI refinement\n')
    try:
        run(args)
    finally:
        lock.unlink(missing_ok=True)
