"""Bounded real-video refinement; fixed point count and motion-aware ROI loss.

Use a training rasterizer (native or the opt-in reference pixel backward).
Never use the inference-only kernel for backward. Inputs stay read-only.
"""
import argparse
import json
import math
import shutil
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
import hybrid
from common import digest, usage, model_timestamp


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
    if not 1<=args.steps<=30000:raise ValueError('Steps must be 1..30000')
    if args.curvature_weight and not args.midpoint_weight:raise ValueError('Curvature requires midpoint regularization')
    if args.geometry_weight and not args.geometry:raise ValueError('Geometry weight requires an explicit surface prior')
    if args.surface_normal_weight and not (args.geometry and args.geometry_weight):raise ValueError('Normal constraint requires geometric supervision')
    if digest(args.video) != '5cbf3042a577359a420d694e0117e2de715e2c5671c7a9fb7f5519917ed0a3bd':
        raise ValueError('Video differs from verified real Lego source')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    if usage(root.parent)>8*1024**3:raise OSError('Model root exceeds 8 GiB')
    r=hybrid.Renderer(SimpleNamespace(repo=args.repo.resolve(),model=args.model.resolve()))
    preserve_rigid=getattr(args,'preserve_rigid',False)
    whole_scene=getattr(args,'whole_scene',False)
    joint_rigid=getattr(args,'joint_rigid',False)
    detail_weight=getattr(args,'detail_weight',0.)
    if joint_rigid and not (preserve_rigid and whole_scene):raise ValueError('Joint detail fitting requires preserved rigid geometry and whole scene')
    rigid_original=None
    if r.meta.get('rigid_bucket') and not preserve_rigid:raise ValueError('Rigid models require --preserve-rigid; refinement must retain their sidecar')
    if preserve_rigid and (r.bucket is None or args.geometry_weight or args.surface_color_pairs):raise ValueError('Preserved articulation requires a rigid model and composite ray supervision, not unmatched whole-source point indices')
    rigid_sha=r.meta['rigid_bucket']['sha256'] if r.bucket is not None else None
    ray_source_sha256=None
    if args.ray_prior:
        from training_kernel import select
        ray_source_sha256=select(args.repo)
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
            # A seeded refinement promises exact preservation of parent
            # geometry, including its row count and ordering.
            if args.freeze_parent_geometry or args.freeze_geometry:keep=torch.ones_like(keep)
            removed_pathological=int((~keep).sum())
            state,_=hybrid.load_checkpoint(args.model/'model.pth','cuda');state=list(state)
            for field in [1,2,3,4,5,6,13,14,15]:state[field]=state[field][keep]
            r.model.restore(tuple(state),None)
            # Saturated sigmoid opacities have zero float32 gradient. Keep
            # them trainable during multiview correction (alpha <= .999665).
            r.model._opacity.clamp_(max=8.)
    names={'_xyz':args.position_lr,'_features_dc':args.appearance_lr,'_features_rest':args.appearance_lr*.05,
           '_opacity':args.opacity_lr if args.opacity_lr is not None else (.02 if args.multiview else .005),'_scaling':.0005,'_rotation':.0001,
           '_t':.0001,'_scaling_t':.0005,'_rotation_r':.0001}
    if args.multiview and args.train_static:
        with torch.no_grad():r.model.static_opacity.clamp_(max=8.)
        names.update({'static'+name:lr for name,lr in list(names.items())
                      if name in ['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation']})
    groups=[]
    frozen_parent_rows=0
    if args.freeze_parent_geometry:
        added=int(r.meta.get('seed_initialization',{}).get('added',0))
        if not 0<added<len(r.model.get_xyz):raise ValueError('Freezing parent geometry requires appended surface seeds')
        frozen_parent_rows=len(r.model.get_xyz)-added
    frozen_fields=['_xyz','_scaling','_rotation','_t','_scaling_t','_rotation_r']
    frozen_before={}
    protected_static_before={}
    if r.meta.get('source_surface_initialization',{}).get('static_appearance_sha256') and not args.train_static:
        protected_static_before={name:getattr(r.model,name).detach().cpu().clone() for name in
            ['static_xyz','static_scaling','static_rotation','static_features_dc','static_features_rest','static_opacity']}
    if args.freeze_geometry:
        if args.freeze_parent_geometry:raise ValueError('Choose only one geometry freeze mode')
        if not r.meta.get('source_surface_initialization'):raise ValueError('Full geometry freeze requires explicit source-surface initialization')
        for name in list(names):
            if name in frozen_fields or name.startswith('static') and name[6:] in frozen_fields:
                value=getattr(r.model,name).detach();frozen_before[name]=value.cpu().clone()
                setattr(r.model,name,value);del names[name]
    elif args.freeze_static_geometry:
        for name in list(names):
            if name.startswith('static') and name[6:] in frozen_fields:
                value=getattr(r.model,name).detach();frozen_before[name]=value.cpu().clone()
                setattr(r.model,name,value);del names[name]
    for name,lr in names.items():
        param=torch.nn.Parameter(getattr(r.model,name).detach())
        if frozen_parent_rows and (name in frozen_fields or name.startswith('static') and name[6:] in frozen_fields):
            stop=len(param) if name.startswith('static') else frozen_parent_rows
            frozen_before[name]=param[:stop].detach().cpu().clone()
            def preserve_parent(gradient,stop=stop):
                result=gradient.clone();result[:stop]=0;return result
            param.register_hook(preserve_parent)
        setattr(r.model,name,param);groups.append(dict(params=[param],lr=lr))
    # Static geometry is opt-in for source-scene multiview correction only.
    # Point counts stay bounded; no densification or optimizer checkpoint.
    if joint_rigid:
        from rigid_bucket import read_arrays
        rigid_original=read_arrays(args.model/'rigid-bucket.npz',r.meta['rigid_bucket']['schema'])
        for key,lr in [('dc',.0005),('rest',.000025),('opacity',.00025),('temporal_dc',.0001)]:
            param=torch.nn.Parameter(r.bucket.values[key].detach().clone())
            r.bucket.values[key]=param;groups.append(dict(params=[param],lr=lr))
    optimizer=torch.optim.Adam(groups,eps=1e-15)
    color_pairs=None; color_pair_metadata=None
    if args.surface_color_pairs:
        if not args.freeze_geometry:raise ValueError('Surface color neighbors require frozen source geometry')
        if args.surface_color_pairs.stat().st_size>32*1024**2:raise ValueError('Surface color neighbors exceed budget')
        with np.load(args.surface_color_pairs,allow_pickle=False) as data:
            cp=data['pairs'];cw=data['weights'];color_pair_metadata=json.loads(str(data['metadata_json']))
        source=r.meta.get('source_surface_initialization',{})
        if color_pair_metadata['source_sha256']!=source.get('source_sha256') or color_pair_metadata['dynamic_points']!=len(r.model.get_xyz):raise ValueError('Surface color source does not match model')
        if cp.ndim!=2 or cp.shape[1]!=2 or not 1<=len(cp)<=len(r.model.get_xyz) or cw.shape!=(len(cp),):raise ValueError('Invalid surface neighbor shape')
        if cp.dtype.kind not in 'iu' or cp.min()<0 or cp.max()>=len(r.model.get_xyz) or not np.isfinite(cw).all() or not ((cw>0)&(cw<=1)).all():raise ValueError('Invalid surface neighbor indices or weights')
        color_pairs=torch.from_numpy(cp.astype(np.int64)).cuda();color_weights=torch.from_numpy(cw.astype(np.float32)).cuda()
    from utils.loss_utils import ssim
    weight=torch.ones((1,640,640),device='cuda')
    if not whole_scene:weight[:,40:470,40:470]=4
    weight/=weight.mean()
    cams=[camera(r,i/60) for i in range(399)]
    novel=[]; validation=[]
    if args.multiview:
        manifest=json.loads((args.multiview/'views.json').read_text());views=manifest['views']
        if len(views)!=manifest['expected_views']:raise ValueError('Source-scene export is incomplete')
        for view in views:
            if view['split'] in ['train','validation']:
                with Image.open(args.multiview/view['file']) as image:
                    if image.size!=(640,640):raise ValueError('Expected 640x640 source-scene views')
                    pixels=torch.from_numpy(np.array(image.convert('RGB')))
                (novel if view['split']=='train' else validation).append((view_camera(view),pixels))
        if not novel:raise ValueError('Multiview export contains no training views')
    rays=[];ray_order=[]
    if args.ray_prior:
        ray_manifest=json.loads((args.ray_prior/'views.json').read_text())
        if len(ray_manifest['views'])!=ray_manifest['expected_views'] or not 1<=len(ray_manifest['views'])<=1536:
            raise ValueError('Incomplete or oversized ray prior')
        for view in ray_manifest['views']:
            if view['split']!='train':continue
            path=(args.ray_prior/view['file']).resolve()
            if not path.is_relative_to(args.ray_prior.resolve()):raise ValueError('Invalid ray file')
            with np.load(path,allow_pickle=False) as data:
                depth=data['inverse_depth'];parts=data['parts'];moving=data['moving']
                if depth.shape not in [(160,160),(320,320)] or parts.shape!=depth.shape or moving.shape!=depth.shape or not np.isfinite(depth).all() or (depth<0).any():
                    raise ValueError('Invalid ray prior arrays')
                cam=view_camera(view);cam.image_width=args.ray_render_width;cam.image_height=args.ray_render_width
                rays.append((cam,torch.from_numpy(depth.copy()),torch.from_numpy(parts.astype(np.int64)),torch.from_numpy(moving.copy())))
        if not rays:raise ValueError('Ray prior contains no training views')
    started=time.monotonic()
    geometry=None;surface_normals=None
    if args.geometry:
        with np.load(args.geometry,allow_pickle=False) as prior:
            points=prior['points'];times=prior['times']
            if points.ndim!=3 or points.shape[1:]!=(8192,3) or not 1<=len(times)<=64 or len(points)!=len(times) or not np.isfinite(points).all() or not np.isfinite(times).all():
                raise ValueError('Invalid bounded surface prior')
            if min(times)<r.meta['time_duration'][0] or max(times)>r.meta['time_duration'][1]+1e-5:raise ValueError('Surface prior times outside model')
            geometry=(times,torch.tensor(points,device='cuda'))
            if args.surface_normal_weight:
                normals=prior['normals']
                if normals.shape!=points.shape or not np.isfinite(normals).all() or not np.allclose(np.linalg.norm(normals,axis=-1),1,atol=1e-4):
                    raise ValueError('Invalid surface normals')
                surface_normals=torch.tensor(normals,device='cuda')
    novel_order=[]
    for step in range(1,args.steps+1):
        if time.monotonic()-started>args.max_minutes*60:raise TimeoutError('Refinement exceeded configured time budget')
        i=int(np.random.choice(pairs));optimizer.zero_grad(set_to_none=True)
        images=[];targets=[]
        for j in [i,i+1]:
            target=torch.tensor(frames[j],device='cuda',dtype=torch.float32).permute(2,0,1)/255
            package=r.render_fn(cams[j],r.model,r.pipe,r.bg)
            out=package['render']
            images.append(out);targets.append(target)
        if whole_scene:
            weight=1+3*((targets[0].mean(0)>8/255)|(targets[1].mean(0)>8/255)).float()[None]
            weight/=weight.mean()
        appearance=sum(.8*((im-gt).abs()*weight).mean()+.2*(1-ssim(im,gt)) for im,gt in zip(images,targets))/2
        # Match actual observed brightness/motion change, not image constancy.
        temporal=(((images[1]-images[0])-(targets[1]-targets[0])).abs()*weight).mean()
        loss=args.appearance_weight*appearance+args.temporal_weight*temporal
        if detail_weight:
            from detail_supervision import detail_loss
            loss=loss+detail_weight*sum(detail_loss(im,gt) for im,gt in zip(images,targets))/2
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
            fractions=[.5] if args.subframe_samples==1 else [float(np.random.uniform(.1,.5)),float(np.random.uniform(.5,.9))]
            if args.curvature_weight:
                center=float(np.random.uniform(.3,.7))
                fractions=[center-.25,center,center+.25]
            midpoint=0; subframes=[]
            for fraction in fractions:
                middle=r.render_fn(camera(r,(i+fraction)/60),r.model,r.pipe,r.bg)['render']
                subframes.append(middle)
                expected=targets[0]*(1-fraction)+targets[1]*fraction
                midpoint=midpoint+((middle-expected).abs()*stable).sum()/(3*stable.sum().clamp_min(1))/len(fractions)
            loss=loss+args.midpoint_weight*midpoint
            if args.curvature_weight:
                curvature=((subframes[2]-2*subframes[1]+subframes[0]).abs()*stable).sum()/(3*stable.sum().clamp_min(1))
                loss=loss+args.curvature_weight*curvature
        if not torch.isfinite(loss):raise ValueError('Non-finite refinement loss')
        # Backpropagate each view separately: more angular supervision without
        # retaining several rasterizer buffers on the GPU at once.
        loss.backward()
        logged_loss=float(loss.detach())
        for _ in range(args.multiview_batch if novel else 0):
            if not novel_order:novel_order=np.random.permutation(len(novel)).tolist()
            cam,pixels=novel[novel_order.pop()]
            target=pixels.cuda().permute(2,0,1).float()/255
            image=r.render_fn(cam,r.model,r.pipe,r.bg)['render']
            # Source-scene views supervise geometry and occlusion; retain full
            # original-video appearance and temporal losses every iteration.
            novel_weight=1+3*(target.mean(0)>8/255).float()
            novel_weight/=novel_weight.mean()
            view_loss=.8*((image-target).abs()*novel_weight).mean()+.2*(1-ssim(image,target))
            if args.edge_weight:
                error=image-target
                view_loss=view_loss+args.edge_weight*.5*((error[:,:,1:]-error[:,:,:-1]).abs().mean()+(error[:,1:,:]-error[:,:-1,:]).abs().mean())
            if detail_weight:view_loss=view_loss+detail_weight*detail_loss(image,target)
            view_loss=view_loss*args.multiview_weight/args.multiview_batch
            if not torch.isfinite(view_loss):raise ValueError('Non-finite multiview loss')
            view_loss.backward();logged_loss+=float(view_loss.detach())
        ray_logged=0.
        for _ in range(args.ray_batch if rays else 0):
            from ray_supervision import inverse_depth_loss,render_inverse_depth
            if not ray_order:ray_order=np.random.permutation(len(rays)).tolist()
            cam,depth,parts,moving=rays[ray_order.pop()]
            depth_render=r.base_render_fn if r.bucket is not None else r.render_fn
            predicted_depth,alpha=render_inverse_depth(depth_render,cam,r.model,r.pipe,r.bg,return_alpha=True,rigid=r.bucket)
            second_moment=render_inverse_depth(depth_render,cam,r.model,r.pipe,r.bg,moment=2,rigid=r.bucket) if args.ray_spread_weight else None
            ray_loss=args.ray_weight/args.ray_batch*inverse_depth_loss(predicted_depth,depth.cuda(),parts.cuda(),moving.cuda(),alpha,args.ray_alpha_weight,second_moment,args.ray_spread_weight)
            if not torch.isfinite(ray_loss):raise ValueError('Non-finite ray loss')
            ray_loss.backward();ray_logged+=float(ray_loss.detach())
        logged_loss+=ray_logged
        if geometry is not None and args.geometry_weight:
            gi=int(np.random.randint(len(geometry[0])));stamp=float(geometry[0][gi])
            # Coverage attracts centers into missing surfaces, while the weaker
            # reverse term discourages floaters. Random bounded subsets keep
            # this explicit mesh prior independent of render resolution.
            covariance,offset=r.model.get_current_covariance_and_mean_offset(1.,stamp)
            means=r.model.get_xyz+offset
            active=(r.model.get_marginal_t(stamp).detach().flatten()>.1)&(r.model.get_opacity.detach().flatten()>.05)
            means=torch.cat((means[active],r.model.get_static_xyz),dim=0)
            if len(means)==0:raise ValueError('No active centers for geometric coverage')
            selected_indices=torch.randperm(len(means),device='cuda')[:4096]
            selected=means[selected_indices]
            surface=geometry[1][gi] if surface_normals is not None else geometry[1][gi,torch.randperm(8192,device='cuda')[:2048]]
            distances=torch.cdist(surface.unsqueeze(0),selected.unsqueeze(0))[0]
            coverage=distances.min(1).values.mean()+.25*distances.min(0).values.mean()
            extent=covariance[active][:,[0,3,5]].clamp_min(1e-12).sqrt().amax(1)
            broad_splats=(extent-.03).clamp_min(0).mean() if len(extent) else coverage*0
            geometric_loss=args.geometry_weight*(coverage+.5*broad_splats)
            if surface_normals is not None:
                from surface_regularization import normal_surface_loss
                # Constrain moving surfaces; static centers retain coverage loss.
                dynamic_covariance=covariance[active]
                dynamic_selected=selected_indices<len(dynamic_covariance)
                normal_loss=normal_surface_loss(selected[dynamic_selected],
                    dynamic_covariance[selected_indices[dynamic_selected]],surface,
                    surface_normals[gi],distances.detach().argmin(0)[dynamic_selected]) if dynamic_selected.any() else coverage*0
                geometric_loss=geometric_loss+args.surface_normal_weight*normal_loss
            if not torch.isfinite(geometric_loss):raise ValueError('Non-finite geometric loss')
            geometric_loss.backward();logged_loss+=float(geometric_loss.detach())
        color_logged=0.
        if color_pairs is not None:
            from surface_color_regularization import surface_color_loss
            selected=torch.randint(len(color_pairs),(min(65536,len(color_pairs)),),device='cuda')
            color_loss=args.surface_color_weight*surface_color_loss(r.model._features_dc,r.model._features_rest,color_pairs[selected],color_weights[selected])
            if not torch.isfinite(color_loss):raise ValueError('Non-finite surface color loss')
            color_loss.backward();color_logged=float(color_loss.detach());logged_loss+=color_logged
        if joint_rigid:
            temporal_dc=r.bucket.values['temporal_dc']
            selected=torch.randint(temporal_dc.shape[1],(4096,),device='cuda')
            knots=temporal_dc[:,selected]
            regularizer=.005*knots.square().mean()+.02*(knots[2:]-2*knots[1:-1]+knots[:-2]).square().mean()
            regularizer.backward()
        optimizer.step()
        if joint_rigid:
            with torch.no_grad():
                r.bucket.values['dc'].clamp_(-1.7,1.77)
                r.bucket.values['temporal_dc'].clamp_(-2,2)
                r.bucket.values['opacity'].clamp_(-4,8)
        if args.multiview:
            with torch.no_grad():
                r.model._opacity.clamp_(max=8.)
                if args.train_static:r.model.static_opacity.clamp_(max=8.)
        if step%100==0 or step==1:
            row=dict(step=step,loss=logged_loss,temporal=float(temporal),ray=ray_logged,surface_color=color_logged,elapsed=time.monotonic()-started)
            with (root/'refinement.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if validation and (step%1000==0 or step==args.steps):
            metrics=[];preview=Image.new('RGB',(4*320,2*320))
            with torch.no_grad():
                for vi,(cam,pixels) in enumerate(validation):
                    target=pixels.cuda().permute(2,0,1).float()/255
                    prediction=r.render_fn(cam,r.model,r.pipe,r.bg)['render'].clamp(0,1)
                    foreground=target.mean(0)>8/255
                    metrics.append(float(-10*torch.log10((prediction[:,foreground]-target[:,foreground]).square().mean().clamp_min(1e-12))))
                    if vi<8:
                        rgb=(prediction.permute(1,2,0)*255).byte().cpu().numpy()
                        preview.paste(Image.fromarray(rgb).resize((320,320)),((vi%4)*320,(vi//4)*320))
            preview.save(root/'validation-preview.jpg',quality=92)
            with (root/'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step,foreground_psnr=float(np.mean(metrics)),views=len(metrics)))+'\n')
    # capture() expects optimizer, but compact() discards its states immediately.
    for name,original in {**frozen_before,**protected_static_before}.items():
        if not torch.equal(getattr(r.model,name)[:len(original)].detach().cpu(),original):
            raise AssertionError('Protected model parameters changed')
    r.model.optimizer=optimizer
    parent_iteration=max(int(hybrid.load_checkpoint(args.model/'model.pth','cpu')[1]),int(r.meta.get('training_iteration',0)))
    artifact=hybrid.compact((r.model.capture(),parent_iteration+args.steps))
    hybrid.save_bounded(artifact,root/'model.pth',512*1024**2)
    (root/'effective.yaml').write_bytes((args.model/'effective.yaml').read_bytes())
    if rigid_sha:
        if digest(args.model/'rigid-bucket.npz')!=rigid_sha:raise ValueError('Protected rigid weights changed during training')
        if joint_rigid:
            trained={k:v.detach().cpu().numpy() for k,v in r.bucket.values.items()}
            for key in rigid_original:
                if key not in ['dc','rest','opacity','temporal_dc'] and not np.array_equal(trained[key],rigid_original[key]):raise AssertionError('Rigid geometry or trajectory changed: '+key)
            np.savez_compressed(root/'rigid-bucket.npz',**trained)
            read_arrays(root/'rigid-bucket.npz',r.meta['rigid_bucket']['schema'])
        else:
            shutil.copyfile(args.model/'rigid-bucket.npz',root/'rigid-bucket.npz')
            if digest(root/'rigid-bucket.npz')!=rigid_sha:raise ValueError('Protected rigid copy mismatch')
    meta=dict(r.meta);meta.update(trained_at=model_timestamp(),sha256=digest(root/'model.pth'),config_sha256=digest(root/'effective.yaml'),
        bytes=usage(root),budget_bytes=512*1024**2,training_iteration=parent_iteration+args.steps,
        refinement=dict(parent_sha256=r.meta['sha256'],steps=args.steps,all_frames=args.all_frames,
                        midpoint_weight=args.midpoint_weight))
    if joint_rigid:meta['rigid_bucket']=dict(meta['rigid_bucket'],sha256=digest(root/'rigid-bucket.npz'))
    if 'seed_initialization' in meta:
        meta['seed_initialization']=dict(meta['seed_initialization'],requires_training=False,fitted_steps=args.steps)
    if 'source_surface_initialization' in meta:
        meta['source_surface_initialization']=dict(meta['source_surface_initialization'],requires_training=False,fitted_steps=parent_iteration+args.steps,
            geometry_preserved=args.freeze_geometry,static_geometry_preserved=args.freeze_geometry or args.freeze_static_geometry)
    (root/'refinement-provenance.json').write_text(json.dumps(dict(
        parent_sha256=r.meta['sha256'],video_sha256=digest(args.video),steps=args.steps,
        evaluation_scope='in-sequence production fit; no held-out quality claim' if args.all_frames else 'preserved development holdouts',
        training_frames=training,excluded_frames=sorted(excluded),time_step=1/60,
        roi_xyxy=[0,0,640,640] if whole_scene else [40,40,470,470],roi_weight=1 if whole_scene else 4,temporal_loss_weight=args.temporal_weight,
        preserved_rigid_sha256=None if joint_rigid else rigid_sha,whole_scene=whole_scene,
        parent_rigid_sha256=rigid_sha,joint_rigid_appearance=joint_rigid,rigid_geometry_exact=bool(rigid_sha),
        spatial_detail_weight=detail_weight,detail_scope='bounded foreground tile balance and native/2x/4x derivative residuals',
        original_foreground_weight=4 if whole_scene else None,
        appearance_loss_weight=args.appearance_weight,
        multiview_manifest_sha256=digest(args.multiview/'views.json') if args.multiview else None,
        multiview_training_count=len(novel),multiview_foreground_weight=4,position_lr=args.position_lr,
        multiview_batch=args.multiview_batch,multiview_sampling='shuffled exhaustive cycles',max_minutes=args.max_minutes,
        geometry_prior_sha256=digest(args.geometry) if args.geometry else None,geometry_weight=args.geometry_weight,
        surface_normal_weight=args.surface_normal_weight,surface_normal_thickness=.003 if args.surface_normal_weight else None,
        ray_prior_sha256=digest(args.ray_prior/'views.json') if args.ray_prior else None,
        ray_training_kernel_fingerprint=ray_source_sha256,ray_weight=args.ray_weight if rays else 0,
        ray_batch=args.ray_batch if rays else 0,ray_training_views=len(rays),
        ray_render_width=args.ray_render_width if rays else None,
        ray_alpha_weight=args.ray_alpha_weight if rays else 0,
        ray_spread_weight=args.ray_spread_weight if rays else 0,
        frozen_parent_geometry_rows=frozen_parent_rows,frozen_geometry_fields=list(frozen_before),
        frozen_all_geometry=args.freeze_geometry,
        frozen_static_geometry=args.freeze_geometry or args.freeze_static_geometry,
        preserved_static_parameters=list(protected_static_before),
        surface_color_pairs_sha256=digest(args.surface_color_pairs) if args.surface_color_pairs else None,
        surface_color_metadata=color_pair_metadata,surface_color_weight=args.surface_color_weight if color_pairs is not None else 0,
        ray_scope=('Explicit mesh production all-time geometry fit; no held-out geometry timestamp claim' if ray_manifest.get('production_geometry_fit') else 'Explicit mesh inverse-depth and part-balanced foreground/background; excludes reference validation frames') if rays else None,
        geometry_soft_extent_limit=.03 if args.geometry else None,
        geometry_scope='explicit animated source mesh, asymmetric sampled surface coverage; not measured novel-view reconstruction' if args.geometry else None,
        train_static=bool(args.multiview and args.train_static),parameter_learning_rates=names,
        multiview_loss_weight=args.multiview_weight,
        removed_pathological_splats=removed_pathological,opacity_logit_cap=8 if args.multiview else None,
        edge_weight=args.edge_weight,
        curvature_weight=args.curvature_weight,subframe_samples=3 if args.curvature_weight else args.subframe_samples,
        subframe_sampling='random-centered triplet' if args.curvature_weight else 'midpoint' if args.subframe_samples==1 else 'stratified random pair',midpoint_weight=args.midpoint_weight,midpoint_scope='stable-interior interpolation regularizer, not measured midpoint ground truth',
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
    p.add_argument('--preserve-rigid',action='store_true',help='Freeze and retain complete bucket/arm sidecar while refining remaining scene')
    p.add_argument('--whole-scene',action='store_true',help='Use full original RGB image instead of legacy bucket ROI weighting')
    p.add_argument('--joint-rigid',action='store_true',help='Opt-in joint appearance fitting; rigid geometry and motion remain exact')
    p.add_argument('--detail-weight',type=float,choices=[0.,.5,1.,2.],default=0.)
    p.add_argument('--geometry',type=Path)
    p.add_argument('--ray-prior',type=Path)
    p.add_argument('--ray-weight',type=float,default=1,choices=[.5,1,2,5])
    p.add_argument('--ray-batch',type=int,default=2,choices=[1,2,4])
    p.add_argument('--ray-render-width',type=int,default=320,choices=[320,640])
    p.add_argument('--ray-alpha-weight',type=float,default=0,choices=[0,.25,.5,1])
    p.add_argument('--ray-spread-weight',type=float,default=0,choices=[0,1,5,10],help='Penalize composited inverse-depth spread around the source surface, not just its mean')
    p.add_argument('--freeze-parent-geometry',action='store_true',help='After bounded surface seeding, fit new support while preserving existing geometry exactly')
    p.add_argument('--freeze-geometry',action='store_true',help='Fit source-surface appearance and opacity while preserving all source geometry exactly')
    p.add_argument('--freeze-static-geometry',action='store_true',help='Keep static surface geometry exact while refining dynamic support and appearance')
    p.add_argument('--appearance-lr',type=float,default=.001,choices=[.001,.003,.01])
    p.add_argument('--surface-color-pairs',type=Path)
    p.add_argument('--surface-color-weight',type=float,default=1,choices=[.5,1,2])
    p.add_argument('--geometry-weight',type=float,default=0,choices=[0,.5,1.,2.])
    p.add_argument('--surface-normal-weight',type=float,default=0,choices=[0,1.,2.,5.])
    p.add_argument('--multiview-weight',type=float,default=1.5,choices=[.25,.5,1.,1.5,3.])
    p.add_argument('--multiview-batch',type=int,default=1,choices=[1,2,4])
    p.add_argument('--max-minutes',type=int,default=20,choices=[20,60,120])
    p.add_argument('--opacity-lr',type=float,choices=[.001,.003,.005,.02])
    p.add_argument('--position-lr',type=float,default=1e-6,choices=[1e-6,3e-6,1e-5,3e-5,1e-4])
    p.add_argument('--train-static',action='store_true')
    p.add_argument('--edge-weight',type=float,default=0,choices=[0,0.25,0.5,1])
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--midpoint-weight',type=float,default=0,choices=[0,0.5,1,2])
    p.add_argument('--temporal-weight',type=float,default=.5,choices=[.5,1,2,4,8],help='Match observed adjacent RGB changes; does not penalize real motion')
    p.add_argument('--curvature-weight',type=float,choices=[0,2,5,10],default=0,help='Penalize subframe curvature only within motion-masked stable interiors')
    p.add_argument('--subframe-samples',type=int,choices=[1,2],default=1,help='Two stratified random subframes cover interpolation between RGB observations')
    p.add_argument('--all-frames',action='store_true',help='Production fit: use all observed timestamps; no holdout claim')
    p.add_argument('--appearance-weight',type=float,default=1,choices=[1,2,4,8],help='Protect original RGB fidelity while retaining novel-view constraints')
    args=p.parse_args()
    # Coordinate with UI training/import jobs; only remove a lock we created.
    lock=args.output.resolve().parent/'.fvv-writer.lock'
    with lock.open('x') as stream:
        stream.write('real Lego ROI refinement\n')
    try:
        run(args)
    finally:
        lock.unlink(missing_ok=True)
