"""Smooth time-dependent Gaussian appearance with optional bounded geometry.

Low-degree Legendre color coefficients share parameters across every time.
Validation timestamps are excluded from gradients. Parent motion is fixed.
Optional canonical offsets follow the parent rotations. This is a single-view RGB
fit, not an event-only model or evidence of novel-view generalization.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.experiment_format import sha256, save_json


def refine(sequence, reference, output, iterations=1600, perceptual_weight=.03, initialize=None, degree=5, spatial_weight=.01, ssim_weight=0., neighbor_count=4, freeze_surface=False, color_bound=None, smooth_field=False, geometry_bound=0., freeze_appearance=False, freeze_geometry=False, geometry_groups=None, rotation_trajectory=None):
    import torch
    from PIL import Image
    from viewer import Renderer
    from gaussian_renderer import render
    from scripts.rendering.render_ply import make_camera
    from utils.appearance_stage import FIELDS, temporal_basis
    from utils.loss_utils import ssim
    from utils.event_refinement import qrotate
    torch.manual_seed(20261027)
    if color_bound is not None and not 0 < color_bound <= 1:
        raise ValueError('Color bound must be in (0, 1] RGB units')
    if smooth_field and (initialize or not freeze_surface or color_bound is None):
        raise ValueError('Smooth field needs a fresh fixed surface and an explicit RGB bound')
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    source=json.loads(reference.read_text());rows=source['samples'];camera=source['camera']
    if not all(r.get('rgb') and r.get('foreground_mask') for r in rows):
        raise ValueError('Aligned RGB and shared mask required')
    r=Renderer(sequence,'cpu');r.load();model=r.model
    pose=np.asarray(camera['pose_opengl_c2w']);h,w=camera['height'],camera['width']
    fovy=2*np.arctan(h/(2*camera['K'][1][1]));cam=make_camera(pose,w,h,fovy)
    bg=torch.tensor(source.get('background_rgb',[0.,0.,0.]),device='cuda')
    rgb=[torch.tensor(np.asarray(Image.open(reference.parent/x['rgb']).convert('RGB')).copy(),device='cuda').permute(2,0,1).float()/255 for x in rows]
    masks=[torch.tensor(np.asarray(Image.open(reference.parent/x['foreground_mask'])).copy()!=0,device='cuda')[None] for x in rows]
    part_weights=[];part_labels=[];part_counts=[]
    if geometry_groups:
        for row,mask in zip(rows,masks):
            labels=np.asarray(Image.open(reference.parent/row['part_mask']))
            ids,counts=np.unique(labels,return_counts=True)
            weights=np.zeros(labels.shape,np.float32)
            for label,count in zip(ids,counts):
                if label:weights[labels==label]=1/np.sqrt(count)
            weights=torch.tensor(weights,device='cuda')[None]*mask
            part_weights.append(weights/weights.sum().clamp_min(1e-8))
            labels=torch.tensor(labels.astype(np.int64),device='cuda').flatten()
            part_labels.append(labels);part_counts.append(torch.bincount(labels))
    achromatic = all(float((x.max(0).values-x.min(0).values).max()) < 1e-6 for x in rgb)
    validation=[i for i in range(len(rows)) if i%5==4];training=[i for i in range(len(rows)) if i not in validation]
    if len(validation)<3:raise ValueError('Too few held-out samples')
    original={attr:getattr(model,attr).detach().clone() for attr in FIELDS.values()}
    for attr in FIELDS.values():setattr(model,attr,torch.nn.Parameter(original[attr].clone()))
    dc = model._features_dc
    rest_xyz=model._xyz.detach().clone()
    rest_inverse=model.get_rotation.detach().clone();rest_inverse[:,1:] *= -1
    displacement=torch.nn.Parameter(torch.zeros_like(rest_xyz)) if geometry_bound else None
    if geometry_bound < 0 or geometry_bound > .1:raise ValueError('Geometry bound must be in [0, .1] scene units')
    coefficients = torch.nn.Parameter(torch.zeros((len(model._xyz), degree, 3), device='cuda'))
    temporal_metadata = dict(kind='legendre',degree=degree,frame_count=len(r.poses))
    temporal_values = temporal_basis(len(r.poses), degree)
    static_rows = None
    if rotation_trajectory:
        from utils.appearance_stage import rotor_basis
        if initialize or smooth_field or not freeze_surface or geometry_bound or freeze_appearance:
            raise ValueError('Rotor appearance requires fresh, fixed geometry and original DC anchor')
        angles=json.loads(rotation_trajectory.read_text())['angles_rad']
        temporal_values=rotor_basis(angles,degree)
        if len(angles)!=len(r.poses) or r.dynamic_indices is None:raise ValueError('Rotor motion identity mismatch')
        temporal_metadata=dict(kind='rotor_fourier',degree=degree,frame_count=len(r.poses),angles_rad=angles,
                               trajectory_sha256=sha256(rotation_trajectory))
        static_rows=torch.ones(len(dc),dtype=torch.bool,device='cuda');static_rows[r.dynamic_indices]=False
        dc.requires_grad_(False)
    basis = torch.tensor(temporal_values, device='cuda')
    temporal_active = False
    geometry_field=None;geometry_bases=[];geometry_indices=[]
    if geometry_groups:
        if not geometry_bound or freeze_geometry:raise ValueError('Continuous geometry requires a trainable bounded geometry stage')
        with np.load(geometry_groups,allow_pickle=False) as z:labels=z['groups']
        if labels.shape!=(len(rest_xyz),) or labels.dtype.kind not in 'iu':raise ValueError('Geometry group identity mismatch')
        axis=torch.linspace(-1,1,4,device='cuda');nodes=torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
        for group in np.unique(labels):
            ids=torch.tensor(np.flatnonzero(labels==group),device='cuda');points=rest_xyz[ids]
            low,high=points.amin(0),points.amax(0);coords=(points-(low+high)/2)/((high-low)/2).clamp_min(1e-5)
            b=torch.cat([torch.softmax(-torch.cdist(c,nodes).square()/(2*.65**2),1) for c in coords.split(8192)])
            geometry_indices.append(ids);geometry_bases.append(b)
        geometry_field=torch.nn.Parameter(torch.zeros((len(geometry_indices),64,3),device='cuda'))
        geometry_order=torch.argsort(torch.cat(geometry_indices))
    field = spatial_basis = None
    if smooth_field:
        xyz=model._xyz.detach()
        low,high=xyz.amin(0),xyz.amax(0)
        coordinates=(xyz-(low+high)/2)/((high-low)/2).clamp_min(1e-6)
        axis=torch.linspace(-1,1,4,device='cuda')
        nodes=torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
        spatial_basis=torch.cat([torch.softmax(-torch.cdist(chunk,nodes).square()/(2*.65**2),dim=1)
                                 for chunk in coordinates.split(32768)],dim=0)
        field=torch.nn.Parameter(torch.zeros((len(nodes),degree+1,3),device='cuda'))
    neighbors = None
    if (spatial_weight and not smooth_field) or (geometry_bound and not geometry_groups):
        from scipy.spatial import cKDTree
        xyz = model._xyz.detach().cpu().numpy()
        if not 2 <= neighbor_count <= 32:raise ValueError('Neighbor count must be 2..32')
        _, nearest = cKDTree(xyz).query(xyz,k=neighbor_count,workers=4)
        neighbors = torch.tensor(nearest[:,1:].copy(),dtype=torch.long,device='cuda')
    for attr in ('_features_rest','_xyz','_rotation'):getattr(model,attr).requires_grad_(False)
    parameters=([{'params':[field],'lr':.004}] if smooth_field else [{'params':[dc],'lr':.008}, {'params':[coefficients],'lr':.008}])
    if rotation_trajectory:parameters=[{'params':[coefficients],'lr':.008}]
    if freeze_appearance:
        if smooth_field:raise ValueError('Cannot freeze a newly initialized appearance field')
        dc.requires_grad_(False);coefficients.requires_grad_(False);parameters=[]
    if displacement is not None:
        displacement.requires_grad_(not freeze_geometry)
        if not freeze_geometry:parameters.append({'params':[geometry_field if geometry_groups else displacement],'lr':.0003})
    for attr, rate in (('_opacity', .015), ('_scaling', .0015)):
        getattr(model, attr).requires_grad_(not freeze_surface)
        if not freeze_surface: parameters.append({'params':[getattr(model,attr)],'lr':rate})
    optimizer=torch.optim.Adam(parameters)
    perceptual=None
    if perceptual_weight:
        from unittest.mock import patch
        import lpips
        torch.hub.set_dir(str(Path.home()/'.cache/scgs-comparison/weights'))
        original_load=torch.load
        def tensor_load(*args,**kwargs):return original_load(*args,**dict(kwargs,weights_only=True))
        with patch('torch.load',tensor_load):perceptual=lpips.LPIPS(net='alex',version='0.1',verbose=False).cuda().eval()
        for parameter in perceptual.parameters():parameter.requires_grad_(False)
    def frame(i):
        nonlocal dc, coefficients, displacement
        if geometry_groups:
            displacement=torch.cat([b@geometry_field[g] for g,b in enumerate(geometry_bases)])[geometry_order]
            displacement=displacement*(geometry_bound/displacement.norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp(max=1)
        if smooth_field:
            mapped=(spatial_basis@field.reshape(len(field),-1)).reshape(-1,degree+1,3)
            dc=original['_features_dc']+mapped[:,:1]
            coefficients=mapped[:,1:]
        model._features_dc = dc + (coefficients * basis[rows[i]['frame']][None,:,None]).sum(1,keepdim=True) if temporal_active else dc
        model._xyz=rest_xyz.clone()
        state=r.poses[rows[i]['frame']].to('cuda')
        if r.dynamic_indices is None:model._xyz=state[:,:3].contiguous();model._rotation=state[:,3:7].contiguous()
        else:
            model._xyz[r.dynamic_indices]=state[:,:3];model._rotation[r.dynamic_indices]=state[:,3:7]
        if displacement is not None:
            model._xyz=model._xyz+qrotate(model.get_rotation,qrotate(rest_inverse,displacement))
        return render(cam,model,r.pipe,bg,0.,0.,0.)
    def score(indices):
        errors=[];ious=[];distances=[];missing=[];parts=[]
        with torch.no_grad():
            for i in indices:
                result=frame(i);image=result['render'].clamp(0,1)
                errors.append(float(((image-rgb[i]).square()*masks[i]).sum()/(masks[i].sum()*3)))
                cover=result['alpha']>=.5;mask=masks[i]
                ious.append(float((cover&mask).sum()/(cover|mask).sum().clamp_min(1)))
                missing.append(float((mask&~cover).sum()/mask.sum().clamp_min(1)))
                if geometry_groups:
                    count=part_counts[i];absent=torch.bincount(part_labels[i],weights=(~cover).flatten().float(),minlength=len(count))
                    eligible=count>0;eligible[0]=False
                    parts.append(float((absent[eligible]/count[eligible]).mean()))
                if perceptual is not None:distances.append(float(perceptual(image[None]*2-1,rgb[i][None]*2-1)))
        return dict(foreground_mse=float(np.mean(errors)),iou=float(np.mean(ious)),lpips=float(np.mean(distances)) if distances else None,
                    foreground_missing=float(np.mean(missing)),part_missing=float(np.mean(parts)) if parts else None,
                    per_frame_mse=errors,
                    foreground_psnr=float(np.mean(-10*np.log10(np.maximum(errors,1e-12)))))
    baseline=score(validation);best=None;history=[]
    if initialize:
        from utils.appearance_stage import validate_attribute
        with np.load(initialize,allow_pickle=False) as data,torch.no_grad():
            if json.loads(str(data['metadata_json']))['parent_sha256']!=sha256(sequence):raise ValueError('Initialization parent mismatch')
            for field,attr in FIELDS.items():
                validate_attribute(field,data[field],original[attr].cpu().numpy())
                getattr(model,attr).copy_(torch.tensor(data[field],device='cuda'))
            if 'temporal_dc' in data:
                from utils.appearance_stage import validate_temporal
                metadata=json.loads(str(data['metadata_json']))
                validate_temporal(data['temporal_dc'],metadata['temporal'],len(model._xyz))
                if metadata['temporal']['degree']!=degree or metadata['temporal']['frame_count']!=len(r.poses):
                    raise ValueError('Initialization temporal basis mismatch')
                coefficients.copy_(torch.tensor(data['temporal_dc'],device='cuda'))
                temporal_active = True
            if 'geometry_offset_local' in data:
                if displacement is None:raise ValueError('Geometry initialization requires an explicit geometry bound')
                local=torch.tensor(data['geometry_offset_local'],device='cuda')
                if local.shape != displacement.shape or not torch.isfinite(local).all() or float(local.norm(dim=1).max())>geometry_bound*(1+1e-5):
                    raise ValueError('Initialization geometry exceeds allowed bound')
                rest_rotation=rest_inverse.clone();rest_rotation[:,1:] *= -1
                displacement.copy_(qrotate(rest_rotation,local))
                if geometry_groups:
                    for g,(ids,b) in enumerate(zip(geometry_indices,geometry_bases)):
                        geometry_field[g].copy_(torch.linalg.solve(b.T@b+len(ids)*1e-5*torch.eye(64,device='cuda'),b.T@displacement[ids]))
    previous = score(validation)
    temporal_active = True
    # Finite equivalents of parent opacity sentinels allow coverage refinement.
    opacity_center = torch.nan_to_num(original['_opacity'],posinf=8.,neginf=-8.)
    if not freeze_surface:
        with torch.no_grad():model._opacity.copy_(torch.nan_to_num(model._opacity,posinf=8.,neginf=-8.))
    print('BASELINE',baseline,flush=True)
    print('PREVIOUS_V3',previous,flush=True)
    for step in range(iterations):
        i=training[step%len(training)];optimizer.zero_grad(set_to_none=True)
        result=frame(i);image=result['render'];mask=masks[i];error=(image-rgb[i]).square()
        photometric=(error*mask).sum()/(mask.sum()*3)+.2*(error*~mask).sum()/((~mask).sum()*3)
        coverage=(result['alpha']-mask.float()).square().mean()
        loss=photometric+.15*coverage+.0002*coefficients.square().mean()
        if displacement is not None:
            alpha=result['alpha']
            dice=1-(2*(alpha*mask).sum()+1)/(alpha.sum()+mask.sum()+1)
            loss=loss+.5*dice+.01*(displacement/geometry_bound).square().mean()
            if geometry_groups:
                loss=loss+.4*((1-alpha)*part_weights[i]).sum()
        if ssim_weight:loss=loss+ssim_weight*(1-ssim(image.clamp(0,1)[None],rgb[i][None]))
        if neighbors is not None:
            sites=torch.randint(len(dc),(min(32768,len(dc)),),device='cuda')
            adjacent=neighbors[sites]
            smooth_dc=(dc[sites][:,None]-dc[adjacent]).square().mean()
            smooth_time=(coefficients[sites][:,None]-coefficients[adjacent]).square().mean()
            loss=loss+spatial_weight*(smooth_dc+smooth_time)
            if displacement is not None:
                loss=loss+.1*((displacement[sites][:,None]-displacement[adjacent])/geometry_bound).square().mean()
        if perceptual is not None:loss=loss+perceptual_weight*perceptual(image.clamp(0,1)[None]*2-1,rgb[i][None]*2-1).mean()
        loss.backward();optimizer.step()
        with torch.no_grad():
            if displacement is not None:
                displacement.mul_((geometry_bound/displacement.norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp(max=1))
            if smooth_field:
                if achromatic:field.copy_(field.mean(-1,keepdim=True).expand_as(field).clone())
                # Positive spatial weights sum to one; each node has its own
                # bounded temporal curve. This cannot create splat-scale grain.
                factor=((color_bound/.28209479177387814)/field.abs().sum(1,keepdim=True).clamp_min(1e-8)).clamp(max=1)
                field.mul_(factor)
            if not freeze_appearance:
                dc.clamp_(min=-1.77245385,max=1.77245385)
                coefficients.clamp_(min=-2.,max=2.)
            if achromatic and not freeze_appearance:
                dc.copy_(dc.mean(-1,keepdim=True).expand_as(dc).clone())
                coefficients.copy_(coefficients.mean(-1,keepdim=True).expand_as(coefficients).clone())
            if color_bound is not None:
                # A conservative bound holds for all times because every
                # Legendre basis element has magnitude <= 1 on [-1, 1].
                limit=color_bound/.28209479177387814
                residual=dc-original['_features_dc']
                budget=residual.abs()+coefficients.abs().sum(1,keepdim=True)
                factor=(limit/budget.clamp_min(1e-8)).clamp(max=1)
                dc.copy_(original['_features_dc']+residual*factor)
                coefficients.mul_(factor)
            if rotation_trajectory:
                dc.copy_(original['_features_dc'])
                coefficients[static_rows]=0
            if not freeze_surface:
                model._opacity.copy_(torch.maximum(torch.minimum(model._opacity,opacity_center+2),opacity_center-2))
                model._scaling.copy_(torch.maximum(torch.minimum(model._scaling,original['_scaling']+.2),original['_scaling']-.2))
        if (step+1)%100==0 or step+1==iterations:
            measured=score(validation);history.append(dict(iteration=step+1,**measured));print('VALIDATION',history[-1],flush=True)
            def objective(value):
                if geometry_groups:return value['foreground_mse']+.2*(value['lpips'] or 0)+.05*(1-value['iou'])+.05*value['foreground_missing']+.05*value['part_missing']
                return value['lpips' if ssim_weight else 'foreground_mse']
            if (measured['foreground_mse']<min(baseline['foreground_mse'],previous['foreground_mse']*(1.1 if geometry_groups else 1.05 if ssim_weight else 1.))
                    and measured['iou']>=min(previous['iou'],baseline['iou'])
                    and (perceptual is None or measured['lpips']<min(baseline['lpips'],previous['lpips']*(1.1 if geometry_groups else 1)))
                    and (not geometry_groups or (measured['foreground_missing']<previous['foreground_missing'] and measured['part_missing']<previous['part_missing']))
                    and (best is None or objective(measured)<objective(best))):
                best=dict(iteration=step+1,**measured)
                metadata=dict(format='scgs_appearance_stage_v1',parent_sha256=sha256(sequence),
                    reference_sha256=sha256(reference),train_frames=[rows[i]['frame'] for i in training],
                    initialization_sha256=sha256(initialize) if initialize else None,perceptual_weight=perceptual_weight,
                    temporal=temporal_metadata,
                    dc_anchor='original parent; static rows unchanged' if rotation_trajectory else None,
                    achromatic_reference=achromatic,spatial_weight=spatial_weight,
                    freeze_surface=freeze_surface,color_bound_rgb=color_bound,smooth_field=smooth_field,
                    spatial_field_basis='normalized Gaussian RBF 4x4x4, sigma 0.65' if smooth_field else None,
                    ssim_weight=ssim_weight,neighbor_count=neighbor_count,
                    selection='Joint MSE/LPIPS/IoU/coverage objective; both missing rates must improve; MSE/LPIPS bounded to previous +10 percent' if geometry_groups else 'Minimum validation LPIPS, MSE within 5 percent of previous stage' if ssim_weight else 'Minimum validation foreground MSE',
                    validation_frames=[rows[i]['frame'] for i in validation],baseline=baseline,validation=best,
                    changes=('Bounded temporal DC color only; opacity, scales, centers, rotations and motion frozen' if freeze_surface else 'Smooth temporal DC color, opacity and bounded log-scale; centers, rotations, identities and motion frozen'),
                    scope='Single-view in-sequence RGB-supervised fit; held-out timestamps, not unseen-view or event-only training')
                if displacement is not None:
                    metadata['geometry_refinement']=dict(max_offset_scene_units=geometry_bound,representation='time-independent local Gaussian-frame offset; follows parent rigid motion',regularization='part-local continuous RBF' if geometry_groups else 'canonical neighbor displacement smoothness',freeze_appearance=freeze_appearance,freeze_geometry=freeze_geometry,groups_sha256=sha256(geometry_groups) if geometry_groups else None)
                    metadata['changes']=('Temporal appearance over fixed refined geometry' if freeze_geometry else 'Bounded canonical geometry offsets; parent motion, scales and opacity frozen' if freeze_surface else 'Bounded canonical geometry offsets and appearance parameters')
                np.savez_compressed(output/'appearance.npz',metadata_json=np.asarray(json.dumps(metadata)),
                    **({'geometry_offset_local':qrotate(rest_inverse,displacement).detach().cpu().numpy()} if displacement is not None else {}),
                    temporal_dc=coefficients.detach().cpu().numpy(),
                    **{field:(dc if field=='features_dc' else getattr(model,attr)).detach().cpu().numpy() for field,attr in FIELDS.items()})
    save_json(output/'training.json',dict(baseline=baseline,selected=best,history=history,iterations=iterations,
        previous_v3=previous, temporal_degree=degree, temporal=temporal_metadata,
        geometry_bound=geometry_bound,freeze_appearance=freeze_appearance,freeze_geometry=freeze_geometry,
        geometry_groups=str(geometry_groups) if geometry_groups else None,geometry_groups_sha256=sha256(geometry_groups) if geometry_groups else None,
        geometry_loss_weights=dict(dice=.5,displacement=.01,neighbor=.1,balanced_parts=.4 if geometry_groups else 0.) if geometry_bound else None,
        achromatic_reference=achromatic,spatial_weight=spatial_weight,
        freeze_surface=freeze_surface,color_bound_rgb=color_bound,smooth_field=smooth_field,
        ssim_weight=ssim_weight,neighbor_count=neighbor_count,
        seed=20261027,learning_rates={**({} if freeze_appearance else dict(spatial_field=.004) if smooth_field else dict(dc=.008,temporal_dc=.008)),**({} if freeze_surface else dict(opacity=.015,log_scale=.0015)),**(dict(geometry=.0003) if geometry_bound and not freeze_geometry else {})},coverage_weight=.15,
        perceptual_weight=perceptual_weight,initialization_sha256=sha256(initialize) if initialize else None,
        parent=str(sequence),parent_sha256=sha256(sequence),reference=str(reference),reference_sha256=sha256(reference)))
    if best is None:raise RuntimeError('No checkpoint improved both held-out foreground error and IoU; not deployable')
    print('ACCEPTED',best,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=1600)
    p.add_argument('--perceptual-weight',type=float,default=.03);p.add_argument('--initialize',type=Path)
    p.add_argument('--freeze-surface',action='store_true')
    p.add_argument('--color-bound',type=float)
    p.add_argument('--smooth-field',action='store_true')
    p.add_argument('--geometry-bound',type=float,default=0.)
    p.add_argument('--freeze-appearance',action='store_true')
    p.add_argument('--freeze-geometry',action='store_true')
    p.add_argument('--geometry-groups',type=Path)
    p.add_argument('--rotation-trajectory',type=Path)
    p.add_argument('--degree',type=int,default=5)
    p.add_argument('--spatial-weight',type=float,default=.01)
    p.add_argument('--ssim-weight',type=float,default=0.)
    p.add_argument('--neighbor-count',type=int,default=4)
    a=p.parse_args();refine(a.sequence,a.reference,a.output,a.iterations,a.perceptual_weight,a.initialize,a.degree,a.spatial_weight,a.ssim_weight,a.neighbor_count,a.freeze_surface,a.color_bound,a.smooth_field,a.geometry_bound,a.freeze_appearance,a.freeze_geometry,a.geometry_groups,a.rotation_trajectory)
