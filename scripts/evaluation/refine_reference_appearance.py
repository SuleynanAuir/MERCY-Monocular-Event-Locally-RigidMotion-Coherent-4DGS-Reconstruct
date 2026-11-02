"""Bounded RGB/coverage refinement; source geometry and motion remain fixed.

Fit only DC color, opacity and Gaussian scale on a predefined 80% time split.
Publish a checkpoint only if held-out foreground error and IoU both improve.
The split is temporal within one supplied view, not unseen-scene generalization.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.experiment_format import sha256, save_json


def refine(sequence, reference, output, iterations=400, perceptual_weight=0., initialize=None, structure_weight=0., anchor_weight=0., smoothness_weight=0.):
    if type(iterations) is not int or iterations < 1:
        raise ValueError('Iterations must be positive')
    if any(not np.isfinite(v) or v < 0 for v in (perceptual_weight, structure_weight, anchor_weight, smoothness_weight)):
        raise ValueError('Loss weights must be finite and nonnegative')
    import torch
    from PIL import Image
    from viewer import Renderer
    from gaussian_renderer import render
    from scripts.rendering.render_ply import make_camera
    from utils.appearance_stage import FIELDS
    torch.manual_seed(20261027)
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
    validation=[i for i in range(len(rows)) if i%5==4];training=[i for i in range(len(rows)) if i not in validation]
    if len(validation)<3:raise ValueError('Too few held-out samples')
    original={attr:getattr(model,attr).detach().clone() for attr in FIELDS.values()}
    for attr in FIELDS.values():setattr(model,attr,torch.nn.Parameter(original[attr].clone()))
    for attr in ('_features_rest','_xyz','_rotation'):getattr(model,attr).requires_grad_(False)
    optimizer=torch.optim.Adam([{'params':[model._features_dc],'lr':.015},
                                {'params':[model._opacity],'lr':.025},
                                {'params':[model._scaling],'lr':.003}])
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
        state=r.poses[rows[i]['frame']].to('cuda')
        if r.dynamic_indices is None:model._xyz=state[:,:3].contiguous();model._rotation=state[:,3:7].contiguous()
        else:
            model._xyz[r.dynamic_indices]=state[:,:3];model._rotation[r.dynamic_indices]=state[:,3:7]
        return render(cam,model,r.pipe,bg,0.,0.,0.)
    def score(indices):
        errors=[];ious=[];distances=[]
        with torch.no_grad():
            for i in indices:
                result=frame(i);image=result['render'].clamp(0,1)
                errors.append(float(((image-rgb[i]).square()*masks[i]).sum()/(masks[i].sum()*3)))
                cover=result['alpha']>=.5;mask=masks[i]
                ious.append(float((cover&mask).sum()/(cover|mask).sum().clamp_min(1)))
                if perceptual is not None:distances.append(float(perceptual(image[None]*2-1,rgb[i][None]*2-1)))
        return dict(foreground_mse=float(np.mean(errors)),iou=float(np.mean(ious)),lpips=float(np.mean(distances)) if distances else None,
                    foreground_psnr=float(np.mean(-10*np.log10(np.maximum(errors,1e-12)))))
    canonical_xyz=model._xyz.detach().cpu().numpy().copy()
    baseline=score(validation);best=None;history=[]
    if initialize:
        from utils.appearance_stage import validate_attribute
        with np.load(initialize,allow_pickle=False) as data,torch.no_grad():
            if json.loads(str(data['metadata_json']))['parent_sha256']!=sha256(sequence):raise ValueError('Initialization parent mismatch')
            for field,attr in FIELDS.items():
                validate_attribute(field,data[field],original[attr].cpu().numpy())
                getattr(model,attr).copy_(torch.tensor(data[field],device='cuda'))
    # Gate against the actual initialized candidate, not its older parent colors.
    parent_baseline=baseline
    baseline=score(validation) if initialize else baseline
    anchor=model._features_dc.detach().clone()
    neighbors=None
    if smoothness_weight:
        from scipy.spatial import cKDTree
        from scipy.spatial.transform import Rotation
        # Local surface neighbors only; reject sharp material/normal boundaries.
        distance, indices=cKDTree(canonical_xyz).query(canonical_xyz,k=5,workers=4)
        indices=indices[:,1:];distance=distance[:,1:]
        colors=anchor.detach().cpu().numpy()[:,0]
        with np.load(sequence,allow_pickle=False) as data:
            vertices=data['base_vertices']
            quats=np.column_stack([vertices[f'rot_{k}'] for k in (1,2,3,0)])
            rotations=Rotation.from_quat(quats).as_matrix()
            axis=np.argmin(np.column_stack([vertices[f'scale_{k}'] for k in range(3)]),axis=1)
            normals=rotations[np.arange(len(vertices)),:,axis]
        agreement=np.abs(np.einsum('nki,ni->nk',normals[indices],normals))
        support=(distance<.015)&(agreement>.85)&(np.linalg.norm(colors[indices]-colors[:,None],axis=2)<.30)
        neighbors=torch.tensor(indices,device='cuda')
        neighbor_support=torch.tensor(support,device='cuda')
    from utils.loss_utils import ssim
    print('BASELINE',baseline,flush=True)
    for step in range(iterations):
        i=training[step%len(training)];optimizer.zero_grad(set_to_none=True)
        result=frame(i);image=result['render'];mask=masks[i];error=(image-rgb[i]).square()
        photometric=(error*mask).sum()/(mask.sum()*3)+.2*(error*~mask).sum()/((~mask).sum()*3)
        coverage=(result['alpha']-mask.float()).square().mean()
        loss=photometric+.15*coverage
        if structure_weight:loss=loss+structure_weight*(1-ssim(image[None],rgb[i][None]))
        if anchor_weight:loss=loss+anchor_weight*(model._features_dc-anchor).square().mean()
        if neighbors is not None:
            ids=torch.randint(len(anchor),(8192,),device='cuda')
            delta=model._features_dc-anchor
            residual=(delta[ids,None]-delta[neighbors[ids]]).square().mean((2,3))
            support=neighbor_support[ids]
            loss=loss+smoothness_weight*(residual*support).sum()/support.sum().clamp_min(1)
        if perceptual is not None:loss=loss+perceptual_weight*perceptual(image.clamp(0,1)[None]*2-1,rgb[i][None]*2-1).mean()
        loss.backward();optimizer.step()
        with torch.no_grad():
            model._features_dc.clamp_(min=-1.77245385,max=1.77245385)
            model._opacity.copy_(torch.maximum(torch.minimum(model._opacity,original['_opacity']+2),original['_opacity']-2))
            model._scaling.copy_(torch.maximum(torch.minimum(model._scaling,original['_scaling']+.2),original['_scaling']-.2))
        if (step+1)%50==0 or step+1==iterations:
            measured=score(validation);history.append(dict(iteration=step+1,**measured));print('VALIDATION',history[-1],flush=True)
            if (measured['foreground_mse']<baseline['foreground_mse'] and measured['iou']>=baseline['iou']
                    and (perceptual is None or measured['lpips']<baseline['lpips'])
                    and (best is None or measured['foreground_mse']<best['foreground_mse'])):
                best=dict(iteration=step+1,**measured)
                metadata=dict(format='scgs_appearance_stage_v1',parent_sha256=sha256(sequence),
                    reference_sha256=sha256(reference),structure_weight=structure_weight,anchor_weight=anchor_weight,smoothness_weight=smoothness_weight,train_frames=[rows[i]['frame'] for i in training],
                    initialization_sha256=sha256(initialize) if initialize else None,perceptual_weight=perceptual_weight,
                    validation_frames=[rows[i]['frame'] for i in validation],baseline=baseline,validation=best,
                    changes='DC color, opacity and bounded log-scale; centers, rotations, point identities and all motion frozen',
                    scope='Single-view in-sequence RGB-supervised fit; held-out timestamps, not unseen-view or event-only training')
                np.savez_compressed(output/'appearance.npz',metadata_json=np.asarray(json.dumps(metadata)),
                    **{field:getattr(model,attr).detach().cpu().numpy() for field,attr in FIELDS.items()})
    save_json(output/'training.json',dict(baseline=baseline,parent_baseline=parent_baseline,structure_weight=structure_weight,anchor_weight=anchor_weight,smoothness_weight=smoothness_weight,selected=best,history=history,iterations=iterations,
        seed=20261027,learning_rates=dict(dc=.015,opacity=.025,log_scale=.003),coverage_weight=.15,
        perceptual_weight=perceptual_weight,initialization_sha256=sha256(initialize) if initialize else None,
        parent=str(sequence),parent_sha256=sha256(sequence),reference=str(reference),reference_sha256=sha256(reference)))
    if best is None:raise RuntimeError('No checkpoint improved both held-out foreground error and IoU; not deployable')
    print('ACCEPTED',best,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=400)
    p.add_argument('--perceptual-weight',type=float,default=0.);p.add_argument('--initialize',type=Path)
    p.add_argument('--structure-weight',type=float,default=0.);p.add_argument('--anchor-weight',type=float,default=0.)
    p.add_argument('--smoothness-weight',type=float,default=0.)
    a=p.parse_args();refine(a.sequence,a.reference,a.output,a.iterations,a.perceptual_weight,a.initialize,a.structure_weight,a.anchor_weight,a.smoothness_weight)
