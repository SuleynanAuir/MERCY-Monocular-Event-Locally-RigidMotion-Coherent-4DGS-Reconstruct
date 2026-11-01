"""Bounded spatially balanced RGB detail supervision (offline training only).

Balance occupied tiles, not inferred semantic parts. Tiny foreground fragments
cannot get unbounded weight. Original full-image losses remain necessary for
black surfaces, silhouette/background and global appearance.
"""
import torch
import torch.nn.functional as F


def detail_loss(image, target, tile=40):
    if image.shape != target.shape or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError('Expected matching CHW RGB images')
    h,w=image.shape[-2:]
    if h % tile or w % tile or tile < 4:
        raise ValueError('Detail tiles must divide the image')
    mask=(target.detach().mean(0,keepdim=True)>8/255).float()
    occupancy=F.avg_pool2d(mask[None],tile,tile)
    # Saturate at 4x: a single bright pixel must not dominate optimization.
    balance=(1/occupancy.clamp_min(.25)).clamp_max(4)
    weight=mask*F.interpolate(balance,size=(h,w),mode='nearest')[0]
    weight=weight/weight.mean().clamp_min(1e-6)
    error=image-target
    loss=(error.abs()*weight).mean()
    # Native and coarser derivative residuals retain fine holes and structural
    # edges without asking adjacent surfaces to share colors.
    for scale in (1,2,4):
        e=error if scale==1 else F.avg_pool2d(error[None],scale,scale)[0]
        v=weight if scale==1 else F.avg_pool2d(weight[None],scale,scale)[0]
        dx=(e[:,:,1:]-e[:,:,:-1]).abs()
        dy=(e[:,1:,:]-e[:,:-1,:]).abs()
        loss=loss+((dx*torch.maximum(v[:,:,1:],v[:,:,:-1])).mean()+
                   (dy*torch.maximum(v[:,1:,:],v[:,:-1,:])).mean())/6
    return loss
