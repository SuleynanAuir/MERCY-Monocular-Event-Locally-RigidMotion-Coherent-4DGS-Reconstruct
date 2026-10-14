"""Bounded appearance-only refinement; no centers, rotations, or motion parameters."""
import math

import torch
from torch import nn


class SurfaceAppearance(nn.Module):
    def __init__(self, dc, log_scale, opacity_logit, minimum_scale=.55, maximum_scale=1.1,
                 anisotropic=False):
        super().__init__()
        count = len(dc)
        if (dc.shape != (count, 1, 3) or log_scale.shape != (count, 3)
                or opacity_logit.shape != (count, 1) or not count
                or not 0 < minimum_scale <= 1 <= maximum_scale):
            raise ValueError('Invalid seed appearance or scale bounds')
        if not all(torch.isfinite(value).all() for value in (dc, log_scale, opacity_logit)):
            raise ValueError('Nonfinite seed appearance')
        if not torch.allclose(dc[..., 0], dc[..., 1]) or not torch.allclose(dc[..., 0], dc[..., 2]):
            raise ValueError('This grayscale refinement requires grayscale seed colors')
        gray = dc[:, 0, 0].detach() * .28209479177387814 + .5
        self.register_buffer('base_gray', gray.clone())
        self.register_buffer('base_log_scale', log_scale.detach().clone())
        self.register_buffer('base_opacity', opacity_logit.detach().clone())
        self.gray = nn.Parameter(gray.clone())
        self.log_scale_ratio = nn.Parameter(torch.zeros((count, 3 if anisotropic else 1), device=dc.device, dtype=dc.dtype))
        self.opacity = nn.Parameter(opacity_logit.detach().clone())
        self.scale_bounds = (math.log(minimum_scale), math.log(maximum_scale))
        self.opacity_bounds = (math.log(.75/.25), math.log(.995/.005))

    def forward(self):
        gray = self.gray.clamp(.001, .999)
        dc = ((gray-.5)/.28209479177387814)[:, None, None].expand(-1, 1, 3)
        scale = self.base_log_scale + self.log_scale_ratio.clamp(*self.scale_bounds)
        opacity = self.opacity.clamp(*self.opacity_bounds)
        return dc, scale, opacity

    @torch.no_grad()
    def project_(self):
        self.gray.clamp_(.001, .999)
        self.log_scale_ratio.clamp_(*self.scale_bounds)
        self.opacity.clamp_(*self.opacity_bounds)

    def regularization(self):
        return (.002*self.log_scale_ratio.square().mean()
                + .0001*(self.gray-self.base_gray).square().mean()
                + .00005*(self.opacity-self.base_opacity).square().mean())


def masked_detail_losses(predicted, target, region):
    """Match measured gradients, not merely increase image sharpness/contrast."""
    if predicted.shape != target.shape or predicted.shape != region.shape or predicted.ndim != 2:
        raise ValueError('Image and mask shapes must agree')
    mask = region.to(predicted.dtype)
    rgb = ((predicted-target).abs()*mask).sum()/mask.sum().clamp_min(1)
    mx, my = mask[:, 1:]*mask[:, :-1], mask[1:]*mask[:-1]
    dx = (predicted[:, 1:]-predicted[:, :-1])-(target[:, 1:]-target[:, :-1])
    dy = (predicted[1:]-predicted[:-1])-(target[1:]-target[:-1])
    gradient = ((dx.abs()*mx).sum()/mx.sum().clamp_min(1)
                + (dy.abs()*my).sum()/my.sum().clamp_min(1))*.5
    wanted = (target > .12)*mask
    coverage = ((.08-predicted).clamp_min(0)*wanted).sum()/(.08*wanted.sum().clamp_min(1))
    return rgb, gradient, coverage


def masked_laplacian_error(predicted, target, region):
    """Match signed second derivatives to the measured image; never reward edge energy alone."""
    if predicted.shape != target.shape or predicted.shape != region.shape or predicted.ndim != 2:
        raise ValueError('Image and mask shapes must agree')
    if min(predicted.shape) < 3:
        raise ValueError('Laplacian needs at least a 3 by 3 image')
    error = predicted-target
    laplacian = (error[1:-1, :-2]+error[1:-1, 2:]+error[:-2, 1:-1]+error[2:, 1:-1]
                 -4*error[1:-1, 1:-1])
    mask = region.to(predicted.dtype)
    valid = (mask[1:-1, 1:-1]*mask[1:-1, :-2]*mask[1:-1, 2:]
             *mask[:-2, 1:-1]*mask[2:, 1:-1])
    return (laplacian.abs()*valid).sum()/valid.sum().clamp_min(1)
