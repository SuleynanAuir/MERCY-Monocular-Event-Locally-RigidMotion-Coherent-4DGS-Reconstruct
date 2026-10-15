"""Confidence-gated, bounded Gaussian color residuals with frozen geometry.

The residual is a scalar linear-RGB increment shared by all color channels.
It preserves source chroma and cannot change opacity, covariance or motion.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

C0 = .28209479177387814
DETAIL_ROIS = {'arm': (160, 140, 380, 380), 'gear': (360, 230, 500, 325),
               'cabin': (399, 146, 540, 321), 'platform': (12, 368, 628, 418)}


def split_timeline(frame_count):
    """Every fifth RGB frame is held out; its neighboring intervals cannot train.

    Event-heldout intervals retain the existing i % 5 == 4 audit convention.
    RGB heldout means timestamps withheld from this residual stage only, of
    the SAME camera. The inherited model may already have used those images.
    """
    if type(frame_count) is not int or frame_count < 20:
        raise ValueError('Need an integer timeline of at least twenty frames')
    frames = np.arange(frame_count, dtype=np.int32)
    held_rgb = frames[frames % 5 == 4]
    intervals = np.arange(frame_count-1, dtype=np.int32)
    held_event = intervals[intervals % 5 == 4]
    train = intervals[~np.isin(intervals, held_rgb)
                      & ~np.isin(intervals+1, held_rgb) & (intervals % 5 != 4)]
    return train, held_event, held_rgb


class BoundedColorResidual(nn.Module):
    def __init__(self, source_dc, indices, maximum_delta=.08):
        super().__init__()
        if (source_dc.ndim != 3 or source_dc.shape[1:] != (1, 3)
                or not torch.isfinite(source_dc).all() or indices.ndim != 1
                or indices.dtype != torch.int64 or not len(indices)
                or torch.any(indices < 0) or torch.any(indices >= len(source_dc))
                or torch.any(indices[1:] <= indices[:-1])
                or not np.isfinite(maximum_delta) or not 0 < maximum_delta <= .15):
            raise ValueError('Invalid bounded residual source, indices or RGB bound')
        self.register_buffer('source_dc', source_dc.detach().clone())
        self.register_buffer('indices', indices.to(source_dc.device).clone())
        self.maximum_delta = float(maximum_delta)
        self.delta = nn.Parameter(source_dc.new_zeros(len(indices)))

    def forward(self):
        delta = self.delta.clamp(-self.maximum_delta, self.maximum_delta)
        selected = self.source_dc[self.indices] + delta[:, None, None]/C0
        return self.source_dc.index_copy(0, self.indices, selected)

    @torch.no_grad()
    def project_(self):
        self.delta.clamp_(-self.maximum_delta, self.maximum_delta)

    def regularization(self):
        return self.delta.square().mean()


def weighted_l1(error, weight):
    if error.shape != weight.shape or error.ndim != 2:
        raise ValueError('Error and weights must be matching 2D images')
    return (error.abs()*weight).sum()/weight.sum().clamp_min(1)


def detail_terms(predicted, target, weight):
    """Match signed measured derivatives; never maximize edge energy blindly."""
    if predicted.shape != target.shape or predicted.shape != weight.shape or min(predicted.shape) < 3:
        raise ValueError('Image and weight shapes must agree and be at least 3x3')
    error = predicted-target
    photo = weighted_l1(error, weight)
    dx = error[:, 1:]-error[:, :-1]
    dy = error[1:]-error[:-1]
    gradient = .5*(weighted_l1(dx, torch.minimum(weight[:, 1:], weight[:, :-1]))
                   + weighted_l1(dy, torch.minimum(weight[1:], weight[:-1])))
    lap = error[1:-1, :-2]+error[1:-1, 2:]+error[:-2, 1:-1]+error[2:, 1:-1]-4*error[1:-1, 1:-1]
    laplacian = weighted_l1(lap, weight[1:-1, 1:-1])
    return photo, gradient, laplacian


def multiscale_detail_terms(predicted, target, weight):
    terms = [predicted.new_zeros(()) for _ in range(3)]
    for scale, factor in ((1, .6), (2, .3), (4, .1)):
        arrays = [predicted, target, weight]
        if scale != 1:
            arrays = [F.avg_pool2d(a[None, None], scale)[0, 0] for a in arrays]
        for index, value in enumerate(detail_terms(*arrays)):
            terms[index] = terms[index]+factor*value
    return tuple(terms)


def local_detail_loss(predicted, target, rois=DETAIL_ROIS):
    values = []
    for x0, y0, x1, y1 in rois.values():
        if not (0 <= x0 < x1 <= predicted.shape[1] and 0 <= y0 < y1 <= predicted.shape[0]):
            raise ValueError('Detail ROI lies outside the native image')
        p, t = predicted[y0:y1, x0:x1], target[y0:y1, x0:x1]
        photo, gradient, laplacian = detail_terms(p, t, torch.ones_like(p))
        values.append(photo+.75*gradient+.1*laplacian)
    if not values:
        raise ValueError('At least one detail ROI is required')
    return torch.stack(values).mean()


def normalized_event_loss(predicted, target):
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError('Event predictions and reference must have identical 2D shape')
    l1 = (predicted-target).abs().sum()/target.abs().sum().clamp_min(1)
    cosine = 1-(predicted*target).sum()/(predicted.norm()*target.norm()).clamp_min(1e-6)
    return l1+.5*cosine


def temporal_nonregression_loss(predicted, target, parent, l1_slack=.015, cosine_slack=.005):
    """Training-only barrier against worsening an inherited event prediction.

    It is not an acceptance gate and does not relax the unchanged audit policy.
    """
    if predicted.shape != target.shape or predicted.shape != parent.shape or predicted.ndim != 2:
        raise ValueError('Temporal guard arrays must have matching 2D shapes')
    mass = target.abs().sum().clamp_min(1)
    excess = ((predicted-target).abs().sum()-(parent-target).abs().sum())/mass
    target_norm = target.norm()
    cosine = (predicted*target).sum()/(predicted.norm()*target_norm).clamp_min(1e-6)
    old_cosine = (parent*target).sum()/(parent.norm()*target_norm).clamp_min(1e-6)
    return (excess-l1_slack).clamp_min(0)+(old_cosine-cosine-cosine_slack).clamp_min(0)
