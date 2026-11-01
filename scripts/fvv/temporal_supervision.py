"""Exact observed-change L1 gradients with one render graph resident at a time."""


def backward_observed_change(first_render,second_render,observed,weight,strength=8.):
    """Render callbacks must be deterministic and share unchanged parameters."""
    import torch
    with torch.no_grad():
        first=first_render();second=second_render();error=second-first-observed
        loss=float(strength*(error.abs()*weight).mean()/weight.mean())
        gradient=strength*error.sign()*weight/(first.numel()*weight.mean())
    del first,second,error
    first_render().backward(-gradient)
    second_render().backward(gradient)
    return loss
