"""Display-only sampling; independent of the audited event fitting operator."""
import math

HYBRID_QUALITIES = ('surface', 'native', 'balanced', 'detail')
DEFAULT_HYBRID_QUALITY = 'surface'
MAX_RENDER_SIDE = 3072
MAX_RENDER_PIXELS = 3072 ** 2


def sampling_plan(width, height, quality=DEFAULT_HYBRID_QUALITY, native_supersample=1):
    if quality not in HYBRID_QUALITIES:
        raise ValueError('Unknown hybrid quality')
    if (type(width) is not int or type(height) is not int or min(width, height) < 1
            or max(width, height) > MAX_RENDER_SIDE):
        raise ValueError('Invalid output dimensions')
    if type(native_supersample) is not int or native_supersample not in (1, 2):
        raise ValueError('Invalid native sampling factor')
    factor = native_supersample if quality in ('surface', 'native') else 2 if quality == 'balanced' else 3
    scale = min(factor, MAX_RENDER_SIDE / max(width, height),
                math.sqrt(MAX_RENDER_PIXELS / (width * height)))
    return max(width, int(width * scale)), max(height, int(height * scale))


def resolve_hybrid_rgb(rgb, width, height, quality=DEFAULT_HYBRID_QUALITY):
    """Clamp RGB, integrating only when the caller requested oversampling.

    Surface/native modes use the source GS dimensions. Display denoising is a
    separate operation requiring depth/alpha guides; it is not applied here.
    """
    import torch.nn.functional as F
    if quality not in HYBRID_QUALITIES:
        raise ValueError('Unknown hybrid quality')
    if rgb.ndim != 3 or rgb.shape[0] != 3 or rgb.shape[1] < height or rgb.shape[2] < width:
        raise ValueError('Expected supersampled CHW RGB')
    source = rgb.clamp(0, 1)
    if source.shape[1] % height == 0 and source.shape[2] % width == 0:
        return F.avg_pool2d(source, kernel_size=(source.shape[1] // height, source.shape[2] // width))
    return F.interpolate(source.unsqueeze(0), size=(height, width), mode='area')[0]


def smooth_surface(rgb, depth, alpha):
    """Conservative display denoising inside opaque, continuous-depth surfaces.

    A 3x3 joint bilateral average rejects depth discontinuities and translucent
    pixels, attenuates smoothing near strong color edges, and caps changes to
    0.035 per channel. Source geometry/colors and previous frames are untouched.
    Depth is the rasterizer's premultiplied expected depth (sum(z * alpha * T)).
    """
    import torch
    import torch.nn.functional as F
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError('Expected CHW RGB')
    height, width = rgb.shape[1:]
    if depth.shape != (1, height, width) or alpha.shape != depth.shape:
        raise ValueError('Depth and alpha must be 1HW at the output resolution')
    valid = (alpha > 1e-6) & torch.isfinite(depth) & (depth > 0)
    z = torch.where(valid, depth / alpha.clamp_min(1e-6), 0.)

    def patches(tensor):
        padded = F.pad(tensor.unsqueeze(0), (1, 1, 1, 1), mode='replicate')
        return F.unfold(padded, kernel_size=3).reshape(tensor.shape[0], 9, height, width)

    neighbors = patches(rgb)
    neighbor_z = patches(z)[0]
    supported = patches(valid.to(rgb.dtype))[0] > .5
    opacity_blend = ((patches(alpha)[0].amin(0) - .98) / .02).clamp(0, 1)
    relative_depth = (neighbor_z - z) / z.clamp_min(1e-6)
    color_difference = ((neighbors - rgb[:, None]) ** 2).mean(0)
    spatial = rgb.new_tensor([.5, .75, .5, .75, 1., .75, .5, .75, .5])[:, None, None]
    weights = torch.exp(-color_difference / (2 * .08 ** 2)
                        -relative_depth.square() / (2 * .005 ** 2)) * spatial * supported
    filtered = (neighbors * weights[None]).sum(1) / weights.sum(0).clamp_min(1e-6)
    color_range = (neighbors.amax(1) - neighbors.amin(1)).amax(0)
    edge_blend = ((.25 - color_range) / .15).clamp(0, 1)
    depth_blend = ((.02 - relative_depth.abs().amax(0)) / .015).clamp(0, 1)
    interior = valid[0] & supported.all(0)
    delta = ((filtered - rgb) * .75).clamp(-.035, .035)
    return (rgb + delta * edge_blend * opacity_blend * depth_blend * interior).clamp(0, 1)
