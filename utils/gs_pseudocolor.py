"""Display-only false colors attached to rest-space Gaussian positions."""
import torch
from utils.gs_clay import gray_gaussians


def srgb_to_oklab(color):
    """Oklab matrices from https://bottosson.github.io/posts/oklab/ (public domain)."""
    linear = torch.where(color <= .04045, color/12.92, ((color+.055)/1.055).pow(2.4))
    cone = linear @ color.new_tensor([
        [.4122214708, .5363325363, .0514459929],
        [.2119034982, .6806995451, .1073969566],
        [.0883024619, .2817188376, .6299787005]]).T
    cone = cone.sign()*cone.abs().pow(1/3)
    return cone @ color.new_tensor([
        [.2104542553, .7936177850, -.0040720468],
        [1.9779984951, -2.4285922050, .4505937099],
        [.0259040371, .7827717662, -.8086757660]]).T


def oklab_to_srgb(color):
    cone = color @ color.new_tensor([
        [1., .3963377774, .2158037573],
        [1., -.1055613458, -.0638541728],
        [1., -.0894841775, -1.2914855480]]).T
    linear = cone.pow(3) @ color.new_tensor([
        [4.0767416621, -3.3077115913, .2309699292],
        [-1.2684380046, 2.6097574011, -.3413193965],
        [-.0041960863, -.7034186147, 1.7076147010]]).T
    return torch.where(linear <= .0031308, 12.92*linear,
                       1.055*linear.clamp_min(0).pow(1/2.4)-.055)


def palette(value):
    """C1-continuous Oklab gradient, with restrained blue/teal/gold/coral colors."""
    stops = srgb_to_oklab(value.new_tensor([
        [.19, .26, .60], [.13, .48, .72], [.16, .70, .72],
        [.40, .77, .61], [.94, .78, .43], [.90, .43, .32]]))
    position = value.clamp(0, 1)*(len(stops)-1)
    index = position.floor().long().clamp_max(len(stops)-2)
    fraction = (position-index)[..., None]
    p0, p1 = stops[(index-1).clamp_min(0)], stops[index]
    p2, p3 = stops[index+1], stops[(index+2).clamp_max(len(stops)-1)]
    # Catmull-Rom avoids the broad flat bands of easing separately at each stop.
    lab = .5*(2*p1+(-p0+p2)*fraction+(2*p0-5*p1+4*p2-p3)*fraction.square()
              +(-p0+3*p1-3*p2+p3)*fraction.pow(3))
    return oklab_to_srgb(lab).clamp(0, 1)


def pseudocolor_materials(rest_positions):
    """One shared range for V1/V3; calculate once before applying any motion."""
    if not rest_positions or any(x.ndim != 2 or x.shape[1] != 3 or len(x) == 0 for x in rest_positions):
        raise ValueError('Expected non-empty Nx3 rest positions for each side')
    values = [xyz @ xyz.new_tensor([.65, .20, .15]) for xyz in rest_positions]
    low = torch.stack([v.min() for v in values]).min()
    high = torch.stack([v.max() for v in values]).max()
    return [palette((value-low)/(high-low).clamp_min(1e-6)).contiguous() for value in values]


def soft_color_finish(color, shade):
    """Bounded diffuse tone and soft neutral highlights; no micro-normal specular."""
    diffuse = (shade/.82).clamp(.25, 1).pow(1.08)
    highlight = ((shade-.68)/.28).clamp(0, 1)
    highlight = .16*highlight.square()*(3-2*highlight)
    base = color*diffuse
    return base+(1-base)*highlight


def color_gaussians(rgb, depth, alpha, material, fovy, background):
    """Combine premultiplied material colors with clean soft-white form shading."""
    if material.shape != rgb.shape:
        raise ValueError('Rendered material colors must match CHW RGB')
    a = alpha.nan_to_num().clamp(0, 1)
    shade = gray_gaussians(rgb, depth, a, fovy, [0., 0., 0.])[:1]/a.clamp_min(1e-5)
    color = (material.nan_to_num()/a.clamp_min(1e-5)).clamp(0, 1)
    surface = soft_color_finish(color, shade)
    bg = torch.as_tensor(background, dtype=rgb.dtype, device=rgb.device).reshape(3, 1, 1)
    return surface*a+bg*(1-a)


def pseudocolor_info():
    return dict(available=True, representation='gaussian', default_resolution=1024,
                left_label='V1 · 伪彩色', right_label='V3 · 伪彩色',
                description='靛蓝、湖青、柔金与珊瑚橙平滑渐变，柔和亮部与清晰部件层次；配色固定在表面、随动画运动，两侧统一配色。颜色为展示配色，不代表真实材质或误差。')
