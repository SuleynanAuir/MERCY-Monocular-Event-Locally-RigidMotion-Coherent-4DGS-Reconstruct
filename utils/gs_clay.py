"""Display-only soft clay shading of native Gaussian RGB/depth/coverage.

No mesh extraction, source edits, temporal accumulation or model retraining.
Depth-derived lighting is illustrative, not physically calibrated relighting.
"""
import math
import torch
import torch.nn.functional as F


def shifted(image, dx, dy):
    h, w = image.shape[-2:]
    pad = max(abs(dx), abs(dy))
    if not pad: return image
    return F.pad(image, (pad, pad, pad, pad), mode='replicate')[..., pad+dy:pad+dy+h, pad+dx:pad+dx+w]


def filtered_depth(depth, alpha):
    valid = torch.isfinite(depth) & (depth > 0) & (alpha > .05)
    z = torch.where(valid, depth/alpha.clamp_min(1e-6), 0.)
    # Only the lighting guide is filtered. Native RGB and coverage remain at
    # full resolution, so this cannot fill holes or blur the silhouette.
    for stride in (1, 2):
        total, mass = z.clone(), torch.ones_like(z)
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                if dx == dy == 0: continue
                neighbor = shifted(z, dx*stride, dy*stride)
                a = shifted(alpha, dx*stride, dy*stride)
                supported = shifted(valid.to(z.dtype), dx*stride, dy*stride) > .5
                relative = (neighbor-z)/z.clamp_min(1e-6)
                weight = torch.exp(-.5*(dx*dx+dy*dy)/1.5**2-.5*(relative/.012)**2)
                weight *= supported * a.clamp(0, 1) * (relative.abs() < .04)
                total += weight*neighbor
                mass += weight
        z = torch.where(valid, total/mass, 0.)
    return z, valid


def depth_normals(z, alpha, fovy):
    h, w = z.shape[-2:]
    focal = h/(2*math.tan(math.radians(fovy)/2))
    gradients = []
    for dx, dy in ((2, 0), (0, 2)):
        plus, minus = shifted(z, dx, dy), shifted(z, -dx, -dy)
        good_plus = (shifted(alpha, dx, dy) > .5) & ((plus-z).abs() < z*.025)
        good_minus = (shifted(alpha, -dx, -dy) > .5) & ((minus-z).abs() < z*.025)
        forward = torch.where(good_plus, (plus-z)/2, 0.)
        backward = torch.where(good_minus, (z-minus)/2, 0.)
        gradients.append((forward+backward)/(good_plus.to(z.dtype)+good_minus.to(z.dtype)).clamp_min(1))
    gx, gy = gradients
    yy, xx = torch.meshgrid(torch.arange(h, device=z.device, dtype=z.dtype)+.5-h/2,
                           torch.arange(w, device=z.device, dtype=z.dtype)+.5-w/2, indexing='ij')
    n = torch.cat((focal*gx, -focal*gy, z-xx*gx-yy*gy), dim=0)
    fallback = torch.zeros_like(n); fallback[2] = 1
    return F.normalize(torch.where((alpha > .05).expand_as(n), n, fallback), dim=0, eps=1e-8)


def canonical_coordinates(xyz, bounds=None):
    """Immutable material coordinates; the splats carry these through motion."""
    if bounds is None:
        low, high = xyz.amin(0), xyz.amax(0)
        bounds = ((low+high)/2, (high-low).amax().clamp_min(1e-6)*1.02)
    center, extent = bounds
    return ((xyz-center)/extent+.5).contiguous(), bounds


def surface_wire(coordinates, alpha, spacing=.022):
    """Antialiased object-space grid, not recovered mesh topology.

    Blend the three planar charts by canonical surface orientation. Coordinates
    are rendered by the same splats/occlusion as GS, never anchored to the screen.
    Fade unresolved cells and unstable depth boundaries rather than alias them.
    """
    p = coordinates.nan_to_num()/alpha.clamp_min(1e-5)
    # Smooth only the material chart, not GS coverage or native RGB. This
    # removes subpixel coordinate wobble while rejecting separate surfaces.
    for stride in (1, 2):
        total, mass = p.clone(), torch.ones_like(alpha)
        for ox, oy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
            neighbor = shifted(p, ox*stride, oy*stride)
            delta = (neighbor-p).square().sum(0, keepdim=True)
            weight = torch.exp(-delta/(spacing*.8)**2)*shifted(alpha, ox*stride, oy*stride)
            weight *= delta < (spacing*1.5)**2
            total += weight*neighbor
            mass += weight
        p = total/mass
    dx = (shifted(p, 1, 0)-shifted(p, -1, 0))*.5
    dy = (shifted(p, 0, 1)-shifted(p, 0, -1))*.5
    normal = F.normalize(torch.linalg.cross(dx, dy, dim=0), dim=0, eps=1e-10)
    normal = F.avg_pool2d(F.pad(normal, (2, 2, 2, 2), mode='replicate'), 5, stride=1)
    weight = normal.abs().pow(8)
    weight /= weight.sum(0, keepdim=True).clamp_min(1e-10)
    width = ((dx.abs()+dy.abs())/spacing).clamp_min(1e-5)
    distance = (torch.remainder(p/spacing+.5, 1)-.5).abs()
    line = (1-distance/(width*.85)).clamp(0, 1)
    # Suppress near-Nyquist pattern and coordinates crossing distinct parts.
    line *= ((.40-width)/.18).clamp(0, 1)
    charts = torch.stack((torch.maximum(line[1], line[2]),
                          torch.maximum(line[0], line[2]),
                          torch.maximum(line[0], line[1])))
    support = alpha > .85
    for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        support &= shifted(alpha, ox, oy) > .85
    continuous = (dx.norm(dim=0) < spacing*.6) & (dy.norm(dim=0) < spacing*.6)
    return (charts*weight).sum(0, keepdim=True)*support*continuous


def suppress_isolated_grain(luminance, alpha, depth):
    """Reject isolated bright/dark impulses, retaining directionally supported lines.

    Only operate inside opaque, depth-continuous surfaces. Coverage, holes and
    thin silhouettes are untouched; this is display filtering, not new geometry.
    """
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))
    neighbors = [shifted(luminance, x, y) for x, y in offsets]
    median = torch.stack(neighbors+[luminance]).median(0).values
    support = torch.zeros_like(luminance)
    for index in (0, 2, 4, 6):
        error = torch.maximum((neighbors[index]-luminance).abs(),
                              (neighbors[index+1]-luminance).abs())
        support = torch.maximum(support, torch.exp(-.5*(error/.065)**2))
    interior = alpha > .98
    for x, y in offsets:
        interior &= (shifted(alpha, x, y) > .98) & ((shifted(depth, x, y)-depth).abs() < depth*.02)
    blend = (1-support)*((luminance-median).abs()/.12).clamp(0, 1)*interior
    return luminance+blend*(median-luminance)


def clean_luminance(luminance, alpha, depth):
    """Suppress isolated grain while preserving dark features and thin edges."""
    luminance = suppress_isolated_grain(luminance, alpha, depth)
    patches = torch.stack([
        torch.where(shifted(alpha, x, y) > .8, shifted(luminance, x, y), luminance)
        for y in (-1, 0, 1) for x in (-1, 0, 1)])
    guide = patches.median(0).values
    total, mass = torch.zeros_like(luminance), torch.zeros_like(alpha)
    for y in range(-2, 3):
        for x in range(-2, 3):
            delta = (shifted(guide, x, y)-guide)/.075
            dz = (shifted(depth, x, y)-depth)/depth.clamp_min(1e-5)
            weight = torch.exp(-.5*(x*x+y*y)/1.25**2-.5*delta**2-.5*(dz/.015)**2)
            weight *= (shifted(alpha, x, y) > .8)*(dz.abs() < .04)
            total += weight*shifted(luminance, x, y)
            mass += weight
    smooth = total/mass.clamp_min(1e-5)
    supported = (alpha > .95) & (mass > 2) & ((luminance-guide) > -.18)
    return torch.where(supported, .85*smooth+.15*luminance, luminance)


def broad_lighting(depth, alpha, fovy):
    """Low-frequency form light; no noisy specular or micro-occlusion relief."""
    size = tuple(max(1, dimension//4) for dimension in alpha.shape[-2:])
    a = F.interpolate(alpha[None], size=size, mode='area')[0]
    d = F.interpolate((depth*alpha)[None], size=size, mode='area')[0]
    z, _ = filtered_depth(d, a)
    normal = depth_normals(z, a, fovy)
    key = F.normalize(normal.new_tensor([-.4, .6, .7]), dim=0)[:, None, None]
    light = .36+.58*(normal*key).sum(0, keepdim=True).clamp_min(0)
    return F.interpolate(light[None], size=alpha.shape[-2:], mode='bilinear', align_corners=False)[0]


def gray_gaussians(rgb, depth, alpha, fovy, background, detail_strength=.68, coordinates=None):
    """Shade premultiplied native GS rendered against black; return CHW RGB."""
    if rgb.ndim != 3 or rgb.shape[0] != 3 or depth.shape != rgb[:1].shape or alpha.shape != depth.shape:
        raise ValueError('Expected CHW RGB and matching 1HW depth/alpha')
    a = alpha.nan_to_num().clamp(0, 1)
    z, valid = filtered_depth(depth, a)
    light = broad_lighting(z, a, fovy)
    albedo = (rgb.nan_to_num()/a.clamp_min(1e-5)).clamp(0, 1)
    luminance = (albedo*albedo.new_tensor([.2126, .7152, .0722])[:, None, None]).sum(0, keepdim=True)
    luminance = clean_luminance(luminance, a, z)
    # Retain supported grooves/studs without turning depth noise into bumps.
    def layer(size):
        pad = (size//2,)*4
        pooled_a = F.avg_pool2d(F.pad(a, pad, mode='replicate'), size, stride=1)
        pooled_l = F.avg_pool2d(F.pad(luminance*a, pad, mode='replicate'), size, stride=1)
        return pooled_l/pooled_a.clamp_min(1e-5)
    middle, base = layer(9), layer(31)
    # Separate part-scale contrast from fine detail: readable ribs and recesses
    # without sharpening the finest residual grain or adding specular relief.
    shape_detail = (middle-base).clamp(-.5, .3)
    fine_detail = (luminance-middle).clamp(-.5, .25)
    shade = (.52+.34*base.sqrt()+shape_detail+detail_strength*fine_detail+.10*(light-.85)).clamp(0, 1)
    if coordinates is not None:
        if coordinates.shape != rgb.shape:
            raise ValueError('Material coordinates must match CHW RGB')
        shade *= 1-.14*surface_wire(coordinates, a)
    shade = torch.where(valid, shade, torch.full_like(shade, .74))
    bg = torch.as_tensor(background, dtype=rgb.dtype, device=rgb.device).reshape(3, 1, 1)
    return (shade.expand_as(rgb)*a + bg*(1-a)).clamp(0, 1)


def gs_clay_info():
    return dict(available=True, representation='gaussian', default_resolution=1024,
                left_label='V1 · 柔光白模', right_label='V3 · 柔光白模',
                description='清晰柔光白模，抑制表面颗粒、增强部件层次；保留连续筋条、孔边与轮廓，淡线框可选。')
