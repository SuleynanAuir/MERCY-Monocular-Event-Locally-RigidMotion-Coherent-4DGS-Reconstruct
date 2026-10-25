"""Display-only Lamp detail finish driven by animated surface normals.

This does not train or modify the Gaussian geometry. Keep the source appearance
as the low-frequency guide; do not treat enhanced shading as accuracy evidence.
"""
import math
import torch
import torch.nn.functional as F
from utils.gs_clay import shifted, filtered_depth, depth_normals


def height_weights(xyz, settings=None):
    """Keep the clean static base unchanged, fading into detail above its top."""
    if settings is None:
        return torch.ones_like(xyz[:, :1])
    axis = xyz.new_tensor(settings['axis'])
    start, end = float(settings['start']), float(settings['end'])
    if (axis.shape != (3,) or not torch.isfinite(axis).all()
            or abs(float(axis.norm())-1) > 1e-4
            or not math.isfinite(start+end) or end <= start):
        raise ValueError('Invalid canonical detail-height envelope')
    weight = ((xyz @ axis-start)/(end-start)).clamp(0, 1)
    return (weight.square()*(3-2*weight))[:, None]


def detail_finish(rgb, normals, depth, alpha, fovy, background, mode):
    """Resolve supported surface normals, local contrast and restrained highlights."""
    a=alpha.clamp(0,1)
    direction=normals/a.clamp_min(1e-5)*2-1
    strength=direction.norm(dim=0,keepdim=True).clamp(0,1)
    n=F.normalize(direction,dim=0,eps=1e-6)
    n=torch.where(n[2:]<0,-n,n)
    bg=torch.as_tensor(background, dtype=rgb.dtype, device=rgb.device).reshape(3,1,1)
    surface=((rgb-bg*(1-a))/a.clamp_min(1e-5)).clamp(0,1)
    z=depth/a.clamp_min(1e-5)
    def blur(x,size):
        return F.avg_pool2d(F.pad(x,(size//2,)*4,mode='replicate'),size,1)
    interior=a>.99
    for x,y in ((2,0),(-2,0),(0,2),(0,-2)):
        interior&=(shifted(a,x,y)>.99)&((shifted(z,x,y)-z).abs()<z*.02)
    # Only supported interior colour detail is enhanced; no silhouette halos.
    middle=blur(surface*a,9)/blur(a,9).clamp_min(1e-5)
    detail=(surface-middle).clamp(-.10,.10)*interior
    key=F.normalize(n.new_tensor([-.45,.65,.9]),dim=0)[:,None,None]
    fill=F.normalize(n.new_tensor([.65,.15,.6]),dim=0)[:,None,None]
    ndl=(n*key).sum(0,keepdim=True).clamp_min(0)
    diffuse=.82+.26*ndl+.08*(n*fill).sum(0,keepdim=True).clamp_min(0)
    h,w=a.shape[-2:];f=h/(2*math.tan(math.radians(fovy)/2))
    yy,xx=torch.meshgrid(torch.arange(h,device=n.device)+.5-h/2,torch.arange(w,device=n.device)+.5-w/2,indexing='ij')
    view=F.normalize(torch.stack([-xx,yy,torch.full_like(xx,f)]),dim=0)
    half=F.normalize(key+view,dim=0)
    spec=(n*half).sum(0,keepdim=True).clamp_min(0).pow(36)
    # Contact shadows from nearby visible depth, restricted to a small radius.
    occ=torch.zeros_like(a);mass=torch.zeros_like(a)
    for radius in (3,8,18):
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)):
            other=shifted(z,dx*radius,dy*radius);oa=shifted(a,dx*radius,dy*radius)
            scale=z*radius/f
            vector=torch.stack([((xx+dx*radius)*other[0]-xx*z[0])/f,
                                -((yy+dy*radius)*other[0]-yy*z[0])/f,
                                (z-other)[0]])
            distance=vector.norm(dim=0,keepdim=True).clamp_min(1e-6)
            height=(vector*n).sum(0,keepdim=True)
            supported=(a>.95)&(oa>.95)&(height>scale*.12)&(distance<scale*5)
            occ+=supported*(height/distance).clamp(0,1)*(1-distance/(scale*5).clamp_min(1e-6)).clamp(0,1)
            mass+=1
    ao=1-.30*occ/mass.clamp_min(1)
    spec_strength=.12 if mode=='gaussian' else .10
    finish=(surface+.35*detail)*diffuse*ao
    finish=finish+(1-finish)*spec*spec_strength
    enhanced=finish.clamp(0,1)*a+bg*(1-a)
    # A transferred normal must agree with the visible GS surface. Unsupported
    # source geometry cannot add a false crease or a new dark lighting patch.
    guide_depth,_=filtered_depth(depth,a)
    geometry=depth_normals(guide_depth,a,fovy)
    agreement=(n*geometry).sum(0,keepdim=True).abs()
    support=((agreement-.65)/.25).clamp(0,1)
    support=blur(support.square()*(3-2*support),5)*strength
    return rgb+support*(enhanced-rgb)
