"""Surface-normal constraints on conditional 4D Gaussian spatial support."""


def normal_surface_loss(centers, packed_covariance, surface, normals, nearest, thickness=.003):
    """Ignore normal sign; penalize displacement and excess normal thickness.

    Packed covariance order is xx,xy,xz,yy,yz,zz. Tangential extent remains
    unconstrained so valid surface coverage is not uniformly shrunk.
    """
    normal=normals[nearest]
    offset=((centers-surface[nearest])*normal).sum(-1).abs()
    x,y,z=normal.unbind(-1)
    c=packed_covariance
    variance=c[:,0]*x*x+2*c[:,1]*x*y+2*c[:,2]*x*z+c[:,3]*y*y+2*c[:,4]*y*z+c[:,5]*z*z
    excess=(variance.clamp_min(1e-12).sqrt()-thickness).clamp_min(0)
    return offset.mean()+excess.mean()
