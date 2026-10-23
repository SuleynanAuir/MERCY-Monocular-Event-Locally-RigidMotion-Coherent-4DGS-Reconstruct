"""Neutral studio material for inspecting the actual animated mesh surface."""
import torch
import torch.nn.functional as F


def posed_clay_normals(normals, quaternions, groups, camera_rotation, frame, deformation=None):
    """Transport a canonical normal field through native and V3 joint motion.

    The deployed auxiliary MLP predicts rigid joint residuals. Translations
    must not affect normals; delta_q must be applied after native rotation,
    just as in deform_mesh_vertices. No screen-space or per-frame filtering.
    """
    from utils.event_refinement import qrotate
    posed = qrotate(quaternions[groups], normals)
    if deformation is not None and frame != 0 and torch.any(deformation.angles[frame]):
        posed = qrotate(deformation.delta_q[frame][groups], posed)
    return F.normalize(posed @ camera_rotation, dim=-1, eps=1e-12)


def vertex_normals(vertices, faces):
    """Area-weighted smooth normals, including finite degenerate/unused vertices."""
    indices = faces.long()
    triangles = vertices[indices]
    normals = torch.cross(triangles[:, 1] - triangles[:, 0],
                          triangles[:, 2] - triangles[:, 0], dim=-1)
    result = torch.zeros_like(vertices)
    for corner in range(3):
        result.index_add_(0, indices[:, corner], normals)
    return F.normalize(result, dim=-1, eps=1e-12)


def studio_clay(normals, positions):
    """Two-sided matte gray, with camera-relative key/fill and broad highlights.

    Positions and normals must be in camera space (camera looks toward -Z).
    No baked Gaussian colors are used. Lighting follows the camera during orbit.
    """
    normal = F.normalize(normals, dim=-1, eps=1e-12)
    view = F.normalize(-positions, dim=-1, eps=1e-12)
    normal = torch.where((normal * view).sum(-1, keepdim=True) < 0, -normal, normal)
    key = F.normalize(normal.new_tensor([-.5, .75, 1.]), dim=0)
    fill = F.normalize(normal.new_tensor([.7, .1, .5]), dim=0)
    diffuse = .25 + .53 * (normal * key).sum(-1).clamp_min(0)
    diffuse = diffuse + .16 * (normal * fill).sum(-1).clamp_min(0)
    halfway = F.normalize(key + view, dim=-1, eps=1e-12)
    highlight = .06 * (normal * halfway).sum(-1).clamp_min(0).pow(24)
    # Linear-light material followed by a display transfer; soft but not flat.
    intensity = (.66 * diffuse + highlight).clamp(0, 1).pow(1 / 2.2)
    return intensity[..., None].expand(*intensity.shape, 3).contiguous()


def clay_info(mesh_info):
    info = dict(mesh_info, left_label='V1 · 灰模 / Clay', right_label='V3 · 灰模 / Clay',
                description='统一灰色材质与柔和棚灯，突出轮廓和表面起伏；左右使用各自网格与运动，支持同步旋转、缩放和播放。')
    if mesh_info.get('surface_regularized'):
        info['description'] = '按运动部件去噪、局部补洞后的灰模表面；保留主要轮廓、关节与同步运动。'
    if mesh_info.get('feature_reconstructed'):
        info['description'] = '保边表面去噪与局部补洞，减少颗粒感并保留轮廓；表面光照随部件同步运动。'
        info['default_resolution'] = 1024
    return info
