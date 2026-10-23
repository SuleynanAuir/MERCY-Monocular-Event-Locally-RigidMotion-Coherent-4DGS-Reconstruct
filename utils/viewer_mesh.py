"""Mesh reconstructed from Gaussian renders, with the current corrected rigid motion."""
import io
import json
from pathlib import Path
import time

import numpy as np

from utils.viewer_cache import EncodedFrameCache
from utils.viewer_state import BACKGROUNDS, image_packet, orbit_pose


def load_mesh_data(path, group_count, fingerprints):
    """Reject stale meshes and any triangles crossing independently moving groups."""
    with np.load(path, allow_pickle=False) as archive:
        vertices, faces, colors, groups = [archive[k] for k in
                                         ('vertices', 'triangles', 'colors', 'vertex_groups')]
        meta = json.loads(str(archive['metadata_json']))
        clay_normals = archive['clay_normals'] if 'clay_normals' in archive else None
        motion = ({k: archive[k] for k in ('motion_quaternions', 'motion_translations')}
                  if meta.get('format') == 'gaussian_tsdf_mesh_v2' else {})
    if (meta.get('format') not in ('gaussian_tsdf_mesh_v1', 'gaussian_tsdf_mesh_v2')
            or meta.get('source_fingerprints') != fingerprints):
        raise ValueError('Mesh was not reconstructed from the current Gaussian candidate')
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or colors.shape != vertices.shape or not np.isfinite(colors).all()
            or np.any(colors < 0) or np.any(colors > 1)
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype.kind not in 'iu' or np.any(faces < 0) or np.any(faces >= len(vertices))
            or groups.shape != (len(vertices),) or groups.dtype.kind not in 'iu'
            or np.any(groups < 0) or np.any(groups >= group_count)):
        raise ValueError('Invalid reconstructed mesh geometry, colors or groups')
    face_groups = groups[faces]
    if clay_normals is not None and (
            not meta.get('clay_regularization', {}).get('feature_reconstruction')
            or clay_normals.shape != vertices.shape or not np.isfinite(clay_normals).all()
            or np.any(np.abs(np.linalg.norm(clay_normals, axis=1)-1) > 1e-3)):
        raise ValueError('Invalid canonical Clay normal field')
    if np.any(face_groups != face_groups[:, :1]):
        raise ValueError('A mesh face crosses independent motion groups')
    if motion:
        q, t = motion['motion_quaternions'], motion['motion_translations']
        frames = meta.get('frame_count')
        if (type(frames) is not int or frames < 2 or meta.get('source_side') not in ('left', 'right')
                or q.shape != (frames, group_count, 4) or t.shape != (frames, group_count, 3)
                or not np.isfinite(q).all() or not np.isfinite(t).all()
                or np.any(np.abs(np.linalg.norm(q, axis=-1) - 1) > 1e-4)
                or np.max(np.abs(q[0] - np.array([1, 0, 0, 0]))) > 1e-4 or np.max(np.abs(t[0])) > 1e-4):
            raise ValueError('Invalid embedded source motion')
    # NPZ may preserve column-major arrays from adaptive subdivision. CUDA
    # rasterization requires packed row-major triangle indices.
    return dict(vertices=np.ascontiguousarray(vertices, dtype=np.float32),
                faces=np.ascontiguousarray(faces, dtype=np.int32),
                colors=np.ascontiguousarray(colors, dtype=np.float32), vertex_groups=groups,
                face_groups=face_groups[:, 0], metadata=meta,
                clay_normals=None if clay_normals is None else np.ascontiguousarray(clay_normals, dtype=np.float32), **motion)


def projection_matrix(width, height, fovy, near=.01, far=100.):
    tangent = np.tan(np.deg2rad(fovy) / 2)
    return np.array([[height / width / tangent, 0, 0, 0],
                     [0, 1 / tangent, 0, 0],
                     [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                     [0, 0, -1, 0]], dtype=np.float32)


class MeshRenderer:
    def __init__(self, path, fingerprints, correction, base_pose, target, source_side=None):
        import torch
        import nvdiffrast.torch as dr

        data = load_mesh_data(Path(path), correction.group_count, fingerprints)
        if source_side is not None and data['metadata'].get('source_side') != source_side:
            raise ValueError('Mesh comparison side does not match its Gaussian source')
        self.motion = None
        if 'motion_quaternions' in data:
            if len(data['motion_quaternions']) != correction.frame_count:
                raise ValueError('Mesh timeline differs from Gaussian comparison')
            self.motion = tuple(torch.from_numpy(data[k].astype(np.float32)).cuda()
                                for k in ('motion_quaternions', 'motion_translations'))
        self.source_path, self.fingerprints = Path(path), fingerprints
        self.dr, self.correction = dr, correction
        self.base_pose, self.target = base_pose, target
        self.context = dr.RasterizeCudaContext()
        self.vertices = torch.from_numpy(data['vertices']).cuda()
        self.faces = torch.from_numpy(data['faces']).cuda()
        self.vertex_groups = torch.from_numpy(data['vertex_groups'].astype(np.int64)).cuda()
        self.colors = torch.from_numpy(data['colors']).cuda()
        self.clay_normals = (None if data['clay_normals'] is None else torch.from_numpy(data['clay_normals']).cuda())
        self.supersample = 2
        self.edge_antialias = True
        self.image_cache = EncodedFrameCache(max_bytes=48 * 1024**2)
        self.info = dict(available=True, triangle_count=len(self.faces), vertex_count=len(self.vertices),
                         excluded_gaussians=data['metadata'].get('excluded_elastic_gaussians', 0),
                         label='Mesh · 高斯重建', default_resolution=768,
                         source_side=data['metadata'].get('source_side', 'right'),
                         surface_denoised=bool(data['metadata'].get('surface_denoising')),
                         color_refined=bool(data['metadata'].get('color_refinement')),
                         surface_subdivided=bool(data['metadata'].get('surface_subdivision')),
                         render_supersample=self.supersample,
                         clay_regularization=data['metadata'].get('clay_regularization'),
                         description='由本侧高斯的多视角深度融合生成，使用本侧运动；低置信度表面和弹性带未包含。')

    def render_pixels(self, request):
        """Shade clay geometry separately; retain baked color in the texture mode."""
        import torch
        from utils.event_refinement import qrotate

        with torch.no_grad():
            q, t = (self.correction.group_transforms(request['frame']) if self.motion is None
                    else (values[request['frame']] for values in self.motion))
            xyz = qrotate(q[self.vertex_groups], self.vertices) + t[self.vertex_groups]
            if getattr(self, 'deformation', None) is not None:
                from utils.v3_deformation import deform_mesh_vertices
                xyz = deform_mesh_vertices(self.deformation, xyz, self.vertex_groups, request['frame'])
            pose = orbit_pose(self.base_pose, self.target, request)
            camera = torch.as_tensor(pose, device='cuda')
            xyz_camera = (xyz - camera[:3, 3]) @ camera[:3, :3]
            projection = torch.as_tensor(projection_matrix(
                request['width'], request['height'], request['fovy']), device='cuda')
            homogeneous = torch.cat((xyz_camera, torch.ones_like(xyz_camera[:, :1])), dim=-1)
            clip = (homogeneous @ projection.T).unsqueeze(0).contiguous()
            raster, _ = self.dr.rasterize(self.context, clip, self.faces,
                                         [request['height'] * self.supersample,
                                          request['width'] * self.supersample], grad_db=False)
            visible = raster[0, ..., 3] > 0
            if request.get('render_mode') in {'clay', 'clay_mesh'}:
                from utils.clay_shading import vertex_normals, studio_clay, posed_clay_normals
                if self.clay_normals is None:
                    normals = vertex_normals(xyz_camera, self.faces)
                else:
                    normals = posed_clay_normals(self.clay_normals, q, self.vertex_groups,
                        camera[:3, :3], request['frame'], getattr(self, 'deformation', None))
                attributes = torch.cat((normals, xyz_camera), dim=-1)
                surface, _ = self.dr.interpolate(attributes.unsqueeze(0).contiguous(), raster, self.faces)
                color = studio_clay(surface[..., :3], surface[..., 3:])
            else:
                color, _ = self.dr.interpolate(self.colors.unsqueeze(0).contiguous(), raster, self.faces)
            background = color.new_tensor(BACKGROUNDS[request['background']])
            rgb = torch.where(visible[..., None], color[0], background)
            if self.edge_antialias:
                rgb = self.dr.antialias(rgb.unsqueeze(0).contiguous(), raster, clip, self.faces)[0]
            # Integrate subpixel triangles before quantization to reduce speckle
            # and crawling edges during animation. Output framing stays identical.
            rgb = rgb.reshape(request['height'], self.supersample,
                              request['width'], self.supersample, 3).mean(dim=(1, 3))
            return (rgb.flip(0).clamp(0, 1) * 255).byte().cpu().numpy()

    def render(self, request):
        from PIL import Image

        start = time.perf_counter()
        jpeg = self.image_cache.get(request)
        cache_hit = jpeg is not None
        render_ms = encode_ms = 0.
        if jpeg is None:
            pixels = self.render_pixels(request)
            rendered = time.perf_counter()
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format='JPEG', quality=request['quality'], subsampling=0)
            jpeg = stream.getvalue()
            self.image_cache.put(request, jpeg)
            render_ms, encode_ms = (rendered - start) * 1000, (time.perf_counter() - rendered) * 1000
        return image_packet(dict(id=request['id'], frame=request['frame'], render_mode=request.get('render_mode', 'mesh'), comparison=False,
                                 width=request['width'], height=request['height'], cache_hit=cache_hit,
                                 render_ms=round(render_ms, 2), encode_ms=round(encode_ms, 2),
                                 **({'stream': request['_stream']} if '_stream' in request else {})), jpeg)


class MeshPairRenderer:
    """Two separately reconstructed meshes, each with its own source trajectory."""
    def __init__(self, paths, fingerprints, correction, base_pose, target):
        if set(paths) != {'left', 'right'} or Path(paths['left']).resolve() == Path(paths['right']).resolve():
            raise ValueError('V1 and V3 need separate source meshes')
        self.sides = [MeshRenderer(paths[side], fingerprints[side], correction, base_pose, target, side)
                      for side in ('left', 'right')]
        self.info = dict(available=True, default_resolution=768,
                         triangle_count=sum(side.info['triangle_count'] for side in self.sides),
                         vertex_count=sum(side.info['vertex_count'] for side in self.sides),
                         left_label='V1 Mesh · 原始结果', right_label='V3 Mesh · 最新优化结果',
                         left=self.sides[0].info, right=self.sides[1].info,
                         description='左右分别从 V1 原始高斯、V3 最新优化高斯重建，使用各自运动；相机与时间同步。弹性带未包含，局部未知表面仍可能缺失。')

    def render_panels(self, request):
        return [side.render_pixels(request) for side in self.sides]


def load_clay_pair(paths, source_pair):
    """Only accept display repairs tied to the exact source geometry and motion."""
    import torch
    from utils.rigid_checkpoint import sha256_file
    first = source_pair.sides[0]
    pair = MeshPairRenderer(paths, dict(zip(('left', 'right'),
                           (side.fingerprints for side in source_pair.sides))),
                           first.correction, first.base_pose, first.target)
    for side, source in zip(pair.sides, source_pair.sides):
        record = side.info.get('clay_regularization') or {}
        if record.get('source_mesh_sha256') != sha256_file(source.source_path):
            raise ValueError('Clay repair belongs to a different source mesh')
        if side.motion is None or source.motion is None or any(
                not torch.equal(a, b) for a, b in zip(side.motion, source.motion)):
            raise ValueError('Clay repair changed the original group motion')
        if set(torch.unique(side.vertex_groups).cpu().tolist()) != set(torch.unique(source.vertex_groups).cpu().tolist()):
            raise ValueError('Clay repair introduced an unknown motion group')
        side.deformation = getattr(source, 'deformation', None)
        side.context = first.context
        side.edge_antialias = False
        side.info['edge_antialias'] = '2x2 supersampled coverage'
    for field in ('vertices', 'faces', 'colors', 'vertex_groups'):
        if torch.equal(getattr(pair.sides[0], field), getattr(pair.sides[1], field)):
            setattr(pair.sides[1], field, getattr(pair.sides[0], field))
    pair.info['surface_regularized'] = True
    if all(getattr(side, 'clay_normals', None) is not None for side in pair.sides):
        if torch.equal(pair.sides[0].clay_normals, pair.sides[1].clay_normals):
            pair.sides[1].clay_normals = pair.sides[0].clay_normals
        pair.info['feature_reconstructed'] = True
    return pair
