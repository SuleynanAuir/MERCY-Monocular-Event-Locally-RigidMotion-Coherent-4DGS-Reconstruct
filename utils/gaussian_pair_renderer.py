"""Synchronized supplied Gaussian versions with independently baked source motion.

Unlike the Lego event-optimized renderer, this renderer makes no event-training
claim. Both sequences must share the exact camera and animation timestamps.
"""
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

from utils.viewer_cache import EncodedFrameCache
from utils.viewer_state import image_packet, orbit_pose


def validate_pair(left, right):
    for key in ('frame_count', 'fps', 'preview_camera'):
        if key not in left or left[key] != right.get(key):
            raise ValueError(f'Gaussian pair differs in {key}')
    if ([f['time_seconds'] for f in left['frames']]
            != [f['time_seconds'] for f in right['frames']]):
        raise ValueError('Gaussian pair timestamps differ')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_detail_normals(path, sequence_sha, canonical_sha, count):
    with np.load(path, allow_pickle=False) as archive:
        normal = archive['local_normals']
        meta = json.loads(str(archive['metadata_json']))
    if (meta.get('format') != 'lamp_detail_normals_v1'
            or meta.get('sequence_sha256') != sequence_sha
            or meta.get('canonical_sha256') != canonical_sha
            or meta.get('normal_count') != count
            or normal.shape != (count, 3) or not np.isfinite(normal).all()
            or np.any(np.abs(np.linalg.norm(normal, axis=1)-1) > 1e-3)):
        raise ValueError('Detail normal field does not match the canonical sequence')
    return np.ascontiguousarray(normal, dtype=np.float32)


class GaussianPairRenderer:
    def __init__(self, path, cache='cpu'):
        self.path, self.cache = Path(path), cache
        self.status = dict(ready=False, progress=0, message='正在加载物件')

    def load(self):
        import torch
        from viewer import Renderer
        from utils.gs_clay import canonical_coordinates, gs_clay_info
        from utils.gs_pseudocolor import pseudocolor_info, pseudocolor_materials

        config = json.loads(self.path.read_text())
        if config.get('format') != 'gaussian_pair_v1':
            raise ValueError('Unsupported Gaussian pair manifest')
        paths, metadata = [], []
        for name in ('left', 'right'):
            path = (self.path.parent / config[name]['path']).resolve()
            if file_sha(path) != config[name]['sha256']:
                raise ValueError('Gaussian pair sequence checksum mismatch')
            with np.load(path, allow_pickle=False) as archive:
                metadata.append(json.loads(str(archive['metadata_json'].item())))
            paths.append(path)
        validate_pair(*metadata)
        self.supersamples = [config[name].get('render_supersample', 1) for name in ('left', 'right')]
        if any(type(value) is not int or value not in (1, 2) for value in self.supersamples):
            raise ValueError('Unsupported pair supersampling factor')
        self.sides = []
        for i, path in enumerate(paths):
            side = Renderer(path, self.cache)
            def progress(value, message, i=i):
                self.status = dict(ready=False, progress=int((i * 100 + value) / 2), message=message)
            side.progress = progress
            side.load()
            self.sides.append(side)
        with torch.no_grad():
            rests = [side.model._xyz for side in self.sides]
            anchors = rests
            if config.get('material_reference'):
                entry = config['material_reference']
                anchor_path = (self.path.parent/entry['path']).resolve()
                if file_sha(anchor_path) != entry['sha256']:
                    raise ValueError('Material reference checksum mismatch')
                with np.load(anchor_path, allow_pickle=False) as archive:
                    vertices = archive['base_vertices']
                    anchor = torch.from_numpy(np.column_stack([vertices[k] for k in ('x', 'y', 'z')]).copy()).cuda()
                anchors = [rests[0], anchor]
            _, bounds = canonical_coordinates(torch.cat(anchors))
            self.coordinates = [canonical_coordinates(x, bounds)[0] for x in rests]
            if anchors is rests:
                self.colors = pseudocolor_materials(rests)
            else:
                # Freeze the original shared material chart so optimizing V3
                # cannot recolor the V1 baseline in either display mode.
                from utils.gs_pseudocolor import palette
                values = [x @ x.new_tensor([.65, .20, .15]) for x in anchors]
                low = torch.stack([v.min() for v in values]).min()
                high = torch.stack([v.max() for v in values]).max()
                self.colors = [palette((x @ x.new_tensor([.65, .20, .15])-low)/(high-low).clamp_min(1e-6)).contiguous() for x in rests]
        self.info = dict(self.sides[1].info, name=config['name'], comparison=True,
            comparison_modes=['gaussian', 'clay', 'pseudocolor'],
            left_label=config['left']['label'], right_label=config['right']['label'],
            description=config['description'], default_background='light',
            surface_refinement=config.get('surface_refinement'), render_supersamples=self.supersamples,
            gaussian_counts=[s.info['gaussian_count'] for s in self.sides],
            clay=gs_clay_info(), pseudocolor=pseudocolor_info())
        self.detail_normals = None
        if config.get('detail_finish'):
            from utils.lamp_detail import height_weights
            entry = config['detail_finish']
            path = (self.path.parent/entry['path']).resolve()
            if entry.get('style') != 'normal_detail_v1' or file_sha(path) != entry['sha256']:
                raise ValueError('Unsupported or modified detail finish asset')
            self.detail_normals = torch.from_numpy(load_detail_normals(path,
                config['right']['sha256'], metadata[1]['base_vertices_sha256'],
                self.info['gaussian_count'])).cuda()
            self.detail_weights = height_weights(self.sides[1].model._xyz, entry.get('preserve_base'))
            self.info['detail_finish'] = dict(style=entry['style'], side='right',
                preserve_base=bool(entry.get('preserve_base')), depth_consistency_guard=True,
                description='V3 表面法线与材质展示增强；不代表重建精度或训练指标提升。')
            for mode, label in (('clay', '白模'), ('pseudocolor', '伪彩色')):
                self.info[mode]['right_label'] = f'V3 · {label} · 细节增强'
                self.info[mode]['description'] += ' V3 额外使用法线细节、接触阴影和柔和高光，仅为展示增强。'
        self.frames = EncodedFrameCache(max_bytes=256 * 1024**2, max_entries=1800)
        self.status = dict(ready=True, progress=100, message='就绪', scene=self.info)

    def render(self, request):
        import torch
        from PIL import Image
        from gaussian_renderer import render
        from render_ply import make_camera
        from utils.gs_clay import gray_gaussians
        from utils.gs_pseudocolor import color_gaussians

        mode = request['render_mode']
        if mode not in self.info['comparison_modes']:
            raise ValueError('This Gaussian pair does not provide the requested mode')
        start = time.perf_counter()
        jpeg = self.frames.get(request)
        hit = jpeg is not None
        render_ms = encode_ms = 0.
        if jpeg is None:
            panels = []
            with torch.no_grad():
                for i, side in enumerate(self.sides):
                    factor = self.supersamples[i]
                    def resolve(value):
                        return value if factor == 1 else torch.nn.functional.avg_pool2d(value, factor)
                    pose = side.poses[request['frame']].to('cuda')
                    if side.dynamic_indices is None:
                        side.model._xyz = pose[:, :3].contiguous()
                        side.model._rotation = pose[:, 3:7].contiguous()
                    else:
                        side.model._xyz[side.dynamic_indices] = pose[:, :3]
                        side.model._rotation[side.dynamic_indices] = pose[:, 3:7]
                    camera_pose = orbit_pose(side.base_pose, side.target, request)
                    camera = make_camera(camera_pose,
                        request['width']*factor, request['height']*factor, np.deg2rad(request['fovy']))
                    background = side.backgrounds[request['background']]
                    output = render(camera, side.model, side.pipe,
                        background if mode == 'gaussian' else side.backgrounds['black'], 0., 0., 0.)
                    rgb = output['render'] if factor == 1 else resolve(output['render'].clamp(0, 1))
                    depth, alpha = resolve(output['depth']), resolve(output['alpha'])
                    if mode == 'pseudocolor':
                        material = render(camera, side.model, side.pipe, side.backgrounds['black'],
                            0., 0., 0., override_color=self.colors[i])['render']
                        rgb = color_gaussians(rgb, depth, alpha, resolve(material),
                                             request['fovy'], background)
                    elif mode == 'clay':
                        coordinates = None
                        if request['clay_style'] == 'wire':
                            coordinates = render(camera, side.model, side.pipe, side.backgrounds['black'],
                                0., 0., 0., override_color=self.coordinates[i])['render']
                        rgb = gray_gaussians(rgb, depth, alpha,
                            request['fovy'], background, coordinates=None if coordinates is None else resolve(coordinates))
                    if i == 1 and self.detail_normals is not None:
                        from utils.event_refinement import qrotate
                        from utils.lamp_detail import detail_finish
                        normal = qrotate(side.model.get_rotation, self.detail_normals)
                        rotation = torch.as_tensor(camera_pose[:3, :3], dtype=normal.dtype, device=normal.device)
                        normal = normal @ rotation
                        position = (side.model._xyz-camera.camera_center) @ rotation
                        normal = torch.where((normal*position).sum(1, keepdim=True) > 0, -normal, normal)
                        guide = render(camera, side.model, side.pipe, side.backgrounds['black'],
                            0., 0., 0., override_color=(normal*self.detail_weights*.5+.5).contiguous())['render']
                        rgb = detail_finish(rgb, resolve(guide), depth, alpha,
                                            request['fovy'], background, mode)
                    panels.append((rgb.clamp(0, 1).permute(1, 2, 0)*255).byte().contiguous().cpu().numpy())
            rendered = time.perf_counter()
            stream = io.BytesIO()
            Image.fromarray(np.concatenate(panels, axis=1)).save(stream, 'JPEG', quality=request['quality'], subsampling=0)
            jpeg = stream.getvalue()
            self.frames.put(request, jpeg)
            render_ms, encode_ms = (rendered-start)*1000, (time.perf_counter()-rendered)*1000
        return image_packet(dict(id=request['id'], frame=request['frame'], comparison=True,
            left_frame=request['frame'], right_frame=request['frame'], panel_width=request['width'],
            width=request['width']*2, height=request['height'], render_mode=mode,
            **({'stream': request['_stream']} if '_stream' in request else {}),
            cache_hit=hit, render_ms=round(render_ms, 2), encode_ms=round(encode_ms, 2)), jpeg)
