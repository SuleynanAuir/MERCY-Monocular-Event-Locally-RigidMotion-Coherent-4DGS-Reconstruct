"""Fan's native GS view plus synchronized before/after Mesh and Mesh+GS panels."""
import io
import json
import logging
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from utils.viewer_cache import EncodedFrameCache
from utils.viewer_state import image_packet, orbit_pose, validate_request


class FanRenderer:
    def __init__(self, path, cache):
        self.path, self.cache = Path(path), cache
        self.status = dict(ready=False, progress=0, message='加载风扇 Mesh / GS 场景')
        self.caches = {mode: EncodedFrameCache(max_bytes=192*1024**2) for mode in ('mesh', 'hybrid')}

    def progress(self, percent, message):
        self.status = dict(ready=False, progress=percent, message=message)
        logging.getLogger('gaussian-viewer').info('%s (%s%%)', message, percent)

    def load(self, prewarm=True):
        import torch
        from viewer import Renderer
        from utils.fan_mesh import validate_source_pair
        from utils.rigid_checkpoint import sha256_file
        from utils.viewer_mesh import MeshPairRenderer
        from utils.mesh_gaussian_binding import MeshGaussianBinding
        config = json.loads(self.path.read_text())
        if config.get('format') != 'fan_mesh_comparison_v1':
            raise ValueError('Unsupported Fan manifest')
        paths = {}
        for key in ('sources', 'mesh_paths', 'hybrid_paths'):
            if set(config[key]) != {'left', 'right'}: raise ValueError('Fan requires two source versions')
            paths[key] = {name: (self.path.parent/value).resolve() for name, value in config[key].items()}
        if paths['sources']['left'] == paths['sources']['right']:
            raise ValueError('Fan versions must be distinct')
        meta = []
        for name in ('left', 'right'):
            path = paths['sources'][name]
            if config['source_fingerprints'][name] != {'sequence_sha256': sha256_file(path)}:
                raise ValueError('Fan source changed after Mesh reconstruction')
            with np.load(path, allow_pickle=False) as archive:
                meta.append(json.loads(str(archive['metadata_json'])))
        validate_source_pair(*meta)
        self.sides = []
        for index, name in enumerate(('left', 'right')):
            self.progress(10+index*20, f'加载风扇 {"V1" if index == 0 else "V3"} 高斯')
            side = Renderer(paths['sources'][name], 'cpu')
            side.load()
            self.sides.append(side)
        right = self.sides[1]
        frames = right.info['frame_count']
        self.progress(55, '加载两侧网格及独立部件运动')
        correction = SimpleNamespace(group_count=3, frame_count=frames)
        self.mesh = MeshPairRenderer(paths['mesh_paths'], config['source_fingerprints'],
                                     correction, right.base_pose, right.target)
        # Five million triangles per side need a >1 GiB topology hash for
        # analytic edge AA. Four subpixel coverage samples already smooth the
        # silhouette, without that temporary allocation or geometry reduction.
        for side in self.mesh.sides:
            side.edge_antialias = False
            side.info['edge_antialias'] = '2x2 supersampled coverage'
        # Panels render serially on the viewer's one GPU executor. Reuse the
        # raster scratch space and byte-identical geometry, not their motion.
        first, second = self.mesh.sides
        second.context = first.context
        for field in ('vertices', 'faces', 'colors', 'vertex_groups'):
            if torch.equal(getattr(first, field), getattr(second, field)):
                setattr(second, field, getattr(first, field))
        with torch.no_grad():
            self.native_rest = (right.model._xyz.clone(), right.model._rotation.clone())
            self.bindings = [MeshGaussianBinding(paths['hybrid_paths'][name], paths['mesh_paths'][name],
                              self.mesh.sides[i], side.model._xyz, side.model._rotation)
                             for i, (name, side) in enumerate(zip(('left', 'right'), self.sides))]
        if any(binding.info['unbound_gaussians'] for binding in self.bindings):
            raise ValueError('Fan hybrid requires every Gaussian to be bound')
        mesh_info = dict(self.mesh.info, default_resolution=1024,
            left_label='V1 Mesh · 修复前', right_label='V3 Mesh · 支架绑定修复后',
            description='左右为修复前 V1／修复后 V3：同一初始高斯表面重建，各自驱动部件运动。V3 支架保持静止；薄片与遮挡表面是高斯推得的近似网格。')
        self.info = dict(right.info, comparison=False, comparison_modes=['mesh', 'hybrid'],
            name='Fan · 风扇', mesh=mesh_info, cache='cpu',
            cache_gib=sum(side.info['cache_gib'] for side in self.sides),
            hybrid=dict(available=True, default_resolution=1024, default_quality='surface',
                left_label='V1 Mesh＋GS · 修复前', right_label='V3 Mesh＋GS · 支架绑定修复后',
                left=self.bindings[0].info, right=self.bindings[1].info,
                description='左侧修复前 V1，右侧修复后 V3。网格驱动绑定高斯运动，保留原始外观；轻度表面去噪。',
                quality_descriptions={
                    'surface': 'V1 修复前／V3 修复后。Mesh 驱动运动，GS 保留纹理；深度引导轻度去噪。',
                    'native': 'V1 修复前／V3 修复后。Mesh 驱动全部高斯，保留 GS 原生外观，不做表面柔化。',
                    'balanced': 'Mesh 驱动高斯，2 倍采样后缩小；用于效果对照。',
                    'detail': 'Mesh 驱动高斯，3 倍采样后缩小；渲染成本较高，用于效果对照。'}))
        if prewarm:
            for mode, quality in (('mesh', 'surface'), ('hybrid', 'surface'), ('hybrid', 'native')):
                for frame in range(frames):
                    self.render(validate_request(dict(render_mode=mode, hybrid_quality=quality, frame=frame,
                        width=1024, height=1024, quality=95, fovy=self.info['fovy'], background='black'), frames))
                    if (frame+1) % 25 == 0:
                        self.progress(60+int(35*(frame+1)/frames), f'预缓存风扇 {mode}/{quality} {frame+1}/{frames}')
            self.info['mesh']['prewarmed_frames'] = frames
            self.info['hybrid'].update(prewarmed_frames=frames, prewarmed_qualities=['surface', 'native'])
        self.status = dict(ready=True, progress=100, message='就绪', scene=self.info)

    def render(self, request):
        import torch
        mode = request.get('render_mode', 'gaussian')
        if mode == 'gaussian':
            # Switching out of hybrid must restore exact source static rows as
            # well; barycentric float rounding must not accumulate in GS mode.
            with torch.no_grad():
                self.sides[1].model._xyz.copy_(self.native_rest[0])
                self.sides[1].model._rotation.copy_(self.native_rest[1])
            return self.sides[1].render(request)
        if mode not in self.caches: raise ValueError('Unknown Fan display mode')
        from PIL import Image
        from gaussian_renderer import render
        from render_ply import make_camera
        from utils.hybrid_sampling import sampling_plan, resolve_hybrid_rgb, smooth_surface
        start = time.perf_counter()
        jpeg = self.caches[mode].get(request)
        hit = jpeg is not None
        render_ms = encode_ms = 0.
        if jpeg is None:
            with torch.no_grad():
                if mode == 'mesh':
                    panels = self.mesh.render_panels(request)
                else:
                    quality = request.get('hybrid_quality', 'surface')
                    width, height = sampling_plan(request['width'], request['height'], quality)
                    panels = []
                    for side, binding in zip(self.sides, self.bindings):
                        binding.apply(side.model, request['frame'])
                        camera = make_camera(orbit_pose(side.base_pose, side.target, request), width, height,
                                             np.deg2rad(request['fovy']))
                        output = render(camera, side.model, side.pipe, side.backgrounds[request['background']], 0., 0., 0.)
                        rgb = resolve_hybrid_rgb(output['render'], request['width'], request['height'], quality)
                        if quality == 'surface': rgb = smooth_surface(rgb, output['depth'], output['alpha'])
                        panels.append((rgb.permute(1, 2, 0)*255).byte().contiguous().cpu().numpy())
            rendered = time.perf_counter()
            stream = io.BytesIO()
            Image.fromarray(np.concatenate(panels, axis=1)).save(stream, format='JPEG', quality=request['quality'], subsampling=0)
            jpeg = stream.getvalue()
            self.caches[mode].put(request, jpeg)
            render_ms, encode_ms = (rendered-start)*1000, (time.perf_counter()-rendered)*1000
        return image_packet(dict(id=request['id'], frame=request['frame'], comparison=True,
            left_frame=request['frame'], right_frame=request['frame'], panel_width=request['width'],
            width=request['width']*2, height=request['height'], render_mode=mode,
            **({'hybrid_quality': request.get('hybrid_quality', 'surface')} if mode == 'hybrid' else {}),
            **({'stream': request['_stream']} if '_stream' in request else {}),
            cache_hit=hit, render_ms=round(render_ms, 2), encode_ms=round(encode_ms, 2)), jpeg)
