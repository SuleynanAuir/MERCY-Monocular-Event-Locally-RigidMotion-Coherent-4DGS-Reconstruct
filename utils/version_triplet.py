"""Insert an actual archived V2 into the existing V1/V3 renderer."""
import io
import json
from pathlib import Path
import time

import numpy as np

from utils.experiment_export import render_pose, sha256
from utils.gaussian_pair_renderer import GaussianPairRenderer
from utils.viewer_cache import EncodedFrameCache
from utils.viewer_state import image_packet


def validate_stage(base, stage, base_pose, stage_pose):
    if (base['frame_count'] != stage['frame_count'] or abs(base['fps']-stage['fps']) > 1e-8
            or not np.allclose(base['times_seconds'], stage['times_seconds'], atol=1e-8, rtol=0)
            or abs(base['fovy']-stage['fovy']) > 1e-5
            or not np.allclose(base_pose, stage_pose, atol=1e-6, rtol=0)):
        raise ValueError('V2 must share the existing world camera and exact timeline')


class CaptureRoute:
    """Capture in render order, persist using the displayed version index."""
    def __init__(self, capture, indices):
        self.capture, self.indices, self.index = capture, indices, 0

    def point_ids(self, count, groups=None):
        return self.capture.point_ids(count, groups, side=self.indices[self.index])

    def __call__(self, state, alpha):
        self.capture(state, alpha, side=self.indices[self.index])
        self.index += 1


class SingleStageRenderer(GaussianPairRenderer):
    """Reuse the established native/clay/color renderer for one real sequence."""
    def load_stage(self, base, progress):
        import torch
        from viewer import Renderer
        from utils.gs_clay import canonical_coordinates
        from utils.gs_pseudocolor import palette
        settings = json.loads(self.path.read_text())
        if settings.get('format') != 'gaussian_stage_v1': raise ValueError('Invalid V2 stage manifest')
        path = (self.path.parent/settings['path']).resolve()
        if sha256(path) != settings['sha256']: raise ValueError('V2 sequence checksum mismatch')
        side = Renderer(path, self.cache)
        side.progress = progress
        side.load()
        self.sides, self.supersamples, self.detail_normals = [side], [1], None
        self.info = dict(side.info, comparison_modes=['gaussian', 'clay', 'pseudocolor'])
        self.frames = EncodedFrameCache(max_bytes=0, max_entries=0)
        anchors = ([geometry[0] for geometry in base.side_geometry]
                   if hasattr(base, 'side_geometry') else [s.model._xyz for s in base.sides])
        with torch.no_grad():
            _, bounds = canonical_coordinates(torch.cat(anchors))
            self.coordinates = [canonical_coordinates(side.model._xyz, bounds)[0]]
            values = [xyz @ xyz.new_tensor([.65, .20, .15]) for xyz in anchors]
            low, high = torch.stack([v.min() for v in values]).min(), torch.stack([v.max() for v in values]).max()
            value = side.model._xyz @ side.model._xyz.new_tensor([.65, .20, .15])
            self.colors = [palette((value-low)/(high-low).clamp_min(1e-6)).contiguous()]
        self.label = settings['label']
        self.provenance = dict(manifest_sha256=sha256(self.path), sequence_sha256=settings['sha256'],
                               branch=settings.get('branch'), label=self.label,
                               parents=settings.get('parents', {}))


class VersionTripletRenderer:
    def __init__(self, base, middle_path, cache='cpu'):
        self.base, self.path, self.cache = base, base.path, cache
        self.middle_path = Path(middle_path)
        self.source_paths = [self.middle_path]
        self.status = dict(ready=False, progress=0, message='正在加载 V1 / V2 / V3')
        self.frames = EncodedFrameCache(max_bytes=128*1024**2, max_entries=1200)

    def load(self):
        started = time.perf_counter()
        # The composite owns its cache. Avoid precomputing unused pair JPEGs.
        self.base.prewarm = False
        original_progress = getattr(self.base, 'progress', lambda value, message: None)
        def base_progress(value, message):
            original_progress(value, message)
            self.status = dict(ready=False, progress=int(value*.65), message=message)
        self.base.progress = base_progress
        self.base.load()
        self.middle = SingleStageRenderer(self.middle_path, self.cache)
        def middle_progress(value, message):
            self.status = dict(ready=False, progress=65+int(value*.34), message='V2 · '+message)
        self.middle.load_stage(self.base, middle_progress)
        camera_source = self.base if hasattr(self.base, 'base_pose') else self.base.sides[0]
        self.base_pose, self.target = camera_source.base_pose, camera_source.target
        validate_stage(self.base.info, self.middle.info, self.base_pose, self.middle.sides[0].base_pose)
        pair = [self.base.info['left_label'], self.base.info['right_label']]
        labels = [pair[0], self.middle.label, pair[1]]
        common = self.middle.info['comparison_modes']
        modes = {mode: labels if mode in common else pair for mode in ('gaussian', 'clay', 'pseudocolor', 'mesh', 'hybrid')}
        settings = json.loads(self.middle_path.read_text())
        self.info = dict(self.base.info, name=self.base.info['name'].split(' · ')[0]+' · V1 / V2 / V3', panel_count=3,
                         export_sample_step=2,
                         version_labels=labels, mode_version_labels=modes,
                         cache_gib=self.base.info['cache_gib']+self.middle.info['cache_gib'],
                         prewarmed_frames=0, image_cache_mib=128,
                         middle_version=self.middle.provenance,
                         gaussian_counts=[self.base.sides[0].info['gaussian_count'], self.middle.info['gaussian_count'], self.base.sides[1].info['gaussian_count']] if hasattr(self.base, 'sides') else self.base.info.get('gaussian_counts'),
                         description=(self.base.info.get('description') or '')+' '+settings.get('description', 'V2 为已存观测校正分支；三列使用同相机、同时间，不表示严格连续的训练链。'))
        if 'source_to_display' in settings:
            self.info['source_to_display'] = settings['source_to_display']
        for mode in ('clay', 'pseudocolor', 'mesh', 'hybrid'):
            self.info[mode] = dict(self.base.info.get(mode, {}), prewarmed_frames=0)
            if mode not in common and self.info[mode].get('available'):
                self.info[mode]['description'] += ' 此模式仅有 V1 / V3；V2 尚无对应资产。'
        self.info['asset_identity'] = dict(base=self.base.info.get('asset_identity'),
                                          base_manifest_sha256=sha256(self.path), middle=self.middle.provenance,
                                          renderer_sha256=sha256(Path(__file__)))
        settings = json.loads(self.middle_path.read_text())
        self.motion_evidence = {}
        self.compact_sources = []
        for entry in settings.get('compact_assets', []):
            path = (self.middle_path.parent/entry['path']).resolve()
            if path.stat().st_size>1024**2 or sha256(path)!=entry['sha256']:
                raise ValueError('Compact parameter source mismatch')
            self.compact_sources.append(path)
        if settings.get('motion_evidence_path'):
            path = (self.middle_path.parent/settings['motion_evidence_path']).resolve()
            evidence = json.loads(path.read_text())
            if (evidence.get('format') != 'scgs_sampled_motion_evidence_v1'
                    or evidence['base_manifest_sha256'] != sha256(self.path)
                    or evidence['middle_sequence_sha256'] != settings['sha256']):
                raise ValueError('Motion evidence belongs to different models')
            for side, record in evidence['versions'].items():
                source = path.parent/record['path']
                if source.stat().st_size > 16*1024**2 or sha256(source) != record['sha256']:
                    raise ValueError('Invalid motion evidence checksum/size')
                with np.load(source, allow_pickle=False) as archive:
                    data = {key:archive[key] for key in archive.files}
                ids = data['point_ids']; times = data['times_seconds']
                if (len(ids)>1024 or len(np.unique(ids))!=len(ids) or ids.dtype.kind not in 'iu'
                        or ids.min()<0 or ids.max()>=record['point_count']
                        or data['points'].shape!=(self.info['frame_count'],len(ids),3)
                        or not np.isfinite(data['points']).all()
                        or not np.array_equal(data['frames'],np.arange(self.info['frame_count']))
                        or not np.allclose(times,self.info['times_seconds'],rtol=0,atol=1e-8)):
                    raise ValueError('Invalid motion evidence samples/times')
                self.motion_evidence[int(side)] = dict(data, point_count=record['point_count'],
                    frame_lookup={int(frame):i for i,frame in enumerate(data['frames'])})
            self.source_paths.append(path)
            self.info['motion_evidence'] = evidence
        self.load_seconds = time.perf_counter()-started
        self.status = dict(ready=True, progress=100, message='V1 / V2 / V3 就绪', scene=self.info)

    def render(self, request):
        mode = request['render_mode']
        if mode not in self.middle.info['comparison_modes']:
            return self.base.render(request)
        started = time.perf_counter()
        jpeg = None if request.get('_raw') else self.frames.get(request)
        hit, render_ms, encode_ms = jpeg is not None, 0., 0.
        if jpeg is None:
            pose = render_pose(self.base_pose, self.target, request)
            left_right = dict(request, _raw=True, _camera_pose=pose, _panel_timings=[])
            middle = dict(request, _raw=True, _camera_pose=pose, _panel_timings=[])
            if request.get('_capture'):
                left_right['_capture'] = CaptureRoute(request['_capture'], [0, 2])
                middle['_capture'] = CaptureRoute(request['_capture'], [1])
            pair = self.base.render(left_right)['panels']
            extra = self.middle.render(middle)['panels']
            if len(pair) != 2 or len(extra) != 1: raise ValueError('Missing version panel')
            panels = [pair[0], extra[0], pair[1]]
            timings = left_right['_panel_timings']
            request['_panel_timings'] = ([timings[0], *middle['_panel_timings'], timings[1]] if len(timings)==2 else [])
            if request.get('_raw'): return dict(panels=panels)
            rendered = time.perf_counter()
            from PIL import Image
            stream = io.BytesIO()
            Image.fromarray(np.concatenate(panels, axis=1)).save(stream, 'JPEG', quality=request['quality'], subsampling=0)
            jpeg = stream.getvalue(); self.frames.put(request, jpeg)
            render_ms, encode_ms = (rendered-started)*1000, (time.perf_counter()-rendered)*1000
        return image_packet(dict(id=request['id'], frame=request['frame'], comparison=True,
            panel_count=3, panel_frames=[request['frame']]*3, panel_width=request['width'],
            left_frame=request['frame'], right_frame=request['frame'],
            width=request['width']*3, height=request['height'], render_mode=mode,
            **({'stream': request['_stream']} if '_stream' in request else {}),
            cache_hit=hit, render_ms=round(render_ms, 2), encode_ms=round(encode_ms, 2)), jpeg)
