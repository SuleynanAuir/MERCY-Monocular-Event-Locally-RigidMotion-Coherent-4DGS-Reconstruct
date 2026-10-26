"""Optional V3 | V4 scene sharing the existing, verified V3 GPU assets."""
import copy
import io
import json
from pathlib import Path
import time

import numpy as np
from utils.experiment_export import capture_gaussians, render_pose, begin_panel, end_panel
from PIL import Image
import torch

from utils.rigid_checkpoint import sha256_file
from utils.scgs_v4_field import load_checkpoint
from utils.viewer_cache import EncodedFrameCache
from utils.viewer_state import orbit_pose, validate_request
from utils.render_sampling import resolve_rgb


class SCGSV4ComparisonRenderer:
    def __init__(self, path, cache, base_renderer):
        self.path, self.cache, self.base = Path(path), cache, base_renderer
        self.status = dict(ready=False, progress=0, message='等待 V3 加载')
        self.image_cache = EncodedFrameCache(max_bytes=96*1024**2)

    @torch.no_grad()
    def load(self):
        if not self.base.status.get('ready') or not hasattr(self.base, 'verified_v3'):
            raise ValueError('V4 requires the already verified current V3')
        self.status = dict(ready=False, progress=5, message='校验 V4 辅助运动场')
        settings = json.loads(self.path.read_text())
        if settings.get('format') != 'scgs_v4_comparison_v1':
            raise ValueError('Invalid V4 comparison manifest')
        paths = {key: (Path(settings[key]) if Path(settings[key]).is_absolute()
                       else self.path.parent/settings[key]) for key in ('checkpoint', 'audit')}
        for key, path in paths.items():
            if sha256_file(path) != settings[key+'_sha256']:
                raise ValueError(f'V4 {key} checksum mismatch')
        self.field, provenance = load_checkpoint(paths['checkpoint'])
        expected = self.base.verified_v3
        if (str(Path(provenance['baseline']).resolve()) != expected['baseline']
                or str(Path(provenance['candidate']).resolve()) != expected['candidate']
                or provenance['baseline_artifact_sha256'] != expected['artifact_sha256']):
            raise ValueError('V4 is bound to a different V3 base')
        audit = json.loads(paths['audit'].read_text())
        frames = self.base.info['frame_count']
        if (self.field.settings['frame_count'] != frames or audit['frame_count'] != frames
                or len(audit['intervals']) != frames-1
                or [r['interval'] for r in audit['intervals']] != list(range(frames-1))
                or int(self.field.rows.max()) >= len(self.base.side_geometry[1][0])):
            raise ValueError('V4 timeline or source rows do not match V3')
        self.info = copy.deepcopy(self.base.info)
        self.info.update(name='Lego · V3 / V4 微运动实验', left_label='V3 · Blender 主运动',
                         right_label='V4 · SC-GS 辅助场（实验）', geometry_repaired=False,
                         left_gaussian_count=self.base.info['gaussian_count'],
                         motion_label='左右同帧 · 主运动相同 · 右侧增加微小形变',
                         mesh={'available': False}, hybrid={'available': False},
                         scgs_v4={'checkpoint_sha256': settings['checkpoint_sha256'],
                                  'baseline_motion_frozen': True, 'event_only_auxiliary_training': True,
                                  'control_nodes': len(self.field.centers)},
                         event_validation={'optimized': provenance.get('best_step', 0) > 0, 'approved': audit['approved_for_export'],
                                           'preview_only': True, 'summary': audit['summary']['all'],
                                           'validation_summary': audit['summary']['validation_events'],
                                           'failed_intervals': audit['failed_intervals'],
                                           'reference_weak_intervals': audit.get('reference_self_check_failed_intervals', []),
                                           'interval_checks': [{'passed': r['passed'], 'reasons': r['reasons'],
                                                                'reference_weak': not r['reference_self_check_passed']}
                                                               for r in audit['intervals']]})
        if provenance.get('field_kind') in ('joint-trajectory-v4', 'local-joint-mlp-v4'):
            geometry = provenance['final_geometry']
            self.info.update(name='Lego · V3 / V4 关节防抖实验',
                             right_label='V4 · 关节约束＋时间平滑',
                             motion_label='左右同帧 · Blender 主运动保留 · 小幅关节修正',
                             description='保留 V3 主运动和刚性零件；辅助关节修正接受时间平滑、事件、连接和运动幅度检查。实验效果以指标和同帧对比为准。',
                             trajectory_stability=geometry)
            self.info['scgs_v4'].update(method='constrained_joint_residual', mlp_trained=False,
                                         control_nodes=len(self.field.parents),
                                         event_budget_fraction=provenance['event_budget_fraction'])
            if provenance['field_kind'] == 'local-joint-mlp-v4':
                self.info.update(name='Lego · V3 / V4 局部 MLP',
                                 right_label='V4 · 局部关节 MLP',
                                 motion_label='左右同帧 · 主传动保持 V3 · 局部分支小幅补偿',
                                 description='MLP 仅作用于局部关节分支；主传动、支架、外观和零件刚性保留。事件监督与关节、幅度、相位约束共同筛选。')
                self.info['scgs_v4'].update(method='local_joint_mlp',mlp_trained=True,
                                             control_nodes=len(provenance.get('enabled_auxiliary_joints',self.field.active_groups)),
                                             active_joint_groups=provenance.get('enabled_auxiliary_joints',self.field.active_groups),
                                             parameters=provenance['parameters'])
                if settings.get('local_metrics'):
                    from utils.local_motion_metrics import load_viewer_metrics
                    local_path=Path(settings['local_metrics'])
                    if not local_path.is_absolute():local_path=self.path.parent/local_path
                    self.info['local_metrics']=load_viewer_metrics(local_path,settings['local_metrics_sha256'],
                        settings['checkpoint_sha256'],provenance['enabled_auxiliary_joints'],frames)
        for frame in range(frames):
            self.render(validate_request(dict(frame=frame, width=640, height=640, quality=95,
                                              fovy=self.info['fovy'], background='black'), frames))
            if frame % 50 == 0:
                self.status = dict(ready=False, progress=10+int(89*frame/frames), message='缓存 V3 / V4 同帧对比')
        self.status = dict(ready=True, progress=100, scene=self.info)

    @torch.no_grad()
    def render(self, request):
        from gaussian_renderer import render
        from scripts.rendering.render_ply import make_camera
        if request.get('render_mode', 'gaussian') != 'gaussian':
            raise ValueError('This V4 field is evaluated on Gaussian geometry')
        jpeg = None if request.get('_raw') else self.image_cache.get(request)
        if jpeg is not None:
            return self.base.packet(request, jpeg, 0., 0., cache_hit=True)
        started = time.perf_counter()
        pose = render_pose(self.base.base_pose, self.base.target, request)
        supersample = self.base.side_sampling[1]['render_supersample']
        camera = make_camera(pose, request['width']*supersample, request['height']*supersample,
                             np.deg2rad(request['fovy']))
        frame, panels = request['frame'], []
        try:
            for auxiliary in (False, True):
                panel_started = begin_panel(request)
                # Both panels begin at exactly the same corrected V3 pose.
                self.base.apply_side_pose(1, frame)
                model = self.base.model
                if auxiliary:
                    group_q = self.base.correction.group_transforms(frame)[0]
                    model._xyz, model._rotation = self.field.deform(model._xyz, model._rotation, group_q, frame)
                    model._scaling = self.field.deform_scaling(model._scaling, frame)
                output = render(camera, model, self.base.pipe, self.base.backgrounds[request['background']], 0., 0., 0.)
                rgb = resolve_rgb(output['render'], supersample)
                capture_gaussians(request, model, output.get('alpha'))
                panels.append(rgb.clamp(0, 1).permute(1, 2, 0).mul(255).byte().contiguous().cpu().numpy())
                end_panel(request, panel_started)
        finally:
            # Shared GPU model must be restored before another scene renders.
            self.base.apply_side_pose(1, frame)
        if request.get('_raw'):
            return {'panels': panels}
        rendered = time.perf_counter()
        stream = io.BytesIO()
        Image.fromarray(np.concatenate(panels, 1)).save(stream, format='JPEG', quality=request['quality'], subsampling=0)
        jpeg = stream.getvalue()
        self.image_cache.put(request, jpeg)
        return self.base.packet(request, jpeg, (rendered-started)*1000, (time.perf_counter()-rendered)*1000)
