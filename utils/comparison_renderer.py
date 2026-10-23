"""Synchronized actual-Gaussian comparison with isolated rest geometry per side."""
from collections import OrderedDict
import io
import hashlib
import json
import logging
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from utils.event_refinement import correction_from_record
from utils.viewer_cache import EncodedFrameCache
from utils.viewer_state import BACKGROUNDS, STREAM_WINDOW, image_packet, orbit_pose, validate_request
from utils.render_sampling import render_sampling_contract, require_same_sampling, resolve_rgb
from utils.hybrid_sampling import DEFAULT_HYBRID_QUALITY, sampling_plan, resolve_hybrid_rgb, smooth_surface
from utils.multiview_completion import (FORMAT as MULTIVIEW_FORMAT, CompactCompletion,
                                        completion_contract, validate_report_shape,
                                        verify_multiview_completion)
from utils.gaussian_cleanup import verify_gaussian_cleanup, apply_gaussian_cleanup
from utils.static_surface_rebuild import verify_static_surface_rebuild, StaticSurfaceRebuild
from utils.color_residual import verify_color_residual, apply_color_residual
from utils.protected_surface_detail import verify_protected_surface_detail, ProtectedSurfaceDetail


def validate_comparison_sources(left_meta, right_meta, audit, left_sha256, geometry_report=None):
    """Require identical source unless an exact, narrowly scoped repair is recorded.

    verify_candidate has already hashed the right source against the audit. This
    helper adds the left-source link without weakening that existing validation.
    Tensor-level appearance and row-count checks are also required after loading.
    """
    if (left_meta['frame_count'] != right_meta['frame_count'] or left_meta['fps'] != right_meta['fps']
            or left_meta['camera'] != right_meta['camera']):
        raise ValueError('Comparison requires identical camera and timestamps')
    right_sha256 = audit.get('artifact_sha256', {}).get('source_ply')
    if not right_sha256:
        raise ValueError('Missing audited right source checksum')
    if geometry_report is None:
        if left_sha256 != right_sha256 or left_meta['gaussian_count'] != right_meta['gaussian_count']:
            raise ValueError('Comparison source mismatch requires a geometry_repair_report')
        return False
    require_same_sampling(right_meta, geometry_report)
    if geometry_report.get('appearance_refinement'):
        require_same_sampling(right_meta, geometry_report['appearance_refinement'])
    if geometry_report.get('format') == MULTIVIEW_FORMAT:
        validate_report_shape(geometry_report)
        contract = completion_contract(right_meta, audit)
        if (contract is None or geometry_report['source_sha256'] != left_sha256
                or geometry_report['completed_sha256'] != right_sha256
                or geometry_report['original_gaussian_count'] != left_meta['gaussian_count']
                or geometry_report['gaussian_count'] != right_meta['gaussian_count']
                or geometry_report['completed_source'] != right_meta['source']
                or geometry_report['extension'] != contract['extension']
                or geometry_report['extension_sha256'] != contract['extension_sha256']):
            raise ValueError('Invalid multiview completion report: identity or counts differ')
        return True
    if geometry_report.get('format') == 'lego_prior_surface_rebuild_v1':
        names = ('original_gaussian_count', 'gaussian_count', 'retained_gaussian_count',
                 'retired_gaussian_count', 'rebuilt_gaussians')
        if any(type(geometry_report.get(key)) is not int for key in names):
            raise ValueError('Invalid surface rebuild report: counts must be integers')
        retained, retired, rebuilt = [geometry_report[key] for key in
                                      ('retained_gaussian_count', 'retired_gaussian_count', 'rebuilt_gaussians')]
        if (geometry_report.get('source_sha256') != left_sha256
                or geometry_report.get('rebuilt_sha256') != right_sha256
                or geometry_report['original_gaussian_count'] != left_meta['gaussian_count']
                or geometry_report['gaussian_count'] != right_meta['gaussian_count']
                or not 0 < retained < left_meta['gaussian_count']
                or retired != left_meta['gaussian_count'] - retained
                or right_meta['gaussian_count'] != retained + rebuilt or not 1 <= rebuilt <= 200000
                or any(not isinstance(geometry_report.get(key), str) or not geometry_report[key]
                       for key in ('retained_indices_path', 'retained_indices_sha256', 'base_repaired_source',
                                   'base_repaired_sha256', 'frozen_retained_vertex_sha256'))):
            raise ValueError('Invalid surface rebuild report: source, counts or retained-row provenance differ')
        return True
    if geometry_report.get('format') == 'lego_part_completion_v1':
        added = right_meta['gaussian_count'] - left_meta['gaussian_count']
        if (geometry_report.get('source_sha256') != left_sha256
                or geometry_report.get('completed_sha256') != right_sha256
                or geometry_report.get('original_gaussian_count') != left_meta['gaussian_count']
                or geometry_report.get('gaussian_count') != right_meta['gaussian_count']
                or geometry_report.get('added_gaussians') != added or not 0 < added <= 100000
                or not geometry_report.get('base_repaired_source')
                or not geometry_report.get('base_repaired_sha256')
                or not geometry_report.get('frozen_prefix_vertex_sha256')):
            raise ValueError('Invalid geometry completion report: source, counts or append-only provenance differ')
        return True
    if left_meta['gaussian_count'] != right_meta['gaussian_count']:
        raise ValueError('Depth-only repair requires identical Gaussian counts')
    if (geometry_report.get('format') != 'lego_part_depth_repair_v1'
            or geometry_report.get('source_sha256') != left_sha256
            or geometry_report.get('repaired_sha256') != right_sha256
            or geometry_report.get('gaussian_count') != left_meta['gaussian_count']):
        raise ValueError('Invalid geometry repair report: format, source checksums or Gaussian count differ')
    return left_sha256 != right_sha256


def validate_repaired_gaussians(left, right, expected_count, completed_count=None):
    """Depth repair may only change xyz/scaling, never appearance/rotation/rows."""
    right_count = expected_count if completed_count is None else completed_count
    if completed_count is not None and not expected_count < completed_count <= expected_count + 100000:
        raise ValueError('Completion Gaussian count must append between 1 and 100000 rows')
    for name, model, count in (('left', left, expected_count), ('right', right, right_count)):
        if model._xyz.shape != (count, 3) or model._scaling.shape != (count, 3):
            raise ValueError(f'Geometry repair {name} Gaussian count or geometry shape differs')
        if not torch.isfinite(model._xyz).all() or not torch.isfinite(model._scaling).all():
            raise ValueError(f'Geometry repair {name} has nonfinite position or scaling')
    for attribute in ('_features_dc', '_features_rest', '_opacity', '_rotation'):
        left_value, right_value = getattr(left, attribute), getattr(right, attribute)
        if (left_value.shape[0] != expected_count
                or right_value.shape != (right_count, *left_value.shape[1:])):
            raise ValueError(f'Geometry repair protected attribute shape differs: {attribute}')
        # torch.equal deliberately permits matching +/-inf in opacity. NaNs do
        # not compare equal, and all fields are checked without tolerance.
        if not torch.equal(left_value, right_value[:expected_count]):
            raise ValueError(f'Geometry repair changed protected Gaussian attribute {attribute}')
        if attribute != '_opacity' and not torch.isfinite(left_value).all():
            raise ValueError(f'Geometry repair has nonfinite protected attribute {attribute}')
        if completed_count and not torch.isfinite(right_value[expected_count:]).all():
            raise ValueError('Completion has nonfinite appended attributes')
    if completed_count and torch.any(right._rotation[expected_count:].norm(dim=-1) < 1e-8):
        raise ValueError('Completion has invalid appended rotations')
    if completed_count and torch.count_nonzero(right._features_rest[expected_count:]).item() != 0:
        raise ValueError('Completion expects DC-only appended color')


def vertex_rows_sha256(vertices, indices=None):
    """Hash exact structured-row bytes without a full-model temporary byte copy."""
    digest = hashlib.sha256()
    count = len(vertices) if indices is None else len(indices)
    for start in range(0, count, 65536):
        rows = vertices[start:start + 65536] if indices is None else vertices[indices[start:start + 65536]]
        digest.update(np.ascontiguousarray(rows).view(np.uint8))
    return digest.hexdigest()


def verify_completion_prefix(report, completed_source):
    """Verify unchanged repaired rows before permitting appended surface points."""
    from plyfile import PlyData
    from utils.rigid_checkpoint import sha256_file
    base = Path(report['base_repaired_source'])
    if sha256_file(base) != report['base_repaired_sha256']:
        raise ValueError('Completion base source checksum mismatch')
    before = PlyData.read(base)['vertex'].data
    after = PlyData.read(completed_source)['vertex'].data
    count = report['original_gaussian_count']
    if len(before) != count or len(after) != report['gaussian_count'] or before.dtype != after.dtype:
        raise ValueError('Completion source count or vertex schema mismatch')
    old_hash = vertex_rows_sha256(before)
    prefix_hash = vertex_rows_sha256(after[:count])
    if old_hash != prefix_hash or prefix_hash != report['frozen_prefix_vertex_sha256']:
        raise ValueError('Completion modified the frozen repaired-source prefix')


def verify_rebuild_provenance(report, original_source, rebuilt_source):
    """Verify original/base retained rows and mapping, not only claimed counts."""
    from plyfile import PlyData
    from utils.rigid_checkpoint import sha256_file
    for path, key in ((original_source, 'source_sha256'), (rebuilt_source, 'rebuilt_sha256'),
                      (report['base_repaired_source'], 'base_repaired_sha256'),
                      (report['retained_indices_path'], 'retained_indices_sha256')):
        if sha256_file(path) != report[key]:
            raise ValueError(f'Rebuild provenance checksum mismatch: {key}')
    indices = np.load(report['retained_indices_path'], mmap_mode='r', allow_pickle=False)
    original_count, retained_count = report['original_gaussian_count'], report['retained_gaussian_count']
    if (indices.ndim != 1 or indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64))
            or len(indices) != retained_count or not 0 < retained_count < original_count
            or np.any(indices < 0) or np.any(indices >= original_count)
            or np.any(indices[1:] <= indices[:-1])
            or report['retired_gaussian_count'] != original_count - retained_count
            or not 1 <= report['rebuilt_gaussians'] <= 200000
            or report['gaussian_count'] != retained_count + report['rebuilt_gaussians']):
        raise ValueError('Invalid rebuild retained indices or complement counts')
    original = PlyData.read(original_source)['vertex'].data
    base = PlyData.read(report['base_repaired_source'])['vertex'].data
    rebuilt = PlyData.read(rebuilt_source)['vertex'].data
    if (len(original) != original_count or len(base) != original_count
            or len(rebuilt) != report['gaussian_count'] or original.dtype != base.dtype
            or original.dtype != rebuilt.dtype or original.dtype.hasobject):
        raise ValueError('Rebuild vertex schema or row counts mismatch')
    hashes = (vertex_rows_sha256(original, indices), vertex_rows_sha256(base, indices),
              vertex_rows_sha256(rebuilt[:retained_count]), report['frozen_retained_vertex_sha256'])
    if len(set(hashes)) != 1:
        raise ValueError('Rebuild modified frozen original/base retained vertex rows')
    return np.array(indices, copy=True)


def validate_rebuilt_gaussians(left, right, expected_count, retained_indices, seed_count):
    """A rebuild may replace moving rows, but every retained row is immutable."""
    if not 1 <= seed_count <= 200000:
        raise ValueError('Rebuild requires between 1 and 200000 new surface rows')
    indices = torch.as_tensor(retained_indices, device=left._xyz.device)
    if (indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64)
            or len(indices) == 0 or len(indices) >= expected_count
            or torch.any(indices < 0) or torch.any(indices >= expected_count)
            or torch.any(indices[1:] <= indices[:-1])):
        raise ValueError('Invalid strictly increasing retained original indices')
    indices = indices.long()
    retained_count = len(indices)
    right_count = retained_count + seed_count
    for attribute in ('_xyz', '_scaling', '_features_dc', '_features_rest', '_opacity', '_rotation'):
        before, after = getattr(left, attribute), getattr(right, attribute)
        if before.shape[0] != expected_count or after.shape != (right_count, *before.shape[1:]):
            raise ValueError(f'Rebuild attribute shape mismatch: {attribute}')
        frozen = before[indices]
        if not torch.equal(frozen, after[:retained_count]):
            raise ValueError(f'Rebuild modified frozen retained attribute {attribute}')
        if attribute != '_opacity' and not torch.isfinite(frozen).all():
            raise ValueError(f'Rebuild retained nonfinite attribute {attribute}')
        if not torch.isfinite(after[retained_count:]).all():
            raise ValueError(f'Rebuild has nonfinite new surface attribute {attribute}')
    if torch.any(right._rotation[retained_count:].norm(dim=-1) < 1e-8):
        raise ValueError('Rebuild has invalid new surface rotations')
    if torch.count_nonzero(right._features_rest[retained_count:]).item() != 0:
        raise ValueError('Rebuild expects DC-only new surface color')


def audited_source_fingerprints(audit):
    """Publish only the two verified fingerprints needed for live scene identity."""
    hashes = audit['artifact_sha256']
    return {'candidate_correction_sha256': hashes['correction'],
            'right_source_sha256': hashes['source_ply']}


class ComparisonRenderer:
    def __init__(self, path, cache):
        self.path, self.cache = Path(path), cache
        self.status = {'ready': False, 'progress': 0, 'message': '加载同步对比场景'}
        self.pose_cache = OrderedDict()
        self.cache_limit = 0 if cache == 'cpu' else 48
        self.image_cache = EncodedFrameCache()
        self.side_sampling = [render_sampling_contract({}), render_sampling_contract({})]
        self.completion = None
        self.deformation = None
        self.mesh = None
        self.mesh_comparison_cache = EncodedFrameCache(max_bytes=192 * 1024**2)
        # Both 400-frame white styles total about 235 MiB at 1024px/quality 95.
        # Keep both complete loops resident; a 192 MiB shared cache evicts the
        # first wire frames while warming the solid style and stalls playback.
        self.clay_comparison_cache = EncodedFrameCache(max_bytes=384 * 1024**2)
        self.pseudocolor_cache = EncodedFrameCache(max_bytes=256 * 1024**2)
        self.hybrid = None
        self.hybrid_comparison_cache = EncodedFrameCache(max_bytes=192 * 1024**2)

    def progress(self, value, message):
        self.status = {'ready': False, 'progress': value, 'message': message}
        logging.getLogger('gaussian-viewer').info('%s (%s%%)', message, value)

    @torch.no_grad()
    def load(self, prewarm=True):
        from scene import GaussianModel
        from scripts.events.export_lego_event_preview import verify_candidate
        from utils.rigid_checkpoint import sha256_file
        settings = json.loads(self.path.read_text())
        if settings.get('format') != 'lego_event_comparison_v1':
            raise ValueError('Unsupported comparison manifest')
        left, right, candidate = [Path(settings[k]) for k in ('baseline', 'right_baseline', 'candidate')]
        self.progress(5, '校验事件校正、初始模型及运动数据')
        right_meta, correction, audit = verify_candidate(right, candidate, allow_unapproved=True)
        self.verified_v3 = {'baseline': str(right.resolve()), 'candidate': str(candidate.resolve()),
                            'artifact_sha256': dict(audit['artifact_sha256'])}
        left_meta = json.loads((left / 'cache.json').read_text())
        self.side_sampling = [render_sampling_contract(meta) for meta in (left_meta, right_meta)]
        geometry_report = None
        if 'geometry_repair_report' in settings:
            report_path = Path(settings['geometry_repair_report'])
            if not report_path.is_absolute():
                report_path = self.path.parent / report_path
            geometry_report = json.loads(report_path.read_text())
        self.geometry_repaired = validate_comparison_sources(
            left_meta, right_meta, audit, sha256_file(left_meta['source']), geometry_report)
        self.poses, self.indices = [], []
        for folder, meta in ((left, left_meta), (right, right_meta)):
            poses = np.load(folder / 'poses.npy', mmap_mode='r', allow_pickle=False)
            indices = np.load(folder / 'dynamic_indices.npy', allow_pickle=False)
            if (poses.shape != (left_meta['frame_count'], len(indices), 7) or poses.dtype != np.float32
                    or np.any(indices < 0) or np.any(indices >= meta['gaussian_count'])
                    or np.any(indices[1:] <= indices[:-1])):
                raise ValueError('Invalid comparison motion arrays')
            self.poses.append(poses)
            self.indices.append(torch.as_tensor(indices.astype(np.int64), device='cuda'))
        self.correction = correction_from_record(correction, 'cuda')
        extension = verify_multiview_completion(right_meta, correction, audit, baseline=right)
        cleaned = verify_gaussian_cleanup(right_meta, correction, audit, baseline=right)
        rebuilt_static = verify_static_surface_rebuild(right_meta, correction, audit, baseline=right)
        static_surface = StaticSurfaceRebuild(rebuilt_static, 'cuda') if rebuilt_static is not None else None
        residual = verify_color_residual(right_meta, correction, audit, baseline=right)
        detail = verify_protected_surface_detail(right_meta, correction, audit, baseline=right)
        surface_detail = ProtectedSurfaceDetail(detail, 'cuda') if detail is not None else None
        if extension is not None:
            if geometry_report != extension['report']:
                raise ValueError('Comparison manifest differs from audited multiview completion report')
            self.completion = CompactCompletion(extension, 'cuda')
        self.rest = torch.from_numpy(self.poses[1][0].copy()).cuda()
        self.progress(65, '加载共享高斯；左右视图共享时间与相机')
        self.model = GaussianModel(3, with_motion_mask=False)
        self.model.load_ply(left_meta['source'])
        if len(self.model._xyz) != left_meta['gaussian_count']:
            raise ValueError('Left source and comparison Gaussian count differ')
        if torch.count_nonzero(self.model._features_rest).item() != 0:
            raise ValueError('Comparison expects DC-only color')
        self.base_xyz = self.model._xyz.detach().clone()
        self.base_rotation = self.model._rotation.detach().clone()
        self.base_scaling = self.model._scaling.detach().clone()
        self.side_geometry = [(self.base_xyz, self.base_rotation, self.base_scaling)]
        self.side_appearance = [(self.model._features_dc.detach(), self.model._opacity.detach(),
                                 torch.empty((left_meta['gaussian_count'], 0, 3), device='cuda'))]
        if self.geometry_repaired:
            repaired = GaussianModel(3, with_motion_mask=False)
            repaired.load_ply(right_meta['source'])
            completion = geometry_report['format'] == 'lego_part_completion_v1'
            rebuild = geometry_report['format'] == 'lego_prior_surface_rebuild_v1'
            multiview = geometry_report['format'] == MULTIVIEW_FORMAT
            if completion:
                verify_completion_prefix(geometry_report, right_meta['source'])
            if rebuild:
                retained = verify_rebuild_provenance(geometry_report, left_meta['source'], right_meta['source'])
                validate_rebuilt_gaussians(self.model, repaired, left_meta['gaussian_count'], retained,
                                           geometry_report['rebuilt_gaussians'])
            elif not multiview:
                validate_repaired_gaussians(self.model, repaired, left_meta['gaussian_count'],
                                            right_meta['gaussian_count'] if completion else None)
            apply_gaussian_cleanup(repaired, cleaned)
            apply_color_residual(repaired, residual)
            right_xyz, right_q = repaired._xyz.detach(), repaired._rotation.detach()
            if static_surface is not None:
                static_surface.apply_appearance(repaired)
                right_xyz, right_q = static_surface.append_geometry(right_xyz, right_q)
            if surface_detail is not None:
                surface_detail.apply_appearance(repaired)
                right_xyz, right_q = surface_detail.append_geometry(right_xyz, right_q)
            self.side_geometry.append((right_xyz.clone(), right_q.clone(), repaired._scaling.detach().clone()))
            self.side_appearance.append((repaired._features_dc.detach(), repaired._opacity.detach(),
                                         torch.empty((len(repaired._features_dc), 0, 3), device='cuda')))
            del repaired
        else:
            self.side_geometry.append(self.side_geometry[0])
            self.side_appearance.append(self.side_appearance[0])
        self.model._features_rest = torch.empty((left_meta['gaussian_count'], 0, 3), device='cuda')
        self.model.active_sh_degree = self.model.max_sh_degree = 0
        if self.cache_limit:
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            pose_bytes = max(len(indices) for indices in self.indices) * 7 * 4
            # A 30 FPS pass through 400 source frames needs 400 paired pose
            # entries. Keep those for the next loop if memory permits, without
            # duplicating both full 60 FPS trajectories on a 10 GB GPU.
            self.cache_limit = min(448, max(0, int((free - 1536 * 1024**2) // pose_bytes)))
        camera = left_meta['camera']
        self.base_pose, self.target = np.asarray(camera['pose']), np.asarray(camera['target'])
        self.pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        self.backgrounds = {key: torch.tensor(value, dtype=torch.float32, device='cuda') for key, value in BACKGROUNDS.items()}
        frames = left_meta['frame_count']
        self.info = {'name': 'Lego · 最原始 / 进一步优化', 'comparison': True,
                     **audited_source_fingerprints(audit),
                     'left_render_sampling': self.side_sampling[0],
                     'right_render_sampling': self.side_sampling[1],
                     'left_label': '最原始结果 · 未校正初始动画',
                     'right_label': settings.get('right_label', '经过事件校正 · 改进候选'),
                     'geometry_repaired': self.geometry_repaired,
                     **({'geometry_repair_scope': (
                         '右侧保留正面已校正高斯，使用已知 Blender 网格补充多视角刚性表面；未观测背面为几何与外观先验，不是单目事件证实的重建。新增点沿用同一硬连接运动，弹性带保持原始高斯。仍有局部缺口，尤其是底板与少数静态件，尚不代表任意视角完整。'
                         if geometry_report['format'] == MULTIVIEW_FORMAT else
                         '右侧依据已知几何先验与多个时刻的视频观测重建运动刚性部件，再经事件校正；静态件与弹性带的原始高斯保留，左侧为原始模型。'
                         if geometry_report['format'] == 'lego_prior_surface_rebuild_v1' else
                         '右侧采用深度修复，并补充后续显露表面的高斯；原有高斯保留，新增点绑定刚性部件；事件验收独立。'
                         if geometry_report['format'] == 'lego_part_completion_v1' else
                         '右侧采用部件深度修复高斯；仅位置和尺度可变，点数、颜色、透明度和初始旋转保持不变；事件验收独立。') +
                         (' 新部件再经 RGB、边缘及事件约束细节拟合；运动保持不变，事件验收结论独立。'
                          if geometry_report.get('appearance_refinement') else '')}
                        if self.geometry_repaired else {}),
                     **({'gaussian_cleanup': {'modified_gaussians': cleaned['report']['modified_gaussian_count'],
                          'attributes': cleaned['report']['attributes'], 'geometry_and_motion_unchanged': True,
                          'report_sha256': audit['artifact_sha256']['cleanup_report'],
                          'overlay_sha256': audit['artifact_sha256']['cleanup_overlay']}}
                        if cleaned is not None else {}),
                     'frame_count': frames, 'fps': left_meta['fps'], 'duration_seconds': frames / left_meta['fps'],
                     'gaussian_count': right_meta.get('rendered_gaussian_count', right_meta['gaussian_count']),
                     'source_gaussian_count': right_meta['gaussian_count'], 'left_gaussian_count': left_meta['gaussian_count'],
                     'dynamic_gaussian_count': len(self.indices[1]) + (self.completion.moving_count if self.completion is not None else 0),
                     'baked_dynamic_gaussian_count': len(self.indices[1]),
                     **({'multiview_completion': {'format': MULTIVIEW_FORMAT,
                          'unobserved_geometry_is_prior': True,
                          'added_gaussians': len(self.completion.rest),
                          'moving_added_gaussians': self.completion.moving_count,
                          'completion_report_sha256': audit['artifact_sha256']['completion_report'],
                          'completion_extension_sha256': audit['artifact_sha256']['completion_extension']}}
                        if self.completion is not None else {}),
                     'times_seconds': [i / left_meta['fps'] for i in range(frames)], 'rpm': [None] * frames,
                     'motion_label': '左右同帧 · 同相机 · 真实高斯渲染', 'default_stream_rate': 30,
                     'default_stream_quality': 95, 'default_still_quality': 95,
                     'default_resolution': 640, 'default_background': 'light', 'fovy': camera['fovy_degrees'],
                     'cache': 'gpu-lru' if self.cache_limit else 'cpu',
                     'cache_gib': self.cache_limit * max(len(v) for v in self.indices) * 28 / 1024**3,
                     'gpu': torch.cuda.get_device_name(), 'baked': False, 'streaming': True, 'stream_window': STREAM_WINDOW,
                     'render_cache': True, 'image_cache_mib': self.image_cache.max_bytes / 1024**2,
                     'buffer_seconds': 0.4,
                     'event_validation': {'optimized': True, 'approved': audit['approved_for_export'],
                                          'preview_only': True, 'summary': audit['summary']['all'],
                                          'validation_summary': audit['summary']['validation_events'],
                                          'reference_weak_intervals': audit.get('reference_self_check_failed_intervals', []),
                                          'interval_checks': [{'passed': row['passed'], 'reasons': row['reasons'],
                                                               'reference_weak': not row['reference_self_check_passed']}
                                                              for row in audit['intervals']],
                                          'failed_intervals': audit['failed_intervals']}}
        if cleaned is not None:
            self.info['geometry_repair_scope'] += (
                ' 已统一部分未观测静态高斯的部件颜色先验；几何、不透明度和运动不变，背面颜色仍非实测纹理。'
                if cleaned['report']['attributes'] == ['dc_gray'] else
                ' 已对部分原始静态高斯施加独立透明度清理层；位置、颜色、行序与运动不变。')
            self.info['geometry_repair_scope'] += (' 结果已重新执行整段事件检查；这不是二维图像去噪，'
                                                  '也不代表所有斑驳或缺口已消除。')
        if residual is not None:
            self.info['color_residual'] = {
                'report_sha256': audit['artifact_sha256']['color_residual_report'],
                'residual_sha256': audit['artifact_sha256']['color_residual'],
                'optimized_gaussians': residual['report']['optimized_count'],
                'maximum_linear_rgb_delta': residual['report']['maximum_delta']}
            self.info['geometry_repair_scope'] += (' 已加入全局与局部细节的有界高斯颜色残差优化，'
                '使用视频、多尺度梯度和训练事件约束；几何、透明度与运动不变。'
                '未观测面的完整性仍不作保证，事件未通过区间继续如实显示。')
        if detail is not None:
            self.info['protected_surface_detail'] = {
                'report_sha256': audit['artifact_sha256']['surface_detail_report'],
                'extension_sha256': audit['artifact_sha256']['surface_detail_extension'],
                'retired_gaussians': detail['report']['retired_gaussian_count'],
                'added_gaussians': detail['report']['added_gaussian_count'],
                'part_ids': detail['report']['allowed_part_ids']}
            self.info['geometry_repair_scope'] += (' 车顶及固定部件的局部表面加入致密薄层高斯，'
                '保留原始视角可见点，并检查替换后的表面缺口；运动与已有颜色优化保留。'
                '未观测的背面仍采用网格和颜色先验，不代表真实背面已由事件验证。')
        if rebuilt_static is not None:
            self.info['static_surface_rebuild'] = {
                'report_sha256': audit['artifact_sha256']['static_surface_report'],
                'extension_sha256': audit['artifact_sha256']['static_surface_extension'],
                'retired_gaussians': rebuilt_static['report']['retired_gaussian_count'],
                'added_gaussians': rebuilt_static['report']['added_gaussian_count'],
                'part_ids': rebuilt_static['report']['allowed_part_ids']}
            self.info['geometry_repair_scope'] += (' 驾驶舱车顶的5个固定零件另按真实网格表面重建高斯；'
                '可见面采用视频颜色，未观测面仍是先验。窗框、立柱和运动参数保留；结果另经完整事件复验。')
        from utils.gs_clay import gs_clay_info, canonical_coordinates
        bounds_xyz = torch.cat([geometry[0] for geometry in self.side_geometry])
        _, bounds = canonical_coordinates(bounds_xyz)
        self.clay_coordinates = [canonical_coordinates(geometry[0], bounds)[0] for geometry in self.side_geometry]
        del bounds_xyz
        self.info['clay'] = gs_clay_info()
        from utils.gs_pseudocolor import pseudocolor_materials, pseudocolor_info
        self.pseudocolor_materials = pseudocolor_materials([geometry[0] for geometry in self.side_geometry])
        self.info['pseudocolor'] = pseudocolor_info()
        self.info['mesh'] = {'available': False, 'description': '当前场景未生成高斯重建网格。'}
        self.info['hybrid'] = {'available': False, 'description': '当前场景未生成 Mesh＋Gaussian 绑定。'}
        if settings.get('deformation'):
            from utils.v3_deformation import load_v3_deformation
            deformation_path = Path(settings['deformation'])
            if not deformation_path.is_absolute():
                deformation_path = self.path.parent / deformation_path
            self.deformation, deformation_info = load_v3_deformation(deformation_path, self)
            self.info.update(deformation_info)
        if not prewarm:
            self.status = {'ready': True, 'progress': 100, 'scene': self.info}
            return
        if settings.get('mesh_paths'):
            try:
                from utils.viewer_mesh import MeshPairRenderer
                paths = {side: (Path(value) if Path(value).is_absolute() else self.path.parent / value)
                         for side, value in settings['mesh_paths'].items()}
                fingerprints = {'left': dict(source_sha256=sha256_file(left_meta['source']),
                                              poses_sha256=sha256_file(left / 'poses.npy'),
                                              dynamic_indices_sha256=sha256_file(left / 'dynamic_indices.npy')),
                                'right': audited_source_fingerprints(audit)}
                self.mesh = MeshPairRenderer(paths, fingerprints, self.correction, self.base_pose, self.target)
                self.mesh.sides[1].deformation = self.deformation
                self.mesh.render_panels(validate_request({'render_mode': 'mesh', 'width': 512, 'height': 512,
                                                          'fovy': camera['fovy_degrees']}, frames))
                self.info['mesh'] = self.mesh.info
            except Exception:
                self.mesh = None
                self.info['mesh'] = {'available': False, 'description': 'Mesh 加载失败；高斯模式仍可使用。'}
                logging.getLogger('gaussian-viewer').exception('Optional Mesh failed to load')
        if self.mesh is not None and settings.get('hybrid_paths'):
            try:
                from utils.mesh_gaussian_binding import MeshGaussianBinding
                if set(settings['hybrid_paths']) != {'left', 'right'}:
                    raise ValueError('Hybrid comparison requires both source bindings')
                bindings = []
                for side, name in enumerate(('left', 'right')):
                    value = Path(settings['hybrid_paths'][name])
                    if not value.is_absolute(): value = self.path.parent / value
                    self.apply_side_pose(side, 0)
                    bindings.append(MeshGaussianBinding(value, paths[name], self.mesh.sides[side],
                                                         self.model._xyz, self.model._rotation))
                self.hybrid = bindings
                self.info['hybrid'] = dict(available=True, default_resolution=self.info['default_resolution'],
                    render_supersample=self.side_sampling[0]['render_supersample'],
                    default_quality=DEFAULT_HYBRID_QUALITY, max_render_side=3072,
                    reconstruction_filter='native_sampling_with_depth_alpha_guided_denoising',
                    quality_descriptions={
                        'surface': '保留 GS 原生外观，在表面内部轻度去噪，保护孔洞与轮廓。Mesh 绑定驱动运动；这是显示去噪，未修改源几何。',
                        'native': '采用与单独 GS 一致的采样，关闭表面柔化；Mesh 绑定仍驱动运动。这是混合渲染，未修改源几何。',
                        'balanced': '旧版均衡效果，仅供对照。高采样可能使细碎颗粒更明显，可切回表面柔化。',
                        'detail': '旧版高采样效果，仅供对照。本场景中颗粒更明显，建议切回表面柔化。'},
                    left_label='V1 · Mesh＋Gaussian', right_label='V3 · Mesh＋Gaussian',
                    left=bindings[0].info, right=bindings[1].info,
                    description='Mesh 绑定驱动运动，保留 GS 原生外观；表面柔化轻度抑制内部颗粒，并保护轮廓、孔洞边缘和透明区域。可切换 GS 原生外观对照。混合渲染，并非纯网格；这是显示去噪，未修复源模型几何。')
            except Exception:
                self.hybrid = None
                self.info['hybrid'] = {'available': False, 'description': '混合模式绑定加载失败；可使用 Gaussian 或纯 Mesh。'}
                logging.getLogger('gaussian-viewer').exception('Optional hybrid binding failed to load')
        if self.deformation is not None:
            for mode in ('mesh', 'hybrid', 'clay', 'pseudocolor'):
                if self.info[mode]['available']:
                    self.info[mode]['deformation_enabled'] = True
                    self.info[mode]['description'] += ' 右侧 V3 同步叠加局部 MLP 关节残差；主运动和刚性保持。'
                    for quality in self.info[mode].get('quality_descriptions', {}):
                        self.info[mode]['quality_descriptions'][quality] += ' V3 已启用局部 MLP 辅助运动。'
        self.progress(95, '预热双视图实时渲染')
        self.render(validate_request({'width': 512, 'height': 512, 'fovy': camera['fovy_degrees']}, frames))
        # Render every source frame of the initial view once, before autoplay.
        # This is an exact JPEG cache of 3D Gaussian renders, not a substituted
        # input video. New camera/resolution/background values render afresh.
        for index in range(frames):
            self.render(validate_request({'frame': index, 'width': self.info['default_resolution'],
                                          'height': self.info['default_resolution'],
                                          'quality': self.info['default_stream_quality'],
                                          'background': self.info['default_background'],
                                          'fovy': camera['fovy_degrees']}, frames))
            if (index + 1) % 40 == 0 or index + 1 == frames:
                self.progress(95 + int(4 * (index + 1) / frames), f'预缓存双视图 {index + 1}/{frames} 帧')
        self.info['prewarmed_frames'] = frames
        for mode, clay_style in (('clay', 'wire'), ('clay', 'solid'), ('pseudocolor', 'solid'), ('mesh', 'wire')):
            if self.info[mode]['available']:
                resolution = self.info[mode]['default_resolution']
                for index in range(frames):
                    self.render(validate_request({'render_mode': mode, 'clay_style': clay_style, 'frame': index,
                                                  'width': resolution, 'height': resolution,
                                                  'quality': self.info['default_stream_quality'],
                                                  'background': self.info['default_background'],
                                                  'fovy': camera['fovy_degrees']}, frames))
                    if (index + 1) % 40 == 0 or index + 1 == frames:
                        self.progress(99, f'预缓存 V1 / V3 {mode} 对比 {index + 1}/{frames} 帧')
                self.info[mode]['prewarmed_frames'] = frames
        if self.hybrid is not None:
            resolution = self.info['hybrid']['default_resolution']
            for quality in ('surface', 'native'):
                for index in range(frames):
                    self.render(validate_request({'render_mode': 'hybrid', 'hybrid_quality': quality, 'frame': index,
                        'width': resolution, 'height': resolution, 'quality': self.info['default_stream_quality'],
                        'background': self.info['default_background'], 'fovy': camera['fovy_degrees']}, frames))
                    if (index + 1) % 40 == 0 or index + 1 == frames:
                        label = '表面柔化' if quality == 'surface' else 'GS 原生外观'
                        self.progress(99, f'预缓存 {label} 动态对比 {index + 1}/{frames} 帧')
            self.info['hybrid']['prewarmed_frames'] = frames
            self.info['hybrid']['prewarmed_qualities'] = ['surface', 'native']
        self.status = {'ready': True, 'progress': 100, 'message': '就绪', 'scene': self.info}
        logging.getLogger('gaussian-viewer').info('Ready: %d paired frames cached, %.1f MiB JPEG',
                                                 frames, self.image_cache.bytes / 1024**2)

    @torch.no_grad()
    def posed(self, side, index):
        key = side, index
        if key in self.pose_cache:
            self.pose_cache.move_to_end(key)
            return self.pose_cache[key]
        pose = torch.from_numpy(self.poses[side][index].copy()).cuda()
        if side == 1 and index != 0:
            pose = self.correction(self.rest, pose, index)
        if not torch.isfinite(pose).all() or torch.any(pose[:, 3:].norm(dim=-1) < 1e-8):
            raise ValueError('Invalid pose')
        if self.cache_limit:
            self.pose_cache[key] = pose
            while len(self.pose_cache) > self.cache_limit:
                self.pose_cache.popitem(last=False)
        return pose

    @torch.no_grad()
    def apply_side_pose(self, side, index, auxiliary=True):
        # Reset ALL rows and scales, including static rows. Repaired geometry
        # and different motion masks must not leak from right into the left.
        xyz, rotation, scaling = self.side_geometry[side]
        for attribute, base in (('_xyz', xyz), ('_rotation', rotation), ('_scaling', scaling)):
            if getattr(self.model, attribute).shape != base.shape:
                setattr(self.model, attribute, base.clone())
            else:
                getattr(self.model, attribute).copy_(base)
        if hasattr(self, 'side_appearance'):
            self.model._features_dc, self.model._opacity, self.model._features_rest = self.side_appearance[side]
        pose = self.posed(side, index)
        self.model._xyz[self.indices[side]] = pose[:, :3]
        self.model._rotation[self.indices[side]] = pose[:, 3:]
        if side == 1 and self.completion is not None:
            suffix = self.completion.posed(index, self.correction)
            start, end = self.completion.prefix_count, self.completion.prefix_count + len(suffix)
            self.model._xyz[start:end] = suffix[:, :3]
            self.model._rotation[start:end] = suffix[:, 3:]
        if auxiliary:
            self.apply_deformation(side, index)

    @torch.no_grad()
    def apply_deformation(self, side, index):
        field = getattr(self, 'deformation', None)
        if side == 1 and field is not None:
            group_q = self.correction.group_transforms(index)[0]
            self.model._xyz, self.model._rotation = field.deform(
                self.model._xyz, self.model._rotation, group_q, index)
            self.model._scaling = field.deform_scaling(self.model._scaling, index)

    @torch.no_grad()
    def render(self, request):
        mesh_mode = request.get('render_mode', 'gaussian') == 'mesh'
        clay_mode = request.get('render_mode') == 'clay'
        color_mode = request.get('render_mode') == 'pseudocolor'
        hybrid_mode = request.get('render_mode', 'gaussian') == 'hybrid'
        if mesh_mode and self.mesh is None:
            raise ValueError('Mesh is unavailable for this scene')
        if hybrid_mode and self.hybrid is None:
            raise ValueError('Hybrid Mesh bindings are unavailable for this scene')
        from gaussian_renderer import render
        from render_ply import make_camera
        start = time.perf_counter()
        cache = self.pseudocolor_cache if color_mode else self.clay_comparison_cache if clay_mode else self.mesh_comparison_cache if mesh_mode else self.hybrid_comparison_cache if hybrid_mode else self.image_cache
        jpeg = cache.get(request)
        if jpeg is not None:
            return self.packet(request, jpeg, 0., 0., cache_hit=True)
        pose = orbit_pose(self.base_pose, self.target, request)
        pixels = []
        for side in (() if mesh_mode else (0, 1)):
            quality = request.get('hybrid_quality', DEFAULT_HYBRID_QUALITY)
            supersample = self.side_sampling[side]['render_supersample']
            render_width, render_height = (sampling_plan(request['width'], request['height'], quality, supersample)
                if hybrid_mode else (request['width'] * supersample, request['height'] * supersample))
            camera = make_camera(pose, render_width, render_height,
                                 np.deg2rad(request['fovy']))
            self.apply_side_pose(side, request['frame'], auxiliary=not hybrid_mode)
            if hybrid_mode:
                self.hybrid[side].apply(self.model, request['frame'])
                # Binding rewrites positions: deform only after it, exactly once.
                self.apply_deformation(side, request['frame'])
            output = render(camera, self.model, self.pipe, self.backgrounds['black' if clay_mode or color_mode else request['background']], 0., 0., 0.)
            rgb = (resolve_hybrid_rgb(output['render'], request['width'], request['height'], quality)
                   if hybrid_mode else resolve_rgb(output['render'], supersample))
            if hybrid_mode and quality == 'surface':
                import torch.nn.functional as F
                guides = [F.interpolate(output[key].unsqueeze(0), size=(request['height'], request['width']),
                                        mode='area')[0] for key in ('depth', 'alpha')]
                rgb = smooth_surface(rgb, *guides)
            if clay_mode:
                import torch.nn.functional as F
                from utils.gs_clay import gray_gaussians
                guides = [F.avg_pool2d(output[key], supersample) for key in ('depth', 'alpha')]
                coordinates = None
                if request.get('clay_style', 'solid') == 'wire':
                    attributes = render(camera, self.model, self.pipe, self.backgrounds['black'],
                                        0., 0., 0., override_color=self.clay_coordinates[side])
                    coordinates = F.avg_pool2d(attributes['render'], supersample)
                rgb = gray_gaussians(rgb, *guides, request['fovy'], self.backgrounds[request['background']],
                                     coordinates=coordinates)
            if color_mode:
                import torch.nn.functional as F
                from utils.gs_pseudocolor import color_gaussians
                material = render(camera, self.model, self.pipe, self.backgrounds['black'],
                                  0., 0., 0., override_color=self.pseudocolor_materials[side])['render']
                guides = [F.avg_pool2d(output[key], supersample) for key in ('depth', 'alpha')]
                rgb = color_gaussians(rgb, *guides, F.avg_pool2d(material, supersample),
                                      request['fovy'], self.backgrounds[request['background']])
            pixels.append((rgb.permute(1, 2, 0) * 255).byte().contiguous().cpu().numpy())
        if mesh_mode:
            pixels = self.mesh.render_panels(request)
        rendered = time.perf_counter()
        stream = io.BytesIO()
        Image.fromarray(np.concatenate(pixels, axis=1)).save(stream, format='JPEG', quality=request['quality'], subsampling=0)
        jpeg = stream.getvalue()
        cache.put(request, jpeg)
        return self.packet(request, jpeg, (rendered-start)*1000, (time.perf_counter()-rendered)*1000)

    def packet(self, request, jpeg, render_ms, encode_ms, cache_hit=False):
        return image_packet({'id': request['id'], 'frame': request['frame'], 'comparison': True,
                             'render_mode': request.get('render_mode', 'gaussian'),
                             **({'hybrid_quality': request.get('hybrid_quality', DEFAULT_HYBRID_QUALITY)}
                                if request.get('render_mode') == 'hybrid' else {}),
                             'left_frame': request['frame'], 'right_frame': request['frame'],
                             **({'stream': request['_stream']} if '_stream' in request else {}),
                             'width': request['width'] * 2, 'height': request['height'],
                             'panel_width': request['width'], 'render_ms': round(render_ms, 2),
                             'encode_ms': round(encode_ms, 2), 'cache_hit': cache_hit}, jpeg)
