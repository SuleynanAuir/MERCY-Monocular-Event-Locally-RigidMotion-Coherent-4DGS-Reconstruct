"""Verified auxiliary deformation inside V3, shared by GS and Mesh displays.

Historical checkpoint names/formats remain readable so trained weights and
their provenance are not rewritten during integration.
"""
import json
from pathlib import Path

import torch

from utils.event_refinement import qrotate
from utils.rigid_checkpoint import sha256_file
from utils.scgs_v4_field import load_checkpoint


def load_v3_deformation(path, renderer):
    path = Path(path)
    settings = json.loads(path.read_text())
    if settings.get('format') not in ('v3_deformation_v1', 'scgs_v4_comparison_v1'):
        raise ValueError('Invalid V3 deformation manifest')
    paths = {}
    for key in ('checkpoint', 'audit'):
        asset = Path(settings[key])
        paths[key] = asset if asset.is_absolute() else path.parent / asset
        if sha256_file(paths[key]) != settings[key + '_sha256']:
            raise ValueError(f'V3 deformation {key} checksum mismatch')
    field, provenance = load_checkpoint(paths['checkpoint'])
    expected = renderer.verified_v3
    if (str(Path(provenance['baseline']).resolve()) != expected['baseline']
            or str(Path(provenance['candidate']).resolve()) != expected['candidate']
            or provenance['baseline_artifact_sha256'] != expected['artifact_sha256']):
        raise ValueError('Deformation is bound to a different V3 base')
    # The deployed method combines a trained MLP and rigid joint FK. Earlier
    # scale-only/free-space experiments must not silently replace this method.
    if provenance.get('field_kind') != 'local-joint-mlp-v4':
        raise ValueError('Integrated V3 requires the trained local joint MLP')
    audit = json.loads(paths['audit'].read_text())
    frames = renderer.info['frame_count']
    if (field.settings['frame_count'] != frames or audit['frame_count'] != frames
            or len(audit['intervals']) != frames - 1
            or [r['interval'] for r in audit['intervals']] != list(range(frames - 1))
            or int(field.rows.min()) < 0
            or int(field.rows.max()) >= len(renderer.side_geometry[1][0])):
        raise ValueError('Deformation timeline or source rows do not match V3')
    info = dict(
        name='Lego · V1 / V3 对比', left_label='V1 · 原始结果',
        right_label='V3 · Blender＋局部 MLP 变形',
        motion_label='左右同帧 · V3：Blender 主运动＋局部 MLP 辅助',
        description='V3 已集成局部 MLP 变形、关节刚性约束和时间平滑；保留 Blender 主运动、外观与运动幅度，采用事件监督及相位、连接检查。',
        deformation=dict(enabled=True, method='local_joint_mlp',
                         checkpoint_sha256=settings['checkpoint_sha256'],
                         audit_sha256=settings['audit_sha256'],
                         baseline_motion_frozen=True, event_only_auxiliary_training=True,
                         active_joint_groups=provenance['enabled_auxiliary_joints'],
                         parameters=provenance['parameters'],
                         display_modes=['gaussian', 'mesh', 'hybrid']),
        trajectory_stability=provenance['final_geometry'],
        event_validation=dict(
            optimized=provenance.get('best_step', 0) > 0,
            approved=audit['approved_for_export'], preview_only=True,
            summary=audit['summary']['all'], validation_summary=audit['summary']['validation_events'],
            failed_intervals=audit['failed_intervals'],
            reference_weak_intervals=audit.get('reference_self_check_failed_intervals', []),
            interval_checks=[dict(passed=r['passed'], reasons=r['reasons'],
                                  reference_weak=not r['reference_self_check_passed'])
                             for r in audit['intervals']]))
    if settings.get('local_metrics'):
        from utils.local_motion_metrics import load_viewer_metrics
        metrics = Path(settings['local_metrics'])
        if not metrics.is_absolute():
            metrics = path.parent / metrics
        info['local_metrics'] = load_viewer_metrics(
            metrics, settings['local_metrics_sha256'], settings['checkpoint_sha256'],
            provenance['enabled_auxiliary_joints'], frames)
        info['local_metrics']['comparison_scope'] = 'v3_auxiliary_off_on'
    return field, info


@torch.no_grad()
def deform_mesh_vertices(field, xyz, groups, frame):
    """Apply the same world-space joint residual once, after native Mesh motion."""
    if frame == 0 or not torch.any(field.angles[frame]):
        return xyz
    dq, dt = field.delta_q[frame], field.delta_t[frame]
    identity = dq.new_tensor([1., 0., 0., 0.])
    changed = torch.any(dq != identity, dim=-1) | torch.any(dt != 0, dim=-1)
    rows = torch.where(changed[groups])[0]
    selected = groups[rows]
    return xyz.index_copy(0, rows, qrotate(dq[selected], xyz[rows]) + dt[selected])
