"""Guarded, memory-only roof Gaussian experiment on the latest audited parent.

This does not package a model or approve export.  It protects original-camera
visibility, then compares native Gaussian RGB/alpha; priors remain priors.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


ROOF_PARTS = np.arange(11, 16, dtype=np.int32)


def guarded_retirement(proposed, maximum, dynamic, count, threshold):
    proposed, maximum, dynamic = map(np.asarray, (proposed, maximum, dynamic))
    if (proposed.ndim != 1 or proposed.dtype.kind not in 'iu'
            or maximum.shape != proposed.shape or not np.isfinite(maximum).all()
            or np.any(maximum < 0) or np.any(proposed < 0) or np.any(proposed >= count)
            or np.any(proposed[1:] <= proposed[:-1])
            or not np.isfinite(threshold) or not 0 < threshold <= .001
            or np.intersect1d(proposed, dynamic).size):
        raise ValueError('Invalid protected retirement selection')
    return proposed[maximum <= threshold].astype(np.int32)


def next_active(active, maximum, threshold):
    active, maximum = np.asarray(active), np.asarray(maximum)
    if (active.dtype != bool or active.ndim != 1 or maximum.shape != active.shape
            or not np.isfinite(maximum).all() or np.any(maximum < 0)
            or not np.isfinite(threshold) or not 0 < threshold <= .001):
        raise ValueError('Invalid addition exposure arrays')
    return active & (maximum <= threshold)


def validate_additional_guard_metadata(metadata, primary):
    """Extra restoration cameras must use the same verified mesh and motion."""
    keys = ('mesh', 'mesh_sha256', 'motion', 'motion_sha256', 'objects',
            'motion_metadata', 'excluded_deforming_objects', 'camera', 'resolution')
    if any(metadata.get(key) != primary.get(key) for key in keys):
        raise ValueError('Additional guards differ from the verified primary mesh, motion or camera')
    views = metadata.get('views')
    if not isinstance(views, list) or not 1 <= len(views) <= 64:
        raise ValueError('Additional restoration guards must contain 1 to 64 native views')
    return views


def validate_target_triangles(arrays, mesh, allowed, tolerance=2e-6):
    """Validate exact on-triangle centers and normal alignment for static parts."""
    xyz, q, ids, parts = (arrays[k] for k in ('new_xyz', 'new_quaternions', 'triangle_ids', 'part_ids'))
    faces = mesh['triangles']
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or q.shape != (len(xyz), 4)
            or ids.shape != (len(xyz),) or ids.dtype.kind not in 'iu'
            or parts.shape != ids.shape or parts.dtype.kind not in 'iu'
            or not np.isfinite(xyz).all() or not np.isfinite(q).all()
            or np.any(ids < 0) or np.any(ids >= len(faces))
            or not np.array_equal(mesh['part_ids'][ids], parts)
            or not np.isin(parts, allowed).all()
            or np.any(np.abs(np.linalg.norm(q, axis=1)-1) > 1e-4)):
        raise ValueError('Invalid targeted triangle sample identities')
    corners = mesh['vertices'][faces[ids]].astype(np.float64)
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    cross = np.cross(b-a, c-a)
    area = np.linalg.norm(cross, axis=1)
    if np.any(area <= 0):
        raise ValueError('Targeted sample references a degenerate triangle')
    normal = cross/area[:, None]
    offset = np.sum((xyz-a)*normal, axis=1)
    projected = xyz-offset[:, None]*normal
    bary = np.column_stack((np.sum(np.cross(b-projected, c-projected)*normal, axis=1),
                            np.sum(np.cross(c-projected, a-projected)*normal, axis=1),
                            np.sum(np.cross(a-projected, b-projected)*normal, axis=1)))/area[:, None]
    edges = np.column_stack((np.linalg.norm(b-c, axis=1), np.linalg.norm(c-a, axis=1),
                             np.linalg.norm(a-b, axis=1)))
    if np.any(np.abs(offset) > tolerance) or np.any(bary < -tolerance*edges/area[:, None]):
        raise ValueError('Targeted sample is not on its actual triangle')
    w, x, y, z = q.astype(np.float64).T
    local_z = np.column_stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))
    if np.any(np.abs(np.sum(local_z*normal, axis=1)) < .999):
        raise ValueError('Targeted sample thin axis is not its triangle normal')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'selection', 'masks', 'front-masks', 'supervision', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--views', nargs='+', type=int, default=[4, 6, 17, 28, 32])
    p.add_argument('--threshold', type=float, default=.0001)
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--variants', nargs='+', choices=['additions_only', 'guarded_remove_add'],
                   default=['additions_only', 'guarded_remove_add'])
    p.add_argument('--full-timeline', action='store_true')
    p.add_argument('--probe-stride', type=int)
    p.add_argument('--probe-frames', nargs='+', type=int,
                   help='Explicit finite visibility probes; never marks full timeline unless exactly all frames')
    p.add_argument('--resume', type=Path, help='Exact prior diagnostic preview.npz; never reactivates rejected rows')
    p.add_argument('--additional-guard-masks', type=Path,
                   help='Extra exact cameras used for retirement restoration, not independent validation')
    p.add_argument('--global-support-preservation', action='store_true',
                   help='Preserve whole-roof and all zero-replacement-part static support before retirement')
    args = p.parse_args()
    if (args.output.exists() or not 1 <= args.rounds <= 128 or not 0 < args.threshold <= .001
            or len(set(args.variants)) != len(args.variants)
            or (args.probe_stride is not None and not 1 <= args.probe_stride <= 100)
            or (args.probe_frames is not None and (args.full_timeline or args.probe_stride is not None))):
        raise ValueError('Use a new output directory and bounded scan settings')
    for name in ('baseline', 'candidate', 'selection', 'masks', 'front_masks', 'supervision'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    from scripts.render_lego_multiview import verify_inputs, sha256, validate_view_plan, validate_mask_arrays
    from scripts.refine_lego_events import MotionRenderer
    from scripts.prune_lego_completion_exposure import contribution_from_dc_gradient
    from utils.event_refinement import correction_from_record
    from gaussian_renderer import render
    from render_ply import make_camera
    from utils.surface_detail_metrics import surface_comparison, mesh_structure_edges, edge_agreement
    from scipy.ndimage import binary_erosion, binary_dilation, label
    from PIL import Image
    import torch

    print(json.dumps({'stage': 'verify_immutable_parent'}), flush=True)
    cache, record, mask_meta, _, paths, hashes, _, _ = verify_inputs(
        args.baseline, args.candidate, args.masks)
    additional_meta, additional_parts = None, None
    if args.additional_guard_masks:
        args.additional_guard_masks = args.additional_guard_masks.resolve(strict=True)
        additional_hash = sha256(args.additional_guard_masks)
        with np.load(args.additional_guard_masks, allow_pickle=False) as archive:
            additional_meta = json.loads(str(archive['metadata_json']))
            validate_additional_guard_metadata(additional_meta, mask_meta)
            validate_view_plan(additional_meta, archive['frame_indices'], cache)
            validate_mask_arrays(archive['part_ids'], archive['depth'], additional_meta)
            additional_parts = archive['part_ids']
        paths['additional_guard_masks'] = args.additional_guard_masks
        hashes['additional_guard_masks'] = additional_hash
    with np.load(args.front_masks, allow_pickle=False) as archive:
        front_meta = json.loads(str(archive['metadata_json']))
        validate_view_plan(front_meta, archive['frame_indices'], cache)
        validate_mask_arrays(archive['part_ids'], archive['depth'], front_meta)
    if any(front_meta.get(key) != mask_meta.get(key) for key in
           ('mesh', 'mesh_sha256', 'motion', 'motion_sha256', 'objects')):
        raise ValueError('Front probes do not identify the exact validated orbit mesh and motion')
    if (front_meta.get('front_only_stride') is None
            or any(v['yaw'] != 0 or v['pitch'] != 0 or v['zoom'] != 1
                   or v['pan_x'] != 0 or v['pan_y'] != 0 for v in front_meta['views'])):
        raise ValueError('Front probes must use only the exact observed camera')
    mask_frames = [v['frame'] for v in front_meta['views']]
    frames = (list(range(cache['frame_count'])) if args.full_timeline else
              sorted(set(range(0, cache['frame_count'], args.probe_stride)) | {cache['frame_count']-1})
              if args.probe_stride is not None else mask_frames)
    if args.probe_frames is not None:
        if (args.probe_frames != sorted(set(args.probe_frames))
                or any(i < 0 or i >= cache['frame_count'] for i in args.probe_frames)):
            raise ValueError('Explicit probe frames must be sorted distinct valid timeline indices')
        frames = args.probe_frames
    metric_frames = sorted(set(frames) & set(mask_frames))
    mask_positions = [mask_frames.index(f) for f in metric_frames]
    frame_positions = [frames.index(f) for f in metric_frames]
    if len(set(args.views)) != len(args.views) or any(i < 0 or i >= len(mask_meta['views']) for i in args.views):
        raise ValueError('Invalid orbit-view indices')
    with np.load(args.selection, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    meta = json.loads(str(arrays['metadata_json']))
    n = len(arrays['new_xyz'])
    allowed = np.asarray(meta.get('target_part_ids'), np.int32)
    if (meta.get('format') not in ('lego_static_roof_rebuild_v1', 'lego_static_back_detail_proposal_v1')
            or meta.get('source_sha256') != hashes['source_ply']
            or meta.get('source_gaussian_count') != cache['gaussian_count']
            or allowed.ndim != 1 or not len(allowed) or np.any(allowed < 0) or np.any(allowed > 141)
            or np.any(allowed[1:] <= allowed[:-1])
            or not 0 < n <= 750000 or not np.isin(arrays['part_ids'], allowed).all()
            or not np.isfinite(arrays['new_dc_gray']).all()
            or np.any(arrays['new_dc_gray'] < 0) or np.any(arrays['new_dc_gray'] > 1)):
        raise ValueError('Roof selection does not identify this immutable source')
    mesh_path = Path(meta['input_paths']['mesh'])
    if sha256(mesh_path) != meta['input_sha256']['mesh'] or sha256(mesh_path) != hashes['mesh']:
        raise ValueError('Roof triangle geometry differs from current mesh prior')
    with np.load(mesh_path, allow_pickle=False) as archive:
        mesh = {key: archive[key] for key in ('vertices', 'triangles', 'part_ids')}
    validate_target_triangles(arrays, mesh, allowed)
    source_geometry = None
    if args.global_support_preservation:
        from plyfile import PlyData
        from utils.protected_surface_detail import static_support_guard
        source_vertices = PlyData.read(paths['source_ply'], mmap='r')['vertex'].data
        source_geometry = tuple(np.column_stack([source_vertices[field] for field in fields])
            for fields in (('x', 'y', 'z'), ('scale_0', 'scale_1', 'scale_2'),
                           ('rot_0', 'rot_1', 'rot_2', 'rot_3')))
        del source_vertices
    else:
        del mesh
    with np.load(paths['motion'], allow_pickle=False) as archive:
        groups = archive['group_ids'][allowed]
        roots = np.flatnonzero(np.asarray(record['joint_structure']['parents']) == -1)
        if (not np.isin(groups, roots).all()
                or not np.allclose(archive['quaternions'][:, groups], [1, 0, 0, 0], rtol=0, atol=1e-7)
                or not np.allclose(archive['translations'][:, groups], 0, rtol=0, atol=1e-7)):
            raise ValueError('Every target part must be identity-root static for all 400 frames')
        if additional_meta is not None:
            # The primary verifier checked its own frames. Check the actual
            # complete hard-joint SE3 at every additional restoration frame too.
            cpu_correction = correction_from_record(record, 'cpu').eval()
            with torch.no_grad():
                for frame in sorted({v['frame'] for v in additional_meta['views']}):
                    q, t = cpu_correction.group_transforms(frame)
                    if (np.max(np.abs(q.numpy()-archive['quaternions'][frame])) > 1e-6
                            or np.max(np.abs(t.numpy()-archive['translations'][frame])) > 1e-6):
                        raise ValueError('Additional guard hard-joint motion differs at frame '+str(frame))
            del cpu_correction
    dynamic = np.load(args.baseline/'dynamic_indices.npy', allow_pickle=False)
    proposed = arrays['retired_indices']
    guarded_retirement(proposed, np.zeros(len(proposed)), dynamic, cache['gaussian_count'], args.threshold)
    completion_report = json.loads(Path(cache['multiview_completion']['report']).read_text())
    with np.load(cache['multiview_completion']['extension'], allow_pickle=False) as archive:
        suffix_parts = archive['part_ids']
    prefix = completion_report['prefix_count']
    if np.any(~np.isin(suffix_parts[proposed[proposed >= prefix]-prefix], allowed)):
        raise ValueError('Proposed retirement touches a nonroof completion row')
    parent_report = json.loads(Path(completion_report['parent_report']).read_text())
    if np.any((proposed >= parent_report['retained_gaussian_count']) & (proposed < prefix)):
        raise ValueError('Proposed retirement touches rebuilt dynamic geometry')
    with np.load(args.front_masks, allow_pickle=False) as archive:
        front_parts = archive['part_ids']
    with np.load(args.masks, allow_pickle=False) as archive:
        orbit_parts = archive['part_ids'][args.views]
        orbit_depth = archive['depth'][args.views]
    with np.load(args.supervision, allow_pickle=False) as archive:
        references = archive['images'][frames].astype(np.float32)/255
    audit_record = json.loads((args.candidate/'event-audit.json').read_text())
    if sha256(args.supervision) != audit_record['artifact_sha256']['supervision']:
        raise ValueError('Reference video differs from the immutable parent event audit')
    if references.shape != (len(frames), 640, 640):
        raise ValueError('Unexpected native supervision shape')
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    print(json.dumps({'stage': 'load_latest_gaussians', 'new_candidates': n,
                     'retirement_candidates': len(proposed)}), flush=True)
    renderer = MotionRenderer(args.baseline, 640)
    correction = correction_from_record(record, 'cuda').eval()
    for parameter in correction.parameters():
        parameter.requires_grad_(False)
    original_camera = renderer.camera
    source_count = cache['gaussian_count']
    base = {key: getattr(renderer.model, key).detach().clone()
            for key in ('_scaling', '_opacity', '_features_dc', '_features_rest')}
    added = {'_scaling': torch.from_numpy(arrays['new_log_scale']).cuda(),
        '_opacity': torch.from_numpy(arrays['new_opacity_logit'][:, None]).cuda(),
        '_features_dc': torch.from_numpy(np.repeat(((arrays['new_dc_gray']-.5)/.28209479177387814)[:, None, None], 3, 2)).cuda(),
        '_features_rest': torch.empty((n, 0, 3), device='cuda')}

    class AppendRoof:
        def __init__(self):
            self.xyz = torch.from_numpy(arrays['new_xyz']).cuda()
            self.rotation = torch.from_numpy(arrays['new_quaternions']).cuda()

        def append_geometry(self, xyz, rotation):
            return torch.cat((xyz, self.xyz), 0), torch.cat((rotation, self.rotation), 0)

    extension = AppendRoof()
    resume = None
    if args.resume:
        with np.load(args.resume.resolve(strict=True), allow_pickle=False) as archive:
            resume = {key: archive[key] for key in archive.files
                      if key.endswith('_active') or key.endswith('_retired_indices') or key == 'metadata_json'}
        resume_meta = json.loads(str(resume['metadata_json']))
        if (resume_meta.get('format') != 'lego_guarded_roof_detail_preview_v1'
                or resume_meta.get('threshold') != args.threshold
                or resume_meta.get('target_part_ids') != allowed.tolist()
                or resume_meta.get('color_residual') != cache.get('color_residual')
                or resume_meta.get('artifact_sha256', {}).get('selection') != sha256(args.selection)
                or any(resume_meta.get('artifact_sha256', {}).get(key) != hashes[key]
                       for key in ('source_ply', 'correction', 'baseline_cache_metadata'))):
            raise ValueError('Resume does not identify the exact parent, selection and threshold')

    def configure(active=None, retired=None):
        renderer.static_surface = extension if active is not None else None
        for key, value in base.items():
            setattr(renderer.model, key, torch.cat((value, added[key]), 0) if active is not None else value.clone())
        if active is not None:
            renderer.model._opacity[source_count:][torch.from_numpy(~active).cuda()] = -30.
        if retired is not None and len(retired):
            rows = torch.from_numpy(retired.astype(np.int64)).cuda()
            renderer.model._opacity[rows] = torch.minimum(renderer.model._opacity[rows],
                                                          torch.full_like(renderer.model._opacity[rows], -30.))

    def scan(rows, phase, number):
        renderer.camera = original_camera
        dc = torch.zeros_like(renderer.model._features_dc, requires_grad=True)
        maxima = torch.zeros(len(rows), device='cuda')
        selected = torch.from_numpy(np.asarray(rows, np.int64)).cuda()
        for position, frame in enumerate(frames):
            dc.grad = None
            renderer.model._features_dc = dc
            value = renderer.image(frame, correction)
            if not torch.isfinite(value).all() or torch.any(value < -1e-7) or torch.any(value > .50001):
                raise ValueError('Uniform-color exposure probe is clamped or nonfinite')
            value.sum().backward()
            maxima = torch.maximum(maxima, contribution_from_dc_gradient(dc.grad)[selected].detach())
            if position == 0 or position+1 == len(frames):
                print(json.dumps({'phase': phase, 'round': number, 'probe': position+1,
                    'total': len(frames), 'maximum': float(maxima.max()) if len(rows) else 0.}), flush=True)
        return maxima.cpu().numpy()

    configure()
    original_exposure = scan(proposed, 'retirement_guard', 1)
    retired = guarded_retirement(proposed, original_exposure, dynamic, source_count, args.threshold)
    result = {'format': 'lego_guarded_roof_detail_preview_v1', 'diagnostic_only': True,
        'approved_for_export': False, 'source_artifacts_changed': False,
        'baseline': str(args.baseline), 'candidate': str(args.candidate),
        'artifact_sha256': {**hashes, 'selection': sha256(args.selection),
            'supervision': sha256(args.supervision), 'front_masks': sha256(args.front_masks)},
        'color_residual': cache.get('color_residual'), 'threshold': args.threshold,
        'front_probe_frames': frames, 'views': [mask_meta['views'][i] for i in args.views],
        'additional_guard_masks': str(args.additional_guard_masks) if additional_meta is not None else None,
        'additional_guard_masks_sha256': hashes.get('additional_guard_masks'),
        'additional_guard_views': additional_meta['views'] if additional_meta is not None else [],
        'global_support_preservation_enabled': args.global_support_preservation,
        'full_timeline_exposure_scan': frames == list(range(cache['frame_count'])),
        'front_roof_metric_frames': metric_frames,
        'resumed_from': str(args.resume.resolve()) if args.resume else None,
        'proposed_retired_count': len(proposed), 'guarded_retired_count': len(retired),
        'front_visible_original_rows_preserved': len(proposed)-len(retired),
        'proposed_new_count': n, 'target_part_ids': allowed.tolist(),
        'all_motion_frozen': True, 'variants': {},
        'scope': 'Native Gaussian diagnostic, no image filtering. Retire only proposed roof rows invisible '
                 'at recorded probes; add only actual roof triangle points with bounded combined-model '
                 'exposure. Front gradient bounds are NOT RGB/event invariance or full-timeline acceptance. '
                 'Hidden geometry/appearance are explicit Blender/same-part priors, not observations. '
                 'Additional restoration cameras are selection guards, not independent held-out evidence.'}
    saved = {'proposed_retired_indices': proposed, 'retired_indices': retired,
             'retirement_maximum_probe_exposure': original_exposure}
    states = {'original': (None, None)}
    guard_views = result['views']+result['additional_guard_views']
    guard_parts = list(orbit_parts)
    if additional_parts is not None:
        guard_parts.extend(additional_parts)
    guard_cameras = [make_camera(np.asarray(view['camera_pose']), 640, 640,
                    np.deg2rad(cache['camera']['fovy_degrees'])) for view in guard_views]
    result['interior_guard_view_count'] = len(guard_views)
    interiors = []
    for parts in guard_parts:
        interior = np.zeros(parts.shape, bool)
        for part in allowed:
            interior |= binary_erosion(parts == part, iterations=1)
        interiors.append(interior)
    configure()
    parent_alpha = []
    with torch.no_grad():
        for camera, view in zip(guard_cameras, guard_views):
            renderer.camera = camera
            renderer.image(view['frame'], correction)
            parent_alpha.append(render(camera, renderer.model, renderer.pipe, renderer.background,
                                       0., 0., 0.)['alpha'].reshape(640, 640).cpu().numpy())

    def interior_guard(active, remove):
        """Restore real old Gaussian rows; never alter rendered image pixels."""
        history = []
        for number in range(1, 6):
            configure(active, remove)
            holes = []
            with torch.no_grad():
                for camera, view, before, interior in zip(guard_cameras, guard_views, parent_alpha, interiors):
                    renderer.camera = camera
                    renderer.image(view['frame'], correction)
                    alpha = render(camera, renderer.model, renderer.pipe, renderer.background,
                                   0., 0., 0.)['alpha'].reshape(640, 640).cpu().numpy()
                    holes.append((before >= .5) & (alpha < .5) & interior)
            counts = [int(mask.sum()) for mask in holes]
            row = {'phase': 'prior_interior_retirement_restore', 'round': number,
                   'new_hole_pixels_by_view': counts, 'retired_before': len(remove),
                   'restore_support_halo_pixels': 2,
                   'restore_support_halo_scope': 'Gaussian contribution selection only; rendered pixels unchanged'}
            if not any(counts):
                row['restored'] = 0; history.append(row)
                return remove, history, True
            if number == 5 or not len(remove):
                row['restored'] = 0; row['unresolved'] = True; history.append(row)
                return remove, history, False
            if number == 4:
                # Old source screen labels are not reliable 3-D part ownership.
                # Restore all remaining targeted retirement rows, a stricter
                # fallback than guessing a narrower part set from those labels.
                row['restored'] = len(remove); row['fallback'] = 'restore_all_remaining_targeted_retirements'
                remove = np.empty(0, np.int32)
            else:
                configure(active, None)  # Old opacities reactivated, new Gaussians still present.
                dc = torch.zeros_like(renderer.model._features_dc, requires_grad=True)
                selection = torch.from_numpy(remove.astype(np.int64)).cuda()
                exposure = torch.zeros(len(remove), device='cuda')
                for camera, view, mask in zip(guard_cameras, guard_views, holes):
                    if not mask.any():
                        continue
                    renderer.camera = camera
                    renderer.model._features_dc = dc
                    dc.grad = None
                    gray = renderer.image(view['frame'], correction)
                    support = binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=2)
                    (gray*torch.from_numpy(support).cuda()).sum().backward()
                    exposure = torch.maximum(exposure, contribution_from_dc_gradient(dc.grad)[selection].detach())
                restore = exposure.cpu().numpy() > 1e-5
                row['restored'] = int(restore.sum()); row['contribution_threshold'] = 1e-5
                remove = remove[~restore]
            history.append(row)
            print(json.dumps(row), flush=True)
        raise AssertionError('Bounded interior guard must terminate')

    for name in args.variants:
        remove = retired.copy() if name == 'guarded_remove_add' else None
        active = np.ones(n, bool)
        if resume is not None:
            prior_active = resume.get(name+'_active')
            if prior_active is None or prior_active.dtype != bool or prior_active.shape != (n,):
                raise ValueError('Resume does not contain exact variant active rows')
            active = prior_active.copy()
            if remove is not None and name+'_retired_indices' in resume:
                old_remove = resume[name+'_retired_indices']
                if old_remove.ndim != 1 or old_remove.dtype.kind not in 'iu' or not np.isin(old_remove, proposed).all():
                    raise ValueError('Invalid resumed retirement rows')
                remove = np.intersect1d(remove, old_remove).astype(np.int32)
        history, global_history, global_metadata = [], [], None

        def preserve_global_support(remove, phase, number):
            nonlocal global_metadata
            if not args.global_support_preservation or remove is None:
                return remove
            protected, global_metadata = static_support_guard(proposed, *source_geometry,
                mesh, allowed, arrays['part_ids'][active], prefix_count=prefix,
                suffix_part_ids=suffix_parts)
            if (protected.dtype != np.int32 or protected.ndim != 1
                    or np.any(protected[1:] <= protected[:-1])
                    or not np.isin(protected, proposed).all()):
                raise ValueError('Global support preservation returned invalid proposed identities')
            remaining = np.setdiff1d(remove, protected, assume_unique=True).astype(np.int32)
            row = {'phase': phase, 'round': number, 'retired_before': len(remove),
                   'restored': len(remove)-len(remaining), 'retired_after': len(remaining),
                   'globally_protected_proposed_count': len(protected)}
            global_history.append(row)
            print(json.dumps({'variant': name, 'global_support_preservation': row}), flush=True)
            return remaining

        for number in range(1, args.rounds+1):
            remove = preserve_global_support(remove, 'before_exposure_peeling', number)
            configure(active, remove)
            maximum = scan(np.arange(source_count, source_count+n), name, number)
            changed = next_active(active, maximum, args.threshold)
            history.append({'round': number, 'before': int(active.sum()), 'retained': int(changed.sum()),
                'removed': int((active & ~changed).sum()),
                'maximum_retained_exposure': float(maximum[changed].max()) if changed.any() else 0.})
            print(json.dumps({'variant': name, **history[-1]}), flush=True)
            if np.array_equal(changed, active):
                break
            active = changed
        converged = history[-1]['removed'] == 0
        guard_history, guard_converged = [], None
        if remove is not None:
            # A last addition may have been removed on the final bounded round;
            # zero-replacement protection must reflect the final actual set.
            remove = preserve_global_support(remove, 'after_exposure_peeling', len(history))
            remove, guard_history, guard_converged = interior_guard(active, remove)
            # Restoring old splats changes visibility; confirm additions again.
            configure(active, remove)
            final_maximum = scan(np.arange(source_count, source_count+n), name+'_after_restore_confirm', 1)
            if np.any(active & (final_maximum > args.threshold)):
                converged = False
            maximum = final_maximum
        result['variants'][name] = {'active_added_count': int(active.sum()),
            'retired_count': len(remove) if remove is not None else 0, 'probe_converged': converged,
            'exposure_history': history, 'interior_restore_history': guard_history,
            'interior_restore_converged': guard_converged,
            'maximum_final_active_exposure': float(maximum[active].max()) if active.any() else 0.}
        if args.global_support_preservation:
            result['variants'][name]['global_support_guard'] = global_metadata
            result['variants'][name]['global_support_guard_history'] = global_history
        saved[name+'_active'] = active
        saved[name+'_retired_indices'] = remove if remove is not None else np.empty(0, np.int32)
        saved[name+'_maximum_probe_exposure'] = maximum
        states[name] = (active, remove)

    with torch.no_grad():
        for name, (active, remove) in states.items():
            configure(active, remove)
            front_gray, front_alpha = [], []
            renderer.camera = original_camera
            for frame in frames:
                gray = renderer.image(frame, correction).cpu().numpy()
                rendered = render(renderer.camera, renderer.model, renderer.pipe, renderer.background, 0., 0., 0.)
                front_gray.append(gray)
                front_alpha.append(rendered['alpha'].reshape(640, 640).cpu().numpy())
            front_gray, front_alpha = np.asarray(front_gray), np.asarray(front_alpha)
            saved[name+'_front_gray'], saved[name+'_front_alpha'] = front_gray, front_alpha
            errors = np.abs(front_gray-references)
            roof_mask = np.isin(front_parts[mask_positions], ROOF_PARTS)
            measured_errors = errors[frame_positions]
            scores = {'front_global_mae': float(errors.mean()),
                'front_roof_mae': float(measured_errors[roof_mask].mean()),
                'front_per_frame_global_mae': errors.mean((1, 2)).tolist(),
                'front_per_frame_roof_mae': [float(e[m].mean()) for e, m in zip(measured_errors, roof_mask)]}
            if name != 'original':
                parent = saved['original_front_gray']
                difference = np.abs(front_gray-parent)
                scores.update({'front_parent_l1': float(difference.mean()),
                    'front_parent_max_abs': float(difference.max()),
                    'front_new_alpha_holes': int(((front_alpha < .5) & (saved['original_front_alpha'] >= .5)).sum())})
            orbit_gray, orbit_alpha = [], []
            for view in result['views']:
                renderer.camera = make_camera(np.asarray(view['camera_pose']), 640, 640,
                    np.deg2rad(cache['camera']['fovy_degrees']))
                gray = renderer.image(view['frame'], correction).cpu().numpy()
                rendered = render(renderer.camera, renderer.model, renderer.pipe, renderer.background, 0., 0., 0.)
                rgb = rendered['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                alpha = rendered['alpha'].reshape(640, 640).cpu().numpy()
                orbit_gray.append(gray); orbit_alpha.append(alpha)
                Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(args.output/f'{name}-view-{view["view_index"]:02d}.png')
                Image.fromarray(np.rint(alpha*255).astype(np.uint8)).save(args.output/f'{name}-view-{view["view_index"]:02d}-alpha.png')
            orbit_gray, orbit_alpha = np.asarray(orbit_gray), np.asarray(orbit_alpha)
            saved[name+'_orbit_gray'], saved[name+'_orbit_alpha'] = orbit_gray, orbit_alpha
            scores['roof_prior_pixels_below_alpha_0_5'] = [int(((a < .5) & np.isin(m, ROOF_PARTS)).sum())
                                                        for a, m in zip(orbit_alpha, orbit_parts)]
            if name != 'original':
                scores['orbit_new_alpha_holes'] = [int(((a < .5) & (b >= .5)).sum())
                                                   for a, b in zip(orbit_alpha, saved['original_orbit_alpha'])]
                scores['orbit_gray_l1_vs_parent'] = np.abs(orbit_gray-saved['original_orbit_gray']).mean((1, 2)).tolist()
                surface_scores = []
                for index, parts in enumerate(orbit_parts):
                    bg, ag = saved['original_orbit_gray'][index], orbit_gray[index]
                    ba, aa = saved['original_orbit_alpha'][index], orbit_alpha[index]
                    item = surface_comparison(bg, ag, ba, aa, parts, allowed)
                    interior = np.zeros(parts.shape, bool)
                    for part in np.unique(parts):
                        if part >= 0:
                            interior |= binary_erosion(parts == part, iterations=1)
                    holes = (ba >= .5) & (aa < .5) & interior
                    components, _ = label(holes, np.ones((3, 3), bool))
                    sizes = np.bincount(components.ravel())[1:]
                    item['new_holes_per_part_eroded_interior'] = {'pixels': int(holes.sum()),
                        'largest_component': int(sizes.max()) if sizes.size else 0}
                    edges, region = mesh_structure_edges(parts, orbit_depth[index], allowed)
                    item['before_prior_edge'] = edge_agreement(bg, edges, region)
                    item['after_prior_edge'] = edge_agreement(ag, edges, region)
                    surface_scores.append(item)
                scores['surface_detail'] = surface_scores
            result['variants'].setdefault(name, {}).update(scores)
            print(json.dumps({'rendered_variant': name, 'front_frames': len(frames),
                'orbit_views': len(args.views), 'front_global_mae': scores['front_global_mae'],
                'front_roof_mae': scores['front_roof_mae'],
                'front_parent_l1': scores.get('front_parent_l1'),
                'front_parent_max_abs': scores.get('front_parent_max_abs'),
                'front_new_alpha_holes': scores.get('front_new_alpha_holes')}), flush=True)
    for key in ('source_ply', 'correction', 'baseline_cache_metadata', *(['additional_guard_masks'] if additional_meta else [])):
        if sha256(paths[key]) != hashes[key]:
            raise ValueError('Immutable input changed during diagnostic: '+key)
    result['seconds'] = time.monotonic()-started
    np.savez_compressed(args.output/'preview.npz', **saved,
        metadata_json=np.asarray(json.dumps(result, allow_nan=False)))
    (args.output/'preview.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
