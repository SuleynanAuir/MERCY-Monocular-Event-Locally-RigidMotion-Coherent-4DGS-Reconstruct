"""Learn a new continuous Gaussian motion from G0 + raw events + K/R/T only.

No recorded RGB, Blender mesh/animation, old binding, old motion, or previous
supervision is loaded. Event-integrated log intensity is a derived training
proxy anchored by the rendered G0, NOT an observed image. Fitting every event
window is reported as in-sample consistency, never independent validation.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.event_only_inputs import preflight_event_only_inputs, sha256_file
from utils.event_only_renderer import EventOnlyRenderer
from utils.event_only_motion import build_graph, EventOnlyMotion
from utils.event_only_supervision import build_event_windows, training_spans, audit_event_prediction
from utils.event_only_checkpoint import save_event_only_checkpoint
from utils.event_only_correspondence import make_feature_evidence, feature_reprojection_loss


def pool_log(image, factor):
    if factor == 1:
        return image
    return F.avg_pool2d(image[None, None], factor, factor)[0, 0]


def robust_abs(value, epsilon=.01):
    return (value.square() + epsilon ** 2).sqrt() - epsilon


def observation_log(rendered, background, log_floor):
    """Event observation model; RGB export may independently use black canvas."""
    intensity = rendered['gray'] + (1 - rendered['alpha'].squeeze(0)) * background
    return intensity.clamp(log_floor, 1).log()


def integral_loss(predicted_log, target_log):
    """Native-pixel log-domain multiscale fit; no log-after-pooling swap."""
    scales = [s for s in (1, 4, 16, 32) if min(predicted_log.shape) >= s]
    return sum(robust_abs(pool_log(predicted_log, s) - pool_log(target_log, s)).mean()
               for s in scales) / len(scales)


def difference_loss(predicted_delta, expected_delta):
    scales = [s for s in (1, 4, 16) if min(predicted_delta.shape) >= s]
    result = predicted_delta.sum() * 0
    for scale in scales:
        actual, expected = pool_log(predicted_delta, scale), pool_log(expected_delta, scale)
        amplitude = robust_abs(actual - expected).mean() / expected.abs().mean().clamp_min(.02)
        denominator = ((actual.square().sum() + 1e-6).sqrt()
                       * (expected.square().sum() + 1e-6).sqrt())
        cosine = 1 - (actual * expected).sum() / denominator
        # With no net event change, polarity/cosine has no identifiable target.
        informative = (expected.square().sum().detach() > 1e-4).to(actual.dtype)
        result = result + amplitude + .1 * cosine * informative
    return result / len(scales)


def restrict_gradient_to_knot(motion, row):
    for parameter in (motion.translations, motion.rotation_vectors):
        if parameter.grad is not None:
            parameter.grad[:row] = 0
            parameter.grad[row + 1:] = 0


def enforce_frozen_groups(motion, frozen, *, gradients_only=False):
    """Keep explicit static/unknown identity priors from drifting via Adam."""
    if frozen is None:
        return
    mask = torch.as_tensor(frozen, device=motion.translations.device)
    if mask.dtype != torch.bool or mask.shape != (motion.translations.shape[1],):
        raise ValueError('Frozen groups must be one boolean per node')
    with torch.no_grad():
        for parameter in (motion.translations, motion.rotation_vectors):
            if parameter.grad is not None:
                parameter.grad[:, mask] = 0
            if not gradients_only:
                parameter[:, mask] = 0


def seed_chronological_knot(motion, row, initial_state=None):
    """Carry fitted corrections forward without erasing event-derived motion."""
    with torch.no_grad():
        for name in ('translations', 'rotation_vectors'):
            parameter = getattr(motion, name)
            if initial_state is None:
                if row:
                    parameter[row].copy_(parameter[row - 1])
            else:
                parameter[row].copy_(initial_state[name][row])
                if row:
                    parameter[row].add_(parameter[row - 1] - initial_state[name][row - 1])


def write_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('g0', 'events', 'intrinsics', 'extrinsics', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--g0-time', type=float, required=True)
    parser.add_argument('--time-scale', type=float, required=True)
    parser.add_argument('--contrast-positive', type=float, default=.2)
    parser.add_argument('--contrast-negative', type=float, default=.2)
    parser.add_argument('--contrast-confirmed', action='store_true')
    parser.add_argument('--identity-time-confirmed', action='store_true')
    parser.add_argument('--fit-split', choices=('all', 'train'), default='all')
    parser.add_argument('--event-flow-initialization', action='store_true',
                        help='Classical flow from G0 + event-integrated proxies only; no input video')
    parser.add_argument('--estimate-event-background', action='store_true',
                        help='Heuristic radiometry estimate from G0 and raw-event decreases, not calibration truth')
    parser.add_argument('--propagate-event-flow', action='store_true',
                        help='Extend confident event flow over a bounded G0 3D graph; no semantic part labels')
    parser.add_argument('--rigid-feature-initialization', action='store_true',
                        help='Infer exact rigid groups from G0 surface features and event-only proxy tracking')
    parser.add_argument('--feature-max-corners', type=int, default=2048)
    parser.add_argument('--feature-method', choices=('corners', 'sift'), default='corners')
    parser.add_argument('--motion-model', choices=('se3', 'camera_plane'), default='se3',
                        help='Opt-in fixed-camera-depth initialization and training prior; not physical motion proof')
    parser.add_argument('--feature-reference-reacquisition', action='store_true')
    parser.add_argument('--rigid-max-groups', type=int, default=6)
    from utils.event_only_feature_init import add_binding_arguments, validate_binding_arguments, bind_feature_groups
    add_binding_arguments(parser)
    parser.add_argument('--feature-reprojection-weight', type=float, default=.02,
                        help='Weight of event-derived 2D correspondence prior, not independent supervision')
    parser.add_argument('--step-seconds', type=float, default=.05)
    parser.add_argument('--substeps', type=int, default=2)
    parser.add_argument('--knot-stride', type=int, default=4)
    parser.add_argument('--nodes', type=int, default=256)
    parser.add_argument('--warmup-steps', type=int, default=24)
    parser.add_argument('--polish-steps', type=int, default=400)
    parser.add_argument('--stop-after-knots', type=int, default=0,
                        help='Diagnostic only: truncate optimization, NOT the input timeline or audit')
    parser.add_argument('--learning-rate', type=float, default=.0005)
    parser.add_argument('--rigidity-weight', type=float, default=200.)
    parser.add_argument('--acceleration-weight', type=float, default=.0001)
    parser.add_argument('--log-floor', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new directory; existing work is never overwritten')
    if (min(args.nodes, args.knot_stride, args.substeps) < 1 or args.seed < 0
            or min(args.warmup_steps, args.polish_steps, args.stop_after_knots) < 0
            or not 0 < args.log_floor < 1 or not 0 < args.learning_rate <= .1
            or not all(math.isfinite(value) and value > 0 for value in
                       (args.contrast_positive, args.contrast_negative, args.step_seconds))
            or not all(math.isfinite(value) and value >= 0 for value in
                       (args.rigidity_weight, args.acceleration_weight))):
        parser.error('Invalid training parameters')
    if args.fit_split == 'train' and (args.warmup_steps or args.event_flow_initialization
                                    or args.estimate_event_background or args.rigid_feature_initialization):
        parser.error('Cumulative warmup uses every event: choose all or set warmup-steps=0 for train split')
    if args.propagate_event_flow and not args.event_flow_initialization:
        parser.error('--propagate-event-flow requires --event-flow-initialization')
    if args.rigid_feature_initialization and args.event_flow_initialization:
        parser.error('Select either soft event-flow initialization or rigid features, not both')
    validate_binding_arguments(args)
    if args.geodesic_binding and not args.rigid_feature_initialization:
        parser.error('--geodesic-binding requires --rigid-feature-initialization')
    if args.motion_model != 'se3' and not args.rigid_feature_initialization:
        parser.error('--motion-model camera_plane requires --rigid-feature-initialization')
    if (min(args.feature_max_corners, args.rigid_max_groups) < 1
            or not 0 < args.binding_radius_fraction < 1
            or not math.isfinite(args.feature_reprojection_weight) or args.feature_reprojection_weight < 0):
        parser.error('Invalid rigid feature or binding options')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for full Gaussian motion fitting')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    provenance = preflight_event_only_inputs(
        args.g0, args.events, args.intrinsics, args.extrinsics,
        g0_time=args.g0_time, time_scale=args.time_scale)
    provenance.update({
        'input_files': {name: str(getattr(args, name).resolve())
                        for name in ('g0', 'events', 'intrinsics', 'extrinsics')},
        'identity_and_time_confirmed': args.identity_time_confirmed,
        'scope': 'fresh_event_only_training', 'legacy_data_used': False,
        'unknowns': [] if args.identity_time_confirmed else [
            'Existing contrastive_g0.ply is treated as the requested corrected G0 candidate.',
            'G0 t=0 and the explicitly supplied event time scale have not been independently verified.'],
    })
    camera = provenance['camera']
    windows = build_event_windows(args.events, camera['width'], camera['height'],
                                  g0_time=args.g0_time, time_scale=args.time_scale,
                                  step_seconds=args.step_seconds, substeps=args.substeps)
    times, counts, train_mask = windows['times'], windows['counts'], windows['train_mask']
    if windows['metadata']['input_sha256'] != provenance['input_sha256']['events']:
        raise RuntimeError('Raw events changed between preflight and supervision construction')
    knot_indices = list(range(0, len(times) - 1, args.knot_stride)) + [len(times) - 1]
    renderer = EventOnlyRenderer(args.g0, camera)
    graph = None if args.rigid_feature_initialization else build_graph(renderer.g0.xyz.cpu().numpy(), nodes=args.nodes, seed=args.seed)
    motion = None if graph is None else EventOnlyMotion(graph, times[knot_indices]).cuda()
    frozen_groups = None
    feature_evidence = None
    print(json.dumps({'phase': 'inputs_ready', 'gaussians': renderer.g0.count,
                      'nodes': None if motion is None else len(motion.centers), 'knots': len(knot_indices),
                      'event_windows': len(counts), 'duration_seconds': float(times[-1]),
                      'fit_split': args.fit_split, 'input_sha256': provenance['input_sha256']}), flush=True)
    args.output.mkdir(parents=True)
    write_json(args.output / 'input-provenance.json', provenance)
    write_json(args.output / 'event-window-metadata.json', windows['metadata'])
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output / 'settings.json', settings)
    source_root = Path(__file__).resolve().parents[1]
    code_paths = [Path(__file__).resolve()] + [
        source_root / 'utils' / name for name in (
            'event_only_inputs.py', 'event_only_renderer.py', 'event_only_motion.py',
            'event_only_supervision.py', 'event_only_checkpoint.py', 'event_only_flow_init.py',
            'event_only_radiometry.py', 'event_only_graph_propagation.py',
            'event_only_feature_tracks.py', 'event_only_feature_init.py',
            'event_only_rigid_discovery.py', 'event_only_hard_binding.py', 'event_only_correspondence.py',
            'event_only_geodesic_binding.py', 'event_only_planar_constraint.py')]
    execution_hashes = {str(path.relative_to(source_root)): sha256_file(path)
                        for path in code_paths if path.is_file()}
    write_json(args.output / 'execution-code-sha256.json', execution_hashes)
    for source in code_paths:
        if source.is_file():
            target = args.output / 'source-snapshot' / source.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if (sha256_file(source) != execution_hashes[str(source.relative_to(source_root))]
                    or sha256_file(target) != execution_hashes[str(source.relative_to(source_root))]):
                raise ValueError('Execution source changed during archival')
    # The copy is byte-for-byte G0, not an improved static model or baked motion.
    shutil.copyfile(args.g0, args.output / 'gaussians.ply')
    if sha256_file(args.output / 'gaussians.ply') != provenance['input_sha256']['g0']:
        raise RuntimeError('G0 changed during the copy')
    history = []
    expected = (counts[:, 0].astype(np.float32) * args.contrast_positive
                - counts[:, 1].astype(np.float32) * args.contrast_negative)
    cumulative = np.cumsum(expected, axis=0, dtype=np.float32) if args.fit_split == 'all' else None
    device = renderer.g0.xyz.device
    event_background = 0.
    radiometry = {'method': 'fixed_black_background', 'estimate': 0., 'confirmed': False}

    def log_render(t):
        xyz, rotation = motion.deform(renderer.g0.xyz, renderer.g0.rotation, float(t))
        result = renderer.render(xyz, rotation)
        return observation_log(result, event_background, args.log_floor)

    def enforce_constraints(*, gradients_only=False):
        if args.motion_model == 'camera_plane':
            from utils.event_only_planar_constraint import enforce_camera_plane_motion
            enforce_camera_plane_motion(motion, camera['w2c'], gradients_only=gradients_only)
        enforce_frozen_groups(motion, frozen_groups, gradients_only=gradients_only)

    with torch.no_grad():
        first_result = renderer.render()
        if args.estimate_event_background:
            from utils.event_only_radiometry import infer_background_from_events
            radiometry = infer_background_from_events(
                first_result['gray'].cpu().numpy(),
                first_result['alpha'].squeeze(0).cpu().numpy(), iter(cumulative))
            if radiometry['estimate'] is None:
                write_json(args.output / 'radiometry-estimate.json', radiometry)
                raise ValueError('Insufficient event evidence for background estimation; do not silently guess')
            event_background = float(radiometry['estimate'])
            print(json.dumps({'phase': 'event_background_estimated', 'estimate': event_background,
                              'independently_calibrated': False}), flush=True)
        write_json(args.output / 'radiometry-estimate.json', radiometry)
        reference_log = observation_log(first_result, event_background, args.log_floor)
        reference_cpu = reference_log.cpu().numpy()
        Image.fromarray((first_result['rgb'].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
                        .round().astype(np.uint8)).save(args.output / 'input-g0.png')
    initial_state = None
    initialization_metadata = {'kind': 'identity_zero_motion', 'external_animation_used': False}
    if args.rigid_feature_initialization:
        from utils.event_only_feature_init import infer_feature_groups
        alpha = first_result['alpha'].squeeze(0).cpu().numpy()
        normalized_depth = first_result['depth'].squeeze(0).cpu().numpy() / np.maximum(alpha, 1e-8)
        inferred = infer_feature_groups(
            reference_log.exp().cpu().numpy(), alpha, normalized_depth, cumulative, camera, times, knot_indices,
            raw_event_counts=counts,
            event_background=event_background,
            max_corners=args.feature_max_corners, reference_reacquisition=args.feature_reference_reacquisition,
            feature_method=args.feature_method,
            motion_model=args.motion_model,
            max_groups=args.rigid_max_groups)
        features, discovery = inferred['features'], inferred['discovery']
        initialization_metadata = {'kind': 'event_surface_feature_rigid_initialization_v1',
                                   'external_animation_used': False, 'feature_metadata': features['metadata'],
                                   'rigid_group_metadata': discovery['metadata'],
                                   'static_hypothesis': inferred['static_hypothesis']}
        diagnostics = {'metadata': initialization_metadata, 'features': features['diagnostics'],
                       'groups': discovery['diagnostics']}
        write_json(args.output / 'initialization-diagnostics.json', diagnostics)
        if not discovery['diagnostics']['dynamic_groups']:
            raise ValueError('No multi-time supported rigid groups; cannot invent part motion')
        xyz = renderer.g0.xyz.cpu().numpy()
        try:
            binding = bind_feature_groups(xyz, features, discovery, times[knot_indices], provenance['input_sha256'], args)
        except (ValueError, RuntimeError) as error:
            diagnostics['binding_failure'] = {'type': type(error).__name__, 'message': str(error)}
            write_json(args.output / 'initialization-diagnostics.json', diagnostics)
            raise
        diagnostics['binding'] = binding['diagnostics']
        if not binding['diagnostics']['retained_dynamic_group_count']:
            write_json(args.output / 'initialization-diagnostics.json', diagnostics)
            raise ValueError('No dynamic group survived bounded binding; refusing an all-frozen fit')
        graph = binding['graph']
        motion = EventOnlyMotion(graph, binding['times']).to(device)
        frozen_groups = torch.as_tensor(binding['recommended_frozen_groups'], device=device)
        feature_evidence = make_feature_evidence(features, binding['supervision_seed_groups'],
                                                 binding['recommended_frozen_groups'], camera, times[knot_indices], device)
        initial_state = {name: torch.as_tensor(binding[name], device=device).clone()
                         for name in ('translations', 'rotation_vectors')}
        with torch.no_grad():
            for name, value in initial_state.items():
                getattr(motion, name).copy_(value)
        enforce_constraints()
        initial_state = {name: getattr(motion, name).detach().clone() for name in initial_state}
        initialization_metadata['training_motion_subspace'] = (
            'camera_xy_translation_and_camera_z_rotation_prior_not_verified_3D_motion'
            if args.motion_model == 'camera_plane' else 'unrestricted_SE3')
        if args.motion_model == 'camera_plane':
            from utils.event_only_planar_constraint import camera_plane_prior_metadata
            initialization_metadata['camera_plane_prior'] = camera_plane_prior_metadata(camera)
        diagnostics['binding'] = binding['diagnostics']
        initialization_metadata['binding'] = graph['metadata']
        initialization_metadata['feature_reprojection_prior'] = {
            'weight': args.feature_reprojection_weight, 'dynamic_seed_count': len(feature_evidence['points']),
            'source': 'same_run_G0_and_events_only', 'independent_observation': False,
            'intermediate_times': 'linear_pixel_interpolation_only_if_both_endpoint_identities_accepted'}
        write_json(args.output / 'initialization-diagnostics.json', diagnostics)
        print(json.dumps({'phase': 'event_rigid_features_initialized', 'features': len(features['points']),
                          'dynamic_groups': discovery['diagnostics']['dynamic_groups'],
                          'motion_groups': len(motion.centers), 'frozen_group_count': int(frozen_groups.sum())}), flush=True)
        del inferred, features, discovery, binding
    if args.event_flow_initialization:
        from utils.event_only_flow_init import infer_event_flow_initialization
        cumulative_at_knots = (np.zeros_like(reference_cpu) if index == 0 else cumulative[index - 1]
                               for index in knot_indices)
        initialization = infer_event_flow_initialization(
            reference_log.exp().cpu().numpy(), cumulative_at_knots,
            graph['centers'], camera, times[knot_indices], log_floor=args.log_floor,
            reference_depth_np=first_result['depth'].squeeze().cpu().numpy(),
            reference_alpha_np=first_result['alpha'].squeeze().cpu().numpy())
        if args.propagate_event_flow:
            from utils.event_only_graph_propagation import propagate_graph_motion
            propagated = propagate_graph_motion(
                graph['centers'], initialization['translations'], initialization['initially_visible_mask'],
                initialization['accepted_masks'], edges=graph['edges'],
                static_anchor_mask_np=initialization['static_anchor_mask'])
            for name in ('translations', 'rotation_vectors'):
                initialization[name] = propagated[name]
            initialization['metadata']['graph_propagation'] = propagated['metadata']
            initialization['debug']['graph_propagation'] = propagated['debug']
            initialization['debug']['event_quiet_static_anchor_count'] = int(initialization['static_anchor_mask'].sum())
        initial_state = {name: torch.as_tensor(initialization[name], device=device).clone()
                         for name in ('translations', 'rotation_vectors')}
        with torch.no_grad():
            for name, value in initial_state.items():
                getattr(motion, name).copy_(value)
        initialization_metadata = initialization['metadata']
        write_json(args.output / 'initialization-diagnostics.json',
                   {'metadata': initialization_metadata, 'debug': initialization['debug']})
        # These images are outputs derived ONLY from G0 and raw events. They
        # expose drift/saturation in the proxy, not independent observations.
        for knot in sorted(set(round(fraction * (len(knot_indices) - 1))
                               for fraction in (0, .25, .5, .75, 1))):
            index = knot_indices[knot]
            delta = 0 if index == 0 else cumulative[index - 1]
            proxy = np.exp(np.clip(reference_cpu + delta, -20, 0))
            Image.fromarray(np.rint(proxy * 255).astype(np.uint8)).save(
                args.output / f'derived-event-proxy-knot-{knot:03d}.png')
        print(json.dumps({'phase': 'event_flow_initialized',
                          'accepted_node_steps': initialization['debug']['total_accepted_node_steps'],
                          'max_translation': float(motion.translations.norm(dim=-1).max().detach()),
                          'metadata': initialization_metadata}), flush=True)
        del initialization
    previous_log = reference_log.detach()
    last_trained_knot = 0
    optimized_windows = np.zeros(len(counts), dtype=bool)
    proxy_saturation = []

    def optimizer(rate):
        return torch.optim.Adam([
            {'params': [motion.translations], 'lr': rate * float(motion.scene_scale)},
            {'params': [motion.rotation_vectors], 'lr': rate * 5},
        ])

    # Chronological initialization makes large displacements reachable without
    # any borrowed animation. Only the newly introduced temporal knot is updated.
    warmup_limit = (min(args.stop_after_knots, len(knot_indices) - 1)
                    if args.stop_after_knots else len(knot_indices) - 1)
    if args.warmup_steps:
        for knot in range(1, warmup_limit + 1):
            row, end, begin = knot - 1, knot_indices[knot], knot_indices[knot - 1]
            timestamp = float(times[end])
            seed_chronological_knot(motion, row, initial_state)
            enforce_constraints()
            expected_step = torch.as_tensor(expected[begin:end].sum(axis=0), device=device)
            target_raw = reference_cpu + cumulative[end - 1]
            target = torch.as_tensor(np.clip(target_raw, math.log(args.log_floor), 0), device=device)
            proxy_saturation.append(float(np.mean((target_raw < math.log(args.log_floor)) | (target_raw > 0))))
            previous_t = motion.translations[row - 1].detach() if row else torch.zeros_like(motion.translations[row])
            previous_r = motion.rotation_vectors[row - 1].detach() if row else torch.zeros_like(motion.rotation_vectors[row])

            def knot_objective():
                rendered = log_render(timestamp)
                integral = integral_loss(rendered, target)
                events = difference_loss(rendered - previous_log, expected_step)
                rigidity = motion.spatial_arap(timestamp)
                smooth = ((motion.translations[row] - previous_t) / motion.scene_scale).square().mean()
                smooth = smooth + .1 * (motion.rotation_vectors[row] - previous_r).square().mean()
                correspondence = feature_reprojection_loss(motion, timestamp, feature_evidence)
                return (integral + .15 * events + args.rigidity_weight * rigidity + .1 * smooth
                        + args.feature_reprojection_weight * correspondence,
                        integral, events, rigidity, smooth, correspondence)

            selected_flow_scale = None
            if initial_state is not None:
                # A flow estimate is a proposal, never a measured 3D trajectory.
                # Always retain the no-flow alternative, and choose using the
                # same actual Gaussian-render/event objective as optimization.
                with torch.no_grad():
                    proposed_t = motion.translations[row].clone() - previous_t
                    proposed_r = motion.rotation_vectors[row].clone() - previous_r
                    proposal_loss = float('inf')
                    for scale in (0., .25, .5, 1.):
                        motion.translations[row].copy_(previous_t + scale * proposed_t)
                        motion.rotation_vectors[row].copy_(previous_r + scale * proposed_r)
                        score = float(knot_objective()[0])
                        if score < proposal_loss:
                            proposal_loss, selected_flow_scale = score, scale
                    if selected_flow_scale is None:
                        raise ValueError('All event-flow proposals have non-finite losses')
                    motion.translations[row].copy_(previous_t + selected_flow_scale * proposed_t)
                    motion.rotation_vectors[row].copy_(previous_r + selected_flow_scale * proposed_r)
            opt = optimizer(args.learning_rate)
            best_loss, first_loss, best_t, best_r = float('inf'), None, None, None
            for iteration in range(args.warmup_steps + 1):
                decay = .2 + .8 * (1 - iteration / max(1, args.warmup_steps))
                opt.param_groups[0]['lr'] = args.learning_rate * float(motion.scene_scale) * decay
                opt.param_groups[1]['lr'] = args.learning_rate * 5 * decay
                opt.zero_grad(set_to_none=True)
                # Prefix-only regularity: never penalize the initialized prefix
                # for not matching the still-untrained future knots.
                loss, integral, events, rigidity, smooth, correspondence = knot_objective()
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite chronological fitting loss')
                score = float(loss.detach())
                if first_loss is None:
                    first_loss = score
                if score < best_loss:
                    best_loss, best_t, best_r = score, motion.translations[row].detach().clone(), motion.rotation_vectors[row].detach().clone()
                if iteration == args.warmup_steps:
                    break
                loss.backward()
                restrict_gradient_to_knot(motion, row)
                enforce_constraints(gradients_only=True)
                torch.nn.utils.clip_grad_norm_(motion.parameters(), 5.)
                opt.step()
                enforce_constraints()
            with torch.no_grad():
                motion.translations[row].copy_(best_t)
                motion.rotation_vectors[row].copy_(best_r)
                previous_log = log_render(timestamp).detach()
            last_trained_knot = knot
            optimized_windows[begin:end] = True
            record = {'phase': 'chronological', 'knot': knot, 'total_knots': len(knot_indices) - 1,
                      'time_seconds': timestamp, 'first_loss': first_loss, 'best_loss': best_loss,
                      'selected_event_flow_scale': selected_flow_scale,
                      'last_iteration_event_integral_loss': float(integral.detach()),
                      'last_iteration_event_pair_loss': float(events.detach()),
                      'last_iteration_spatial_arap': float(rigidity.detach()),
                      'last_iteration_feature_reprojection_prior': float(correspondence.detach()),
                      'seconds_elapsed': time.perf_counter() - started,
                      'peak_cuda_mb': torch.cuda.max_memory_allocated() / 1024 ** 2}
            history.append(record)
            print(json.dumps(record), flush=True)
            write_json(args.output / 'training-progress.json', {'history': history, 'whole_timeline_fitted': False})
            del loss, integral, events, rigidity, smooth, correspondence, opt
            if knot % 20 == 0:
                save_event_only_checkpoint(args.output / f'deform-knot-{knot:03d}.pth', motion, graph,
                                           camera, provenance, {'history': history, 'trained_until_seconds': timestamp,
                                                                'whole_timeline_fitted': False,
                                                                'fit_split': args.fit_split,
                                                                'thresholds': [args.contrast_positive, args.contrast_negative],
                                                                'thresholds_confirmed': args.contrast_confirmed,
                                                                'identity_and_time_confirmed': args.identity_time_confirmed,
                                                                'event_background': event_background,
                                                                'event_radiometry': radiometry,
                                                                'log_intensity_floor': args.log_floor,
                                                                'initialization': initialization_metadata})

    # Joint refinement evaluates true endpoint log differences. In train-only
    # mode no pair spans a held-out window, and no cumulative proxy is created.
    fitting_mask = np.ones(len(counts), dtype=bool) if args.fit_split == 'all' else train_mask.copy()
    if args.stop_after_knots:
        fitting_mask[knot_indices[warmup_limit]:] = False
    spans = training_spans(fitting_mask, max_length=args.knot_stride)
    rng = np.random.default_rng(args.seed)
    single_order = rng.permutation(np.flatnonzero(fitting_mask))
    individually_optimized = np.zeros(len(counts), dtype=bool)
    opt = optimizer(args.learning_rate * .25)
    for iteration in range(args.polish_steps):
        if iteration < len(single_order):
            begin = int(single_order[iteration])
            end = begin + 1
        else:
            begin, end = spans[int(rng.integers(len(spans)))]
        opt.zero_grad(set_to_none=True)
        first, second = log_render(times[begin]), log_render(times[end])
        target_delta = torch.as_tensor(expected[begin:end].sum(axis=0), device=device)
        events = difference_loss(second - first, target_delta)
        anchor = (integral_loss(second, torch.as_tensor(
            np.clip(reference_cpu + cumulative[end - 1], math.log(args.log_floor), 0), device=device))
            if args.fit_split == 'all' else events * 0)
        rigidity = motion.spatial_arap(float(times[end]))
        smooth = motion.temporal_smoothness()
        correspondence = feature_reprojection_loss(motion, float(times[end]), feature_evidence)
        loss = (events + .5 * anchor + args.rigidity_weight * rigidity + args.acceleration_weight * smooth
                + args.feature_reprojection_weight * correspondence)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite joint event optimization loss')
        loss.backward()
        enforce_constraints(gradients_only=True)
        torch.nn.utils.clip_grad_norm_(motion.parameters(), 5.)
        opt.step()
        enforce_constraints()
        optimized_windows[begin:end] = True
        if end == begin + 1:
            individually_optimized[begin] = True
        if iteration % 25 == 0 or iteration + 1 == args.polish_steps:
            record = {'phase': 'joint_events', 'iteration': iteration + 1,
                      'loss': float(loss.detach()), 'event_loss': float(events.detach()),
                      'event_integral_loss': float(anchor.detach()), 'spatial_arap': float(rigidity.detach()),
                      'temporal_acceleration': float(smooth.detach()),
                      'feature_reprojection_prior': float(correspondence.detach()),
                      'seconds_elapsed': time.perf_counter() - started,
                      'peak_cuda_mb': torch.cuda.max_memory_allocated() / 1024 ** 2}
            history.append(record)
            print(json.dumps(record), flush=True)
            write_json(args.output / 'training-progress.json', {'history': history, 'whole_timeline_fitted': False})

    whole_timeline = bool(not args.stop_after_knots and fitting_mask.any()
                          and optimized_windows[fitting_mask].all())
    summary = {
        'format': 'event_only_training_summary_v1',
        'training_performed': bool(optimized_windows.any()),
        'whole_timeline_fitted': whole_timeline, 'duration_seconds': float(times[-1]),
        'explicitly_optimized_window_count': int(optimized_windows.sum()),
        'individually_optimized_native_window_count': int(individually_optimized.sum()),
        'unoptimized_fitting_windows': np.flatnonzero(fitting_mask & ~optimized_windows).tolist(),
        'fit_split': args.fit_split,
        'evaluation_scope': 'in_sample_consistency_only' if args.fit_split == 'all' else 'time_window_holdout',
        'thresholds': [args.contrast_positive, args.contrast_negative],
        'thresholds_confirmed': args.contrast_confirmed,
        'initialization': initialization_metadata,
        'frozen_group_count': 0 if frozen_groups is None else int(frozen_groups.sum()),
        'hard_rigid_group_binding': args.rigid_feature_initialization,
        'motion_model': args.motion_model,
        'event_radiometry': radiometry, 'event_background': event_background,
        'log_intensity_floor': args.log_floor,
        'g0_appearance_frozen': True, 'all_gaussians_preserved': True,
        'identity_and_time_confirmed': args.identity_time_confirmed,
        'event_integral_proxy_saturation_fraction': proxy_saturation,
        'history': history, 'gpu': torch.cuda.get_device_name(),
        'peak_cuda_mb': torch.cuda.max_memory_allocated() / 1024 ** 2,
        'elapsed_seconds_before_audit': time.perf_counter() - started,
        'limitations': [
            'Event-integrated training images are derived proxies, not observed RGB.',
            'Estimated background radiometry is a G0/event-derived nuisance parameter, not an independently calibrated observation.',
            'Uncalibrated contrast thresholds and assumed time alignment limit correctness.',
            'Single fixed view does not identify unique 3D motion or unseen geometry.',
            'The learned deformation graph is not a known rigid-part or joint mechanism.',
        ],
    }
    save_event_only_checkpoint(args.output / 'deform.pth', motion, graph, camera, provenance, summary)
    write_json(args.output / 'training-summary.json', summary)
    predicted = np.empty((len(counts), camera['height'], camera['width']), dtype=np.float32)
    with torch.no_grad():
        previous = log_render(times[0])
        for index in range(len(counts)):
            current = log_render(times[index + 1])
            predicted[index] = (current - previous).cpu().numpy()
            previous = current
            if index % 40 == 0:
                print(json.dumps({'phase': 'full_event_audit', 'window': index, 'windows': len(counts)}), flush=True)
    audit = audit_event_prediction(
        predicted, counts, train_mask=train_mask, fit_split=args.fit_split,
        threshold_positive=args.contrast_positive, threshold_negative=args.contrast_negative,
        thresholds_confirmed=args.contrast_confirmed)
    audit.update({'whole_timeline_fitted': whole_timeline, 'input_sha256': provenance['input_sha256'],
                  'checkpoint_sha256': sha256_file(args.output / 'deform.pth'),
                  'event_background': event_background, 'log_intensity_floor': args.log_floor,
                  'identity_and_time_confirmed': args.identity_time_confirmed})
    if not whole_timeline or not args.identity_time_confirmed:
        audit['approved_for_export'] = False
        audit['status'] = 'candidate_unapproved'
        if not whole_timeline:
            audit['approval_blockers'].append('not_all_fitting_windows_optimized')
        if not args.identity_time_confirmed:
            audit['approval_blockers'].append('g0_identity_or_time_unconfirmed')
    write_json(args.output / 'event-audit.json', audit)
    print(json.dumps({'phase': 'finished', 'output': str(args.output),
                      'seconds_elapsed': time.perf_counter() - started,
                      'approved_for_export': audit.get('approved_for_export', False)}), flush=True)


if __name__ == '__main__':
    main()
