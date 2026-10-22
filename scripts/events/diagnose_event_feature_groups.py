"""Fresh G0 surface-feature / raw-event rigid-motion initialization diagnostic.

Reads only four explicitly supplied raw inputs. Output images/tracks are
derived hypotheses, not additional RGB observations or ground-truth parts.
Does not load old trajectories, models, bindings, Blender, or video.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    from utils.event_only_feature_init import add_binding_arguments, validate_binding_arguments, bind_feature_groups
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('g0', 'events', 'intrinsics', 'extrinsics', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--g0-time', type=float, required=True)
    parser.add_argument('--time-scale', type=float, required=True)
    parser.add_argument('--identity-time-confirmed', action='store_true')
    parser.add_argument('--knot-stride', type=int, default=4)
    parser.add_argument('--max-corners', type=int, default=2048)
    parser.add_argument('--feature-method', choices=('corners', 'sift'), default='corners')
    parser.add_argument('--motion-model', choices=('se3', 'camera_plane'), default='se3')
    parser.add_argument('--reference-reacquisition', action='store_true')
    parser.add_argument('--max-groups', type=int, default=6)
    add_binding_arguments(parser)
    parser.add_argument('--make-binding', action='store_true')
    parser.add_argument('--score-binding-photometry', action='store_true',
                        help='Diagnostic only: compare candidate motion using same-input event photometry')
    parser.add_argument('--score-binding-photometry-pairs', action='store_true',
                        help='Separate identity-vs-dynamic comparisons; different time-set scores cannot be mixed')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.knot_stride, args.max_corners, args.max_groups) < 1:
        parser.error('Counts must be positive')
    validate_binding_arguments(args)
    import numpy as np
    import torch
    from PIL import Image
    from utils.event_only_inputs import preflight_event_only_inputs, sha256_file
    from utils.event_only_renderer import EventOnlyRenderer
    from utils.event_only_supervision import build_event_windows
    from utils.event_only_radiometry import infer_background_from_events
    from utils.event_only_feature_init import infer_feature_groups
    from scripts.events.reconstruct_lego_event_only import observation_log, write_json

    started = time.perf_counter()
    provenance = preflight_event_only_inputs(args.g0, args.events, args.intrinsics, args.extrinsics,
                                            g0_time=args.g0_time, time_scale=args.time_scale)
    provenance.update(identity_and_time_confirmed=args.identity_time_confirmed,
                      source_authenticity_verified=False, legacy_data_used=False,
                      identity_confirmation_source='user_confirmation' if args.identity_time_confirmed else None)
    camera = provenance['camera']
    windows = build_event_windows(args.events, camera['width'], camera['height'],
                                  g0_time=args.g0_time, time_scale=args.time_scale,
                                  step_seconds=.05, substeps=2)
    if windows['metadata']['input_sha256'] != provenance['input_sha256']['events']:
        raise ValueError('Raw events changed during preparation')
    counts, times = windows['counts'], windows['times']
    cumulative = np.cumsum(.2 * counts[:, 0].astype(np.float32) - .2 * counts[:, 1].astype(np.float32), axis=0)
    indices = list(range(0, len(times) - 1, args.knot_stride)) + [len(times) - 1]
    renderer = EventOnlyRenderer(args.g0, camera)
    with torch.no_grad():
        rendered = renderer.render()
        gray = rendered['gray'].cpu().numpy()
        alpha = rendered['alpha'].squeeze(0).cpu().numpy()
        depth = rendered['depth'].squeeze(0).cpu().numpy() / np.maximum(alpha, 1e-8)
        radiometry = infer_background_from_events(gray, alpha, iter(cumulative))
        if radiometry['estimate'] is None:
            raise ValueError('No supported background estimate')
        reference = observation_log(rendered, radiometry['estimate'], .0001).exp().cpu().numpy()
    args.output.mkdir(parents=True)
    write_json(args.output / 'input-provenance.json', provenance)
    print(json.dumps({'phase': 'tracking', 'knots': len(indices), 'gaussians': renderer.g0.count}), flush=True)
    inferred = infer_feature_groups(reference, alpha, depth, cumulative, camera, times, indices,
                                    raw_event_counts=counts,
                                    event_background=radiometry['estimate'],
                                    max_corners=args.max_corners,
                                    feature_method=args.feature_method,
                                    motion_model=args.motion_model,
                                    reference_reacquisition=args.reference_reacquisition, max_groups=args.max_groups)
    features, discovery, static = inferred['features'], inferred['discovery'], inferred['static_prior_mask']
    report = {'input_provenance': provenance, 'fit_split': 'all', 'thresholds': [.2, .2],
              'thresholds_confirmed': False, 'radiometry': radiometry,
              'feature_metadata': features['metadata'], 'feature_diagnostics': features['diagnostics'],
              'metadata': discovery['metadata'], 'diagnostics': discovery['diagnostics'],
              'static_hypothesis': inferred['static_hypothesis'],
              'approved_for_export': False, 'training_performed': False,
              'all_input_modalities': ['g0', 'events', 'camera']}
    source_root = Path(__file__).resolve().parents[2]
    code_names = ['scripts/events/diagnose_event_feature_groups.py', 'scripts/events/reconstruct_lego_event_only.py'] + [
        'utils/event_only_' + name + '.py' for name in ('inputs', 'renderer', 'supervision', 'motion',
                                                       'feature_tracks', 'feature_init', 'radiometry', 'rigid_discovery',
                                                       'hard_binding', 'geodesic_binding', 'correspondence', 'checkpoint',
                                                       'binding_unaries', 'planar_constraint')]
    report['execution_source_sha256'] = {name: sha256_file(source_root / name) for name in code_names
                                        if (source_root / name).is_file()}
    for name, expected_hash in report['execution_source_sha256'].items():
        source, target = source_root / name, args.output / 'source-snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(source, target)
        if sha256_file(target) != expected_hash:
            raise ValueError('Execution source changed during archival')
    with (args.output / 'derived-tracks-and-poses.npz').open('xb') as stream:
        np.savez_compressed(stream, points=features['points'], tracks=features['tracks'],
                            accepted=features['accepted'], tracking_continuous=features['tracking_continuous'],
                            times=times[indices], pose_times=times[indices][1:], labels=discovery['labels'], poses=discovery['poses'],
                            pose_valid=discovery['pose_valid'], static_prior_mask=static)
    for index in (0, len(indices) // 4, len(indices) // 2, 3 * len(indices) // 4, len(indices) - 1):
        end = indices[index]
        proxy = reference if not end else np.exp(np.clip(np.log(np.maximum(reference, .0001)) + cumulative[end - 1], -20, 0))
        Image.fromarray(np.rint(proxy * 255).astype(np.uint8)).save(args.output / f'derived-proxy-{index:03d}.png')
    if (args.score_binding_photometry or args.score_binding_photometry_pairs) and discovery['diagnostics']['dynamic_groups']:
        from utils.event_only_binding_unaries import score_binding_event_unaries
        poses_with_zero = np.concatenate((np.broadcast_to(np.eye(4), (1,) + discovery['poses'].shape[1:]),
                                          discovery['poses']))
        valid_with_zero = np.concatenate((np.zeros((1, discovery['pose_valid'].shape[1]), dtype=bool),
                                          discovery['pose_valid']))
        comparisons = [('binding_photometry', list(range(poses_with_zero.shape[1])))] if args.score_binding_photometry else []
        if args.score_binding_photometry_pairs:
            comparisons += [(f'binding_photometry_pair_0_{group}', [0, group])
                            for group in range(1, poses_with_zero.shape[1])]
        for name, columns in comparisons:
            unaries = score_binding_event_unaries(
                np.log(np.maximum(reference, .0001)), alpha, depth,
                (np.zeros_like(reference) if index == 0 else cumulative[index - 1] for index in indices),
                camera, times[indices], poses_with_zero[:, columns], valid_with_zero[:, columns],
                background=radiometry['estimate'], log_floor=.0001)
            report[name] = {key: value for key, value in unaries.items() if not isinstance(value, np.ndarray)}
            report[name].update(candidate_source_labels=columns, cross_comparison_scores_comparable=False,
                                binding_modified=False)
            with (args.output / f'derived-{name}.npz').open('xb') as stream:
                np.savez_compressed(stream, **{key: value for key, value in unaries.items()
                                              if isinstance(value, np.ndarray)})
    if args.make_binding and discovery['diagnostics']['dynamic_groups']:
        from utils.event_only_motion import EventOnlyMotion
        from utils.event_only_checkpoint import save_event_only_checkpoint
        xyz = renderer.g0.xyz.cpu().numpy()
        try:
            binding = bind_feature_groups(xyz, features, discovery, times[indices], provenance['input_sha256'], args)
        except (ValueError, RuntimeError) as error:
            report['binding_failure'] = {'type': type(error).__name__, 'message': str(error)}
            report['elapsed_seconds'] = time.perf_counter() - started
            write_json(args.output / 'feature-rigid-discovery.json', report)
            raise
        report['binding'] = binding['diagnostics']
        if not binding['diagnostics']['retained_dynamic_group_count']:
            write_json(args.output / 'feature-rigid-discovery.json', report)
            raise ValueError('No dynamic group retained after bounded G0 binding')
        motion = EventOnlyMotion(binding['graph'], binding['times'])
        with torch.no_grad():
            motion.translations.copy_(torch.as_tensor(binding['translations']))
            motion.rotation_vectors.copy_(torch.as_tensor(binding['rotation_vectors']))
        summary = {'training_performed': False, 'whole_timeline_fitted': False,
                   'fit_split': 'all', 'thresholds': [.2, .2], 'thresholds_confirmed': False,
                   'identity_and_time_confirmed': args.identity_time_confirmed,
                   'event_background': radiometry['estimate'], 'log_intensity_floor': .0001,
                   'initialization_only': True, 'initialization': features['metadata'],
                   'binding': binding['diagnostics'], 'approved_for_export': False}
        save_event_only_checkpoint(args.output / 'initialization-only.pth', motion, binding['graph'], camera, provenance, summary)
        report['binding'] = binding['diagnostics']
    report['elapsed_seconds'] = time.perf_counter() - started
    write_json(args.output / 'feature-rigid-discovery.json', report)
    print(json.dumps({'output': str(args.output), 'points': len(features['points']),
                      'groups': discovery['diagnostics']['dynamic_groups'],
                      'group_node_counts': [group['nodes'] for group in discovery['diagnostics']['groups']],
                      'valid_pose_counts': discovery['pose_valid'].sum(axis=0).tolist(),
                      'seconds': report['elapsed_seconds'], 'approved_for_export': False}), flush=True)


if __name__ == '__main__':
    main()
