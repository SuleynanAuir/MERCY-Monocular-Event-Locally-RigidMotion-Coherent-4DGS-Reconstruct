"""Test rigid-motion identifiability using only explicit G0/events/K/R/T.

This produces inferred correspondence/pose diagnostics, not an approved
reconstruction. Unknown node labels and invalid poses are retained.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('g0', 'events', 'intrinsics', 'extrinsics', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--g0-time', type=float, required=True)
    parser.add_argument('--time-scale', type=float, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    import torch
    from utils.event_only_inputs import preflight_event_only_inputs, sha256_file
    from utils.event_only_renderer import EventOnlyRenderer
    from utils.event_only_supervision import build_event_windows
    from utils.event_only_motion import build_graph
    from utils.event_only_flow_init import infer_event_flow_initialization
    from utils.event_only_radiometry import infer_background_from_events
    from utils.event_only_rigid_discovery import discover_rigid_groups
    from scripts.events.reconstruct_lego_event_only import observation_log

    provenance = preflight_event_only_inputs(args.g0, args.events, args.intrinsics, args.extrinsics,
                                            g0_time=args.g0_time, time_scale=args.time_scale)
    camera = provenance['camera']
    windows = build_event_windows(args.events, camera['width'], camera['height'],
                                  g0_time=args.g0_time, time_scale=args.time_scale,
                                  step_seconds=.05, substeps=2)
    if windows['metadata']['input_sha256'] != provenance['input_sha256']['events']:
        raise ValueError('Raw events changed during input preparation')
    counts, times = windows['counts'], windows['times']
    expected = .2 * counts[:, 0].astype(np.float32) - .2 * counts[:, 1].astype(np.float32)
    cumulative = np.cumsum(expected, axis=0)
    indices = list(range(0, len(times) - 1, 4)) + [len(times) - 1]
    renderer = EventOnlyRenderer(args.g0, camera)
    graph = build_graph(renderer.g0.xyz.cpu().numpy(), nodes=256, seed=0)
    with torch.no_grad():
        rendered = renderer.render()
        radiometry = infer_background_from_events(rendered['gray'].cpu().numpy(),
                                                   rendered['alpha'].cpu().numpy(), iter(cumulative))
        if radiometry['estimate'] is None:
            raise ValueError('No supported background estimate')
        reference = observation_log(rendered, radiometry['estimate'], .0001).exp().cpu().numpy()
        initialized = infer_event_flow_initialization(
            reference, (np.zeros_like(reference) if i == 0 else cumulative[i - 1] for i in indices),
            graph['centers'], camera, times[indices], log_floor=.0001,
            reference_depth_np=rendered['depth'].squeeze(0).cpu().numpy(),
            reference_alpha_np=rendered['alpha'].squeeze(0).cpu().numpy())
    world = graph['centers'][None] + initialized['translations']
    w2c = np.asarray(camera['w2c'])
    camera_points = world @ w2c[:3, :3].T + w2c[:3, 3]
    depth = camera_points[..., 2]
    tracks = np.stack((camera['fx'] * camera_points[..., 0] / np.maximum(depth, 1e-8) + camera['cx'],
                       camera['fy'] * camera_points[..., 1] / np.maximum(depth, 1e-8) + camera['cy']), axis=-1)
    discovery = discover_rigid_groups(graph['centers'], tracks, initialized['accepted_masks'], camera,
                                      static_anchor_mask_np=initialized['static_anchor_mask'], max_groups=5)
    args.output.mkdir(parents=True)
    report = {'input_provenance': provenance, 'fit_split': 'all', 'thresholds': [.2, .2],
              'thresholds_confirmed': False, 'radiometry': radiometry,
              'flow_initialization': {'metadata': initialized['metadata'], 'debug': initialized['debug']},
              'metadata': discovery['metadata'], 'diagnostics': discovery['diagnostics'],
              'approved_for_export': False, 'training_performed': False,
              'all_input_modalities': ['g0', 'events', 'camera']}
    source_root = Path(__file__).resolve().parents[2]
    report['execution_source_sha256'] = {
        name: sha256_file(source_root / name) for name in (
            'scripts/events/diagnose_event_rigid_groups.py', 'scripts/events/reconstruct_lego_event_only.py',
            'utils/event_only_inputs.py', 'utils/event_only_renderer.py',
            'utils/event_only_supervision.py', 'utils/event_only_motion.py',
            'utils/event_only_flow_init.py', 'utils/event_only_radiometry.py',
            'utils/event_only_rigid_discovery.py')}
    with (args.output / 'rigid-discovery.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    with (args.output / 'inferred-node-poses.npz').open('xb') as stream:
        np.savez_compressed(stream, centers=graph['centers'], times=times[indices][1:],
                            labels=discovery['labels'], poses=discovery['poses'],
                            pose_valid=discovery['pose_valid'])
    print(json.dumps({'output': str(args.output),
                      'groups': discovery['diagnostics']['dynamic_groups'],
                      'unknown_nodes': discovery['diagnostics']['unknown_nodes'],
                      'group_node_counts': [group['nodes'] for group in discovery['diagnostics']['groups']],
                      'valid_pose_counts': discovery['pose_valid'].sum(axis=0).tolist(),
                      'approved_for_export': False}), flush=True)


if __name__ == '__main__':
    main()
