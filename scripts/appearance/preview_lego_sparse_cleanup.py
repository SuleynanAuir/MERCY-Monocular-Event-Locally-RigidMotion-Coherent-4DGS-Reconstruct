"""Preview actual static-Gaussian attribute changes; never modify the input model.

This diagnostic is not an audited export. Thin mechanical structures, all
baked motion and the compact completion suffix are protected from pruning.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from utils.rigid_checkpoint import sha256_file


def protected_selection(indices, total, retained, dynamic):
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= retained)
            or np.any(indices[1:] <= indices[:-1]) or retained >= total
            or np.intersect1d(indices, dynamic).size):
        raise ValueError('Cleanup must select sorted unique retained-static rows only')
    return indices.astype(np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'selection', 'masks', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--keys', nargs='*', default=['conservative_indices', 'balanced_indices'])
    p.add_argument('--color-overlay', type=Path)
    p.add_argument('--views', nargs='+', type=int, default=[4, 6, 17, 24, 28, 32, 35])
    p.add_argument('--opacity-factor', type=float, default=0.)
    args = p.parse_args()
    if args.output.exists() or not 0 <= args.opacity_factor < 1:
        raise ValueError('Use a new output directory and an opacity factor in [0,1)')
    from scripts.events.export_lego_event_preview import verify_candidate
    from scripts.events.refine_lego_events import MotionRenderer
    from scripts.geometry.render_lego_multiview import verify_inputs
    from utils.event_refinement import correction_from_record
    from gaussian_renderer import render
    from render_ply import make_camera
    from PIL import Image
    import torch

    cache, correction_record, _ = verify_candidate(args.baseline, args.candidate, allow_unapproved=True)
    _, _, mask_meta, _, _, hashes, _, _ = verify_inputs(args.baseline, args.candidate, args.masks)
    report = json.loads(Path(cache['multiview_completion']['report']).read_text())
    parent_report = json.loads(Path(report['parent_report']).read_text())
    retained = parent_report['retained_gaussian_count']
    dynamic = np.load(args.baseline/'dynamic_indices.npy', allow_pickle=False)
    with np.load(args.selection, allow_pickle=False) as archive:
        variants = {key: protected_selection(archive[key], cache['gaussian_count'], retained, dynamic)
                    for key in args.keys}
        metadata = json.loads(str(archive['metadata_json']))
    # Diagnostic selections must be attached to this exact cloud, not row IDs
    # from another rebuild. Accept the diagnostic's explicit hash map only.
    selection_hashes = metadata.get('input_sha256', metadata.get('artifact_sha256', {}))
    source_hash = sha256_file(cache['source'])
    if source_hash != selection_hashes.get('source_ply'):
        raise ValueError('Selection does not identify this exact source PLY')
    colors = None
    if args.color_overlay:
        with np.load(args.color_overlay, allow_pickle=False) as archive:
            rows = protected_selection(archive['indices'], cache['gaussian_count'], retained, dynamic)
            colors = archive['dc_gray']
            color_meta = json.loads(str(archive['metadata_json']))
        if (colors.dtype != np.float32 or colors.shape != rows.shape or not np.isfinite(colors).all()
                or np.any(colors < 0) or np.any(colors > 1) or color_meta.get('source_sha256') != source_hash):
            raise ValueError('Invalid source-specific static color overlay')
        variants['surface_color_prior'] = rows
    views = [mask_meta['views'][i] for i in args.views]
    if len(set(args.views)) != len(args.views) or any(i < 0 for i in args.views):
        raise ValueError('Use distinct nonnegative view indices')
    renderer = MotionRenderer(args.baseline, 640)
    correction = correction_from_record(correction_record, 'cuda')
    original = renderer.model._opacity.detach().clone()
    original_dc = renderer.model._features_dc.detach().clone()
    renderer.model._opacity = original.clone()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'format': 'lego_sparse_cleanup_preview_v1', 'diagnostic_only': True,
        'baseline': str(args.baseline), 'candidate': str(args.candidate),
        'source_ply_sha256': source_hash, 'selection_sha256': sha256_file(args.selection),
        'masks_sha256': hashes['masks'], 'views': views, 'opacity_factor': args.opacity_factor,
        'selected_counts': {key: len(value) for key, value in variants.items()},
        'color_overlay_sha256': sha256_file(args.color_overlay) if args.color_overlay else None,
        'scope': 'Actual Gaussian opacity or DC prior modification in memory, not image masking. '
                 'No input PLY or motion is changed. Requires full event and multiview validation.',
        'samples': []}
    all_arrays = {}
    with torch.no_grad():
        for key, indices in {'original': np.empty(0, np.int64), **variants}.items():
            renderer.model._opacity.copy_(original)
            renderer.model._features_dc = original_dc.clone()
            rows = torch.from_numpy(indices).cuda()
            if key == 'surface_color_prior':
                dc = (torch.from_numpy(colors).cuda()-.5)/.28209479177387814
                renderer.model._features_dc[rows] = dc[:, None, None].expand(-1, 1, 3)
            elif len(indices):
                value = (original[rows].sigmoid()*args.opacity_factor).clamp_min(1e-13)
                renderer.model._opacity[rows] = torch.logit(value)
            gray, alpha = [], []
            for view in views:
                renderer.image(view['frame'], correction)
                camera = make_camera(np.asarray(view['camera_pose']), 640, 640,
                                     np.deg2rad(cache['camera']['fovy_degrees']))
                rendered = render(camera, renderer.model, renderer.pipe, renderer.background, 0., 0., 0.)
                image = rendered['render'].clamp(0, 1)
                g = (image*renderer.luma).sum(0).cpu().numpy()
                a = rendered['alpha'].reshape(640, 640).cpu().numpy()
                if not np.isfinite(g).all() or not np.isfinite(a).all():
                    raise ValueError('Nonfinite preview')
                gray.append(g); alpha.append(a)
                name = f'{key}-view-{view["view_index"]:02d}.png'
                Image.fromarray(np.rint(image.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)).save(args.output/name)
                result['samples'].append({'variant': key, 'view_index': view['view_index'], 'image': name})
                print(json.dumps({'variant': key, 'view': view['view_index'], 'selected': len(indices)}), flush=True)
            all_arrays[key+'_gray'] = np.asarray(gray, np.float32)
            all_arrays[key+'_alpha'] = np.asarray(alpha, np.float32)
    if sha256_file(cache['source']) != source_hash:
        raise ValueError('Source changed during preview')
    np.savez_compressed(args.output/'preview.npz', **all_arrays,
                        metadata_json=np.asarray(json.dumps(result, allow_nan=False)))
    (args.output/'preview.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
