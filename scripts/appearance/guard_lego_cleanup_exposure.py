"""Protect every original-camera time when proposing static splat cleanup.

Candidates come from a separate geometric diagnosis. Only those with low
actual raster alpha contribution at all 400 source times are proposed for
attenuation. This is not proof of RGB invariance: final RGB/event checks are
mandatory, including early-termination effects in the rasterizer.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.events.export_lego_event_preview import verify_candidate
from scripts.appearance.preview_lego_sparse_cleanup import protected_selection
from scripts.geometry.prune_lego_completion_exposure import contribution_from_dc_gradient
from utils.event_refinement import correction_from_record
from utils.rigid_checkpoint import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'selection', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--key', default='surface_outlier_indices')
    p.add_argument('--threshold', type=float, default=.0005)
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError(args.output)
    if not 0 < args.threshold <= .01 or args.output.suffix != '.npz':
        raise ValueError('Require a new NPZ and a bounded pixel-contribution threshold')
    cache, record, audit = verify_candidate(args.baseline, args.candidate, allow_unapproved=True)
    if cache['frame_count'] != 400 or cache.get('gaussian_cleanup'):
        raise ValueError('Expected the original 400-frame completion model without cleanup')
    report = json.loads(Path(cache['multiview_completion']['report']).read_text())
    parent = json.loads(Path(report['parent_report']).read_text())
    dynamic = np.load(args.baseline/'dynamic_indices.npy', allow_pickle=False)
    with np.load(args.selection, allow_pickle=False) as archive:
        indices = protected_selection(archive[args.key], cache['gaussian_count'], parent['retained_gaussian_count'], dynamic)
        selection_meta = json.loads(str(archive['metadata_json']))
        original_variants = {key: archive[key] for key in ('conservative_indices', 'balanced_indices') if key in archive}
    selection_hashes = selection_meta.get('input_sha256', selection_meta.get('artifact_sha256', {}))
    if cache['source_sha256'] != selection_hashes.get('source_ply'):
        raise ValueError('Diagnosis source checksum differs')
    from scripts.events.refine_lego_events import MotionRenderer
    torch.set_num_threads(2)
    renderer = MotionRenderer(args.baseline, 640)
    correction = correction_from_record(record, 'cuda').eval()
    for value in correction.parameters():
        value.requires_grad_(False)
    rows = torch.from_numpy(indices).cuda()
    candidate_dc = torch.zeros((len(indices), 1, 3), device='cuda', requires_grad=True)
    fixed_dc = torch.zeros_like(renderer.model._features_dc)
    maxima = torch.zeros(len(indices), device='cuda')
    for frame in range(400):
        candidate_dc.grad = None
        renderer.model._features_dc = fixed_dc.index_copy(0, rows, candidate_dc)
        image = renderer.image(frame, correction)
        if not torch.isfinite(image).all() or torch.any(image < -1e-7) or torch.any(image > .50001):
            raise ValueError('Invalid uniform-color alpha probe')
        image.sum().backward()
        maxima = torch.maximum(maxima, contribution_from_dc_gradient(candidate_dc.grad).detach())
        if frame % 40 == 0 or frame == 399:
            print(json.dumps({'frame': frame, 'source_frames': 400, 'candidates': len(indices),
                              'protected_so_far': int(torch.count_nonzero(maxima > args.threshold))}), flush=True)
    maxima = maxima.cpu().numpy()
    kept = indices[maxima <= args.threshold].astype(np.int32)
    metadata = {'format': 'lego_cleanup_original_camera_guard_v1',
        'frames_checked': 400, 'original_camera': cache['camera'],
        'candidates': len(indices), 'selected_count': len(kept),
        'threshold_output_pixels': args.threshold,
        'maximum_selected_contribution': float(maxima[maxima <= args.threshold].max()) if len(kept) else 0.,
        'sum_selected_maximum_contribution': float(maxima[maxima <= args.threshold].sum(dtype=np.float64)),
        'input_selection_key': args.key, 'input_selection_metadata': selection_meta,
        'input_sha256': {'source_ply': cache['source_sha256'],
                        'baseline_cache': sha256_file(args.baseline/'cache.json'),
                        'correction': sha256_file(args.candidate/'correction.json'),
                        'event_audit': sha256_file(args.candidate/'event-audit.json'),
                        'diagnosis': sha256_file(args.selection)},
        'scope': 'Potential cleanup only: original-camera contribution screening over the full timeline. '
                 'Not event training, not new evidence about back-side appearance. Requires final actual RGB/event and novel-view checks.'}
    output = dict(original_variants, indices=kept, guarded_surface_indices=kept,
                  candidate_indices=indices.astype(np.int32), maximum_contribution=maxima,
                  metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps(metadata | {'input_selection_metadata': 'see file'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
