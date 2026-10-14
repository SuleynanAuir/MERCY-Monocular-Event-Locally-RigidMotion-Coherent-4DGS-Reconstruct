"""Fit only new rigid-surface appearance with measured RGB, gradients and raw events.

Geometry, binding, motion, original static rows and elastic-band rows are frozen.
Validation events never enter gradients or checkpoint selection. A lower training
loss is not event acceptance; the unchanged full audit runs after fitting.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
import torch
import torch.nn.functional as F

from scripts.refine_lego_events import MotionRenderer
from scripts.audit_lego_event_motion import audit
from utils.appearance_refinement import SurfaceAppearance, masked_detail_losses, masked_laplacian_error
from utils.event_refinement import correction_from_record
from utils.rigid_checkpoint import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'base-rebuild', 'supervision', 'masks', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=600)
    parser.add_argument('--event-weight', type=float, default=.3)
    parser.add_argument('--supersample', type=int, choices=(1, 2), default=1)
    parser.add_argument('--anisotropic', action='store_true')
    parser.add_argument('--minimum-scale', type=float, default=.55)
    parser.add_argument('--maximum-scale', type=float, default=1.1)
    parser.add_argument('--gradient-weight', type=float, default=8.)
    parser.add_argument('--laplacian-weight', type=float, default=0.)
    parser.add_argument('--coverage-weight', type=float, default=3.)
    parser.add_argument('--lr-multiplier', type=float, default=1.)
    parser.add_argument('--cosine-lr', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.iterations < 1 or not all(np.isfinite(value) and value >= 0 for value in
            (args.event_weight, args.gradient_weight, args.laplacian_weight, args.coverage_weight))
            or not np.isfinite(args.lr_multiplier) or not 0 < args.lr_multiplier <= 2
            or not 0 < args.minimum_scale <= 1 <= args.maximum_scale <= 2):
        raise ValueError('Invalid optimization settings')
    torch.set_num_threads(4)
    report = json.loads((args.base_rebuild/'rebuild-report.json').read_text())
    renderer = MotionRenderer(args.baseline, 640, supersample=args.supersample)
    if (report['format'] != 'lego_prior_surface_rebuild_v1'
            or report['rebuilt_sha256'] != sha256_file(renderer.meta['source'])
            or report['gaussian_count'] != renderer.meta['gaussian_count']):
        raise ValueError('Baseline is not the expected rebuilt model')
    retained, seeds = report['retained_gaussian_count'], report['rebuilt_gaussians']
    if retained+seeds != renderer.meta['gaussian_count']:
        raise ValueError('Seed suffix does not match the source')
    record = json.loads((args.candidate/'correction.json').read_text())
    correction = correction_from_record(record, 'cuda').eval()
    for parameter in correction.parameters():
        parameter.requires_grad_(False)
    original = {key: getattr(renderer.model, key).detach() for key in ('_features_dc', '_scaling', '_opacity')}
    appearance = SurfaceAppearance(*(original[key][retained:] for key in ('_features_dc', '_scaling', '_opacity')),
                                   minimum_scale=args.minimum_scale, maximum_scale=args.maximum_scale,
                                   anisotropic=args.anisotropic).cuda()
    with np.load(args.supervision, allow_pickle=False) as archive:
        reference = torch.from_numpy(archive['images'].astype(np.float32)/255).cuda()
        counts = archive['counts']
        metadata = json.loads(str(archive['metadata_json']))
    with np.load(args.masks, allow_pickle=False) as archive:
        masks = archive['part_ids']
        mask_meta = json.loads(str(archive['metadata_json']))
    region = binary_dilation(np.isin(masks, mask_meta['moving_rigid_parts']).any(0), iterations=3)
    region = torch.from_numpy(region).cuda()
    if reference.shape != (len(renderer.poses), 640, 640) or not region.any():
        raise ValueError('Invalid supervision timeline or moving-part region')
    positive, negative = metadata['calibration']['thresholds']
    target = np.empty((len(counts), 80, 80), np.float32)
    for index, count in enumerate(counts):
        signed = count[0].astype(np.float32)*positive-count[1].astype(np.float32)*negative
        target[index] = signed.reshape(80, 8, 80, 8).mean((1, 3))
    del counts, masks
    floor = metadata['calibration']['log_floor']
    train = np.flatnonzero((np.arange(len(target)) % 5 != 4) & (np.abs(target).sum((1, 2)) > 1))
    if len(train) < 32:
        raise ValueError('Too few observable training intervals')
    probe = train[np.linspace(0, len(train)-1, 24).astype(int)]
    events = torch.from_numpy(target).cuda()
    optimizer = torch.optim.Adam([
        {'params': [appearance.gray], 'lr': .012*args.lr_multiplier},
        {'params': [appearance.log_scale_ratio], 'lr': .015*args.lr_multiplier},
        {'params': [appearance.opacity], 'lr': .02*args.lr_multiplier}])
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        .1+.9*.5*(1+math.cos(math.pi*min(step, args.iterations)/args.iterations))) if args.cosine_lr else None
    rng = np.random.default_rng(23)
    args.output.mkdir(parents=True)

    def apply_attributes():
        values = appearance()
        for key, value in zip(('_features_dc', '_scaling', '_opacity'), values):
            setattr(renderer.model, key, torch.cat((original[key][:retained], value), dim=0))

    def loss(index):
        apply_attributes()
        first, second = renderer.image(index, correction), renderer.image(index+1, correction)
        a = masked_detail_losses(first, reference[index], region)
        b = masked_detail_losses(second, reference[index+1], region)
        rgb, gradient, coverage = [(x+y)*.5 for x, y in zip(a, b)]
        laplacian = ((masked_laplacian_error(first, reference[index], region)
                      + masked_laplacian_error(second, reference[index+1], region))*.5
                     if args.laplacian_weight else rgb.new_zeros(()))
        predicted = F.avg_pool2d((second.clamp_min(floor).log()-first.clamp_min(floor).log())[None, None], 8)[0, 0]
        expected = events[index]
        event = ((predicted-expected).abs().sum()/expected.abs().sum().clamp_min(1)
                 + .5*(1-(predicted*expected).sum()/(predicted.norm()*expected.norm()).clamp_min(1e-6)))
        total = (8*rgb + args.gradient_weight*gradient + args.coverage_weight*coverage
                 + args.laplacian_weight*laplacian + args.event_weight*event + appearance.regularization())
        return total, (rgb, gradient, coverage, event, laplacian)

    @torch.no_grad()
    def evaluate():
        values = []
        for index in probe:
            total, components = loss(int(index))
            values.append([total.item()]+[value.item() for value in components])
        return dict(zip(('loss', 'rgb_l1', 'gradient_l1', 'coverage_deficit', 'event_loss', 'laplacian_l1'), np.mean(values, axis=0).tolist()))

    initial = evaluate()
    best_score = initial['loss']
    best = {key: value.detach().clone() for key, value in appearance.state_dict().items()}
    history = [{'iteration': 0, **initial}]
    print(json.dumps(history[-1]), flush=True)
    for iteration in range(1, args.iterations+1):
        optimizer.zero_grad(set_to_none=True)
        total, _ = loss(int(rng.choice(train)))
        if not torch.isfinite(total):
            raise ValueError('Nonfinite detail-refinement loss')
        total.backward()
        torch.nn.utils.clip_grad_norm_(appearance.parameters(), 5.)
        optimizer.step(); appearance.project_()
        if schedule:
            schedule.step()
        if iteration % 100 == 0 or iteration == args.iterations:
            score = evaluate()
            history.append({'iteration': iteration, **score})
            if score['loss'] < best_score:
                best_score = score['loss']
                best = {key: value.detach().clone() for key, value in appearance.state_dict().items()}
            print(json.dumps(history[-1]), flush=True)
    appearance.load_state_dict(best)
    with torch.no_grad():
        dc, scale, opacity = appearance()
        np.savez_compressed(args.output/'attributes.npz',
                            dc_gray=(dc[:, 0, 0]*.28209479177387814+.5).cpu().numpy(),
                            log_scale=scale.cpu().numpy(), opacity_logit=opacity[:, 0].cpu().numpy())
        result = {
            'format': 'lego_frozen_motion_surface_appearance_v1',
            'baseline': str(args.baseline), 'candidate': str(args.candidate), 'base_rebuild': str(args.base_rebuild),
            'source': renderer.meta['source'], 'seed_count': seeds, 'retained_count': retained,
            'iterations': args.iterations, 'history': history, 'best_training_probe_loss': best_score,
            'training_intervals': train.tolist(), 'probe_intervals': probe.tolist(),
            'weights': {'rgb': 8., 'gradient': args.gradient_weight, 'coverage': args.coverage_weight,
                        'event': args.event_weight, 'laplacian': args.laplacian_weight},
            'radius_ratio_bounds': [args.minimum_scale, args.maximum_scale], 'opacity_bounds': [.75, .995],
            'anisotropic_scale_refinement': args.anisotropic,
            'render_supersample': args.supersample,
            'render_downsample_filter': 'area_after_clamp',
            'training_native_resolution': 640*args.supersample,
            'lr_multiplier': args.lr_multiplier, 'cosine_lr': args.cosine_lr,
            'positions_rotations_motion_frozen': True, 'original_static_and_bands_frozen': True,
            'validation_events_not_used_for_fitting_or_selection': True,
            'rgb_note': 'RGB observations at adjacent times may be used; no independent RGB-heldout claim.',
            'input_sha256': {key: sha256_file(path) for key, path in {
                'source_ply': renderer.meta['source'], 'correction': args.candidate/'correction.json',
                'supervision': args.supervision, 'baseline_cache_metadata': args.baseline/'cache.json',
                'rebuild_report': args.base_rebuild/'rebuild-report.json', 'masks': args.masks}.items()},
            'attributes_sha256': sha256_file(args.output/'attributes.npz'),
            'scale_ratio_quantiles': torch.quantile(appearance.log_scale_ratio.reshape(-1).exp(),
                torch.tensor([0., .1, .5, .9, 1.], device='cuda')).cpu().tolist(),
            'scope': 'Actual 3D Gaussian appearance refinement of previously observed moving surfaces, not image sharpening or new hidden geometry.'}
        (args.output/'optimization.json').write_text(json.dumps(result, indent=2)+'\n')
        apply_attributes()
        rendered = np.lib.format.open_memmap(args.output/'rendered_gray.npy', mode='w+', dtype=np.float16,
                                            shape=(len(renderer.poses), 640, 640))
        for index in range(len(rendered)):
            frame = renderer.image(index, correction).cpu().numpy()
            rendered[index] = frame
            if index in (0, 50, 100, 150, 200, 300, 399):
                Image.fromarray(np.rint(frame*255).astype(np.uint8)).save(args.output/f'frame_{index:04d}.png')
            if index % 100 == 0:
                print('Rendered detail candidate', index, flush=True)
        rendered.flush()
    assessment = audit(rendered, args.supervision)
    (args.output/'appearance-event-audit.json').write_text(json.dumps(assessment, indent=2)+'\n')
    print(json.dumps({'event_summary': assessment['summary'], 'approved': assessment['approved_for_export'],
                      'scale_ratios': result['scale_ratio_quantiles']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
