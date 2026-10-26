"""Optimize motion alignment using raw events; never approve by appearance alone."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from gaussian_renderer import render
from scripts.rendering.render_ply import make_camera
from scene import GaussianModel
from utils.event_refinement import RigidMotionCorrection
from utils.event_supervision import require_event_export_approval
from utils.rigid_checkpoint import sha256_file
from scripts.events.audit_lego_event_motion import audit
from utils.render_sampling import require_native_sampling


class MotionRenderer:
    def __init__(self, baseline, width=320, supersample=None):
        from utils.render_sampling import render_sampling_contract
        self.meta = json.loads((baseline / 'cache.json').read_text())
        self.render_sampling = render_sampling_contract(self.meta, supersample)
        self.supersample = self.render_sampling['render_supersample']
        self.poses = np.load(baseline / 'poses.npy', mmap_mode='r', allow_pickle=False)
        self.indices = torch.as_tensor(np.load(baseline / 'dynamic_indices.npy').astype(np.int64), device='cuda')
        self.rest = torch.from_numpy(self.poses[0].copy()).cuda()
        self.model = GaussianModel(3, with_motion_mask=False)
        self.model.load_ply(self.meta['source'])
        if torch.count_nonzero(self.model._features_rest).item() != 0:
            raise ValueError('Expected DC-only source color')
        self.model._features_rest = torch.empty((self.meta['gaussian_count'], 0, 3), device='cuda')
        self.model.active_sh_degree = self.model.max_sh_degree = 0
        for key in ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest'):
            getattr(self.model, key).requires_grad_(False)
        self.base_xyz, self.base_rotation = self.model._xyz.detach(), self.model._rotation.detach()
        from utils.multiview_completion import CompactCompletion, verify_multiview_completion
        completed = verify_multiview_completion(self.meta, baseline=baseline)
        self.completion = CompactCompletion(completed, 'cuda') if completed is not None else None
        from utils.gaussian_cleanup import verify_gaussian_cleanup, apply_gaussian_cleanup
        cleaned = verify_gaussian_cleanup(self.meta, baseline=baseline)
        apply_gaussian_cleanup(self.model, cleaned)
        from utils.static_surface_rebuild import verify_static_surface_rebuild, StaticSurfaceRebuild
        rebuilt = verify_static_surface_rebuild(self.meta, baseline=baseline)
        self.static_surface = StaticSurfaceRebuild(rebuilt, 'cuda') if rebuilt is not None else None
        if self.static_surface is not None:
            self.static_surface.apply_appearance(self.model)
        from utils.color_residual import verify_color_residual, apply_color_residual
        residual = verify_color_residual(self.meta, baseline=baseline)
        apply_color_residual(self.model, residual)
        from utils.protected_surface_detail import verify_protected_surface_detail, ProtectedSurfaceDetail
        detail = verify_protected_surface_detail(self.meta, baseline=baseline)
        self.surface_detail = ProtectedSurfaceDetail(detail, 'cuda') if detail is not None else None
        if self.surface_detail is not None:
            self.surface_detail.apply_appearance(self.model)
        if self.completion is not None and torch.any(self.indices >= self.completion.prefix_count):
            raise ValueError('Parent baked motion must not reference the compact suffix')
        self.pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        self.background = torch.zeros(3, device='cuda')
        self.luma = torch.tensor([.299, .587, .114], device='cuda')[:, None, None]
        self.set_resolution(width)

    def set_resolution(self, width):
        if type(width) is not int or width < 1:
            raise ValueError('Render width must be a positive integer')
        camera = self.meta['camera']
        self.width = width
        native_width = width * self.supersample
        self.camera = make_camera(np.asarray(camera['pose']), native_width, native_width,
                                  np.deg2rad(camera['fovy_degrees']))

    def posed(self, index, correction):
        baked = torch.from_numpy(self.poses[index].copy()).cuda()
        # Preserve the source PLY exactly at rest; no color/opacity optimization.
        return baked if index == 0 else correction(self.rest, baked, index)

    def image(self, index, correction):
        from utils.render_sampling import resolve_rgb
        pose = self.posed(index, correction)
        self.model._xyz = self.base_xyz.index_copy(0, self.indices, pose[:, :3])
        self.model._rotation = self.base_rotation.index_copy(0, self.indices, pose[:, 3:])
        if self.completion is not None:
            suffix = self.completion.posed(index, correction)
            prefix = self.completion.prefix_count
            self.model._xyz = torch.cat((self.model._xyz[:prefix], suffix[:, :3]), 0)
            self.model._rotation = torch.cat((self.model._rotation[:prefix], suffix[:, 3:]), 0)
        if getattr(self, 'static_surface', None) is not None:
            self.model._xyz, self.model._rotation = self.static_surface.append_geometry(
                self.model._xyz, self.model._rotation)
        if getattr(self, 'surface_detail', None) is not None:
            self.model._xyz, self.model._rotation = self.surface_detail.append_geometry(
                self.model._xyz, self.model._rotation)
        rgb = resolve_rgb(render(self.camera, self.model, self.pipe, self.background, 0., 0., 0.)['render'],
                          self.supersample)
        return (rgb * self.luma).sum(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--supervision', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=400)
    parser.add_argument('--train-resolution', type=int, choices=(320, 640), default=640,
                        help='640 matches the full-resolution log-intensity audit operator')
    parser.add_argument('--initial-correction', type=Path,
                        help='Continue from a previous motion correction, never from appearance parameters')
    parser.add_argument('--foreground-weight', type=float, default=0.,
                        help='Additional visible foreground coverage constraint in training-event-active region')
    args = parser.parse_args()
    from utils.gaussian_cleanup import require_no_cleanup
    from utils.protected_surface_detail import require_no_protected_surface_detail
    require_no_protected_surface_detail(json.loads((args.baseline/'cache.json').read_text()), 'Legacy motion training')
    require_no_cleanup(json.loads((args.baseline/'cache.json').read_text()), 'Legacy motion training')
    from utils.color_residual import require_no_color_residual
    require_no_color_residual(json.loads((args.baseline/'cache.json').read_text()), 'Legacy motion training')
    if json.loads((args.baseline/'cache.json').read_text()).get('static_surface_rebuild'):
        raise ValueError('Legacy motion training does not support static surface replacement')
    if args.output.exists():
        raise FileExistsError(args.output)
    require_native_sampling(json.loads((args.baseline / 'cache.json').read_text()),
                            'Legacy rigid-motion training')
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    if not np.isfinite(args.foreground_weight) or args.foreground_weight < 0:
        parser.error('--foreground-weight must be finite and nonnegative')
    with np.load(args.supervision, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata_json']))
        images = archive['images']
        counts = archive['counts']
    floor = meta['calibration']['log_floor']
    positive, negative = meta['calibration']['thresholds']
    # Event change targets at 80x80; retain full 640x640 data for final audit.
    target = counts[:, 0].astype(np.float32) * positive - counts[:, 1].astype(np.float32) * negative
    target = target.reshape(len(target), 80, 8, 80, 8).mean((2, 4))
    active = np.zeros(counts.shape[-2:], dtype=bool)
    for index, count in enumerate(counts):
        if index % 5 != 4:
            active |= np.any(count > 0, axis=0)
    active = binary_dilation(active, iterations=3)
    del counts
    event_targets = torch.from_numpy(target).cuda()
    width = args.train_resolution
    rgb_targets = torch.from_numpy(np.asarray([cv2.resize(img, (width, width), interpolation=cv2.INTER_AREA)
                                               for img in images], dtype=np.float32) / 255).cuda()
    foreground_region = torch.from_numpy(cv2.resize(active.astype(np.uint8), (width, width),
                                                     interpolation=cv2.INTER_NEAREST) > 0).cuda()
    del images
    renderer = MotionRenderer(args.baseline, width)
    correction = RigidMotionCorrection().cuda()
    if args.initial_correction:
        previous = json.loads(args.initial_correction.read_text())
        raw = torch.as_tensor(previous['raw'], dtype=torch.float32, device='cuda')
        if (previous.get('format') != 'event_supervised_rigid_motion_alignment_v1'
                or raw.shape != (7,) or not torch.isfinite(raw).all()):
            raise ValueError('Invalid initialization correction')
        with torch.no_grad():
            correction.raw.copy_(raw)
    optimizer = torch.optim.Adam(correction.parameters(), lr=.025)
    train = np.array([i for i in range(len(target)) if i % 5 != 4 and np.abs(target[i]).sum() > 1])
    if len(train) < 20:
        raise ValueError('Too few observable training event intervals')
    probe = train[np.linspace(0, len(train) - 1, 12).astype(int)]
    rng = np.random.default_rng(0)
    args.output.mkdir(parents=True)

    def loss(index):
        first, second = renderer.image(index, correction), renderer.image(index + 1, correction)
        delta = (second.clamp_min(floor).log() - first.clamp_min(floor).log())
        delta = F.avg_pool2d(delta[None, None], width // 80)[0, 0]
        expected = event_targets[index]
        event_loss = (delta - expected).abs().sum() / expected.abs().sum().clamp_min(1.)
        cosine_loss = 1 - (delta * expected).sum() / (delta.norm() * expected.norm()).clamp_min(1e-6)
        rgb_loss = (second - rgb_targets[index + 1]).abs().mean()
        foreground = (rgb_targets[index + 1] > .08) & foreground_region
        coverage_loss = ((.08 - second).clamp_min(0) * foreground).sum() / (.08 * foreground.sum().clamp_min(1))
        total = (event_loss + .5 * cosine_loss + 5 * rgb_loss
                 + args.foreground_weight * coverage_loss + .001 * correction.raw.square().mean())
        return total, event_loss, cosine_loss, rgb_loss

    @torch.no_grad()
    def probe_loss():
        return float(np.mean([loss(int(i))[0].item() for i in probe]))

    initial = probe_loss()
    best_score, best_raw = initial, correction.raw.detach().clone()
    history = [{'iteration': 0, 'probe_loss': initial, 'raw': best_raw.cpu().tolist()}]
    for iteration in range(1, args.iterations + 1):
        index = int(rng.choice(train))
        optimizer.zero_grad(set_to_none=True)
        total, event, cosine, rgb = loss(index)
        if not torch.isfinite(total):
            raise ValueError('Non-finite optimization loss')
        total.backward()
        torch.nn.utils.clip_grad_norm_(correction.parameters(), 5.)
        optimizer.step()
        if iteration % 50 == 0 or iteration == args.iterations:
            score = probe_loss()
            history.append({'iteration': iteration, 'probe_loss': score, 'event_loss': event.item(),
                            'cosine_loss': cosine.item(), 'rgb_loss': rgb.item(), 'raw': correction.raw.detach().cpu().tolist()})
            if score < best_score:
                best_score, best_raw = score, correction.raw.detach().clone()
            print(json.dumps(history[-1]), flush=True)
    with torch.no_grad():
        correction.raw.copy_(best_raw)
        q, translation, scale = correction.parameters_physical()
        result = {'format': 'event_supervised_rigid_motion_alignment_v1', 'initial_probe_loss': initial,
                  'best_probe_loss': best_score, 'raw': best_raw.cpu().tolist(),
                  'rotation_wxyz': q.cpu().tolist(), 'translation': translation.cpu().tolist(),
                  'scale': scale.item(), 'iterations': args.iterations, 'history': history,
                  'training_render_resolution': width,
                  'foreground_weight': args.foreground_weight,
                  'initial_correction': str(args.initial_correction) if args.initial_correction else None,
                  'calibration': meta['calibration'], 'rgb_appearance_parameters_frozen': True,
                  'event_validation_intervals_not_trained': True, 'scope': 'Shared similarity conjugation; per-part rigidity and existing links preserved. No unique-depth claim.'}
        (args.output / 'correction.json').write_text(json.dumps(result, indent=2) + '\n')
        renderer.set_resolution(640)
        rendered = np.lib.format.open_memmap(args.output / 'rendered_gray.npy', mode='w+', dtype=np.float16,
                                            shape=(len(renderer.poses), 640, 640))
        for i in range(len(renderer.poses)):
            frame = renderer.image(i, correction).cpu().numpy()
            rendered[i] = frame
            if i in (0, 50, 100, 200, 300, 399):
                Image.fromarray((frame * 255).astype(np.uint8)).save(args.output / f'frame_{i:04d}.png')
            if i % 100 == 0:
                print('Render corrected', i, flush=True)
        rendered.flush()
    validation = audit(rendered, args.supervision)
    validation['event_optimization_performed'] = True
    validation['artifact_sha256'] = {
        'correction': sha256_file(args.output / 'correction.json'),
        'rendered_gray': sha256_file(args.output / 'rendered_gray.npy'),
        'supervision': sha256_file(args.supervision),
        'baseline_cache_metadata': sha256_file(args.baseline / 'cache.json'),
        'baseline_poses': sha256_file(args.baseline / 'poses.npy'),
        'dynamic_indices': sha256_file(args.baseline / 'dynamic_indices.npy'),
        'source_ply': sha256_file(renderer.meta['source']),
    }
    validation['correction'] = {key: value for key, value in result.items() if key != 'history'}
    (args.output / 'event-audit.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps({key: value for key, value in validation.items()
                      if key not in ('intervals', 'failed_intervals', 'correction')}, indent=2), flush=True)
    if not validation['approved_for_export']:
        print('NOT APPROVED: no final dynamic Gaussian sequence exported. See failed intervals.', flush=True)
        raise SystemExit(2)
    require_event_export_approval(validation)
    print('Audit passed; export/playback integration is a separate step. No final sequence exported by this script.', flush=True)


if __name__ == '__main__':
    main()
