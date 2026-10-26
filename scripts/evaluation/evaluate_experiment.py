"""Evaluate an extracted viewer experiment without a viewer or GPU renderer.

Usage: python -m scripts.evaluation.evaluate_experiment EXPORT_DIRECTORY [--lpips]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from utils.experiment_format import FORMAT, save_json, sha256


def artifact(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Invalid or missing artifact: {name}')
    return path


def verify(root, manifest):
    if manifest.get('format') != FORMAT:
        raise ValueError('Unsupported experiment format')
    for name, record in manifest['files'].items():
        path = artifact(root, name)
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {name}')


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float64) / 255


def ssim_map(reference, prediction):
    """Wang SSIM: 11x11 Gaussian sigma=1.5, population covariance, valid crop."""
    if min(reference.shape[:2]) < 11:
        raise ValueError('SSIM requires at least 11x11 pixels')
    kernel = np.exp(-np.arange(-5, 6, dtype=float)**2 / (2*1.5**2))
    kernel /= kernel.sum()

    def blur(value):
        height, width = value.shape[:2]
        horizontal = sum(weight*value[:, i:i+width-10] for i, weight in enumerate(kernel))
        return sum(weight*horizontal[i:i+height-10] for i, weight in enumerate(kernel))
    mu_a, mu_b = blur(reference), blur(prediction)
    var_a = np.maximum(0, blur(reference*reference)-mu_a*mu_a)
    var_b = np.maximum(0, blur(prediction*prediction)-mu_b*mu_b)
    covariance = blur(reference*prediction)-mu_a*mu_b
    return (((2*mu_a*mu_b+.01**2)*(2*covariance+.03**2)) /
            ((mu_a*mu_a+mu_b*mu_b+.01**2)*(var_a+var_b+.03**2))).mean(axis=2)


def image_metrics(reference, prediction, mask=None, similarity=None):
    if reference.shape != prediction.shape:
        raise ValueError('Prediction/GT size mismatch; implicit resizing is forbidden')
    if mask is None: mask = np.ones(reference.shape[:2], bool)
    if mask.shape != reference.shape[:2]: raise ValueError('Mask size mismatch')
    if not mask.any(): return dict(available=False, reason='Empty fixed reference mask')
    difference = reference[mask]-prediction[mask]
    mse = float(np.square(difference).mean())
    similarity = ssim_map(reference, prediction) if similarity is None else similarity
    interior = mask[5:-5, 5:-5]
    return dict(available=True, pixels=int(mask.sum()), l1=float(np.abs(difference).mean()), mse=mse,
                psnr_db=float(-10*np.log10(mse)) if mse else None, psnr_infinite=mse == 0,
                ssim=float(similarity[interior].mean()) if interior.any() else None)


def third_derivative(values, times):
    """Third divided differences x 3!, defined at each four-sample interval."""
    values, times = np.asarray(values, dtype=float), np.asarray(times, dtype=float)
    if len(times) < 4 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Jerk needs >=4 strictly increasing finite timestamps')
    for degree in range(1, 4):
        span = times[degree:]-times[:-degree]
        values = np.diff(values, axis=0) / span.reshape((-1,) + (1,)*(values.ndim-1))
    return values * 6


def rotation_jerk(quaternions, times):
    """World angular-velocity increments, then a second divided derivative."""
    from utils.mesh_motion import quaternion_product
    q=np.asarray(quaternions,dtype=float);times=np.asarray(times,dtype=float)
    if len(times)<4 or q.shape[0]!=len(times) or q.shape[-1]!=4 or np.any(np.diff(times)<=0):
        raise ValueError('Rotation jerk needs four ordered samples and wxyz rotations')
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    delta=quaternion_product(q[1:],q[:-1]*[1,-1,-1,-1])
    delta=np.where(delta[...,:1]<0,-delta,delta)
    norm=np.linalg.norm(delta[...,1:],axis=-1,keepdims=True)
    vector=delta[...,1:]*(2*np.arctan2(norm,delta[...,:1])/np.maximum(norm,1e-15))
    velocity=vector/np.diff(times).reshape((-1,)+(1,)*(vector.ndim-1))
    centers=(times[1:]+times[:-1])/2
    for degree in (1,2):
        span=centers[degree:]-centers[:-degree]
        velocity=np.diff(velocity,axis=0)/span.reshape((-1,)+(1,)*(velocity.ndim-1))
    return 2*velocity


def motion_metrics(path):
    with np.load(path, allow_pickle=False) as source:
        points, times, ids = source['points'], source['times_seconds'], source['point_ids']
        if points.shape != (len(times), len(ids), 3) or not np.isfinite(points).all():
            raise ValueError('Invalid fixed-point trajectories')
        result = dict(sample_count=len(times), point_count=len(ids),
            scope='Fixed row IDs within this version; cross-version correspondence is unverified',
            motion_amplitude=float(np.linalg.norm(points-points[0], axis=2).max(initial=0)),
            position_jerk_rms=None, angular_jerk_rms=None, joint_gap_max=None, rigidity_error=None,
            unavailable={})
        if len(times) >= 4:
            jerk = third_derivative(points, times)
            result['position_jerk_rms'] = float(np.sqrt(np.square(jerk).sum(axis=2).mean()))
        else: result['unavailable']['position_jerk_rms'] = 'Fewer than four timestamps'
        if 'joint_angles' in source and source['joint_angles'].size and len(times) >= 4:
            result['angular_jerk_rms'] = float(np.sqrt(np.square(third_derivative(np.unwrap(source['joint_angles'], axis=0), times)).mean()))
        else: result['unavailable']['angular_jerk_rms'] = 'No verified joint angles (radians) or insufficient timestamps'
        if 'group_quaternions' in source and len(times)>=4:
            result['rotation_jerk_rms']=float(np.sqrt(np.square(rotation_jerk(source['group_quaternions'],times)).sum(-1).mean()))
            result['rotation_jerk_scope']='SO(3) shortest increments divided by dt; second divided derivative at interval midpoints; rad/s^3'
        if 'joint_twist_angles' in source and source['joint_twist_angles'].size and len(times)>=4:
            result['joint_twist_jerk_rms']=float(np.sqrt(np.square(third_derivative(np.unwrap(source['joint_twist_angles'],axis=0),times)).mean()))
            result['joint_twist_scope']='Swing-twist projection on the declared source-prior axis; not a pure hinge angle when swing is nonzero'
        if 'joint_swing_angles' in source:
            result['joint_swing_max_rad']=float(np.abs(source['joint_swing_angles']).max(initial=0))
        if {'group_quaternions', 'group_translations', 'joint_pairs', 'joint_pivots'} <= set(source.files):
            from utils.mesh_motion import rotate
            q, t = source['group_quaternions'], source['group_translations']
            pairs, pivots = source['joint_pairs'], source['joint_pivots']
            a = rotate(q[:, pairs[:, 0]], pivots)+t[:, pairs[:, 0]]
            b = rotate(q[:, pairs[:, 1]], pivots)+t[:, pairs[:, 1]]
            result['joint_gap_max'] = float(np.linalg.norm(a-b, axis=-1).max(initial=0))
        else: result['unavailable']['joint_gap_max'] = 'No verified joint pivots and actual part transforms'
        if 'point_groups' in source:
            errors = []
            for group in np.unique(source['point_groups']):
                if group < 0: continue
                subset = points[:, source['point_groups'] == group]
                if subset.shape[1] < 2: continue
                distances = np.linalg.norm(subset[:, :, None]-subset[:, None, :], axis=-1)
                errors.append(float(np.abs(distances-distances[0]).max()))
            if errors: result['rigidity_error'] = max(errors)
        if result['rigidity_error'] is None:
            result['unavailable']['rigidity_error'] = 'No verified rigid part membership with >=2 fixed points'
        return result


def event_metrics(first, second, event_path, calibration, t0, t1):
    if not calibration or calibration.get('space') != 'rgb_image':
        return dict(available=False, reason='No event-to-RGB calibration')
    if calibration.get('transfer') not in ('linear', 'srgb', 'encoded'):
        return dict(available=False, reason='No declared RGB radiometric transfer')
    cp, cn = calibration.get('contrast_positive'), calibration.get('contrast_negative')
    if not all(isinstance(c, (int, float)) and np.isfinite(c) and c > 0 for c in (cp, cn)):
        return dict(available=False, reason='Missing fixed positive/negative contrast thresholds')

    def log_luminance(image):
        if calibration['transfer'] == 'srgb':
            image = np.where(image <= .04045, image/12.92, ((image+.055)/1.055)**2.4)
        if calibration.get('luminance') == 'opencv_bt601_u8':
            # OpenCV's fixed-point RGB-to-gray coefficients, preserving the
            # response convention used by the archived calibration script.
            pixels = np.rint(image*255).astype(np.int64)
            luminance = ((pixels @ np.array([9798, 19235, 3735]) + 16384) >> 15)/255.
        else:
            luminance = image @ np.array([.2126, .7152, .0722])
        floor = calibration.get('log_floor', 1e-6)
        if not isinstance(floor, (int, float)) or not 0 < floor < 1:
            raise ValueError('Invalid calibrated log intensity floor')
        return np.log(np.maximum(luminance, floor))
    with np.load(event_path, allow_pickle=False) as data:
        if abs(float(data['t0'])-t0) > 1e-8 or abs(float(data['t1'])-t1) > 1e-8:
            return dict(available=False, reason='Event window does not equal sampled frame interval')
        positive, negative = data['positive'].astype(np.float64), data['negative'].astype(np.float64)
    if positive.shape != first.shape[:2] or negative.shape != positive.shape:
        raise ValueError('Calibrated event shape mismatch')
    if not np.isfinite([positive, negative]).all() or np.any(positive < 0) or np.any(negative < 0):
        raise ValueError('Invalid event counts')
    delta = log_luminance(second)-log_luminance(first)
    observed = cp*positive-cn*negative
    denominator = float(np.abs(observed).sum())
    polarity = observed != 0
    silent = positive+negative == 0
    norm = float(np.linalg.norm(observed)*np.linalg.norm(delta))
    error = float(np.abs(delta-observed).sum())
    return dict(available=True, relative_l1=error/denominator if denominator else None,
        error_l1_sum=error, observed_abs_sum=denominator,
        cosine=float(np.sum(observed*delta)/norm) if norm else None,
        polarity_agreement=float((np.sign(delta[polarity]) == np.sign(observed[polarity])).mean()) if polarity.any() else None,
        no_event_change_mean=float(np.abs(delta[silent]).mean()) if silent.any() else None,
        no_event_pixels=int(silent.sum()), t0=t0, t1=t1)


def evaluate(root, use_lpips=False, version_ids=None):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    verify(root, manifest)
    result = dict(format='scgs_experiment_metrics_v1', manifest_sha256=sha256(root/'manifest.json'),
        protocol=dict(range=1, psnr='mean per-frame dB; infinite results explicitly flagged',
            ssim='11x11 Gaussian, sigma=1.5, population covariance, valid crop, channel mean',
            masked_ssim='mean of SSIM map at fixed mask centers; windows include neighbors',
            jerk='6 times third divided differences using actual timestamps; units per second cubed'), versions={})
    lpips_model = None
    if use_lpips:
        try:
            import torch
            import lpips
        except ImportError as error:
            raise RuntimeError('--lpips requires torch and lpips; install evaluation requirements') from error
        lpips_model = lpips.LPIPS(net='alex').eval()
        result['protocol']['lpips'] = 'alex, RGB [-1,1], full frame, CPU'
        # Record the actual model, including installed/pretrained weights.
        import hashlib
        digest = hashlib.sha256()
        for key, tensor in sorted(lpips_model.state_dict().items()):
            digest.update(key.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
        result['protocol']['lpips_weights_sha256'] = digest.hexdigest()
    references = manifest.get('reference', {}).get('samples', [])
    if version_ids is not None and not set(version_ids)<=set(v['id'] for v in manifest['versions']):
        raise ValueError('Unknown evaluation version')
    for version in manifest['versions']:
        name = version['id']
        if version_ids is not None and name not in version_ids: continue
        if not isinstance(name, str) or not name.isascii() or not name.isdigit():
            raise ValueError('Invalid version identifier')
        rows, events, previous = [], [], None
        for index, sample in enumerate(manifest['samples']):
            ref = references[index]
            predicted = rgb(artifact(root, f'versions/{name}/frames/{index:06d}.png'))
            row = dict(sample=index, frame=sample['frame'], time_seconds=sample['time_seconds'])
            truth = similarity = None
            alpha_name = f'versions/{name}/alpha/{index:06d}.png'
            coverage = None
            if alpha_name in manifest['files']:
                coverage = np.asarray(Image.open(artifact(root, alpha_name))) / 65535 >= .5
            if not ref.get('aligned') or 'rgb' not in ref or not ref.get('background_matches', True):
                row.update(available=False, reason=ref.get('reason', 'GT camera/time/background mismatch or missing RGB'))
            else:
                truth = rgb(artifact(root, ref['rgb']))
                similarity = ssim_map(truth, predicted)
                row.update(image_metrics(truth, predicted, similarity=similarity))
                if sample['render_mode'] != 'gaussian':
                    row['scope'] = 'Display appearance discrepancy; not native RGB reconstruction quality'
                row['lpips'] = None
                if lpips_model:
                    with torch.no_grad():
                        tensors = [torch.from_numpy(np.ascontiguousarray(value.transpose(2,0,1))).float()[None]*2-1 for value in (truth, predicted)]
                        row['lpips'] = float(lpips_model(*tensors).item())
            if ref.get('aligned'):
                if 'foreground_mask' in ref:
                    mask = np.asarray(Image.open(artifact(root, ref['foreground_mask'])))
                    if mask.ndim != 2: raise ValueError('Fixed masks must be single-channel')
                    mask = mask > 0
                    if truth is not None: row['foreground'] = image_metrics(truth, predicted, mask, similarity)
                    if coverage is not None and mask.any():
                        row['silhouette_iou'] = float((coverage & mask).sum() / max((coverage | mask).sum(), 1))
                        row['foreground_missing_fraction'] = float((mask & ~coverage).sum()/mask.sum())
                if 'part_mask' in ref:
                    mask = np.asarray(Image.open(artifact(root, ref['part_mask'])))
                    if mask.ndim != 2: raise ValueError('Part masks must contain integer labels')
                    row['parts'] = {str(label): (image_metrics(truth, predicted, mask == label, similarity)
                                    if truth is not None else dict(available=False, reason='No aligned RGB reference'))
                                    for label in np.unique(mask) if label > 0}
                    if coverage is not None:
                        for label, part in row['parts'].items():
                            selected = mask == int(label)
                            part['alpha_missing_fraction'] = float((selected & ~coverage).sum()/selected.sum())
            rows.append(row)
            if 'events' in ref:
                eligible = (previous is not None and ref.get('aligned') and previous[2].get('aligned')
                    and ref.get('background_matches', True) and previous[2].get('background_matches', True)
                    and sample['camera'] == previous[1]['camera'] and sample['render_mode'] == 'gaussian'
                    and previous[1]['render_mode'] == 'gaussian' and sample['background'] == previous[1]['background'])
                metric = (event_metrics(previous[0], predicted, artifact(root, ref['events']), ref.get('event_calibration'),
                                       previous[1]['time_seconds'], sample['time_seconds']) if eligible else
                          dict(available=False, reason='Need two native frames at the calibrated fixed event camera'))
                split = ref.get('event_split', 'unspecified')
                calibration_ids = (ref.get('event_calibration') or {}).get('calibration_ids')
                if calibration_ids is not None and previous is not None:
                    split = ('overlaps_response_calibration' if any(previous[1]['frame'] <= i < sample['frame']
                             for i in calibration_ids) else 'response_validation')
                events.append(dict(sample=index, split=split, **metric))
            previous = predicted, sample, ref
        valid = [row for row in rows if row.get('available')]
        finite_psnr = [row['psnr_db'] for row in valid if row['psnr_db'] is not None]
        infinite = sum(row['psnr_infinite'] for row in valid)
        mean = lambda values: float(np.mean(values)) if values else None
        result['versions'][name] = dict(label=version['label'], frames=rows, events=events,
            summary=dict(evaluated_frames=len(valid), total_frames=len(rows), infinite_psnr_frames=infinite,
                psnr_db=None if infinite else mean(finite_psnr), psnr_infinite=bool(infinite),
                finite_psnr_db_mean=mean(finite_psnr), ssim=mean([row['ssim'] for row in valid]),
                l1=mean([row['l1'] for row in valid])),
            motion=motion_metrics(artifact(root, f'versions/{name}/motion.npz')),
            unavailable=dict(lpips=None if use_lpips else 'Not requested; use --lpips',
                             events=None if events else 'No raw calibrated event counts'))
    timing = json.loads(artifact(root, 'timings.json').read_text())
    measurements = timing['samples']
    latencies = [row['render_and_transfer_ms'] for row in measurements]
    result['performance'] = dict(operation=timing.get('operation', manifest['operation']),
        export_operation=manifest['operation'], samples=len(latencies),
        warmup=timing['warmup'], repeats=timing['repeats'], scope=manifest['timing_scope'],
        includes_state_capture=any(row['includes_state_capture'] for row in measurements),
        mean_ms=float(np.mean(latencies)), p50_ms=float(np.percentile(latencies, 50)),
        p95_ms=float(np.percentile(latencies, 95)),
        comparison_samples_per_second=1000/float(np.mean(latencies)) if np.mean(latencies)>0 else None,
        load_seconds=timing['load_seconds'], memory_scope=timing['memory_scope'],
        memory_peak=max((row['memory']['peak_allocated_bytes'] for row in measurements if row['memory']), default=None))
    panel_count = max((len(row.get('panel_ms', [])) for row in measurements), default=0)
    result['performance']['versions'] = {}
    for side in range(panel_count):
        values = [row['panel_ms'][side] for row in measurements if len(row.get('panel_ms', [])) > side]
        result['performance']['versions'][str(side)] = dict(mean_ms=float(np.mean(values)),
            p50_ms=float(np.percentile(values,50)), p95_ms=float(np.percentile(values,95)),
            frames_per_second=1000/float(np.mean(values)) if np.mean(values)>0 else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, help='Defaults to DIRECTORY/metrics.json')
    parser.add_argument('--lpips', action='store_true', help='Enable optional pretrained AlexNet LPIPS')
    parser.add_argument('--version', action='append', help='Evaluate only this version ID; may be repeated')
    args = parser.parse_args()
    output = args.output or args.directory/'metrics.json'
    manifest = json.loads((args.directory/'manifest.json').read_text())
    if output.resolve() == (args.directory/'manifest.json').resolve() or output.resolve() in {
            (args.directory/name).resolve() for name in manifest['files']}:
        parser.error('Output must not overwrite an exported artifact')
    save_json(output, evaluate(args.directory, args.lpips, args.version))
    print(output)


if __name__ == '__main__': main()
