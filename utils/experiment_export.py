"""Compact lossless evaluation bundles; source models stay in the project."""
import importlib.metadata
import json
import math
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from utils.viewer_state import orbit_pose, validate_request
from utils.experiment_format import FORMAT, save_json, sha256

SAMPLE_STEP = 4
MAX_EXPORT_BYTES = 2 * 1024**3


def copy_chunks(source, target, progress):
    """Report bytes for large copies and ZIP members without buffering files."""
    for chunk in iter(lambda: source.read(1024 * 1024), b''):
        target.write(chunk)
        progress(len(chunk))


class ExportProgress:
    """Ordered workflow progress; percentages describe work, not time remaining."""
    def __init__(self, operation, callback):
        self.callback = callback
        self.stages = [('prepare', '准备相机与配置')]
        if operation == 'benchmark':
            self.stages += [('warmup', '预热渲染器'), ('benchmark', '重复性能测量')]
        if operation != 'benchmark':
            self.stages += [('frames', '导出采样帧与关键运动数据'), ('motion', '保存运动轨迹'),
                            ('reference', '保存参考数据')]
        self.stages += [('sources', '记录模型来源与哈希'), ('code', '保存配置与评测脚本'),
                        ('hashes', '校验文件哈希'), ('archive', '打包下载文件')]
        self.stage = None
        self.last_update = 0.

    def update(self, stage, completed=0, total=1, unit='items', detail=''):
        now = time.monotonic()
        if stage == self.stage and completed not in (0, total) and now - self.last_update < .1:
            return
        index = next(i for i, row in enumerate(self.stages) if row[0] == stage)
        fraction = min(1., max(0., completed / total)) if total else 1.
        self.callback(dict(stage=stage, stage_index=index+1, stage_count=len(self.stages),
            message=self.stages[index][1], completed=completed, total=total, unit=unit, detail=detail,
            stage_percent=round(fraction*100, 1),
            percent=min(99.9, round((index+fraction)/len(self.stages)*100, 1))))
        self.stage, self.last_update = stage, now


def array(value):
    return value.detach().cpu().numpy().copy() if hasattr(value, 'detach') else np.asarray(value).copy()


def render_pose(base, target, request):
    """The export supplies one absolute OpenGL camera pose to every version."""
    if '_camera_pose' in request:
        return np.asarray(request['_camera_pose'], dtype=np.float32)
    return orbit_pose(base, target, request)


def render_background(backgrounds, request):
    background = backgrounds[request['background']]
    if request.get('_raw') and '_background_rgb' in request:
        return background.new_tensor(request['_background_rgb'])
    return background


def joint_snapshot(structure, quaternions):
    """Export declared hinges only; reject relative rotations off their axes."""
    if not all(hasattr(structure, key) for key in ('joint_groups', 'joint_pivots', 'joint_axes')):
        return {}
    from utils.mesh_motion import quaternion_product
    pairs, pivots, axes = [array(getattr(structure, key)) for key in ('joint_groups', 'joint_pivots', 'joint_axes')]
    q = array(quaternions).astype(np.float64)
    relative = quaternion_product(q[pairs[:, 0]] * [1, -1, -1, -1], q[pairs[:, 1]])
    projection = (relative[:, 1:]*axes).sum(-1)
    result = dict(joint_pairs=pairs, joint_pivots=pivots, joint_axes=axes)
    result['joint_twist_angles'] = 2*np.arctan2(projection, relative[:,0])
    result['joint_swing_angles'] = 2*np.arctan2(np.linalg.norm(relative[:,1:]-projection[:,None]*axes,axis=-1),
                                               np.hypot(relative[:,0],projection))
    if len(pairs) and np.max(np.abs(relative[:, 1:]-projection[:, None]*axes)) < 1e-4:
        result['joint_angles'] = 2*np.arctan2(projection, relative[:, 0])
    return result


def comparison_motion(renderer, side, request):
    """Verified hard-joint baseline/correction; do not label auxiliary fields rigid."""
    if not request.get('_capture') or getattr(renderer, 'deformation', None) is not None:
        return {}
    correction = renderer.correction
    if not all(hasattr(correction, key) for key in ('source_quaternions', 'source_translations', 'row_groups')):
        return {}
    frame = request['frame']
    q, t = ((correction.source_quaternions[frame], correction.source_translations[frame]) if side == 0
            else correction.group_transforms(frame))
    indices, groups = array(renderer.indices[side]), array(correction.row_groups)
    if len(indices) != len(groups): return {}
    mapping = np.full(len(renderer.model._xyz), -1, dtype=np.int32)
    mapping[indices] = groups
    # Verify the claimed transforms against the actual rendered rigid rows.
    # Completion rows without shared binding IDs remain explicitly unassigned.
    from utils.mesh_motion import rotate
    selected = indices[groups >= 0]
    q, t = array(q), array(t)
    canonical = array(renderer.side_geometry[side][0][selected])
    expected = rotate(q[mapping[selected]], canonical) + t[mapping[selected]]
    actual = array(renderer.model._xyz[selected])
    scale = max(float(np.linalg.norm(np.ptp(canonical, axis=0))), 1e-6) if len(canonical) else 1
    if not len(selected) or np.max(np.linalg.norm(actual-expected, axis=1)) > max(scale*1e-5, 1e-6):
        return {'point_groups': mapping}
    return dict(point_groups=mapping, group_quaternions=q, group_translations=t, **joint_snapshot(correction, q))


def capture_points(request, xyz, groups=None, motion=None, alpha=None):
    callback = request.get('_capture')
    if callback:
        ids = callback.point_ids(len(xyz), groups)
        # Select on the GPU before transferring: never read full SH/geometry arrays.
        state = dict(xyz=array(xyz[ids]), point_ids=ids, point_count=len(xyz))
        if groups is not None: state['point_groups'] = array(groups[ids])
        if motion:
            state.update({key: array(value) for key, value in motion.items() if key != 'point_groups'})
        callback(state, array(alpha) if alpha is not None else None)


def capture_gaussians(request, model, alpha=None, motion=None):
    if request.get('_capture'):
        capture_points(request, model._xyz, (motion or {}).get('point_groups'), motion, alpha)


def begin_panel(request):
    if not request.get('_raw'): return None
    cuda_stats()
    return time.perf_counter()


def end_panel(request, started):
    if started is None: return
    cuda_stats()
    request.setdefault('_panel_timings', []).append((time.perf_counter()-started)*1000)


def validate_plan(data, info):
    if not isinstance(data, dict) or data.get('operation') not in ('frame', 'sequence', 'benchmark', 'complete'):
        raise ValueError('请选择一键保存、当前帧、固定序列或性能测量')
    if data['operation'] == 'complete':
        if not 1 <= info['frame_count'] <= 10000:
            raise ValueError('一键保存支持 1–10000 帧')
        settings = validate_request({'fovy': info['fovy'], **data.get('settings', {})}, info['frame_count'])
        step = data.get('sample_step', info.get('export_sample_step', SAMPLE_STEP))
        if type(step) is not int or not 1 <= step <= 10000:
            raise ValueError('采样间隔必须为 1–10000 的整数')
        # The server expands the whole source timeline, independent of the UI's
        # range controls, previously imported plans or client-supplied samples.
        data = {**data, 'samples': [dict(settings, frame=frame, time_seconds=info['times_seconds'][frame])
                                    for frame in range(0, info['frame_count'], step)]}
    samples = data.get('samples')
    if not isinstance(samples, list) or not 1 <= len(samples) <= 10000:
        raise ValueError('采样表需要 1–10000 个采样点')
    if data['operation'] == 'frame' and len(samples) != 1:
        raise ValueError('当前帧导出只能包含一个采样点')
    times = np.asarray(info['times_seconds'], dtype=float)
    if times.shape != (info['frame_count'],) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('源时间轴必须有限且严格递增')
    validated = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError('无效采样点')
        request = validate_request(sample, len(times))
        mode = request['render_mode']
        if mode != 'gaussian' and mode not in info.get('comparison_modes', []) and not info.get(mode, {}).get('available'):
            raise ValueError(f'当前模型不支持显示模式：{mode}')
        actual = float(times[request['frame']])
        if not math.isfinite(sample.get('time_seconds', math.nan)) or abs(sample['time_seconds'] - actual) > 1e-8:
            raise ValueError('保存的时间表与当前模型不一致；请重新保存采样表')
        if 'camera' in sample:
            camera = sample['camera']
            pose = np.asarray(camera['pose_opengl_c2w'], dtype=float)
            if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                    or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
                    or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5, rtol=0)
                    or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5, rtol=0)):
                raise ValueError('无效的已保存相机姿态')
            f = request['height'] / (2*np.tan(np.deg2rad(request['fovy'])/2))
            expected = [[f, 0, request['width']/2], [0, f, request['height']/2], [0, 0, 1]]
            if (camera['width'] != request['width'] or camera['height'] != request['height']
                    or np.asarray(camera['K']).shape != (3, 3)
                    or not np.allclose(camera['K'], expected, rtol=0, atol=1e-5)):
                raise ValueError('已保存相机内参与渲染尺寸不一致')
            request['camera'] = dict(pose_opengl_c2w=pose.astype(np.float32).tolist(), K=expected,
                                     width=request['width'], height=request['height'], fovy_degrees=request['fovy'])
        validated.append({**request, 'time_seconds': actual})
    if any(b['frame'] <= a['frame'] for a, b in zip(validated, validated[1:])):
        raise ValueError('采样帧必须严格递增，不能重复')
    for key, default, low, high in [('warmup', 20, 1, 100), ('repeats', 5, 1, 100)]:
        value = data.get(key, default)
        if data['operation'] == 'benchmark' and (type(value) is not int or not low <= value <= high):
            raise ValueError(f'{key} 超出范围 {low}–{high}')
    if data['operation'] == 'benchmark' and len(samples) * data.get('repeats', 5) > 50000:
        raise ValueError('性能测量最多 50000 次渲染')
    plan = dict(operation=data['operation'], samples=validated,
                warmup=data.get('warmup', 20), repeats=data.get('repeats', 5))
    if data['operation'] != 'benchmark': plan.update(warmup=0, repeats=1)
    if data['operation'] == 'complete': plan.update(selected_frame=settings['frame'], sample_step=step)
    plan['export_profile'] = 'evaluation_compact_v2'
    return plan


def camera_record(renderer, request):
    if 'camera' in request:
        return request['camera']
    base = getattr(renderer, 'base', renderer)
    if hasattr(base, 'sides'):
        base = base.sides[0]
    pose = render_pose(base.base_pose, base.target, request)
    f = request['height'] / (2 * np.tan(np.deg2rad(request['fovy']) / 2))
    return dict(pose_opengl_c2w=pose.tolist(), width=request['width'], height=request['height'],
                fovy_degrees=request['fovy'], K=[[f, 0, request['width']/2], [0, f, request['height']/2], [0, 0, 1]])


def reference_plan(plan, info, reference_path):
    """One-click evaluation uses the recorded source camera, never a fitted view."""
    if plan['operation'] != 'complete' or not reference_path:
        return plan
    source = json.loads(Path(reference_path).read_text())
    if source.get('format') != 'scgs_experiment_reference_v1' or not source.get('provenance'):
        raise ValueError('无效的独立参考数据清单')
    lookup = {row['frame']: row for row in source['samples']}
    samples = []
    for sample in plan['samples']:
        row = lookup.get(sample['frame'])
        if not row or abs(row['time_seconds']-sample['time_seconds']) > 1e-8:
            raise ValueError(f'参考数据未覆盖每 {plan["sample_step"]} 帧采样表；请重新准备参考数据')
        camera = row.get('camera', source.get('camera'))
        fovy = float(np.rad2deg(2*np.arctan(camera['height']/(2*camera['K'][1][1]))))
        samples.append(dict(sample, camera=camera, width=camera['width'], height=camera['height'],
                            fovy=fovy, render_mode='gaussian', background=source['background'],
                            yaw=0, pitch=0, zoom=1, pan_x=0, pan_y=0))
    checked = validate_plan(dict(operation='sequence', samples=samples), info)
    if 'background_rgb' in source:
        color = np.asarray(source['background_rgb'], dtype=float)
        if color.shape != (3,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
            raise ValueError('Invalid reference background')
        for sample in checked['samples']: sample['background_rgb'] = color.tolist()
    return dict(plan, samples=checked['samples'], requested_settings=plan['samples'][0],
                reference_alignment='source camera, native RGB resolution, Gaussian and source background')


def compact_counts(values):
    """Lossless integer packing; calibrated fractional counts retain float64."""
    values = np.asarray(values)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Invalid event counts')
    if np.all(values == np.floor(values)):
        maximum = float(values.max(initial=0))
        for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
            if maximum < float(np.iinfo(dtype).max)+1:
                return values.astype(dtype)
    return values.astype(np.float64)


def bundle_sources(root, paths, progress=lambda **value: None, include_paths=()):
    """Keep small configuration files; identify external models by path and hash."""
    directory = root / 'configs'
    directory.mkdir()
    records, seen, files = [], set(), []
    included = {Path(path).resolve() for path in include_paths}
    if any(path.stat().st_size>1024**2 for path in included):
        raise ValueError('Explicit compact parameter file exceeds 1 MiB')

    def visit(path):
        path = Path(path).resolve()
        if path in seen:
            return
        seen.add(path)
        if not path.is_file():
            records.append(dict(source=str(path), available=False))
            return
        files.append(path)
        if path.suffix == '.json':
            def walk(value):
                if isinstance(value, dict):
                    for child in value.values(): walk(child)
                elif isinstance(value, list):
                    for child in value: walk(child)
                elif isinstance(value, str) and len(value) < 4096:
                    candidate = path.parent / value
                    try:
                        if candidate.is_file(): visit(candidate)
                    except OSError:
                        pass
            walk(json.loads(path.read_text()))
    for path in [*paths, *included]:
        if path is not None: visit(path)
    total, completed = sum(path.stat().st_size for path in files), 0
    progress(completed=0, total=total, unit='bytes')
    for path in files:
        def advance(size):
            nonlocal completed
            completed += size
            progress(completed=completed, total=total, unit='bytes', detail=path.name)
        digest = sha256(path, advance)
        record = dict(source=str(path), sha256=digest, bytes=path.stat().st_size, available=True, included=False)
        if (path.suffix == '.json' and path.stat().st_size <= 1024**2) or path in included:
            dest = directory / (digest + path.suffix)
            if not dest.exists(): shutil.copyfile(path, dest)
            record.update(file=str(dest.relative_to(root)), included=True)
        records.append(record)
    return records


class StateCapture:
    def __init__(self, root, index, sampling=None, *, evidence=None, frame=None):
        self.root, self.index, self._states = root, index, {}
        self.sampling = sampling if sampling is not None else {}
        self.evidence, self.frame = evidence or {}, frame
        self.encode_ms = 0.

    @property
    def states(self):
        return [self._states[side] for side in sorted(self._states)]

    def point_ids(self, count, groups=None, side=None):
        side = len(self._states) if side is None else side
        if side in self.evidence and side not in self.sampling:
            record = self.evidence[side]
            self.sampling[side] = (record['point_count'], record['point_ids'])
        if side not in self.sampling:
            ids = np.linspace(0, count-1, min(count, 256), dtype=np.int64)
            if groups is not None:
                labels = array(groups)
                selected = []
                for group in np.unique(labels):
                    if group < 0: continue
                    members = np.flatnonzero(labels == group)
                    selected.extend(members[np.linspace(0, len(members)-1, min(len(members), 64), dtype=np.int64)])
                if selected:
                    ids = np.asarray(selected, dtype=np.int64)
                    if len(ids) > 1024: ids = ids[np.linspace(0, len(ids)-1, 1024, dtype=np.int64)]
            self.sampling[side] = (count, ids)
        if self.sampling[side][0] != count:
            raise ValueError('Point count changed during export; fixed trajectories would be invalid')
        return self.sampling[side][1]

    def __call__(self, state, alpha, side=None):
        side = len(self._states) if side is None else side
        if side in self._states: raise ValueError('Duplicate version capture')
        dest = self.root / 'versions' / str(side)
        dest.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        xyz = state.get('xyz', state.get('vertices'))
        groups = state.get('vertex_groups', state.get('point_groups'))
        preselected = 'point_ids' in state
        ids = state['point_ids'] if preselected else self.point_ids(len(xyz), groups, side=side)
        motion = {key: value for key, value in state.items() if key in (
            'group_quaternions', 'group_translations', 'joint_pairs', 'joint_pivots', 'joint_axes', 'joint_angles',
            'joint_twist_angles', 'joint_swing_angles')}
        if groups is not None: motion['point_groups'] = groups if preselected else groups[ids]
        if side in self.evidence:
            record = self.evidence[side]
            frame = record['frame_lookup'][self.frame]
            actual = xyz if preselected else xyz[ids]
            if not np.array_equal(ids, record['point_ids']) or not np.allclose(actual, record['points'][frame], atol=2e-6, rtol=0):
                raise ValueError('Motion evidence differs from actual rendered points; refusing stale transforms')
            for key in ('point_groups', 'joint_pairs', 'joint_pivots', 'joint_axes'):
                if key in record: motion[key] = record[key]
            for key in ('group_quaternions', 'group_translations', 'joint_angles', 'rigid_fit_max_error',
                        'joint_twist_angles', 'joint_swing_angles'):
                if key in record: motion[key] = record[key][frame]
        self._states[side] = dict(ids=ids, points=xyz if preselected else xyz[ids],
                                 count=state.get('point_count', len(xyz)), motion=motion)
        if alpha is not None:
            from PIL import Image
            mask = np.squeeze(alpha)
            # Coverage downsampling follows the output pixel grid. This is
            # predicted alpha, never the common evaluation mask.
            image = Image.fromarray(mask.astype(np.float32), mode='F')
            image = image.resize(self.size, Image.Resampling.BOX)
            (dest / 'alpha').mkdir(exist_ok=True)
            Image.fromarray(np.rint(np.clip(np.asarray(image), 0, 1)*65535).astype(np.uint16)).save(
                dest / 'alpha' / f'{self.index:06d}.png', compress_level=1)
        self.encode_ms += (time.perf_counter() - start)*1000


def camera_matches(camera, actual):
    return bool(camera and all(camera[k] == actual[k] for k in ('width', 'height'))
        and np.allclose(camera['K'], actual['K'], rtol=0, atol=1e-5)
        and np.allclose(camera['pose_opengl_c2w'], actual['pose_opengl_c2w'], rtol=0, atol=1e-6))


def aggregate_events(source, reference_path, previous, sample, target):
    """Sum only calibrated contiguous windows; never omit the three skipped bins."""
    start, end, cursor = previous['time_seconds'], sample['time_seconds'], sample['time_seconds']
    if previous['camera'] != sample['camera']: return 'Sample cameras differ'
    rows = sorted((row for row in source['samples'] if start < row['time_seconds'] <= end+1e-8),
                  key=lambda row: row['time_seconds'], reverse=True)
    positive = negative = None
    for row in rows:
        if abs(row['time_seconds']-cursor) > 1e-8: continue
        if not row.get('events') or not camera_matches(row.get('camera', source.get('camera')), sample['camera']):
            return 'Missing event bin or mismatched camera inside sampled interval'
        with np.load(reference_path.parent / row['events'], allow_pickle=False) as event:
            if not {'positive', 'negative', 't0', 't1'} <= set(event.files):
                raise ValueError('Events require positive/negative counts and t0/t1')
            t0, t1 = float(event['t0']), float(event['t1'])
            if not np.isfinite([t0, t1]).all() or t0 >= t1:
                raise ValueError('Invalid event interval')
            if abs(t1-cursor) > 1e-8 or t0 < start-1e-8:
                return 'Event windows do not cover the sampled interval exactly'
            counts = []
            for polarity in ('positive', 'negative'):
                values = event[polarity]
                if values.shape != (sample['camera']['height'], sample['camera']['width']) or not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError('Invalid calibrated event counts')
                counts.append(values.astype(np.float64))
        if positive is None: positive, negative = counts
        else: positive += counts[0]; negative += counts[1]
        cursor = t0
        if abs(cursor-start) <= 1e-8:
            np.savez_compressed(target, positive=compact_counts(positive), negative=compact_counts(negative), t0=start, t1=end)
            return None
    return 'Event windows are incomplete for the sampled interval'


def export_reference(root, reference_path, preview, samples, progress=lambda **value: None):
    """Only an administrator's calibrated, independent data can supply GT."""
    result = dict(samples=[], unavailable={})
    directory = root / 'reference'
    directory.mkdir()
    source = None
    if reference_path:
        reference_path = Path(reference_path)
        source = json.loads(reference_path.read_text())
        if source.get('format') != 'scgs_experiment_reference_v1':
            raise ValueError('Unsupported experiment reference format')
        save_json(directory / 'source.json', source)
        if not source.get('provenance'):
            raise ValueError('Reference needs independent data and mask provenance')
        for key in ('part_labels', 'mask_definition', 'source_motion_scope'):
            if key in source: result[key] = source[key]
        if source.get('source_motion'):
            path = reference_path.parent/source['source_motion']
            if path.stat().st_size > 16*1024**2: raise ValueError('Reference motion exceeds 16 MiB')
            with np.load(path, allow_pickle=False) as motion:
                frames = motion['frames']; indices = [np.flatnonzero(frames == s['frame']) for s in samples]
                if all(len(i)==1 for i in indices):
                    ids = np.asarray([i[0] for i in indices])
                    np.savez_compressed(directory/'source-motion.npz', frames=frames[ids],
                        times_seconds=motion['times_seconds'][ids], object_matrices=motion['object_matrices'][ids],
                        source_camera_pose=motion['source_camera_pose'], part_ids=motion['part_ids'])
                    result['source_motion'] = 'reference/source-motion.npz'
    lookup = {row['frame']: row for row in source['samples']} if source else {}
    if source and len(lookup) != len(source['samples']):
        raise ValueError('Duplicate reference frame')
    copied = {}
    for index, sample in enumerate(samples):
        record = dict(sample=index, frame=sample['frame'], aligned=False)
        ref = lookup.get(sample['frame'])
        if ref:
            camera = ref.get('camera', source.get('camera'))
            if not camera:
                raise ValueError('Reference camera calibration is missing')
            record.update(time_seconds=ref['time_seconds'], camera=camera)
            if 'background' in source:
                record['background_matches'] = (sample['background'] == source['background']
                    and sample.get('background_rgb') == source.get('background_rgb'))
            actual = sample['camera']
            record['aligned'] = bool(abs(ref['time_seconds']-sample['time_seconds']) <= 1e-8 and camera_matches(camera, actual))
            if not record['aligned']: record['reason'] = 'GT camera/time mismatch; unused reference files omitted'
            for key in (('rgb', 'foreground_mask', 'part_mask') if record['aligned'] else ()):
                if not ref.get(key): continue
                path = reference_path.parent / ref[key]
                target = directory / f'{index:06d}-{key}{".npz" if key == "events" else ".png"}'
                cache_key = (key, str(path.resolve()))
                if cache_key in copied:
                    target = copied[cache_key]
                else:
                    with Image.open(path) as image:
                        if image.size != (camera['width'], camera['height']):
                            raise ValueError('Reference image and calibration sizes differ')
                        if key == 'rgb': image = image.convert('RGB')
                        else:
                            labels = np.asarray(image)
                            if labels.ndim != 2 or labels.dtype.kind not in 'bui' or np.any(labels < 0):
                                raise ValueError('Reference masks must be nonnegative single-channel integer labels')
                            if key == 'foreground_mask':
                                image = Image.fromarray(labels != 0)
                            else:
                                if labels.max(initial=0) > 65535: raise ValueError('Part labels exceed PNG uint16')
                                image = Image.fromarray(labels.astype(np.uint8 if labels.max(initial=0) < 256 else np.uint16))
                        image.save(target, compress_level=1)
                    copied[cache_key] = target
                record[key] = str(target.relative_to(root))
            record['event_calibration'] = source.get('event_calibration')
            if 'event_split' in ref: record['event_split'] = ref['event_split']
            if index and record['aligned'] and result['samples'][-1].get('aligned'):
                target = directory / f'{index:06d}-events.npz'
                reason = aggregate_events(source, reference_path, samples[index-1], sample, target)
                if reason: record['events_unavailable'] = reason
                else: record['events'] = str(target.relative_to(root))
        elif preview is not None:
            record['reason'] = 'UI preview is not calibrated ground truth; omitted from compact export'
        else:
            record['reason'] = 'No independent reference configured'
        result['samples'].append(record)
        progress(completed=index+1, total=len(samples), unit='frames')
    for key in ('rgb', 'foreground_mask', 'part_mask', 'events'):
        if not any(key in row for row in result['samples']):
            result['unavailable'][key] = (source or {}).get('unavailable', {}).get(key, 'No independent source configured; not generated from method outputs')
    return result


def cuda_stats(reset=False):
    # Fake renderers / CPU tests need no torch installation.
    torch = sys.modules.get('torch')
    if torch is None or not torch.cuda.is_available(): return None
    torch.cuda.synchronize()
    if reset: torch.cuda.reset_peak_memory_stats()
    return dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())


def save_bundle_guide(root, manifest, histories):
    """Describe exactly what was saved, including partial/missing inputs."""
    references = manifest.get('reference', {}).get('samples', [])
    expected = len(manifest['samples'])
    availability = dict(frame_count=expected, version_count=len(manifest['versions']), reference={}, motion={},
                        limitations=manifest['unavailable'])
    missing = []
    for key, label in (('rgb', '独立 RGB 真值'), ('foreground_mask', '固定前景掩码'),
                       ('part_mask', '固定部件掩码'), ('events', '原始事件计数')):
        # A sequence's first frame has no preceding event interval.
        required = max(expected-1, 0) if key == 'events' else expected
        rows = references[1:] if key == 'events' else references
        count = sum(key in row for row in rows)
        aligned = sum(key in row and row.get('aligned', False) for row in rows)
        availability['reference'][key] = dict(saved=count, expected=required, aligned=aligned)
        if count < required: missing.append(f'{label}：{count}/{required}')
        elif aligned < required: missing.append(f'{label}：{required-aligned} 个采样点与导出相机/时间不匹配')
    for side, rows in histories.items():
        present = set.intersection(*(set(row['motion']) for row in rows))
        availability['motion'][str(side)] = sorted(present)
        for key, label in (('group_quaternions', '部件旋转'), ('group_translations', '部件平移'),
                           ('joint_angles', '关节角'), ('joint_pairs', '关节连接'), ('joint_pivots', '关节枢轴')):
            if key not in present:
                missing.append(f'版本 {side}：转动包含偏轴分量；已保存完整旋转、扭转投影与偏轴角，不能作为纯单轴铰链角'
                    if key=='joint_angles' and 'joint_twist_angles' in present else f'版本 {side}：未提供已验证的{label}')
    if any(not record['available'] for record in manifest['sources']):
        missing.append('部分来源文件不可用；参见 manifest.json 的 sources')
    if any(row.get('events') and not all((row.get('event_calibration') or {}).get(k)
            for k in ('contrast_positive', 'contrast_negative')) for row in references):
        missing.append('已保存原始事件计数，但缺少正负对比阈值标定；暂不能计算事件响应误差')
    availability['missing_data'] = missing
    save_json(root/'data_availability.json', availability)
    manifest['data_availability'] = 'data_availability.json'
    (root/'evaluate.py').write_text(
        '"""Evaluate this folder using its saved evaluator; NumPy and Pillow required."""\n'
        'from pathlib import Path\nimport sys\n\n'
        "root = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(root / 'code'))\n"
        'from scripts.evaluation.evaluate_experiment import main\n\n'
        "if __name__ == '__main__':\n    sys.argv.insert(1, str(root))\n    main()\n", encoding='utf-8')
    notes = '\n'.join('- '+item for item in missing) or '无缺失项目。'
    (root/'README.txt').write_text(
        f"SC-GS 实验数据 — {manifest['scene']}\n\n"
        f'本文件夹保存 {expected} 个采样时刻、{len(manifest["versions"])} 个版本。\n'
        '相机、真实时间和显示模式见 plan.json。无损图像、覆盖率和固定点轨迹见 versions/。\n'
        '公共参考数据见 reference/，小型配置与显式指定的小型参数文件见 configs/，来源路径与文件哈希见 manifest.json。\n'
        '不含逐帧完整模型、原始资产副本或 UI 展示视频。原始模型保留在项目目录。\n'
        '本包可独立计算指标；重新渲染必须使用清单所标识的原模型及对应项目代码。\n'
        'timings.json 记录本次导出耗时（含关键数据捕获）；严格性能比较请单独运行性能测量。\n'
        'code/ 仅包含离线评测需要的三个脚本。\n\n'
        '独立评测（安装 NumPy、Pillow 后，在本文件夹执行）：\n'
        '  python evaluate.py\n'
        '可选 LPIPS（另需 torch、lpips 及预训练权重）：\n'
        '  python evaluate.py --lpips\n'
        '结果写入 metrics.json；评测前会核对导出文件哈希。\n\n'
        f'未提供或未对齐的数据（详见 data_availability.json）：\n{notes}\n'
        '指标适用范围及来源限制见 data_availability.json 的 limitations 和 manifest.json 的 scene_info.motion_evidence。\n'
        '固定点在各版本内部对应，跨版本点对应未验证；推断的部件与关节先验不等同于独立物理关节真值。\n', encoding='utf-8')


def check_export_budget(root, archive_reserve=False):
    size = sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
    if size > MAX_EXPORT_BYTES:
        raise ValueError(f'导出超过 {MAX_EXPORT_BYTES/1024**3:g} GiB 容量上限，请缩小时间范围或降低分辨率')
    required = (size if archive_reserve else 0) + 32*1024**2
    if shutil.disk_usage(root).free < required:
        raise OSError(28, '服务器空间不足，导出已停止，临时文件将自动清理')
    return size


def run_export(renderer, plan, destination, scene, *, reference_path=None, preview=None, progress=lambda value: None):
    """Own only this job's files; retain one finalized ZIP, clean failed work."""
    root = Path(destination)
    archive = Path(str(root)+'.zip')
    partial = Path(str(archive)+'.partial')
    if archive.exists() or partial.exists(): raise FileExistsError(archive)
    root.mkdir(parents=True, exist_ok=False)
    last_check = 0.

    def checked_progress(update):
        nonlocal last_check
        now = time.monotonic()
        if now-last_check >= .5:
            check_export_budget(root)
            last_check = now
        progress(update)
    try:
        check_export_budget(root)
        _write_export(renderer, plan, root, scene, reference_path=reference_path, preview=preview, progress=checked_progress)
        shutil.rmtree(root)
        return archive
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        partial.unlink(missing_ok=True)
        archive.unlink(missing_ok=True)
        raise


def _write_export(renderer, plan, destination, scene, *, reference_path=None, preview=None, progress=lambda value: None):
    reporter = ExportProgress(plan['operation'], progress)
    reporter.update('prepare')
    root = Path(destination)
    started = time.perf_counter()
    if getattr(renderer, 'loaded_source_sha256', None) and sha256(renderer.path) != renderer.loaded_source_sha256:
        raise ValueError('模型来源文件在加载后已改变；请重新加载模型')
    samples = [{**request, 'camera': camera_record(renderer, request)} for request in plan['samples']]
    def version_labels(sample):
        return renderer.info.get('mode_version_labels', {}).get(sample['render_mode'], renderer.info.get('version_labels', []))
    labels = version_labels(samples[0])
    if any(version_labels(sample) != labels for sample in samples):
        raise ValueError('同一个导出计划必须包含相同的版本；请分别导出三版本模式和双版本模式')
    save_json(root / 'plan.json', dict(plan, format='scgs_experiment_plan_v1', scene=scene, samples=samples))
    manifest = dict(format=FORMAT, scene=scene, operation=plan['operation'], scene_info=renderer.info,
                    export_profile='evaluation_compact_v2',
                    model_state_policy='fixed sampled points and verified part transforms only; full models are external',
                    sample_step=plan.get('sample_step'),
                    samples=samples, versions=[], environment=dict(python=sys.version, platform=platform.platform()),
                    unavailable={'physical_joint_ground_truth': 'Declared source-prior joints are not independently measured physical joint ground truth',
                                 'cross_version_keypoints': 'No verified cross-version correspondence between Gaussian row IDs'},
                    timing_scope='serial render of all panels, GPU synchronization and CPU pixel transfer; no JPEG cache',
                    load_seconds=getattr(renderer, 'load_seconds', None))
    def configuration(component):
        return dict(renderer_class=type(component).__name__, cache=getattr(component, 'cache', None),
            side_sampling=getattr(component, 'side_sampling', None), supersamples=getattr(component, 'supersamples', None),
            modules={key: getattr(component, key, None) is not None for key in
                     ('correction', 'completion', 'deformation', 'field', 'mesh', 'hybrid', 'bindings', 'surface_refinement', 'detail_normals')})
    manifest['renderer_config'] = configuration(renderer)
    if hasattr(renderer, 'middle'):
        manifest['renderer_config'] = dict(renderer_class=type(renderer).__name__,
            components=[dict(versions=[0, len(labels)-1], **configuration(renderer.base)),
                        dict(versions=[1] if len(labels)==3 else [], **configuration(renderer.middle))])
    manifest['environment']['packages'] = {}
    for package in ('numpy', 'Pillow', 'torch', 'aiohttp', 'nvdiffrast', 'diff-gaussian-rasterization', 'lpips'):
        try: manifest['environment']['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    torch = sys.modules.get('torch')
    if torch is not None:
        manifest['environment']['cuda'] = torch.version.cuda
        if torch.cuda.is_available(): manifest['environment']['gpu'] = torch.cuda.get_device_name()
    histories, measurements, export_measurements, sampling = {}, [], [], {}
    benchmark = plan['operation'] == 'benchmark'

    def render(index, capture=False):
        sample = samples[index]
        request = {key: value for key, value in sample.items() if key not in ('camera', 'time_seconds')}
        request.update(_raw=True, _camera_pose=sample['camera']['pose_opengl_c2w'])
        if 'background_rgb' in sample: request['_background_rgb'] = sample['background_rgb']
        evidence = getattr(renderer, 'motion_evidence', {}) if len(labels)==3 else {}
        recorder = StateCapture(root, index, sampling, evidence=evidence, frame=sample['frame']) if capture else None
        if recorder:
            recorder.size = (sample['width'], sample['height'])
            request['_capture'] = recorder
        cuda_stats()
        start = time.perf_counter()
        output = renderer.render(request)
        memory = cuda_stats()
        elapsed = (time.perf_counter()-start)*1000
        if not isinstance(output, dict) or 'panels' not in output:
            raise ValueError('Renderer does not support lossless experiment export')
        panels = output['panels']
        if labels and len(panels) != len(labels):
            raise ValueError('Rendered versions do not match the export plan')
        encode_ms = 0.
        if capture:
            if len(recorder.states) != len(panels):
                raise ValueError('Every version must export its actual sampled geometry')
            encode_started = time.perf_counter()
            for side, pixels in enumerate(panels):
                if pixels.shape != (sample['height'], sample['width'], 3) or pixels.dtype != np.uint8:
                    raise ValueError('Invalid raw render dimensions/type')
                dest = root / 'versions' / str(side) / 'frames'
                dest.mkdir(exist_ok=True)
                Image.fromarray(pixels).save(dest / f'{index:06d}.png', compress_level=1)
                histories.setdefault(side, []).append(recorder.states[side])
            encode_ms = (time.perf_counter()-encode_started)*1000 + recorder.encode_ms
        return dict(sample=index, render_and_transfer_ms=elapsed, encode_write_ms=encode_ms,
                    panel_ms=request.get('_panel_timings', []),
                    includes_state_capture=capture, memory=memory)

    if benchmark:
        reporter.update('warmup', total=plan['warmup'], unit='renders')
        for i in range(plan['warmup']):
            render(i % len(samples))
            reporter.update('warmup', completed=i+1, total=plan['warmup'], unit='renders')
        baseline = cuda_stats(reset=True)
        reporter.update('benchmark', total=len(samples)*plan['repeats'], unit='renders')
        for repeat in range(plan['repeats']):
            for index in range(len(samples)):
                measurements.append(dict(repeat=repeat, **render(index)))
                reporter.update('benchmark', completed=len(measurements), total=len(samples)*plan['repeats'],
                                unit='renders', detail=f'第 {repeat+1}/{plan["repeats"]} 轮 · 帧 {index+1}/{len(samples)}')
        manifest['memory_baseline'] = baseline
    if plan['operation'] != 'benchmark':
        reporter.update('frames', total=len(samples), unit='frames')
        cuda_stats(reset=True)
        for index in range(len(samples)):
            export_measurements.append(render(index, capture=True))
            reporter.update('frames', completed=index+1, total=len(samples), unit='frames')
        reporter.update('motion', total=len(histories), unit='versions')
        for side, rows in histories.items():
            if len(rows) != len(samples):
                raise ValueError('A version is missing frames or sampled geometry')
            if any(not np.array_equal(row['ids'], rows[0]['ids']) for row in rows):
                raise ValueError('Point identities changed during export')
            motion = {}
            for key in rows[0]['motion']:
                if all(key in row['motion'] for row in rows):
                    motion[key] = (rows[0]['motion'][key] if key in ('point_groups', 'joint_pairs', 'joint_pivots', 'joint_axes')
                                   else np.stack([row['motion'][key] for row in rows]))
            np.savez_compressed(root / 'versions' / str(side) / 'motion.npz', **motion,
                point_ids=rows[0]['ids'], points=np.stack([row['points'] for row in rows]),
                times_seconds=np.asarray([sample['time_seconds'] for sample in samples]))
            label = labels[side] if side < len(labels) else renderer.info.get('left_label' if side == 0 else 'right_label', f'版本 {side+1}')
            manifest['versions'].append(dict(id=str(side), label=label,
                point_count=rows[0]['count'], sampled_point_count=len(rows[0]['ids']),
                point_correspondence='fixed row IDs within this version; cross-version correspondence unverified'))
            reporter.update('motion', completed=len(manifest['versions']), total=len(histories), unit='versions')
        reporter.update('reference', total=len(samples), unit='frames')
        manifest['reference'] = export_reference(root, reference_path, preview, samples,
            progress=lambda **value: reporter.update('reference', **value))
    reporter.update('sources')
    base = getattr(renderer, 'base', None)
    manifest['sources'] = bundle_sources(root, [renderer.path, getattr(base, 'path', None), reference_path, *getattr(renderer, 'source_paths', [])],
        progress=lambda **value: reporter.update('sources', **value), include_paths=getattr(renderer, 'compact_sources', ()))
    # Keep only portable evaluator dependencies, identify renderer code by hashes.
    reporter.update('code')
    code_root = Path(__file__).resolve().parents[1]
    code_files = [code_root/name for name in ('scripts/evaluation/evaluate_experiment.py',
                                             'utils/experiment_format.py', 'utils/mesh_motion.py')]
    for index, path in enumerate(code_files):
        target = root / 'code' / path.relative_to(code_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        reporter.update('code', completed=index+1, total=len(code_files), unit='files', detail=path.name)
    implementation = set()
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(code_root) and path.suffix == '.py' and path.is_file(): implementation.add(path)
    manifest['implementation'] = {str(path.relative_to(code_root)): sha256(path) for path in sorted(implementation)}
    save_json(root / 'timings.json', dict(operation='benchmark' if benchmark else plan['operation'],
        warmup=plan['warmup'] if benchmark else 0, repeats=plan['repeats'] if benchmark else 1,
        samples=measurements if benchmark else export_measurements, load_seconds=manifest['load_seconds'],
        export_wall_seconds=time.perf_counter()-started,
        memory_scope='PyTorch process-wide allocations; includes all loaded scenes, not per-version memory'))
    save_bundle_guide(root, manifest, histories)
    reporter.update('hashes')
    files = [path for path in sorted(root.rglob('*')) if path.is_file()]
    total, completed = sum(path.stat().st_size for path in files), 0
    reporter.update('hashes', total=total, unit='bytes')
    manifest['files'] = {}
    for path in files:
        def advance_hash(size):
            nonlocal completed
            completed += size
            reporter.update('hashes', completed=completed, total=total, unit='bytes', detail=str(path.relative_to(root)))
        manifest['files'][str(path.relative_to(root))] = dict(sha256=sha256(path, advance_hash), bytes=path.stat().st_size)
    save_json(root / 'manifest.json', manifest)
    archive = Path(str(root)+'.zip')
    partial = Path(str(archive)+'.partial')
    files.append(root / 'manifest.json')
    total, completed = sum(path.stat().st_size for path in files), 0
    reporter.update('archive', total=total, unit='bytes')
    check_export_budget(root, archive_reserve=True)
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for path in files:
            relative = path.relative_to(root)
            def advance_archive(size):
                nonlocal completed
                completed += size
                reporter.update('archive', completed=completed, total=total, unit='bytes', detail=str(relative))
            # A single root folder survives extraction into Downloads.
            name = str(Path(root.name)/relative if plan['operation'] == 'complete' else relative)
            info = zipfile.ZipInfo.from_file(path, name)
            if path.suffix in ('.json', '.txt', '.py'):
                info.compress_type = zipfile.ZIP_DEFLATED
                info._compresslevel = 1
            with path.open('rb') as source, bundle.open(info, 'w', force_zip64=True) as target:
                copy_chunks(source, target, advance_archive)
    partial.replace(archive)
    return archive
