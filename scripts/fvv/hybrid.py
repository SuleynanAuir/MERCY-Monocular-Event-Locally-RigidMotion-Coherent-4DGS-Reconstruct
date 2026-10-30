"""Run only inside the separate upstream Hybrid environment; see integration guide."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import math
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from common import LimitedWriter, digest, usage, validate_pose, model_timestamp


def revision(repo):
    marker = Path(repo) / 'UPSTREAM_REVISION'
    if marker.is_file():
        value = marker.read_text().strip()
        if not re.fullmatch(r'[0-9a-f]{40}', value):
            raise ValueError('Invalid upstream archive revision marker')
        return value
    return subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()


def setup(repo, config):
    # Upstream's scene/utils/renderer names collide with MERCY: process isolation
    # and path precedence are mandatory. Never import this module in viewer.py.
    sys.path.insert(0, str(repo))
    from omegaconf import OmegaConf
    from arguments import ModelParams, PipelineParams, OptimizationParams
    parser = argparse.ArgumentParser()
    mp, pp, op = ModelParams(parser), PipelineParams(parser), OptimizationParams(parser)
    args = parser.parse_args([])
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    for key, value in cfg.items():
        if isinstance(value, dict):
            vars(args).update(value)
        else:
            setattr(args, key, value)
    if getattr(args, 'gaussian_dim', None) != 4 or args.env_map_res:
        raise ValueError('Phase 1 supports Hybrid 4D checkpoints without environment maps')
    return args, mp, pp, op


def load_checkpoint(path, device):
    import torch
    import numpy as np
    # Upstream camera normalization may serialize a NumPy scalar. Permit only
    # scalar/dtype constructors, never arbitrary checkpoint globals.
    scalar = np.core.multiarray.scalar
    allowed = [scalar, np.dtype, type(np.dtype('float32')), type(np.dtype('float64'))]
    if not hasattr(torch.serialization, 'safe_globals'):
        # PyTorch 2.1 can load our tensor-only packed artifacts. Legacy source
        # checkpoints containing NumPy scalars require packing on >=2.4.
        return torch.load(path, map_location=device, weights_only=True)
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location=device, weights_only=True)


def compact(state):
    """Preserve every inference tensor, dtype and coefficient; drop optimizer only."""
    import torch
    params, iteration = state
    if len(params) != 28:
        raise ValueError('Unsupported upstream checkpoint schema (expected Hybrid 28 fields)')
    values = list(params)
    for i in (7, 8, 9, 10, 25, 26, 27):
        values[i] = torch.empty(0)
    values[11] = None
    values[12] = float(values[12])
    return (tuple(v.detach().cpu() if torch.is_tensor(v) else v for v in values), iteration)


def save_bounded(state, path, budget):
    import torch
    temporary = path.with_suffix('.partial')
    remaining = budget - usage(path.parent)
    if remaining <= 0 or shutil.disk_usage(path.parent).free < remaining + 1024**3:
        raise OSError('Insufficient reserved storage; require budget plus 1 GiB free')
    try:
        with temporary.open('xb') as f:
            torch.save(state, LimitedWriter(f, remaining))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def manifest(args, settings, root):
    duration = [float(v) / settings.frame_ratio for v in settings.time_duration]
    if not all(math.isfinite(v) for v in duration) or duration[1] <= duration[0]:
        raise ValueError('Invalid model time duration')
    validate_pose(dict(position=args.position, yaw=args.yaw, pitch=args.pitch, time=duration[0], width=640, height=480), duration)
    data = dict(schema=1, trained_at=model_timestamp(), backend='ohsngjun/3D-4DGS', revision=revision(args.repo),
                scene=args.scene, checkpoint='model.pth', sha256=digest(root/'model.pth'),
                config='effective.yaml', config_sha256=digest(root/'effective.yaml'),
                time_duration=duration, position=args.position, yaw=args.yaw, pitch=args.pitch,
                fov=float(getattr(settings, 'viewer_fov', 60)),
                bytes=usage(root), budget_bytes=int(args.budget_gib * 1024**3))
    (root/'manifest.json').write_text(json.dumps(data, indent=2))


def prepare(args):
    from omegaconf import OmegaConf
    root = args.output.resolve()
    # Never overwrite an existing run or accept a path inside original assets.
    root.mkdir(parents=True, exist_ok=False)
    try:
        settings, mp, pp, op = setup(args.repo, args.config)
        cfg = OmegaConf.load(args.config)
        cfg.ModelParams.model_path = str(root)
        if args.command == 'train':
            if (args.source/'sparse').exists() or not (args.source/'points3d.ply').is_file():
                raise ValueError('Require prepared Blender-format points3d.ply; source is never generated or modified')
            if not (args.source/'transforms_train.json').is_file():
                raise ValueError('Expected upstream-preprocessed calibrated dynamic dataset')
            cfg.ModelParams.source_path = str(args.source.resolve())
        OmegaConf.save(cfg, root/'effective.yaml')
        settings, mp, pp, op = setup(args.repo, root/'effective.yaml')
        budget = int(args.budget_gib * 1024**3)
        started = time.monotonic()
        if args.command == 'pack':
            import torch
            # Only tensor checkpoints; no arbitrary pickle execution.
            state = load_checkpoint(args.checkpoint, 'cpu')
            save_bounded(compact(state), root/'model.pth', budget)
        else:
            import main as upstream
            import torch
            upstream.TENSORBOARD_FOUND = False  # no growing image/event logs
            upstream.setup_seed(6666)
            # A fixed-camera sequence has zero camera-center extent. Derive
            # the explicit spatial learning scale from the real G0 bounds.
            if getattr(settings, 'single_view_extent', None) is not None:
                extent = float(settings.single_view_extent)
                if not math.isfinite(extent) or extent <= 0:
                    raise ValueError('single_view_extent must be positive')
                OriginalScene = upstream.Scene
                class FixedCameraScene(OriginalScene):
                    def __init__(self, *a, **kw):
                        super().__init__(*a, **kw)
                        if self.cameras_extent < 1e-6:
                            self.cameras_extent = extent
                            self.gaussians.spatial_lr_scale = extent
                upstream.Scene = FixedCameraScene
            original_report = upstream.training_report
            def measured_report(writer, iteration, l1, loss, *a, **kw):
                result = original_report(writer, iteration, l1, loss, *a, **kw)
                if iteration == 2 or iteration % 100 == 0:
                    row = dict(iteration=iteration, loss=float(loss.detach()), l1=float(l1.detach()),
                               elapsed_seconds=time.monotonic()-started,
                               allocated_bytes=torch.cuda.memory_allocated(),
                               validation_psnr=float(result) if iteration == settings.iterations else None)
                    with (root/'training-metrics.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                return result
            upstream.training_report = measured_report
            original_save = torch.save
            def inference_save(value, filename, *a, **kw):
                # Final-only validation emits one checkpoint. Bypass recursively
                # patched torch.save while writing the bounded inference artifact.
                torch.save = original_save
                try:
                    save_bounded(compact(value), root/'model.pth', budget)
                finally:
                    torch.save = inference_save
            torch.save = inference_save
            try:
                upstream.training(mp.extract(settings), op.extract(settings), pp.extract(settings),
                    [settings.iterations], [], None, -1, settings.gaussian_dim,
                    list(settings.time_duration), settings.num_pts, settings.num_pts_ratio,
                    settings.rot_4d, settings.force_sh_3d, settings.batch_size)
            finally:
                torch.save = original_save
        if not (root/'model.pth').is_file():
            raise RuntimeError('Upstream did not produce a final checkpoint')
        # Only disposable files created by this new isolated job are removed.
        for name in ('input.ply', 'cameras.json', 'cfg_args'):
            (root/name).unlink(missing_ok=True)
        manifest(args, settings, root)
        (root/'timing.json').write_text(json.dumps({'offline_seconds': time.monotonic()-started}))
        if usage(root) > budget:
            raise OSError('Output exceeded storage budget')
        print(root/'manifest.json')
    except BaseException:
        # Keep failed runs inspectable, never publish them as ready.
        (root/'manifest.json').unlink(missing_ok=True)
        raise


class Renderer:
    def __init__(self, args):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Hybrid FVV requires CUDA')
        self.torch = torch
        self.root = args.model.resolve()
        self.meta = json.loads((self.root/'manifest.json').read_text())
        if self.meta['schema'] != 1 or self.meta['revision'] != revision(args.repo):
            raise ValueError('Upstream revision differs from the trained artifact')
        # Fixed names, no manifest-controlled traversal or executable configs.
        for name, key in [('model.pth', 'sha256'), ('effective.yaml', 'config_sha256')]:
            if digest(self.root/name) != self.meta[key]:
                raise ValueError('Artifact hash mismatch: '+name)
        settings, _, pp, _ = setup(args.repo, self.root/'effective.yaml')
        from scene import GaussianModel
        from gaussian_renderer import render
        self.render_fn = render
        self.pipe = pp.extract(settings)
        self.model = GaussianModel(settings.sh_degree, gaussian_dim=4,
            time_duration=self.meta['time_duration'], rot_4d=settings.rot_4d,
            force_sh_3d=settings.force_sh_3d, sh_degree_t=2 if self.pipe.eval_shfs_4d else 0)
        state, _ = load_checkpoint(self.root/'model.pth', 'cuda')
        self.model.restore(state, None)
        self.temporal_sigma = float(getattr(args, 'temporal_sigma', 0))
        if not math.isfinite(self.temporal_sigma) or not 0 <= self.temporal_sigma <= 1/30:
            raise ValueError('Temporal sigma must be between 0 and 1/30 seconds')
        if self.temporal_sigma:
            if not settings.rot_4d or self.pipe.compute_cov3D_python:
                raise ValueError('Temporal filtering requires the CUDA rot4D path')
            from temporal_kernel import select, REVISION
            if self.meta['revision'] != REVISION:
                raise ValueError('Temporal filtering requires the pinned Hybrid revision')
            select(args.repo, True)
            self.pipe.opa_threshold = self.temporal_sigma
        self.bg = torch.tensor([1.,1.,1.] if settings.white_background else [0.,0.,0.], device='cuda')

    def frame(self, pose, raw=False, pixels=False):
        import numpy as np
        from utils.graphics_utils import getProjectionMatrix
        t = self.torch
        yaw, pitch = map(math.radians, (pose['yaw'], pose['pitch']))
        forward = np.array([math.sin(yaw)*math.cos(pitch), math.sin(pitch), math.cos(yaw)*math.cos(pitch)])
        right = np.array([math.cos(yaw), 0., -math.sin(yaw)])
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward])
        view = np.eye(4, dtype=np.float32)
        view[:3,:3] = rotation
        view[:3,3] = -rotation @ np.array(pose['position'])
        fy = math.radians(pose['fov'])
        fx = 2 * math.atan(math.tan(fy/2)*pose['width']/pose['height'])
        with t.no_grad():
            world = t.tensor(view, device='cuda').T.contiguous()
            proj = getProjectionMatrix(znear=0.01, zfar=10000., fovX=fx, fovY=fy).T.cuda()
            camera = SimpleNamespace(FoVx=fx, FoVy=fy, image_width=pose['width'], image_height=pose['height'],
                timestamp=pose['time'], world_view_transform=world, full_proj_transform=world @ proj,
                camera_center=t.tensor(pose['position'], device='cuda', dtype=t.float32))
            if pose.get('temporal_samples', 1) == 3:
                # Same camera, short symmetric shutter; no historical image
                # reuse, screen-space trails or extra CPU copies per sample.
                lo, hi = self.meta['time_duration']
                rgb = None
                for offset, weight in [(-1/240, .25), (0, .5), (1/240, .25)]:
                    camera.timestamp = max(lo, min(hi, pose['time'] + offset))
                    sample = self.render_fn(camera, self.model, self.pipe, self.bg)['render']
                    rgb = sample * weight if rgb is None else rgb + sample * weight
            else:
                rgb = self.render_fn(camera, self.model, self.pipe, self.bg)['render']
            if raw:
                return rgb.detach().cpu()
            array = (rgb.clamp(0,1)*255).byte().permute(1,2,0).contiguous().cpu().numpy()
        if pixels:
            return array
        return self.encode(array,int(pose.get('jpeg_quality',95)))

    @staticmethod
    def encode(array, quality):
        from PIL import Image
        stream = io.BytesIO()
        # Optimized Huffman tables reduce bandwidth without changing the
        # decoded pixels at the selected quality and chroma sampling.
        Image.fromarray(array).save(stream, format='JPEG', quality=quality, subsampling=0,optimize=True)
        return stream.getvalue()


async def stream_frames(ws, renderer, duration, render_pool, encode_pool):
    """Overlap GPU rendering with CPU encoding; queue at most two pixel buffers."""
    from aiohttp import WSMsgType
    loop=asyncio.get_running_loop()
    frames=asyncio.Queue(maxsize=2)
    poses=asyncio.Queue(maxsize=8)
    latest_camera=None
    async def receive():
        nonlocal latest_camera
        previous_id=-1
        async for msg in ws:
            if msg.type!=WSMsgType.TEXT:break
            try:
                pose=validate_pose(json.loads(msg.data),duration)
                request_id=pose['request_id']
                if request_id is not None:
                    if request_id<=previous_id:raise ValueError('Frame identifiers must increase')
                    previous_id=request_id
            except (ValueError,KeyError,TypeError) as exc:
                await ws.send_json({'error':str(exc)});break
            # Refresh queued camera positions without discarding frame credits.
            # Dropping a whole TCP burst starves playback for another RTT.
            latest_camera=pose if request_id is not None and pose['coalesce'] else None
            await poses.put(pose)
        await poses.put(None)
    async def render():
        while True:
            pose=await poses.get()
            if pose is None:
                await frames.put(None);return
            if pose['coalesce'] and latest_camera is not None:
                pose={**pose,**{key:latest_camera[key] for key in ('position','yaw','pitch','fov')}}
            started=time.monotonic()
            pixels=await loop.run_in_executor(render_pool,lambda:renderer.frame(pose,pixels=True))
            await frames.put((pixels,pose['jpeg_quality'],pose['request_id']))
            await asyncio.sleep(max(0,1/120-(time.monotonic()-started)))
    async def encode():
        while True:
            item=await frames.get()
            if item is None:return
            pixels,quality,request_id=item
            jpeg=await loop.run_in_executor(encode_pool,renderer.encode,pixels,quality)
            if request_id is not None:jpeg=b'FVV1'+request_id.to_bytes(4,'big')+jpeg
            await ws.send_bytes(jpeg)
    tasks=[asyncio.create_task(receive()),asyncio.create_task(render()),asyncio.create_task(encode())]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)


def serve(args):
    from aiohttp import web
    pool = ThreadPoolExecutor(max_workers=1)
    encode_pool = ThreadPoolExecutor(max_workers=1)
    state = {'ready': False, 'message': '正在加载 Hybrid 检查点…'}
    renderer = None
    busy = False
    active_ws = None
    @web.middleware
    async def local(request, handler):
        if request.host.split(':')[0] not in {'127.0.0.1', 'localhost'}:
            raise web.HTTPForbidden()
        if request.headers.get('Origin') not in (None, f'{request.scheme}://{request.host}'):
            raise web.HTTPForbidden()
        response = await handler(request)
        if not response.prepared:
            response.headers['Cache-Control'] = 'no-store'
        return response
    app = web.Application(middlewares=[local], client_max_size=4096)
    async def load_model(model):
        nonlocal renderer
        previous, previous_state = renderer, dict(state)
        if active_ws is not None:
            await active_ws.close(code=1012, message=b'Model switching')
        state.update(ready=False, message='正在校验并加载模型…')
        try:
            candidate = await asyncio.get_running_loop().run_in_executor(
                pool, Renderer, SimpleNamespace(repo=args.repo, model=model,
                                               temporal_sigma=getattr(args, 'temporal_sigma', 0)))
            pose = dict(position=candidate.meta['position'], yaw=candidate.meta['yaw'], pitch=candidate.meta['pitch'],
                        time=candidate.meta['time_duration'][0], width=640, height=480, fov=candidate.meta.get('fov',60))
            for _ in range(3):
                await asyncio.get_running_loop().run_in_executor(pool, candidate.frame, pose)
            renderer = candidate
            args.model = model
            state.update(ready=True, message='Hybrid 模型已就绪', **renderer.meta)
            state['temporal_sigma_seconds'] = renderer.temporal_sigma
        except Exception:
            if previous is not None:
                state.clear(); state.update(previous_state)
            raise
    async def load():
        try:
            await load_model(args.model)
        except Exception as exc:
            state.update(ready=False, message=f'加载失败：{exc}')
    async def lifecycle(app):
        task = asyncio.create_task(load())
        yield
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        pool.shutdown(wait=True, cancel_futures=True)
        encode_pool.shutdown(wait=True, cancel_futures=True)
    app.cleanup_ctx.append(lifecycle)
    def matches(request):
        return request.query.get('scene', state.get('scene')) == state.get('scene')
    async def status(request):
        if state['ready'] and not matches(request):
            return web.json_response({'ready': False, 'message': '该场景尚未加载 Hybrid 模型，当前服务场景为 '+state['scene']})
        return web.json_response({**state, 'busy': busy, 'frame_protocol':'fvv-jpeg-v1', 'max_in_flight': 8, 'max_fps': 120})
    async def socket(request):
        nonlocal busy, active_ws
        if not state['ready'] or busy or not matches(request):
            raise web.HTTPServiceUnavailable()
        busy = True
        ws = web.WebSocketResponse(max_msg_size=4096, heartbeat=20, compress=False)
        try:
            await ws.prepare(request)
            active_ws = ws
            try:
                await stream_frames(ws,renderer,state['time_duration'],pool,encode_pool)
            except Exception:
                logging.exception('FVV frame failed')
                await ws.close(code=1011,message=b'Render failed; inspect GPU service')
        finally:
            busy = False
            if active_ws is ws:
                active_ws = None
            if renderer is not None:
                await asyncio.get_running_loop().run_in_executor(pool,renderer.torch.cuda.empty_cache)
        return ws
    from management import ModelManager
    manager = ModelManager(args, revision(args.repo), lambda: args.model.resolve(), load_model)
    manager.install(app)
    app.router.add_get('/status', status)
    app.router.add_get('/ws', socket)
    web.run_app(app, host='127.0.0.1', port=args.port)


def benchmark(args):
    import numpy as np
    renderer = Renderer(args)
    t = renderer.torch
    meta = renderer.meta
    lo, hi = meta['time_duration']
    poses = [dict(position=[meta['position'][0] + .1*math.sin(i), *meta['position'][1:]],
                  yaw=meta['yaw'] + 20*math.sin(i), pitch=meta['pitch'], time=lo+(hi-lo)*i/11,
                  width=960, height=640, fov=60) for i in range(12)]
    for pose in poses[:3]:
        renderer.frame(pose)
    durations = []
    for i in range(120):
        start = time.monotonic()
        renderer.frame(poses[i % len(poses)])
        durations.append((time.monotonic()-start)*1000)
    result = dict(scene=meta['scene'], revision=meta['revision'], model_sha256=meta['sha256'],
                  gpu=t.cuda.get_device_name(), torch=t.__version__, width=960, height=640,
                  scope='GPU render + CPU copy + JPEG, excludes network/browser',
                  mean_ms=float(np.mean(durations)), p95_ms=float(np.percentile(durations,95)),
                  mean_fps=1000/float(np.mean(durations)), samples_ms=durations)
    if args.checkpoint:
        # Compare raw floating-point renders, before JPEG, against the source.
        expected = [renderer.frame(pose, raw=True) for pose in poses]
        state = load_checkpoint(args.checkpoint, 'cuda')
        renderer.model.restore(state[0], None)
        del state
        errors = []
        for pose, reference in zip(poses, expected):
            actual = renderer.frame(pose, raw=True)
            errors.append(float((actual-reference).abs().max()))
        result['source_checkpoint_sha256'] = digest(args.checkpoint)
        result['raw_max_abs_error'] = max(errors)
        result['raw_parity_pass'] = result['raw_max_abs_error'] <= 1e-6
    print(json.dumps(result, indent=2))


def supervise(args):
    """One writer per artifact store; monitor *all* generated files, not just PTH."""
    root = args.output.resolve()
    store = root.parent
    store.mkdir(parents=True, exist_ok=True)
    limit = int(args.total_budget_gib * 1024**3)
    job_limit = int(args.budget_gib * 1024**3)
    lock = store / '.fvv-writer.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    process = None
    paused_viewers = []
    try:
        if root.exists():
            raise FileExistsError('Use a new output directory; existing assets are never overwritten')
        if usage(store) + job_limit > limit or shutil.disk_usage(store).free < job_limit + 1024**3:
            raise OSError('Artifact store budget or free-space reservation exhausted')
        env = dict(os.environ, MERCY_FVV_WORKER='1', PYTHONDONTWRITEBYTECODE='1')
        if args.command == 'train':
            registries = json.loads(os.environ.get('MERCY_FVV_RENDER_REGISTRIES', '[]'))
            if registries:
                from training_resources import pause
                print('Training priority: pausing original viewer services...', flush=True)
                pause(registries, paused_viewers)
            print('Checking CUDA math/indexing before training...', flush=True)
            try:
                health = subprocess.run([sys.executable, str(Path(__file__).with_name('cuda_health.py'))],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
            except subprocess.TimeoutExpired as exc:
                root.mkdir(exist_ok=False)
                (root/'cuda-health.log').write_bytes((exc.output or b'')[-65536:])
                raise RuntimeError('GPU health check timed out; training was not started') from exc
            if health.returncode:
                root.mkdir(exist_ok=False)
                (root/'cuda-health.log').write_bytes(health.stdout[-65536:])
                print(health.stdout[-4096:].decode(errors='replace'), flush=True)
                raise RuntimeError('GPU health check failed; training was not started')
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                                   env=env, start_new_session=True)
        start = time.monotonic()
        while process.poll() is None:
            if usage(store) > limit or usage(root) > job_limit or shutil.disk_usage(store).free < 1024**3:
                raise OSError('Storage watchdog stopped the offline job')
            if time.monotonic() - start > args.max_minutes * 60:
                raise TimeoutError('Offline time budget reached; no ready model is published')
            time.sleep(.25)
        if process.returncode:
            raise RuntimeError('Offline worker failed; inspect its console output')
        if args.command == 'train':
            (root/'cuda-health.log').write_bytes(health.stdout[-65536:])
        if usage(store) > limit or usage(root) > job_limit:
            raise OSError('Final artifact exceeds storage budget')
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if process is not None:
            (root/'manifest.json').unlink(missing_ok=True)
        raise
    finally:
        try:
            if paused_viewers:
                from training_resources import resume
                print('Restoring original viewer services...', flush=True)
                resume(paused_viewers)
        finally:
            os.close(fd)
            lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'pack'):
        p = commands.add_parser(name)
        p.add_argument('--config', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--scene', required=True)
        p.add_argument('--budget-gib', type=float, default=8)
        p.add_argument('--total-budget-gib', type=float, default=32)
        p.add_argument('--max-minutes', type=float, default=30)
        p.add_argument('--position', type=float, nargs=3, default=[0,0,-3])
        p.add_argument('--yaw', type=float, default=0)
        p.add_argument('--pitch', type=float, default=0)
        if name == 'train':
            p.add_argument('--source', type=Path, required=True)
        else:
            p.add_argument('--checkpoint', type=Path, required=True)
    p = commands.add_parser('serve')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--port', type=int, default=8897)
    p.add_argument('--profiles', type=Path, help='Trusted offline training profiles JSON')
    p.add_argument('--temporal-sigma', type=float, default=0,
                   help='Opt-in temporal opacity antialiasing, seconds (0 disables; max 1/30)')
    p = commands.add_parser('benchmark')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, help='Optional original checkpoint for raw render parity')
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    if args.command == 'serve':
        serve(args)
    elif args.command == 'benchmark':
        benchmark(args)
    else:
        if not math.isfinite(args.budget_gib) or args.budget_gib <= 0:
            parser.error('--budget-gib must be positive and finite')
        for value in (args.total_budget_gib, args.max_minutes):
            if not math.isfinite(value) or value <= 0:
                parser.error('Storage and time budgets must be positive and finite')
        if os.environ.get('MERCY_FVV_WORKER') == '1':
            prepare(args)
        else:
            supervise(args)

if __name__ == '__main__':
    main()
