"""Loopback-only browser UI: render baked Gaussian states on the server GPU.

The NPZ stores each Gaussian's position and orientation at every frame.
Optional RGB/event references share the exact animation timeline.
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import io
import json
import logging
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from aiohttp import WSMsgType, web
import numpy as np

from utils.viewer_state import (BACKGROUNDS, STREAM_WINDOW, image_packet, orbit_pose,
                                cached_image_reference, stream_frame, validate_request, validate_stream)


LOG = logging.getLogger("gaussian-viewer")
STATIC = Path(__file__).resolve().parent / "viewer"


class Renderer:
    def __init__(self, path, cache):
        self.path, self.cache = Path(path), cache
        self.status = {"ready": False, "progress": 0, "message": "等待加载高斯序列"}

    def progress(self, value, message):
        self.status = {"ready": False, "progress": value, "message": message}
        LOG.info("%s (%d%%)", message, value)

    def load(self):
        import torch
        from scene import GaussianModel
        from utils.gaussian_sequence import GaussianSequence

        if self.path.suffix != ".npz":
            raise ValueError("Realtime viewer requires a baked .npz; use pack_gaussian_sequence.py first")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required on the rendering server")
        self.progress(5, "正在解压动态高斯 NPZ，首次加载约需数十秒")
        with GaussianSequence(self.path) as sequence, torch.no_grad():
            meta = sequence.metadata
            self.model = GaussianModel(meta["sh_degree"], with_motion_mask=False)
            self.progress(35, "校验首帧并加载颜色、尺度和不透明度")
            with tempfile.TemporaryFile() as stream:
                stream.write(sequence.frame_bytes(0))
                stream.seek(0)
                self.model.load_ply(stream)
            for value in (self.model.get_xyz, self.model.get_rotation,
                          self.model.get_scaling, self.model.get_opacity, self.model.get_features):
                if not torch.isfinite(value).all().item():
                    raise ValueError("Non-finite activated Gaussian attributes")
            # These supplied assets contain DC-only color in a degree-3 PLY.
            # Removing exactly zero higher-order coefficients is lossless and
            # avoids copying hundreds of MB of zero features every frame.
            if torch.count_nonzero(self.model._features_rest).item() == 0:
                self.model._features_rest = torch.empty((len(self.model.get_xyz), 0, 3), device="cuda")
                self.model.active_sh_degree = self.model.max_sh_degree = 0
            del value
            torch.cuda.empty_cache()
            self.dynamic_indices = (None if sequence.dynamic_indices is None else
                                    torch.as_tensor(sequence.dynamic_indices.astype(np.int64), device="cuda"))
            n, frames, _ = sequence.poses.shape
            needed = n * frames * 7 * 4
            free, _ = torch.cuda.mem_get_info()
            device = self.cache
            if device == "auto":
                device = "gpu" if free > needed + 1536 * 1024**2 else "cpu"
            if device == "gpu" and free < needed + 512 * 1024**2:
                raise RuntimeError("Insufficient GPU memory; restart with --cache cpu")
            self.poses = torch.empty((frames, n, 7), dtype=torch.float32,
                                     device="cuda" if device == "gpu" else "cpu")
            for start in range(0, frames, 16):
                end = min(start + 16, frames)
                for index in range(start, end):
                    sequence.validate_pose_frame(index)
                chunk = np.ascontiguousarray(sequence.poses[:, start:end, :7].transpose(1, 0, 2))
                if not np.isfinite(chunk).all() or np.any(np.linalg.norm(chunk[:, :, 3:], axis=-1) < 1e-8):
                    raise ValueError("Invalid baked position or quaternion")
                self.poses[start:end].copy_(torch.from_numpy(chunk))
                self.progress(40 + int(50 * end / frames), "缓存逐帧高斯位置与朝向")
            camera = meta.get("preview_camera")
            if not camera:
                raise ValueError("Sequence needs preview_camera metadata for an initial view")
            self.base_pose = np.asarray(camera["pose"], dtype=np.float64)
            vertices = np.column_stack([sequence.base_vertices[k] for k in ("x", "y", "z")])
            center = camera.get("target", meta.get("motion_provenance_only", {}).get("center"))
            self.target = np.asarray(center if center is not None else
                                     (vertices.min(0) + vertices.max(0)) / 2, dtype=np.float64)
            self.pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            self.backgrounds = {key: torch.tensor(value, dtype=torch.float32, device="cuda")
                                for key, value in BACKGROUNDS.items()}
            self.info = {
                "name": meta.get('name', self.path.name), "description": meta.get('description'),
                "frame_count": frames, "gaussian_count": len(sequence.base_vertices),
                "dynamic_gaussian_count": n, "motion_label": meta.get("motion_label"),
                "default_stream_rate": meta.get("default_stream_rate", 60),
                "default_stream_quality": meta.get("default_stream_quality", 78),
                "default_still_quality": meta.get("default_still_quality", 85),
                "default_resolution": meta.get("default_resolution", 768),
                "fps": float(meta["fps"]), "duration_seconds": float(meta["duration_seconds"]),
                "times_seconds": [f["time_seconds"] for f in meta["frames"]],
                "rpm": [f.get("rpm") for f in meta["frames"]],
                "ramp_seconds": meta.get("motion_provenance_only", {}).get("ramp_seconds"),
                "cache": device, "cache_gib": round(needed / 1024**3, 2),
                "gpu": torch.cuda.get_device_name(), "fovy": float(camera["fovy_degrees"]),
                "baked": True, "streaming": True, "stream_window": STREAM_WINDOW,
                "event_validation": meta.get("event_validation"),
                "default_background": meta.get("default_background", "dark"),
            }
        self.progress(95, "预热 GPU 光栅化器")
        self.render(validate_request({"width": 512, "height": 512, "fovy": self.info["fovy"]}, frames))
        self.status = {"ready": True, "progress": 100, "message": "就绪", "scene": self.info}
        LOG.info("Ready: %s / %d Gaussians / %d moving / %d frames / %s cache",
                 self.path.name, self.info["gaussian_count"], n, frames, device)

    def render(self, request):
        import torch
        from PIL import Image
        from gaussian_renderer import render
        from render_ply import make_camera

        start = time.perf_counter()
        with torch.no_grad():
            pose = self.poses[request["frame"]].to("cuda")
            if self.dynamic_indices is None:
                self.model._xyz = pose[:, :3].contiguous()
                self.model._rotation = pose[:, 3:7].contiguous()
            else:
                self.model._xyz[self.dynamic_indices] = pose[:, :3]
                self.model._rotation[self.dynamic_indices] = pose[:, 3:7]
            camera = make_camera(orbit_pose(self.base_pose, self.target, request),
                                 request["width"], request["height"], np.deg2rad(request["fovy"]))
            result = render(camera, self.model, self.pipe, self.backgrounds[request["background"]],
                            0., 0., 0.)["render"]
            # HWC-contiguous on the GPU avoids Pillow's slow strided-array copy.
            pixels = (result.clamp(0, 1).permute(1, 2, 0) * 255).byte().contiguous().cpu().numpy()
        rendered = time.perf_counter()
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="JPEG", quality=request["quality"])
        return image_packet({"id": request["id"], "frame": request["frame"],
                             **({"stream": request["_stream"]} if "_stream" in request else {}),
                             "width": request["width"], "height": request["height"],
                             "render_ms": round((rendered - start) * 1000, 2),
                             "encode_ms": round((time.perf_counter() - rendered) * 1000, 2)},
                            stream.getvalue())


@web.middleware
async def local_only(request, handler):
    # Loopback binding + Host/Origin checks also protect against DNS rebinding
    # and another website silently opening this local viewer's WebSocket.
    if request.url.host not in {"localhost", "127.0.0.1", "::1"}:
        raise web.HTTPForbidden(text="Use localhost or an SSH local forward")
    origin = request.headers.get("Origin")
    if origin and origin != f"{request.scheme}://{request.host}":
        raise web.HTTPForbidden(text="Cross-origin requests are not allowed")
    response = await handler(request)
    if not response.prepared:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; connect-src 'self'; img-src 'self' blob: data:; "
            "style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        response.headers["Cache-Control"] = "no-store"
    return response


def read_scene_catalog(path):
    """Only an administrator's catalog can select files or renderer types."""
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    scenes = {}
    for item in config['scenes']:
        key = item['id']
        if not isinstance(key, str) or not key or not all(c.isascii() and (c.isalnum() or c in '-_') for c in key) or key in scenes:
            raise ValueError('Scene IDs must be unique ASCII names')
        if item['type'] not in {'sequence', 'comparison', 'fan_mesh', 'scgs_v4', 'gaussian_pair'}:
            raise ValueError('Unknown scene renderer')
        if item['type'] == 'scgs_v4' and (item.get('base_scene') not in scenes
                or scenes[item['base_scene']]['type'] != 'comparison'):
            raise ValueError('V4 requires an earlier V3 comparison scene')
        if not isinstance(item.get('name'), str) or not item['name'].strip():
            raise ValueError('Scenes require a display name')
        if item.get('cache', 'auto') not in {'auto', 'cpu', 'gpu'}:
            raise ValueError('Unknown motion cache mode')
        asset = (path.parent / item['path']).resolve()
        if not asset.is_file():
            raise ValueError(f'Missing scene asset: {asset}')
        scenes[key] = {**item, 'path': asset}
        if item.get('reference_path'):
            scenes[key]['reference_path'] = (path.parent / item['reference_path']).resolve()
    default = config.get('default', next(iter(scenes), None))
    if default not in scenes:
        raise ValueError('Default scene must exist')
    return scenes, default


def create_app(path=None, cache="auto", renderer_factory=None, *, scenes=None, default_scene=None):
    if scenes is None:
        default_scene = 'default'
        scenes = {'default': {'path': path, 'name': '当前物件', 'factory': renderer_factory or Renderer}}
    default_scene = default_scene or next(iter(scenes))
    renderers = {}
    for key, spec in scenes.items():
        factory = spec.get('factory', Renderer)
        if spec.get('type') == 'scgs_v4':
            from utils.scgs_v4_viewer import SCGSV4ComparisonRenderer
            if spec.get('base_scene') not in renderers:
                raise ValueError('V4 base scene must load first')
            renderers[key] = SCGSV4ComparisonRenderer(spec['path'], spec.get('cache', cache),
                                                     renderers[spec['base_scene']])
            continue
        if spec.get('type') == 'comparison':
            from utils.comparison_renderer import ComparisonRenderer
            factory = ComparisonRenderer
        elif spec.get('type') == 'fan_mesh':
            from utils.fan_renderer import FanRenderer
            factory = FanRenderer
        elif spec.get('type') == 'gaussian_pair':
            from utils.gaussian_pair_renderer import GaussianPairRenderer
            factory = GaussianPairRenderer
        renderers[key] = factory(spec['path'], spec.get('cache', cache))
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gaussian-gpu")
    references = {}
    clients = set()
    loading = {'id': None}
    app = web.Application(middlewares=[local_only], client_max_size=4096)

    async def load_one(key, renderer):
        try:
            await asyncio.get_running_loop().run_in_executor(executor, renderer.load)
            if scenes[key].get('reference_path'):
                from utils.viewer_reference import ReferenceSequence
                reference = await asyncio.to_thread(ReferenceSequence, scenes[key]['reference_path'], renderer.info)
                references[key] = reference
                renderer.info['reference'] = reference.info
        except Exception as error:
            LOG.exception("Failed to load sequence")
            renderer.status = {"ready": False, "error": True, "progress": 0,
                               "message": "加载失败，请查看服务器日志", "detail": type(error).__name__}

    async def load():
        for key, renderer in renderers.items():
            loading['id'] = key
            await load_one(key, renderer)
        loading['id'] = None

    def selected(request):
        key = request.query.get('scene', default_scene)
        if key not in renderers:
            raise web.HTTPNotFound(text='Unknown scene')
        return key, renderers[key]

    async def lifecycle(app):
        task = asyncio.create_task(load())
        yield
        for ws in list(clients):
            await ws.close(code=1001, message=b"Server shutting down")
        await task
        executor.shutdown(wait=True)

    async def status(request):
        key, renderer = selected(request)
        state = renderer.status
        if state.get('ready') and loading['id'] is not None:
            busy = loading['id']
            state = {**state, 'ready': False, 'progress': renderers[busy].status.get('progress', 0),
                     'message': f"正在初始化 {scenes[busy]['name']}，完成后即可切换物件"}
        return web.json_response({**state, 'scene_id': key,
            'objects': [{'id': name, 'name': scenes[name]['name'],
                         'ready': value.status.get('ready', False)} for name, value in renderers.items()]})

    async def reference_frame(request):
        key, _ = selected(request)
        reference = references.get(key)
        index = request.match_info["frame"]
        if reference is None or not index.isascii() or not index.isdigit() or len(index) > 8:
            raise web.HTTPNotFound()
        try:
            path = reference.frame_path(int(index))
        except IndexError:
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Cache-Control": "private, max-age=0, must-revalidate"})

    async def static(request):
        filename = request.match_info.get("file", "index.html")
        if filename not in {"index.html", "style.css", "app.js", "playback.js", "references.js"}:
            raise web.HTTPNotFound()
        return web.FileResponse(STATIC / filename)

    async def websocket(request):
        _, renderer = selected(request)
        if not renderer.status.get("ready") or loading['id'] is not None:
            raise web.HTTPServiceUnavailable(text="Model is still loading")
        if len(clients) >= 4:
            raise web.HTTPServiceUnavailable(text="At most four viewer connections are allowed")
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=4096, compress=False)
        await ws.prepare(request)
        clients.add(ws)
        stream_task, flow = None, None

        async def stop_stream():
            nonlocal stream_task, flow
            if stream_task is not None:
                stream_task.cancel()
                with suppress(asyncio.CancelledError, ConnectionResetError):
                    await stream_task
            stream_task, flow = None, None

        async def produce(settings, credit):
            # Credits bound unpresented frames across GPU, TCP and browser to
            # 24. An independent receiver handles ACK/camera/stop immediately;
            # no per-frame network round trip is needed for continuous playback.
            seq = 0
            try:
                while not ws.closed:
                    while seq - credit["ack"] > STREAM_WINDOW:
                        credit["event"].clear()
                        await credit["event"].wait()
                    frame, last = stream_frame(settings, seq, renderer.info["fps"], renderer.info["frame_count"])
                    sample = {**settings, "frame": frame,
                              "_stream": {"seq": seq, "rate": settings["rate"]}}
                    if frame in credit['cached_frames']:
                        paired = (renderer.info.get('comparison', False)
                                  or sample.get('render_mode') in renderer.info.get('comparison_modes', []))
                        packet = cached_image_reference(sample, paired)
                    else:
                        packet = await asyncio.get_running_loop().run_in_executor(executor, renderer.render, sample)
                    credit['offered_frames'].add(frame)
                    credit["sent"] = seq
                    await ws.send_bytes(packet)
                    if last:
                        await ws.send_json({"type": "stream_end", "id": settings["id"], "seq": seq})
                        break
                    seq += 1
            except (ConnectionResetError, asyncio.CancelledError):
                return
            except Exception:
                LOG.exception("Streaming failed")
                if not ws.closed:
                    await ws.send_json({"type": "error", "message": "连续渲染失败，请降低清晰度后重试"})

        try:
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    break
                try:
                    raw = message.json()
                    if not isinstance(raw, dict):
                        raise ValueError("Request must be an object")
                    kind = raw.get("type", "frame")
                    if kind == "ack":
                        seq = raw.get("seq")
                        if flow is None or raw.get("id") != flow["id"]:
                            continue  # An acknowledgement for a replaced stream.
                        if isinstance(seq, bool) or not isinstance(seq, int) or not -1 <= seq <= flow["sent"]:
                            raise ValueError("Invalid stream acknowledgement")
                        retained = raw.get('cached_frames', [])
                        if (not isinstance(retained, list) or len(retained) > STREAM_WINDOW
                                or any(isinstance(f, bool) or not isinstance(f, int)
                                       or f not in flow['offered_frames'] for f in retained)):
                            raise ValueError('Invalid client frame cache acknowledgement')
                        # Optional protocol: old clients never acknowledge a
                        # cache, so continue to receive complete JPEG packets.
                        flow['cached_frames'].update(retained)
                        flow["ack"] = max(flow["ack"], seq)
                        flow["event"].set()
                        continue
                    if kind == "stop":
                        await stop_stream()
                        continue
                    if kind not in {"frame", "stream"}:
                        raise ValueError("Unknown request type")
                    data = (validate_stream if kind == "stream" else validate_request)(raw, renderer.info["frame_count"])
                    if data['render_mode'] in {'mesh', 'hybrid', 'clay', 'pseudocolor'} and not renderer.info.get(data['render_mode'], {}).get('available'):
                        await ws.send_json({'type': 'error', 'message': '当前场景未提供所选的 Mesh 或混合模式'})
                        continue
                except (ValueError, TypeError, OverflowError):
                    await ws.send_json({"type": "error", "message": "无效的相机或帧参数"})
                    continue
                await stop_stream()
                if kind == "stream":
                    flow = {"id": data["id"], "ack": -1, "sent": -1, "event": asyncio.Event(),
                            'offered_frames': set(), 'cached_frames': set()}
                    stream_task = asyncio.create_task(produce(data, flow))
                    continue
                try:
                    packet = await asyncio.get_running_loop().run_in_executor(executor, renderer.render, data)
                    await ws.send_bytes(packet)
                except (ConnectionResetError, asyncio.CancelledError):
                    break
                except Exception:
                    LOG.exception("Render failed")
                    await ws.send_json({"type": "error", "message": "渲染失败，请降低分辨率或查看服务器日志"})
        finally:
            await stop_stream()
            clients.discard(ws)
        return ws

    app.cleanup_ctx.append(lifecycle)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/reference/{frame}", reference_frame)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/", static)
    app.router.add_get("/{file}", static)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence_path", type=Path)
    source.add_argument("--comparison_path", type=Path, help="Manifest for synchronized before/after views")
    source.add_argument("--scenes_path", type=Path, help="JSON catalog of selectable objects")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1")
    parser.add_argument("--cache", choices=["auto", "gpu", "cpu"], default="auto")
    args = parser.parse_args()
    path = args.sequence_path or args.comparison_path or args.scenes_path
    if not path.is_file():
        parser.error("Sequence file does not exist")
    factory = None
    if args.comparison_path:
        from utils.comparison_renderer import ComparisonRenderer
        factory = ComparisonRenderer
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scenes, default = read_scene_catalog(path) if args.scenes_path else (None, None)
    web.run_app(create_app(path, args.cache, factory, scenes=scenes, default_scene=default), host=args.host, port=args.port,
                access_log=None)


if __name__ == "__main__":
    main()
