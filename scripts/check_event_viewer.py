"""Check the live GPU viewer protocol and its agreement with audited candidate renders."""
import argparse
import asyncio
import io
import json
from pathlib import Path
import struct
import sys
import time

import aiohttp
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.render_sampling import require_same_sampling


def verify_live_sampling(scene, audit, baseline_metadata=None):
    if scene.get('comparison'):
        require_same_sampling(scene.get('right_render_sampling', {}), audit)
        if baseline_metadata is not None:
            require_same_sampling(scene.get('left_render_sampling', {}), baseline_metadata)


def verify_live_cleanup(scene, audit):
    """Source PLY identity alone cannot identify an independent attribute layer."""
    detail, live_detail = audit.get('protected_surface_detail'), scene.get('protected_surface_detail')
    if detail is None:
        if live_detail is not None:
            raise ValueError('Live viewer has an unaudited protected surface detail layer')
    elif (not isinstance(live_detail, dict)
          or any(live_detail.get(field+'_sha256') != detail[field+'_sha256']
                 or live_detail.get(field+'_sha256') != audit.get('artifact_sha256', {}).get(artifact)
                 for field, artifact in (('report', 'surface_detail_report'), ('extension', 'surface_detail_extension')))):
        raise ValueError('Live protected surface detail differs from the fresh event audit')
    residual, live_residual = audit.get('color_residual'), scene.get('color_residual')
    if residual is None:
        if live_residual is not None:
            raise ValueError('Live viewer has an unaudited color residual')
    elif (not isinstance(live_residual, dict)
          or any(live_residual.get(field+'_sha256') != residual[field+'_sha256']
                 or live_residual.get(field+'_sha256') != audit.get('artifact_sha256', {}).get(artifact)
                 for field, artifact in (('report', 'color_residual_report'), ('residual', 'color_residual')))):
        raise ValueError('Live color residual identity differs from the fresh event audit')
    surface, live_surface = audit.get('static_surface_rebuild'), scene.get('static_surface_rebuild')
    if surface is None:
        if live_surface is not None:
            raise ValueError('Live viewer has an unaudited static surface extension')
    elif (not isinstance(live_surface, dict)
          or any(live_surface.get(field+'_sha256') != surface[field+'_sha256']
                 or live_surface.get(field+'_sha256') != audit.get('artifact_sha256', {}).get(artifact)
                 for field, artifact in (('report', 'static_surface_report'),
                                         ('extension', 'static_surface_extension')))):
        raise ValueError('Live static surface identity differs from the fresh event audit')
    contract, actual = audit.get('gaussian_cleanup'), scene.get('gaussian_cleanup')
    if contract is None:
        if actual is not None:
            raise ValueError('Live viewer has an unaudited Gaussian attribute overlay')
        return
    if (not isinstance(actual, dict)
            or actual.get('report_sha256') != audit['artifact_sha256'].get('cleanup_report')
            or actual.get('overlay_sha256') != audit['artifact_sha256'].get('cleanup_overlay')
            or actual.get('report_sha256') != contract['report_sha256']
            or actual.get('overlay_sha256') != contract['overlay_sha256']):
        raise ValueError('Live Gaussian cleanup identity differs from the fresh event audit')


async def check(args):
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(args.url + '/api/status') as response:
            response.raise_for_status()
            status = await response.json()
        if not status.get('ready'):
            raise RuntimeError(f'Viewer not ready: {status}')
        scene = status['scene']
        if not scene.get('event_validation', {}).get('optimized'):
            raise ValueError('Viewer is not showing the event-corrected candidate')
        audit = json.loads((args.candidate / 'event-audit.json').read_text())
        baseline_metadata = json.loads((args.baseline / 'cache.json').read_text()) if args.baseline else None
        verify_live_sampling(scene, audit, baseline_metadata)
        verify_live_cleanup(scene, audit)
        if scene.get('comparison'):
            for field, key in (('candidate_correction_sha256', 'correction'), ('right_source_sha256', 'source_ply')):
                if scene.get(field) != audit['artifact_sha256'][key]:
                    raise ValueError(f'Viewer is showing a different candidate/source: {field}')
        if (scene['event_validation']['summary'] != audit['summary']['all']
                or scene['event_validation']['approved'] != audit['approved_for_export']):
            raise ValueError('Viewer event decision differs from the actual candidate audit')
        expected = np.load(args.candidate / 'rendered_gray.npy', mmap_mode='r', allow_pickle=False)
        baseline = np.load(args.baseline / 'rendered_gray.npy', mmap_mode='r', allow_pickle=False) if args.baseline else None
        if scene.get('comparison') and baseline is None:
            raise ValueError('Comparison verification requires --baseline')
        results = {'scene': scene, 'samples': []}
        async with session.ws_connect(args.url + '/ws', origin=args.url) as ws:
            async def receive():
                packet = await ws.receive(timeout=30)
                if packet.type != aiohttp.WSMsgType.BINARY:
                    raise ValueError(f'Expected rendered frame: {packet}')
                size = struct.unpack('<I', packet.data[:4])[0]
                return json.loads(packet.data[4:4 + size]), packet.data[4 + size:]

            for frame, yaw in ((0, 0), (100, 0), (200, 0), (100, 35)):
                await ws.send_json({'id': len(results['samples']) + 1, 'frame': frame,
                                    'width': 640, 'height': 640, 'fovy': scene['fovy'],
                                    'background': 'black', 'quality': 95, 'yaw': yaw})
                meta, jpeg = await receive()
                image = np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB')) / 255
                width = 1280 if scene.get('comparison') else 640
                if image.shape != (640, width, 3) or meta['frame'] != frame:
                    raise ValueError('Incorrect frame response')
                gray = image[:, -640:] @ np.array([.299, .587, .114])
                left_error = None
                if scene.get('comparison'):
                    if meta['left_frame'] != frame or meta['right_frame'] != frame:
                        raise ValueError('Comparison panels are not synchronized')
                    left_gray = image[:, :640] @ np.array([.299, .587, .114])
                    if yaw == 0:
                        left_error = float(np.abs(left_gray - baseline[frame]).mean())
                        if left_error > .015:
                            raise ValueError('Left view is not the unchanged baseline')
                    if frame == 100 and yaw == 0 and np.abs(gray-left_gray).mean() < .001:
                        raise ValueError('Left and right unexpectedly show the same result')
                error = float(np.abs(gray - expected[frame]).mean())
                if yaw == 0 and error > .015:
                    raise ValueError(f'Viewer differs from audited Gaussian rendering: frame={frame}, MAE={error}')
                if frame == 100 and yaw == 0:
                    reference = gray
                if yaw != 0 and np.abs(gray - reference).mean() < .005:
                    raise ValueError('Changing camera did not change the rendered view')
                results['samples'].append({'frame': frame, 'yaw': yaw, 'render_ms': meta['render_ms'],
                                           'audit_render_mae': error if yaw == 0 else None,
                                           'left_baseline_mae': left_error})
                (args.output / f'ui_frame_{frame:04d}_yaw_{yaw}.jpg').write_bytes(jpeg)
            if args.multiview_render is not None:
                if args.parent_multiview_render is None:
                    raise ValueError('Novel-view cleanup verification needs the parent render too')
                with np.load(args.multiview_render, allow_pickle=False) as a, np.load(args.parent_multiview_render, allow_pickle=False) as b:
                    view_meta = json.loads(str(a['metadata_json']))
                    parent_meta = json.loads(str(b['metadata_json']))
                    if (view_meta['views'] != parent_meta['views']
                            or view_meta['artifact_sha256']['correction'] != audit['artifact_sha256']['correction']):
                        raise ValueError('Novel-view reference does not identify the current candidate/cameras')
                    results['novel_view_checks'] = []
                    for vi in (6, 17, 32):
                        view = view_meta['views'][vi]
                        await ws.send_json({'id': 100+vi, 'frame': view['frame'],
                            'width': 640, 'height': 640, 'fovy': scene['fovy'], 'quality': 95,
                            'background': 'black', **{k: view[k] for k in ('yaw', 'pitch', 'zoom', 'pan_x', 'pan_y')}})
                        meta, jpeg = await receive()
                        actual = np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB'), np.float32)[:, -640:]/255
                        actual_gray = actual @ np.array([.299, .587, .114])
                        target, parent_gray = a['gray'][vi], b['gray'][vi]
                        changed = np.abs(target-parent_gray) > .04
                        error = float(np.abs(actual_gray-target).mean())
                        changed_error = float(np.abs(actual_gray[changed]-target[changed]).mean()) if changed.any() else None
                        if error > .006 or (changed_error is not None and changed_error > .02):
                            raise ValueError('Live novel-view image does not apply the audited Gaussian attributes')
                        results['novel_view_checks'].append({'view_index': vi, 'mean_absolute_error': error,
                            'changed_pixels': int(changed.sum()), 'changed_region_mae': changed_error})
                        (args.output/f'ui_novel_view_{vi:02d}.jpg').write_bytes(jpeg)
            start = time.monotonic()
            await ws.send_json({'type': 'stream', 'id': 10, 'frame': 0, 'rate': 30, 'loop': True,
                                'width': 640, 'height': 640, 'fovy': scene['fovy'],
                                'background': 'black', 'quality': scene.get('default_stream_quality', 78)})
            times = []
            for seq in range(60):
                meta, _ = await receive()
                if meta['stream']['seq'] != seq or meta['frame'] != seq * 2:
                    raise ValueError('Playback is not advancing at the requested source timing')
                if scene.get('comparison') and meta['left_frame'] != meta['right_frame']:
                    raise ValueError('Comparison stream lost synchronization')
                times.append(meta['render_ms'] + meta['encode_ms'])
                if seq % 4 == 3:
                    await ws.send_json({'type': 'ack', 'id': 10, 'seq': seq})
            await ws.send_json({'type': 'stop'})
            results['stream'] = {'frames_checked': 60, 'quality': scene.get('default_stream_quality', 78),
                                  'mean_render_encode_ms': float(np.mean(times)),
                                  'server_delivery_fps': 60 / (time.monotonic() - start),
                                  'note': 'Server throughput only; not measured browser display FPS.'}
        (args.output / 'viewer-check.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({'samples': results['samples'], 'stream': results['stream']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8892')
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--multiview-render', type=Path)
    parser.add_argument('--parent-multiview-render', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    asyncio.run(check(args))


if __name__ == '__main__':
    main()
