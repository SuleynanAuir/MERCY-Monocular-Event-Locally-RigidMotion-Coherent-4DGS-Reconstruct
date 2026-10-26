"""Evaluate actual UI WebSocket images, including its white-clay display finish.

No alternate renderer or postprocessing is used. Saved JPEGs are the exact
bytes delivered to the browser; the reference is the same HTTP asset as UI.
Scores against RGB describe display discrepancy, not material-free geometry.
"""
import argparse
import asyncio
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from .protocol import save_json


def decode_packet(packet, frame, size):
    if len(packet) < 5:
        raise ValueError('Truncated UI packet')
    length = struct.unpack('<I', packet[:4])[0]
    if length > len(packet)-4:
        raise ValueError('Truncated UI header')
    meta = json.loads(packet[4:4+length])
    if (meta.get('frame') != frame or meta.get('panel_width') != size
            or meta.get('left_frame') != frame or meta.get('right_frame') != frame):
        raise ValueError('UI frame or comparison identity mismatch')
    jpeg = packet[4+length:]
    pixels = np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB'))
    if pixels.shape != (size, 2*size, 3):
        raise ValueError('UI image dimensions mismatch')
    return meta, jpeg, pixels


def score(reference, predicted):
    if reference.shape != predicted.shape:
        raise ValueError('Reference and UI render dimensions differ')
    ref = reference.astype(np.float64)/255
    pred = predicted.astype(np.float64)/255
    mse = float(np.square(ref-pred).mean())
    return dict(rgb_l1=float(np.abs(ref-pred).mean()),
                psnr=float(-10*np.log10(mse)) if mse else None,
                exact_match=mse == 0,
                ssim=float(structural_similarity(ref, pred, channel_axis=2,
                    data_range=1, gaussian_weights=True, sigma=1.5,
                    use_sample_covariance=False)))


async def capture(args):
    import aiohttp
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('frames', 'references', 'visualization'):
        (output/folder).mkdir()
    base = args.url.rstrip('/')
    params = {'scene': args.scene}
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(base+'/api/status', params=params) as response:
            response.raise_for_status()
            status = await response.json()
        if not status.get('ready'):
            raise ValueError('UI is not ready: '+json.dumps(status, ensure_ascii=False))
        scene = status['scene']
        count = scene['frame_count']
        # Native reference pixels avoid adding an arbitrary resize to scores.
        reference_info = scene.get('reference', {})
        size = reference_info.get('width')
        if size is None or reference_info.get('height') != size:
            raise ValueError('A verified square UI RGB reference is required')
        frames = list(range(count)) if args.frames is None else args.frames
        if len(set(frames)) != len(frames) or any(i < 0 or i >= count for i in frames):
            raise ValueError('Invalid frame list')
        fovy = scene['fovy']
        settings = dict(width=size, height=size, quality=95, render_mode=args.mode,
                        clay_style='solid', background='light', yaw=0, pitch=0,
                        zoom=1, pan_x=0, pan_y=0, fovy=fovy)
        record = dict(format='ui_websocket_display_evaluation_v1', scene=args.scene,
            status=status, request=settings, source_url=base,
            scope='full_sequence' if len(frames) == count else 'explicit_frame_subset',
            metric_scope='UI white-clay/presentation pixels versus UI RGB; appearance discrepancy, not independent reconstruction accuracy',
            exposure='Existing G0 and animation exposure unknown; no held-out generalization claim',
            unavailable_metrics={key: reason for key, reason in [
                ('event_relative_l1', 'Requires physical/native intensity and raw held-out events; white-clay brightness is not sensor radiance'),
                ('event_pass', 'Not inferred from the colored event preview'),
                ('visible_missing', 'Requires renderer alpha and a declared independent coverage reference'),
                ('back_missing', 'No independent registered rear-view observation'),
                ('joint_gap_max', 'Requires the actual version geometry and joint transforms'),
                ('angular_jerk', 'Requires the actual version joint trajectory'),
                ('position_jerk', 'Requires corresponding 3D trajectories, not screenshot differences')]},
            frames=[])
        save_json(output/'protocol.json', record)
        async with session.ws_connect(base+'/ws', params=params, max_msg_size=32*1024**2) as ws:
            for i in frames:
                await ws.send_json(dict(settings, type='frame', id=i, frame=i))
                message = await ws.receive(timeout=300)
                if message.type != aiohttp.WSMsgType.BINARY:
                    raise ValueError('UI did not return an image: '+str(message.data))
                meta, jpeg, pair = decode_packet(message.data, i, size)
                (output/'frames'/f'{i:04d}.jpg').write_bytes(jpeg)
                async with session.get(base+f'/api/reference/{i}', params=params) as response:
                    response.raise_for_status()
                    data = await response.read()
                (output/'references'/f'{i:04d}.image').write_bytes(data)
                refpair = np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))
                if refpair.shape != (size, 2*size, 3):
                    raise ValueError('Unexpected UI reference layout')
                reference = refpair[:, :size]
                metrics = {side: score(reference, pair[:, k*size:(k+1)*size])
                           for k, side in enumerate(('left', 'right'))}
                record['frames'].append(dict(frame=i, time_seconds=scene['times_seconds'][i],
                    jpeg_sha256=hashlib.sha256(jpeg).hexdigest(),
                    reference_sha256=hashlib.sha256(data).hexdigest(), metadata=meta, metrics=metrics))
                if i in {frames[0], frames[len(frames)//2], frames[-1]}:
                    Image.fromarray(np.concatenate((reference, pair), axis=1)).save(
                        output/'visualization'/f'actual-ui-frame-{i:04d}.png')
                if len(record['frames']) % 10 == 0:
                    save_json(output/'progress.json', dict(completed=len(record['frames']), total=len(frames)))
                    print(args.scene, args.mode, len(record['frames']), '/', len(frames), flush=True)
        record['means'] = {side: {key: float(np.mean(values)) if all(v is not None for v in values) else None
            for key in ('rgb_l1', 'psnr', 'ssim')
            for values in [[row['metrics'][side][key] for row in record['frames']]]}
            for side in ('left', 'right')}
        save_json(output/'results.json', record)
        print(json.dumps(record['means']), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8894')
    p.add_argument('--scene', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--mode', choices=('clay', 'gaussian', 'pseudocolor'), default='clay')
    p.add_argument('--frames', type=int, nargs='+')
    asyncio.run(capture(p.parse_args()))


if __name__ == '__main__':
    main()
