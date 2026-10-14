"""Read-only live UI orbit snapshots, with artifact identity and scope recorded.

These are Gaussian renderings, not unseen-view ground truth or event approval.
"""
import argparse
import asyncio
import hashlib
import io
import json
from pathlib import Path
import struct
import sys

import aiohttp
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.render_sampling import require_same_sampling


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def snapshot_plan():
    requests = [(frame, yaw, 0) for frame in (0, 100, 200, 300)
                for yaw in (0, 90, 180, 270)]
    requests += [(100, yaw, 0) for yaw in (45, 135, 225, 315)]
    requests += [(100, yaw, pitch) for pitch in (-35, 35) for yaw in range(0, 360, 45)]
    return requests


def verify_scene(scene, audit, source_hash, correction_hash):
    from scripts.check_event_viewer import verify_live_cleanup
    verify_live_cleanup(scene, audit)
    if not scene.get('comparison'):
        raise ValueError('Expected the paired Gaussian comparison UI')
    for scene_key, audit_key, actual in (
            ('right_source_sha256', 'source_ply', source_hash),
            ('candidate_correction_sha256', 'correction', correction_hash)):
        if scene.get(scene_key) != actual or audit['artifact_sha256'][audit_key] != actual:
            raise ValueError(f'Live scene/audit/file hash mismatch: {scene_key}')
    require_same_sampling(scene.get('right_render_sampling', {}), audit)
    decision = scene.get('event_validation', {})
    if (decision.get('summary') != audit['summary']['all']
            or decision.get('approved') != audit['approved_for_export']):
        raise ValueError('Live event decision differs from candidate audit')


def decode_packet(data, request):
    if len(data) < 4:
        raise ValueError('Truncated packet')
    size = struct.unpack('<I', data[:4])[0]
    if size < 2 or 4 + size >= len(data):
        raise ValueError('Invalid packet header length')
    metadata = json.loads(data[4:4 + size])
    for field, expected in (('id', request['id']), ('frame', request['frame']),
                            ('left_frame', request['frame']), ('right_frame', request['frame']),
                            ('width', request['width'] * 2), ('height', request['height'])):
        if metadata.get(field) != expected:
            raise ValueError(f'Unexpected rendered packet {field}')
    jpeg = data[4 + size:]
    image = Image.open(io.BytesIO(jpeg)).convert('RGB')
    if image.size != (request['width'] * 2, request['height']):
        raise ValueError('Unexpected rendered JPEG dimensions')
    return metadata, jpeg, image


def contact_sheet(output, samples):
    chosen = [row for row in samples if row['request']['frame'] == 100]
    ordered = sorted(chosen, key=lambda row: (row['request']['pitch'], row['request']['yaw']))
    columns, tile, title = 8, 256, 24
    sheet = Image.new('RGB', (columns * tile, 3 * (tile + title)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(ordered):
        x, y = index % columns * tile, index // columns * (tile + title)
        with Image.open(output / row['right_image']) as image:
            sheet.paste(image.resize((tile, tile)), (x, y + title))
        request = row['request']
        draw.text((x + 4, y + 5), f"frame 100 / yaw {request['yaw']} / pitch {request['pitch']}", fill='white')
    sheet.save(output / 'contact-frame-0100.png')


async def capture(args):
    audit_path = args.candidate / 'event-audit.json'
    audit = json.loads(audit_path.read_text())
    baseline_path = args.baseline / 'cache.json'
    baseline = json.loads(baseline_path.read_text())
    source_hash = sha256_file(baseline['source'])
    correction_hash = sha256_file(args.candidate / 'correction.json')
    report = {
        'format': 'lego_live_multiview_snapshots_v1',
        'scope': 'Actual paired Gaussian UI renderings at sampled orbit views and times. '
                 'Only the original camera has RGB/event observation. Other views, especially '
                 'the back, have no observed ground truth; these snapshots alone prove neither '
                 '3D completeness nor event consistency. A mesh-prior coverage audit must be '
                 'reported separately from original-camera event supervision.',
        'candidate': str(args.candidate), 'baseline': str(args.baseline),
        'artifact_sha256': {'source_ply': source_hash, 'correction': correction_hash,
                            'event_audit': sha256_file(audit_path), 'baseline_cache': sha256_file(baseline_path)},
        'camera': baseline['camera'], 'jpeg_quality': 95,
        'image_encoding_note': 'The live protocol returns JPEG quality 95. Right-only PNGs are '
                               'lossless crops of that decoded JPEG, not raw float renders.',
        'contact_sheet_note': 'Resized navigation overview only; full-resolution images are retained.',
        'samples': [],
    }
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def status():
            async with session.get(args.url.rstrip('/') + '/api/status') as response:
                response.raise_for_status()
                result = await response.json()
            if not result.get('ready'):
                raise ValueError('Live viewer is not ready')
            verify_scene(result['scene'], audit, source_hash, correction_hash)
            return result

        first = await status()
        report['scene'] = first['scene']
        require_same_sampling(baseline, audit)
        if first['scene']['frame_count'] <= 300:
            raise ValueError('Snapshot plan requires at least 301 frames')
        args.output.mkdir(parents=True, exist_ok=False)
        async with session.ws_connect(args.url.rstrip('/') + '/ws', origin=args.url.rstrip('/')) as ws:
            for index, (frame, yaw, pitch) in enumerate(snapshot_plan()):
                request = {'id': index + 1, 'frame': frame, 'yaw': yaw, 'pitch': pitch,
                           'zoom': 1, 'pan_x': 0, 'pan_y': 0, 'width': 640, 'height': 640,
                           'fovy': first['scene']['fovy'], 'background': 'black', 'quality': 95}
                await ws.send_json(request)
                packet = await ws.receive(timeout=120)
                if packet.type != aiohttp.WSMsgType.BINARY:
                    raise ValueError(f'Expected Gaussian render packet, got {packet.type}: {packet.data}')
                meta, jpeg, image = decode_packet(packet.data, request)
                name = f'frame-{frame:04d}_yaw-{yaw:03d}_pitch-{pitch:+03d}'
                paired_name, right_name = name + '-paired.jpg', name + '-right.png'
                (args.output / paired_name).write_bytes(jpeg)
                image.crop((640, 0, 1280, 640)).save(args.output / right_name)
                row = {'request': request, 'response': meta, 'paired_image': paired_name,
                       'right_image': right_name, 'paired_jpeg_sha256': hashlib.sha256(jpeg).hexdigest()}
                report['samples'].append(row)
                print(json.dumps({'sample': index + 1, 'frame': frame, 'yaw': yaw, 'pitch': pitch,
                                  'render_ms': meta['render_ms']}), flush=True)
        await status()
    report['identity_rechecked_after_capture'] = True
    report['sample_count'] = len(report['samples'])
    contact_sheet(args.output, report['samples'])
    (args.output / 'multiview-snapshots.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8892')
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path,
                        help='Right candidate baseline, not the unchanged left baseline')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    asyncio.run(capture(args))


if __name__ == '__main__':
    main()
