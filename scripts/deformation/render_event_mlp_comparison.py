"""Render baseline | event MLP video from one self-contained MLP checkpoint."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image, ImageDraw

from utils.event_mlp import load_mlp_checkpoint
from utils.event_only_renderer import EventOnlyRenderer
from scripts.events.render_event_only_sequence import sequence_times, _ffmpeg_executable


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--g0', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fps', type=float, default=30.)
    p.add_argument('--width', type=int, default=640)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 32 <= args.width <= 1280:
        p.error('width must be in [32,1280]')
    executable = _ffmpeg_executable()
    if not executable:
        raise RuntimeError('ffmpeg is required')
    model, camera, payload = load_mlp_checkpoint(args.checkpoint, args.g0)
    renderer = EventOnlyRenderer(args.g0, camera)
    timestamps = sequence_times(float(model.times[-1]), args.fps)
    height = max(1, round(camera['height'] * args.width/camera['width']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([
            executable, '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-video_size', f'{args.width*2}x{height+32}', '-framerate', str(args.fps), '-i', 'pipe:0',
            '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black', '-movflags', '+faststart', str(args.output)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            with torch.no_grad():
                for index, timestamp in enumerate(timestamps):
                    images = []
                    for motion in (model.base, model):
                        x, q = motion.deform(renderer.g0.xyz, renderer.g0.rotation, timestamp)
                        rgb = renderer.render(x, q, width=args.width)['rgb']
                        images.append(rgb.clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
                    canvas = Image.new('RGB', (args.width*2, height+32), (24, 24, 24))
                    canvas.paste(Image.fromarray(np.concatenate(images, axis=1)), (0, 32))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((12, 10), f'Event-only base   t={timestamp:.3f}s', fill='white')
                    draw.text((args.width+12, 10), 'Base + residual MLP (candidate)', fill='white')
                    process.stdin.write(canvas.tobytes())
                    if index % 50 == 0:
                        print(json.dumps({'frame': index, 'total': len(timestamps)}), flush=True)
            process.stdin.close()
            if process.wait(timeout=60):
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors='replace'))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    print(json.dumps({'video': str(args.output), 'frames': len(timestamps),
                      'event_accuracy_approved': False, 'checkpoint_format': payload['format']}), flush=True)


if __name__ == '__main__':
    main()
