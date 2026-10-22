"""Compare the exact frozen V3 with its auxiliary MLP at the original camera."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image, ImageDraw
import torch

from scripts.events.export_lego_event_preview import verify_candidate
from scripts.events.render_event_only_sequence import _ffmpeg_executable
from utils.rigid_checkpoint import sha256_file
from utils.v3_auxiliary_mlp import load_checkpoint
from utils.v3_auxiliary_renderer import FrozenV3Renderer


@torch.no_grad()
def main(field_kind='v3-auxiliary'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    executable = _ffmpeg_executable()
    if not executable:
        raise RuntimeError('ffmpeg is required')
    load_field = load_checkpoint
    if field_kind == 'scgs-v4':
        from utils.scgs_v4_field import load_checkpoint as load_field, make_scgs_graph
    model, provenance = load_field(args.checkpoint)
    cache, record, report = verify_candidate(Path(provenance['baseline']), Path(provenance['candidate']), allow_unapproved=True)
    if report['artifact_sha256'] != provenance['baseline_artifact_sha256']:
        raise ValueError('Checkpoint is bound to a different V3')
    renderer = FrozenV3Renderer(Path(provenance['baseline']), record)
    if hasattr(model, 'validate_renderer'):
        model.validate_renderer(renderer)
    else:
        graph = make_scgs_graph(renderer) if field_kind == 'scgs-v4' else renderer.graph()
        for key, value in graph.items():
            if not torch.equal(torch.as_tensor(value), getattr(model, key).cpu()):
                raise ValueError(f'Auxiliary binding differs from frozen V3: {key}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    keys = {0, 50, 99, 150, 199, 250, 299, 350, 399}
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([executable, '-nostdin', '-v', 'error', '-n',
                                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-video_size', '1280x672',
                                    '-framerate', str(cache['fps']), '-i', 'pipe:0', '-an',
                                    '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)],
                                   stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            for frame in range(cache['frame_count']):
                panels = [renderer.rgb(frame, auxiliary).clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
                          for auxiliary in (None, model)]
                canvas = Image.new('RGB', (1280, 672), (24, 24, 24))
                canvas.paste(Image.fromarray(np.concatenate(panels, 1)), (0, 32))
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 10), f'V3: frozen Blender motion   t={frame/cache["fps"]:.3f}s', fill='white')
                label = 'V4: V3 + SC-GS residual field' if field_kind == 'scgs-v4' else 'V3 + auxiliary MLP (event supervision)'
                if provenance.get('field_kind') == 'joint-trajectory-v4':
                    label = 'V4: constrained joint trajectory residual'
                if provenance.get('field_kind') == 'local-joint-mlp-v4':
                    label = 'V4: local joint MLP / frozen main drive'
                draw.text((652, 10), label, fill='white')
                process.stdin.write(canvas.tobytes())
                if frame in keys:
                    canvas.save(args.output.parent/f'comparison-{frame:04d}.png')
                if frame % 100 == 0:
                    print(json.dumps(dict(frame=frame, total=cache['frame_count'])), flush=True)
            process.stdin.close()
            if process.wait(timeout=60):
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors='replace'))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    result = dict(field_kind=field_kind, video=str(args.output), frames=cache['frame_count'], fps=cache['fps'],
                  checkpoint_sha256=sha256_file(args.checkpoint), video_sha256=sha256_file(args.output))
    (args.output.parent/'comparison-video.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
