"""Diagnostic contact sheets of the same joint region over time, not UI imagery."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--supervision', required=True, type=Path)
    p.add_argument('--renders', required=True, type=Path, nargs='+')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--frames', type=int, nargs='+', default=[0, 25, 50, 75, 100, 125, 150, 175])
    p.add_argument('--roi', type=int, nargs=4, default=[160, 140, 380, 380])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.supervision, allow_pickle=False) as data:
        reference = data['images']
    sources = [('reference', reference)] + [(path.name, np.load(path/'rendered_gray.npy', mmap_mode='r')) for path in args.renders]
    x0, y0, x1, y1 = args.roi
    width, height = x1-x0, y1-y0
    if width <= 0 or height <= 0 or min(x0, y0) < 0 or max(x1, y1) > 640:
        raise ValueError('Invalid ROI')
    canvas = Image.new('RGB', (width*len(sources), (height+35)*len(args.frames)), '#171c23')
    draw = ImageDraw.Draw(canvas)
    for row, index in enumerate(args.frames):
        for column, (name, sequence) in enumerate(sources):
            frame = sequence[index, y0:y1, x0:x1]
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1)*255).astype(np.uint8)
            canvas.paste(Image.fromarray(frame).convert('RGB'), (column*width, row*(height+35)+35))
            draw.text((column*width+4, row*(height+35)+4), f'{name}\nf={index} t={index/60:.3f}s', fill='white')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
