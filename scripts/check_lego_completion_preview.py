"""Compare localized parts in an eight-frame, quantized completion preview."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervision', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--preview', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.supervision, allow_pickle=False) as data:
        reference = data['images']
    with np.load(args.masks, allow_pickle=False) as data:
        masks = data['part_ids']; indices = data['frame_indices']; meta = json.loads(str(data['metadata_json']))
    baseline = np.load(args.baseline/'rendered_gray.npy', mmap_mode='r', allow_pickle=False)
    pngs = {int(path.stem.rsplit('_', 1)[1]): path for path in args.preview.glob('lego-completed-parts_*.png')}
    report = {'scope': 'Sparse PNG readback diagnostic only. Both old and new are quantized to 8 bits; '
                      'not the full-frame float event/completeness audit.', 'parts': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for part in (235, 229, 237, 256, 227):
        rows, panels = [], []
        for frame, png in pngs.items():
            if frame not in indices:
                continue
            sample = int(np.flatnonzero(indices == frame)[0]); mask = masks[sample] == part
            interior = binary_erosion(mask, structure=np.ones((3, 3), bool))
            expected = interior & (reference[frame] / 255 > .12)
            old = (np.clip(baseline[frame], 0, 1) * 255).astype(np.uint8)
            new = np.asarray(Image.open(png).convert('L'))
            if not expected.sum():
                continue
            rows.append({'frame': frame, 'reference_pixels': int(expected.sum()),
                         'before_missing': int((expected & (old/255 < .08)).sum()),
                         'after_missing': int((expected & (new/255 < .08)).sum())})
        worst = max(rows, key=lambda row: row['before_missing'] / row['reference_pixels'])
        frame = worst['frame']; sample = int(np.flatnonzero(indices == frame)[0]); mask = masks[sample] == part
        yy, xx = np.nonzero(mask); cx, cy = int(xx.mean()), int(yy.mean())
        half = max(48, int(max(np.ptp(xx), np.ptp(yy)) / 2) + 12)
        x0, y0, x1, y1 = max(0,cx-half), max(0,cy-half), min(640,cx+half), min(640,cy+half)
        edge = binary_dilation(mask) & ~mask
        for label, image in [('RGB reference', reference[frame]),
                             ('before append', (np.clip(baseline[frame],0,1)*255).astype(np.uint8)),
                             ('after append 609 Gaussians', np.asarray(Image.open(pngs[frame]).convert('L')))]:
            color = np.repeat(image[...,None], 3, -1)
            color[edge] = [255,220,40]
            crop = Image.fromarray(color[y0:y1,x0:x1]).resize(((x1-x0)*3,(y1-y0)*3),Image.Resampling.NEAREST)
            panel = Image.new('RGB',(crop.width,crop.height+40));panel.paste(crop,(0,40))
            draw=ImageDraw.Draw(panel);draw.text((5,5),label,fill='white');draw.text((5,22),f'part {part}, f{frame}',fill='white')
            panels.append(panel)
        canvas=Image.new('RGB',(sum(panel.width for panel in panels),panels[0].height));offset=0
        for panel in panels:canvas.paste(panel,(offset,0));offset+=panel.width
        path=args.output.parent/f'{args.output.stem}-part-{part}.png';canvas.save(path)
        denominator=sum(row['reference_pixels'] for row in rows)
        report['parts'][str(part)]={'name':meta['objects'][part],
                                   'before_missing_fraction':sum(row['before_missing'] for row in rows)/denominator,
                                   'after_missing_fraction':sum(row['after_missing'] for row in rows)/denominator,
                                   'frames':rows,'image':str(path)}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({part:{k:v for k,v in row.items() if k not in ('frames','image')} for part,row in report['parts'].items()},indent=2))


if __name__=='__main__':
    main()
