"""CPU-only image/alpha diagnostics for explicitly selected mesh-prior parts."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'masks', 'mesh', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--target', action='append', required=True, help='view_index:part_id')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.masks, allow_pickle=False) as z:
        masks = z['part_ids']; depths = z['depth']; metadata = json.loads(str(z['metadata_json']))
    with np.load(args.mesh, allow_pickle=False) as z:
        vertices = z['vertices']; triangles = z['triangles']; parts = z['part_ids']
    before, after = {}, {}
    for path, value in ((args.before, before), (args.after, after)):
        with np.load(path, allow_pickle=False) as z:
            value.update(alpha=z['alpha'], gray=z['gray'])
            if json.loads(str(z['metadata_json']))['views'] != metadata['views']:
                raise ValueError('Render/mask view mismatch')
    args.output.mkdir(parents=True)
    report = {'scope': 'Selected opaque mesh-prior visible part pixels, not observed hidden-view truth. '
                       'No images/model are modified; overlays are diagnostic only.', 'parts': []}
    for text in args.target:
        index, part = map(int, text.split(':'))
        mask = masks[index] == part
        if not mask.any():
            raise ValueError(f'Target {text} has no prior-visible pixels')
        interior = binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)
        yy, xx = np.nonzero(mask)
        shape = mask.shape
        selected_vertices = np.unique(triangles[parts == part])
        xyz = vertices[selected_vertices]
        row = {'view_index': index, 'view': metadata['views'][index], 'part_id': part,
               'name': metadata['objects'][part], 'visible_prior_pixels': int(mask.sum()),
               'interior_prior_pixels': int(interior.sum()),
               'bbox_xyxy': [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1],
               'touches_image_boundary': bool(np.any(mask[[0, -1]]) or np.any(mask[:, [0, -1]])),
               'canonical_part_bounds': [xyz.min(0).tolist(), xyz.max(0).tolist()],
               'visible_depth_bounds': [float(depths[index][mask].min()), float(depths[index][mask].max())],
               'models': {}}
        for name, values in (('before', before), ('after', after)):
            alpha, gray = values['alpha'][index], values['gray'][index]
            row['models'][name] = {
                'alpha_quantiles_visible': np.quantile(alpha[mask], [0, .1, .5, .9, 1]).tolist(),
                'gray_quantiles_visible': np.quantile(gray[mask], [0, .1, .5, .9, 1]).tolist(),
                'interior_missing_by_alpha': {str(t): int((interior & (alpha < t)).sum()) for t in (.2, .5, .8)},
                'interior_gray_below_003': int((interior & (gray < .03)).sum()),
                'interior_pixels': int(interior.sum())}
        panels = []
        for name, values in (('before gray', before), ('after gray', after)):
            rgb = np.repeat(np.clip(values['gray'][index, :, :, None]*255, 0, 255).astype(np.uint8), 3, axis=2)
            panels.append((name, rgb))
        base = panels[1][1].copy()
        base[mask] = (.45*base[mask] + .55*np.array([64, 220, 64])).astype(np.uint8)
        missing = interior & (after['alpha'][index] < .5)
        base[missing] = [255, 32, 32]
        panels.append(('prior part green / alpha<.5 red', base))
        alpha_image = np.repeat(np.clip(after['alpha'][index, :, :, None]*255, 0, 255).astype(np.uint8), 3, axis=2)
        panels.append(('after alpha', alpha_image))
        canvas = Image.new('RGB', (shape[1]*len(panels), shape[0]+42), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for col, (name, rgb) in enumerate(panels):
            canvas.paste(Image.fromarray(rgb), (col*shape[1], 42))
            draw.text((col*shape[1]+5, 5), f'view {index} / part {part} / {name}', fill='white')
            draw.text((col*shape[1]+5, 22), metadata['objects'][part], fill='white')
        filename = f'view-{index:02d}-part-{part:03d}.png'
        canvas.save(args.output/filename)
        row['diagnostic_image'] = filename
        report['parts'].append(row)
    (args.output/'static-part-inspection.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
