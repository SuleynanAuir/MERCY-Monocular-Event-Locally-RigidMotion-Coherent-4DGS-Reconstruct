"""Compare white materials on identical cached GS fields, without loading models."""
import argparse
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
import torch
from PIL import Image
from utils.gs_clay import gray_gaussians


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields-root', type=Path, required=True)
    parser.add_argument('--baseline-module', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('baseline_white_finish', args.baseline_module)
    baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = []
    for scene in ('fan', 'lego'):
        dest = args.output/scene; dest.mkdir(parents=True, exist_ok=True)
        for path in sorted((args.fields_root/scene).glob('fields-*.npz')):
            with np.load(path, allow_pickle=False) as data:
                fields = {key: torch.as_tensor(data[key], device=device)
                          for key in ('render', 'depth', 'alpha', 'coordinates')}
                fovy = float(data['fovy'])
            row = dict(scene=scene, fields=path.name, styles={})
            for style in ('solid', 'wire'):
                params = dict(coordinates=fields['coordinates'] if style == 'wire' else None)
                inputs = [fields[key] for key in ('render', 'depth', 'alpha')]
                before = baseline.gray_gaussians(*inputs, fovy, [0, 0, 0], **params)
                after = gray_gaussians(*inputs, fovy, [0, 0, 0], **params)
                changed = ((before.sum(0) > 1e-5) != (after.sum(0) > 1e-5)).sum().item()
                row['styles'][style] = dict(changed_foreground_mask_pixels=int(changed))
                for name, rgb in (('before', before), ('after', after)):
                    Image.fromarray((rgb.permute(1, 2, 0)*255).byte().cpu().numpy()).save(
                        dest/f'{path.stem}-{name}-{style}.png')
            rows.append(row)
            print(scene, path.stem, flush=True)
    (args.output/'comparison-checks.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
