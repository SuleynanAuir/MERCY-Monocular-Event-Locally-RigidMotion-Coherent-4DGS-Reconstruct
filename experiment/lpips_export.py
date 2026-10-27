"""Compute native-resolution LPIPS without rerunning the slower part metrics."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def run(root, output, weights, progress=None):
    import torch
    import lpips
    from torchvision.models import AlexNet_Weights
    torch.set_num_threads(4)
    torch.manual_seed(20261027)
    torch.hub.set_dir(str(weights))
    # Both the official ImageNet trunk and packaged LPIPS calibration are
    # tensor-only loads. No experiment checkpoint is deserialized here.
    original_load = torch.load
    def tensor_load(*args, **kwargs):
        return original_load(*args, **dict(kwargs, weights_only=True))
    with patch('torch.load', tensor_load):
        model = lpips.LPIPS(net='alex', version='0.1', verbose=False).eval()
    fingerprint = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        fingerprint.update(key.encode())
        fingerprint.update(tensor.detach().cpu().numpy().tobytes())
    manifest = json.loads((root/'manifest.json').read_text())
    result = dict(manifest_sha256=digest(root/'manifest.json'), network='alex', version='0.1',
        package_version=importlib.metadata.version('lpips'), torch_version=torch.__version__,
        weights_sha256=fingerprint.hexdigest(), trunk_url=AlexNet_Weights.IMAGENET1K_V1.url,
        protocol='RGB [-1,1], full native resolution, no crop or resize, CPU float32, eval mode',
        versions={v['id']: [] for v in manifest['versions']})
    def tensor(path):
        pixels = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)/127.5-1
        return torch.from_numpy(pixels.transpose(2, 0, 1).copy())
    with torch.inference_mode():
        for index, reference in enumerate(manifest['reference']['samples']):
            truth = tensor(root/reference['rgb'])
            predicted = torch.stack([tensor(root/f'versions/{v}/frames/{index:06d}.png')
                                     for v in result['versions']])
            scores = model(predicted, truth[None].expand_as(predicted)).flatten().tolist()
            if index == 0:
                assert abs(model(truth[None], truth[None]).item()) < 1e-8
            for version, score in zip(result['versions'], scores):
                assert np.isfinite(score) and score >= 0
                result['versions'][version].append(score)
            if progress: progress(index+1,len(manifest["samples"]))
            if (index+1) % 25 == 0:
                print(f'LPIPS {index+1}/{len(manifest["samples"])} x {len(scores)} versions', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('LPIPS means', {v: float(np.mean(s)) for v, s in result['versions'].items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=Path('/tmp/scgs-report-models'))
    args = parser.parse_args()
    run(args.export.resolve(), args.output, args.weights)
