"""Exercise CUDA math and indexing without importing any Gaussian extension.

Run in a disposable process: a device assertion poisons its CUDA context.
This is a smoke test, not a certification of GPU hardware health.
"""
import argparse
import json
import time


def check(iterations=10000):
    import torch
    report = dict(torch=torch.__version__, cuda=torch.version.cuda,
                  iterations=iterations, completed=0, custom_extensions=False, passed=False)
    start = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        report['device'] = torch.cuda.get_device_name()
        torch.manual_seed(6666)
        for i in range(iterations):
            x = torch.rand(150000, 4, 4, device='cuda')
            y = x / x.square().sum(-1, keepdim=True).sqrt()
            z = y @ y.transpose(-1, -2)
            mask = z[:, 0, 0] > .5
            counts = torch.zeros((150000, 1), device='cuda')
            counts[mask] += 1
            torch.cuda.synchronize()
            if not bool(torch.isfinite(z).all()) or not torch.equal(counts[:, 0], mask.float()):
                raise RuntimeError('CUDA math/indexing produced an incorrect result')
            report['completed'] = i + 1
        report['passed'] = True
    except Exception as exc:
        report['error'] = str(exc)
    report['elapsed_seconds'] = time.monotonic() - start
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=10000)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 10000:
        parser.error('iterations must be 1..10000')
    result = check(args.iterations)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if result['passed'] else 1)
