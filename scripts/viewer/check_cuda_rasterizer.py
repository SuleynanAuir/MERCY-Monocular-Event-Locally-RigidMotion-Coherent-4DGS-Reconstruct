"""Probe the original viewer's installed CUDA extension before loading assets.

Run in the scgs-env interpreter, not the independent Hybrid environment.
Uses one transient Gaussian; writes no models, cache or image files.
"""
import json


def check():
    import torch
    import diff_gaussian_rasterization as raster

    report = dict(passed=False, torch=torch.__version__, cuda=torch.version.cuda,
                  extension=raster._C.__file__)
    try:
        report.update(gpu=torch.cuda.get_device_name(),
                      capability=list(torch.cuda.get_device_capability()))
        with torch.no_grad():
            identity = torch.eye(4, device='cuda')
            settings = raster.GaussianRasterizationSettings(
                image_height=32, image_width=32, tanfovx=1., tanfovy=1.,
                bg=torch.zeros(3, device='cuda'), scale_modifier=1.,
                viewmatrix=identity, projmatrix=identity, sh_degree=0,
                campos=torch.zeros(3, device='cuda'), prefiltered=False,
                # Detect incompatible kernels before an invalid tile count can
                # be mistaken for an enormous allocation request.
                debug=True)
            renderer = raster.GaussianRasterizer(settings)
            for _ in range(8):
                color, radii, depth, alpha = renderer(
                    means3D=torch.tensor([[0., 0., 2.]], device='cuda'),
                    means2D=torch.zeros((1, 3), device='cuda'),
                    opacities=torch.full((1, 1), .5, device='cuda'),
                    colors_precomp=torch.ones((1, 3), device='cuda'),
                    scales=torch.full((1, 3), .1, device='cuda'),
                    rotations=torch.tensor([[1., 0., 0., 0.]], device='cuda'))
                torch.cuda.synchronize()
                if (not all(bool(torch.isfinite(x).all()) for x in (color, depth, alpha))
                        or not bool((radii > 0).all()) or not bool(alpha.max() > 0)):
                    raise RuntimeError('Rasterizer returned invalid or empty output')
            report.update(passed=True, rendered_frames=8)
    except Exception as error:
        report['error'] = str(error)
    return report


if __name__ == '__main__':
    result = check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
