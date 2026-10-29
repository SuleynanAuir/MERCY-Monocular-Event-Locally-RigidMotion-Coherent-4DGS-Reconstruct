"""Inference-only temporal opacity filter for the pinned Hybrid rasterizer.

Never edits the upstream checkout or training extension. No frame blending.
"""
import hashlib
import os
from pathlib import Path
import shutil

REVISION = '9528828d79148f032c924ccc840bd81374ab800c'
PATCH = 'temporal-prefilter-v2'
_native = None
_stable = None


def select(repo, enabled):
    global _native, _stable
    import gaussian_renderer.diff_gaussian_rasterization as raster
    if _native is None:
        _native = raster._C
    if not enabled:
        raster._C = _native
        return
    if _stable is None:
        from torch.utils.cpp_extension import load
        source = Path(repo)/'diff-gaussian-rasterization'
        original = (source/'cuda_rasterizer/forward.cu').read_text()
        old = 'mask = marginal_t > 0.05;'
        if original.count(old) != 1:
            raise ValueError('Unsupported upstream temporal culling kernel')
        # Pixel alpha cannot exceed opacity*marginal_t. The same 1/255
        # threshold is already used in rasterization, so only invisible
        # contributions are skipped, rather than popping at 5% support.
        patched = original.replace(old, 'mask = opacity * marginal_t >= (1.0f / 255.0f);')
        signature = 'float t, const float timestamp, int idx, bool& mask, float& opacity)'
        call = 'p_orig, ts[idx], timestamp, idx, time_mask, opacity);'
        marginal = 'float marginal_t = __expf(-0.5*dt*dt/cov_t);'
        for token in (signature,call,marginal):
            if patched.count(token) != 1:
                raise ValueError('Unsupported conditional covariance implementation')
        patched = patched.replace(signature,signature[:-1]+', const float temporal_sigma)')
        patched = patched.replace(call,call[:-2]+', opa_threshold);')
        patched = patched.replace(marginal,'''// Convolve only temporal opacity support. Keep the original
        // covariance and motion trajectory for spatial conditioning below.
        float filtered_cov_t = cov_t + temporal_sigma * temporal_sigma;
        opacity *= sqrtf(cov_t / filtered_cov_t);
        float marginal_t = __expf(-0.5*dt*dt/filtered_cov_t);''')
        fingerprint = hashlib.sha256(patched.encode()).hexdigest()[:12]
        cache = Path(os.environ.get('TORCH_EXTENSIONS_DIR',Path.home()/'.cache/torch_extensions'))
        target = cache/('fvv-temporal-source-'+fingerprint)
        if not target.exists():
            shutil.copytree(source,target,ignore=shutil.ignore_patterns('.git','build','__pycache__','*.so'))
        forward = target/'cuda_rasterizer/forward.cu'
        if forward.read_text() != patched:
            forward.write_text(patched)
        _stable = load(name='fvv_temporal_'+fingerprint,
            sources=[str(target/name) for name in ['cuda_rasterizer/rasterizer_impl.cu',
                'cuda_rasterizer/forward.cu','cuda_rasterizer/backward.cu','rasterize_points.cu','ext.cpp']],
            extra_cuda_cflags=['-I '+str(target/'third_party/glm'),'-g'],verbose=True)
    raster._C = _stable
