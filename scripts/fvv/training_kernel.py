"""Isolated reference pixel backward for geometry-supervised offline training.

Forward image computation stays byte-for-byte upstream. Serving never imports
this module. The fast per-Gaussian backward is not used by this opt-in path.
"""
import hashlib
import os
from pathlib import Path
import shutil


def patch_backward(source):
    start=source.index('void BACKWARD::render(')
    body=source.index('{',start);level=1;end=body+1
    while level:
        level+=(source[end]=='{')-(source[end]=='}');end+=1
    if source[body:end].count('PerGaussianRenderCUDA<NUM_CHANNELS>')!=1:
        raise ValueError('Unsupported upstream backward implementation')
    replacement='''{
    renderCUDA<NUM_CHANNELS><<<grid, block>>>(
        P, ranges, point_list, W, H, bg_color, means2D,
        conic_opacity, conic_opacity_static, colors, colors_static, depths,
        flows_2d, final_Ts, n_contrib, dL_dpixels, dL_depths, dL_masks,
        dL_dpix_flow, dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors,
        dL_dflows, dL_dmean2D_static, dL_dconic2D_static,
        dL_dopacity_static, dL_dcolors_static);
}'''
    result=source[:body]+replacement+source[end:]
    # The legacy pixel branch left static depths uninitialized. Although this
    # route uses color gradients, avoid any undefined float multiplied by zero.
    marker='collected_conic_opacity[block.thread_rank()] = conic_opacity_static[coll_id-P];'
    if result.count(marker)!=1:raise ValueError('Unsupported static gather schema')
    return result.replace(marker,'collected_depths[block.thread_rank()] = depths[coll_id];\n\t\t\t\t'+marker)


def select(repo):
    import gaussian_renderer.diff_gaussian_rasterization as raster
    from torch.utils.cpp_extension import load
    source=Path(repo)/'diff-gaussian-rasterization'
    original=(source/'cuda_rasterizer/backward.cu').read_text()
    patched=patch_backward(original);fingerprint=hashlib.sha256(patched.encode()).hexdigest()[:12]
    cache=Path(os.environ['TORCH_EXTENSIONS_DIR'])
    target=cache/('fvv-training-pixel-source-'+fingerprint)
    if not target.exists():shutil.copytree(source,target,ignore=shutil.ignore_patterns('.git','build','__pycache__','*.so'))
    path=target/'cuda_rasterizer/backward.cu'
    if path.read_text()!=patched:path.write_text(patched)
    raster._C=load(name='fvv_training_pixel_'+fingerprint,
        sources=[str(target/name) for name in ['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/forward.cu','cuda_rasterizer/backward.cu','rasterize_points.cu','ext.cpp']],
        extra_cuda_cflags=['-I '+str(target/'third_party/glm'),'-g'],verbose=True)
    return fingerprint
