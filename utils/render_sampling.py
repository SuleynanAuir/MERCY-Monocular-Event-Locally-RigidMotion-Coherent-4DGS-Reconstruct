"""Bounded, recorded Gaussian raster sampling shared by fitting and playback.

This does not sharpen an image or change the CUDA rasterizer. At factor two,
the same camera projects to twice as many pixels in each axis; clamped RGB is
then reduced by an exact area average before grayscale/event supervision.
"""

RENDER_DOWNSAMPLE_FILTER = 'area_after_clamp'


def validate_supersample(value):
    if type(value) is not int or value not in (1, 2):
        raise ValueError('render_supersample must be the integer 1 or 2')
    return value


def render_sampling_contract(metadata, supersample=None):
    """Resolve legacy factor one or an explicitly documented sampling operator.

    An override is permitted for a new fitting experiment on an old baseline;
    persisted factor-two records always require an explicit filter contract.
    """
    factor = validate_supersample(metadata.get('render_supersample', 1))
    stored_filter = metadata.get('render_downsample_filter')
    if stored_filter is not None and stored_filter != RENDER_DOWNSAMPLE_FILTER:
        raise ValueError('Unsupported render_downsample_filter')
    if factor != 1 and stored_filter is None:
        raise ValueError('Supersampled rendering requires an explicit render_downsample_filter')
    if supersample is not None:
        factor = validate_supersample(supersample)
    return {'render_supersample': factor, 'render_downsample_filter': RENDER_DOWNSAMPLE_FILTER}


def require_same_sampling(*records):
    contracts = [render_sampling_contract(record) for record in records]
    if not contracts or any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError('Render sampling contract mismatch')
    return contracts[0]


def require_native_sampling(metadata, context):
    """Guard legacy routes that cannot preserve the audited SSAA operator."""
    if render_sampling_contract(metadata)['render_supersample'] != 1:
        raise ValueError(f'{context} supports render_supersample=1 only. '
                         'Use the audited comparison UI for supersampled candidates; '
                         'this legacy route does not preserve their sampling contract.')


def resolve_rgb(rgb, supersample):
    """Clamp CHW RGB before area reduction, preserving differentiability."""
    import torch.nn.functional as F
    factor = validate_supersample(supersample)
    if (rgb.ndim != 3 or rgb.shape[0] != 3 or rgb.shape[1] < factor
            or rgb.shape[2] < factor or rgb.shape[1] % factor or rgb.shape[2] % factor):
        raise ValueError('RGB render shape must be CHW with spatial dimensions divisible by sampling factor')
    rgb = rgb.clamp(0, 1)
    return rgb if factor == 1 else F.avg_pool2d(rgb, kernel_size=factor, stride=factor)
