"""Minimal fixed-appearance G0 renderer for explicit camera/event supervision.

Only ``g0_path`` is read. No scene loader, learned deformation, image/video,
mesh, appearance correction, or render cache is consulted. Camera intrinsics
use pixel-boundary coordinates: the centre of array pixel (0, 0) is (0.5, 0.5).
World-to-camera matrices use OpenCV axes (+x right, +y down, +z forward).

The repository CUDA rasterizer has a hard-coded camera-depth cutoff of 0.2
and a 0.3-pixel covariance low-pass term. This module does not change either.
Rendering is CUDA-only; PLY loading and camera construction also support CPU.
"""

from collections.abc import Mapping
from math import atan, isfinite, isqrt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from plyfile import PlyData


def _indexed_fields(names, prefix):
    """Read standard numeric suffix order, never lexicographic/header order."""
    fields = [name for name in names if name.startswith(prefix)]
    expected = [f"{prefix}{i}" for i in range(len(fields))]
    if set(fields) != set(expected):
        raise ValueError(f"{prefix} fields must be contiguous canonical indices from 0")
    return expected


def _normalize_quaternion(rotation, name="rotation"):
    if rotation.ndim != 2 or rotation.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4), ordered w, x, y, z")
    if not bool(torch.isfinite(rotation).all()):
        raise ValueError(f"{name} contains non-finite values")
    # Scaling first avoids overflow/underflow for otherwise valid quaternions.
    largest = rotation.abs().amax(dim=1, keepdim=True)
    if bool((largest == 0).any()):
        raise ValueError(f"{name} contains a zero quaternion")
    scaled = rotation / largest
    return scaled / torch.linalg.vector_norm(scaled, dim=1, keepdim=True)


def load_g0(g0_path, device="cpu"):
    """Load all rows of one standard 3DGS PLY without changing any appearance.

    Returned ``ply`` and ``raw_vertex`` retain the complete original properties,
    property order, other PLY elements, and metadata. Activated tensor views are
    float32 and require no gradients. ``rotation`` is normalized wxyz;
    ``scaling`` is exp(log-scale), ``opacity`` is sigmoid(logit), and ``sh`` has
    shape (N, (degree + 1)**2, 3). No coefficients or Gaussians are dropped.
    """
    path = Path(g0_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    # Binary PLY can be mapped in bulk; mmap=False in older plyfile releases
    # decodes millions of rows through Python. Copy each element immediately
    # so the returned source data owns no open file mapping.
    ply = PlyData.read(str(path), mmap='c')
    for element in ply.elements:
        element.data = np.array(element.data, copy=True)
    if "vertex" not in ply:
        raise ValueError("G0 PLY is missing its vertex element")
    raw = ply["vertex"].data
    names = tuple(raw.dtype.names or ())
    if len(raw) == 0:
        raise ValueError("G0 PLY has no Gaussian rows")
    required = {"x", "y", "z", "opacity"}
    missing = required.difference(names)
    if missing:
        raise ValueError(f"G0 PLY is missing required properties: {sorted(missing)}")
    dc_names = _indexed_fields(names, "f_dc_")
    rest_names = _indexed_fields(names, "f_rest_")
    scale_names = _indexed_fields(names, "scale_")
    rotation_names = _indexed_fields(names, "rot_")
    if len(dc_names) != 3 or len(scale_names) != 3 or len(rotation_names) != 4:
        raise ValueError("G0 requires 3 f_dc, 3 scale, and 4 rot properties")
    if len(rest_names) % 3:
        raise ValueError("f_rest property count must be divisible by three")
    coefficient_count = 1 + len(rest_names) // 3
    degree = isqrt(coefficient_count) - 1
    if (degree + 1) ** 2 != coefficient_count or not 0 <= degree <= 3:
        raise ValueError("G0 SH coefficients must encode a complete degree 0, 1, 2, or 3")
    target = torch.device(device)

    def columns(fields, *, allow_infinite_logits=False):
        if any(raw.dtype.fields[name][0].kind not in "fiu" for name in fields):
            raise ValueError(f"G0 properties must be numeric scalars: {fields}")
        values = np.stack([raw[name] for name in fields], axis=1).astype(np.float32)
        if np.isnan(values).any() or (not allow_infinite_logits and not np.isfinite(values).all()):
            raise ValueError(f"G0 contains non-finite or unrepresentable values in {fields}")
        return torch.as_tensor(values, dtype=torch.float32, device=target).contiguous()

    xyz = columns(["x", "y", "z"])
    rotation = _normalize_quaternion(columns(rotation_names))
    scaling = columns(scale_names).exp()
    if not bool(torch.isfinite(scaling).all()) or bool((scaling <= 0).any()):
        raise ValueError("G0 activated scales must be finite and strictly positive")
    # Saturated logit(+/-inf) represents exactly opaque/transparent G0 rows.
    # Preserve the raw PLY and exact activated 1/0 values, never clip/re-fit.
    # NaN is still invalid; no other attribute accepts infinite values.
    opacity = columns(["opacity"], allow_infinite_logits=True).sigmoid()
    dc = columns(dc_names).unsqueeze(1)
    # Standard 3DGS PLY flattens all red coefficients, then green, then blue.
    rest = (columns(rest_names).reshape(len(raw), 3, -1).transpose(1, 2)
            if rest_names else dc.new_empty((len(raw), 0, 3)))
    sh = torch.cat((dc, rest), dim=1).contiguous()
    return SimpleNamespace(
        path=str(path), ply=ply, raw_vertex=raw, property_names=names,
        count=len(raw), xyz=xyz, rotation=rotation.contiguous(),
        scaling=scaling, opacity=opacity, sh=sh, sh_degree=degree,
    )


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def make_camera(camera, width=None, device="cpu"):
    """Build a small camera from only width/height/fx/fy/cx/cy/w2c.

    ``w2c`` is a column-vector 4x4 rigid OpenCV world-to-camera transform.
    Optional ``width`` rescales the image while preserving its aspect ratio
    (rounded height); x/y intrinsics use the corresponding actual scale. Since
    cx/cy are boundary coordinates, resizing adds no half-pixel offset.

    ``projection_matrix`` is column-vector P; transforms ending in ``transform``
    are transposed row-vector tensors expected by diff_gaussian_rasterization.
    Its ndc2Pix maps projected boundary coordinates to array indices minus 0.5.
    Extra dictionary keys are deliberately ignored, never interpreted as paths.
    """
    if not isinstance(camera, Mapping):
        raise TypeError("camera must be an explicit calibrated-camera mapping")
    needed = {"width", "height", "fx", "fy", "cx", "cy", "w2c"}
    missing = needed.difference(camera)
    if missing:
        raise ValueError(f"camera is missing required fields: {sorted(missing)}")
    original_width = _positive_integer(camera["width"], "camera width")
    original_height = _positive_integer(camera["height"], "camera height")
    image_width = original_width if width is None else _positive_integer(width, "width")
    image_height = max(1, int(round(original_height * image_width / original_width)))
    sx, sy = image_width / original_width, image_height / original_height
    intrinsics = [float(camera[name]) for name in ("fx", "fy", "cx", "cy")]
    if not all(isfinite(value) for value in intrinsics) or min(intrinsics[:2]) <= 0:
        raise ValueError("camera intrinsics must be finite, with strictly positive fx/fy")
    fx, fy, cx, cy = (intrinsics[0] * sx, intrinsics[1] * sy,
                      intrinsics[2] * sx, intrinsics[3] * sy)
    extrinsic = np.asarray(camera["w2c"], dtype=np.float64)
    if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
        raise ValueError("camera w2c must be a finite 4x4 matrix")
    if not np.allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("camera w2c must be homogeneous with last row [0,0,0,1]")
    rotation = extrinsic[:3, :3]
    if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0)):
        raise ValueError("camera w2c rotation must be a proper orthonormal rotation")
    target = torch.device(device)
    w2c = torch.tensor(extrinsic, dtype=torch.float32, device=target)
    # These depth mapping constants do not override CUDA's fixed 0.2 cutoff.
    znear, zfar = 0.2, 100.0
    projection = torch.zeros((4, 4), dtype=torch.float32, device=target)
    projection[0, 0] = 2 * fx / image_width
    projection[1, 1] = 2 * fy / image_height
    projection[0, 2] = 2 * cx / image_width - 1
    projection[1, 2] = 2 * cy / image_height - 1
    projection[2, 2] = zfar / (zfar - znear)
    projection[2, 3] = -zfar * znear / (zfar - znear)
    projection[3, 2] = 1
    world_view = w2c.transpose(0, 1).contiguous()
    projection_transform = projection.transpose(0, 1).contiguous()
    return SimpleNamespace(
        image_width=image_width, image_height=image_height,
        width=image_width, height=image_height, fx=fx, fy=fy, cx=cx, cy=cy,
        w2c=w2c, projection_matrix=projection,
        world_view_transform=world_view, projection_transform=projection_transform,
        full_proj_transform=(world_view @ projection_transform).contiguous(),
        camera_center=(-w2c[:3, :3].transpose(0, 1) @ w2c[:3, 3]).contiguous(),
        tanfovx=image_width / (2 * fx), tanfovy=image_height / (2 * fy),
        FoVx=2 * atan(image_width / (2 * fx)), FoVy=2 * atan(image_height / (2 * fy)),
        effective_near=0.2, znear=znear, zfar=zfar,
    )


class EventOnlyRenderer:
    """Render fixed G0 appearance at explicit differentiable poses.

    Posed rotations are absolute wxyz quaternions, not additive residuals.
    ``render()`` with no pose arguments renders every original Gaussian at t0.
    Gray is 0.299 R + 0.587 G + 0.114 B on a fixed black background, without
    post-render clipping, exposure fitting, gamma fitting, or image sharpening.
    Depth is the rasterizer's alpha-weighted depth sum, not normalized depth.
    """

    def __init__(self, g0_path, camera, device="cuda"):
        self.device = torch.device(device)
        self.g0 = load_g0(g0_path, device=self.device)
        # Copy only the permitted numeric camera fields; never retain old scene
        # config objects that could later introduce an implicit dependency.
        self.camera_spec = {name: camera[name] for name in
                            ("width", "height", "fx", "fy", "cx", "cy")}
        self.camera_spec["w2c"] = np.array(camera["w2c"], dtype=np.float64, copy=True)
        self.camera = make_camera(self.camera_spec, device=self.device)

    def render(self, posed_xyz=None, posed_rotation=None, width=None):
        if self.device.type != "cuda":
            raise RuntimeError("Rendering requires this repository's CUDA rasterizer; CPU supports loading/camera tests only")
        # Lazy import keeps CPU schema/projection checks independent of CUDA.
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

        xyz = self.g0.xyz if posed_xyz is None else posed_xyz
        rotation = self.g0.rotation if posed_rotation is None else posed_rotation
        for name, value, columns in (("posed_xyz", xyz, 3), ("posed_rotation", rotation, 4)):
            if not isinstance(value, torch.Tensor) or value.shape != (self.g0.count, columns):
                raise ValueError(f"{name} must be a tensor with shape ({self.g0.count}, {columns})")
            if value.dtype != torch.float32 or value.device != self.g0.xyz.device:
                raise ValueError(f"{name} must be float32 on {self.g0.xyz.device}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        # No detach: gradients must reach both the explicit positions and the
        # actual orientation, including this quaternion normalization.
        rotation = _normalize_quaternion(rotation, "posed_rotation").contiguous()
        camera = self.camera if width is None else make_camera(self.camera_spec, width, self.device)
        settings = GaussianRasterizationSettings(
            image_height=camera.image_height, image_width=camera.image_width,
            tanfovx=camera.tanfovx, tanfovy=camera.tanfovy,
            bg=torch.zeros(3, device=self.g0.xyz.device, dtype=torch.float32),
            scale_modifier=1.0, viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform, sh_degree=self.g0.sh_degree,
            campos=camera.camera_center, prefiltered=False, debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)
        rgb, radii, depth, alpha = rasterizer(
            means3D=xyz.contiguous(), means2D=torch.zeros_like(xyz),
            shs=self.g0.sh.detach(), colors_precomp=None,
            opacities=self.g0.opacity.detach(), scales=self.g0.scaling.detach(),
            rotations=rotation, cov3D_precomp=None,
        )
        gray = (rgb * rgb.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(dim=0)
        return {"rgb": rgb, "gray": gray, "alpha": alpha, "depth": depth, "radii": radii}
