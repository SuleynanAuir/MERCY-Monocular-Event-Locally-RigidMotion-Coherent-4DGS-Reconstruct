"""Opt-in camera-plane motion prior; no data, geometry or motion is loaded.

This constrains the existing G0-centered deformation parameters, not their
physical correctness. Rotation about the camera's world-space optical axis
and perpendicular center translations preserve each Gaussian center's
camera Z, including interpolated knots and normalized soft graph blends.
It does not make single-view depth or hidden motion observable. Native raw
event auditing remains required and is not relaxed by this prior.
"""

from collections.abc import Mapping

import numpy as np
import torch


def _validated_camera(camera_w2c):
    """Return an independent CPU float64 proper world-to-camera transform."""
    if isinstance(camera_w2c, torch.Tensor):
        if camera_w2c.is_complex() or camera_w2c.dtype == torch.bool or camera_w2c.layout != torch.strided:
            raise ValueError('camera_w2c must be a finite real proper rigid 4-by-4 transform')
        matrix = camera_w2c.detach().to(device='cpu', dtype=torch.float64).clone()
    else:
        try:
            supplied = np.asarray(camera_w2c)
            if supplied.dtype.kind not in 'iuf':
                raise ValueError('camera_w2c must contain real numeric values')
            # Lists also work on installations whose NumPy/torch binary
            # bridge is unavailable. No input file or cached camera is read.
            matrix = torch.tensor(supplied.tolist(), dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError, OverflowError) as error:
            raise ValueError('camera_w2c must be a finite real proper rigid 4-by-4 transform') from error
    if (matrix.shape != (4, 4) or not torch.isfinite(matrix).all()
            or not torch.allclose(matrix[3], matrix.new_tensor([0., 0., 0., 1.]), atol=1e-8, rtol=0)):
        raise ValueError('camera_w2c must be a finite real proper rigid 4-by-4 transform')
    rotation = matrix[:3, :3]
    if (not torch.allclose(rotation @ rotation.T, torch.eye(3, dtype=torch.float64), atol=1e-5, rtol=0)
            or not torch.isclose(torch.linalg.det(rotation), matrix.new_tensor(1.), atol=1e-5, rtol=0)):
        raise ValueError('camera_w2c rotation must be orthonormal with determinant +1')
    return matrix


def _camera_axis(matrix):
    # R.T @ [0,0,1] is the third row of world-to-camera R. Normalizing
    # removes the small norm error permitted by camera calibration tolerance.
    axis = matrix[2, :3].clone()
    return axis / torch.linalg.vector_norm(axis)


def _validate_motion(motion):
    parameters = [getattr(motion, name, None) for name in ('translations', 'rotation_vectors')]
    if any(not isinstance(value, torch.Tensor) or not value.is_floating_point()
           or value.layout != torch.strided for value in parameters):
        raise ValueError('Motion translations and rotation_vectors must be dense real floating tensors')
    translations, rotations = parameters
    if (translations.ndim != 3 or translations.shape[-1] != 3
            or translations.shape[0] < 1 or translations.shape[1] < 1
            or rotations.shape != translations.shape or rotations.device != translations.device
            or rotations.dtype != translations.dtype
            or not all(torch.isfinite(value).all() for value in parameters)):
        raise ValueError('Motion parameters must share a finite nonempty (T-1)-by-K-by-3 shape, dtype and device')
    centers, times = getattr(motion, 'centers', None), getattr(motion, 'times', None)
    if (not isinstance(centers, torch.Tensor) or not centers.is_floating_point()
            or centers.layout != torch.strided or centers.shape != (translations.shape[1], 3)
            or centers.device != translations.device or not torch.isfinite(centers).all()):
        raise ValueError('Motion centers must be finite floating K-by-3 tensors on the parameter device')
    if (not isinstance(times, torch.Tensor) or not times.is_floating_point() or times.layout != torch.strided
            or times.device != translations.device or times.shape != (translations.shape[0] + 1,)
            or not torch.isfinite(times).all() or times[0] != 0 or torch.any(times[1:] <= times[:-1])):
        raise ValueError('Motion times must increase from zero with t0 structurally absent from the parameters')
    for value in parameters:
        gradient = value.grad
        if gradient is not None and (gradient.layout != torch.strided or gradient.shape != value.shape
                                     or gradient.dtype != value.dtype or gradient.device != value.device
                                     or not torch.isfinite(gradient).all()):
            raise ValueError('Existing motion gradients must be finite dense tensors matching their parameter')
    return parameters


def enforce_camera_plane_motion(motion, camera_w2c, *, gradients_only=False):
    """Project parameters and existing gradients onto an explicit plane prior.

    ``a = R_camera.T @ [0,0,1]`` is a unit vector in world coordinates.
    Center translations become ``t - a*(t dot a)`` and rotation vectors
    become ``a*(r dot a)``. Buffers, centers, times, t0 and parameter identity
    are unchanged. Missing gradients stay None. With gradients_only=True,
    parameters are validated but never modified, even if outside the plane.

    Call before using newly initialized parameters and after every optimizer
    step. Projecting gradients alone is insufficient for coordinatewise Adam
    preconditioning. Optimizer state is not read or modified here. This
    function supplies no event evidence and never approves a reconstruction.
    Validation and finite-result checks precede all in-place changes.
    """
    if type(gradients_only) is not bool:
        raise ValueError('gradients_only must be boolean')
    with torch.no_grad():
        matrix = _validated_camera(camera_w2c)
        parameters = _validate_motion(motion)
        updates = []
        for index, parameter in enumerate(parameters):
            for value in ((parameter.grad,) if gradients_only else (parameter, parameter.grad)):
                if value is None:
                    continue
                # Accumulating float32/half values in double avoids gratuitous
                # overflow/cancellation. The final dtype must still be finite.
                working_dtype = torch.float64 if value.device.type != 'mps' else torch.float32
                work = value.detach().to(dtype=working_dtype)
                axis = _camera_axis(matrix).to(device=value.device, dtype=working_dtype)
                parallel = (work * axis).sum(dim=-1, keepdim=True) * axis
                projected = (work - parallel if index == 0 else parallel).to(dtype=value.dtype)
                if not torch.isfinite(projected).all():
                    raise ValueError('Camera-plane projection exceeds the finite motion dtype range')
                updates.append((value, projected))
        for destination, projected in updates:
            destination.copy_(projected)


def camera_plane_prior_metadata(camera):
    """JSON-safe disclosure for an explicitly enabled camera-plane prior."""
    if not isinstance(camera, Mapping) or 'w2c' not in camera:
        raise ValueError('Camera metadata must explicitly include w2c')
    matrix = _validated_camera(camera['w2c'])
    return {
        'kind': 'explicit_camera_plane_motion_prior', 'version': 1, 'enabled': True,
        'source': 'supplied_camera_extrinsics_only',
        'world_optical_axis': _camera_axis(matrix).tolist(),
        'rotation_projection': 'rotation_vector_parallel_to_R_camera_transpose_ez',
        'translation_projection': 'node_center_translation_perpendicular_to_R_camera_transpose_ez',
        'camera_depth_policy': 'preserve_each_G0_Gaussian_center_camera_Z_up_to_floating_point_error',
        'applies_to_interpolated_knots': True,
        'normalized_graph_blends_preserve_depth': True,
        't0_parameter_added_or_changed': False,
        'optimizer_state_modified': False,
        'post_optimizer_parameter_projection_required': True,
        'physical_motion_validated': False,
        'hidden_motion_identifiable': False,
        'independent_observations_added': 0,
        'native_event_audit_required': True,
        'native_event_gates_relaxed': False,
        'approved_for_export': False,
        'limitations': [
            'Camera-plane motion is a chosen prior, not motion derived uniquely from events.',
            'Real motion may contain out-of-plane translation or rotation excluded by this prior.',
            'Preserved G0 depth does not validate initial geometry, binding or hidden surfaces.',
        ],
    }
