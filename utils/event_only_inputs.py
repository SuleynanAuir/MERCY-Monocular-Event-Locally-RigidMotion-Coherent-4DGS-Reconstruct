"""Fail-closed input preflight for G0 + raw events + explicit K/R/T only.

This module never follows paths embedded in JSON. It validates supported file
and camera structure, not the authenticity of a file's claimed origin, event
semantics, or reconstruction quality. It does not train or render anything.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import stat


CAMERA_CONVENTION = "3DGS R stored transposed; camera x-right/y-down/z-forward"
ALLOWED_MODALITIES = ("g0", "events", "camera")
_ROLES = ("g0", "events", "intrinsics", "extrinsics")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_HEADER_BYTES = 256 * 1024
_MAX_HEADER_LINE = 8192
_FLOAT_TYPES = {"float", "float32", "double", "float64"}
_SCALAR_TYPES = _FLOAT_TYPES | {
    "char", "uchar", "short", "ushort", "int", "uint",
    "int8", "uint8", "int16", "uint16", "int32", "uint32",
}
_REQUIRED_PROPERTIES = (
    "x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2",
)


class InputValidationError(ValueError):
    """The explicit inputs do not satisfy the supported input contract."""


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise InputValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise InputValidationError(f"{name} must be a finite number")
    return result


def _dimension(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputValidationError(f"{name} must be a positive integer")
    return value


def _vector(value, length, name):
    if not isinstance(value, list) or len(value) != length:
        raise InputValidationError(f"{name} must contain {length} numbers")
    return [_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _matrix(value, name):
    if not isinstance(value, list) or len(value) != 3:
        raise InputValidationError(f"{name} must be a 3 by 3 matrix")
    return [_vector(row, 3, f"{name}[{index}]") for index, row in enumerate(value)]


def _close(left, right):
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-8)


def _distinct_regular_paths(paths):
    resolved, identities = {}, set()
    for role, value in zip(_ROLES, paths):
        if not isinstance(value, (str, Path)) or not str(value):
            raise InputValidationError(f"{role} requires an explicit file path")
        try:
            path = Path(value).resolve(strict=True)
            info = path.stat()
        except (OSError, RuntimeError) as exc:
            raise InputValidationError(f"{role} is not an accessible file") from exc
        if not stat.S_ISREG(info.st_mode):
            raise InputValidationError(f"{role} must be a regular file")
        identity = (info.st_dev, info.st_ino)
        if path in resolved.values() or identity in identities:
            raise InputValidationError("The four explicit inputs must be distinct files")
        resolved[role] = path
        identities.add(identity)
    return resolved


def _stream_read(path, capture=False):
    digest, chunks, size = hashlib.sha256(), [], 0
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                size += len(chunk)
                if capture and size > _MAX_JSON_BYTES:
                    raise InputValidationError("Camera JSON exceeds the input size limit")
                digest.update(chunk)
                if capture:
                    chunks.append(chunk)
    except OSError as exc:
        raise InputValidationError("Unable to read an explicit input file") from exc
    return digest.hexdigest(), b"".join(chunks) if capture else None


def sha256_file(path):
    """Hash one explicitly named file in bounded chunks; never follow references."""
    return _stream_read(Path(path))[0]


def inspect_gaussian_ply_header(path):
    """Read only the bounded ASCII header of an ASCII or binary 3DGS PLY.

    Scalar values, the body length, and provenance are deliberately not inferred
    from its header. SH coefficients must form complete RGB SH bands.
    """
    consumed, file_format, current_element, vertex_count = 0, None, None, None
    elements, properties = set(), {}
    try:
        with Path(path).open("rb") as source:
            first = source.readline(_MAX_HEADER_LINE + 1)
            consumed += len(first)
            if first.rstrip(b"\r\n") != b"ply":
                raise InputValidationError("G0 does not start with a PLY header")
            while True:
                raw = source.readline(_MAX_HEADER_LINE + 1)
                consumed += len(raw)
                if not raw or len(raw) > _MAX_HEADER_LINE or consumed > _MAX_HEADER_BYTES:
                    raise InputValidationError("PLY header is missing, unterminated, or too large")
                try:
                    tokens = raw.decode("ascii").strip().split()
                except UnicodeDecodeError as exc:
                    raise InputValidationError("PLY header must be ASCII") from exc
                if not tokens:
                    raise InputValidationError("Empty PLY header directive")
                kind = tokens[0]
                if kind in {"comment", "obj_info"}:
                    continue
                if kind == "format":
                    if (file_format is not None or current_element is not None or len(tokens) != 3 or
                            tokens[1] not in {"ascii", "binary_little_endian", "binary_big_endian"} or tokens[2] != "1.0"):
                        raise InputValidationError("Unsupported or duplicate PLY format")
                    file_format = tokens[1]
                elif kind == "element":
                    if file_format is None or len(tokens) != 3 or not re.fullmatch(r"\d+", tokens[2]):
                        raise InputValidationError("Invalid PLY element")
                    current_element = tokens[1]
                    if current_element in elements:
                        raise InputValidationError("Duplicate PLY element")
                    elements.add(current_element)
                    if current_element == "vertex":
                        vertex_count = int(tokens[2])
                elif kind == "property":
                    if current_element is None:
                        raise InputValidationError("PLY property has no element")
                    scalar = len(tokens) == 3 and tokens[1] in _SCALAR_TYPES
                    listed = (len(tokens) == 5 and tokens[1] == "list" and
                              tokens[2] in _SCALAR_TYPES - _FLOAT_TYPES and tokens[3] in _SCALAR_TYPES)
                    if not scalar and not listed:
                        raise InputValidationError("Unsupported PLY property")
                    if current_element == "vertex":
                        name = tokens[-1]
                        if name in properties:
                            raise InputValidationError("Duplicate PLY vertex property")
                        properties[name] = tokens[1] if scalar else "list"
                elif kind == "end_header" and len(tokens) == 1:
                    break
                else:
                    raise InputValidationError("Unsupported PLY header directive")
    except OSError as exc:
        raise InputValidationError("Unable to read G0 PLY header") from exc
    if file_format is None or vertex_count is None or vertex_count <= 0:
        raise InputValidationError("PLY must declare a format and nonempty vertex element")
    for name in _REQUIRED_PROPERTIES:
        if properties.get(name) not in _FLOAT_TYPES:
            raise InputValidationError(f"PLY requires floating-point Gaussian property {name}")
    dc = {name for name in properties if name.startswith("f_dc_")}
    if dc != {"f_dc_0", "f_dc_1", "f_dc_2"}:
        raise InputValidationError("PLY must have exactly three DC SH coefficients")
    rest = [name for name in properties if name.startswith("f_rest_")]
    for name in rest:
        if not re.fullmatch(r"f_rest_(0|[1-9]\d*)", name) or properties[name] not in _FLOAT_TYPES:
            raise InputValidationError("Invalid higher-order SH coefficient property")
    if set(rest) != {f"f_rest_{index}" for index in range(len(rest))} or len(rest) % 3:
        raise InputValidationError("SH coefficients must be contiguous and have three color channels")
    bands_squared = len(rest) // 3 + 1
    bands = math.isqrt(bands_squared)
    if bands * bands != bands_squared:
        raise InputValidationError("SH coefficient count does not form complete spherical-harmonic bands")
    return {"format": file_format, "vertex_count": vertex_count,
            "sh_degree": bands - 1, "header_bytes": consumed}


def _json_object(content, role):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InputValidationError(f"Duplicate JSON key in {role}")
            result[key] = value
        return result
    try:
        result = json.loads(content.decode("utf-8"), object_pairs_hook=pairs,
                            parse_constant=lambda value: (_ for _ in ()).throw(
                                InputValidationError(f"Non-finite JSON literal in {role}")))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise InputValidationError(f"{role} must be a valid JSON object") from exc
    if not isinstance(result, dict):
        raise InputValidationError(f"{role} must be a JSON object")
    return result


def _no_distortion(value):
    if value is None or value == "none":
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value) and value == 0
    if isinstance(value, list):
        return all(_no_distortion(item) and not isinstance(item, (str, dict, list)) for item in value)
    return False


def _validate_untransformed(data, role, explicit=False):
    if explicit and any(key not in data for key in ("post_render_transform", "image_mirror", "mirror_axis")):
        raise InputValidationError("Extrinsics must explicitly declare no post-render transform or mirroring")
    if data.get("post_render_transform", "none") != "none":
        raise InputValidationError(f"{role}: post-render transforms are unsupported")
    if data.get("mirror_axis", "none") != "none":
        raise InputValidationError(f"{role}: image mirroring is unsupported")
    for key in ("image_mirror", "mirror_x", "mirror_y", "flip_x", "flip_y"):
        if key in data and data[key] is not False:
            raise InputValidationError(f"{role}: image mirroring is unsupported")
    for key in ("skew", "s"):
        if key in data and _number(data[key], key) != 0:
            raise InputValidationError(f"{role}: nonzero camera skew is unsupported")
    for key in ("distortion", "distortion_coefficients", "distortion_coeffs", "dist_coeffs", "D",
                "k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"):
        if key in data and not _no_distortion(data[key]):
            raise InputValidationError(f"{role}: lens distortion is unsupported")
    if "distortion_model" in data and data["distortion_model"] not in (None, "none"):
        raise InputValidationError(f"{role}: lens distortion models are unsupported")


def _coordinate_convention(value):
    if not isinstance(value, dict):
        raise InputValidationError("Intrinsics require an explicit coordinate_convention")
    expected = {"image_axes": "u right, v down", "camera_axes": "X right, Y down, Z forward",
                "pixel_origin": "top-left image boundary", "top_left_pixel_center": [0.5, 0.5]}
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise InputValidationError(f"Unsupported coordinate convention: {key}")


def _intrinsics(data, role, require_matrix=False):
    try:
        values = {key: _dimension(data[key], f"{role}.{key}") for key in ("width", "height")}
        values.update({key: _number(data[key], f"{role}.{key}") for key in ("fx", "fy", "cx", "cy")})
    except KeyError as exc:
        raise InputValidationError(f"Missing {role} intrinsic field {exc.args[0]}") from exc
    if values["fx"] <= 0 or values["fy"] <= 0:
        raise InputValidationError("Focal lengths must be positive")
    if require_matrix and "K" not in data:
        raise InputValidationError("Intrinsics require an explicit K matrix")
    if "K" in data:
        matrix = _matrix(data["K"], f"{role}.K")
        if any(matrix[row][column] != 0 for row, column in ((0, 1), (1, 0), (2, 0), (2, 1))):
            raise InputValidationError("K must have zero skew and an untransformed projective row")
        expected = [[values["fx"], 0, values["cx"]], [0, values["fy"], values["cy"]], [0, 0, 1]]
        if any(not _close(matrix[row][column], expected[row][column]) for row in range(3) for column in range(3)):
            raise InputValidationError(f"{role} K matrix disagrees with scalar intrinsics")
    return values


def parse_camera(intrinsics, extrinsics):
    """Whitelist supported K/R/T fields; never use embedded source references."""
    if not isinstance(intrinsics, dict) or not isinstance(extrinsics, dict):
        raise InputValidationError("Camera inputs must be JSON objects")
    _coordinate_convention(intrinsics.get("coordinate_convention"))
    if "coordinate_convention" in extrinsics:
        _coordinate_convention(extrinsics["coordinate_convention"])
    if extrinsics.get("camera_convention") != CAMERA_CONVENTION:
        raise InputValidationError("Only the explicit 3DGS transposed-R camera convention is supported")
    if "camera_convention" in intrinsics and intrinsics["camera_convention"] != CAMERA_CONVENTION:
        raise InputValidationError("Intrinsics camera convention conflicts with the supported convention")
    _validate_untransformed(intrinsics, "intrinsics")
    _validate_untransformed(extrinsics, "extrinsics", explicit=True)
    first, second = _intrinsics(intrinsics, "intrinsics", True), _intrinsics(extrinsics, "extrinsics")
    if any(not _close(first[key], second[key]) for key in first):
        raise InputValidationError("The two explicit camera files have inconsistent intrinsics")
    try:
        rotation = _matrix(extrinsics["R"], "R")
        translation = _vector(extrinsics["T"], 3, "T")
    except KeyError as exc:
        raise InputValidationError(f"Missing explicit extrinsic field {exc.args[0]}") from exc
    for row in range(3):
        for column in range(3):
            dot = sum(rotation[row][index] * rotation[column][index] for index in range(3))
            if abs(dot - (1 if row == column else 0)) > 1e-5:
                raise InputValidationError("R must be orthonormal")
    a, b, c = rotation
    determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
    if abs(determinant - 1) > 1e-5:
        raise InputValidationError("R must have determinant +1; reflections are unsupported")
    w2c = [[rotation[column][row] for column in range(3)] + [translation[row]] for row in range(3)]
    w2c.append([0.0, 0.0, 0.0, 1.0])
    return {**first, "pixel_center": "boundary", "R_stored_transposed": rotation,
            "T": translation, "w2c": w2c}


def preflight_event_only_inputs(g0_path, events_path, intrinsics_path, camera_path, *,
                                g0_time, time_scale, allowed_modalities=ALLOWED_MODALITIES):
    """Validate four explicit inputs and return minimal hash-bound provenance.

    ``g0_time`` is explicitly supplied in the user's chosen timeline, while
    positive ``time_scale`` converts raw event timestamps to that timeline.
    Their numeric validity is checked, not their physical alignment. Raw events
    are hashed as opaque bytes here; their format/content is not validated.
    """
    if (not isinstance(allowed_modalities, (tuple, list)) or len(allowed_modalities) != 3 or
            any(not isinstance(item, str) for item in allowed_modalities) or
            set(allowed_modalities) != set(ALLOWED_MODALITIES)):
        raise InputValidationError("Only G0, raw events, and explicit camera inputs are allowed")
    g0_time = _number(g0_time, "g0_time")
    time_scale = _number(time_scale, "time_scale")
    if time_scale <= 0:
        raise InputValidationError("time_scale must be positive")
    paths = _distinct_regular_paths((g0_path, events_path, intrinsics_path, camera_path))
    def signature(path):
        info = path.stat()
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
    before = {role: signature(path) for role, path in paths.items()}
    inspect_gaussian_ply_header(paths["g0"])
    hashes, camera_json = {}, {}
    for role, path in paths.items():
        hashes[role], content = _stream_read(path, capture=role in {"intrinsics", "extrinsics"})
        if content is not None:
            camera_json[role] = _json_object(content, role)
    camera = parse_camera(camera_json["intrinsics"], camera_json["extrinsics"])
    if any(signature(paths[role]) != value for role, value in before.items()):
        raise InputValidationError("An explicit input changed during preflight; retry with stable files")
    return {"scope": "preflight_only", "source_authenticity_verified": False,
            "allowed_modalities": list(ALLOWED_MODALITIES), "g0_time": g0_time,
            "time_scale": time_scale, "input_sha256": hashes, "camera": camera}
