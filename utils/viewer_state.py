"""CPU-only camera math and bounded wire protocol for the Gaussian viewer."""

import json
import math
import struct

import numpy as np


BACKGROUNDS = {"dark": (0.055, 0.067, 0.085), "light": (0.94, 0.95, 0.97),
               "gray": (0.3, 0.32, 0.35), "black": (0., 0., 0.)}
STREAM_WINDOW = 24


def validate_stream(data, frame_count):
    result = validate_request(data, frame_count)
    speed, rate, loop = data.get("speed", 1), data.get("rate", 60), data.get("loop", False)
    if isinstance(speed, bool) or not isinstance(speed, (float, int)) or speed not in (.25, .5, 1, 2):
        raise ValueError("Unsupported playback speed")
    if isinstance(rate, bool) or rate not in (30, 60):
        raise ValueError("Unsupported stream rate")
    if not isinstance(loop, bool):
        raise ValueError("Loop must be boolean")
    result.update(speed=float(speed), rate=int(rate), loop=loop)
    return result


def stream_frame(request, seq, source_fps, frame_count):
    """Output at a fixed rate, preserving source timestamps and requested speed."""
    position = request["frame"] + seq * source_fps * request["speed"] / request["rate"]
    frame = math.floor(position + 1e-9)
    if request["loop"]:
        return frame % frame_count, False
    return min(frame, frame_count - 1), frame >= frame_count - 1


def validate_request(data, frame_count):
    if not isinstance(data, dict):
        raise ValueError("Request must be a JSON object")
    result = {}
    for key, default, low, high, integer in (
        ("id", 0, 0, 2**31 - 1, True), ("frame", 0, 0, frame_count - 1, True),
        ("width", 768, 128, 1280, True), ("height", 768, 128, 1280, True),
        ("yaw", 0, -3600, 3600, False), ("pitch", 0, -85, 85, False),
        ("zoom", 1, 0.2, 8, False), ("pan_x", 0, -5, 5, False),
        ("pan_y", 0, -5, 5, False), ("fovy", 60, 30, 90, False),
        ("quality", 85, 60, 95, True),
    ):
        value = data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} is out of range")
        if integer and int(value) != value:
            raise ValueError(f"{key} must be an integer")
        result[key] = int(value) if integer else float(value)
    if result["width"] * result["height"] > 1280 * 960:
        raise ValueError("Image exceeds the pixel budget")
    background = data.get("background", "dark")
    if not isinstance(background, str) or background not in BACKGROUNDS:
        raise ValueError("Unknown background")
    result["background"] = background
    return result


def orbit_pose(base_pose, target, request):
    """Orbit relative to the saved OpenGL camera; angles are degrees."""
    base = np.asarray(base_pose, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    radius = float(np.linalg.norm(base[:3, 3] - target))
    if radius < 1e-6:
        raise ValueError("Camera is coincident with its target")
    yaw, pitch = np.deg2rad([request["yaw"], request["pitch"]])
    back = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch),
                     np.cos(yaw) * np.cos(pitch)])
    right = np.array([np.cos(yaw), 0., -np.sin(yaw)])
    up = np.cross(back, right)
    basis = base[:3, :3] @ np.stack((right, up, back), axis=1)
    shift = radius * (request["pan_x"] * basis[:, 0] + request["pan_y"] * basis[:, 1])
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = basis
    pose[:3, 3] = target + shift + radius * request["zoom"] * basis[:, 2]
    return pose


def image_packet(metadata, jpeg):
    header = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return struct.pack("<I", len(header)) + header + jpeg


def cached_image_reference(request, comparison=False):
    """Reference a JPEG explicitly retained by this stream's browser client."""
    frame = request['frame']
    return image_packet({'id': request['id'], 'frame': frame, 'stream': request['_stream'],
                         'width': request['width'] * (2 if comparison else 1),
                         'height': request['height'], 'render_ms': 0., 'encode_ms': 0.,
                         'cache_hit': True, 'client_cache_hit': True,
                         **({'comparison': True, 'panel_width': request['width'],
                             'left_frame': frame, 'right_frame': frame} if comparison else {})}, b'')
