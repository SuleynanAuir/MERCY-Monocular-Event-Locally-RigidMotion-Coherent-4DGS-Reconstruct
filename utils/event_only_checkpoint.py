"""Hash-bound, weights-only-safe checkpoints for explicit G0/event motion.

This is a custom embedded-deformation format, not SC-GS network weights.
Loading reads only the explicitly supplied checkpoint and G0; paths recorded
inside provenance are descriptive strings and are never followed.
"""

from collections.abc import Mapping
import math
from numbers import Integral, Real
from pathlib import Path
import re

import torch

from utils.event_only_inputs import inspect_gaussian_ply_header, sha256_file
from utils.event_only_motion import (EventOnlyMotion, HARD_RIGID_GROUP_KIND,
                                     _hard_rigid_group_metadata)
from utils.event_only_renderer import make_camera


FORMAT = "event_only_embedded_deformation_v1"
_CAMERA_KEYS = {"width", "height", "fx", "fy", "cx", "cy", "w2c", "pixel_center"}
_GRAPH_KEYS = {"centers", "knn_indices", "knn_weights", "edges", "scene_scale", "metadata"}
_BUFFER_KEYS = {"centers", "knn_indices", "knn_weights", "edges", "scene_scale", "times"}
_STATE_KEYS = _BUFFER_KEYS | {"rotation_vectors", "translations"}
_PAYLOAD_KEYS = {"format", "graph", "times", "state_dict", "camera", "input_provenance",
                 "input_sha256", "training_summary", "gaussian_count", "chunk_size"}


def _plain(value, location="metadata"):
    """Copy only finite JSON-like values; never pickle numpy/path/class objects."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite number in {location}")
        return number
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"Non-string key in {location}")
        return {key: _plain(item, f"{location}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{location}[]") for item in value]
    raise ValueError(f"Unsupported object in {location}: {type(value).__name__}")


def _provenance(value):
    value = _plain(value, "input_provenance")
    if not isinstance(value, dict):
        raise ValueError("input_provenance must be a dictionary")
    allowed = value.get("allowed_modalities")
    if (not isinstance(allowed, list) or len(allowed) != 3 or
            any(not isinstance(item, str) for item in allowed) or
            set(allowed) != {"g0", "events", "camera"}):
        raise ValueError("Only exactly G0, events, and camera modalities are permitted")
    hashes = value.get("input_sha256")
    if (not isinstance(hashes, dict) or set(hashes) != {"g0", "events", "intrinsics", "extrinsics"} or
            any(not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for digest in hashes.values())):
        raise ValueError("Four explicit input SHA256 values are required")
    if value.get("forbidden_inputs_loaded") or value.get("legacy_pipeline_used", False):
        raise ValueError("Checkpoint provenance declares a forbidden/legacy input")
    return value


def _camera(value, strict=False):
    if not isinstance(value, Mapping) or (strict and set(value) != _CAMERA_KEYS):
        raise ValueError("Checkpoint camera must contain only the numeric camera whitelist")
    if value.get("pixel_center", "boundary") != "boundary":
        raise ValueError("Only boundary-coordinate camera intrinsics are supported")
    for key in ("fx", "fy", "cx", "cy"):
        if key not in value or isinstance(value[key], bool) or not isinstance(value[key], Real):
            raise ValueError(f"Camera {key} must be numeric, not a boolean/string")
    camera = make_camera(value, device="cpu")
    return {"width": camera.width, "height": camera.height,
            "fx": camera.fx, "fy": camera.fy, "cx": camera.cx, "cy": camera.cy,
            "w2c": _plain(value["w2c"], "camera.w2c"), "pixel_center": "boundary"}


def _validate_payload(payload):
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS or payload.get("format") != FORMAT:
        raise ValueError("Unsupported event-only checkpoint format or payload fields")
    count, chunk = payload["gaussian_count"], payload["chunk_size"]
    if (type(count) is not int or count <= 0 or type(chunk) is not int or chunk <= 0):
        raise ValueError("Invalid Gaussian count or motion chunk size")
    provenance = _provenance(payload["input_provenance"])
    if payload["input_sha256"] != provenance["input_sha256"]:
        raise ValueError("Duplicated input hashes do not agree")
    _plain(payload["training_summary"], "training_summary")
    _camera(payload["camera"], strict=True)
    graph, state, times = payload["graph"], payload["state_dict"], payload["times"]
    if not isinstance(graph, dict) or set(graph) != _GRAPH_KEYS:
        raise ValueError("Invalid graph fields")
    metadata = _plain(graph["metadata"], "graph.metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Graph metadata must be a dictionary")
    if metadata.get("kind") == HARD_RIGID_GROUP_KIND:
        # Event-derived membership must not masquerade as a G0-only graph.
        # The motion constructor below validates this mode's exact hard
        # binding, empty edges, all-group membership, and source declarations.
        if metadata.get("input_sha256") != payload["input_sha256"]:
            raise ValueError("Hard rigid group input SHA256 values do not match checkpoint provenance")
    elif (metadata.get("kind") != "g0_only_latent_deformation_graph" or
          metadata.get("node_source") != "explicit_G0_positions_only" or
          metadata.get("blender_parts") is not False):
        raise ValueError("Graph must be G0-only latent or explicitly event-inferred hard rigid groups")
    if not isinstance(state, dict) or set(state) != _STATE_KEYS:
        raise ValueError("Invalid motion state_dict fields")
    for name, tensor in state.items():
        dtype = torch.int64 if name in {"knn_indices", "edges"} else torch.float64 if name == "times" else torch.float32
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or
                tensor.dtype != dtype or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"State tensor {name} has invalid dtype/device/values")
    if (not isinstance(times, torch.Tensor) or times.dtype != torch.float64 or
            times.device.type != "cpu" or not torch.equal(times, state["times"])):
        raise ValueError("Checkpoint time vectors do not agree")
    for key in _GRAPH_KEYS - {"metadata"}:
        if (not isinstance(graph[key], torch.Tensor) or graph[key].device.type != "cpu" or
                graph[key].dtype != state[key].dtype or not torch.equal(graph[key], state[key])):
            raise ValueError(f"Graph/state binding mismatch for {key}")
    if state["knn_indices"].ndim != 2 or len(state["knn_indices"]) != count:
        raise ValueError("Graph binding length does not match G0 count")
    if state["scene_scale"].ndim != 0:
        raise ValueError("scene_scale must be scalar")
    # Constructor validates finite knots, t0 anchoring, proper graph shapes,
    # neighbor range, normalized nonnegative weights, and graph edge range.
    motion = EventOnlyMotion(graph, times, chunk_size=chunk)
    expected = motion.state_dict()
    if any(state[key].shape != expected[key].shape for key in _STATE_KEYS):
        raise ValueError("Motion parameter shapes disagree with graph and knot times")
    motion.load_state_dict(state, strict=True)
    return motion


def save_event_only_checkpoint(path, motion, graph, camera, provenance, training_summary):
    """Write one new checkpoint; existing files are never overwritten.

    ``provenance`` must contain ``allowed_modalities`` and four ``input_sha256``
    hashes (g0/events/intrinsics/extrinsics). The existing preflight record can
    be passed directly. Returns the weights-only-safe payload that was saved.
    """
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    if not isinstance(motion, EventOnlyMotion):
        raise TypeError("motion must be EventOnlyMotion, not a legacy deformation network")
    state = {key: value.detach().to("cpu").clone().contiguous()
             for key, value in motion.state_dict().items()}
    if not isinstance(graph, Mapping) or set(graph) != _GRAPH_KEYS:
        raise ValueError("Graph must contain exactly the event-only graph fields")
    # The motion constructor normalizes weights once. Check the provided graph
    # against that canonical state, allowing only its float32 rounding error.
    for key in _GRAPH_KEYS - {"metadata"}:
        original = torch.as_tensor(graph[key], dtype=state[key].dtype, device="cpu")
        agrees = (torch.allclose(original, state[key], atol=1e-6, rtol=1e-6)
                  if key in {"knn_weights", "scene_scale"} and original.shape == state[key].shape
                  else torch.equal(original, state[key]))
        if not agrees:
            raise ValueError(f"Supplied graph does not match motion buffer {key}")
    graph_metadata = graph["metadata"]
    graph_is_hard = (isinstance(graph_metadata, Mapping) and
                     graph_metadata.get("kind") == HARD_RIGID_GROUP_KIND)
    motion_is_hard = motion.metadata.get("kind") == HARD_RIGID_GROUP_KIND
    if graph_is_hard or motion_is_hard:
        if graph_is_hard != motion_is_hard:
            raise ValueError("Supplied graph mode does not match motion provenance")
        _hard_rigid_group_metadata(graph, state["knn_indices"], state["edges"], len(state["centers"]))
        for key in ("binding_source", "node_source", "blender_parts", "input_sha256"):
            if graph_metadata.get(key) != motion.metadata.get(key):
                raise ValueError(f"Supplied graph {key} does not match motion provenance")
    canonical_graph = {key: state[key] for key in _GRAPH_KEYS - {"metadata"}}
    canonical_graph["metadata"] = _plain(graph["metadata"], "graph.metadata")
    source = _provenance(provenance)
    payload = {"format": FORMAT, "graph": canonical_graph, "times": state["times"],
               "state_dict": state, "camera": _camera(camera), "input_provenance": source,
               "input_sha256": dict(source["input_sha256"]),
               "training_summary": _plain(training_summary, "training_summary"),
               "gaussian_count": len(state["knn_indices"]), "chunk_size": motion.chunk_size}
    _validate_payload(payload)
    with destination.open("xb") as output:
        torch.save(payload, output)
    return payload


def load_event_only_checkpoint(path, g0_path, device="cuda"):
    """Return (motion, numeric camera, payload), checking the explicit G0 hash.

    ``weights_only=True`` is mandatory; there is no unsafe-pickle fallback.
    No path from the checkpoint/provenance is opened.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    motion = _validate_payload(payload)
    if sha256_file(g0_path) != payload["input_sha256"]["g0"]:
        raise ValueError("Explicit G0 SHA256 does not match checkpoint input")
    header = inspect_gaussian_ply_header(g0_path)
    if header["vertex_count"] != payload["gaussian_count"]:
        raise ValueError("Explicit G0 row count does not match graph binding length")
    return motion.to(device), _camera(payload["camera"], strict=True), payload
