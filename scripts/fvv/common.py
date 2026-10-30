"""CPU-only contracts for the isolated Hybrid 3D-4DGS service."""
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path


def model_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def model_name(scene):
    import re
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', scene):
        raise ValueError('场景名称无效')
    return scene + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def usage(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())


def validate_pose(data, duration):
    def number(value, lo, hi):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('相机参数或时间超出范围')
        return float(value)
    request_id=data.get('request_id')
    if request_id is not None and (type(request_id) is not int or not 0<=request_id<2**32):
        raise ValueError('Invalid frame request identifier')
    coalesce=data.get('coalesce',True)
    if type(coalesce) is not bool:raise ValueError('coalesce must be boolean')
    temporal_samples = data.get('temporal_samples', 1)
    if type(temporal_samples) is not int or temporal_samples not in (1, 3):
        raise ValueError('temporal_samples must be 1 or 3')
    position = data['position']
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError('position must have three coordinates')
    return dict(request_id=request_id,coalesce=coalesce,temporal_samples=temporal_samples, position=[number(v, -1e6, 1e6) for v in position],
                yaw=number(data['yaw'], -1e6, 1e6), pitch=number(data['pitch'], -89.9, 89.9),
                time=number(data['time'], *duration), width=int(number(data['width'], 256, 1280)),
                height=int(number(data['height'], 256, 1280)), fov=number(data.get('fov', 60), 20, 100),
                jpeg_quality=int(number(data.get('jpeg_quality',95),92,98)))


class LimitedWriter:
    """Stop serialization before a file can exceed its reserved disk budget."""
    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit

    def write(self, data):
        if self.stream.tell() + len(data) > self.limit:
            raise OSError('FVV storage budget exceeded')
        return self.stream.write(data)

    def flush(self):
        return self.stream.flush()

    def tell(self):
        return self.stream.tell()
