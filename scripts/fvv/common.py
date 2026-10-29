"""CPU-only contracts for the isolated Hybrid 3D-4DGS service."""
import hashlib
import math
from pathlib import Path


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
    position = data['position']
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError('position must have three coordinates')
    return dict(position=[number(v, -1e6, 1e6) for v in position],
                yaw=number(data['yaw'], -1e6, 1e6), pitch=number(data['pitch'], -89.9, 89.9),
                time=number(data['time'], *duration), width=int(number(data['width'], 256, 1280)),
                height=int(number(data['height'], 256, 1280)), fov=number(data.get('fov', 60), 20, 100))


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
