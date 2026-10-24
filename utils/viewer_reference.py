"""Read-only, frame-addressed RGB/event references for the synchronized viewer."""
import json
from pathlib import Path

import numpy as np


class ReferenceSequence:
    def __init__(self, path, scene):
        self.root = Path(path).resolve().parent
        data = json.loads(Path(path).read_text())
        if data.get('format') != 'rgb_event_reference_v1':
            raise ValueError('Unsupported reference manifest')
        count = data['frame_count']
        if isinstance(count, bool) or not isinstance(count, int) or count != scene['frame_count']:
            raise ValueError('Reference and render frame counts differ')
        if not np.isclose(data['fps'], scene['fps'], atol=1e-6, rtol=0):
            raise ValueError('Reference and render FPS differ')
        times = np.asarray(data['times_seconds'], dtype=float)
        render_times = np.asarray(scene['times_seconds'], dtype=float)
        if (times.shape != (count,) or not np.isfinite(times).all()
                or not np.allclose(times, render_times, atol=.0011, rtol=0)):
            raise ValueError('Reference and render timestamps differ')
        if not all(isinstance(data[k], int) and 1 <= data[k] <= 4096 for k in ('width', 'height')):
            raise ValueError('Invalid reference dimensions')
        if data['alignment'].get('space') != 'rgb_image':
            raise ValueError('References must be prepared in RGB image coordinates')
        extension = data.get('frame_extension', '.jpg')
        if extension not in ('.jpg', '.png'):
            raise ValueError('Unsupported reference image format')
        self.paths = [(self.root / f'{i:04d}{extension}').resolve() for i in range(count)]
        if any(p.parent != self.root or not p.is_file() for p in self.paths):
            raise ValueError('Incomplete reference frames')
        self.info = {k: data[k] for k in ('width', 'height', 'frame_count', 'fps', 'alignment')}
        self.info['available'] = True

    def frame_path(self, index):
        if not 0 <= index < len(self.paths):
            raise IndexError('Reference frame out of range')
        return self.paths[index]
