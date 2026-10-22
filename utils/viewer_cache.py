"""Bounded, lossless reuse of already encoded renders of an immutable scene."""
from collections import OrderedDict
from utils.hybrid_sampling import DEFAULT_HYBRID_QUALITY


class EncodedFrameCache:
    # Include every validated field that affects pixels. Request IDs and stream
    # sequence numbers intentionally aren't cached and must be rebuilt per send.
    FIELDS = ('frame', 'width', 'height', 'yaw', 'pitch', 'zoom', 'pan_x', 'pan_y',
              'fovy', 'background', 'quality')

    def __init__(self, max_bytes=96 * 1024**2, max_entries=1200):
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries, self.bytes = OrderedDict(), 0

    def key(self, request):
        hybrid_quality = request.get('hybrid_quality', DEFAULT_HYBRID_QUALITY) if request.get('render_mode') == 'hybrid' else None
        return tuple(request[field] for field in self.FIELDS) + (hybrid_quality,)

    def get(self, request):
        key = self.key(request)
        result = self.entries.get(key)
        if result is not None:
            self.entries.move_to_end(key)
        return result

    def put(self, request, jpeg):
        key = self.key(request)
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.bytes -= len(previous)
        if len(jpeg) > self.max_bytes or self.max_entries <= 0:
            return
        self.entries[key] = jpeg
        self.bytes += len(jpeg)
        while self.bytes > self.max_bytes or len(self.entries) > self.max_entries:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= len(removed)
