import unittest

from utils.viewer_cache import EncodedFrameCache
from utils.viewer_state import validate_request


class EncodedFrameCacheTests(unittest.TestCase):
    def setUp(self):
        self.request = validate_request({'frame': 1}, 400)

    def test_ids_and_stream_timing_do_not_change_pixels(self):
        cache = EncodedFrameCache()
        cache.put(self.request, b'pixels')
        self.assertEqual(cache.get({**self.request, 'id': 99, '_stream': {'seq': 44}}), b'pixels')

    def test_every_pixel_parameter_invalidates_cached_image(self):
        cache = EncodedFrameCache()
        cache.put(self.request, b'pixels')
        for field in cache.FIELDS:
            value = 'light' if field == 'background' else self.request[field] + 1
            with self.subTest(field=field):
                self.assertIsNone(cache.get({**self.request, field: value}))

    def test_lru_is_bounded_by_bytes_and_count(self):
        cache = EncodedFrameCache(max_bytes=6, max_entries=2)
        requests = [{**self.request, 'frame': i} for i in range(3)]
        cache.put(requests[0], b'ab'); cache.put(requests[1], b'cde')
        self.assertEqual(cache.get(requests[0]), b'ab')
        cache.put(requests[2], b'fgh')
        self.assertIsNone(cache.get(requests[1]))
        self.assertEqual(cache.bytes, 5)
        cache.put(requests[0], b'ij')
        self.assertEqual(cache.bytes, 5)
        cache.put(requests[0], b'toolarge')
        self.assertEqual(cache.bytes, 3)
        self.assertIsNone(cache.get(requests[0]))
        cache = EncodedFrameCache(max_bytes=100, max_entries=1)
        cache.put(requests[0], b'a'); cache.put(requests[1], b'b')
        self.assertEqual(len(cache.entries), 1)
        self.assertIsNone(cache.get(requests[0]))


if __name__ == '__main__':
    unittest.main()
