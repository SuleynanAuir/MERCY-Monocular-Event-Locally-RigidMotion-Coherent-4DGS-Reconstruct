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

    def test_display_modes_never_reuse_other_material_pixels(self):
        cache = EncodedFrameCache()
        for mode in ('mesh', 'clay', 'gaussian', 'hybrid', 'pseudocolor'):
            request = validate_request(dict(self.request, render_mode=mode), 400)
            self.assertIsNone(cache.get(request))
            cache.put(request, mode.encode())
        for mode in ('mesh', 'clay', 'gaussian', 'hybrid', 'pseudocolor'):
            self.assertEqual(cache.get(dict(self.request, render_mode=mode)), mode.encode())

    def test_white_wire_and_solid_are_isolated_and_do_not_change_gs(self):
        cache = EncodedFrameCache()
        wire = validate_request({'render_mode': 'clay', 'clay_style': 'wire'}, 3)
        solid = dict(wire, clay_style='solid')
        self.assertEqual(validate_request({'render_mode': 'clay'}, 3)['clay_style'], 'solid')
        cache.put(wire, b'wire')
        self.assertIsNone(cache.get(solid))
        cache.put(solid, b'solid')
        self.assertEqual(cache.get(wire), b'wire')
        gs = dict(wire, render_mode='gaussian')
        cache.put(gs, b'original')
        self.assertEqual(cache.get(dict(gs, clay_style='solid')), b'original')
        for value in ('quad', None, 4):
            with self.assertRaises(ValueError):
                validate_request(dict(wire, clay_style=value), 3)
        with self.assertRaises(ValueError):
            validate_request({'render_mode': 'clay_mesh'}, 3)

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

    def test_hybrid_quality_isolated_but_does_not_change_gaussian_cache(self):
        cache = EncodedFrameCache()
        request = dict(self.request, render_mode='hybrid', hybrid_quality='balanced')
        cache.put(request, b'previous')
        self.assertIsNone(cache.get(dict(request, hybrid_quality='detail')))
        cache.put(dict(request, hybrid_quality='detail'), b'detailed')
        self.assertEqual(cache.get(request), b'previous')
        cache.put(self.request, b'gaussian')
        self.assertEqual(cache.get(dict(self.request, hybrid_quality='balanced')), b'gaussian')
        implicit = dict(self.request, render_mode='hybrid')
        implicit.pop('hybrid_quality')
        cache.put(implicit, b'surface')
        self.assertEqual(cache.get(dict(implicit, hybrid_quality='surface')), b'surface')
        self.assertIsNone(cache.get(dict(implicit, hybrid_quality='native')))


if __name__ == '__main__':
    unittest.main()
