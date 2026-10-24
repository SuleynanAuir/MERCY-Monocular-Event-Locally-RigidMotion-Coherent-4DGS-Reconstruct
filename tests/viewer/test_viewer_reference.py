import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp.test_utils import TestClient, TestServer

import viewer
from utils.viewer_reference import ReferenceSequence


class ReferenceTests(unittest.IsolatedAsyncioTestCase):
    def test_lossless_png_sequence_and_extension_allowlist(self):
        self.data['frame_extension'] = '.png'
        self.path.write_text(json.dumps(self.data))
        for i in range(3):
            (self.root / f'{i:04d}.png').write_bytes(b'png-frame')
        sequence = ReferenceSequence(self.path, self.scene)
        self.assertEqual(sequence.frame_path(1).suffix, '.png')
        self.assertEqual(sequence.frame_path(1).read_bytes(), b'png-frame')
        self.data['frame_extension'] = '/../../private'
        self.path.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, 'format'):
            ReferenceSequence(self.path, self.scene)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'reference.json'
        self.scene = dict(frame_count=3, fps=30, times_seconds=[0, 1/30, 2/30])
        self.data = dict(format='rgb_event_reference_v1', **self.scene, width=640, height=640,
                         alignment=dict(space='rgb_image', event_frame_offset=-1))
        self.path.write_text(json.dumps(self.data))
        for i in range(3):
            (self.root / f'{i:04d}.jpg').write_bytes(f'frame-{i}'.encode())

    def test_exact_timeline_required_and_no_truncated_references(self):
        self.assertEqual(ReferenceSequence(self.path, self.scene).frame_path(2).read_bytes(), b'frame-2')
        for override in (dict(frame_count=2), dict(fps=60), dict(times_seconds=[.01, .04, .07]),
                         dict(times_seconds=[0, float('nan'), 2/30])):
            with self.subTest(override=override), self.assertRaises(ValueError):
                ReferenceSequence(self.path, {**self.scene, **override})
        (self.root / '0002.jpg').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            ReferenceSequence(self.path, self.scene)

    def test_wrong_space_and_file_symlinks_rejected(self):
        self.data['alignment']['space'] = 'event_sensor_unregistered'
        self.path.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, 'RGB'):
            ReferenceSequence(self.path, self.scene)
        self.data['alignment']['space'] = 'rgb_image'
        self.path.write_text(json.dumps(self.data))
        with tempfile.TemporaryDirectory() as other:
            outside = Path(other) / 'outside.jpg'; outside.write_bytes(b'private')
            (self.root / '0001.jpg').unlink(); (self.root / '0001.jpg').symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                ReferenceSequence(self.path, self.scene)

    async def test_http_frames_are_scene_scoped_bounded_and_guarded(self):
        scene = self.scene
        class Renderer:
            def __init__(self, path, cache):
                self.info = dict(scene)
                self.status = dict(ready=True, scene=self.info)
            def load(self):
                pass
        scenes = {'lego': dict(path='unused', name='Lego', factory=Renderer, reference_path=self.path),
                  'other': dict(path='unused', name='Other', factory=Renderer)}
        async with TestClient(TestServer(viewer.create_app(scenes=scenes))) as client:
            for _ in range(100):
                data = await (await client.get('/api/status?scene=lego')).json()
                if data['ready']:
                    break
                await asyncio.sleep(.01)
            self.assertTrue(data['scene']['reference']['available'])
            for i in range(3):
                response = await client.get(f'/api/reference/{i}?scene=lego')
                self.assertEqual(response.status, 200)
                self.assertEqual(await response.read(), f'frame-{i}'.encode())
            for suffix in ('-1?scene=lego', '3?scene=lego', '00000000000000?scene=lego',
                           '0?scene=other', '0?scene=unknown', 'reference.json?scene=lego'):
                self.assertEqual((await client.get('/api/reference/'+suffix)).status, 404)
            response = await client.get('/api/reference/0?scene=lego', headers={'Origin': 'https://foreign.test'})
            self.assertEqual(response.status, 403)


if __name__ == '__main__':
    unittest.main()
