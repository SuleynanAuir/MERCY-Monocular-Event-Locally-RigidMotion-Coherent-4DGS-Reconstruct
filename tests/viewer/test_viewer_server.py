"""HTTP/WebSocket tests use a fake renderer; no CUDA or model files required."""
import json
import asyncio
import struct
import unittest
from unittest.mock import patch

from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

import viewer
from utils.viewer_state import image_packet


class FakeRenderer:
    def __init__(self, path, cache):
        self.status = {"ready": True, "scene": {"frame_count": 360}}
        self.info = {"frame_count": 360, "fps": 60}

    def load(self):
        pass

    def render(self, request):
        return image_packet({"id": request["id"], "frame": request["frame"],
                             **({"stream": request["_stream"]} if "_stream" in request else {})}, b"jpeg")


class ViewerServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        with patch.object(viewer, "Renderer", FakeRenderer):
            self.client = TestClient(TestServer(viewer.create_app("unused.npz")))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_status_and_assets(self):
        for path in ("/", "/app.js", "/playback.js", "/style.css", "/api/status"):
            response = await self.client.get(path)
            self.assertEqual(response.status, 200)
            self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        response = await self.client.get("/api/status")
        self.assertTrue((await response.json())["ready"])

    async def test_unknown_file(self):
        response = await self.client.get("/viewer.py")
        self.assertEqual(response.status, 404)

    async def test_unavailable_mesh_rejects_request_without_breaking_gaussian(self):
        ws = await self.client.ws_connect('/ws')
        for mode in ('mesh', 'hybrid', 'clay', 'pseudocolor'):
            await ws.send_json({'render_mode': mode})
            self.assertIn('Mesh', (await ws.receive_json())['message'])
        await ws.send_json({'frame': 5})
        self.assertEqual((await self.read_packet(ws))['frame'], 5)
        await ws.close()

    async def test_foreign_host(self):
        response = await self.client.get("/api/status", headers={"Host": "untrusted.example:8890"})
        self.assertEqual(response.status, 403)

    async def test_foreign_origin(self):
        response = await self.client.get("/ws", headers={"Origin": "https://untrusted.example"})
        self.assertEqual(response.status, 403)

    async def test_socket_rejects_invalid_then_renders_valid(self):
        ws = await self.client.ws_connect("/ws")
        for data in ("not JSON", "null", '{"frame":360}', '{"yaw":NaN}', '{"zoom":true}'):
            await ws.send_str(data)
            self.assertEqual((await ws.receive_json())["type"], "error")
        await ws.send_json({"id": 7, "frame": 75})
        response = await ws.receive()
        self.assertEqual(response.type, WSMsgType.BINARY)
        size = struct.unpack("<I", response.data[:4])[0]
        self.assertEqual(json.loads(response.data[4:4 + size]), {"id": 7, "frame": 75})
        await ws.close()

    async def test_same_origin_socket(self):
        origin = str(self.client.make_url("/")).rstrip("/")
        ws = await self.client.ws_connect("/ws", headers={"Origin": origin})
        await ws.send_json({"frame": 359})
        self.assertEqual((await ws.receive()).type, WSMsgType.BINARY)
        await ws.close()

    async def read_packet(self, ws):
        response = await ws.receive(timeout=2)
        self.assertEqual(response.type, WSMsgType.BINARY)
        size = struct.unpack("<I", response.data[:4])[0]
        return json.loads(response.data[4:4 + size])

    async def test_stream_is_sequential_and_credit_bounded(self):
        ws = await self.client.ws_connect("/ws")
        await ws.send_json({"type": "stream", "id": 8, "frame": 0, "loop": True})
        for i in range(24):
            packet = await self.read_packet(ws)
            self.assertEqual(packet["frame"], i)
            self.assertEqual(packet["stream"]["seq"], i)
        # Producer cannot queue a 25th frame without an acknowledgement.
        with self.assertRaises(asyncio.TimeoutError):
            await ws.receive(timeout=.05)
        await ws.send_json({"type": "ack", "id": 8, "seq": 3})
        for i in range(24, 28):
            self.assertEqual((await self.read_packet(ws))["frame"], i)
        await ws.send_json({"type": "stop"})
        await ws.send_json({"id": 10, "frame": 99})
        self.assertEqual((await self.read_packet(ws))["frame"], 99)
        await ws.close()

    async def test_stream_end_and_replacement(self):
        ws = await self.client.ws_connect("/ws")
        await ws.send_json({"type": "stream", "id": 1, "frame": 358})
        self.assertEqual((await self.read_packet(ws))["frame"], 358)
        self.assertEqual((await self.read_packet(ws))["frame"], 359)
        self.assertEqual(await ws.receive_json(), {"type": "stream_end", "id": 1, "seq": 1})
        await ws.send_json({"type": "ack", "id": 0, "seq": 999})  # stale stream is ignored
        await ws.send_json({"type": "stream", "id": 2, "frame": 359, "loop": True})
        self.assertEqual((await self.read_packet(ws))["frame"], 359)
        self.assertEqual((await self.read_packet(ws))["frame"], 0)
        await ws.close()

    async def test_invalid_stream_options(self):
        ws = await self.client.ws_connect("/ws")
        for options in ({"rate": 120}, {"speed": 99}, {"speed": True}, {"loop": "yes"}):
            await ws.send_json({"type": "stream", **options})
            self.assertEqual((await ws.receive_json())["type"], "error")
        await ws.close()

    async def test_retained_frames_use_references_but_new_streams_never_reuse_them(self):
        ws = await self.client.ws_connect('/ws')
        # At 0.25x, the same source frame is used by four stream packets.
        await ws.send_json({'type': 'stream', 'id': 4, 'speed': .25})
        for _ in range(24):
            self.assertNotIn('client_cache_hit', await self.read_packet(ws))
        # Source frame 5 has been offered; the following packets 24..27 use 6.
        await ws.send_json({'type': 'ack', 'id': 4, 'seq': 3, 'cached_frames': [5]})
        for _ in range(4):
            self.assertNotIn('client_cache_hit', await self.read_packet(ws))
        await ws.send_json({'type': 'stop'})
        await ws.send_json({'type': 'stream', 'id': 5, 'frame': 0, 'speed': .25})
        for _ in range(24):
            self.assertNotIn('client_cache_hit', await self.read_packet(ws))
        # A source frame not yet offered must never be accepted as retained.
        await ws.send_json({'type': 'ack', 'id': 5, 'seq': 3, 'cached_frames': [100]})
        self.assertEqual((await ws.receive_json())['type'], 'error')
        await ws.close()

    async def test_cache_reference_preserves_time_and_bounds(self):
        ws = await self.client.ws_connect('/ws')
        await ws.send_json({'type': 'stream', 'id': 6, 'frame': 354, 'loop': True, 'speed': .25})
        for _ in range(24):
            await self.read_packet(ws)
        await ws.send_json({'type': 'ack', 'id': 6, 'seq': 23, 'cached_frames': list(range(354, 360))})
        # Drive through one loop; references appear only for the retained six
        # source frames and retain the new stream sequence, not the first pass.
        found = 0
        for seq in range(24, 1464):
            meta = await self.read_packet(ws)
            if meta.get('client_cache_hit'):
                found += 1
                self.assertGreaterEqual(meta['frame'], 354)
                self.assertEqual(meta['stream']['seq'], seq)
            if seq % 4 == 3:
                await ws.send_json({'type': 'ack', 'id': 6, 'seq': seq})
        self.assertEqual(found, 24)
        await ws.close()


class ObjectSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_mode_specific_comparison_keeps_both_panels_on_cached_loops(self):
        class ModeRenderer(FakeRenderer):
            def __init__(self, path, cache):
                super().__init__(path, cache)
                self.info.update(frame_count=2, comparison=False, comparison_modes=['mesh', 'hybrid', 'clay', 'pseudocolor'],
                                 mesh={'available': True}, hybrid={'available': True}, clay={'available': True}, pseudocolor={'available': True})
        scenes = {'fan': {'name': 'Fan', 'path': '', 'factory': ModeRenderer}}
        async with TestClient(TestServer(viewer.create_app(scenes=scenes))) as client:
            for mode in ('gaussian', 'mesh', 'hybrid', 'clay', 'pseudocolor'):
                ws = await client.ws_connect('/ws?scene=fan')
                await ws.send_json(dict(type='stream', id=1, frame=0, loop=True, render_mode=mode,
                                        width=512, height=512))
                for _ in range(24): await ws.receive()
                await ws.send_json(dict(type='ack', id=1, seq=23, cached_frames=[0, 1]))
                packet = (await ws.receive()).data
                size = struct.unpack('<I', packet[:4])[0]
                meta = json.loads(packet[4:4+size])
                self.assertTrue(meta['client_cache_hit'])
                self.assertEqual(meta.get('comparison', False), mode != 'gaussian')
                self.assertEqual(meta['width'], 512 if mode == 'gaussian' else 1024)
                if mode != 'gaussian': self.assertEqual(meta['left_frame'], meta['right_frame'])
                await ws.close()

    async def test_objects_are_scoped_to_each_connection(self):
        class NamedRenderer(FakeRenderer):
            def __init__(self, path, cache):
                super().__init__(path, cache)
                self.name = path
            def render(self, request):
                return image_packet({'object': self.name, 'frame': request['frame']}, b'jpeg')
        scenes = {key: {'name': key.title(), 'path': key, 'factory': NamedRenderer}
                  for key in ('fan', 'lego')}
        async with TestClient(TestServer(viewer.create_app(scenes=scenes, default_scene='fan'))) as client:
            response = await client.get('/api/status')
            status = await response.json()
            self.assertEqual(status['scene_id'], 'fan')
            self.assertEqual([item['id'] for item in status['objects']], ['fan', 'lego'])
            for url in ('/api/status?scene=missing', '/ws?scene=../secret'):
                self.assertEqual((await client.get(url)).status, 404)
            fan = await client.ws_connect('/ws?scene=fan')
            lego = await client.ws_connect('/ws?scene=lego')
            for socket, name in ((fan, 'fan'), (lego, 'lego'), (fan, 'fan')):
                await socket.send_json({'frame': 9})
                packet = (await socket.receive()).data
                size = struct.unpack('<I', packet[:4])[0]
                self.assertEqual(json.loads(packet[4:4+size]), {'object': name, 'frame': 9})
            await fan.close()
            await lego.close()

    async def test_failed_object_does_not_hide_other_objects(self):
        class Broken(FakeRenderer):
            def load(self):
                raise ValueError('bad asset')
        scenes = {'bad': {'name': 'Bad', 'path': '', 'factory': Broken},
                  'fan': {'name': 'Fan', 'path': '', 'factory': FakeRenderer}}
        async with TestClient(TestServer(viewer.create_app(scenes=scenes, default_scene='fan'))) as client:
            await asyncio.sleep(.05)
            bad = await (await client.get('/api/status?scene=bad')).json()
            good = await (await client.get('/api/status?scene=fan')).json()
            self.assertTrue(bad['error'])
            self.assertTrue(good['ready'])


if __name__ == "__main__":
    unittest.main()
