"""Exercise a running viewer's actual GPU renderer over its WebSocket protocol."""
import argparse
import asyncio
import io
import json
from pathlib import Path
import statistics
import struct
import time

from aiohttp import ClientSession, ClientTimeout, WSMsgType
import numpy as np
from PIL import Image


async def check(url, samples):
    timings = []
    async with ClientSession(timeout=ClientTimeout(total=120)) as client:
        async with client.get(url + "/api/status") as response:
            status = await response.json()
            assert status["ready"], status
        async with client.get(url + "/ws", headers={"Origin": "https://untrusted.example"}) as response:
            assert response.status == 403
        async with client.ws_connect(url + "/ws", origin=url) as ws:
            await ws.send_json({"frame": -1})
            assert (await ws.receive_json())["type"] == "error"
            images = []
            requests = [{"frame": 0}, {"frame": 75}, {"frame": 75, "yaw": 40},
                        {"frame": 75, "yaw": 40, "background": "light"}]
            for i, values in enumerate(requests + [{"frame": i % status["scene"]["frame_count"]}
                                                  for i in range(samples)]):
                started = time.perf_counter()
                await ws.send_json({"id": i, "width": 768, "height": 768,
                                    "fovy": status['scene']['fovy'], **values})
                message = await ws.receive(timeout=30)
                assert message.type == WSMsgType.BINARY, message
                size = struct.unpack("<I", message.data[:4])[0]
                meta = json.loads(message.data[4:4 + size])
                assert meta["id"] == i and meta["frame"] == values["frame"]
                pixels = np.asarray(Image.open(io.BytesIO(message.data[4 + size:])))
                assert pixels.shape == (768, 768, 3)
                if i < 4:
                    images.append(pixels.astype(np.float32))
                else:
                    timings.append({"roundtrip_ms": (time.perf_counter() - started) * 1000,
                                    "render_ms": meta["render_ms"], "encode_ms": meta["encode_ms"]})
            differences = [float(np.abs(a - b).mean()) for a, b in zip(images[:-1], images[1:])]
            assert all(value > 1 for value in differences), differences
    return {"passed": True, "samples": samples, "resolution": [768, 768],
            "gaussians": status["scene"]["gaussian_count"], "cache": status["scene"]["cache"],
            "mean_render_ms": statistics.mean(t["render_ms"] for t in timings),
            "mean_encode_ms": statistics.mean(t["encode_ms"] for t in timings),
            "mean_roundtrip_ms": statistics.mean(t["roundtrip_ms"] for t in timings),
            "frame_camera_background_mae": differences,
            "invalid_frame_rejected": True, "foreign_origin_rejected": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8890")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    result = asyncio.run(check(args.url.rstrip("/"), args.samples))
    data = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data)
    print(data)


if __name__ == "__main__":
    main()
