"""Measure bounded WebSocket pipelines and JPEG decode, not browser display FPS."""
import argparse
import asyncio
from collections import deque
import io
import json
import math
import statistics
import time

import aiohttp
from PIL import Image


async def probe(args):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        async with session.get(args.url.rstrip('/')+'/status', params={'scene':args.scene}) as response:
            response.raise_for_status()
            status = await response.json()
        if not status['ready']:
            raise RuntimeError(status['message'])
        intervals, roundtrips, sizes = [], [], []
        sent = deque()
        credits = asyncio.Semaphore(args.window)
        warmup = 5
        count = args.frames + warmup + 1
        async with session.ws_connect(args.url.rstrip('/')+'/ws', params={'scene':args.scene},
                                      max_msg_size=8*1024**2) as ws:
            async def send():
                last = 0.
                for i in range(count):
                    await credits.acquire()
                    await asyncio.sleep(max(0, 1/60-(time.perf_counter()-last)))
                    lo, hi = status['time_duration']
                    pose = dict(position=[status['position'][0]+.1*math.sin(i*.1),*status['position'][1:]],
                        yaw=status['yaw']+10*math.sin(i*.1),pitch=status['pitch'],time=lo+(hi-lo)*(i%61)/60,
                        width=args.width,height=args.height,fov=status.get('fov',60))
                    if args.motion == 'dolly':
                        pose.update(position=[*status['position'][:2],status['position'][2]-(1-math.cos(i*2*math.pi/360))],
                                    yaw=status['yaw'],time=lo+(i/60)%(hi-lo))
                    last = time.perf_counter()
                    sent.append(last)
                    await ws.send_json(pose)
            sender = asyncio.create_task(send())
            previous = None
            try:
                for i in range(count):
                    message = await ws.receive(timeout=30)
                    if message.type != aiohttp.WSMsgType.BINARY:
                        raise RuntimeError(f'Expected JPEG, received {message.type}: {str(message.data)[:200]}')
                    started = sent.popleft()
                    with Image.open(io.BytesIO(message.data)) as image:
                        image.load()
                        if image.size != (args.width,args.height):
                            raise ValueError('Unexpected image dimensions')
                    now = time.perf_counter()
                    if i > warmup:
                        intervals.append((now-previous)*1000)
                        roundtrips.append((now-started)*1000)
                        sizes.append(len(message.data))
                    previous = now
                    credits.release()
                await sender
            finally:
                sender.cancel()
                await asyncio.gather(sender,return_exceptions=True)
        def p95(values):
            return sorted(values)[math.ceil(.95*len(values))-1]
        return dict(scope='bounded WebSocket + network + JPEG decode; NOT browser display FPS',
            scene=args.scene,motion=args.motion,model_sha256=status['sha256'],revision=status['revision'],
            max_in_flight=args.window,frames=len(intervals),width=args.width,height=args.height,
            duration_seconds=sum(intervals)/1000,mean_ms=statistics.mean(intervals),p95_ms=p95(intervals),
            mean_fps=1000/statistics.mean(intervals),roundtrip_mean_ms=statistics.mean(roundtrips),
            roundtrip_p95_ms=p95(roundtrips),mean_jpeg_bytes=statistics.mean(sizes),
            samples_ms=intervals,roundtrip_samples_ms=roundtrips)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True)
    parser.add_argument('--scene',required=True)
    parser.add_argument('--motion',choices=['orbit','dolly'],default='orbit')
    parser.add_argument('--frames',type=int,default=120)
    parser.add_argument('--window',type=int,choices=[1,2,3],default=1)
    parser.add_argument('--width',type=int,default=960)
    parser.add_argument('--height',type=int,default=640)
    args=parser.parse_args()
    if not 1<=args.frames<=10000:parser.error('frames must be in [1,10000]')
    if not all(256<=v<=1280 for v in [args.width,args.height]):parser.error('dimensions must be 256..1280')
    print(json.dumps(asyncio.run(probe(args)),indent=2))
