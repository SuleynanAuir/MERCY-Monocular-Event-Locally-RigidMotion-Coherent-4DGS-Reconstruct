"""Replace border-keyed Lego UI references with audited source coverage."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from .protocol import save_json, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--data',required=True,type=Path)
    args=p.parse_args();directory=args.root/'lego'
    matte=directory/'source-mattes/matte.json'
    deadline=time.monotonic()+1800
    while not matte.is_file():
        if time.monotonic()>deadline:raise TimeoutError('Source coverage did not complete')
        time.sleep(5)
    original=directory/'references';new=directory/'references-geometry'
    old_hash=sha256(original/'reference.json')
    subprocess.run([sys.executable,'scripts/viewer/prepare_reference_videos.py',
        '--rgb',str(args.data/'Lego_data/Lego.mkv'),
        '--events',str(args.data/'Lego_data/Lego_event_video.mp4'),
        '--output',str(new),'--white-background','--matte-directory',str(matte.parent),
        '--event-reference',str(original/'reference.json')],check=True)
    original.rename(directory/'references-border-original')
    new.rename(original)
    save_json(directory/'reference-upgrade.json',dict(previous_manifest_sha256=old_hash,
        current_manifest_sha256=sha256(original/'reference.json'),matte_sha256=sha256(matte),
        scope='UI display compositing only; original RGB and raw events remain unchanged'))


if __name__=='__main__':main()
