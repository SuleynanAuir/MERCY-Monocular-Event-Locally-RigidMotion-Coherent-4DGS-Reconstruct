"""Collect full-sequence measurements from the running, frozen UI catalog."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .protocol import sha256, save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--url',default='http://127.0.0.1:8894')
    p.add_argument('--scenes',nargs='+')
    p.add_argument('--mode',default='clay',choices=('clay','gaussian','pseudocolor'))
    args=p.parse_args()
    config=json.loads(args.catalog.read_text())
    records=[]
    for entry in config['scenes']:
        if args.scenes and entry['id'] not in args.scenes:continue
        destination=args.output/entry['id']/args.mode
        # Wait only for this application's genuine initialization; failures
        # are surfaced, not replaced by cached images from a different run.
        import urllib.request,time
        deadline=time.monotonic()+600
        while True:
            with urllib.request.urlopen(args.url+'/api/status?scene='+entry['id'],timeout=30) as response:
                status=json.load(response)
            if status.get('error'):raise RuntimeError(status)
            if status.get('ready'):break
            if time.monotonic()>deadline:raise TimeoutError('UI initialization')
            time.sleep(2)
        manifest=(args.catalog.parent/entry['path']).resolve()
        if status['scene'].get('asset_identity',{}).get('manifest_sha256') != sha256(manifest):
            raise ValueError('Running UI has not loaded the frozen experiment manifest')
        if (destination/'results.json').exists():
            previous=json.loads((destination/'results.json').read_text())
            if previous['status']['scene'].get('asset_identity') != status['scene']['asset_identity']:
                raise ValueError('Existing capture belongs to a different UI asset or renderer: '+entry['id'])
            if previous['scope'] != 'full_sequence':
                raise ValueError('Cannot reuse a partial capture')
            # A reference video can change without changing model assets.
            # Verify the actual HTTP bytes before reusing an earlier capture.
            for row in previous['frames']:
                with urllib.request.urlopen(args.url+f"/api/reference/{row['frame']}?scene="+entry['id'],timeout=30) as response:
                    digest=hashlib.sha256(response.read()).hexdigest()
                if digest != row['reference_sha256']:
                    raise ValueError('Existing capture uses an outdated UI reference: '+entry['id'])
            print('Verified frozen capture:',entry['id'],flush=True)
        else:
            subprocess.run([sys.executable,'-m','experiment.ui_capture','--url',args.url,
                '--scene',entry['id'],'--mode',args.mode,'--output',str(destination)],check=True)
        record=dict(scene=entry['id'],manifest_sha256=sha256(manifest),
            results_sha256=sha256(destination/'results.json'))
        save_json(destination/'capture-identity.json',record)
        records.append(record)
    save_json(args.output/('capture-index-'+args.mode+'.json'),dict(catalog_sha256=sha256(args.catalog),records=records))


if __name__=='__main__':main()
