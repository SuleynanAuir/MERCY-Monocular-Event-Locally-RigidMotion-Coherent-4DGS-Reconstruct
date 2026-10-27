"""Attach archived stages without copying models or changing historical labels.

The deployment JSON supplies catalog paths and each scene's middle sequence and
optional replacement pair. The output contains new catalogs and hashed stage
manifests; original catalogs and assets remain immutable.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.experiment_format import sha256, save_json


def prepare(config, output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for catalog_path in config['catalogs']:
        catalog = json.loads(Path(catalog_path).read_text())
        for scene in catalog['scenes']:
            stage = config['objects'].get(scene['id'])
            if not stage: continue
            if stage.get('pair'): scene['path'] = str(Path(stage['pair']).resolve())
            middle = Path(stage['middle']).resolve()
            if not middle.is_file(): raise FileNotFoundError(middle)
            if not Path(scene['path']).is_file(): raise FileNotFoundError(scene['path'])
            manifest = output/(scene['id']+'-middle.json')
            save_json(manifest, dict(format='gaussian_stage_v1', path=str(middle), sha256=sha256(middle),
                label='V2 · '+stage['label'], branch='archived-independent-stage',
                parents=dict(sequence=str(middle), pair_manifest_sha256=sha256(scene['path'])),
                source_to_display={v:v for v in ('V1','V2','V3')}, description=stage['description']))
            scene['middle_path'] = str(manifest)
            if stage.get('evaluation_path'): scene['evaluation_path'] = stage['evaluation_path']
        save_json(output/(Path(catalog_path).parent.name+'-scenes.json'), catalog)
    save_json(output/'deployment.json', config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.output)
