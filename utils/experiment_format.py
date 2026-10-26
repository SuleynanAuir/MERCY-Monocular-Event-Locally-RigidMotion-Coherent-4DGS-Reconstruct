"""Small, renderer-independent helpers shared by export and offline evaluation."""
import hashlib
import json
from pathlib import Path

FORMAT = 'scgs_experiment_v1'


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path, progress=lambda size: None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            progress(len(chunk))
    return digest.hexdigest()
