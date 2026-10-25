"""Package a declared V3 normal/material finish without changing either sequence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from utils.fan_surface import attributes
from utils.gaussian_pair_renderer import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'parts', 'binding-report', 'blend', 'blender', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.manifest.read_text())
    if config.get('format') != 'gaussian_pair_v1':
        raise ValueError('Expected a Gaussian pair manifest')
    for key in ('left', 'right'):
        path = (args.manifest.parent/config[key]['path']).resolve()
        if file_sha(path) != config[key]['sha256']:
            raise ValueError('Sequence checksum differs')
        config[key]['path'] = str(path)
    with np.load(config['right']['path'], allow_pickle=False) as archive:
        base = archive['base_vertices']
        meta = json.loads(str(archive['metadata_json']))
    source_report = Path(config['surface_refinement']['source_report'])
    report = json.loads(source_report.read_text())
    groups_path = source_report.parent/'binding.npz'
    if (file_sha(groups_path) != report['binding_sha256']
            or meta['provenance']['source_ply_sha256'] != report['surface_sha256']
            or report['parts_sha256'] != file_sha(args.parts)
            or report['original_binding_sha256'] != file_sha(args.binding_report)):
        raise ValueError('Surface and motion binding identities differ')
    with np.load(groups_path, allow_pickle=False) as archive:
        groups = archive['groups']
    xyz, basis, scales, _, _ = attributes(base)
    args.output.mkdir(parents=True)
    input_path, projection_path = args.output/'input.npz', args.output/'projection.npz'
    np.savez(input_path, xyz=xyz.astype(np.float32), groups=groups)
    subprocess.run([str(args.blender), '-b', '--disable-autoexec', str(args.blend), '--python',
        str(Path(__file__).with_name('export_lamp_detail_normals.py')), '--', '--input', str(input_path),
        '--parts', str(args.parts), '--binding-report', str(args.binding_report),
        '--output', str(projection_path)], check=True)
    with np.load(projection_path, allow_pickle=False) as archive:
        normal, distance = archive['normals'], archive['distance']
        provenance = json.loads(str(archive['metadata_json']))
    fallback = basis[np.arange(len(base)), :, scales.argmin(1)]
    alignment = (normal*fallback).sum(1)
    normal = np.where((alignment < 0)[:, None], -normal, normal)
    weight = np.exp(-.5*(distance/.030)**2)*np.clip((np.abs(alignment)-.3)/.4, 0, 1)
    normal = normal*weight[:, None]+fallback*(1-weight[:, None])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    local = np.einsum('nij,ni->nj', basis, normal).astype(np.float32)
    record = dict(format='lamp_detail_normals_v1', sequence_sha256=config['right']['sha256'],
        canonical_sha256=meta['base_vertices_sha256'], normal_count=len(local),
        projection=provenance, distance_kernel=.030, supported_fraction=float((weight>.5).mean()),
        method='Source normal transfer within rigid parts, confidence-weighted against Gaussian normals',
        lighting='Camera-relative detail lighting and screen-space contact shading',
        limitation='Display enhancement only; geometry and motion unchanged; no training or accuracy claim')
    normal_path = args.output/'normals.npz'
    np.savez_compressed(normal_path, local_normals=local, metadata_json=np.asarray(json.dumps(record)))
    config['detail_finish'] = dict(style='normal_detail_v1', path=normal_path.name, sha256=file_sha(normal_path))
    with np.load(args.parts, allow_pickle=False) as archive:
        objects = json.loads(str(archive['metadata_json']))['objects']
    registration = np.array(json.loads(args.binding_report.read_text())['registration'])
    up = registration[:3, 2]/np.linalg.norm(registration[:3, 2])
    height = xyz @ up
    start = float(np.quantile(height[groups == objects.index('Circle')], .95))
    end = float(np.quantile(height[groups == objects.index('Circle.002')], .25))
    if end <= start:
        raise ValueError('Base and lower-joint height ranges overlap unexpectedly')
    config['detail_finish']['preserve_base'] = dict(axis=up.tolist(), start=start, end=end)
    config['right']['label'] = 'V3 · 法线细节与材质增强'
    config['description'] = 'V1 保留原始呈现；V3 在精细 GS 上增加部件法线、局部明暗、接触阴影与柔和高光。属于展示增强，几何和 Blender 运动未再修改，不能作为重建精度提升的证据。'
    if config.get('material_reference'):
        config['material_reference']['path'] = str((args.manifest.parent/config['material_reference']['path']).resolve())
    (args.output/'manifest.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
