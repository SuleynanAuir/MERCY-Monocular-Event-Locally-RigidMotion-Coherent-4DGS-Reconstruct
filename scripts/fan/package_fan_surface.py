"""Package refined V3 GS and topology-preserving Mesh, keeping V1 unchanged."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.fan.refine_fan_surface import sha
from utils.mesh_gaussian_binding import attachment_centers


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new output directory')
    config = json.loads(args.manifest.read_text()); root = args.manifest.parent
    for key in ('sources', 'mesh_paths', 'hybrid_paths'):
        config[key] = {side: str((root/path).resolve()) for side, path in config[key].items()}
    old, new = arrays(config['mesh_paths']['right']), arrays(args.mesh)
    for key in ('triangles', 'vertex_groups', 'motion_quaternions', 'motion_translations'):
        if not np.array_equal(old[key], new[key]): raise ValueError('Mesh changed topology or motion: '+key)
    if old['vertices'].shape != new['vertices'].shape: raise ValueError('Vertex identity changed')
    binding = arrays(config['hybrid_paths']['right'])
    args.output.mkdir(parents=True)
    old_center = attachment_centers(old['vertices'], old['triangles'], binding['face_indices'], binding['barycentric'], binding['offsets'])
    new_surface = attachment_centers(new['vertices'], new['triangles'], binding['face_indices'], binding['barycentric'], np.zeros_like(binding['offsets']))
    binding['offsets'] = (old_center-new_surface).astype(np.float32)
    reconstructed = attachment_centers(new['vertices'], new['triangles'], binding['face_indices'], binding['barycentric'], binding['offsets'])
    error = float(np.linalg.norm(reconstructed-old_center, axis=1).max())
    if error > 2e-6: raise ValueError('Updated Mesh binding changed canonical centers')
    mesh_hash = sha(args.mesh)
    binding_record = json.loads(str(binding['metadata_json']))
    binding_record.update(mesh_sha256=mesh_hash, offsets_preserved=False,
                          rebinding_center_error=error, appearance_preserved=True)
    binding['metadata_json'] = np.asarray(json.dumps(binding_record))
    binding_path = args.output/'right-binding.npz'
    np.savez_compressed(binding_path, **binding)
    surface = arrays(root/config['surface_refinement'])
    record = json.loads(str(surface['metadata_json']))
    record.update(mesh_sha256=mesh_hash, binding_sha256=sha(binding_path))
    record['report']['mesh_surface'] = json.loads(str(new['metadata_json']))['surface_denoising']
    record['report']['display_filter'] = 'Depth/alpha-guided 3x3 interior-only filtering; not geometry improvement'
    surface['metadata_json'] = np.asarray(json.dumps(record))
    np.savez_compressed(args.output/'surface.npz', **surface)
    config['mesh_paths']['right'] = str(args.mesh.resolve())
    config['hybrid_paths']['right'] = str(binding_path.resolve())
    config['surface_refinement'] = 'surface.npz'
    (args.output/'manifest.json').write_text(json.dumps(config, indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(record, indent=2)+'\n')
    print('Saved', args.output/'manifest.json', 'binding center error', error, flush=True)


if __name__ == '__main__': main()
