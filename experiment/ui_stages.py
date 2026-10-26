"""Freeze observation/auxiliary stages on the actual UI canonical geometry."""
import argparse
import copy
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .protocol import save_json, sha256


def write_sequence_archive(path, **arrays):
    """Standard NPZ, streamed with fast deflate; no lossy pose compression."""
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                         allowZip64=True) as archive:
        for name, array in arrays.items():
            with archive.open(name+'.npy','w',force_zip64=True) as stream:
                np.lib.format.write_array(stream,np.asanyarray(array),allow_pickle=False)


def prepare_rigid(scene, root, source, sequence, destination, labels_path=None, reference_prior=None):
    from plyfile import PlyData
    from .prepare import save_prior
    from .supervision import prepare as prepare_supervision
    from scripts.lamp.prepare_lamp_viewer import rigid_deltas
    from scripts.motion.bind_lego_parts import transform
    from scripts.diagnostics.diagnose_lego_joints import diagnose
    if (destination/'prior.json').exists():
        raise FileExistsError(destination/'prior.json')
    destination.mkdir(parents=True, exist_ok=True)
    vertices = PlyData.read(source)['vertex'].data
    xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')])
    with np.load(sequence, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata_json']))
        if hashlib.sha256(vertices.tobytes()).hexdigest() != meta['base_vertices_sha256']:
            raise ValueError('Training source differs from the UI canonical geometry')
    binding = json.loads((root/'supplied-pair/binding-report.json').read_text())
    with np.load(root/'parts.npz', allow_pickle=False) as z:
        parts = {k: z[k] for k in z.files}
    reg = np.asarray(binding['registration'])
    delta = rigid_deltas(parts['object_matrices'], reg)
    if labels_path:
        with np.load(labels_path, allow_pickle=False) as z:
            labels = z['groups'].copy()
    else:
        surface = transform(parts['rest_points'], reg)
        labels = parts['point_parts'][cKDTree(surface).query(xyz, workers=4)[1]]
    if labels.shape != (len(xyz),) or labels.min() < 0 or labels.max() >= delta.shape[1]:
        raise ValueError('UI part ownership mismatch')
    centers = np.asarray([xyz[labels == g].mean(0) if np.any(labels == g) else np.zeros(3)
                          for g in range(delta.shape[1])])
    report = diagnose(parts, max_surface_samples=3000)
    representatives = {g['group']: g['representative_object'] for g in report['groups']}
    joints = []
    for edge in report['candidate_joints']:
        if edge['rank'] == 2 and edge.get('rest_axis') is not None:
            axis = reg[:3, :3]@np.asarray(edge['rest_axis'])
            axis /= np.linalg.norm(axis)
            joints.append(dict(groups=[representatives[g] for g in edge['groups']],
                pivot=transform(np.asarray(edge['rest_pivot']), reg).tolist(), axis=axis.tolist()))
    prior = save_prior(scene, destination, str(source), xyz, labels, delta, centers,
        meta['preview_camera'], joints, np.empty(0, np.int64),
        dict(registration=reg.tolist(), parts_sha256=sha256(root/'parts.npz'),
             ui_source_sequence_sha256=sha256(sequence), attachment='Exact UI rigid attachment'))
    prior['common_g0'] = 'Exact canonical geometry of this UI stage; versions may differ'
    prior['ui_source_sequence'] = str(sequence)
    if reference_prior:
        reference = json.loads(reference_prior.read_text())
        if prior['camera'] != reference['camera'] or prior['groups'] != reference['groups']:
            raise ValueError('V3 must preserve its V2 camera and motion-group identities')
        prior['diagonal'] = reference['diagonal']
        prior['joints'] = reference['joints']
        prior['frozen_v1_units_prior_sha256'] = sha256(reference_prior)
    save_json(destination/'prior.json', prior)
    prepare_supervision(scene, destination/'supervision')


def export_sequence(prepared, checkpoint, method, template, destination):
    import torch
    from plyfile import PlyData
    from .model import Adapter
    from .training import restore
    from utils.articulated_motion import quaternion_product
    from scipy.spatial.transform import Rotation
    from utils.gaussian_sequence import GaussianSequence
    if destination.exists():
        raise FileExistsError(destination)
    adapter = Adapter(prepared, method).cpu()
    auxiliary_trained = method == 'C3'
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu')
        expected = state['config']['inputs']['prior.json']
        if expected != sha256(prepared/'prior.json'):
            raise ValueError('Checkpoint belongs to a different UI asset')
        restore(adapter, state['weights'])
        auxiliary_trained = method == 'C3' and state['config'].get('auxiliary_trained', True)
    with np.load(template, allow_pickle=False) as z:
        metadata = copy.deepcopy(json.loads(str(z['metadata_json'])))
    vertices = PlyData.read(adapter.metadata['source'])['vertex'].data.copy()
    if hashlib.sha256(vertices.tobytes()).hexdigest() != metadata['base_vertices_sha256']:
        raise ValueError('Template and learned-stage canonical geometry differ')
    xyz = np.column_stack([vertices[k] for k in ('x','y','z')])
    rotation = np.column_stack([vertices[f'rot_{i}'] for i in range(4)])
    with torch.no_grad():
        q, t = adapter.transforms(torch.arange(len(adapter.q)))
        q, t = q.numpy(), t.numpy()
    labels = adapter.groups.numpy()
    with np.load(prepared/'motion.npz', allow_pickle=False) as z:
        bands = z['band_indices'].copy()
    moving = adapter.active.numpy()[labels]
    moving[bands] = True
    indices = np.flatnonzero(moving)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = destination.with_suffix('.poses.partial.npy')
    poses = np.lib.format.open_memmap(scratch, mode='w+', dtype='float32',
                                    shape=(len(indices), len(q), 7))
    band_poses = np.load(prepared/'bands.npy', mmap_mode='r') if len(bands) else None
    band_dest = np.searchsorted(indices, bands)
    ownership = {int(g):np.flatnonzero(labels[indices] == g) for g in np.unique(labels[indices])}
    for frame in range(len(q)):
        for group, rows in ownership.items():
            ids = indices[rows]
            matrix = Rotation.from_quat(q[frame, group][[1,2,3,0]]).as_matrix()
            poses[rows, frame, :3] = xyz[ids]@matrix.T+t[frame, group]
            poses[rows, frame, 3:] = quaternion_product(q[frame, group], rotation[ids])
        if band_poses is not None:
            pose = torch.tensor(np.asarray(band_poses[frame]))
            if method != 'C1':
                with torch.no_grad():
                    rest = torch.tensor(np.column_stack((xyz[bands], rotation[bands])))
                    pose = adapter.alignment(rest, pose, frame)
            poses[band_dest, frame] = pose.numpy()
        if frame == 0:
            poses[:, 0, :3], poses[:, 0, 3:] = xyz[indices], rotation[indices]
        metadata['frames'][frame]['pose_sha256'] = hashlib.sha256(poses[:, frame].tobytes()).hexdigest()
        metadata['frames'][frame]['file'] = f'frames/frame_{frame:04d}.ply'
        if frame % 50 == 0:
            print('Baking actual UI poses',frame,'/',len(q),flush=True)
    metadata.update(motion_label='源主运动＋冻结的观测校正'+('＋局部辅助 MLP' if auxiliary_trained else ''),
        description='从 data 重新生成的分阶段 UI 资产；输入来源与训练配置见 provenance。',
        default_background='light', motion_is_baked=True, event_validation=False)
    metadata.setdefault('provenance', {})['ui_stage'] = dict(method=method,
        auxiliary_trained=auxiliary_trained,
        prior_sha256=sha256(prepared/'prior.json'),
        checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
        template_sha256=sha256(template), canonical_source_sha256=sha256(adapter.metadata['source']))
    write_sequence_archive(destination.with_suffix('.npz.partial'), base_vertices=vertices, poses=poses,
            dynamic_indices=indices, times_seconds=np.arange(len(q))/metadata['fps'],
            metadata_json=np.asarray(json.dumps(metadata)))
    del poses
    destination.with_suffix('.npz.partial').rename(destination)
    with GaussianSequence(destination) as seq:
        for frame in range(len(seq)):
            seq.validate_pose_frame(frame)
    scratch.unlink()
    print('Validated UI stage:', destination, flush=True)


def package(root, name, surface_manifest):
    """One frozen material anchor keeps V1 byte-identical in both pairings."""
    config = json.loads(surface_manifest.read_text())
    original = (root/'supplied-pair/left.npz').resolve()
    for version in ('v2', 'v3'):
        right = (root/(version+'.npz')).resolve()
        if not original.is_file() or not right.is_file():
            raise FileNotFoundError(right)
        manifest = dict(format='gaussian_pair_v1', name=name+' · V1 / '+version.upper(),
            description='从 data 重建的冻结 UI 阶段；实验直接测量此渲染器输出。V3 额外包含结构／表面先验和展示柔光，效果需按实测评估。',
            left=dict(path=str(original), sha256=sha256(original), label='V1 · 原始高斯＋基础绑定'),
            right=dict(path=str(right), sha256=sha256(right), label=version.upper()+' · '+('观测校正' if version=='v2' else '结构／表面修复＋局部辅助变形'), render_supersample=1 if version=='v2' else 2),
            material_reference=dict(path=str(original), sha256=sha256(original)))
        if version == 'v3' and config.get('auxiliary_trained') is False:
            manifest['right']['label'] = 'V3 · 表面修复（原始事件不完整，未训练辅助 MLP）'
            manifest['description'] += ' 此场景原始事件不完整，辅助 MLP 未训练，保留 V2 校正。'
        if version == 'v3' and config.get('detail_finish'):
            entry = dict(config['detail_finish'])
            path = (surface_manifest.parent/entry['path']).resolve()
            if sha256(path) != entry['sha256']:
                raise ValueError('Source normal asset changed')
            with np.load(path, allow_pickle=False) as z:
                normal = z['local_normals'].copy()
                meta = json.loads(str(z['metadata_json']))
            with np.load(right, allow_pickle=False) as z:
                rightmeta = json.loads(str(z['metadata_json']))
                if rightmeta['base_vertices_sha256'] != meta['canonical_sha256']:
                    raise ValueError('Normal field cannot be transferred to different geometry')
            meta['source_normal_sha256'] = sha256(path)
            meta['sequence_sha256'] = sha256(right)
            normal_path = root/'v3-normals.npz'
            np.savez_compressed(normal_path, local_normals=normal, metadata_json=np.asarray(json.dumps(meta)))
            entry.update(path=str(normal_path.resolve()), sha256=sha256(normal_path))
            manifest['detail_finish'] = entry
        save_json(root/f'pair-v1-{version}.json', manifest)


def inherit_without_events(prepared, parent, destination):
    """Explicit Fan exception: inherit V2, do not fabricate event-trained MLP."""
    import torch
    from .model import Adapter
    supervision = json.loads((prepared/'supervision/metadata.json').read_text())
    if supervision['status'] == 'ready':
        raise ValueError('Use actual auxiliary training when complete events are available')
    if destination.exists(): raise FileExistsError(destination)
    checkpoint = torch.load(parent, map_location='cpu')
    if checkpoint['config']['method'] != 'C2': raise ValueError('Expected V2 parent')
    adapter = Adapter(prepared, 'C3').cpu()
    adapter.alignment.load_state_dict(checkpoint['weights']['alignment'])
    config = dict(method='C3', steps=0, auxiliary_trained=False,
        reason=supervision.get('reason', 'raw_events_unavailable'),
        parent_v2_sha256=sha256(parent),
        inputs={k:sha256(prepared/k) for k in ('prior.json','motion.npz','supervision/metadata.json')})
    destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(weights=dict(alignment=adapter.alignment.state_dict(),mlp=adapter.mlp.state_dict()),
                    config=config,best_step=0),destination)
    save_json(destination.with_suffix('.json'),config)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare-rigid')
    for name in ('inventory', 'root', 'source', 'sequence', 'output'):
        prep.add_argument('--'+name, required=True, type=Path)
    prep.add_argument('--scene', required=True)
    prep.add_argument('--labels', type=Path)
    prep.add_argument('--reference-prior', type=Path)
    train = sub.add_parser('train')
    for name in ('prepared', 'output'):
        train.add_argument('--'+name, required=True, type=Path)
    train.add_argument('--method', choices=('C2','C3'), required=True)
    train.add_argument('--initial-alignment', type=Path)
    train.add_argument('--freeze-alignment', action='store_true')
    train.add_argument('--event-only', action='store_true')
    train.add_argument('--steps', type=int, default=200)
    train.add_argument('--seed', type=int, default=17)
    train.add_argument('--variant', default='full', choices=('full','no_mlp','no_event','no_joint','no_jerk'))
    export = sub.add_parser('export')
    for name in ('prepared', 'template', 'output'):
        export.add_argument('--'+name, required=True, type=Path)
    export.add_argument('--method', choices=('C1','C2','C3'), required=True)
    export.add_argument('--checkpoint', type=Path)
    pkg = sub.add_parser('package')
    pkg.add_argument('--root', required=True, type=Path)
    pkg.add_argument('--name', required=True)
    pkg.add_argument('--surface-manifest', required=True, type=Path)
    inherited = sub.add_parser('inherit-without-events')
    for key in ('prepared','parent','output'):
        inherited.add_argument('--'+key,required=True,type=Path)
    args = p.parse_args()
    if args.action == 'prepare-rigid':
        scene = next(s for s in json.loads(args.inventory.read_text())['scenes'] if s['scene'] == args.scene)
        prepare_rigid(scene, args.root, args.source, args.sequence, args.output, args.labels, args.reference_prior)
    elif args.action == 'train':
        from .training import run
        run(args)
    elif args.action == 'export':
        export_sequence(args.prepared, args.checkpoint, args.method, args.template, args.output)
    elif args.action == 'package':
        package(args.root, args.name, args.surface_manifest)
    else:
        inherit_without_events(args.prepared,args.parent,args.output)


if __name__ == '__main__':
    main()
