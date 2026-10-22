"""Build rigid groups and conservative contact constraints from the motion prior."""
import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import cKDTree


def group_structure(parts_path, binding_path, baseline):
    with np.load(parts_path, allow_pickle=False) as data:
        matrices = data['object_matrices'].astype(np.float64)
    with np.load(binding_path, allow_pickle=False) as data:
        labels, indices = data['labels'], data['dynamic_indices']
    if not np.array_equal(indices, np.load(baseline / 'dynamic_indices.npy')):
        raise ValueError('Binding must exactly match the baseline dynamic rows')
    poses = np.load(baseline / 'poses.npy', mmap_mode='r')
    delta = matrices @ np.linalg.inv(matrices[0])
    sampled = np.linspace(0, len(poses)-1, 17).astype(int)
    features = delta[sampled, :, :3].transpose(1, 0, 2, 3).reshape(len(matrices[0]), -1)
    clusters = fclusterdata(features, .002, criterion='distance', method='complete')
    _, groups = np.unique(clusters[labels[indices]], return_inverse=True)
    count = int(groups.max())+1
    rest = poses[0, :, :3].copy()
    rows = [np.flatnonzero(groups == g) for g in range(count)]
    representatives = np.array([v[0] for v in rows])
    centers = np.array([rest[v].mean(0) for v in rows])
    trajectory = np.asarray(poses[:, representatives, :3])
    # Squared displacement is smooth at every return to rest; an absolute norm
    # would introduce a cusp for a rotating point crossing its initial pose.
    movement = np.square(trajectory-trajectory[:1]).sum(-1)
    envelopes = movement / np.maximum(movement.max(0, keepdims=True), 1e-8)
    envelopes[:, movement.max(0) < 1e-8] = 0
    envelopes[0] = 0
    # Keep only near contacts that stay near throughout the original motion.
    # These are prior-based constraints, not asserted ground-truth joints.
    sparse = [v[np.linspace(0, len(v)-1, min(1500, len(v))).astype(int)] for v in rows]
    contacts = []
    for g in range(count):
        for h in range(g+1, count):
            distance, neighbor = cKDTree(rest[sparse[h]]).query(rest[sparse[g]])
            selected = np.flatnonzero(distance < .008)
            if not len(selected):
                continue
            a, b = sparse[g][selected], sparse[h][neighbor[selected]]
            worst = np.zeros(len(a))
            for t in sampled:
                worst = np.maximum(worst, np.linalg.norm(poses[t, a, :3]-poses[t, b, :3], axis=-1))
            selected = np.flatnonzero(worst < .012)
            selected = selected[np.argsort(worst[selected])[:24]]
            contacts.extend(np.column_stack((a[selected], b[selected])).tolist())
    # Source points that hardly move serve as conservative attachment anchors.
    candidates = np.concatenate(sparse)
    displacement = np.zeros(len(candidates))
    for t in sampled:
        displacement = np.maximum(displacement, np.linalg.norm(poses[t, candidates, :3]-rest[candidates], axis=-1))
    anchors = candidates[displacement < .002]
    if len(anchors) > 64:
        anchors = anchors[np.linspace(0, len(anchors)-1, 64).astype(int)]
    result = {'groups': groups.tolist(), 'representatives': representatives.tolist(),
              'centers': centers.tolist(), 'envelopes': envelopes.tolist(),
              'contacts': contacts, 'anchors': anchors.tolist()}
    print('Motion groups', count, 'contacts', len(contacts), 'anchors', len(anchors), flush=True)
    return result
