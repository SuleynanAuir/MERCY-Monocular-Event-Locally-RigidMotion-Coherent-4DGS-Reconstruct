"""Triangle-attached Gaussian appearance driven by each source Mesh's motion."""
import hashlib
import json
from pathlib import Path
import numpy as np


def pose_fingerprint(xyz, rotation):
    digest = hashlib.sha256()
    for value in (xyz, rotation):
        digest.update(np.ascontiguousarray(value, dtype=np.float32).tobytes())
    return digest.hexdigest()


def attachment_centers(vertices, triangles, face_indices, barycentric, offsets):
    """Reconstruct canonical centers, retaining the surface-to-volume offset."""
    faces = np.asarray(triangles)[face_indices]
    return (np.asarray(vertices)[faces].astype(np.float64) * barycentric[..., None]).sum(1) + offsets


def validate_attachment(data, vertices, triangles, vertex_groups, xyz, rotation, metadata, mesh_sha256):
    meta = json.loads(str(data['metadata_json']))
    rows, face_ids, bary, offsets, groups = [data[k] for k in
        ('rows', 'face_indices', 'barycentric', 'offsets', 'groups')]
    if (meta.get('format') != 'mesh_gaussian_binding_v1' or meta.get('mesh_sha256') != mesh_sha256
            or meta.get('source_fingerprints') != metadata['source_fingerprints']
            or meta.get('source_side') != metadata['source_side']
            or meta.get('canonical_pose_sha256') != pose_fingerprint(xyz, rotation)
            or meta.get('gaussian_count') != len(xyz)):
        raise ValueError('Hybrid binding does not match this Mesh and Gaussian source')
    n = len(rows)
    if (rows.shape != (n,) or rows.dtype.kind not in 'iu' or not n
            or np.any(rows < 0) or np.any(rows >= len(xyz)) or np.any(np.diff(rows.astype(np.int64)) <= 0)
            or face_ids.shape != (n,) or face_ids.dtype.kind not in 'iu'
            or np.any(face_ids < 0) or np.any(face_ids >= len(triangles))
            or bary.shape != (n, 3) or offsets.shape != (n, 3) or groups.shape != (n,)
            or groups.dtype.kind not in 'iu' or not np.isfinite(bary).all() or not np.isfinite(offsets).all()
            or np.any(bary < -1e-6) or np.any(bary > 1 + 1e-6)
            or not np.allclose(bary.sum(1), 1, atol=1e-6, rtol=0)):
        raise ValueError('Invalid triangle attachment arrays')
    if np.any(vertex_groups[triangles[face_ids]] != groups[:, None]):
        raise ValueError('Triangle attachment crosses independent motion groups')
    centers = attachment_centers(vertices, triangles, face_ids, bary, offsets).astype(np.float32)
    if np.max(np.linalg.norm(centers - xyz[rows], axis=1)) > 2e-6:
        raise ValueError('Triangle attachment changes canonical Gaussian geometry')
    return centers, meta


class MeshGaussianBinding:
    def __init__(self, path, mesh_path, mesh_renderer, xyz, rotation):
        import torch
        from utils.rigid_checkpoint import sha256_file
        if mesh_renderer.motion is None:
            raise ValueError('Hybrid mode requires independent embedded Mesh motion')
        with np.load(path, allow_pickle=False) as z:
            data = {key: z[key] for key in z.files}
        with np.load(mesh_path, allow_pickle=False) as z:
            centers, meta = validate_attachment(data, z['vertices'], z['triangles'], z['vertex_groups'],
                xyz.cpu().numpy(), rotation.cpu().numpy(), json.loads(str(z['metadata_json'])), sha256_file(mesh_path))
        self.rows = torch.from_numpy(data['rows'].astype(np.int64)).cuda()
        self.groups = torch.from_numpy(data['groups'].astype(np.int64)).cuda()
        # Rigid group motion commutes with the barycentric reconstruction, so
        # canonical centers can be cached without rasterizing the Mesh each frame.
        self.centers = torch.from_numpy(centers).cuda()
        self.rotation = rotation[self.rows].clone()
        self.motion = mesh_renderer.motion
        self.info = dict(bound_gaussians=len(self.rows), unbound_gaussians=len(xyz)-len(self.rows),
                         source_side=meta['source_side'], mesh_sha256=meta['mesh_sha256'],
                         representation='triangle-attached Gaussians with preserved canonical offsets')

    def apply(self, model, frame):
        from utils.event_refinement import qmul, qrotate
        q, t = (value[frame, self.groups] for value in self.motion)
        model._xyz[self.rows] = qrotate(q, self.centers) + t
        model._rotation[self.rows] = qmul(q, self.rotation)
