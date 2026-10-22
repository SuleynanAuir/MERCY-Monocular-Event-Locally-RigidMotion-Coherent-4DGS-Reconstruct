"""Keep the complete V3 rendering stack fixed and add only a local pose residual."""
import numpy as np
import torch

from scripts.events.refine_lego_events import MotionRenderer
from utils.event_refinement import correction_from_record
from utils.v3_auxiliary_mlp import make_graph


class FrozenV3Renderer(MotionRenderer):
    def __init__(self, baseline, record, width=640):
        if record.get('format') != 'event_supervised_hard_joint_alignment_v4':
            raise ValueError('Requires the existing Blender/V3 hard-joint motion')
        super().__init__(baseline, width)
        if self.static_surface is not None or self.surface_detail is not None:
            raise ValueError('Appended surface geometry needs an explicit auxiliary binding')
        self.correction = correction_from_record(record, 'cuda').requires_grad_(False)
        if self.correction.roots != [0]:
            raise ValueError('Expected V3 static root group 0')
        for key in ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest'):
            getattr(self.model, key).requires_grad_(False)
        self.base_scaling = self.model._scaling.detach()
        self.group_cache = {}

    def graph(self, nodes_per_group=32):
        labels = np.zeros(len(self.base_xyz), np.int64)
        labels[self.indices.cpu().numpy()] = self.correction.row_groups.cpu().numpy()
        canonical = self.base_xyz.cpu().numpy().copy()
        canonical[self.indices.cpu().numpy()] = self.rest[:, :3].cpu().numpy()
        if self.completion is not None:
            labels[self.completion.prefix_count:] = self.completion.groups.cpu().numpy()
            canonical[self.completion.prefix_count:] = self.completion.rest[:, :3].cpu().numpy()
        return make_graph(canonical, labels, self.correction.joint_pivots.cpu().numpy(),
                          self.correction.joint_groups.cpu().numpy(), nodes_per_group)

    @torch.no_grad()
    def base_pose(self, frame):
        pose = self.posed(frame, self.correction)
        xyz = self.base_xyz.index_copy(0, self.indices, pose[:, :3])
        rotation = self.base_rotation.index_copy(0, self.indices, pose[:, 3:])
        if self.completion is not None:
            suffix = self.completion.posed(frame, self.correction)
            prefix = self.completion.prefix_count
            xyz = torch.cat((xyz[:prefix], suffix[:, :3]), 0)
            rotation = torch.cat((rotation[:prefix], suffix[:, 3:]), 0)
        if frame not in self.group_cache:
            self.group_cache[frame] = self.correction.group_transforms(frame)[0].detach()
        return xyz, rotation, self.group_cache[frame]

    def rgb(self, frame, auxiliary=None):
        from gaussian_renderer import render
        from utils.render_sampling import resolve_rgb
        xyz, rotation, group_q = self.base_pose(frame)
        if auxiliary is not None:
            xyz, rotation = auxiliary.deform(xyz, rotation, group_q, frame)
        self.model._xyz, self.model._rotation = xyz, rotation
        self.model._scaling = (auxiliary.deform_scaling(self.base_scaling, frame)
                               if auxiliary is not None and hasattr(auxiliary, 'deform_scaling')
                               else self.base_scaling)
        return resolve_rgb(render(self.camera, self.model, self.pipe, self.background, 0., 0., 0.)['render'],
                           self.supersample)

    def gray(self, frame, auxiliary=None):
        return (self.rgb(frame, auxiliary) * self.luma).sum(0)
