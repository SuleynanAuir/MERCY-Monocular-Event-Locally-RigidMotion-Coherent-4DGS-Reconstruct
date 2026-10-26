import io
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from utils.comparison_renderer import (ComparisonRenderer, validate_comparison_sources,
                                       validate_repaired_gaussians, verify_completion_prefix,
                                       audited_source_fingerprints, validate_rebuilt_gaussians,
                                       verify_rebuild_provenance, vertex_rows_sha256)
from utils.viewer_state import validate_request
from utils.render_sampling import render_sampling_contract


class ComparisonRendererTests(unittest.TestCase):
    def test_native_clay_needs_no_mesh_and_uses_both_source_poses_and_auxiliary_motion(self):
        renderer = ComparisonRenderer('unused.json', 'cpu')
        renderer.base_pose = np.eye(4); renderer.base_pose[2, 3] = 2; renderer.target = np.zeros(3)
        renderer.model = SimpleNamespace(value=.2)
        renderer.clay_coordinates = [torch.full((1, 3), .2), torch.full((1, 3), .8)]
        renderer.pipe = None
        renderer.backgrounds = {'black': torch.zeros(3), 'light': torch.ones(3)}
        poses, backgrounds = [], []
        def pose(side, frame, auxiliary=True):
            poses.append((side, frame, auxiliary)); renderer.model.value = .2+.6*side
        renderer.apply_side_pose = pose
        def render(camera, model, pipe, background, *args, override_color=None):
            backgrounds.append(background.clone())
            alpha = torch.ones(1, 128, 128); alpha[:, 55:65, 55:65] = 0
            return {'render': torch.full((3, 128, 128), model.value)*alpha,
                    'depth': 2*alpha, 'alpha': alpha}
        with patch.dict(sys.modules, {'gaussian_renderer': SimpleNamespace(render=render),
                                     'scripts.rendering.render_ply': SimpleNamespace(make_camera=lambda *args: None)}):
            req = validate_request(dict(render_mode='clay', frame=1, width=128, height=128, background='light'), 3)
            packet = renderer.render(req); cached = renderer.render(req)
        self.assertEqual(poses, [(0, 1, True), (1, 1, True)])
        self.assertTrue(all(torch.equal(b, torch.zeros(3)) for b in backgrounds))
        size = struct.unpack('<I', packet[:4])[0]
        pixels = np.asarray(Image.open(io.BytesIO(packet[4+size:])))
        self.assertGreater(float(pixels[:, 128:].mean()), float(pixels[:, :128].mean())+10)
        self.assertGreater(pixels[60, 60].min(), 250)
        size = struct.unpack('<I', cached[:4])[0]
        self.assertTrue(json.loads(cached[4:4+size])['cache_hit'])

    def test_hybrid_applies_mesh_bindings_and_has_its_own_sampling_and_cache(self):
        renderer = ComparisonRenderer('unused.json', 'cpu')
        renderer.base_pose = np.eye(4); renderer.base_pose[2, 3] = 2; renderer.target = np.zeros(3)
        renderer.model = SimpleNamespace(value=0.)
        renderer.pipe = None; renderer.backgrounds = {'dark': torch.zeros(3)}
        renderer.apply_side_pose = lambda side, frame, auxiliary=True: setattr(renderer.model, 'value', .2 + .1 * side)
        calls = []
        def bind(model, frame):
            calls.append(frame); model.value += .1
        renderer.hybrid = [SimpleNamespace(apply=bind), SimpleNamespace(apply=bind)]
        sizes = []
        def camera(pose, width, height, fovy):
            sizes.append((width, height)); return (width, height)
        def pixels(camera, model, *args):
            return {'render': torch.full((3, camera[1], camera[0]), model.value),
                    'depth': torch.full((1, camera[1], camera[0]), 2.), 'alpha': torch.ones((1, camera[1], camera[0]))}
        with patch.dict(sys.modules, {'gaussian_renderer': SimpleNamespace(render=pixels),
                                     'scripts.rendering.render_ply': SimpleNamespace(make_camera=camera)}):
            request=validate_request({'frame': 1, 'width': 128, 'height': 128}, 3)
            original=renderer.render(request)
            hybrid=renderer.render(dict(request,render_mode='hybrid'))
            balanced=renderer.render(dict(request,render_mode='hybrid',hybrid_quality='balanced'))
            repeated=renderer.render(dict(request,render_mode='hybrid'))
            again=renderer.render(request)
        self.assertEqual(calls,[1,1,1,1])
        self.assertEqual(sizes,[(128,128),(128,128),(128,128),(128,128),(256,256),(256,256)])
        def unpack(packet):
            size=struct.unpack('<I',packet[:4])[0]
            return json.loads(packet[4:4+size]),packet[4+size:]
        self.assertEqual(unpack(hybrid)[0]['render_mode'],'hybrid')
        self.assertEqual(unpack(hybrid)[0]['hybrid_quality'],'surface')
        self.assertEqual(unpack(balanced)[0]['hybrid_quality'],'balanced')
        self.assertTrue(unpack(repeated)[0]['cache_hit'])
        self.assertNotEqual(unpack(original)[1],unpack(hybrid)[1])
        self.assertEqual(unpack(original)[1],unpack(again)[1])

    def test_mesh_mode_renders_two_independent_meshes_without_gaussian_pixels(self):
        renderer = ComparisonRenderer('unused.json', 'cpu')
        renderer.base_pose = np.eye(4); renderer.base_pose[2, 3] = 2; renderer.target = np.zeros(3)
        calls = []
        def panels(request):
            calls.append(request['frame'])
            return [np.full((128, 128, 3), 40 + request['frame'], np.uint8),
                    np.full((128, 128, 3), 180 + request['frame'], np.uint8)]
        renderer.mesh = SimpleNamespace(render_panels=panels)
        def unexpected(*args):
            raise AssertionError('Mesh comparison must not render GS in either panel')
        with patch.dict(sys.modules, {'gaussian_renderer': SimpleNamespace(render=unexpected),
                                      'scripts.rendering.render_ply': SimpleNamespace(make_camera=unexpected)}):
            request = validate_request({'render_mode': 'mesh', 'frame': 1, 'width': 128, 'height': 128}, 3)
            for _ in range(2):
                packet = renderer.render(request)
                size = struct.unpack('<I', packet[:4])[0]
                meta = json.loads(packet[4:4+size])
                image = np.asarray(Image.open(io.BytesIO(packet[4+size:])))
                self.assertTrue(meta['comparison'])
                self.assertEqual(meta['left_frame'], meta['right_frame'])
                self.assertAlmostEqual(image[:, :128].mean(), 41, delta=1)
                self.assertAlmostEqual(image[:, 128:].mean(), 181, delta=1)
        self.assertEqual(calls, [1])

    def test_scene_identity_exposes_only_exact_audited_correction_and_source_hashes(self):
        audit = {'artifact_sha256': {'correction': 'c' * 64, 'source_ply': 'a' * 64,
                                     'supervision': 'private-unneeded-field'}, 'internal_note': 'not public'}
        self.assertEqual(audited_source_fingerprints(audit),
                         {'candidate_correction_sha256': 'c' * 64, 'right_source_sha256': 'a' * 64})

    @staticmethod
    def gaussian_model(count):
        return SimpleNamespace(_xyz=torch.zeros((count, 3)), _scaling=torch.full((count, 3), -3.),
                               _rotation=torch.tensor([[1., 0., 0., 0.]] * count),
                               _features_dc=torch.ones((count, 1, 3)), _features_rest=torch.zeros((count, 15, 3)),
                               _opacity=torch.zeros((count, 1)))

    def test_panels_are_synced_and_different_motion_masks_do_not_leak(self):
        renderer = ComparisonRenderer('unused.json', 'cpu')
        renderer.base_pose = np.eye(4); renderer.base_pose[2, 3] = 2
        renderer.target = np.zeros(3)
        renderer.base_xyz = torch.zeros((3, 3))
        renderer.base_rotation = torch.tensor([[1., 0., 0., 0.]] * 3)
        renderer.base_scaling = torch.zeros((3, 3))
        renderer.side_geometry = [(renderer.base_xyz, renderer.base_rotation, renderer.base_scaling)] * 2
        renderer.model = SimpleNamespace(_xyz=renderer.base_xyz.clone(), _rotation=renderer.base_rotation.clone(),
                                         _scaling=renderer.base_scaling.clone())
        renderer.indices = [torch.tensor([0]), torch.tensor([1])]
        renderer.backgrounds, renderer.pipe = {'dark': torch.zeros(3)}, None
        renderer.posed = lambda side, index: torch.tensor([[float(side + 1), 0., 0., 1., 0., 0., 0.]])
        observed = []

        def render(camera, model, *args):
            observed.append(model._xyz.clone())
            return {'render': torch.full((3, 128, 128), float(model._xyz.sum()) / 4)}

        with patch.dict(sys.modules, {'gaussian_renderer': SimpleNamespace(render=render),
                                      'scripts.rendering.render_ply': SimpleNamespace(make_camera=lambda *args: None)}):
            for index in (1, 0):
                packet = renderer.render(validate_request({'id': 5, 'frame': index, 'width': 128, 'height': 128}, 2))
                size = struct.unpack('<I', packet[:4])[0]
                meta = json.loads(packet[4:4+size])
                self.assertTrue(meta['comparison'])
                self.assertEqual(meta['left_frame'], meta['right_frame'])
                self.assertEqual(meta['panel_width'], 128)
                image = np.asarray(Image.open(io.BytesIO(packet[4+size:])))
                self.assertEqual(image.shape, (128, 256, 3))
                self.assertGreater(image[:, 128:].mean(), image[:, :128].mean())
            # A repeated render reuses pixels but never the old ID or stream seq.
            request = validate_request({'id': 99, 'frame': 1, 'width': 128, 'height': 128}, 2)
            request['_stream'] = {'seq': 12, 'rate': 30}
            packet = renderer.render(request)
            size = struct.unpack('<I', packet[:4])[0]
            meta = json.loads(packet[4:4+size])
            self.assertTrue(meta['cache_hit'])
            self.assertEqual(meta['id'], 99)
            self.assertEqual(meta['frame'], 1)
            self.assertEqual(meta['stream']['seq'], 12)
            self.assertEqual(len(observed), 4)
        expected_left = torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        expected_right = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 0., 0.]])
        for actual, expected in zip(observed, [expected_left, expected_right] * 2):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(renderer.base_xyz.sum().item(), 0)

    def test_cache_memory_is_bounded_and_cpu_mode_disables_gpu_cache(self):
        self.assertEqual(ComparisonRenderer('unused', 'cpu').cache_limit, 0)
        self.assertLessEqual(ComparisonRenderer('unused', 'auto').cache_limit, 48)

    def test_repaired_rest_positions_scales_and_motion_masks_do_not_leak(self):
        renderer = ComparisonRenderer('unused', 'cpu')
        rotation = torch.tensor([[1., 0., 0., 0.]] * 3)
        left_xyz, right_xyz = torch.zeros((3, 3)), torch.full((3, 3), 5.)
        left_scale, right_scale = torch.full((3, 3), -3.), torch.full((3, 3), -2.)
        renderer.side_geometry = [(left_xyz, rotation, left_scale), (right_xyz, rotation, right_scale)]
        renderer.model = SimpleNamespace(_xyz=left_xyz.clone(), _rotation=rotation.clone(), _scaling=left_scale.clone())
        renderer.indices = [torch.tensor([0]), torch.tensor([1])]
        renderer.posed = lambda side, index: torch.tensor([[float(10 + side), 0., 0., 0., 1., 0., 0.]])
        for side in (0, 1, 0, 1, 0):
            renderer.apply_side_pose(side, 1)
            expected_xyz = renderer.side_geometry[side][0].clone()
            expected_rotation = rotation.clone()
            expected_xyz[renderer.indices[side]] = torch.tensor([10. + side, 0., 0.])
            expected_rotation[renderer.indices[side]] = torch.tensor([0., 1., 0., 0.])
            torch.testing.assert_close(renderer.model._xyz, expected_xyz)
            torch.testing.assert_close(renderer.model._rotation, expected_rotation)
            torch.testing.assert_close(renderer.model._scaling, renderer.side_geometry[side][2])
        torch.testing.assert_close(left_xyz, torch.zeros((3, 3)))
        torch.testing.assert_close(right_xyz, torch.full((3, 3), 5.))

    def test_comparison_sources_require_matching_repair_provenance(self):
        meta = {'frame_count': 4, 'fps': 60, 'camera': {'fovy': 40}, 'gaussian_count': 3}
        audit = {'artifact_sha256': {'source_ply': 'right-hash'}}
        report = {'format': 'lego_part_depth_repair_v1', 'source_sha256': 'left-hash',
                  'repaired_sha256': 'right-hash', 'gaussian_count': 3}
        self.assertTrue(validate_comparison_sources(meta, meta, audit, 'left-hash', report))
        sampling = render_sampling_contract({}, supersample=2)
        self.assertTrue(validate_comparison_sources(meta, {**meta, **sampling}, audit,
                                                    'left-hash', {**report, **sampling}))
        with self.assertRaisesRegex(ValueError, 'sampling contract mismatch'):
            validate_comparison_sources(meta, {**meta, **sampling}, audit, 'left-hash', report)
        with self.assertRaisesRegex(ValueError, 'sampling contract mismatch'):
            validate_comparison_sources(meta, {**meta, **sampling}, audit, 'left-hash',
                                        {**report, **sampling, 'appearance_refinement': {'render_supersample': 1}})
        self.assertFalse(validate_comparison_sources(meta, meta, audit, 'right-hash'))
        with self.assertRaisesRegex(ValueError, 'geometry_repair_report'):
            validate_comparison_sources(meta, meta, audit, 'left-hash')
        for key, wrong in [('format', 'untrusted'), ('source_sha256', 'wrong'),
                           ('repaired_sha256', 'wrong'), ('gaussian_count', 2)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'repair report'):
                validate_comparison_sources(meta, meta, audit, 'left-hash', {**report, key: wrong})
        for key, wrong in [('gaussian_count', 2), ('frame_count', 5), ('fps', 30), ('camera', {})]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identical'):
                validate_comparison_sources(meta, {**meta, key: wrong}, audit, 'left-hash', report)

    def test_repair_only_permits_xyz_and_scale_changes_with_exact_appearance(self):
        def model():
            return SimpleNamespace(_xyz=torch.zeros((3, 3)), _scaling=torch.zeros((3, 3)),
                                   _rotation=torch.tensor([[1., 0., 0., 0.]] * 3),
                                   _features_dc=torch.ones((3, 1, 3)), _features_rest=torch.zeros((3, 15, 3)),
                                   _opacity=torch.tensor([[float('inf')], [0.], [float('-inf')]]))
        left, right = model(), model()
        right._xyz += 2
        right._scaling -= 1
        validate_repaired_gaussians(left, right, 3)
        for attribute in ('_rotation', '_features_dc', '_features_rest', '_opacity'):
            changed = model()
            getattr(changed, attribute).reshape(-1)[1] += .01
            with self.subTest(attribute=attribute), self.assertRaisesRegex(ValueError, 'protected'):
                validate_repaired_gaussians(left, changed, 3)
        with self.assertRaisesRegex(ValueError, 'count'):
            validate_repaired_gaussians(left, right, 4)
        right._xyz[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            validate_repaired_gaussians(left, right, 3)

    def test_pose_cache_entries_are_keyed_by_side_and_frame(self):
        renderer = ComparisonRenderer('unused', 'auto')
        left = torch.tensor([[1., 0., 0., 1., 0., 0., 0.]])
        right = torch.tensor([[2., 0., 0., 1., 0., 0., 0.]])
        renderer.pose_cache[(0, 1)] = left
        renderer.pose_cache[(1, 1)] = right
        self.assertIs(renderer.posed(0, 1), left)
        self.assertIs(renderer.posed(1, 1), right)

    def test_completion_requires_exact_report_link_and_bounded_append(self):
        left = {'frame_count': 4, 'fps': 60, 'camera': {'fovy': 40}, 'gaussian_count': 3}
        right = {**left, 'gaussian_count': 5}
        audit = {'artifact_sha256': {'source_ply': 'completed-hash'}}
        report = {'format': 'lego_part_completion_v1', 'source_sha256': 'original-hash',
                  'completed_sha256': 'completed-hash', 'original_gaussian_count': 3,
                  'gaussian_count': 5, 'added_gaussians': 2, 'base_repaired_source': '/tmp/base.ply',
                  'base_repaired_sha256': 'base-hash', 'frozen_prefix_vertex_sha256': 'prefix-hash'}
        self.assertTrue(validate_comparison_sources(left, right, audit, 'original-hash', report))
        for key, wrong in [('source_sha256', 'wrong'), ('completed_sha256', 'wrong'),
                           ('original_gaussian_count', 4), ('gaussian_count', 6), ('added_gaussians', 1),
                           ('base_repaired_source', ''), ('base_repaired_sha256', ''),
                           ('frozen_prefix_vertex_sha256', '')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'completion report'):
                validate_comparison_sources(left, right, audit, 'original-hash', {**report, key: wrong})
        for added in (0, -1, 100001):
            candidate = {**right, 'gaussian_count': 3 + added}
            altered = {**report, 'gaussian_count': 3 + added, 'added_gaussians': added}
            with self.subTest(added=added), self.assertRaisesRegex(ValueError, 'completion report'):
                validate_comparison_sources(left, candidate, audit, 'original-hash', altered)
        for key, value in [('frame_count', 5), ('fps', 30), ('camera', {})]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'camera and timestamps'):
                validate_comparison_sources(left, {**right, key: value}, audit, 'original-hash', report)
        with self.assertRaisesRegex(ValueError, 'geometry_repair_report'):
            validate_comparison_sources(left, right, audit, 'original-hash')

    def test_completion_preserves_prefix_appearance_and_validates_appended_fields(self):
        left, right = self.gaussian_model(3), self.gaussian_model(5)
        left._opacity[0] = float('inf'); left._opacity[2] = float('-inf')
        right._opacity[:3] = left._opacity
        right._xyz += 1.; right._scaling += .1
        validate_repaired_gaussians(left, right, 3, 5)
        for attribute in ('_features_dc', '_features_rest', '_opacity', '_rotation'):
            altered = self.gaussian_model(5); altered._opacity[:3] = left._opacity
            getattr(altered, attribute).reshape(-1)[1] += .25
            with self.subTest(prefix_attribute=attribute), self.assertRaisesRegex(ValueError, 'protected'):
                validate_repaired_gaussians(left, altered, 3, 5)
            altered = self.gaussian_model(5); altered._opacity[:3] = left._opacity
            getattr(altered, attribute)[-1].reshape(-1)[0] = float('inf')
            with self.subTest(appended_attribute=attribute), self.assertRaisesRegex(ValueError, 'nonfinite appended'):
                validate_repaired_gaussians(left, altered, 3, 5)
        altered = self.gaussian_model(5); altered._opacity[:3] = left._opacity
        altered._features_rest[-1, 0, 0] = 1.
        with self.assertRaisesRegex(ValueError, 'DC-only'):
            validate_repaired_gaussians(left, altered, 3, 5)
        altered._features_rest.zero_(); altered._rotation[-1].zero_()
        with self.assertRaisesRegex(ValueError, 'invalid appended rotations'):
            validate_repaired_gaussians(left, altered, 3, 5)
        altered = self.gaussian_model(5); altered._opacity[:3] = left._opacity
        altered._opacity = torch.cat([altered._opacity, torch.zeros((1, 1))])
        with self.assertRaisesRegex(ValueError, 'attribute shape'):
            validate_repaired_gaussians(left, altered, 3, 5)
        for count in (0, 3, 100004):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'append between'):
                validate_repaired_gaussians(left, right, 3, count)

    def test_completion_prefix_checks_base_file_hash_vertex_bytes_and_schema(self):
        from plyfile import PlyData, PlyElement
        from utils.rigid_checkpoint import sha256_file
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / 'base.ply'; completed = Path(folder) / 'completed.ply'
            vertices = np.array([(1., float('inf')), (2., 0.)], dtype=[('x', '<f4'), ('opacity', '<f4')])
            tail = np.array([(3., .2)], dtype=vertices.dtype)
            def write(path, data):
                PlyData([PlyElement.describe(data, 'vertex')], text=False).write(path)
            write(base, vertices); write(completed, np.concatenate([vertices, tail]))
            report = {'base_repaired_source': str(base), 'base_repaired_sha256': sha256_file(base),
                      'original_gaussian_count': 2, 'gaussian_count': 3,
                      'frozen_prefix_vertex_sha256': hashlib.sha256(vertices.tobytes()).hexdigest()}
            verify_completion_prefix(report, completed)
            with self.assertRaisesRegex(ValueError, 'base source checksum'):
                verify_completion_prefix({**report, 'base_repaired_sha256': 'wrong'}, completed)
            with self.assertRaisesRegex(ValueError, 'frozen repaired-source prefix'):
                verify_completion_prefix({**report, 'frozen_prefix_vertex_sha256': 'wrong'}, completed)
            changed = np.concatenate([vertices, tail]); changed['x'][0] += .1
            write(completed, changed)
            with self.assertRaisesRegex(ValueError, 'frozen repaired-source prefix'):
                verify_completion_prefix(report, completed)
            write(completed, np.zeros(3, dtype=[('different_schema', '<f4')]))
            with self.assertRaisesRegex(ValueError, 'vertex schema'):
                verify_completion_prefix(report, completed)

    def test_different_counts_switch_geometry_appearance_and_reuse_jpeg_without_alias(self):
        renderer = ComparisonRenderer('unused', 'cpu')
        renderer.side_sampling[1] = render_sampling_contract({}, supersample=2)
        renderer.base_pose = np.eye(4); renderer.base_pose[2, 3] = 2
        renderer.target = np.zeros(3)
        left, right = self.gaussian_model(2), self.gaussian_model(4)
        right._xyz.fill_(5); right._scaling.fill_(-2); right._features_dc.fill_(.25); right._opacity.fill_(.4)
        renderer.side_geometry = [(m._xyz.clone(), m._rotation.clone(), m._scaling.clone()) for m in (left, right)]
        renderer.side_appearance = [(m._features_dc.clone(), m._opacity.clone(), torch.empty((len(m._xyz), 0, 3))) for m in (left, right)]
        renderer.model = self.gaussian_model(2)
        renderer.indices = [torch.tensor([0]), torch.tensor([2, 3])]
        renderer.posed = lambda side, index: torch.tensor([[10. + side, float(index), 0., 0., 1., 0., 0.]] * len(renderer.indices[side]))
        renderer.backgrounds, renderer.pipe = {'dark': torch.zeros(3)}, None
        calls = []
        def draw(camera, model, *args):
            count = len(model._xyz); side = int(count == 4)
            self.assertEqual(camera, (128 * (side + 1), 128 * (side + 1)))
            expected = renderer.side_geometry[side][0].clone()
            expected[renderer.indices[side]] = torch.tensor([10. + side, 1., 0.])
            torch.testing.assert_close(model._xyz, expected)
            torch.testing.assert_close(model._scaling, renderer.side_geometry[side][2])
            torch.testing.assert_close(model._features_dc, renderer.side_appearance[side][0])
            torch.testing.assert_close(model._opacity, renderer.side_appearance[side][1])
            self.assertEqual(model._features_rest.shape, (count, 0, 3))
            for tensor, base in zip((model._xyz, model._rotation, model._scaling), renderer.side_geometry[side]):
                self.assertNotEqual(tensor.data_ptr(), base.data_ptr())
            calls.append(side)
            return {'render': torch.full((3, camera[1], camera[0]), .2 + .4 * side)}
        with patch.dict(sys.modules, {'gaussian_renderer': SimpleNamespace(render=draw),
                                      'scripts.rendering.render_ply': SimpleNamespace(make_camera=lambda pose, width, height, fov: (width, height))}):
            for ident, yaw in ((1, 0), (2, 1), (3, 0)):
                packet = renderer.render(validate_request({'id': ident, 'frame': 1, 'width': 128, 'height': 128, 'yaw': yaw}, 2))
                size = struct.unpack('<I', packet[:4])[0]; metadata = json.loads(packet[4:4 + size])
                self.assertEqual(metadata['left_frame'], metadata['right_frame'])
                self.assertEqual(metadata['id'], ident)
                self.assertEqual(metadata['cache_hit'], ident == 3)
                image = np.asarray(Image.open(io.BytesIO(packet[4 + size:])))
                self.assertEqual(image.shape, (128, 256, 3))
                self.assertGreater(image[:, 128:].mean(), image[:, :128].mean())
            self.assertEqual(calls, [0, 1, 0, 1])
        torch.testing.assert_close(renderer.side_geometry[0][0], torch.zeros((2, 3)))
        torch.testing.assert_close(renderer.side_geometry[1][0], torch.full((4, 3), 5.))

    def test_rebuild_preserves_every_attribute_of_indexed_original_subset(self):
        left = self.gaussian_model(5)
        left._xyz = torch.arange(15, dtype=torch.float).reshape(5, 3)
        left._features_dc = torch.arange(15, dtype=torch.float).reshape(5, 1, 3)
        left._opacity[0] = float('inf'); left._opacity[2] = float('-inf')
        retained = np.array([0, 2, 4], np.int32)
        def candidate():
            result = self.gaussian_model(4)
            for attribute in ('_xyz', '_scaling', '_features_dc', '_features_rest', '_opacity', '_rotation'):
                getattr(result, attribute)[:3] = getattr(left, attribute)[retained]
            return result
        validate_rebuilt_gaussians(left, candidate(), 5, retained, 1)
        for attribute in ('_xyz', '_scaling', '_features_dc', '_features_rest', '_opacity', '_rotation'):
            right = candidate(); getattr(right, attribute)[2].reshape(-1)[0] += .01
            with self.subTest(attribute=attribute), self.assertRaisesRegex(ValueError, 'frozen retained attribute'):
                validate_rebuilt_gaussians(left, right, 5, retained, 1)
        for wrong in (np.array([2, 0]), np.array([0, 0]), np.array([0, 5]), np.array([0., 2., 4.])):
            with self.subTest(indices=wrong.tolist()), self.assertRaisesRegex(ValueError, 'retained original indices'):
                validate_rebuilt_gaussians(left, candidate(), 5, wrong, 1)
        for seed_count in (0, 200001):
            with self.subTest(seed_count=seed_count), self.assertRaisesRegex(ValueError, '200000'):
                validate_rebuilt_gaussians(left, candidate(), 5, retained, seed_count)
        right = candidate(); right._opacity[-1] = float('inf')
        with self.assertRaisesRegex(ValueError, 'nonfinite new surface'):
            validate_rebuilt_gaussians(left, right, 5, retained, 1)
        right = candidate(); right._features_rest[-1, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'DC-only'):
            validate_rebuilt_gaussians(left, right, 5, retained, 1)
        right = candidate(); right._rotation[-1].zero_()
        with self.assertRaisesRegex(ValueError, 'invalid new surface rotations'):
            validate_rebuilt_gaussians(left, right, 5, retained, 1)

    def test_rebuild_source_report_allows_explicit_replacement_not_unscoped_count_change(self):
        left = {'frame_count': 4, 'fps': 60, 'camera': {'fovy': 40}, 'gaussian_count': 5}
        right = {**left, 'gaussian_count': 4}
        audit = {'artifact_sha256': {'source_ply': 'rebuilt'}}
        report = {'format': 'lego_prior_surface_rebuild_v1', 'source_sha256': 'original',
                  'rebuilt_sha256': 'rebuilt', 'original_gaussian_count': 5, 'gaussian_count': 4,
                  'retained_gaussian_count': 3, 'retired_gaussian_count': 2, 'rebuilt_gaussians': 1,
                  'retained_indices_path': '/tmp/retained.npy', 'retained_indices_sha256': 'mapping',
                  'base_repaired_source': '/tmp/base.ply', 'base_repaired_sha256': 'base',
                  'frozen_retained_vertex_sha256': 'frozen'}
        self.assertTrue(validate_comparison_sources(left, right, audit, 'original', report))
        for key, wrong in [('source_sha256', 'wrong'), ('rebuilt_sha256', 'wrong'),
                           ('original_gaussian_count', 6), ('gaussian_count', 5), ('retained_gaussian_count', 4),
                           ('retired_gaussian_count', 3), ('rebuilt_gaussians', 0), ('rebuilt_gaussians', 200001),
                           ('rebuilt_gaussians', 1.), ('retained_indices_sha256', ''),
                           ('base_repaired_source', ''), ('frozen_retained_vertex_sha256', '')]:
            with self.subTest(key=key, wrong=wrong), self.assertRaisesRegex(ValueError, 'surface rebuild report'):
                validate_comparison_sources(left, right, audit, 'original', {**report, key: wrong})
        for key, wrong in [('frame_count', 5), ('fps', 30), ('camera', {})]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'camera and timestamps'):
                validate_comparison_sources(left, {**right, key: wrong}, audit, 'original', report)
        with self.assertRaisesRegex(ValueError, 'geometry_repair_report'):
            validate_comparison_sources(left, right, audit, 'original')

    def test_rebuild_file_provenance_checks_retained_subset_all_bytes_and_mapping(self):
        from plyfile import PlyData, PlyElement
        from utils.rigid_checkpoint import sha256_file
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            original_path, base_path, rebuilt_path = [folder / name for name in ('original.ply', 'base.ply', 'rebuilt.ply')]
            mapping_path = folder / 'retained.npy'
            original = np.array([(1., float('inf')), (2., 0.), (3., .2), (4., .3)],
                                dtype=[('x', '<f4'), ('opacity', '<f4')])
            base = original.copy(); base['x'][[1, 3]] += 5  # Only retired rows were depth-repaired.
            indices = np.array([0, 2], np.int32)
            appended = np.array([(6., .6)], dtype=original.dtype)
            rebuilt = np.concatenate([base[indices], appended])
            def write(path, data):
                PlyData([PlyElement.describe(data, 'vertex')], text=False).write(path)
            write(original_path, original); write(base_path, base); write(rebuilt_path, rebuilt)
            np.save(mapping_path, indices, allow_pickle=False)
            report = {'source_sha256': sha256_file(original_path), 'rebuilt_sha256': sha256_file(rebuilt_path),
                      'base_repaired_source': str(base_path), 'base_repaired_sha256': sha256_file(base_path),
                      'retained_indices_path': str(mapping_path), 'retained_indices_sha256': sha256_file(mapping_path),
                      'original_gaussian_count': 4, 'gaussian_count': 3, 'retained_gaussian_count': 2,
                      'retired_gaussian_count': 2, 'rebuilt_gaussians': 1,
                      'frozen_retained_vertex_sha256': hashlib.sha256(original[indices].tobytes()).hexdigest()}
            np.testing.assert_array_equal(verify_rebuild_provenance(report, original_path, rebuilt_path), indices)
            self.assertEqual(vertex_rows_sha256(original, indices), report['frozen_retained_vertex_sha256'])
            for key in ('source_sha256', 'rebuilt_sha256', 'base_repaired_sha256', 'retained_indices_sha256'):
                with self.subTest(hash=key), self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    verify_rebuild_provenance({**report, key: 'wrong'}, original_path, rebuilt_path)
            # A reporter cannot bless retained geometry edits merely by updating the full-file SHA.
            mutated = rebuilt.copy(); mutated['x'][0] += .1; write(rebuilt_path, mutated)
            with self.assertRaisesRegex(ValueError, 'frozen original/base'):
                verify_rebuild_provenance({**report, 'rebuilt_sha256': sha256_file(rebuilt_path)}, original_path, rebuilt_path)
            write(rebuilt_path, rebuilt)
            mutated_base = base.copy(); mutated_base['opacity'][2] += .1; write(base_path, mutated_base)
            with self.assertRaisesRegex(ValueError, 'frozen original/base'):
                verify_rebuild_provenance({**report, 'base_repaired_sha256': sha256_file(base_path)}, original_path, rebuilt_path)
            write(base_path, base)
            for wrong in (np.array([2, 0], np.int32), np.array([0, 0], np.int32), np.array([0., 2.]), np.array([0, 4], np.int32)):
                np.save(mapping_path, wrong, allow_pickle=False)
                with self.subTest(indices=wrong.tolist()), self.assertRaisesRegex(ValueError, 'retained indices'):
                    verify_rebuild_provenance({**report, 'retained_indices_sha256': sha256_file(mapping_path)}, original_path, rebuilt_path)


if __name__ == '__main__':
    unittest.main()
