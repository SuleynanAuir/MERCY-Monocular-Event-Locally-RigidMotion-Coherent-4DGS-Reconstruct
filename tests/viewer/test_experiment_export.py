"""Synthetic end-to-end exports verify alignment, provenance and metrics on CPU."""
import asyncio
import io
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
from aiohttp.test_utils import TestClient, TestServer

import viewer
from utils.experiment_export import camera_record, joint_snapshot, run_export, save_json, validate_plan, capture_gaussians, StateCapture, reference_plan, compact_counts
from utils.viewer_state import validate_request
from utils.version_triplet import VersionTripletRenderer, validate_stage
from scripts.evaluation.evaluate_experiment import evaluate, image_metrics, third_derivative, event_metrics, motion_metrics, rotation_jerk


class SyntheticRenderer:
    def __init__(self, path, cache='cpu'):
        self.path = Path(path)
        self.base_pose = np.eye(4); self.base_pose[2, 3] = 3
        self.target = np.zeros(3)
        self.info = dict(frame_count=4, times_seconds=[0., .1, .3, .6], fps=10., fovy=60.,
                         comparison=True, left_label='V1', right_label='V3')
        self.status = dict(ready=True, scene=self.info)
        self.calls = []

    def load(self): pass

    def render(self, request):
        self.calls.append(dict(request))
        assert request['_raw']
        assert np.asarray(request['_camera_pose']).shape == (4, 4)
        panels = [np.full((request['height'], request['width'], 3), value, np.uint8) for value in (80, 90)]
        if request.get('_capture'):
            for side in range(2):
                points = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
                points[:, 0] += self.info['times_seconds'][request['frame']]**3
                request['_capture'](dict(xyz=points, point_groups=np.array([0, 0])), np.ones((128, 128)))
        request['_panel_timings'] = [.1, .2]
        return dict(panels=panels)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/'source.json'; save_json(self.source, dict(model='synthetic fixture'))
        self.renderer = SyntheticRenderer(self.source)
        self.samples = [dict(frame=i, time_seconds=t, width=128, height=128) for i,t in enumerate(self.renderer.info['times_seconds'])]

    def tearDown(self): self.temp.cleanup()

    def export(self, renderer, plan, output, scene, **kwargs):
        archive=run_export(renderer,plan,output,scene,**kwargs)
        self.assertFalse(output.exists(),'Server must retain only the archive')
        self.assertFalse(Path(str(archive)+'.partial').exists())
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(output.parent if plan['operation']=='complete' else output)
        return archive

    def plan(self, operation='sequence'):
        return validate_plan(dict(operation=operation, samples=self.samples, warmup=3, repeats=2), self.renderer.info)

    def reference(self, mismatch=False):
        camera = camera_record(self.renderer, validate_request(self.samples[0], 4))
        if mismatch: camera['pose_opengl_c2w'][0][3] += 1
        Image.fromarray(np.full((128,128,3),80,np.uint8)).save(self.root/'truth.png')
        Image.fromarray(np.full((128,128),255,np.uint8)).save(self.root/'mask.png')
        reference = self.root/'reference.json'
        save_json(reference, dict(format='scgs_experiment_reference_v1', provenance='independent synthetic truth', camera=camera,
             samples=[dict(frame=i,time_seconds=t,rgb='truth.png',foreground_mask='mask.png') for i,t in enumerate(self.renderer.info['times_seconds'])]))
        return reference

    def test_triplet_preserves_version_order_camera_geometry_and_shared_truth(self):
        class Stage(SyntheticRenderer):
            values = [80, 90]
            def render(self, request):
                self.calls.append(dict(request))
                capture = request.get('_capture')
                if capture:
                    for value in self.values:
                        count = value*5
                        ids = capture.point_ids(count)
                        capture(dict(xyz=np.full((len(ids),3), value, dtype=float),
                                     point_ids=ids, point_count=count), np.full((128,128),value/100))
                request['_panel_timings'] = [float(v) for v in self.values]
                return dict(panels=[np.full((128,128,3),v,np.uint8) for v in self.values])
        base = Stage(self.source)
        base.correction = object()
        middle_source = self.root/'middle.json'; save_json(middle_source, dict(branch='archived V2'))
        middle = Stage(middle_source); middle.values = [85]
        middle.info['comparison_modes'] = ['gaussian','clay','pseudocolor']
        renderer = VersionTripletRenderer(base, middle_source)
        renderer.middle = middle; renderer.base_pose = base.base_pose; renderer.target = base.target
        renderer.info = dict(base.info, version_labels=['V1','V2','V3'],
                             mode_version_labels={'mesh':['V1','V3']})
        output = self.root/'triplet'
        self.export(renderer, self.plan(), output, 'triplet', reference_path=self.reference())
        manifest = json.loads((output/'manifest.json').read_text())
        self.assertEqual([row['label'] for row in manifest['versions']], ['V1','V2','V3'])
        self.assertTrue(manifest['renderer_config']['components'][0]['modules']['correction'])
        self.assertIn(str(middle_source.resolve()), [row['source'] for row in manifest['sources']])
        self.assertEqual(len({row['rgb'] for row in manifest['reference']['samples']}), 1)
        for side, value in enumerate([80,85,90]):
            self.assertTrue(np.all(np.asarray(Image.open(output/f'versions/{side}/frames/000000.png'))==value))
            alpha = np.asarray(Image.open(output/f'versions/{side}/alpha/000000.png'))
            np.testing.assert_allclose(alpha/65535, value/100, atol=1/65535)
            with np.load(output/f'versions/{side}/motion.npz') as motion:
                self.assertTrue(np.all(motion['points']==value))
                self.assertEqual(int(motion['point_ids'][-1]), value*5-1)
            self.assertEqual(manifest['versions'][side]['point_count'], value*5)
        for a,b in zip(base.calls, middle.calls):
            np.testing.assert_array_equal(a['_camera_pose'], b['_camera_pose'])
            self.assertEqual(a['frame'], b['frame'])
        scores = evaluate(output)
        self.assertEqual(set(scores['versions']), {'0','1','2'})
        plan = self.plan(); plan['samples'][1]['render_mode'] = 'mesh'
        with self.assertRaisesRegex(ValueError, '相同的版本'):
            run_export(renderer, plan, self.root/'mixed', 'triplet')
        self.assertFalse((self.root/'mixed').exists())

    def test_triplet_rejects_camera_or_timeline_mismatch(self):
        info = self.renderer.info
        pose = self.renderer.base_pose
        validate_stage(info, info, pose, pose)
        for changed in (dict(info, frame_count=5), dict(info, fps=30),
                        dict(info, times_seconds=[0.,.2,.3,.6]), dict(info, fovy=45)):
            with self.assertRaisesRegex(ValueError,'timeline'):
                validate_stage(info, changed, pose, pose)
        shifted = pose.copy(); shifted[0,3] = .01
        with self.assertRaisesRegex(ValueError,'camera'):
            validate_stage(info, info, pose, shifted)

    def test_lossless_export_metrics_and_tamper_detection(self):
        output = self.root/'export'
        archive = self.export(self.renderer, self.plan(), output, 'test', reference_path=self.reference())
        self.assertTrue(archive.is_file())
        with zipfile.ZipFile(archive) as bundle: self.assertIn('manifest.json', bundle.namelist())
        for side,value in ((0,80),(1,90)):
            pixels = np.asarray(Image.open(output/f'versions/{side}/frames/000000.png'))
            self.assertTrue(np.all(pixels==value))
        result = evaluate(output)
        self.assertTrue(result['versions']['0']['summary']['psnr_infinite'])
        self.assertAlmostEqual(result['versions']['0']['summary']['ssim'],1)
        self.assertAlmostEqual(result['versions']['0']['motion']['position_jerk_rms'],6)
        self.assertAlmostEqual(result['versions']['0']['motion']['rigidity_error'],0)
        self.assertLess(result['versions']['1']['summary']['ssim'],1)
        self.assertEqual(result['versions']['0']['frames'][0]['foreground_missing_fraction'],0)
        (output/'versions/0/frames/000000.png').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'checksum'): evaluate(output)

    def test_camera_mismatch_and_missing_reference_do_not_produce_image_scores(self):
        for label,path in [('mismatch',self.reference(mismatch=True)),('missing',None)]:
            output=self.root/label
            self.export(self.renderer,self.plan(),output,'test',reference_path=path)
            score=evaluate(output)['versions']['0']
            self.assertEqual(score['summary']['evaluated_frames'],0)
            self.assertIsNone(score['summary']['psnr_db'])

    def test_one_click_uses_source_camera_resolution_and_background(self):
        path = self.reference()
        source = json.loads(path.read_text()); source['background'] = 'black'; save_json(path, source)
        plan = validate_plan(dict(operation='complete', settings=dict(width=1024, height=1024,
            yaw=45, render_mode='gaussian', background='light')), self.renderer.info)
        aligned = reference_plan(plan, self.renderer.info, path)
        sample = aligned['samples'][0]
        self.assertEqual((sample['width'], sample['height'], sample['background']), (128, 128, 'black'))
        np.testing.assert_allclose(sample['camera']['pose_opengl_c2w'], source['camera']['pose_opengl_c2w'])
        np.testing.assert_allclose(sample['camera']['K'], source['camera']['K'])
        self.assertEqual(aligned['requested_settings']['yaw'], 45)
        self.assertEqual(reference_plan(self.plan(), self.renderer.info, path), self.plan())
        source['samples'] = []; save_json(path, source)
        with self.assertRaisesRegex(ValueError, '覆盖'): reference_plan(plan, self.renderer.info, path)

    def test_shared_masks_are_lossless_compact_and_work_without_rgb(self):
        path = self.reference()
        labels = np.zeros((128,128), np.uint16); labels[10:100,10:100] = 400
        Image.fromarray(labels).save(self.root/'parts.png')
        source = json.loads(path.read_text())
        for row in source['samples']:
            del row['rgb']; row['part_mask'] = 'parts.png'
        save_json(path, source)
        output = self.root/'masks-export'
        self.export(self.renderer, self.plan(), output, 'test', reference_path=path)
        manifest = json.loads((output/'manifest.json').read_text())
        refs = manifest['reference']['samples']
        self.assertEqual(len({row['foreground_mask'] for row in refs}), 1)
        self.assertEqual(len({row['part_mask'] for row in refs}), 1)
        with Image.open(output/refs[0]['foreground_mask']) as mask_image:
            self.assertEqual(mask_image.mode, '1')
        np.testing.assert_array_equal(np.asarray(Image.open(output/refs[0]['part_mask'])), labels)
        metric = evaluate(output)['versions']['0']['frames'][0]
        self.assertFalse(metric['available'])
        self.assertEqual(metric['silhouette_iou'], 1)
        self.assertEqual(metric['parts']['400']['alpha_missing_fraction'], 0)

    def test_event_packing_preserves_counts_without_unsigned_overflow(self):
        for maximum, dtype in [(255,np.uint8),(256,np.uint16),(65536,np.uint32)]:
            values = np.array([[0,maximum]], float)
            packed = compact_counts(values)
            self.assertEqual(packed.dtype, dtype)
            np.testing.assert_array_equal(values, packed)
        np.testing.assert_array_equal(compact_counts(np.array([.25])), [.25])
        path = self.root/'counts.npz'
        np.savez_compressed(path,positive=np.full((12,12),128,np.uint8),negative=np.full((12,12),128,np.uint8),t0=0.,t1=1.)
        calibration = dict(space='rgb_image',transfer='encoded',luminance='opencv_bt601_u8',log_floor=.03,
                           contrast_positive=.2,contrast_negative=.2)
        image = np.ones((12,12,3))*.5
        metric = event_metrics(image,image,path,calibration,0.,1.)
        self.assertTrue(metric['available'])
        self.assertEqual(metric['no_event_pixels'],0,'uint8 128+128 must not wrap to silence')

    def test_benchmark_warmup_repeats_and_no_state_capture(self):
        output=self.root/'benchmark'
        self.export(self.renderer,self.plan('benchmark'),output,'test')
        self.assertEqual(len(self.renderer.calls),3+2*4)
        self.assertFalse(any('_capture' in call for call in self.renderer.calls))
        self.assertFalse((output/'versions').exists())
        performance=evaluate(output)['performance']
        self.assertEqual(performance['samples'],8)
        self.assertFalse(performance['includes_state_capture'])

    def test_rotation_jerk_and_rejection_of_nonrigid_motion_evidence(self):
        times=np.arange(8,dtype=float)*.05
        angle=times**3
        q=np.zeros((8,1,4));q[:,0,0]=np.cos(angle/2);q[:,0,3]=np.sin(angle/2)
        q[3]*=-1  # Equivalent quaternion sign must not create a spike.
        jerk=rotation_jerk(q,times)
        np.testing.assert_allclose(jerk[...,2],6,atol=1e-9)
        np.testing.assert_allclose(jerk[...,:2],0,atol=1e-9)
        from scripts.evaluation.prepare_lego_motion_evidence import fit_motion
        rest=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
        points=np.broadcast_to(rest,(8,4,3)).copy()+times[:,None,None]
        joints=dict(joint_groups=np.empty((0,2),int),joint_pivots=np.empty((0,3)),joint_axes=np.empty((0,3)))
        fitted=fit_motion(rest,points,np.zeros(4,int),joints)
        self.assertLess(fitted['rigid_fit_max_error'].max(),1e-10)
        points[4,0,0]+=.1
        with self.assertRaisesRegex(ValueError,'not rigid'):fit_motion(rest,points,np.zeros(4,int),joints)

    def test_two_frame_sampling_and_compact_source_motion(self):
        weights=self.root/'weights.pt';weights.write_bytes(b'opaque parameter data; never deserialize during export')
        self.renderer.compact_sources=[weights]
        self.renderer.info['export_sample_step'] = 2
        plan = validate_plan(dict(operation='complete',settings=dict(width=128,height=128)),self.renderer.info)
        self.assertEqual([r['frame'] for r in plan['samples']],[0,2])
        self.assertEqual(plan['sample_step'],2)
        override = validate_plan(dict(operation='complete',sample_step=1),self.renderer.info)
        self.assertEqual(len(override['samples']),4)
        for step in (0,True,2.5,10001):
            with self.assertRaises(ValueError):validate_plan(dict(operation='complete',sample_step=step),self.renderer.info)
        reference=self.reference();source=json.loads(reference.read_text())
        matrices=np.tile(np.eye(4),(4,2,1,1));matrices[:,0,0,3]=np.arange(4)
        np.savez_compressed(self.root/'source-motion.npz',frames=np.arange(4),times_seconds=self.renderer.info['times_seconds'],
            object_matrices=matrices,part_ids=[1,2],source_camera_pose=np.eye(4))
        source.update(source_motion='source-motion.npz',source_motion_scope='source only',part_labels={'1':'arm','2':'base'})
        save_json(reference,source)
        output=self.root/'step2';self.export(self.renderer,plan,output,'test',reference_path=reference)
        manifest=json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['reference']['part_labels']['1'],'arm')
        copied=next(r for r in manifest['sources'] if r['source']==str(weights.resolve()))
        self.assertTrue(copied['included'])
        self.assertEqual((output/copied['file']).read_bytes(),weights.read_bytes())
        with np.load(output/manifest['reference']['source_motion']) as motion:
            np.testing.assert_array_equal(motion['frames'],[0,2])
            np.testing.assert_array_equal(motion['object_matrices'],matrices[[0,2]])

    def test_motion_evidence_is_checked_against_actual_rendered_samples(self):
        points=np.array([[1.,2.,3.],[4.,5.,6.]])
        evidence={0:dict(point_count=10,point_ids=np.array([2,7]),points=points[None],frame_lookup={6:0},
            point_groups=np.array([0,0]),group_quaternions=np.array([[[1.,0,0,0]]]),group_translations=np.zeros((1,1,3)))}
        capture=StateCapture(self.root,0,evidence=evidence,frame=6)
        np.testing.assert_array_equal(capture.point_ids(10),[2,7])
        state=dict(xyz=points,point_ids=np.array([2,7]),point_count=10)
        capture(state,None)
        np.testing.assert_array_equal(capture.states[0]['motion']['group_quaternions'],[[1,0,0,0]])
        stale=StateCapture(self.root,1,evidence=evidence,frame=6)
        with self.assertRaisesRegex(ValueError,'actual rendered'):
            stale(dict(state,xyz=points+.01),None)
        with self.assertRaisesRegex(ValueError,'Point count'):
            StateCapture(self.root,2,evidence=evidence,frame=6).point_ids(11)

    def test_one_click_saves_all_data_and_portable_evaluator_together(self):
        self.renderer.info.update(frame_count=13,times_seconds=[i*.1+i*i*.001 for i in range(13)])
        plan=validate_plan(dict(operation='complete',settings=dict(frame=2,width=128,height=128),
            samples=[self.samples[2]],warmup=2,repeats=2),self.renderer.info)
        self.assertEqual([sample['frame'] for sample in plan['samples']],[0,4,8,12])
        self.assertEqual(plan['sample_step'],4)
        self.assertEqual(plan['selected_frame'],2)
        output=self.root/'complete-export'
        updates=[]
        archive=self.export(self.renderer,plan,output,'test',reference_path=self.reference(),progress=updates.append)
        self.assertEqual(list(dict.fromkeys(row['stage'] for row in updates)),
            ['prepare','frames','motion','reference','sources','code','hashes','archive'])
        self.assertEqual([row['percent'] for row in updates],sorted(row['percent'] for row in updates))
        self.assertTrue(all(0<=row['percent']<100 for row in updates),'Only the job completion may report 100%')
        self.assertEqual(updates[-1]['completed'],updates[-1]['total'])
        self.assertEqual(updates[-1]['stage_count'],8)
        self.assertEqual(len(self.renderer.calls),4,'one render per sampled time, no repeated benchmark')
        self.assertEqual([call['frame'] for call in self.renderer.calls],[0,4,8,12])
        self.assertTrue(all('_capture' in call for call in self.renderer.calls))
        with zipfile.ZipFile(archive) as bundle:
            self.assertEqual({Path(name).parts[0] for name in bundle.namelist()},{output.name})
            for path in ('manifest.json','plan.json','timings.json','evaluate.py',
                         'data_availability.json','README.txt','versions/0/frames/000003.png',
                         'versions/1/motion.npz','reference/000000-rgb.png'):
                self.assertIn(output.name+'/'+path,bundle.namelist())
            self.assertFalse(any('/states/' in name or '/sources/' in name for name in bundle.namelist()))
            self.assertEqual(sum('/code/' in name for name in bundle.namelist()),3)
        availability=json.loads((output/'data_availability.json').read_text())
        self.assertEqual(availability['reference']['rgb']['saved'],4)
        self.assertTrue(any('事件' in item for item in availability['missing_data']))
        self.assertFalse((output/'metrics.json').exists(),'UI export must leave image metrics to the standalone script')
        self.source.unlink()
        moved=self.root/'moved-export';output.rename(moved)
        process=subprocess.run([sys.executable,str(moved/'evaluate.py')],cwd=self.root,capture_output=True,text=True)
        self.assertEqual(process.returncode,0,process.stderr)
        metrics=json.loads((moved/'metrics.json').read_text())
        self.assertEqual(metrics['versions']['0']['summary']['evaluated_frames'],4)
        self.assertEqual(metrics['performance']['operation'],'complete')
        self.assertEqual(metrics['performance']['samples'],4)
        self.assertTrue(metrics['performance']['includes_state_capture'])

    def test_large_file_copy_hash_and_zip_report_partial_bytes(self):
        payload=self.root/'model.bin';payload.write_bytes(b'fixture\x00'*(300*1024))
        save_json(self.source,dict(model=payload.name))
        updates=[]
        with patch('utils.experiment_export.time.monotonic',side_effect=itertools.count(0,.2)):
            archive=self.export(self.renderer,self.plan(),self.root/'large-export','test',progress=updates.append)
        for stage in ('sources','hashes','archive'):
            rows=[row for row in updates if row['stage']==stage and row['unit']=='bytes']
            self.assertTrue(any(0<row['completed']<row['total'] for row in rows),stage)
            self.assertEqual(rows[-1]['completed'],rows[-1]['total'],stage)
        with zipfile.ZipFile(archive) as bundle:
            self.assertIsNone(bundle.testzip())
            self.assertFalse(any(name.endswith('.bin') for name in bundle.namelist()))
        self.assertLess(archive.stat().st_size,payload.stat().st_size/10)
        manifest=json.loads((self.root/'large-export/manifest.json').read_text())
        record=next(row for row in manifest['sources'] if row['source']==str(payload.resolve()))
        self.assertFalse(record['included']);self.assertEqual(len(record['sha256']),64)
        self.assertEqual([row['percent'] for row in updates],sorted(row['percent'] for row in updates))

    def test_saved_absolute_camera_survives_base_camera_change(self):
        plan=self.plan()
        for sample in plan['samples']: sample['camera']=camera_record(self.renderer,sample)
        self.renderer.base_pose[0,3]=100
        validated=validate_plan(plan,self.renderer.info)
        self.export(self.renderer,validated,self.root/'frozen','test')
        self.assertTrue(all(call['_camera_pose'][0][3]==0 for call in self.renderer.calls))

    def test_sparse_capture_never_transfers_full_model_attributes(self):
        selections=[]
        class LargeGeometry:
            def __len__(self): return 1000000
            def __getitem__(self, ids):
                selections.append(len(ids))
                return np.column_stack((ids,ids*0,ids*0)).astype(np.float32)
            def __array__(self, *args, **kwargs):
                raise AssertionError('Full geometry transfer is forbidden')
        output=self.root/'points'
        recorder=StateCapture(output,0)
        capture_gaussians({'_capture':recorder},SimpleNamespace(_xyz=LargeGeometry()))
        self.assertEqual(selections,[256])
        self.assertEqual(recorder.states[0]['count'],1000000)
        self.assertEqual(recorder.states[0]['points'].shape,(256,3))
        self.assertFalse(any(output.rglob('*.npz')),'No full-state snapshots')

    def test_size_limit_and_render_failure_remove_only_owned_partial_files(self):
        sentinel=self.root/'keep-model.bin';sentinel.write_bytes(b'original model')
        output=self.root/'limited'
        with patch('utils.experiment_export.MAX_EXPORT_BYTES',1024):
            with self.assertRaisesRegex(ValueError,'容量上限'):
                run_export(self.renderer,self.plan(),output,'test')
        self.assertFalse(output.exists());self.assertFalse(Path(str(output)+'.zip').exists())
        self.assertFalse(Path(str(output)+'.zip.partial').exists())
        with patch.object(self.renderer,'render',side_effect=OSError(28,'disk full')):
            with self.assertRaises(OSError):run_export(self.renderer,self.plan(),output,'test')
        self.assertFalse(output.exists())
        self.assertEqual(sentinel.read_bytes(),b'original model')
        output.mkdir();(output/'keep').write_text('existing result')
        with self.assertRaises(FileExistsError):run_export(self.renderer,self.plan(),output,'test')
        self.assertEqual((output/'keep').read_text(),'existing result')

    def test_stride_four_aggregates_all_event_bins_without_gaps(self):
        self.renderer.info.update(frame_count=9,times_seconds=[i/10 for i in range(9)])
        reference=self.reference();data=json.loads(reference.read_text())
        data['event_calibration']=dict(space='rgb_image',transfer='linear',contrast_positive=.2,contrast_negative=.2)
        data['samples']=[]
        for i,t in enumerate(self.renderer.info['times_seconds']):
            row=dict(frame=i,time_seconds=t,rgb='truth.png',foreground_mask='mask.png')
            if i:
                row['events']=f'events-{i}.npz'
                np.savez(self.root/row['events'],positive=np.full((128,128),i,dtype=np.uint16),negative=np.zeros((128,128),np.uint16),t0=(i-1)/10,t1=t)
            data['samples'].append(row)
        save_json(reference,data)
        plan=validate_plan(dict(operation='complete',settings=dict(width=128,height=128)),self.renderer.info)
        two_step=validate_plan(dict(operation='complete',sample_step=2,settings=dict(width=128,height=128)),self.renderer.info)
        two_output=self.root/'two-step-events';self.export(self.renderer,two_step,two_output,'test',reference_path=reference)
        two_refs=json.loads((two_output/'manifest.json').read_text())['reference']['samples']
        for index,expected in enumerate([3,7,11,15],1):
            with np.load(two_output/two_refs[index]['events']) as events:
                np.testing.assert_array_equal(events['positive'],expected)
                self.assertAlmostEqual(float(events['t1'])-float(events['t0']),.2)
        for label,broken in [('valid',False),('gap',True)]:
            if broken:
                del data['samples'][2]['events'];save_json(reference,data)
            output=self.root/label
            self.export(self.renderer,plan,output,'test',reference_path=reference)
            manifest=json.loads((output/'manifest.json').read_text());refs=manifest['reference']['samples']
            self.assertNotIn('events',refs[0])
            self.assertEqual([s['frame'] for s in manifest['samples']],[0,4,8])
            self.assertEqual(refs[0]['foreground_mask'],refs[2]['foreground_mask'],'fixed mask stored once')
            if broken:
                self.assertNotIn('events',refs[1]);self.assertIn('events_unavailable',refs[1])
            else:
                with np.load(output/refs[1]['events']) as events:
                    np.testing.assert_array_equal(events['positive'],10)
                    self.assertEqual((float(events['t0']),float(events['t1'])),(0.,.4))
            with np.load(output/refs[2]['events']) as events:
                np.testing.assert_array_equal(events['positive'],26)
                self.assertEqual((float(events['t0']),float(events['t1'])),(.4,.8))

    def test_invalid_times_order_resolution_and_pose(self):
        for update in ({'time_seconds':.01},{'frame':True},{'width':4000},{'camera':{'pose_opengl_c2w':np.zeros((4,4)).tolist()}}):
            samples=[{**self.samples[0],**update}]
            with self.assertRaises((ValueError,KeyError)):
                validate_plan(dict(operation='frame',samples=samples),self.renderer.info)
        with self.assertRaises(ValueError):
            validate_plan(dict(operation='sequence',samples=list(reversed(self.samples))),self.renderer.info)

    def test_nonuniform_jerk_and_event_intervals(self):
        times=np.array([0.,.1,.4,1.,1.3])
        np.testing.assert_allclose(third_derivative(times**3,times),6)
        events=self.root/'events.npz'
        np.savez(events,positive=np.zeros((12,12)),negative=np.zeros((12,12)),t0=0.,t1=.1)
        calibration=dict(space='rgb_image',transfer='linear',contrast_positive=.2,contrast_negative=.2)
        image=np.full((12,12,3),.5)
        score=event_metrics(image,image,events,calibration,0,.1)
        self.assertEqual(score['no_event_change_mean'],0)
        self.assertIsNone(score['relative_l1'])
        self.assertFalse(event_metrics(image,image,events,calibration,0,.2)['available'])
        self.assertFalse(event_metrics(image,image,events,None,0,.1)['available'])

    def test_ssim_matches_reference_library(self):
        try: from skimage.metrics import structural_similarity
        except ImportError: self.skipTest('Optional independent skimage reference unavailable')
        rng=np.random.default_rng(19)
        a,b=rng.random((32,35,3)),rng.random((32,35,3))
        expected=structural_similarity(a,b,channel_axis=2,data_range=1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)
        self.assertAlmostEqual(image_metrics(a,b)['ssim'],expected,places=12)

    def test_declared_hinge_angles_gap_and_angular_jerk(self):
        times=np.array([0.,.1,.3,.6])
        structure=SimpleNamespace(joint_groups=np.array([[0,1]]),joint_pivots=np.zeros((1,3)),joint_axes=np.array([[0,0,1]]))
        rotations=np.zeros((4,2,4));rotations[:,0,0]=1
        rotations[:,1,0]=np.cos(times**3/2);rotations[:,1,3]=np.sin(times**3/2)
        snapshots=[joint_snapshot(structure,q) for q in rotations]
        path=self.root/'motion.npz'
        np.savez(path,points=np.zeros((4,1,3)),point_ids=[0],times_seconds=times,
            group_quaternions=rotations,group_translations=np.zeros((4,2,3)),
            joint_pairs=structure.joint_groups,joint_pivots=structure.joint_pivots,
            joint_angles=np.stack([row['joint_angles'] for row in snapshots]))
        score=motion_metrics(path)
        self.assertAlmostEqual(score['angular_jerk_rms'],6)
        self.assertAlmostEqual(score['joint_gap_max'],0)
        structure.joint_axes=np.array([[1,0,0]])
        self.assertNotIn('joint_angles',joint_snapshot(structure,rotations[-1]))


class ExportHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        source=self.root/'source.json';save_json(source,{'fixture':True})
        self.client=TestClient(TestServer(viewer.create_app(source,renderer_factory=SyntheticRenderer,export_root=self.root/'exports')))
        await self.client.start_server()
        for _ in range(100):
            status=await (await self.client.get('/api/status')).json()
            if status['ready']:break
            await asyncio.sleep(.01)

    async def asyncTearDown(self):
        await self.client.close();self.temp.cleanup()

    async def test_plan_job_poll_download_and_origin(self):
        data=dict(operation='frame',samples=[dict(frame=0,time_seconds=0,width=128,height=128)])
        frozen=await self.client.post('/api/experiments/plan',json=data)
        self.assertEqual(frozen.status,200)
        plan=await frozen.json();self.assertIn('camera',plan['samples'][0])
        response=await self.client.post('/api/experiments',json=plan)
        self.assertEqual(response.status,202)
        job=await response.json()
        for _ in range(200):
            job=await (await self.client.get('/api/experiments/'+job['id'])).json()
            if job['status']!='running':break
            await asyncio.sleep(.01)
        self.assertEqual(job['status'],'complete',job)
        response=await self.client.get('/api/experiments/'+job['id']+'/download')
        with zipfile.ZipFile(io.BytesIO(await response.read())) as bundle:
            self.assertIn('versions/0/frames/000000.png',bundle.namelist())
        for route in ('/api/experiments/missing','/api/experiments/missing/download'):
            self.assertEqual((await self.client.get(route)).status,404)
        response=await self.client.post('/api/experiments',json=data,headers={'Origin':'https://example.com'})
        self.assertEqual(response.status,403)
        data['samples'][0]['time_seconds']=1
        self.assertEqual((await self.client.post('/api/experiments',json=data)).status,400)

    async def test_one_click_returns_saved_folder_and_archive_without_plan_setup(self):
        response=await self.client.post('/api/experiments',json=dict(operation='complete',
            settings=dict(width=128,height=128,frame=1),warmup=1,repeats=1))
        self.assertEqual(response.status,202)
        job=await response.json()
        for _ in range(300):
            job=await (await self.client.get('/api/experiments/'+job['id'])).json()
            if job['status']!='running':break
            await asyncio.sleep(.01)
        self.assertEqual(job['status'],'complete',job)
        archive=Path(job['output_archive']);directory=archive.with_suffix('')
        self.assertEqual(job['percent'],100)
        self.assertEqual(job['stage'],'complete')
        self.assertEqual(directory.parent,(self.root/'exports').resolve())
        self.assertTrue(directory.name.startswith('default_'))
        self.assertFalse(directory.exists())
        self.assertEqual(job['sample_step'],4)
        self.assertEqual(job['sampled_frames'],1)
        self.assertEqual(job['output_bytes'],archive.stat().st_size)
        self.assertTrue(job['missing_data'])
        response=await self.client.get('/api/experiments/'+job['id']+'/download')
        self.assertIn(directory.name+'.zip',response.headers['Content-Disposition'])
        with zipfile.ZipFile(io.BytesIO(await response.read())) as bundle:
            self.assertIn(directory.name+'/evaluate.py',bundle.namelist())

    async def test_failed_job_preserves_phase_and_never_exposes_partial_archive(self):
        def disk_full(*args, progress, **kwargs):
            progress(dict(stage='frames',message='导出无损帧与模型状态',percent=32.5,
                          stage_index=4,stage_count=10,completed=1,total=4,unit='frames'))
            raise OSError(28,'No space left on device')
        with patch('utils.viewer_experiments.run_export',side_effect=disk_full), self.assertLogs('gaussian-viewer',level='ERROR'):
            response=await self.client.post('/api/experiments',json=dict(operation='complete',settings={},warmup=1,repeats=1))
            self.assertEqual(response.status,202)
            job=await response.json()
            for _ in range(100):
                job=await (await self.client.get('/api/experiments/'+job['id'])).json()
                if job['status']!='running':break
                await asyncio.sleep(.01)
        self.assertEqual(job['status'],'failed')
        self.assertEqual(job['percent'],32.5)
        self.assertEqual(job['stage'],'frames')
        self.assertIn('导出无损帧与模型状态失败',job['message'])
        self.assertIn('No space left on device',job['message'])
        self.assertEqual((await self.client.get('/api/experiments/'+job['id']+'/download')).status,409)


if __name__=='__main__':unittest.main()
