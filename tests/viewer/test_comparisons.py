import asyncio
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from aiohttp import web,FormData
from aiohttp.test_utils import TestClient,TestServer
from scripts.evaluation.evaluate_experiment import part_metrics,image_metrics,ssim_map
from utils.comparison_io import extract,relative_name
from utils.comparison_outputs import allocate,object_name
from datetime import datetime,timezone
from utils.viewer_comparisons import install_comparisons
from viewer import local_only
from tests.viewer.test_experiment_export import SyntheticRenderer
from utils.experiment_export import run_export,validate_plan,save_json


class GroupedPartTests(unittest.TestCase):
    def test_equivalent_to_individual_masks_with_small_and_edge_parts(self):
        rng=np.random.default_rng(10);truth=rng.random((38,41,3));pred=rng.random(truth.shape)
        masks=rng.integers(0,8,(38,41),dtype=np.uint16);masks[:5,:5]=60000
        coverage=rng.random(masks.shape)>.2;ss=ssim_map(truth,pred)
        result=part_metrics(truth,pred,masks,ss,coverage)
        for pid,row in result.items():
            selected=masks==int(pid);expected=image_metrics(truth,pred,selected,ss)
            for key in ('pixels','l1','mse','psnr_db','ssim'):
                if expected[key] is None:self.assertIsNone(row[key])
                else:self.assertAlmostEqual(row[key],expected[key],places=11)
            self.assertAlmostEqual(row['alpha_missing_fraction'],(selected&~coverage).sum()/selected.sum(),places=12)
    def test_perfect_and_missing_rgb_are_explicit(self):
        image=np.ones((12,12,3));mask=np.ones((12,12),np.uint8)
        row=part_metrics(image,image,mask,ssim_map(image,image),None)['1']
        self.assertTrue(row['psnr_infinite']);self.assertIsNone(row['psnr_db'])
        missing=part_metrics(None,image,mask,None,None)['1'];self.assertFalse(missing['available']);self.assertEqual(missing['pixels'],144)
    def test_reject_zip_traversal_duplicate_and_symlink(self):
        for name in ('../manifest.json','/root/manifest.json','x/../../a','x\\evil'):
            with self.assertRaises(ValueError):relative_name(name)
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            for name in ('../outside','link'):
                archive=folder/'test.zip'
                with zipfile.ZipFile(archive,'w') as z:
                    entry=zipfile.ZipInfo(name)
                    if name=='link':entry.external_attr=0o120777<<16
                    z.writestr(entry,'x')
                with self.assertRaises(ValueError):extract(archive,folder/'extract')
            self.assertFalse((folder/'outside').exists())


class ComparisonHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.repo=Path(__file__).resolve().parents[2]
        app=web.Application(middlewares=[local_only]);install_comparisons(app,self.repo,self.root,self.root/'results')
        self.client=TestClient(TestServer(app));await self.client.start_server()
    async def asyncTearDown(self):await self.client.close();self.temp.cleanup()
    async def test_path_and_origin_and_incomplete_results(self):
        response=await self.client.get('/comparison');self.assertEqual(response.status,200);self.assertIn('对照试验可视化',await response.text())
        for path in ('/etc/passwd',str(self.root/'missing')):
            r=await self.client.post('/api/comparisons',json={'path':path,'object_name':'lego'});self.assertEqual(r.status,400)
        r=await self.client.post('/api/comparisons',json={'path':'x'},headers={'Origin':'https://example.com'});self.assertEqual(r.status,403)
        r=await self.client.get('/api/comparisons/no/files/report.html');self.assertEqual(r.status,404)
        form=FormData();form.add_field('files',b'x',filename='../manifest.json')
        r=await self.client.post('/api/comparisons',data=form);self.assertEqual(r.status,400)
    async def test_real_upload_progress_complete_without_rgb_and_download(self):
        await self.exercise_import(False)

    async def test_real_zip_upload_progress_report_and_download(self):
        await self.exercise_import(True)

    async def test_named_lookup_survives_restart_and_legacy_links_still_resolve(self):
        named,meta=allocate(self.root/'results','lego')
        legacy=self.root/'results'/'comparisons'/('a'*32);legacy.mkdir(parents=True)
        for target in (named,legacy):
            (target/'progress.json').write_text(json.dumps(dict(status='complete',stage='complete',percent=100)))
            (target/'report.html').write_text('existing report')
            (target/'artifact_manifest.json').write_text(json.dumps(dict(files={'report.html':{}})))
        for job in (meta['id'],'a'*32):
            r=await self.client.get('/api/comparisons/'+job);self.assertEqual(r.status,200)
            self.assertEqual((await (await self.client.get('/api/comparisons/'+job+'/files/report.html')).text()),'existing report')
        listing=await (await self.client.get('/api/comparisons')).json()
        self.assertEqual({j['id'] for j in listing['history']},{meta['id'],'a'*32})

    async def test_bad_name_is_rejected_before_starting_a_job(self):
        for name in ('','../escape','a/b','x'*41):
            response=await self.client.post('/api/comparisons',json={'path':'anything','object_name':name})
            self.assertEqual(response.status,400)
        self.assertFalse((self.root/'results').exists())

    async def exercise_import(self, zipped):
        source=self.root/'model.json';save_json(source,{'fixture':True});renderer=SyntheticRenderer(source)
        plan=validate_plan(dict(operation='sequence',samples=[dict(frame=i,time_seconds=t,width=128,height=128) for i,t in enumerate(renderer.info['times_seconds'])],warmup=0,repeats=1),renderer.info)
        archive=run_export(renderer,plan,self.root/'synthetic','fixture')
        form=FormData(quote_fields=False)
        form.add_field('object_name','测试 物品')
        if zipped:
            form.add_field('archive',archive.read_bytes(),filename='experiment.zip',content_type='application/zip')
        else:
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    if not name.endswith('/'):form.add_field('files',z.read(name),filename='fixture/'+name)
        response=await self.client.post('/api/comparisons',data=form);self.assertEqual(response.status,202,await response.text());job=await response.json();values=[];seen=set()
        self.assertEqual(job['object_name'],'测试_物品');self.assertEqual(Path(job['output_directory']).parent,(self.root/'results').resolve())
        self.assertRegex(Path(job['output_directory']).name,r'^测试_物品_\d{8}_\d{6}_\d{6}$')
        stem=job['output_stem'];self.assertEqual(job['download_name'],stem+'.zip')
        for _ in range(160):
            job=await (await self.client.get('/api/comparisons/'+job['id'])).json();values.append(job['percent']);seen.add(job['stage'])
            if job['status']!='running':break
            await asyncio.sleep(.25)
        self.assertEqual(job['status'],'complete',job);self.assertEqual(values,sorted(values));self.assertEqual(job['percent'],100)
        self.assertTrue(all(r['psnr'] is None and r['lpips'] is None for r in job['summary']))
        self.assertTrue(job['warnings']);self.assertGreater(len(seen),2)
        report=await self.client.get(job['report_url']);self.assertEqual(report.status,200);self.assertIn('不可用',await report.text())
        self.assertIn("frame-ancestors 'self'",report.headers['Content-Security-Policy'])
        self.assertEqual((await self.client.get(job['report_url'].replace('report.html','report.js'))).status,200)
        self.assertEqual((await self.client.get(job['report_url'].replace('report.html','report.css'))).status,200)
        response=await self.client.get(job['download_url']);self.assertEqual(response.status,200)
        self.assertIn("filename*=UTF-8''",response.headers['Content-Disposition'])
        with zipfile.ZipFile(io.BytesIO(await response.read())) as z:
            self.assertNotIn('figures/03_paired_deltas.png',z.namelist());self.assertIn('figures/'+stem+'_01_quality.png',z.namelist());self.assertFalse(any(p.startswith('versions/') for p in z.namelist()))
            manifest=json.loads(z.read('artifact_manifest.json'));self.assertNotIn(job['download_name'],manifest['files']);self.assertIn('job.json',manifest['files'])
            self.assertTrue(all(Path(p).name.startswith(stem+'_') for p in z.namelist() if p.startswith('figures/')))
        listing=await (await self.client.get('/api/comparisons')).json();self.assertEqual(listing['history'][0]['output_stem'],stem)
        self.assertEqual((await self.client.get('/api/comparisons/'+job['id']+'/files/results.zip')).status,200)
        self.assertEqual((await self.client.get('/api/comparisons/'+job['id']+'/files/worker.log')).status,404)

class OutputNamingTests(unittest.TestCase):
    def test_same_instant_never_overwrites_and_names_use_beijing_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            now=datetime(2026,10,27,8,9,10,tzinfo=timezone.utc)
            first,a=allocate(tmp,'吊机 A',now=now);second,b=allocate(tmp,'吊机 A',now=now)
            self.assertEqual(first.name,'吊机_A_20261027_160910_000000')
            self.assertEqual(second.name,'吊机_A_20261027_160910_000001')
            self.assertNotEqual(a['id'],b['id']);self.assertTrue((first/'job.json').is_file())
    def test_name_cannot_escape_output_or_inject_headers(self):
        for value in ('','../lego','lego/a','a\\b','<script>','..','_'*4,'a'*41,None):
            with self.assertRaises(ValueError):object_name(value)
        self.assertEqual(object_name(' 乐高 吊机-1 '),'乐高_吊机-1')
