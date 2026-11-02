import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.evaluation.audit_spatial_improvement import audit


class SpatialAuditTest(unittest.TestCase):
    def test_reports_local_regression_despite_mean_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = dict(versions=[dict(id=str(i)) for i in range(3)], samples=[dict(frame=7)],
                            reference=dict(samples=[dict(aligned=True, rgb='rgb.png', foreground_mask='mask.png')]))
            with zipfile.ZipFile(root/'data.zip', 'w') as z:
                z.writestr('run/manifest.json', json.dumps(manifest))
                def put(name, a):
                    b=io.BytesIO(); Image.fromarray(a.astype('uint8')).save(b, format='PNG');z.writestr('run/'+name,b.getvalue())
                put('rgb.png',np.full((2,2,3),100));put('mask.png',np.full((2,2,3),255))
                for i,value in enumerate((150,140,100)):
                    a=np.full((2,2,3),value)
                    if i==2:a[0,0]=150
                    put(f'versions/{i}/frames/000000.png',a)
            result=audit(root/'data.zip',root/'audit.json')
            self.assertTrue(result['all_frames_improved'])
            self.assertEqual(result['samples'][0]['foreground_degraded_fraction'],.25)
            self.assertEqual(result['samples'][0]['foreground_improved_fraction'],.75)

if __name__=='__main__':unittest.main()
