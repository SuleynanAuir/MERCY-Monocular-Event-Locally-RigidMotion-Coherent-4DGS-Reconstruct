import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from utils.local_motion_metrics import angular_metrics,event_terms,summarize_events,local_sensitivity_mask,load_viewer_metrics,FORMAT


class LocalMetricsTests(unittest.TestCase):
    def test_unselected_motion_does_not_dilute_local_score(self):
        t=np.linspace(0,6,80);a=np.stack((np.sin(t),np.sin(t)*100),1);b=a.copy();b[:,0]+=.001*np.sin(15*t)
        first=angular_metrics(a,b,[0],60)
        a[:,1]*=1e8;b[:,1]=a[:,1]+1e9*np.sin(20*t)
        self.assertEqual(first,angular_metrics(a,b,[0],60))

    def test_v3_range_is_shared_and_motion_shrinkage_reported(self):
        a=np.sin(np.linspace(0,6,80))[:,None];b=a*.5
        r=angular_metrics(a,b,[0],60)
        self.assertAlmostEqual(r['max_relative_range_change'],.5)
        self.assertAlmostEqual(r['normalized_jerk']['improvement_percent'],50)

    def test_event_numerator_and_denominator_are_both_local(self):
        target=np.array([[2.,10000.]]);predicted=np.array([[3.,-10000.]])
        self.assertEqual(event_terms(predicted,target,[[True,False]]),(1.,2.))

    def test_zero_event_windows_keep_false_changes_in_summed_loss(self):
        r=summarize_events([2,3],[1,2],[2,0],[0,1])
        self.assertEqual(r['observable_intervals'],1);self.assertEqual(r['zero_event_intervals'],1)
        self.assertEqual(r['mean_interval_relative_l1']['v3'],1)
        self.assertEqual(r['summed_relative_l1']['v3'],2.5)
        self.assertEqual(r['zero_event_false_change_v4'],2)

    def test_mask_is_from_selected_source_groups_only(self):
        jac=np.zeros((3,2,8,8));jac[:,0,2,2]=1;jac[:,1,6,6]=1e8
        mask=local_sensitivity_mask(jac,[6,7],[6])
        self.assertTrue(mask[:,1:4,1:4].all());self.assertFalse(mask[:,5:,5:].any())
        self.assertEqual(int(mask.sum()),18)

    def test_wrong_checkpoint_or_scope_never_reaches_ui(self):
        report=dict(format=FORMAT,checkpoint_sha256='a',scope=dict(joint_groups=[6,8],frame_count=400),ui_summary={'valid':True})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'metrics.json';path.write_text(json.dumps(report));sha=hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertTrue(load_viewer_metrics(path,sha,'a',[6,8],400)['valid'])
            for model,groups,frames in [('b',[6,8],400),('a',[6],400),('a',[6,8],399)]:
                with self.assertRaises(ValueError):load_viewer_metrics(path,sha,model,groups,frames)


if __name__=='__main__':unittest.main()
