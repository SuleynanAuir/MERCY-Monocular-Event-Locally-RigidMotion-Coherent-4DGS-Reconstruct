"""Numerical semantics of the presentation (not pixel snapshots)."""
import unittest
import numpy as np
from experiment.comparison_figures import radar_values, improvement, select_gap_frames, estimate_cycle, area_contributions, error_difference
from experiment.comparison_figures import profile_groups, temporal_keys


class PresentationRules(unittest.TestCase):
    def test_missing_events_use_explicit_image_profile_without_invented_values(self):
        rows = [dict(psnr=30, foreground_psnr=20, ssim=.9, foreground_ssim=.8,
                     lpips=.1, iou=.8, missing=3, event_l1=None) for _ in range(3)]
        title, keys = profile_groups(rows)[1]
        self.assertIn('events unavailable', title)
        self.assertGreaterEqual(len(keys), 3)
        self.assertNotIn('event_l1', keys)
        series = {k: np.ones((3, 4)) for k in rows[0]}
        series['event_l1'][:] = np.nan
        selected = temporal_keys(series)
        self.assertEqual(len(selected), 6)
        self.assertNotIn('event_l1', selected)
        self.assertIn('psnr', selected)

    def test_lower_error_has_smaller_raw_radius(self):
        raw=np.array([[.14,1.5,.8],[.045,1.3,.92],[.022,.69,.93]])
        radius,caps=radar_values(raw,['lpips','event_l1','iou'])
        np.testing.assert_allclose(radius*caps,raw)
        self.assertTrue(np.all(np.diff(radius[:,:2],axis=0)<0))
        self.assertEqual(caps[2],1.)

    def test_gap_selection_is_global_pairwise_sum_not_fixed_times(self):
        psnr=np.array([[10,10,10,10,10,10],[11,12,11,13,14,11],[12,18,13,20,21,12]])
        selected,score=select_gap_frames(psnr)
        np.testing.assert_array_equal(selected,[4,3,1,2])
        for i in range(6):self.assertEqual(score[i],sum(abs(psnr[a,i]-psnr[b,i]) for a,b in [(0,1),(0,2),(1,2)]))

    def test_rank_excludes_unpaired_inf_and_breaks_ties_by_earlier_time(self):
        selected,_=select_gap_frames([[0,0,np.inf,0],[1,1,3,1],[2,2,4,3]])
        np.testing.assert_array_equal(selected,[3,0,1])
        self.assertEqual(select_gap_frames(np.full((3,2),np.nan))[0].size,0)

    def test_oriented_gains_preserve_regressions_and_undefined_baseline(self):
        self.assertAlmostEqual(improvement(.1,.05,-1),50)
        self.assertLess(improvement(3.3,6.2,-1),0)
        self.assertTrue(np.isnan(improvement(0,1,1)))
        self.assertTrue(np.isnan(improvement(None,1,1)))

    def test_cycle_uses_recurrence_not_half_cycle_magnitude(self):
        t=np.arange(200)/30
        y=np.column_stack((np.cos(2*np.pi*t/(100/30)),np.sin(2*np.pi*t/(100/30))))
        result=estimate_cycle(t,y)
        self.assertEqual(result['period_samples'],100)
        self.assertAlmostEqual(result['period_seconds'],100/30)
        self.assertIsNone(estimate_cycle(t,np.ones((200,3))))
        self.assertIsNone(estimate_cycle(t,t[:,None]))

    def test_area_contribution_is_complete_and_not_forced_monotonic(self):
        support=np.array([[10.,100.,0.],[10.,100.,1000.]])
        detail=np.array([[[.8,.1,np.nan],[.8,.1,.01]],[[.9,.2,np.nan],[.9,.2,.02]],[[.7,.15,np.nan],[.7,.15,.03]]])
        result=area_contributions(support,detail)
        np.testing.assert_array_equal(sorted(np.concatenate(result['groups'])),[0,1,2])
        expected=np.nansum(detail*support,axis=2)/support.sum(1)*100
        np.testing.assert_allclose(result['per_frame'].sum(2),expected)
        self.assertTrue(np.all(expected[1]>expected[0]))  # Real regressions must survive the transform.
        self.assertEqual(result['excluded_pixel_frames'],0)

    def test_area_contribution_denominator_is_common_across_versions(self):
        support=np.array([[10.,90.],[10.,90.]])
        detail=np.array([[[.1,.2],[.1,np.nan]],[[.2,.3],[.2,.3]]])
        result=area_contributions(support,detail)
        np.testing.assert_allclose(result['denominator'],[100,10])
        self.assertEqual(result['excluded_pixel_frames'],90)
        np.testing.assert_allclose(np.nansum(result['per_frame'],axis=2),[[19,10],[29,20]])

    def test_missing_parts_fall_back_to_measured_common_foreground(self):
        import tempfile, json
        from pathlib import Path
        from PIL import Image
        from experiment.comparison_figures import parts
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';(out/'figures').mkdir(parents=True);(out/'tables').mkdir()
            mask=np.array([[255,255],[0,0]],dtype=np.uint8)
            Image.fromarray(mask).save(root/'mask.png')
            for v,alpha in enumerate([[[0,65535],[0,0]],[[65535,65535],[0,0]]]):
                dest=root/f'versions/{v}/alpha';dest.mkdir(parents=True)
                Image.fromarray(np.array(alpha,dtype=np.uint16)).save(dest/'000000.png')
            a={'summary':[{'version':'V1'},{'version':'V2'}], 'protocol':{'source_version_ids':[0,1]}}
            z={'times':np.array([0.])};audit={}
            parts(a,z,out,'',audit,root,{'reference':{'samples':[{'foreground_mask':'mask.png','aligned':True}]}})
            self.assertEqual(audit['parts']['gt_area_pooled_percent'],[50.,0.])
            self.assertFalse(audit['parts']['part_annotations_available'])
            self.assertEqual(audit['parts']['sample_indices'],[0])
            self.assertTrue((out/'tables/foreground_coverage.csv').is_file())
            self.assertGreater((out/'figures/06_parts.png').stat().st_size,10000)

    def test_motion_profiles_use_measured_diagnostics_without_fabricated_constraints(self):
        from experiment.comparison_figures import motion_profile_keys
        rows=[dict(joint_gap_max=None,rotation_jerk_rms=None,motion_amplitude=.6,position_jerk_rms=5.7,points=256)]
        keys,title=motion_profile_keys(rows)
        self.assertEqual([k[0] for k in keys],['motion_amplitude','position_jerk_rms','points'])
        self.assertIn('diagnostics',title)
        rows[0]['joint_gap_max']=0.
        keys,title=motion_profile_keys(rows)
        self.assertEqual([k[0] for k in keys],['joint_gap_max'])
        self.assertEqual(motion_profile_keys([{}])[0],[])

    def test_error_delta_keeps_signed_regressions_and_zero(self):
        d=error_difference([[.2,.1,.3]],[[.1,.2,.3]])
        np.testing.assert_allclose(d,[[-.1,.1,0.]])

if __name__=='__main__':unittest.main()
