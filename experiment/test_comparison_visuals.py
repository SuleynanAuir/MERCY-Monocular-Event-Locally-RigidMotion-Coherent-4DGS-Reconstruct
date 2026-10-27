"""Numerical semantics of the presentation (not pixel snapshots)."""
import unittest
import numpy as np
from experiment.comparison_figures import radar_values, improvement, select_gap_frames, estimate_cycle, area_contributions, error_difference


class PresentationRules(unittest.TestCase):
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

    def test_error_delta_keeps_signed_regressions_and_zero(self):
        d=error_difference([[.2,.1,.3]],[[.1,.2,.3]])
        np.testing.assert_allclose(d,[[-.1,.1,0.]])

if __name__=='__main__':unittest.main()
