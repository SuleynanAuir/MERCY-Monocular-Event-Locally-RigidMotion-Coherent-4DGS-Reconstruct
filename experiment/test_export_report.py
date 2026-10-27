"""Statistical checks that detect unpaired resampling and temporal loss."""
import unittest
import numpy as np
from experiment.export_report_analysis import stationary_indices, paired_interval


class PairedTemporalInference(unittest.TestCase):
    def test_shared_temporal_trend_cancels(self):
        trend=np.arange(200,dtype=float)**2
        indices=stationary_indices(200,10,count=200)
        result=paired_interval(trend,trend+3,1,indices)
        self.assertEqual((result['low'],result['high'],result['oriented_delta']),(3,3,3))

    def test_lower_is_better_and_excluded_windows(self):
        a=np.arange(200,dtype=float); b=a-2
        a[10:29]=np.nan; b[10:29]=np.nan
        result=paired_interval(a,b,-1,stationary_indices(200,10,count=200))
        self.assertEqual(result['n'],181)
        self.assertEqual(result['win_fraction'],1)
        self.assertEqual(result['oriented_delta'],2)
        self.assertEqual(result['raw_delta'],-2)

    def test_reproducible_and_temporally_contiguous(self):
        first=stationary_indices(200,10,count=500)
        np.testing.assert_array_equal(first,stationary_indices(200,10,count=500))
        continuation=(first[:,1:]==(first[:,:-1]+1)%200).mean()
        self.assertTrue(.88<continuation<.92)
        self.assertGreater(np.std(first[:,0]),50)


if __name__=='__main__':unittest.main()
