import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location('surface_rebuild',
    Path(__file__).resolve().parents[2]/'scripts/geometry/rebuild_lego_rigid_surfaces.py')
rebuild = importlib.util.module_from_spec(spec); spec.loader.exec_module(rebuild)


class SurfaceRebuildTest(unittest.TestCase):
    def test_only_active_moving_rigid_rows_are_retired(self):
        labels=np.array([0,1,2,1,0,2]);dynamic=np.array([1,2,5]);moving_rigid=np.array([False,True,False])
        retained,active,retired=rebuild.retained_rows(labels,dynamic,moving_rigid)
        np.testing.assert_array_equal(retained,[0,2,3,4,5])
        np.testing.assert_array_equal(np.flatnonzero(retired),[1])
        np.testing.assert_array_equal(np.flatnonzero(active[retained]),[1,4])

    def test_retained_indices_stay_strictly_ordered(self):
        retained,_,_=rebuild.retained_rows(np.array([0,1,1,0]),np.array([1,2]),np.array([False,True]))
        self.assertEqual(retained.dtype,np.int32)
        self.assertTrue(np.all(np.diff(retained)>0))
        np.testing.assert_array_equal(retained,[0,3])


if __name__=='__main__':
    unittest.main()
