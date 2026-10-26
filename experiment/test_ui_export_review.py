import unittest
import numpy as np
from .ui_export_review import layout, valid_mask, masked_scores


class OriginalExportTests(unittest.TestCase):
    def test_layout_and_annotation_mask(self):
        meta=dict(width=2560,height=1006,overview=dict(top=.0546875,bottom=.7109375),
                  regions=[dict(x=.28,y=.28,w=.2,h=.2),dict(x=.335,y=.335,w=.2,h=.2)])
        self.assertEqual(layout(meta),(640,420,51))
        mask,centers=valid_mask(meta)
        self.assertFalse(mask[150,180])  # A's left border
        self.assertFalse(mask[120,190])  # A's colored label
        self.assertTrue(mask[350,500])   # untouched image interior
        self.assertTrue(np.all(~centers | mask))

    def test_overlay_does_not_change_masked_metrics(self):
        rng=np.random.default_rng(5)
        ref=rng.integers(40,210,size=(64,64,3),dtype=np.uint8)
        pred=np.clip(ref.astype(int)+3,0,255).astype('uint8')
        mask=np.ones((64,64),bool);mask[20:30,20:30]=False
        from scipy.ndimage import binary_erosion
        centers=binary_erosion(mask,structure=np.ones((11,11)),border_value=0)
        before=masked_scores(ref,pred,mask,centers)
        pred[20:30,20:30]=[255,0,0]
        after=masked_scores(ref,pred,mask,centers)
        for key in before:self.assertAlmostEqual(before[key],after[key],places=6)


if __name__=='__main__':unittest.main()
