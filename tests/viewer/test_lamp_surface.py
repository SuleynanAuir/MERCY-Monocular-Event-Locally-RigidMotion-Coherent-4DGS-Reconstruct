import unittest
import numpy as np
from utils.fan_surface import attributes
from utils.lamp_surface import reconstruct
from tests.fan.test_fan_surface import plane


class LampSurfaceTests(unittest.TestCase):
    def test_subdivision_preserves_surface_moments_and_part_identity(self):
        source=plane()[:24];groups=np.arange(24,dtype=np.int16)%2
        child,parents,labels,report=reconstruct(source,source,groups,groups,split=4)
        np.testing.assert_array_equal(labels,groups[parents])
        x,b,s,a,_=attributes(source);cx,cb,cs,ca,_=attributes(child)
        for i in range(len(source)):
            ids=np.flatnonzero(parents==i)
            np.testing.assert_allclose(cx[ids].mean(0),x[i],atol=1e-8)
            original=b[i]@np.diag(s[i]**2)@b[i].T
            mixed=sum(cb[j]@np.diag(cs[j]**2)@cb[j].T+np.outer(cx[j]-x[i],cx[j]-x[i]) for j in ids)/4
            np.testing.assert_allclose(mixed,original,atol=1e-10)
            mass=(-np.log1p(-ca[ids])*np.prod(np.sort(cs[ids])[:,1:],axis=1)).sum()
            expected=-np.log1p(-min(a[i],.9999))*np.prod(np.sort(s[i])[1:])
            self.assertAlmostEqual(mass/expected,1,places=5)

    def test_unreliable_or_other_part_colors_do_not_transfer(self):
        source=plane();other=source.copy();groups=np.zeros(len(source),np.int16)
        other['f_dc_0']+=.2;other['opacity']=-15
        result,parents,_,_=reconstruct(source,other,groups,groups,split=1)
        np.testing.assert_array_equal(result['f_dc_0'],source['f_dc_0'])
        other['opacity']=4
        result,_,_,_=reconstruct(source,other,groups,groups+1,split=1)
        np.testing.assert_array_equal(result['f_dc_0'],source['f_dc_0'])

    def test_invalid_input_rejected(self):
        source=plane();groups=np.zeros(len(source),np.int16)
        with self.assertRaises(ValueError):reconstruct(source,source,groups,groups,split=3)
        with self.assertRaises(ValueError):reconstruct(source,source,groups[:-1],groups)
        broken=source.copy();broken['x'][0]=np.nan
        with self.assertRaises(ValueError):reconstruct(broken,source,groups,groups)


if __name__=='__main__':unittest.main()
