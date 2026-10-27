import unittest
import numpy as np
from utils.appearance_stage import validate_attribute


class AppearanceStageTests(unittest.TestCase):
    def test_preserve_existing_exact_zero_opacity(self):
        parent=np.array([[-np.inf],[np.inf],[0.]],dtype=np.float32)
        validate_attribute('opacity',np.array([[-np.inf],[np.inf],[.5]],dtype=np.float32),parent)

    def test_new_infinity_nan_or_nonopacity_infinity_rejected(self):
        parent=np.zeros((2,1),dtype=np.float32)
        for value in (-np.inf,np.inf,np.nan):
            with self.assertRaises(ValueError):validate_attribute('opacity',np.full_like(parent,value),parent)
        with self.assertRaises(ValueError):validate_attribute('scaling',np.full_like(parent,-np.inf),np.full_like(parent,-np.inf))

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):validate_attribute('opacity',np.zeros((1,2),np.float32),np.zeros((2,1),np.float32))
