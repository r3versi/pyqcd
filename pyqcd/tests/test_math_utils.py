import sys
import unittest

import numpy as np

from pyqcd.gates import U3
from pyqcd.math_utils import d1, d2, tr_distance


class TestMathUtils(unittest.TestCase):
    def test_tr_distance(self):
        params = np.random.rand(3)
        a = U3(*params).to_matrix()

        params = np.random.rand(3)
        b = U3(*params).to_matrix()

        d_ab = tr_distance(a, b)
        d_aa = tr_distance(a, a)
        d_bb = tr_distance(b, b)

        self.assertTrue(d_ab >= 0 and d_ab <= 1)
        self.assertTrue(np.isclose(d_aa, 0))
        self.assertTrue(np.isclose(d_bb, 0))

    def test_d1_distance(self):
        params = np.random.rand(3)
        a = U3(*params).to_matrix()

        params = np.random.rand(3)
        b = U3(*params).to_matrix()

        d_ab = d1(a, b)
        d_aa = d1(a, a)
        d_bb = d1(b, b)

        self.assertTrue(d_ab >= 0)
        self.assertTrue(np.isclose(d_aa, 0))
        self.assertTrue(np.isclose(d_bb, 0))

    def test_d2_distance(self):
        params = np.random.rand(3)
        a = U3(*params).to_matrix()

        params = np.random.rand(3)
        b = U3(*params).to_matrix()

        d_ab = d2(a, b)
        d_aa = d2(a, a)
        d_bb = d2(b, b)

        self.assertTrue(d_ab >= 0)
        self.assertTrue(np.isclose(d_aa, 0))
        self.assertTrue(np.isclose(d_bb, 0))
