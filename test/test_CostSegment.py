import unittest
import lmlib as lm
import numpy as np

__all__ = ["TestCostSegment"]

class TestCostSegment(unittest.TestCase):

    def test_CostSegment(self):
        alssm = lm.AlssmPoly(poly_degree=1, label='poly')
        segment = lm.Segment(a=-3, b=-1, direction=lm.BW, g=10)
        cost = lm.CostSegment(alssm, segment)
        self.assertTrue(isinstance(cost, lm.CostSegment))
        self.assertTrue(cost.get_alssm_order() == alssm.N)
        self.assertTrue(cost.get_alssm_output_dimension() == 0)
        self.assertTrue((cost.eval_alssm_output([[1, 0], [2, 0], [3, 0]]) == np.array([1, 2, 3])).all())
        self.assertTrue(cost.get_state_var_indices('poly.x0') == (0,))