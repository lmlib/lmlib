import unittest
import lmlib as lm
import numpy as np

__all__ = ["TestTrajectory"]

class TestTrajectory(unittest.TestCase):
    np.random.seed(12345)
    alssms = (lm.AlssmPoly(poly_degree=1), lm.AlssmPoly(poly_degree=1))
    segments = (lm.Segment(a=-3, b=-1, direction=lm.BW, g=10), lm.Segment(a=0, b=3, direction=lm.FW, g=10))
    F = np.eye(2)
    cost = lm.CompositeCost(alssms, segments, F)
    K = 10
    xs = np.random.randn(K, cost.get_model_order())
    ks = np.random.randint(0, K, 1)

    def test_get_local(self):
        trajs_test = lm.Trajectory.get_local(self.cost, self.xs[self.ks])
        trajs_true = np.array([[
            (range(-3, 0),np.array([-1.97069648, -1.49371147, -1.01672646]))],
            [(range(0, 4), np.array([3.24894392, 2.2277164 , 1.20648887, 0.18526135]))]], dtype=tuple)
        for ttrue_ks, ttest_ks in zip(trajs_true, trajs_test):
            for ttrue, ttest in zip(ttrue_ks, ttest_ks):
                self.assertTrue(np.isclose(ttrue[0], ttest[0]).all())
                self.assertTrue(np.isclose(ttrue[1], ttest[1]).all())

    def test_get_mapped(self):
        trajs_test = lm.Trajectory.get_mapped(self.cost, self.xs[self.ks], self.ks, self.K, merged_seg=False, merged_ks=False)
        trajs_true = np.array([[[np.nan, np.nan]], [[np.nan, np.nan]], [[-1.9706964767585613, np.nan]], [[-1.4937114663462618, np.nan]], [[-1.0167264559339624, np.nan]], [[np.nan, 3.248943919430755]], [[np.nan, 2.227716395075158]], [[np.nan, 1.2064888707195611]], [[np.nan, 0.1852613463639643]], [[np.nan, np.nan]]]).T
        self.assertTrue(np.isclose(trajs_test, trajs_true, equal_nan=True).all())