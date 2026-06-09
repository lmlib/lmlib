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
    xs = np.random.randn(K, cost.get_alssm_order())
    ks = np.random.randint(0, K, 1)

    def test_get_local(self):
        trajs_test = lm.Trajectory.eval(self.cost, self.xs[self.ks],merged_seg=False, merged_ks=False)
        trajs_true = np.array([[
            (np.arange(-3, 0),np.array([-1.97069648, -1.49371147, -1.01672646]))],
            [(np.arange(0, 4), np.array([3.24894392, 2.2277164 , 1.20648887, 0.18526135]))]], dtype=tuple)
        for ttrue_ks, ttest_ks in zip(trajs_true, trajs_test):
            for ttrue, ttest in zip(ttrue_ks, ttest_ks):
                self.assertTrue(np.isclose(ttrue[0], ttest[0]).all())
                self.assertTrue(np.isclose(ttrue[1], ttest[1]).all())

    def test_get_mapped(self):
        trajs_test = lm.Trajectory.eval_y(self.cost, self.xs[self.ks], self.ks, self.K, merged_seg=False, merged_ks=False)
        trajs_true = np.array([[[np.nan, np.nan]], [[np.nan, np.nan]], [[-1.9706964767585613, np.nan]], [[-1.4937114663462618, np.nan]], [[-1.0167264559339624, np.nan]], [[np.nan, 3.248943919430755]], [[np.nan, 2.227716395075158]], [[np.nan, 1.2064888707195611]], [[np.nan, 0.1852613463639643]], [[np.nan, np.nan]]]).T
        self.assertTrue(np.isclose(trajs_test, trajs_true, equal_nan=True).all())


class TestTrajectoryND(unittest.TestCase):
    """Regression tests for Trajectory.eval on an NDCompositeCost.

    A 2-D cost built from a forward + backward window per dimension has
    per-segment offset ranges of *different* length (e.g. -20..-1 vs 0..20).
    The merged ND trajectory must tile the four segment-combination quadrants
    onto the full union offset grid rather than element-wise maxing them.
    """

    def _build(self, a_l, b_l, a_r, b_r, L1=80):
        alssm = lm.AlssmSum((lm.AlssmPoly(0),
                             lm.AlssmSin(2 * np.pi / L1),
                             lm.AlssmSin(2 * np.pi / (0.5 * L1))))
        seg_l = lm.Segment(a=a_l, b=b_l, direction=lm.FORWARD, g=100)
        seg_r = lm.Segment(a=a_r, b=b_r, direction=lm.BACKWARD, g=100)
        cost_1d = lm.CompositeCost((alssm,), (seg_l, seg_r), [[1, 1]])
        return alssm, lm.NDCompositeCost([cost_1d, cost_1d])

    def test_merged_matches_separable_reference(self):
        """Asymmetric windows (lengths 20 vs 21): merged eval == Phi @ Xc @ Phi.T."""
        half = 20
        alssm, nd = self._build(-half, -1, 0, half)
        N = alssm.N
        rng = np.random.default_rng(0)
        state = rng.standard_normal(N * N)

        ab, values = lm.Trajectory.eval(nd, state)          # must not raise
        self.assertEqual(values.shape, (2 * half + 1, 2 * half + 1))
        self.assertFalse(np.isnan(values).any())            # full grid covered
        for a in ab:
            self.assertTrue(np.array_equal(a, np.arange(-half, half + 1)))

        offs = np.arange(-half, half + 1)
        Phi = np.stack([alssm.C @ np.linalg.matrix_power(alssm.A, int(o)) for o in offs], axis=0)
        ref = Phi @ state.reshape(N, N) @ Phi.T
        self.assertTrue(np.allclose(values, ref, atol=1e-9))

    def test_unmerged_returns_per_combo(self):
        """merged_seg=False keeps the documented per-combo object array."""
        alssm, nd = self._build(-20, -1, 0, 20)
        state = np.random.default_rng(1).standard_normal(alssm.N ** 2)
        out = lm.Trajectory.eval(nd, state, merged_seg=False, merged_ks=False)
        self.assertEqual(out.shape, (4, 1))                 # 2 segments^2 combos, 1 anchor
        self.assertIsInstance(out.flat[0], tuple)

    def test_offset_zero_excluded_when_uncovered(self):
        """Windows skipping offset 0 yield a union grid without it."""
        half = 20
        alssm, nd = self._build(-half, -1, 1, half)         # neither segment covers 0
        state = np.random.default_rng(2).standard_normal(alssm.N ** 2)
        ab, values = lm.Trajectory.eval(nd, state)
        self.assertEqual(values.shape, (2 * half, 2 * half))
        self.assertNotIn(0, ab[0])


if __name__ == "__main__":
    unittest.main()
