import unittest
import lmlib as lm
import numpy as np

__all__ = ["TestCompositeCost"]


class TestCompositeCost(unittest.TestCase):
    """Regression tests for single ALSSM / single Segment support in CompositeCost."""

    def _alssm(self, degree=2, label='poly'):
        return lm.AlssmPoly(poly_degree=degree, label=label)

    def _segment(self, a=0, b=20):
        return lm.Segment(a=a, b=b, direction=lm.BW, g=10)

    # ------------------------------------------------------------------
    # Accepting single (non-iterable) ALSSM / Segment arguments
    # ------------------------------------------------------------------

    def test_single_alssm_single_segment_no_F(self):
        """A single ALSSM and a single Segment; F defaults to ones((1, 1))."""
        cost = lm.CompositeCost(self._alssm(), self._segment())
        self.assertTrue(isinstance(cost, lm.CompositeCost))
        self.assertEqual(cost.M, 1)
        self.assertEqual(cost.P, 1)
        self.assertEqual(cost.F.shape, (1, 1))
        np.testing.assert_array_equal(cost.F, np.ones((1, 1)))

    def test_single_alssm_multiple_segments_no_F(self):
        """A single ALSSM with several Segments; F defaults to ones((1, P))."""
        cost = lm.CompositeCost(self._alssm(), [self._segment(0, 20), self._segment(-20, -1)])
        self.assertEqual(cost.M, 1)
        self.assertEqual(cost.P, 2)
        np.testing.assert_array_equal(cost.F, np.ones((1, 2)))

    def test_multiple_alssms_single_segment_no_F(self):
        """Several ALSSMs with a single Segment; F defaults to ones((M, 1))."""
        cost = lm.CompositeCost([self._alssm(2), self._alssm(1)], self._segment())
        self.assertEqual(cost.M, 2)
        self.assertEqual(cost.P, 1)
        np.testing.assert_array_equal(cost.F, np.ones((2, 1)))

    def test_single_elements_wrapped_in_lists_still_work(self):
        """Passing single elements as length-1 lists keeps working, F defaults to ones."""
        cost = lm.CompositeCost([self._alssm()], [self._segment()])
        self.assertEqual((cost.M, cost.P), (1, 1))
        np.testing.assert_array_equal(cost.F, np.ones((1, 1)))

    # ------------------------------------------------------------------
    # F handling
    # ------------------------------------------------------------------

    def test_explicit_F_is_respected_for_single_alssm(self):
        """An explicitly passed F must not be overwritten by the default."""
        cost = lm.CompositeCost(self._alssm(), [self._segment(0, 20), self._segment(-20, -1)], F=[[2, 3]])
        np.testing.assert_array_equal(cost.F, np.array([[2, 3]]))

    def test_F_mandatory_for_two_by_two(self):
        """With >=2 ALSSMs and >=2 Segments and no F, construction must fail."""
        alssms = [self._alssm(2), self._alssm(1)]
        segments = [self._segment(0, 20), self._segment(-20, -1)]
        with self.assertRaises(AssertionError):
            lm.CompositeCost(alssms, segments)

    def test_two_by_two_with_F_ok(self):
        """With >=2 ALSSMs and >=2 Segments, an explicit F constructs fine."""
        alssms = [self._alssm(2), self._alssm(1)]
        segments = [self._segment(0, 20), self._segment(-20, -1)]
        cost = lm.CompositeCost(alssms, segments, F=[[1, 0], [0, 1]])
        self.assertEqual(cost.F.shape, (2, 2))

    # ------------------------------------------------------------------
    # End-to-end equivalence with CostSegment (the motivating example)
    # ------------------------------------------------------------------

    def test_single_composite_matches_cost_segment(self):
        """A single-element CompositeCost must reproduce the equivalent CostSegment fit."""
        np.random.seed(0)
        K = 200
        y = np.random.randn(K)

        alssm = self._alssm(degree=2)
        segment = lm.Segment(0, 30, lm.BW, 10)

        rls_cs = lm.RLSAlssm(lm.CostSegment(alssm, segment))
        rls_cs.filter(y)
        xs_cs = rls_cs.minimize_x(solver='lstsq')

        rls_cc = lm.RLSAlssm(lm.CompositeCost(alssm, segment))
        rls_cc.filter(y)
        xs_cc = rls_cc.minimize_x(solver='lstsq')

        np.testing.assert_allclose(xs_cc, xs_cs, rtol=1e-9, atol=1e-9, equal_nan=True)


if __name__ == '__main__':
    unittest.main()
