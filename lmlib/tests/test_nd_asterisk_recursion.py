"""
Tests for the ND asterisk recursion bug fix in lmlib.

Three bugs were present in _nd_xi_q_asterisk_l_recursion (and one in
_nd_xi_q_recursion) that collectively caused eval_errors to return zero or
garbage whenever per-dimension polynomial degrees differed, or whenever more
than two spatial dimensions were used.

Bug 1 – wrong Kronecker stride (rls.py, scatter-write in q==1 branch)
    The flat index used `n_prev * Nq_prev + n_curr` instead of
    `n_prev * N + n_curr`, where N is the total order of the *current*
    dimension and Nq_prev is the accumulated trailing size from all
    previous dimensions.  They coincide only when N == Nq_prev, which
    happens to be true for 2-D equal-order filters (e.g. pd=2×pd=2).

Bug 2 – spurious m_prev loop (rls.py, q==1 branch)
    The inner loop iterated over the *current* dimension's ALSSM offsets
    for both m_curr and m_prev, re-slicing xi_prev as though it were
    partitioned by the current-dimension ALSSMs.  xi_prev is an already-
    accumulated monolithic block of size Nq_prev and must not be re-sliced.
    For single-ALSSM sub-costs (M=1) this produced xi_tmp of the wrong
    size (N_curr×N_curr instead of Nq_prev×N_curr), causing either a shape
    error or a silent misalignment identical to Bug 1.

Bug 3 – reshape-copies-not-views (_nd_xi_q_recursion, line ~1092)
    xi_curr was allocated with order='F'.  The subsequent moveaxis(…,-2) +
    reshape(…) chain is non-contiguous for any spatial signal with 3+
    dimensions and any model_dimension other than the outermost axis, so
    numpy.reshape silently returned a *copy*.  Writes by the inner
    xi_q_recursion loop never propagated back to xi_curr, leaving the
    whole array zero.  Fix: use ascontiguousarray before reshape and copy
    the working buffer back to xi_curr after the loop.

All three bugs are simultaneously triggered as soon as either condition is
met:
  • two or more spatial dimensions with *different* per-dimension polynomial
    degrees (triggers Bug 1 + Bug 2), OR
  • three or more spatial dimensions of any degree (triggers Bug 3, which
    also silences Bug 1+2).

The 2-D equal-order case (pd=k × pd=k) happened to pass on the unfixed code
because:
  • Bug 3: a 2-D signal has only one non-trivial "other" axis, so the
    moveaxis+reshape chain can sometimes remain a view.
  • Bug 1+2: N == Nq_prev coincidentally when both dimensions have equal
    order, so the wrong stride and wrong slice are accidentally correct.
"""

import unittest
import numpy as np
import lmlib as lm

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_cost(poly_degree, g=5):
    """One forward + one backward segment CompositeCost wrapping AlssmPoly."""
    alssm = lm.AlssmPoly(poly_degree=poly_degree)
    seg_fw = lm.Segment(a=-4, b=-1, direction=lm.FW, g=g, delta=0)
    seg_bw = lm.Segment(a=0,  b=4,  direction=lm.BW, g=g, delta=0)
    return lm.CompositeCost([alssm], [seg_fw, seg_bw], F=[[1, 1]])


def _run_nd(costs, Y, dim_order, backend='numpy'):
    """Filter Y with an NDCompositeCost built from *costs*, return eval_errors."""
    nd = lm.NDCompositeCost(costs)
    rls = lm.RLSAlssm(nd, steady_state=True, backend=backend)
    rls.filter(Y, dim_order=dim_order)
    return rls.eval_errors(rls.minimize_x())


# Fixed random seed so expected values are deterministic across runs.
_RNG = np.random.default_rng(42)


def _Y(shape):
    """Draw a standard-normal array of *shape* from the fixed RNG."""
    return _RNG.standard_normal(shape)


# ---------------------------------------------------------------------------
# Pre-computed reference values (obtained from the *fixed* library)
# ---------------------------------------------------------------------------

# 2-D, pd = 1 + 2, shape (15, 15)
_Y12 = _Y((15, 15))
_E12_ROW0 = [
    9.992930587808743, 11.355275368239962, 12.076972464683074,
    13.478136811037054, 14.0061462345366, 12.063385761973043,
    11.319148159515684, 10.773242961339262, 10.04735400875207,
    9.054582713276499, 8.38240389882399, 8.132747404518033,
    6.9618234576244316, 6.000588055276984, 4.630839002972176,
]
_E12_77 = 23.937775766491953

# 3-D, pd = 1 + 2 + 3, shape (10, 10, 10)
_Y3 = _Y((10, 10, 10))
_E3_ROW000 = [
    43.26491371078278, 55.63403610317164, 61.744012322811,
    66.76065485184083, 71.82561610185154, 71.07510755738141,
    62.55618003350374, 56.832906955984704, 49.6003754937245,
    40.131894320309804,
]
_E3_555 = 179.54505841688405

# 4-D, pd = 2 + 2 + 2 + 2, shape (8, 8, 8, 8)
_Y4 = _Y((8, 8, 8, 8))
_E4_ROW0000 = [
    126.902251103053, 157.79741047485874, 187.00786532547133,
    207.71274851755604, 211.1618114771566, 191.3651587531189,
    166.72241591081254, 133.21945248416077,
]
_E4_4444 = 786.0971708620782


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestNDCostAsteriskRecursion(unittest.TestCase):
    """
    Regression tests for the three bugs in the ND asterisk recursion.

    Each test follows the same structure:
      1. Construct an NDCompositeCost from sub-costs of specified degrees.
      2. Filter a fixed random signal.
      3. Assert that eval_errors is strictly positive (non-zero ↔ fit is
         actually happening) and, where pre-computed, matches reference values.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assert_positive_finite(self, errors, label=''):
        msg = f'{label}: '.rstrip(': ')
        self.assertTrue(
            np.all(np.isfinite(errors)),
            f'{msg}eval_errors contains non-finite values',
        )
        self.assertTrue(
            np.all(errors > 0),
            f'{msg}eval_errors contains zeros (bug not fixed — xs = 0)',
        )

    def _assert_allclose(self, actual, expected, label='', rtol=1e-10):
        np.testing.assert_allclose(
            actual, expected, rtol=rtol,
            err_msg=f'Numerical mismatch in {label}',
        )

    # ------------------------------------------------------------------
    # 2-D equal-order (regression guard — must stay passing after fix)
    # ------------------------------------------------------------------

    def test_2d_equal_order_pd1_numpy(self):
        """2-D pd=1×1: equal-order guard — was accidentally correct before fix."""
        K = 15
        y = np.sin(np.linspace(0, 2 * np.pi, K))
        Y = np.repeat([y], K, axis=0)
        e = _run_nd(
            [_make_cost(1, g=10), _make_cost(1, g=10)],
            Y, dim_order=[0, 1], backend='numpy',
        )
        self._assert_positive_finite(e, '2D pd=1×1 numpy')

    def test_2d_equal_order_pd1_lfilter(self):
        """2-D pd=1×1: lfilter backend must agree with numpy."""
        K = 15
        y = np.sin(np.linspace(0, 2 * np.pi, K))
        Y = np.repeat([y], K, axis=0)
        e_np = _run_nd([_make_cost(1, g=10), _make_cost(1, g=10)], Y, [0, 1], 'numpy')
        e_lf = _run_nd([_make_cost(1, g=10), _make_cost(1, g=10)], Y, [0, 1], 'lfilter')
        self._assert_allclose(e_lf, e_np, '2D pd=1×1 lfilter vs numpy')

    def test_2d_equal_order_pd2_numpy(self):
        """2-D pd=2×2: the classic hiding case — equal order masks both bugs."""
        K = 12
        Y = np.random.default_rng(0).standard_normal((K, K))
        e = _run_nd([_make_cost(2)] * 2, Y, [0, 1], 'numpy')
        self._assert_positive_finite(e, '2D pd=2×2 numpy')

    # ------------------------------------------------------------------
    # 2-D mixed order (Bug 1 + Bug 2)
    # ------------------------------------------------------------------

    def test_2d_mixed_pd1_pd2_numpy(self):
        """
        2-D pd=1+2, numpy backend.

        Before the fix: eval_errors ≈ 0 everywhere (xs = 0) because the
        wrong Kronecker stride scattered xi1 values into the wrong slots,
        making the Gram matrix appear to have no correlation with the data.
        """
        e = _run_nd([_make_cost(1), _make_cost(2)], _Y12, [0, 1], 'numpy')
        self._assert_positive_finite(e, '2D pd=1+2 numpy')
        self._assert_allclose(e[0], _E12_ROW0, '2D pd=1+2 numpy row-0')
        self.assertAlmostEqual(float(e[7, 7]), _E12_77, places=8,
                               msg='2D pd=1+2 numpy e[7,7]')

    def test_2d_mixed_pd1_pd2_lfilter(self):
        """2-D pd=1+2, lfilter backend — must agree with numpy reference."""
        e = _run_nd([_make_cost(1), _make_cost(2)], _Y12, [0, 1], 'lfilter')
        self._assert_positive_finite(e, '2D pd=1+2 lfilter')
        self._assert_allclose(e[0], _E12_ROW0, '2D pd=1+2 lfilter row-0', rtol=1e-6)

    def test_2d_mixed_pd2_pd1_numpy(self):
        """
        2-D pd=2+1 (reversed): also exercises mixed-order path.

        Verifies that the fix works regardless of which dimension carries
        the higher degree.
        """
        Y = np.random.default_rng(1).standard_normal((12, 12))
        e = _run_nd([_make_cost(2), _make_cost(1)], Y, [0, 1], 'numpy')
        self._assert_positive_finite(e, '2D pd=2+1 numpy')

    def test_2d_mixed_pd1_pd3_numpy(self):
        """2-D pd=1+3: larger degree gap stresses the stride calculation."""
        Y = np.random.default_rng(2).standard_normal((12, 12))
        e = _run_nd([_make_cost(1), _make_cost(3)], Y, [0, 1], 'numpy')
        self._assert_positive_finite(e, '2D pd=1+3 numpy')

    # ------------------------------------------------------------------
    # 3-D mixed order (Bug 1 + Bug 2 + Bug 3)
    # ------------------------------------------------------------------

    def test_3d_mixed_pd1_pd2_pd3_numpy(self):
        """
        3-D pd=1+2+3, numpy backend.

        Before the fix: eval_errors = 0 everywhere.  Bug 3 (order='F'
        + moveaxis + reshape creates a copy) meant xi1 was never filled
        at all; Bugs 1 and 2 would have corrupted the result even if Bug 3
        had been absent.
        """
        e = _run_nd(
            [_make_cost(1), _make_cost(2), _make_cost(3)],
            _Y3, [0, 1, 2], 'numpy',
        )
        self._assert_positive_finite(e, '3D pd=1+2+3 numpy')
        self._assert_allclose(e[0, 0], _E3_ROW000, '3D pd=1+2+3 numpy row-[0,0]')
        self.assertAlmostEqual(float(e[5, 5, 5]), _E3_555, places=7,
                               msg='3D pd=1+2+3 numpy e[5,5,5]')

    def test_3d_mixed_pd1_pd2_pd3_lfilter(self):
        """3-D pd=1+2+3, lfilter backend — must agree with numpy reference."""
        e = _run_nd(
            [_make_cost(1), _make_cost(2), _make_cost(3)],
            _Y3, [0, 1, 2], 'lfilter',
        )
        self._assert_positive_finite(e, '3D pd=1+2+3 lfilter')
        self._assert_allclose(e[0, 0], _E3_ROW000, '3D pd=1+2+3 lfilter row-[0,0]', rtol=1e-6)

    def test_3d_mixed_dim_order_permutation(self):
        """
        3-D pd=1+2+3: processing dim_order=[0,1,2] vs [2,1,0] must give the
        same eval_errors (the cost is separable and symmetric in processing
        order).
        """
        costs = [_make_cost(1), _make_cost(2), _make_cost(3)]
        Y = np.random.default_rng(3).standard_normal((8, 8, 8))
        e_012 = _run_nd(costs, Y, [0, 1, 2], 'numpy')
        e_210 = _run_nd(costs, Y, [2, 1, 0], 'numpy')
        self._assert_allclose(e_012, e_210, '3D dim_order permutation invariance')

    # ------------------------------------------------------------------
    # 4-D equal-order (Bug 3 alone triggers, even with equal degrees)
    # ------------------------------------------------------------------

    def test_4d_equal_order_pd2_numpy(self):
        """
        4-D pd=2×4, numpy backend.

        Before the fix this also returned zeros: even though all degrees
        are equal (so Bug 1 and Bug 2 cancel), Bug 3 still leaves xi_curr
        all-zero for any 4-D signal processed along a non-outermost axis.
        """
        e = _run_nd([_make_cost(2)] * 4, _Y4, [0, 1, 2, 3], 'numpy')
        self._assert_positive_finite(e, '4D pd=2×4 numpy')
        self._assert_allclose(e[0, 0, 0], _E4_ROW0000, '4D pd=2×4 numpy row-[0,0,0]')
        self.assertAlmostEqual(float(e[4, 4, 4, 4]), _E4_4444, places=6,
                               msg='4D pd=2×4 numpy e[4,4,4,4]')

    def test_4d_equal_order_pd2_lfilter(self):
        """4-D pd=2×4, lfilter backend — must agree with numpy reference."""
        e = _run_nd([_make_cost(2)] * 4, _Y4, [0, 1, 2, 3], 'lfilter')
        self._assert_positive_finite(e, '4D pd=2×4 lfilter')
        self._assert_allclose(e[0, 0, 0], _E4_ROW0000, '4D pd=2×4 lfilter row-[0,0,0]', rtol=1e-6)

    # ------------------------------------------------------------------
    # Output shape sanity checks
    # ------------------------------------------------------------------

    def test_xs_trailing_size_equals_kronecker_product(self):
        """
        xs.shape[-1] must equal the Kronecker product of per-dimension orders.

        For AlssmPoly(pd) the state order is N = pd + 1, and for q=1 the
        trailing axis of xs must be N_0 * N_1 * … * N_{L-1}.
        """
        # 2-D pd=1+2: N_total = 2 * 3 = 6
        nd = lm.NDCompositeCost([_make_cost(1), _make_cost(2)])
        rls = lm.RLSAlssm(nd, steady_state=True, backend='numpy')
        rls.filter(_Y12, dim_order=[0, 1])
        xs = rls.minimize_x()
        self.assertEqual(xs.shape[-1], 2 * 3,
                         'xs trailing size for pd=1+2 must be 6')

        # 3-D pd=1+2+3: N_total = 2 * 3 * 4 = 24
        nd3 = lm.NDCompositeCost([_make_cost(1), _make_cost(2), _make_cost(3)])
        rls3 = lm.RLSAlssm(nd3, steady_state=True, backend='numpy')
        rls3.filter(_Y3, dim_order=[0, 1, 2])
        xs3 = rls3.minimize_x()
        self.assertEqual(xs3.shape[-1], 2 * 3 * 4,
                         'xs trailing size for pd=1+2+3 must be 24')

        # 4-D pd=2×4: N_total = 3^4 = 81
        nd4 = lm.NDCompositeCost([_make_cost(2)] * 4)
        rls4 = lm.RLSAlssm(nd4, steady_state=True, backend='numpy')
        rls4.filter(_Y4, dim_order=[0, 1, 2, 3])
        xs4 = rls4.minimize_x()
        self.assertEqual(xs4.shape[-1], 3 ** 4,
                         'xs trailing size for pd=2×4 must be 81')

    # ------------------------------------------------------------------
    # Backend consistency (numpy == lfilter to machine precision)
    # ------------------------------------------------------------------

    def test_backends_agree_2d_mixed(self):
        """numpy and lfilter backends must produce identical eval_errors for 2-D pd=1+2."""
        e_np = _run_nd([_make_cost(1), _make_cost(2)], _Y12, [0, 1], 'numpy')
        e_lf = _run_nd([_make_cost(1), _make_cost(2)], _Y12, [0, 1], 'lfilter')
        np.testing.assert_allclose(e_np, e_lf, rtol=1e-10, atol=1e-10,
                                   err_msg='numpy vs lfilter mismatch for 2D pd=1+2')

    def test_backends_agree_3d_mixed(self):
        """numpy and lfilter backends must produce identical eval_errors for 3-D pd=1+2+3."""
        e_np = _run_nd([_make_cost(1), _make_cost(2), _make_cost(3)],
                       _Y3, [0, 1, 2], 'numpy')
        e_lf = _run_nd([_make_cost(1), _make_cost(2), _make_cost(3)],
                       _Y3, [0, 1, 2], 'lfilter')
        np.testing.assert_allclose(e_np, e_lf, rtol=1e-8, atol=1e-8,
                                   err_msg='numpy vs lfilter mismatch for 3D pd=1+2+3')

    def test_backends_agree_4d_equal(self):
        """numpy and lfilter backends must produce identical eval_errors for 4-D pd=2×4."""
        e_np = _run_nd([_make_cost(2)] * 4, _Y4, [0, 1, 2, 3], 'numpy')
        e_lf = _run_nd([_make_cost(2)] * 4, _Y4, [0, 1, 2, 3], 'lfilter')
        np.testing.assert_allclose(e_np, e_lf, rtol=1e-8, atol=1e-8,
                                   err_msg='numpy vs lfilter mismatch for 4D pd=2×4')

    # ------------------------------------------------------------------
    # dim_order permutation invariance
    # ------------------------------------------------------------------

    def test_dim_order_invariance_2d(self):
        """
        For 2-D pd=1+2, dim_order=[0,1] and dim_order=[1,0] must yield the
        same eval_errors (the cost is separable so processing order is
        irrelevant).
        """
        Y = np.random.default_rng(4).standard_normal((12, 12))
        e01 = _run_nd([_make_cost(1), _make_cost(2)], Y, [0, 1], 'numpy')
        e10 = _run_nd([_make_cost(1), _make_cost(2)], Y, [1, 0], 'numpy')
        np.testing.assert_allclose(
            e01, e10, rtol=1e-10,
            err_msg='dim_order=[0,1] vs [1,0] mismatch for 2D pd=1+2',
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
