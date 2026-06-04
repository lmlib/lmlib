r"""
Tests for the lfilter *parallel* backend on N-dimensional cost functions.

The parallel filter form realises both the first-dimension $\xi^{(1)}$
recursion and the cross-dimensional (asterisk) step via per-ALSSM transfer
functions (Option A).  The asterisk step uses a per-ALSSM-block split followed
by a Kronecker scatter (``lfilter_parallel_xi_asterisk_split``); ``q==0`` (kappa)
and ``q==2`` (W) use the cascade / numpy realizations.

These tests pin the end-to-end behaviour — ``filter_form='parallel'`` must agree
with the numpy reference on ND costs, including different ALSSMs per dimension —
and that the genuine parallel path is taken (no silent numpy fallback).
"""

import unittest
import numpy as np
import lmlib as lm

TOL = 1e-6


def _cc(poly_degrees, g=80):
    """CompositeCost of one fw + one bw segment over the given poly ALSSMs."""
    alssms = [lm.AlssmPoly(poly_degree=d) for d in poly_degrees]
    segs = [lm.Segment(a=-25, b=-1, direction=lm.FW, g=g),
            lm.Segment(a=0, b=25, direction=lm.BW, g=g)]
    F = np.ones((len(poly_degrees), 2))
    return lm.CompositeCost(alssms, segs, F)


class TestNDParallel(unittest.TestCase):

    def _compare(self, nd, Y, msg):
        ref = lm.RLSAlssm(nd, steady_state=True, backend='numpy')
        ref.filter(Y)
        par = lm.RLSAlssm(nd, steady_state=True,
                          backend='lfilter', filter_form='parallel')
        par.filter(Y)
        dxi = np.max(np.abs(ref.xi - par.xi))
        self.assertLess(dxi, TOL, f"{msg}: xi mismatch (Δ={dxi:.2e})")
        # W is identical (closed-form steady-state, backend-independent)
        dW = np.max(np.abs(ref.W - par.W))
        self.assertLess(dW, TOL, f"{msg}: W mismatch (Δ={dW:.2e})")
        # downstream estimate must agree as well
        dx = np.max(np.abs(ref.minimize_x() - par.minimize_x()))
        self.assertLess(dx, TOL, f"{msg}: minimize_x mismatch (Δ={dx:.2e})")

    def test_2d_equal_order_single_alssm(self):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((18, 22)) * 0.1
        Y[6:12, 8:16] += 1.0
        nd = lm.NDCompositeCost([_cc([1]), _cc([1])])
        self._compare(nd, Y, "2-D pd1 x pd1")

    def test_2d_mixed_order_single_alssm(self):
        rng = np.random.default_rng(1)
        Y = rng.standard_normal((18, 22)) * 0.1
        Y[6:12, 8:16] += 1.0
        nd = lm.NDCompositeCost([_cc([1]), _cc([3])])
        self._compare(nd, Y, "2-D pd1 x pd3")

    def test_2d_multi_alssm_per_dim(self):
        """CompositeCost with several ALSSMs per dimension (the case the
        per-ALSSM split / block handling must get right)."""
        rng = np.random.default_rng(2)
        Y = rng.standard_normal((16, 20)) * 0.1
        Y[5:11, 7:15] += 1.0
        nd = lm.NDCompositeCost([_cc([1, 2]), _cc([2, 1])])
        self._compare(nd, Y, "2-D (pd1,pd2) x (pd2,pd1)")

    def test_3d_mixed(self):
        rng = np.random.default_rng(3)
        Y = rng.standard_normal((9, 10, 11)) * 0.1
        Y[2:6, 3:7, 4:8] += 1.0
        nd = lm.NDCompositeCost([_cc([1]), _cc([2]), _cc([1])])
        self._compare(nd, Y, "3-D pd1 x pd2 x pd1")

    def test_2d_multi_alssm_both_dims(self):
        rng = np.random.default_rng(4)
        Y = rng.standard_normal((16, 20)) * 0.1
        Y[5:11, 7:15] += 1.0
        nd = lm.NDCompositeCost([_cc([1, 3]), _cc([2, 2])])
        self._compare(nd, Y, "2-D (pd1,pd3) x (pd2,pd2)")

    def test_parallel_asterisk_uses_no_numpy_fallback(self):
        """The genuine parallel asterisk must run without silently falling
        back to the numpy recursion (which would mask a broken parallel path)."""
        import lmlib.statespace.backends.rec as R
        rng = np.random.default_rng(5)
        Y = rng.standard_normal((16, 20)) * 0.1
        Y[5:11, 7:15] += 1.0
        nd = lm.NDCompositeCost([_cc([1, 3]), _cc([2, 2])])

        ref = lm.RLSAlssm(nd, steady_state=True, backend='numpy')
        ref.filter(Y)

        orig = R.numpy_xi_asterisk_l_recursion
        calls = {'n': 0}

        def guard(*a, **k):
            calls['n'] += 1
            return orig(*a, **k)

        R.numpy_xi_asterisk_l_recursion = guard
        try:
            par = lm.RLSAlssm(nd, steady_state=True,
                              backend='lfilter', filter_form='parallel')
            par.filter(Y)
        finally:
            R.numpy_xi_asterisk_l_recursion = orig

        self.assertEqual(calls['n'], 0,
                         "parallel asterisk silently fell back to numpy")
        dxi = np.max(np.abs(ref.xi - par.xi))
        self.assertLess(dxi, TOL, f"xi mismatch (Δ={dxi:.2e})")


if __name__ == '__main__':
    unittest.main()
