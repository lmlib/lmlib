"""
End-to-end pipeline tests for the ND RLS filter: from filter() to minimize_x().

These tests go beyond the existing eval_errors > 0 smoke-tests by verifying that
both W (Gram matrix) and xi (cross-correlation vector) are numerically correct,
using two independent reference methods:

Reference 1 — Known polynomial ground truth (GT)
    The input signal Y is a noiseless separable polynomial:
        Y(i, j, ...) = p_0(i) * p_1(j) * ...
    where each p_d is a polynomial of degree <= pd_d.
    Because the signal lies exactly in the ALSSM model space, the RLS minimizer
    must recover the true Kronecker-product state vector:
        xs[i, j, ...] == kron(x0_true(i), x1_true(j), ...)
    to floating-point precision.  This tests that W and xi are both correct,
    since xs = W^{-1} xi.

Reference 2 — scipy.linalg.lstsq cross-check
    For one interior point, the test solves W @ x = xi[idx] using scipy's
    independent least-squares solver and verifies that lmlib's minimize_x
    returns the same result.  This is a direct numerical check that W and xi
    are mutually consistent.

Configurations tested (all two-segment FW+BW, plus single FW-only variant):
  • 2-D, pd = 1 + 2   (the primary Bug 1 + Bug 2 config)
  • 3-D, pd = 1 + 2 + 3
  • 4-D, pd = 2 + 2 + 2 + 2

AlssmPoly state-vector convention
    AlssmPoly(poly_degree=d) uses the monomial basis:
        y(k + offset) = x[0] + offset*x[1] + offset^2*x[2] + ...
    So x[q] at anchor i is the q-th binomial expansion coefficient of the
    polynomial shifted to origin i:
        x[q](i) = sum_{p >= q} c_p * C(p,q) * i^{p-q}
    (see `true_state_1d` below).
    The ND Kronecker state at (i, j, ...) is kron(x0(i), x1(j), ...).

Window / conditioning notes
    A half-width of 6 samples and g=10 gives gamma ≈ 1.58, which is
    well-conditioned for steady-state W while large enough to fully support
    polynomials up to degree 3.  All assertions use tolerances sized to the
    observed numerical error for these parameters (~1e-4 for 3-D, ~1e-3 for 4-D).
"""

import unittest
import warnings
from math import comb

import numpy as np
from scipy.linalg import lstsq
import lmlib as lm


# ---------------------------------------------------------------------------
# Constants — shared across all tests
# ---------------------------------------------------------------------------

_HW = 6    # half-window width (samples)
_G  = 10   # effective number of samples (controls gamma, must be > 1)

# 1-D polynomial coefficients  [c0, c1, ...] → y(k) = sum c_p k^p
_COEFFS_PD1 = [2.0,  3.0]
_COEFFS_PD2 = [1.0, -1.0,  0.5]
_COEFFS_PD3 = [3.0, -1.0,  0.5, -0.1]
_COEFFS_PD2_ALT = [1.0, -0.5, 0.1]   # used for all four dims in the 4-D test


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def true_state_1d(anchor, coeffs):
    """
    Exact AlssmPoly monomial-basis state vector at *anchor*.

    For a polynomial y(k) = sum_p c_p k^p, the ALSSM state at anchor i
    satisfies y(i + offset) = x[0] + offset*x[1] + offset^2*x[2] + ...,
    so x[q] = sum_{p >= q} c_p C(p,q) i^{p-q}.
    """
    n = len(coeffs)
    x = np.zeros(n)
    for q in range(n):
        for p in range(q, n):
            x[q] += coeffs[p] * comb(p, q) * (float(anchor) ** (p - q))
    return x


def poly_eval(k_array, coeffs):
    """Evaluate polynomial with *coeffs* at every element of *k_array*."""
    return sum(c * k_array ** p for p, c in enumerate(coeffs))


def make_nd_cost(poly_degrees, hw=_HW, g=_G, two_segment=True):
    """
    Build an NDCompositeCost from AlssmPoly sub-costs.

    Parameters
    ----------
    poly_degrees : sequence of int
    hw           : half-window width (int)
    g            : effective number of samples (float > 1)
    two_segment  : if True, use FW + BW segments; if False, use FW only.
    """
    seg_fw = lm.Segment(a=-hw, b=-1, direction=lm.FW, g=g, delta=0)
    costs = []
    for pd in poly_degrees:
        alssm = lm.AlssmPoly(poly_degree=pd)
        if two_segment:
            seg_bw = lm.Segment(a=0, b=hw, direction=lm.BW, g=g, delta=0)
            cost = lm.CompositeCost([alssm], [seg_fw, seg_bw], F=[[1, 1]])
        else:
            cost = lm.CompositeCost([alssm], [seg_fw], F=[[1]])
        costs.append(cost)
    return lm.NDCompositeCost(costs)


def run_nd_rls(Y, poly_degrees, hw=_HW, g=_G, backend='numpy', two_segment=True):
    """Filter *Y*, return (rls_object, minimize_x result)."""
    nd = make_nd_cost(poly_degrees, hw=hw, g=g, two_segment=two_segment)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')   # suppress W-conditioning warnings for large g
        rls = lm.RLSAlssm(nd, steady_state=True, backend=backend)
        rls.filter(Y, dim_order=list(range(nd.L)))
        return rls, rls.minimize_x()


def build_separable_signal(*dim_coeffs):
    """
    Build a separable ND polynomial signal on a grid.

    Each element of *dim_coeffs* is (K, coeffs).  Returns a numpy array of
    shape (K_0, K_1, ...) whose value at (i, j, ...) is
    p_0(i) * p_1(j) * ...
    """
    grids = [np.arange(K, dtype=float) for K, _ in dim_coeffs]
    arrs  = np.meshgrid(*grids, indexing='ij')
    return np.prod([poly_eval(arrs[d], c) for d, (K, c) in enumerate(dim_coeffs)],
                   axis=0)


def kron_state(*per_dim_states):
    """Kronecker product of a sequence of 1-D state vectors (left-to-right)."""
    result = per_dim_states[0]
    for s in per_dim_states[1:]:
        result = np.kron(result, s)
    return result


# ---------------------------------------------------------------------------
# Shared signal fixtures (built once at module load time)
# ---------------------------------------------------------------------------

_K2 = 25
_Y_2D_12 = build_separable_signal((_K2, _COEFFS_PD1), (_K2, _COEFFS_PD2))

_K3 = 18
_Y_3D_123 = build_separable_signal((_K3, _COEFFS_PD1), (_K3, _COEFFS_PD2), (_K3, _COEFFS_PD3))

_K4 = 14
_Y_4D_2222 = build_separable_signal(*[(_K4, _COEFFS_PD2_ALT)] * 4)

# Interior check indices — must satisfy hw <= idx < K - hw in every dimension
# (otherwise the window is clipped and the GT no longer holds exactly)
_IDX_2D  = [(10, 10), (11, 12), (12, 10)]
_IDX_3D  = [(8, 8, 8), (10, 9, 8), (8, 9, 8)]
_IDX_4D  = [(6, 6, 6, 6), (7, 7, 7, 7)]
_SCIPY_2D = (10, 10)
_SCIPY_3D = (8, 8, 8)
_SCIPY_4D = (6, 6, 6, 6)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestNDPipelineGroundTruth(unittest.TestCase):
    """
    End-to-end correctness tests for the ND RLS pipeline: W, xi, minimize_x.

    Each test group contains:
      GT tests  — xs[idx] matches the Kronecker-product polynomial GT.
      Scipy test — W @ xs[idx] == xi[idx] verified via scipy.linalg.lstsq.
      Both FW-only and FW+BW CompositeCost variants are exercised.
    """

    # ------------------------------------------------------------------
    # 2-D, pd = 1 + 2  (single FW segment per dimension)
    # ------------------------------------------------------------------

    def test_2d_pd1_pd2_fw_only_gt(self):
        """
        2-D pd=1+2, single FW segment: minimize_x must recover the Kronecker GT.

        With a noiseless polynomial signal and a window that fully contains the
        fit region, the WLS solution is exact regardless of the exponential
        weighting (gamma).  This test would fail on the unfixed library because
        the wrong Kronecker stride in the scatter-write produces a garbled xi.
        """
        _, xs = run_nd_rls(_Y_2D_12, [1, 2], two_segment=False)
        for i, j in _IDX_2D:
            gt = kron_state(true_state_1d(i, _COEFFS_PD1),
                            true_state_1d(j, _COEFFS_PD2))
            np.testing.assert_allclose(
                xs[i, j], gt, atol=1e-5,
                err_msg=f'GT mismatch at ({i},{j}) — 2D pd=1+2 FW-only',
            )

    def test_2d_pd1_pd2_fw_bw_gt(self):
        """2-D pd=1+2, two-segment FW+BW: minimize_x must recover the Kronecker GT."""
        _, xs = run_nd_rls(_Y_2D_12, [1, 2], two_segment=True)
        for i, j in _IDX_2D:
            gt = kron_state(true_state_1d(i, _COEFFS_PD1),
                            true_state_1d(j, _COEFFS_PD2))
            np.testing.assert_allclose(
                xs[i, j], gt, atol=1e-5,
                err_msg=f'GT mismatch at ({i},{j}) — 2D pd=1+2 FW+BW',
            )

    def test_2d_pd1_pd2_scipy_xi_check(self):
        """
        2-D pd=1+2: scipy.linalg.lstsq(W, xi[idx]) must agree with minimize_x.

        This independently verifies W and xi are mutually consistent: if either
        is wrong the scipy solution will differ from lmlib's.
        """
        rls, xs = run_nd_rls(_Y_2D_12, [1, 2], two_segment=True)
        i, j = _SCIPY_2D
        x_scipy, _, _, _ = lstsq(rls.W, rls.xi[i, j])
        np.testing.assert_allclose(
            xs[i, j], x_scipy, atol=1e-10,
            err_msg='lmlib minimize_x vs scipy lstsq mismatch — 2D pd=1+2',
        )

    def test_2d_pd2_pd1_reversed_gt(self):
        """
        2-D pd=2+1 (reversed order): both orderings of mixed degrees must work.

        Before the fix, Bug 1 (wrong Kronecker stride) was triggered regardless
        of which dimension had the higher degree.
        """
        c_i, c_j = _COEFFS_PD2, _COEFFS_PD1
        Y_21 = build_separable_signal((_K2, c_i), (_K2, c_j))
        _, xs = run_nd_rls(Y_21, [2, 1], two_segment=True)
        for i, j in [(10, 10), (11, 12)]:
            gt = kron_state(true_state_1d(i, c_i), true_state_1d(j, c_j))
            np.testing.assert_allclose(
                xs[i, j], gt, atol=1e-5,
                err_msg=f'GT mismatch at ({i},{j}) — 2D pd=2+1',
            )

    def test_2d_backends_agree_fw_bw(self):
        """numpy and lfilter backends must give identical minimize_x for 2-D pd=1+2."""
        _, xs_np = run_nd_rls(_Y_2D_12, [1, 2], backend='numpy',   two_segment=True)
        _, xs_lf = run_nd_rls(_Y_2D_12, [1, 2], backend='lfilter', two_segment=True)
        np.testing.assert_allclose(xs_np, xs_lf, rtol=1e-8, atol=1e-8,
                                   err_msg='numpy vs lfilter xs mismatch — 2D pd=1+2')

    # ------------------------------------------------------------------
    # 3-D, pd = 1 + 2 + 3
    # ------------------------------------------------------------------

    def test_3d_pd1_pd2_pd3_fw_only_gt(self):
        """
        3-D pd=1+2+3, single FW segment: minimize_x must recover the Kronecker GT.

        Before the fix, Bug 3 (reshape-copies-not-views for 3-D signals) caused
        xi to remain all-zero, so minimize_x returned xs = 0 everywhere.
        """
        _, xs = run_nd_rls(_Y_3D_123, [1, 2, 3], two_segment=False)
        for idx in _IDX_3D:
            gt = kron_state(true_state_1d(idx[0], _COEFFS_PD1),
                            true_state_1d(idx[1], _COEFFS_PD2),
                            true_state_1d(idx[2], _COEFFS_PD3))
            np.testing.assert_allclose(
                xs[idx], gt, atol=0.02,rtol=1e-5,
                err_msg=f'GT mismatch at {idx} — 3D pd=1+2+3 FW-only',
            )

    def test_3d_pd1_pd2_pd3_fw_bw_gt(self):
        """3-D pd=1+2+3, two-segment FW+BW: minimize_x must recover the Kronecker GT."""
        _, xs = run_nd_rls(_Y_3D_123, [1, 2, 3], two_segment=True)
        for idx in _IDX_3D:
            gt = kron_state(true_state_1d(idx[0], _COEFFS_PD1),
                            true_state_1d(idx[1], _COEFFS_PD2),
                            true_state_1d(idx[2], _COEFFS_PD3))
            np.testing.assert_allclose(
                xs[idx], gt, atol=1e-4,
                err_msg=f'GT mismatch at {idx} — 3D pd=1+2+3 FW+BW',
            )

    def test_3d_pd1_pd2_pd3_scipy_xi_check(self):
        """
        3-D pd=1+2+3: scipy.linalg.lstsq(W, xi[idx]) must agree with minimize_x.
        """
        rls, xs = run_nd_rls(_Y_3D_123, [1, 2, 3], two_segment=True)
        idx = _SCIPY_3D
        x_scipy, _, _, _ = lstsq(rls.W, rls.xi[idx])
        np.testing.assert_allclose(
            xs[idx], x_scipy, atol=1e-10,
            err_msg='lmlib minimize_x vs scipy lstsq mismatch — 3D pd=1+2+3',
        )

    def test_3d_backends_agree_fw_bw(self):
        """numpy and lfilter backends must give identical minimize_x for 3-D pd=1+2+3."""
        _, xs_np = run_nd_rls(_Y_3D_123, [1, 2, 3], backend='numpy',   two_segment=True)
        _, xs_lf = run_nd_rls(_Y_3D_123, [1, 2, 3], backend='lfilter', two_segment=True)
        np.testing.assert_allclose(xs_np, xs_lf, rtol=1e-8, atol=1e-8,
                                   err_msg='numpy vs lfilter xs mismatch — 3D pd=1+2+3')

    # ------------------------------------------------------------------
    # 4-D, pd = 2 + 2 + 2 + 2
    # ------------------------------------------------------------------

    def test_4d_pd2x4_fw_bw_gt(self):
        """
        4-D pd=2×4, two-segment FW+BW: minimize_x must recover the Kronecker GT.

        This config exposes Bug 3 alone: all degrees are equal so Bugs 1+2
        cancel, but the order='F' + moveaxis + reshape problem causes xi to
        be all-zero for any signal with 4 spatial dimensions.
        """
        _, xs = run_nd_rls(_Y_4D_2222, [2, 2, 2, 2], two_segment=True)
        c = _COEFFS_PD2_ALT
        for idx in _IDX_4D:
            gt = kron_state(*[true_state_1d(idx[d], c) for d in range(4)])
            np.testing.assert_allclose(
                xs[idx], gt, atol=1e-3,
                err_msg=f'GT mismatch at {idx} — 4D pd=2×4 FW+BW',
            )

    def test_4d_pd2x4_scipy_xi_check(self):
        """
        4-D pd=2×4: scipy.linalg.lstsq(W, xi[idx]) must agree with minimize_x.
        """
        rls, xs = run_nd_rls(_Y_4D_2222, [2, 2, 2, 2], two_segment=True)
        idx = _SCIPY_4D
        x_scipy, _, _, _ = lstsq(rls.W, rls.xi[idx])
        np.testing.assert_allclose(
            xs[idx], x_scipy, atol=1e-10,
            err_msg='lmlib minimize_x vs scipy lstsq mismatch — 4D pd=2×4',
        )

    def test_4d_backends_agree_fw_bw(self):
        """numpy and lfilter backends must give identical minimize_x for 4-D pd=2×4."""
        _, xs_np = run_nd_rls(_Y_4D_2222, [2, 2, 2, 2], backend='numpy',   two_segment=True)
        _, xs_lf = run_nd_rls(_Y_4D_2222, [2, 2, 2, 2], backend='lfilter', two_segment=True)
        np.testing.assert_allclose(xs_np, xs_lf, rtol=1e-8, atol=1e-8,
                                   err_msg='numpy vs lfilter xs mismatch — 4D pd=2×4')

    # ------------------------------------------------------------------
    # 1-D sanity baseline (NDCompositeCost wrapping a single CompositeCost)
    # ------------------------------------------------------------------

    def test_1d_pd2_fw_bw_gt(self):
        """
        1-D pd=2, FW+BW: baseline correctness check for the single-dimension path.

        NDCompositeCost([cost]) with L=1 exercises _nd_xi_q_recursion without
        the asterisk step.  If this fails the problem is in the base recursion,
        not in the ND chaining.
        """
        c = _COEFFS_PD2
        K = 25
        y1d = poly_eval(np.arange(K, dtype=float), c)
        _, xs = run_nd_rls(y1d, [2], two_segment=True)
        for i in range(_HW, K - _HW):
            gt = true_state_1d(i, c)
            np.testing.assert_allclose(
                xs[i], gt, atol=1e-6,
                err_msg=f'GT mismatch at i={i} — 1D pd=2 FW+BW',
            )

    def test_1d_scipy_xi_check(self):
        """1-D pd=2: scipy.linalg.lstsq(W, xi[i]) must agree with minimize_x."""
        c = _COEFFS_PD2
        K = 25
        y1d = poly_eval(np.arange(K, dtype=float), c)
        rls, xs = run_nd_rls(y1d, [2], two_segment=True)
        i = 12
        x_scipy, _, _, _ = lstsq(rls.W, rls.xi[i])
        np.testing.assert_allclose(
            xs[i], x_scipy, atol=1e-10,
            err_msg='lmlib minimize_x vs scipy lstsq mismatch — 1D pd=2',
        )

    # ------------------------------------------------------------------
    # dim_order permutation invariance of minimize_x
    # ------------------------------------------------------------------

    def test_dim_order_both_orderings_recover_gt_2d(self):
        """
        2-D pd=1+2: both dim_order=[0,1] and dim_order=[1,0] must recover the GT.

        Note: dim_order controls the *accumulation order* of the Kronecker product,
        so the two orderings produce state vectors with different element orderings:
          dim_order=[0,1] → xs[i,j] = kron(x0(i), x1(j))   (cost-0 ⊗ cost-1)
          dim_order=[1,0] → xs[i,j] = kron(x1(j), x0(i))   (cost-1 ⊗ cost-0)
        Both must agree with their respective GT; xs itself is NOT identical across
        orderings (the Kronecker blocks are permuted), but both are correct LS fits.
        """
        nd = make_nd_cost([1, 2], two_segment=True)
        i, j = 10, 10
        gt_01 = kron_state(true_state_1d(i, _COEFFS_PD1), true_state_1d(j, _COEFFS_PD2))
        gt_10 = kron_state(true_state_1d(j, _COEFFS_PD2), true_state_1d(i, _COEFFS_PD1))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            rls01 = lm.RLSAlssm(nd, steady_state=True, backend='numpy')
            rls01.filter(_Y_2D_12, dim_order=[0, 1])
            xs_01 = rls01.minimize_x()

            rls10 = lm.RLSAlssm(nd, steady_state=True, backend='numpy')
            rls10.filter(_Y_2D_12, dim_order=[1, 0])
            xs_10 = rls10.minimize_x()

        np.testing.assert_allclose(
            xs_01[i, j], gt_01, atol=1e-4,
            err_msg='dim_order=[0,1]: GT mismatch at (10,10) — 2D pd=1+2',
        )
        np.testing.assert_allclose(
            xs_10[i, j], gt_10, atol=1e-4,
            err_msg='dim_order=[1,0]: GT mismatch at (10,10) — 2D pd=1+2',
        )


class TestFiniteSegmentExactW(unittest.TestCase):
    r"""Regression: steady-state W on a finite segment must equal the exact sum.

    For a finite segment the Gram matrix is the finite sum
    ``W = sum_{t=a}^{b} gamma^{t-delta} (A^t)^T C^T C A^t``.  ``AlssmPolyJordan``
    is caught by neither the Legendre nor the Meixner fast path, so before the
    fix it fell to the ill-conditioned N^2 x N^2 Stein solve, which lost ~4
    digits for short, high-degree segments and pushed the 2-D kron condition
    number from ~1e12 to ~1e13 — enough to wreck the (already borderline)
    multi-channel ECG fit.  The direct-sum path is exact.
    """

    @staticmethod
    def _exact_W(alssm, seg):
        from numpy.linalg import matrix_power as mpow
        A, C = alssm.A, np.atleast_2d(alssm.C)
        Q = C.T @ C
        W = np.zeros((alssm.N, alssm.N))
        for t in range(seg.a, seg.b + 1):
            At = mpow(A, t)
            W += seg.gamma ** (t - seg.delta) * (At.T @ Q @ At)
        return W

    def test_polyjordan_channel_W_matches_exact(self):
        # the short, high-degree channel block that was inaccurate
        alssm = lm.AlssmPolyJordan(poly_degree=5)
        seg = lm.Segment(a=0, b=6, direction=lm.BACKWARD, g=15.0)
        W = lm.CostSegment(alssm, seg).get_steady_state_W('schur')
        np.testing.assert_allclose(W, self._exact_W(alssm, seg), rtol=0, atol=1e-10)

    def test_polyjordan_time_W_matches_exact(self):
        alssm = lm.AlssmProd((lm.AlssmPolyJordan(poly_degree=5), lm.AlssmExp(gamma=1)))
        seg = lm.Segment(a=0, b=15, direction=lm.BACKWARD, g=15.0)
        W = lm.CostSegment(alssm, seg).get_steady_state_W('schur')
        np.testing.assert_allclose(W, self._exact_W(alssm, seg), rtol=0, atol=1e-9)

    def test_limited_sum_helper(self):
        from lmlib.statespace.backends.steady_state import covariance_matrix_limited_sum
        alssm = lm.AlssmPolyJordan(poly_degree=4)
        seg = lm.Segment(a=0, b=8, direction=lm.BACKWARD, g=20.0)
        W = covariance_matrix_limited_sum(alssm.A, alssm.C, seg.gamma, seg.a, seg.b, seg.delta)
        np.testing.assert_allclose(W, self._exact_W(alssm, seg), rtol=0, atol=1e-12)

    def test_2d_kron_well_conditioned_and_reconstructs(self):
        # 2-D (time x channel) ECG-style model: exact W keeps cond ~1e12 (solvable),
        # so a windowed beat is reconstructed (not the ~1e13 / failing case).
        from numpy.linalg import matrix_power as mpow
        rng = np.random.default_rng(0)
        K, M, b_r1, b_r2, g = 200, 7, 15, 6, 15.0
        a1 = lm.AlssmProd((lm.AlssmPolyJordan(poly_degree=5), lm.AlssmExp(gamma=1)))
        a2 = lm.AlssmPolyJordan(poly_degree=5)
        s1 = lm.Segment(a=0, b=b_r1, direction=lm.BACKWARD, g=g)
        s2 = lm.Segment(a=0, b=b_r2, direction=lm.BACKWARD, g=g)
        nd = lm.NDCompositeCost([lm.CostSegment(a1, s1), lm.CostSegment(a2, s2)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            W = nd.get_steady_state_W(method='schur')
        self.assertLess(np.linalg.cond(W), 5e12)         # was ~1e13 before the fix
        # build a separable polynomial signal (degree <= 5 per axis) that lies
        # exactly in the model space, fit it, and check the reference-window
        # reconstruction is exact (before the fix it failed, MSE ~1).
        K_REF = 80
        i = np.arange(b_r1 + 1); j = np.arange(b_r2 + 1)
        shape = ((1 - 0.05 * i + 0.002 * i ** 2)[:, None]
                 * (1 + 0.1 * j - 0.01 * j ** 2)[None, :])
        y = np.zeros((K, M)); y[K_REF:K_REF + b_r1 + 1, :b_r2 + 1] = shape
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rls = lm.RLSAlssm(nd, steady_state=True); rls.filter(y, dim_order=[0, 1])
            xs = rls.minimize_x()[K_REF, 0, :]
        _, surf = lm.Trajectory.eval(nd, xs)
        recon = surf[:b_r1 + 1, :b_r2 + 1]
        self.assertLess(np.mean((recon - shape) ** 2), 1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
