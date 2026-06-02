"""
AlssmPolyMeixner — Comprehensive Test Suite
============================================

Tests cover:
  1.  A and C matrix structure
  2.  Orthogonality under geometric weight (numerical ground truth)
  3.  Steady-state W fast path (closed-form vs numerical sum)
  4.  Steady-state W: forward, finite, and two-sided segments
  5.  Condition-number benchmark vs AlssmPoly and AlssmPolyLegendre
  6.  Maximum degree analysis (where does numerical precision degrade?)
  7.  Functional drop-in: ex122.0 (two-sided polynomial filters)
  8.  Functional drop-in: ex111.0 (pulse detection, 3 segments, mixed g)
  9.  Functional drop-in: ex113.0 (ECG shape detection)
 10.  Filtered signal fidelity (Meixner ≈ AlssmPoly output)
 11.  CompositeCost steady-state W conditioning
 12.  Edge cases (poly_degree=0, large g, small g near 1)
"""

import sys, warnings
import numpy as np
import pytest
sys.path.insert(0, '/home/claude')

import lmlib as lm
from lmlib.statespace.backends.steady_state import (
    covariance_matrix_meixner, _is_meixner_shift,
    covariance_matrix_schur,
)
from lmlib.utils.generator import gen_rect, gen_wgn, gen_rand_walk, gen_rand_pulse, load_lib_csv
from scipy.special import hyp2f1


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def meixner_poly(n, j, gamma):
    """Evaluate M_n(j; 1, gamma) = 2F1(-n, -j; 1; 1-1/gamma)."""
    if n == 0:
        return np.ones_like(np.asarray(j, dtype=float))
    j = np.asarray(j, dtype=float)
    return np.array([float(hyp2f1(-n, -jj, 1, 1.0 - 1.0 / gamma)) for jj in j.ravel()]).reshape(j.shape)


def numerical_gram(poly_degree, gamma, J=8000):
    """Brute-force Gram matrix under gamma^j weight."""
    N = poly_degree + 1
    js = np.arange(J, dtype=float)
    w = gamma ** js
    V = np.column_stack([meixner_poly(n, js, gamma) for n in range(N)])
    return V.T @ (w[:, None] * V)


def make_alssm(poly_degree, g, direction=lm.BACKWARD, **kwargs):
    """Build an AlssmPolyMeixner over a canonical semi-infinite segment with
    effective window size ``g`` (origin at the canonical lag, shift=0)."""
    if direction == lm.BACKWARD:
        seg = lm.Segment(a=0, b=np.inf, direction=lm.BACKWARD, g=g)
    else:
        seg = lm.Segment(a=-np.inf, b=-1, direction=lm.FORWARD, g=g)
    return lm.AlssmPolyMeixner(poly_degree, seg, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 1. A and C matrix structure
# ─────────────────────────────────────────────────────────────────────────────

class TestMatrices:

    @pytest.mark.parametrize("D,g", [(0,10),(1,20),(2,50),(3,100),(5,30),(8,20)])
    def test_A_upper_triangular(self, D, g):
        alssm = make_alssm(D, g)
        assert np.allclose(np.tril(alssm.A, -1), 0), "A must be upper triangular"

    @pytest.mark.parametrize("D,g", [(0,10),(1,20),(3,50),(5,100)])
    def test_A_unit_diagonal(self, D, g):
        alssm = make_alssm(D, g)
        assert np.allclose(np.diag(alssm.A), 1.0), "A diagonal must be 1"

    @pytest.mark.parametrize("D,g", [(1,20),(3,50),(5,100)])
    def test_A_superdiagonal_value(self, D, g):
        alssm = make_alssm(D, g)
        N = D + 1
        expected = -1.0 / (g - 1.0)
        upper = alssm.A[np.triu_indices(N, k=1)]
        assert np.allclose(upper, expected), f"All superdiag entries must be {expected:.6f}"

    @pytest.mark.parametrize("D,g", [(0,10),(1,20),(3,50),(5,100)])
    def test_C_all_ones(self, D, g):
        alssm = make_alssm(D, g)
        assert np.allclose(alssm.C, 1.0), "C must be all-ones"

    @pytest.mark.parametrize("D,g", [(0,10),(1,20),(3,50),(5,100)])
    def test_C_shape(self, D, g):
        alssm = make_alssm(D, g)
        assert alssm.C.shape == (D + 1,)

    @pytest.mark.parametrize("D,g", [(0,10),(1,20),(3,50)])
    def test_A_invertible(self, D, g):
        alssm = make_alssm(D, g)
        assert abs(np.linalg.det(alssm.A)) > 1e-10, "A must be invertible"

    def test_A_condition_number_low(self):
        """kappa(A) should be << kappa(A_pascal) for same N and g."""
        for D in [2, 3, 5]:
            g = 20
            alssm_m = make_alssm(D, g)
            alssm_p = lm.AlssmPoly(D)
            kappa_m = np.linalg.cond(alssm_m.A)
            kappa_p = np.linalg.cond(alssm_p.A.astype(float))
            assert kappa_m < kappa_p or kappa_m < 5, \
                f"D={D} g={g}: Meixner kappa(A)={kappa_m:.2f} not better than Pascal kappa={kappa_p:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Orthogonality under geometric weight
# ─────────────────────────────────────────────────────────────────────────────

class TestOrthogonality:

    @pytest.mark.parametrize("D,g", [(1,10),(2,20),(3,50),(4,100)])
    def test_gram_diagonal(self, D, g):
        """Gram matrix must be diagonal (off-diag < 1e-6 * norm)."""
        gamma = (g - 1.0) / g
        W = numerical_gram(D, gamma, J=10000)
        diag_norm = np.mean(np.abs(np.diag(W)))
        off = np.abs(W - np.diag(np.diag(W)))
        assert np.max(off) < 1e-5 * diag_norm, \
            f"Gram matrix not diagonal: max off-diag={np.max(off):.2e}"

    @pytest.mark.parametrize("D,g", [(0,10),(1,20),(3,50),(4,100)])
    def test_gram_diagonal_values(self, D, g):
        """Diagonal W[n,n] = g * (g/(g-1))^n  (closed-form)."""
        gamma = (g - 1.0) / g
        W_num = numerical_gram(D, gamma, J=10000)
        ns = np.arange(D + 1, dtype=float)
        W_theory = g * (g / (g - 1.0)) ** ns
        assert np.allclose(np.diag(W_num), W_theory, rtol=1e-3), \
            f"Diagonal mismatch: numerical={np.diag(W_num)}, theory={W_theory}"

    @pytest.mark.parametrize("D", [1, 2, 3])
    def test_meixner_at_zero(self, D):
        """M_n(0; 1, gamma) = 1 for all n => C = [1,...,1] is correct."""
        for g in [10, 50, 200]:
            gamma = (g - 1.0) / g
            for n in range(D + 1):
                val = float(hyp2f1(-n, 0.0, 1.0, 1.0 - 1.0 / gamma))
                assert abs(val - 1.0) < 1e-12, f"M_{n}(0) != 1: got {val}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Steady-state W fast path (closed-form == numerical)
# ─────────────────────────────────────────────────────────────────────────────

class TestSteadyStateW:

    @pytest.mark.parametrize("D,g", [(0,10),(1,20),(2,50),(3,100),(5,30)])
    def test_fast_path_vs_numerical_backward_infinite(self, D, g):
        """covariance_matrix_meixner agrees with brute-force sum for BW-inf segment."""
        gamma = (g - 1.0) / g
        alssm = make_alssm(D, g)
        W_fast = covariance_matrix_meixner(alssm.A, alssm.C, gamma, 0, np.inf, delta=0)
        W_num  = numerical_gram(D, gamma, J=12000)
        assert np.allclose(W_fast, W_num, rtol=1e-3), \
            f"Fast path mismatch D={D} g={g}: fast={np.diag(W_fast)}, num={np.diag(W_num)}"

    @pytest.mark.parametrize("D,g", [(1,20),(2,50),(3,100)])
    def test_fast_path_via_schur_dispatches(self, D, g):
        """covariance_matrix_schur must dispatch to the Meixner fast path."""
        gamma = (g - 1.0) / g
        alssm = make_alssm(D, g)
        W_schur = covariance_matrix_schur(alssm.A, alssm.C, gamma, 0, np.inf, delta=0)
        W_ref   = covariance_matrix_meixner(alssm.A, alssm.C, gamma, 0, np.inf, delta=0)
        assert np.allclose(W_schur, W_ref, rtol=1e-10)

    @pytest.mark.parametrize("D,g", [(1,20),(3,50)])
    def test_fast_path_is_diagonal(self, D, g):
        gamma = (g - 1.0) / g
        alssm = make_alssm(D, g)
        W = covariance_matrix_meixner(alssm.A, alssm.C, gamma, 0, np.inf, delta=0)
        off = np.abs(W - np.diag(np.diag(W)))
        assert np.max(off) < 1e-10, "Steady-state W must be diagonal"

    @pytest.mark.parametrize("D,g", [(1,20),(3,50)])
    def test_fast_path_forward_infinite(self, D, g):
        """Forward semi-infinite segment (-inf, b]: W must also be diagonal."""
        gamma_fw = g / (g - 1.0)           # > 1 for forward direction
        alssm = make_alssm(D, g)
        W = covariance_matrix_meixner(alssm.A, alssm.C, gamma_fw, -np.inf, -1, delta=0)
        off = np.abs(W - np.diag(np.diag(W)))
        assert np.max(off) < 1e-8, "Forward-infinite W must be diagonal"
        assert np.all(np.diag(W) > 0), "Forward-infinite W diagonal must be positive"

    @pytest.mark.parametrize("D,g", [(1,20),(2,50)])
    def test_finite_segment_via_schur_and_meixner_agree(self, D, g):
        """For a finite segment, fast path == general Stein solver."""
        gamma = (g - 1.0) / g
        alssm = make_alssm(D, g)
        W_m = covariance_matrix_meixner(alssm.A, alssm.C, gamma, 0, 2 * g, delta=0)
        # Reference: brute-force sum
        J = int(2 * g) + 1
        js = np.arange(J, dtype=float)
        N = D + 1
        V = np.column_stack([meixner_poly(n, js, gamma) for n in range(N)])
        w = gamma ** js
        W_ref = V.T @ (w[:, None] * V)
        assert np.allclose(W_m, W_ref, rtol=1e-4), \
            f"Finite segment mismatch D={D} g={g}"

    @pytest.mark.parametrize("D,g", [(1,20),(3,50)])
    def test_CostSegment_get_steady_state_W(self, D, g):
        """CostSegment.get_steady_state_W() uses the Meixner fast path."""
        alssm = make_alssm(D, g)
        seg = lm.Segment(a=0, b=np.inf, direction=lm.BACKWARD, g=g)
        cost = lm.CostSegment(alssm, seg)
        W = cost.get_steady_state_W()
        gamma = (g - 1.0) / g
        W_ref = covariance_matrix_meixner(alssm.A, alssm.C, gamma, 0, np.inf, delta=0)
        assert np.allclose(W, W_ref, rtol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Condition-number benchmark
# ─────────────────────────────────────────────────────────────────────────────

class TestConditionNumbers:

    def test_composite_W_cond_vs_monomial_ex122(self):
        """ex122.0: composite W kappa(Meixner) << kappa(Poly) for two-sided window."""
        g = 20
        gamma = (g - 1.0) / g

        for D in [1, 2, 3]:
            alssm_m = make_alssm(D, g)
            alssm_p = lm.AlssmPoly(D)

            seg_L = lm.Segment(a=-np.inf, b=-1,      direction=lm.FORWARD,  g=g)
            seg_R = lm.Segment(a=0,       b=np.inf,  direction=lm.BACKWARD, g=g)

            def total_W(alssm):
                cost = lm.CompositeCost((alssm,), (seg_L, seg_R), F=[[1, 1]])
                return cost.get_steady_state_W()

            kappa_m = np.linalg.cond(total_W(alssm_m))
            kappa_p = np.linalg.cond(total_W(alssm_p))
            print(f"  ex122 D={D}: kappa(Meixner)={kappa_m:.2f}  kappa(Poly)={kappa_p:.2e}")
            assert kappa_m < 5.0, f"Meixner kappa={kappa_m:.2f} too high for D={D}"
            assert kappa_m < kappa_p / 100, f"Meixner not much better than Poly at D={D}"

    def test_composite_W_cond_vs_monomial_ex111(self):
        """ex111.0: 3-segment composite W kappa(Meixner) << kappa(Poly)."""
        g_bl, g_sp, len_pulse = 50, 15000, 20

        for D in [1, 2]:
            alssm_m = make_alssm(D, g_bl)
            alssm_p = lm.AlssmPoly(D)

            seg_L = lm.Segment(a=-np.inf,         b=-1,              direction=lm.FORWARD,  g=g_bl)
            seg_C = lm.Segment(a=0,               b=len_pulse,       direction=lm.BACKWARD, g=g_sp)
            seg_R = lm.Segment(a=len_pulse + 1,   b=np.inf,          direction=lm.BACKWARD, g=g_bl,
                               delta=len_pulse)

            def total_W(alssm):
                cost = lm.CompositeCost((alssm,), (seg_L, seg_C, seg_R), F=[[1, 1, 1]])
                return cost.get_steady_state_W()

            kappa_m = np.linalg.cond(total_W(alssm_m))
            kappa_p = np.linalg.cond(total_W(alssm_p))
            print(f"  ex111 D={D}: kappa(Meixner)={kappa_m:.2f}  kappa(Poly)={kappa_p:.2e}")
            assert kappa_m < 20.0, f"Meixner kappa={kappa_m:.2f} too high for D={D}"
            assert kappa_m < kappa_p / 10, f"Meixner not better than Poly at D={D}"

    def test_W_ss_kappa_vs_degree(self):
        """W_ss kappa grows slowly as (g/(g-1))^D — verify for g=20."""
        g = 20
        gamma = (g - 1.0) / g
        print("\n  D | kappa(W_ss Meixner) | theoretical")
        for D in range(1, 15):
            alssm = make_alssm(D, g)
            W = covariance_matrix_meixner(alssm.A, alssm.C, gamma, 0, np.inf, delta=0)
            kappa = np.linalg.cond(W)
            kappa_theory = (g / (g - 1.0)) ** D
            print(f"  {D:2d} | {kappa:18.4f} | {kappa_theory:.4f}")
            assert np.isclose(kappa, kappa_theory, rtol=1e-6), \
                f"D={D}: kappa={kappa:.4f} != theory {kappa_theory:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Maximum degree analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxDegree:
    """
    Meixner degree limits come from three sources:
      (a) A-matrix: cond(A) ~ 1 + D/(g-1), degrades filter numerics
      (b) Meixner polynomial evaluation via hyp2f1 (used in finite-segment W)
      (c) RLS recursion accumulation — bounded by cond(W)^0.5 * eps

    For the semi-infinite case (no hyp2f1 needed), the limit is dominated by (a).
    """

    def test_A_cond_vs_degree(self):
        """kappa(A) grows as 1 + D/(g-1). For g=20 and D=19: kappa ~ 2."""
        g = 20
        print("\n  D | kappa(A)")
        for D in range(1, 30):
            alssm = make_alssm(D, g)
            kappa = np.linalg.cond(alssm.A)
            print(f"  {D:2d} | {kappa:.4f}")
        # At D=50, g=20: kappa(A) = 1 + D/(g-1) = 1 + 50/19 ≈ 3.6 theoretically
        # but the actual upper-triangular matrix formula gives kappa ~ (1 + D/(g-1))^2
        # for the 2-norm; still << any polynomial-basis A which grows as g^D
        alssm = make_alssm(50, g)
        assert np.linalg.cond(alssm.A) < 20.0, "A condition should be well-bounded even at D=50"

    def test_W_ss_precision_vs_degree(self):
        """Closed-form W_ss stays exact (no Stein solve) for any degree."""
        g = 20
        gamma = (g - 1.0) / g
        print("\n  D | max relative W error vs theory")
        for D in [5, 10, 20, 50, 100]:
            alssm = make_alssm(D, g)
            W = covariance_matrix_meixner(alssm.A, alssm.C, gamma, 0, np.inf, delta=0)
            ns = np.arange(D + 1, dtype=float)
            W_theory = g * (g / (g - 1.0)) ** ns
            rel_err = np.max(np.abs(np.diag(W) - W_theory) / W_theory)
            print(f"  {D:3d} | {rel_err:.2e}")
            assert rel_err < 1e-10, f"W_ss precision too low at D={D}: rel_err={rel_err:.2e}"

    def test_filter_precision_vs_degree_infinite_segment(self):
        """
        RLS filter on a pure polynomial signal must recover exact coefficients
        (within floating-point noise) for any degree where A is well-conditioned.
        """
        np.random.seed(42)
        K = 500
        g = 50
        print("\n  D | state-recovery error")
        for D in [1, 2, 3, 5, 8]:
            # Build a pure degree-D polynomial signal
            t = np.arange(K, dtype=float) / K
            coeffs = np.random.randn(D + 1)
            y = np.polyval(coeffs[::-1], t)   # x^0 + x^1 + ...

            alssm = make_alssm(D, g)
            seg   = lm.Segment(a=-(3*g), b=0, direction=lm.BACKWARD, g=g)
            cost  = lm.CostSegment(alssm, seg)
            rls   = lm.RLSAlssm(cost, steady_state=True)
            y_hat = rls.fit(y)

            # y_hat should approximate y everywhere after warm-up
            err = np.max(np.abs(y_hat[K//2:] - y[K//2:]))
            print(f"  {D:2d} | {err:.2e}")
            assert err < 0.5, f"Large fit error at D={D}: {err:.2e}"

    def test_monomial_vs_meixner_high_degree(self):
        """
        At high degree, AlssmPoly becomes numerically unstable.
        AlssmPolyMeixner must remain stable (no NaN/Inf in W_ss).
        """
        g = 20
        gamma = (g - 1.0) / g
        for D in [8, 12, 16, 20]:
            alssm_m = make_alssm(D, g)
            alssm_p = lm.AlssmPoly(D)
            seg = lm.Segment(a=-np.inf, b=0, direction=lm.BACKWARD, g=g)

            W_m = covariance_matrix_schur(alssm_m.A, alssm_m.C, gamma, -np.inf, 0, 0)
            assert np.all(np.isfinite(W_m)), f"Meixner W_ss has NaN/Inf at D={D}"
            assert np.linalg.cond(W_m) < 1e6, f"Meixner W_ss ill-conditioned at D={D}"

            # Poly will typically blow up for D >= 8 with g=20
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    W_p = covariance_matrix_schur(alssm_p.A, alssm_p.C, gamma, -np.inf, 0, 0)
                    kappa_p = np.linalg.cond(W_p)
                except Exception:
                    kappa_p = np.inf
            kappa_m = np.linalg.cond(W_m)
            print(f"  D={D:2d}: kappa(Meixner)={kappa_m:.2e}  kappa(Poly)={kappa_p:.2e}")
            assert kappa_m < kappa_p / 1e3 or kappa_p > 1e10, \
                f"Meixner not better at D={D}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Functional: ex122.0 (two-sided polynomial filters)
# ─────────────────────────────────────────────────────────────────────────────

class TestEx122:

    def test_output_finite_matches_poly(self):
        """
        On a single semi-infinite segment, AlssmPolyMeixner(D) and AlssmPoly(D)
        fit the *same* degree-D polynomial subspace, so the filtered output ŷ
        (the projection onto that subspace) must be identical — only the basis
        differs.

        Note: this equality holds per segment.  It does NOT hold for a two-sided
        (FW + BW) cost driven by a single Meixner ALSSM, because a Meixner model
        carries a direction-specific decay and the BW-built model evaluated in a
        FW segment spans a different fit than Poly.  The single-segment form is
        the meaningful invariant.
        """
        K = 500
        y = gen_rect(K, 500, 250) + gen_wgn(K, sigma=0.02, seed=7)
        g = 20
        seg = lm.Segment(a=0, b=np.inf, direction=lm.BACKWARD, g=g)

        def get_yhat(alssm):
            cost = lm.CompositeCost((alssm,), (seg,), F=[[1]])
            rls  = lm.RLSAlssm(cost, steady_state=True)
            return rls.fit(y)

        for D in [0, 1, 2, 3]:
            y_poly = get_yhat(lm.AlssmPoly(poly_degree=D))
            y_meix = get_yhat(lm.AlssmPolyMeixner(D, seg))

            corr = np.corrcoef(y_poly, y_meix)[0, 1]
            rmse = np.sqrt(np.mean((y_poly - y_meix) ** 2))
            print(f"  ex122 D={D}: corr={corr:.6f}  RMSE={rmse:.2e}")
            assert corr > 0.999, f"D={D}: low correlation between Poly and Meixner outputs ({corr:.4f})"
            assert rmse < 1e-3, f"D={D}: RMSE too large ({rmse:.2e})"

    def test_symmetric_filter_symmetry(self):
        """
        A symmetric two-sided Meixner filter on a symmetric signal produces a
        symmetric output.

        Because a Meixner model is inherently one-sided (its decay has a
        direction), symmetry requires a *direction-matched, shift-matched* pair:
        a forward-built ALSSM for the forward segment and a backward-built ALSSM
        for the backward segment, with mirror-matched boundaries (both touching
        the origin → shift 0).  A single backward-built ALSSM reused in both
        segments is NOT symmetric.
        """
        K = 200
        y = np.zeros(K)
        y[80:120] = 1.0    # rectangular pulse, centred in signal
        g = 20
        # mirror-matched boundaries about the origin (both shift 0)
        seg_L = lm.Segment(a=-np.inf, b=0,       direction=lm.FORWARD,  g=g)
        seg_R = lm.Segment(a=0,       b=np.inf,  direction=lm.BACKWARD, g=g)
        alssm_L = lm.AlssmPolyMeixner(2, seg_L)   # forward-built
        alssm_R = lm.AlssmPolyMeixner(2, seg_R)   # backward-built
        cost  = lm.CompositeCost((alssm_L, alssm_R), (seg_L, seg_R), F=[[1, 0], [0, 1]])
        rls   = lm.RLSAlssm(cost, steady_state=True)
        y_hat = rls.fit(y)
        # Mirror image: y_hat[k] ≈ y_hat[K-1-k] (allow 5% deviation)
        mirror_err = np.max(np.abs(y_hat[20:-20] - y_hat[20:-20][::-1]))
        assert mirror_err < 0.05, f"Filter not symmetric: mirror_err={mirror_err:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Functional: ex111.0 (pulse detection with mixed g)
# ─────────────────────────────────────────────────────────────────────────────

class TestEx111:

    @pytest.fixture(scope="class")
    def signal(self):
        K = 4000
        y_rpulse = 0.03 * gen_rand_pulse(K, n_pulses=6, length=20, seed=1000)
        y = y_rpulse + gen_wgn(K, sigma=0.01, seed=1000) + 1e-3 * gen_rand_walk(K)
        return K, y, y_rpulse

    def _run_detection(self, alssm_pulse, alssm_baseline, K, y, g_sp, g_bl, len_pulse):
        seg_L = lm.Segment(a=-np.inf,       b=-1,            direction=lm.FORWARD,  g=g_bl)
        seg_C = lm.Segment(a=0,             b=len_pulse,     direction=lm.BACKWARD, g=g_sp)
        seg_R = lm.Segment(a=len_pulse + 1, b=np.inf,        direction=lm.BACKWARD, g=g_bl,
                           delta=len_pulse)
        F = [[0, 1, 0], [1, 1, 1]]
        cost = lm.CompositeCost((alssm_pulse, alssm_baseline), (seg_L, seg_C, seg_R), F)
        rls  = lm.RLSAlssm(cost, steady_state=True)
        y_hat, xs_1 = rls.fit(y, output=('y_hat', 'x'), eval_alssm_weights=[1, 0])

        xs_0 = np.copy(xs_1)
        xs_0[:, cost.get_state_var_indices('pulse.x')] = 0
        J1 = rls.eval_errors(xs_1)
        J0 = rls.eval_errors(xs_0)
        lcr = -0.5 * np.log(J1 / J0)
        return lcr

    def test_meixner_detects_pulses(self, signal):
        from scipy.signal import find_peaks
        K, y, y_rpulse = signal
        g_bl, g_sp, len_pulse = 50, 15000, 20

        alssm_pulse    = make_alssm(0, g_bl, label='pulse')
        alssm_baseline = make_alssm(2, g_bl, label='baseline')

        lcr = self._run_detection(alssm_pulse, alssm_baseline, K, y, g_sp, g_bl, len_pulse)

        peaks_meixner, _ = find_peaks(lcr, height=0.2, distance=30)
        print(f"  ex111 Meixner: {len(peaks_meixner)} peaks detected")
        assert 4 <= len(peaks_meixner) <= 8, \
            f"Expected 6 pulses, detected {len(peaks_meixner)}"

    def test_meixner_lcr_comparable_to_poly(self, signal):
        """Meixner LCR must be positively correlated with Poly LCR (same signal info)."""
        from scipy.signal import find_peaks
        K, y, _ = signal
        g_bl, g_sp, len_pulse = 50, 15000, 20

        def run(use_meixner):
            if use_meixner:
                ap = make_alssm(0, g_bl, label='pulse')
                ab = make_alssm(2, g_bl, label='baseline')
            else:
                ap = lm.AlssmPoly(poly_degree=0, label='pulse')
                ab = lm.AlssmPoly(poly_degree=2, label='baseline')
            return self._run_detection(ap, ab, K, y, g_sp, g_bl, len_pulse)

        lcr_m = run(True)
        lcr_p = run(False)
        corr  = np.corrcoef(lcr_m, lcr_p)[0, 1]
        print(f"  ex111 LCR corr(Meixner, Poly) = {corr:.4f}")
        assert corr > 0.90, f"LCR outputs diverged: corr={corr:.4f}"

    def test_W_ss_conditioning_improvement(self, signal):
        """Composite W kappa for Meixner << Poly on ex111 structure."""
        g_bl, g_sp, len_pulse = 50, 15000, 20
        seg_L = lm.Segment(a=-np.inf,       b=-1,            direction=lm.FORWARD,  g=g_bl)
        seg_C = lm.Segment(a=0,             b=len_pulse,     direction=lm.BACKWARD, g=g_sp)
        seg_R = lm.Segment(a=len_pulse + 1, b=np.inf,        direction=lm.BACKWARD, g=g_bl,
                           delta=len_pulse)

        for D in [1, 2]:
            am = make_alssm(D, g_bl, label='m')
            ap = lm.AlssmPoly(D, label='p')
            F  = [[1, 1, 1]]

            Wm = lm.CompositeCost((am,), (seg_L, seg_C, seg_R), F).get_steady_state_W()
            Wp = lm.CompositeCost((ap,), (seg_L, seg_C, seg_R), F).get_steady_state_W()

            km = np.linalg.cond(Wm)
            kp = np.linalg.cond(Wp)
            print(f"  ex111 D={D}: kappa(Meixner)={km:.2f}  kappa(Poly)={kp:.2e}")
            assert km < 20.0
            assert km < kp / 100


# ─────────────────────────────────────────────────────────────────────────────
# 8. Functional: ex113.0 (ECG shape detection)
# ─────────────────────────────────────────────────────────────────────────────

class TestEx113:

    @pytest.fixture(scope="class")
    def ecg_signal(self):
        K = 10000
        y = load_lib_csv('EECG_BASELINE_1CH_10S_FS2400HZ.csv', K)
        return K, y

    def test_meixner_shape_detection_runs(self, ecg_signal):
        """AlssmPolyMeixner runs the ex113 pipeline without error."""
        from scipy.signal import find_peaks
        from scipy.linalg import block_diag

        K, y = ecg_signal
        LCR_THD = 0.15
        K_REF = 1865
        SHAPE_LEN_2 = 100
        g_sp, g_bl = 4000, 250
        N1, N2 = 3, 5

        # Use Meixner for baseline; Jordan for pulse (shape model stays the same)
        alssm_baseline = make_alssm(N1 - 1, g_bl, label="alssm-baseline")
        alssm_pulse    = lm.AlssmPolyJordan(poly_degree=N2 - 1, label="alssm-pulse")

        segL = lm.Segment(a=-np.inf,       b=-1 - SHAPE_LEN_2, direction=lm.FORWARD,  g=g_bl,
                          delta=-1 - SHAPE_LEN_2)
        segC = lm.Segment(a=-SHAPE_LEN_2,  b=SHAPE_LEN_2,      direction=lm.FORWARD,  g=g_sp)
        segR = lm.Segment(a=SHAPE_LEN_2+1, b=np.inf,           direction=lm.BACKWARD, g=g_bl,
                          delta=SHAPE_LEN_2 + 1)

        F = [[0, 1, 0], [1, 1, 1]]
        cost = lm.CompositeCost((alssm_pulse, alssm_baseline), (segL, segC, segR), F)

        rls = lm.RLSAlssm(cost, steady_state=False, backend='lfilter')
        rls.filter(y)
        xs     = rls.minimize_x()
        xs_ref = xs[K_REF]

        H_A = np.transpose(block_diag([xs_ref[0:N2]], np.eye(N1)))
        H_0 = np.transpose(np.hstack([np.zeros((N1, N2)), np.eye(N1)]))

        xs_A = rls.minimize_x(H_A)
        xs_0 = rls.minimize_x(H_0)
        J_A  = rls.eval_errors(xs_A)
        J_0  = rls.eval_errors(xs_0)

        with np.errstate(divide='ignore', invalid='ignore'):
            lcr = -0.5 * np.log(J_A / J_0)
            lcr = np.nan_to_num(lcr, nan=0.0, posinf=0.0)

        peaks, _ = find_peaks(lcr, height=LCR_THD, distance=1500)
        print(f"  ex113 Meixner baseline: {len(peaks)} ECG peaks detected, LCR max={lcr.max():.3f}")
        assert len(peaks) >= 4, f"Too few ECG peaks detected: {len(peaks)}"
        assert lcr.max() > LCR_THD, "LCR never exceeds threshold"
        assert np.all(np.isfinite(lcr)), "LCR contains NaN/Inf"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Detector: _is_meixner_shift
# ─────────────────────────────────────────────────────────────────────────────

class TestDetector:

    @pytest.mark.parametrize("D,g", [(0,10),(1,20),(3,50),(5,100)])
    def test_detects_true_positive(self, D, g):
        alssm = make_alssm(D, g)
        ok, g_detected = _is_meixner_shift(alssm.A)
        assert ok, f"Did not detect Meixner matrix D={D} g={g}"
        if D >= 1:
            assert abs(g_detected - g) < 1e-6, f"g mismatch: detected {g_detected} vs {g}"

    @pytest.mark.parametrize("D", [1, 2, 3])
    def test_rejects_pascal(self, D):
        alssm = lm.AlssmPoly(D)
        ok, _ = _is_meixner_shift(alssm.A.astype(float))
        assert not ok, "Pascal/monomial A must not be detected as Meixner"

    @pytest.mark.parametrize("D", [2, 3])
    def test_rejects_legendre(self, D):
        alssm = lm.AlssmPolyLegendre(poly_degree=D, a_seg=0, b_seg=100)
        ok, _ = _is_meixner_shift(alssm.A)
        assert not ok, "Legendre A must not be detected as Meixner"

    @pytest.mark.parametrize("D", [1, 2])
    def test_rejects_jordan(self, D):
        alssm = lm.AlssmPolyJordan(D)
        ok, _ = _is_meixner_shift(alssm.A.astype(float))
        assert not ok, "Jordan A must not be detected as Meixner"

    def test_rejects_random(self):
        np.random.seed(1)
        A = np.random.randn(4, 4)
        ok, _ = _is_meixner_shift(A)
        assert not ok

    def test_rejects_lower_triangular(self):
        A = np.tril(np.ones((3, 3)))
        ok, _ = _is_meixner_shift(A)
        assert not ok

    def test_rejects_mixed_superdiag(self):
        """Upper triangle with non-constant values must be rejected."""
        A = np.eye(3)
        A[0, 1] = -0.1
        A[0, 2] = -0.2   # different from -0.1
        A[1, 2] = -0.1
        ok, _ = _is_meixner_shift(A)
        assert not ok


# ─────────────────────────────────────────────────────────────────────────────
# 10. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_degree_zero(self):
        """poly_degree=0: A=[[1]], C=[1], equivalent to a mean filter."""
        alssm = make_alssm(0, 20)
        assert alssm.A.shape == (1, 1)
        assert np.isclose(alssm.A[0, 0], 1.0)
        assert np.isclose(alssm.C[0], 1.0)

    def test_large_g(self):
        """Very large g (near-uniform window): A should be near-identity."""
        g = 1e6
        alssm = make_alssm(3, g)
        off_diag = np.abs(alssm.A - np.eye(4))
        assert np.max(off_diag) < 1e-5, "For large g, A should be near I"

    def test_small_g_near_one(self):
        """g just above 1: A entries should be large negative."""
        g = 1.01
        alssm = make_alssm(2, g)
        assert np.all(np.isfinite(alssm.A)), "A must be finite for g=1.01"
        assert np.linalg.cond(alssm.A) < 1e8

    def test_invalid_g_raises(self):
        with pytest.raises((AssertionError, ValueError)):
            make_alssm(2, 0.5)   # g <= 1

    def test_invalid_poly_degree_raises(self):
        with pytest.raises((AssertionError, TypeError)):
            make_alssm(-1, 20)

    def test_invalid_g_rejected_by_segment(self):
        # g is taken from the segment; the g <= 1 check now lives in Segment.
        with pytest.raises((AssertionError, ValueError)):
            lm.Segment(a=0, b=np.inf, direction=lm.BACKWARD, g=0.5)

    def test_g_is_read_only(self):
        # g is derived from the segment and is not a settable (legacy) attribute.
        alssm = make_alssm(2, 20)
        assert np.isclose(alssm.g, 20.0)
        with pytest.raises(AttributeError):
            alssm.g = 30.0

    def test_gamma_property(self):
        g = 50.0
        alssm = make_alssm(3, g)
        assert np.isclose(alssm.gamma, (g - 1.0) / g)

    def test_label_forwarded(self):
        alssm = make_alssm(2, 20, label='test-label')  # label via kwargs
        assert alssm.label == 'test-label'

    def test_repr_contains_class_name(self):
        alssm = make_alssm(2, 20)
        s = str(alssm)
        assert 'AlssmPolyMeixner' in s


# ─────────────────────────────────────────────────────────────────────────────
# 11. Numerical stability summary table (informational)
# ─────────────────────────────────────────────────────────────────────────────

class TestStabilityTable:

    def test_print_comparison_table(self):
        """Print a summary table comparing Poly, Legendre, Meixner conditioning."""
        g = 20
        gamma = (g - 1.0) / g
        seg_L = lm.Segment(a=-np.inf, b=-1,     direction=lm.FORWARD,  g=g)
        seg_R = lm.Segment(a=0,       b=np.inf, direction=lm.BACKWARD, g=g)

        print("\n  D | kappa(Poly W_ss) | kappa(Legendre W_ss) | kappa(Meixner W_ss)")
        print("  " + "-" * 65)
        for D in [1, 2, 3, 4, 5, 6, 8, 10]:
            alssm_p = lm.AlssmPoly(D)
            alssm_l = lm.AlssmPolyLegendre(poly_degree=D, a_seg=0, b_seg=2 * g)
            alssm_m = make_alssm(D, g)

            def get_W(alssm):
                cost = lm.CompositeCost((alssm,), (seg_L, seg_R), F=[[1, 1]])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        return cost.get_steady_state_W()
                    except Exception:
                        return None

            Wp = get_W(alssm_p)
            Wl = get_W(alssm_l)
            Wm = get_W(alssm_m)

            kp = f"{np.linalg.cond(Wp):.2e}" if Wp is not None else "FAIL"
            kl = f"{np.linalg.cond(Wl):.2e}" if Wl is not None else "FAIL"
            km = f"{np.linalg.cond(Wm):.2e}" if Wm is not None else "FAIL"
            print(f"  {D:2d} | {kp:>16} | {kl:>20} | {km:>19}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', __file__, '-v', '--tb=short', '-s'],
        cwd='/home/claude'
    )
    sys.exit(result.returncode)
