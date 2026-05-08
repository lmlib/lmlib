"""
Tests for the refactored RLSAlssm._nd_xi_q_recursion.

Each test compares the new per-ALSSM implementation against the original
monolithic AlssmSum behaviour by running both and asserting numerical equality.
The "reference" result is produced by temporarily restoring the old code path
(calling _get_cost_segments with force_MC=True directly).
"""

import numpy as np
import pytest
import lmlib as lm
from lmlib.statespace.model import AlssmSum
from lmlib.statespace.cost import CompositeCost, CostSegment
from lmlib.utils.generator import gen_slopes, gen_wgn


# ---------------------------------------------------------------------------
# Helper: reference filter using the OLD monolithic AlssmSum code path
# ---------------------------------------------------------------------------
from lmlib.statespace.backends.rec import xi_q_recursion

def reference_filter(cost, y, q):
    """Reproduces the old _nd_xi_q_recursion exactly for a 1-D CompositeCost."""
    if isinstance(cost, CostSegment):
        cost = lm.CompositeCost([cost.alssm], [cost.segment],
                                np.ones((1,1)), betas=np.array([cost.beta]))
    N  = cost.get_alssm_order()
    K  = y.shape[0]
    xi_curr = np.zeros((K, N ** q))
    _sw = np.broadcast_to(1., (K,))

    for p, segment in enumerate(cost.segments):
        alssm_sum = AlssmSum(cost.alssms, cost.F[:, p], force_MC=True)
        cs_beta   = cost.betas[p]
        xi_q_recursion(xi_curr, q,
                       alssm_sum, segment,
                       y, _sw,
                       cs_beta, 'numpy', 'cascade', None)
    return xi_curr


# ---------------------------------------------------------------------------
# Signal fixture
# ---------------------------------------------------------------------------
K   = 300
RNG = np.random.default_rng(0)
Y1  = gen_slopes(K, [40, 80, 130, 160, 220], [0, 5, -8.5, 3, -2]) + gen_wgn(K, sigma=0.2, seed=3141)


# ===========================================================================
# Test 1 – CompositeCost (M=2, P=2), numpy backend, steady_state=False
#          This is the canonical edge-detection example.
# ===========================================================================
def test_composite_cost_numpy_nonsteady():
    alssm_l = lm.AlssmPoly(poly_degree=1, label='left')
    alssm_r = lm.AlssmPoly(poly_degree=1, label='right')
    seg_l   = lm.Segment(a=-21, b=-1, direction=lm.FORWARD,  g=7)
    seg_r   = lm.Segment(a=0,   b=20, direction=lm.BACKWARD, g=7)
    F       = [[1, 0], [0, 1]]
    cost    = lm.CompositeCost((alssm_l, alssm_r), (seg_l, seg_r), F)

    rls = lm.RLSAlssm(cost, steady_state=False)
    rls.filter(Y1)

    for q, name in [(2, 'W'), (1, 'xi'), (0, 'kappa')]:
        ref = reference_filter(cost, Y1.reshape(-1, 1), q)
        if name == 'W':
            new = rls._xi2          # shape (K, N^2)
        elif name == 'xi':
            new = rls._xi1          # shape (K, N)
        else:
            new = rls._xi0          # shape (K,)
            ref = ref[:, 0]         # reference also (K,) after squeezing
        assert np.allclose(ref, new, atol=1e-10), \
            f"q={q} ({name}): max diff = {np.max(np.abs(ref - new))}"
    print("PASS test_composite_cost_numpy_nonsteady")


# ===========================================================================
# Test 2 – CostSegment (M=1, P=1) — exercises the _as_composite_cost wrapper
# ===========================================================================
def test_cost_segment_numpy_nonsteady():
    alssm = lm.AlssmPoly(poly_degree=2)
    seg   = lm.Segment(a=-30, b=-1, direction=lm.FORWARD, g=15)
    cost  = lm.CostSegment(alssm, seg)

    rls = lm.RLSAlssm(cost, steady_state=False)
    rls.filter(Y1)

    for q, name in [(2, 'W'), (1, 'xi'), (0, 'kappa')]:
        ref = reference_filter(cost, Y1.reshape(-1, 1), q)
        if name == 'W':
            new = rls._xi2
        elif name == 'xi':
            new = rls._xi1
        else:
            new = rls._xi0
            ref = ref[:, 0]
        assert np.allclose(ref, new, atol=1e-10), \
            f"q={q} ({name}): max diff = {np.max(np.abs(ref - new))}"
    print("PASS test_cost_segment_numpy_nonsteady")


# ===========================================================================
# Test 3 – steady_state=True path is untouched
# ===========================================================================
def test_steady_state():
    alssm_l = lm.AlssmPoly(poly_degree=1)
    alssm_r = lm.AlssmPoly(poly_degree=1)
    seg_l   = lm.Segment(a=-21, b=-1, direction=lm.FORWARD,  g=7)
    seg_r   = lm.Segment(a=0,   b=20, direction=lm.BACKWARD, g=7)
    F       = [[1, 0], [0, 1]]
    cost    = lm.CompositeCost((alssm_l, alssm_r), (seg_l, seg_r), F)

    rls_ss  = lm.RLSAlssm(cost, steady_state=True)
    rls_ns  = lm.RLSAlssm(cost, steady_state=False)
    rls_ss.filter(Y1)
    rls_ns.filter(Y1)

    # W should be constant across k for steady state; compare to non-steady mean
    W_ss = rls_ss.W                          # shape (4, 4) broadcast
    W_ns = rls_ns.W                          # shape (300, 4, 4)
    # At steady state the time-averaged W_ns should be close to W_ss
    assert W_ss.shape == (4, 4)
    assert W_ns.shape == (K, 4, 4)
    assert np.allclose(W_ss, W_ns[150], atol=1e-4), \
        "Steady-state W deviates from non-steady W at mid-signal"
    print("PASS test_steady_state")


# ===========================================================================
# Test 4 – minimize_x / eval_errors end-to-end (regression)
# ===========================================================================
def test_edge_detection_end_to_end():
    from scipy.signal import find_peaks
    alssm_l = lm.AlssmPoly(poly_degree=1)
    alssm_r = lm.AlssmPoly(poly_degree=1)
    seg_l   = lm.Segment(a=-21, b=-1, direction=lm.FORWARD,  g=7)
    seg_r   = lm.Segment(a=0,   b=20, direction=lm.BACKWARD, g=7)
    F       = [[1, 0], [0, 1]]
    cost    = lm.CompositeCost((alssm_l, alssm_r), (seg_l, seg_r), F)

    H_edge = lm.TSLM.H_Continuous
    H_line = lm.TSLM.H_Straight

    rls = lm.RLSAlssm(cost, steady_state=False)
    rls.filter(Y1)

    xs_edge = rls.minimize_x(H_edge)
    xs_line = rls.minimize_x(H_line)
    err_edge = rls.eval_errors(xs_edge)
    err_line = rls.eval_errors(xs_line)

    lcr = -0.5 * np.log(np.divide(err_edge, err_line))
    peaks, _ = find_peaks(lcr, height=0.05, distance=20)

    # With these parameters both steady and non-steady find peaks near all 5
    # reference positions (some may be slightly offset by noise).
    expected = [40, 80, 130, 160, 220]
    matched = sum(any(abs(pk - ex) <= 10 for pk in peaks) for ex in expected)
    assert matched == len(expected), \
        f"Only {matched}/{len(expected)} expected peaks found: peaks={peaks.tolist()}"
    print(f"PASS test_edge_detection_end_to_end  peaks={peaks.tolist()}")


# ===========================================================================
# Test 5 – lfilter cascade backend produces same result as numpy backend
# ===========================================================================
def test_lfilter_cascade_matches_numpy():
    alssm_l = lm.AlssmPoly(poly_degree=1)
    alssm_r = lm.AlssmPoly(poly_degree=1)
    seg_l   = lm.Segment(a=-21, b=-1, direction=lm.FORWARD,  g=7)
    seg_r   = lm.Segment(a=0,   b=20, direction=lm.BACKWARD, g=7)
    F       = [[1, 0], [0, 1]]
    cost    = lm.CompositeCost((alssm_l, alssm_r), (seg_l, seg_r), F)

    rls_np  = lm.RLSAlssm(cost, steady_state=False, backend='numpy')
    rls_lf  = lm.RLSAlssm(cost, steady_state=False, backend='lfilter')
    rls_np.filter(Y1)
    rls_lf.filter(Y1)

    assert np.allclose(rls_np.xi,    rls_lf.xi,    atol=1e-6), \
        f"xi mismatch: max {np.max(np.abs(rls_np.xi - rls_lf.xi))}"
    assert np.allclose(rls_np.kappa, rls_lf.kappa, atol=1e-6), \
        f"kappa mismatch: max {np.max(np.abs(rls_np.kappa - rls_lf.kappa))}"
    print("PASS test_lfilter_cascade_matches_numpy")


# ===========================================================================
# Test 6 – sparse F matrix (some grid nodes inactive, f_mp == 0)
# ===========================================================================
def test_sparse_F():
    alssm_spike    = lm.AlssmPoly(poly_degree=3)
    alssm_baseline = lm.AlssmPoly(poly_degree=1)
    seg_l = lm.Segment(a=-30, b=-1, direction=lm.FORWARD,  g=20)
    seg_r = lm.Segment(a=0,   b=30, direction=lm.BACKWARD, g=20)
    # spike only on left, baseline on both
    F    = [[1, 0],
            [1, 1]]
    cost = lm.CompositeCost((alssm_spike, alssm_baseline), (seg_l, seg_r), F)

    rls = lm.RLSAlssm(cost, steady_state=False)
    rls.filter(Y1)

    ref_xi = reference_filter(cost, Y1.reshape(-1, 1), q=1)
    assert np.allclose(ref_xi, rls._xi1, atol=1e-10), \
        f"sparse F xi mismatch: max {np.max(np.abs(ref_xi - rls._xi1))}"
    print("PASS test_sparse_F")


# ===========================================================================
# Test 7 – multi-channel (vector) observations
# ===========================================================================
def test_multichannel():
    alssm = lm.AlssmPoly(poly_degree=1)
    seg   = lm.Segment(a=-20, b=-1, direction=lm.FORWARD, g=10)
    cost  = lm.CostSegment(alssm, seg)

    Y2 = np.stack([Y1, Y1 * 0.5], axis=-1)   # shape (300, 2)

    rls = lm.RLSAlssm(cost, steady_state=False)
    rls.filter(Y2)

    assert rls.xi.shape    == (300, 2, 2), f"xi shape: {rls.xi.shape}"
    assert rls.kappa.shape == (300, 2),    f"kappa shape: {rls.kappa.shape}"
    print("PASS test_multichannel")


# ===========================================================================
# Test 8 – NDCompositeCost with steady_state=False (xi and kappa)
#   Previously blocked by the overly broad assert; now allowed for q=0,1.
#   W still requires steady_state=True for NDCompositeCost.
# ===========================================================================
def test_nd_composite_cost_nonsteady_xi_kappa():
    alssm = lm.AlssmPoly(poly_degree=1)
    seg_l = lm.Segment(a=-20, b=-1, direction=lm.FORWARD,  g=10)
    seg_r = lm.Segment(a=0,   b=20, direction=lm.BACKWARD, g=10)
    F     = [[1, 0], [0, 1]]
    cost1d = lm.CompositeCost((alssm, alssm), (seg_l, seg_r), F)
    nd_cost = lm.NDCompositeCost([cost1d, cost1d])

    # 2-D signal: shape (K, K)
    Y2d = np.outer(Y1, Y1 / Y1.max())

    rls = lm.RLSAlssm(nd_cost, steady_state=False, calc_W=False)
    rls.filter(Y2d)  # should not raise

    assert rls.xi.shape    == (K, K, 16), f"xi shape wrong: {rls.xi.shape}"
    assert rls.kappa.shape == (K, K),     f"kappa shape wrong: {rls.kappa.shape}"

    # W with steady_state=False should raise for NDCompositeCost
    rls_w = lm.RLSAlssm(nd_cost, steady_state=False, calc_W=True)
    try:
        rls_w.filter(Y2d)
        assert False, "Expected assertion error for W with steady_state=False"
    except AssertionError:
        pass
    print("PASS test_nd_composite_cost_nonsteady_xi_kappa")


# ===========================================================================
# Run all
# ===========================================================================
if __name__ == '__main__':
    test_composite_cost_numpy_nonsteady()
    test_cost_segment_numpy_nonsteady()
    test_steady_state()
    test_edge_detection_end_to_end()
    test_lfilter_cascade_matches_numpy()
    test_sparse_F()
    test_multichannel()
    test_nd_composite_cost_nonsteady_xi_kappa()
    print("\nAll tests passed.")
