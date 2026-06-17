"""
Tests for the GPU (cupy / cupyx) backend.

The GPU backend is a faithful port of the ``lfilter`` cascade path, so the
canonical correctness check is **parity**: for every configuration the GPU
result must match the ``lfilter`` backend (and hence ``numpy``) to within
floating-point tolerance.

These tests are skipped automatically when the ``cupy`` backend is not available
(no cupy package or no visible CUDA device), mirroring the ``skip_no_jit``
convention in ``test_rls.py``.
"""
import unittest
import numpy as np
import lmlib as lm

CUPY_AVAILABLE = lm.is_backend_available('cupy')
skip_no_cupy = unittest.skipUnless(
    CUPY_AVAILABLE, "cupy backend unavailable (no cupy package or no CUDA device)")

# tolerance: lfilter (scipy) vs cupyx may differ only by float rounding
ATOL = 1e-8
RTOL = 1e-6


@skip_no_cupy
class TestCupyBackend(unittest.TestCase):

    def _parity(self, cost, y, sample_weights=None, steady_state=True):
        ref = lm.RLSAlssm(cost, steady_state=steady_state, backend='lfilter')
        gpu = lm.RLSAlssm(cost, steady_state=steady_state, backend='cupy')
        ref.filter(y, sample_weights=sample_weights)
        gpu.filter(y, sample_weights=sample_weights)
        np.testing.assert_allclose(np.asarray(gpu.xi),    np.asarray(ref.xi),    rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.asarray(gpu.kappa), np.asarray(ref.kappa), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.asarray(gpu.W),     np.asarray(ref.W),     rtol=RTOL, atol=ATOL)
        return ref, gpu

    # ── the gu130.0 guide configuration ──────────────────────────────────────
    def test_guide_config(self):
        alssm = lm.AlssmPoly(poly_degree=1)
        seg = lm.Segment(a=-21, b=-1, direction=lm.FW, g=100)
        cost = lm.CompositeCost([alssm], [seg], F=[[1]])
        np.random.seed(0)
        self._parity(cost, np.random.randn(5000))

    # ── forward segment, several model orders ────────────────────────────────
    def test_forward_orders(self):
        for order in (0, 1, 2, 3):
            alssm = lm.AlssmPoly(poly_degree=order)
            seg = lm.Segment(a=-20, b=-1, direction=lm.FW, g=30)
            cost = lm.CompositeCost([alssm], [seg], F=[[1]])
            y = np.sin(np.linspace(0, 6 * np.pi, 2000))
            self._parity(cost, y)

    # ── backward segment ─────────────────────────────────────────────────────
    def test_backward(self):
        alssm = lm.AlssmPoly(poly_degree=2)
        seg = lm.Segment(a=0, b=20, direction=lm.BW, g=30)
        cost = lm.CompositeCost([alssm], [seg], F=[[1]])
        y = np.cos(np.linspace(0, 6 * np.pi, 2000))
        self._parity(cost, y)

    # ── two-sided composite cost (fw + bw), downstream minimize_x ─────────────
    def test_two_sided_minimize(self):
        y = np.sin(np.linspace(0, 2 * np.pi, 40))
        alssm = lm.AlssmPoly(poly_degree=2)
        sfw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        sbw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [sfw, sbw], F=[[1, 1]])
        ref, gpu = self._parity(cost, y)
        e_ref = ref.eval_errors(ref.minimize_x())
        e_gpu = gpu.eval_errors(gpu.minimize_x())
        np.testing.assert_allclose(e_gpu, e_ref, rtol=RTOL, atol=ATOL)

    # ── time-varying W (q==2 cascade path, steady_state=False) ────────────────
    def test_time_varying_W(self):
        alssm = lm.AlssmPoly(poly_degree=2)
        seg = lm.Segment(a=-25, b=-1, direction=lm.FW, g=20)
        cost = lm.CompositeCost([alssm], [seg], F=[[1]])
        y = np.sin(np.linspace(0, 8 * np.pi, 1500))
        self._parity(cost, y, steady_state=False)

    # ── custom per-sample weights ─────────────────────────────────────────────
    def test_sample_weights(self):
        alssm = lm.AlssmPoly(poly_degree=2)
        seg = lm.Segment(a=-15, b=-1, direction=lm.FW, g=20)
        cost = lm.CompositeCost([alssm], [seg], F=[[1]])
        y = np.sin(np.linspace(0, 8 * np.pi, 2000))
        w = np.abs(np.cos(np.linspace(0, 3, 2000)))
        self._parity(cost, y, sample_weights=w)

    # ── batched multichannel (the key #3 path): cupy processes all channels in
    #    one GPU sweep; must match the lfilter per-channel result ──────────────
    def test_multichannel_forward(self):
        np.random.seed(0)
        for S in (1, 2, 8, 33):
            alssm = lm.AlssmPoly(poly_degree=2)
            seg = lm.Segment(a=-21, b=-1, direction=lm.FW, g=100)
            cost = lm.CompositeCost([alssm], [seg], F=[[1]])
            Y = np.random.randn(1500, S)
            self._parity(cost, Y)

    def test_multichannel_backward(self):
        np.random.seed(1)
        alssm = lm.AlssmPoly(poly_degree=2)
        seg = lm.Segment(a=0, b=21, direction=lm.BW, g=100)
        cost = lm.CompositeCost([alssm], [seg], F=[[1]])
        Y = np.random.randn(1500, 6)
        self._parity(cost, Y)

    def test_multichannel_weights_and_nonss(self):
        np.random.seed(2)
        alssm = lm.AlssmPoly(poly_degree=2)
        seg = lm.Segment(a=-25, b=-1, direction=lm.FW, g=20)
        cost = lm.CompositeCost([alssm], [seg], F=[[1]])
        Y = np.random.randn(1200, 5)
        W = np.abs(np.random.rand(1200, 5))
        self._parity(cost, Y, sample_weights=W)            # per-(K,S) weights
        self._parity(cost, Y, steady_state=False)          # batched q=2 cascade

    def test_multichannel_minimize_x(self):
        Y = np.sin(np.linspace(0, 2 * np.pi, 40))[:, None] * np.array([1.0, 0.5, 2.0])
        alssm = lm.AlssmPoly(poly_degree=2)
        sfw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        sbw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [sfw, sbw], F=[[1, 1]])
        ref, gpu = self._parity(cost, Y)
        np.testing.assert_allclose(gpu.minimize_x(), ref.minimize_x(), rtol=RTOL, atol=ATOL)

    # ── float32 device path (lever #1): faster, ~1e-6 relative accuracy ───────
    def test_float32_precision(self):
        # default float64 must stay exact; float32 must be small but nonzero.
        np.random.seed(0)
        Y = np.random.randn(4000, 16)
        alssm = lm.AlssmPoly(poly_degree=1)
        seg = lm.Segment(a=-21, b=-1, direction=lm.FW, g=100)
        cost = lm.CompositeCost([alssm], [seg], F=[[1]])
        ref = lm.RLSAlssm(cost, backend='lfilter'); ref.filter(Y)
        try:
            lm.set_gpu_dtype('float64')
            g64 = lm.RLSAlssm(cost, backend='cupy'); g64.filter(Y)
            np.testing.assert_allclose(np.asarray(g64.xi), np.asarray(ref.xi), rtol=RTOL, atol=ATOL)

            lm.set_gpu_dtype('float32')
            g32 = lm.RLSAlssm(cost, backend='cupy'); g32.filter(Y)
            rel = np.max(np.abs(np.asarray(ref.xi) - np.asarray(g32.xi))) / np.max(np.abs(np.asarray(ref.xi)))
            self.assertLess(rel, 1e-3, f"float32 relative error too large: {rel:.2e}")
            self.assertGreater(rel, 0.0, "float32 path appears identical to float64 (dtype not applied?)")
        finally:
            lm.set_gpu_dtype('float64')  # never leak precision state into other tests

    # ── parallel filter form on GPU (vs lfilter parallel) ─────────────────────
    def _parity_parallel(self, cost, y, sample_weights=None):
        ref = lm.RLSAlssm(cost, backend='lfilter', filter_form='parallel')
        gpu = lm.RLSAlssm(cost, backend='cupy', filter_form='parallel')
        ref.filter(y, sample_weights=sample_weights)
        gpu.filter(y, sample_weights=sample_weights)
        np.testing.assert_allclose(np.asarray(gpu.xi),    np.asarray(ref.xi),    rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.asarray(gpu.kappa), np.asarray(ref.kappa), rtol=RTOL, atol=ATOL)
        return ref, gpu

    def test_parallel_poly_orders(self):
        for order in (1, 2, 3):
            alssm = lm.AlssmPoly(poly_degree=order)
            seg = lm.Segment(a=-21, b=-1, direction=lm.FW, g=100)
            cost = lm.CompositeCost([alssm], [seg], F=[[1]])
            np.random.seed(order)
            self._parity_parallel(cost, np.random.randn(3000))

    def test_parallel_backward(self):
        alssm = lm.AlssmPoly(poly_degree=2)
        seg = lm.Segment(a=0, b=21, direction=lm.BW, g=100)
        cost = lm.CompositeCost([alssm], [seg], F=[[1]])
        np.random.seed(0)
        self._parity_parallel(cost, np.random.randn(3000))

    def test_parallel_alssmsin_complex_poles(self):
        # non-upper-triangular model with complex-conjugate poles: the genuine
        # reason the parallel form exists; exercises the sosfilt (non gamma-shift) path.
        sin = lm.AlssmSin(omega=0.1, rho=0.99)
        seg = lm.Segment(a=-40, b=-1, direction=lm.FW, g=200)
        cost = lm.CompositeCost([sin], [seg], F=[[1]])
        np.random.seed(0)
        self._parity_parallel(cost, np.random.randn(3000))

    def test_parallel_minimize_x(self):
        y = np.sin(np.linspace(0, 2 * np.pi, 40))
        alssm = lm.AlssmPoly(poly_degree=2)
        sfw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10)
        sbw = lm.Segment(a=0, b=5, direction=lm.BW, g=10)
        cost = lm.CompositeCost([alssm], [sfw, sbw], F=[[1, 1]])
        ref, gpu = self._parity_parallel(cost, y)
        np.testing.assert_allclose(gpu.minimize_x(), ref.minimize_x(), rtol=RTOL, atol=ATOL)

    # ── N-dimensional (asterisk-l) recursion on GPU vs numpy ──────────────────
    @staticmethod
    def _mk_nd_cost(pd, g=5):
        alssm = lm.AlssmPoly(poly_degree=pd)
        sfw = lm.Segment(a=-4, b=-1, direction=lm.FW, g=g, delta=0)
        sbw = lm.Segment(a=0, b=4, direction=lm.BW, g=g, delta=0)
        return lm.CompositeCost([alssm], [sfw, sbw], F=[[1, 1]])

    def _parity_nd(self, costs, Y, dim_order, filter_form):
        nd = lm.NDCompositeCost(costs)
        rn = lm.RLSAlssm(nd, steady_state=True, backend='numpy')
        rg = lm.RLSAlssm(nd, steady_state=True, backend='cupy', filter_form=filter_form)
        rn.filter(Y, dim_order=dim_order)
        rg.filter(Y, dim_order=dim_order)
        np.testing.assert_allclose(np.asarray(rg.xi), np.asarray(rn.xi), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.asarray(rg.kappa), np.asarray(rn.kappa), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(rg.eval_errors(rg.minimize_x()),
                                   rn.eval_errors(rn.minimize_x()), rtol=RTOL, atol=ATOL)

    def test_nd_2d_equal_order(self):
        rng = np.random.default_rng(0); Y = rng.standard_normal((15, 15))
        for form in ('cascade', 'parallel'):
            self._parity_nd([self._mk_nd_cost(2), self._mk_nd_cost(2)], Y, [0, 1], form)

    def test_nd_2d_unequal_order(self):
        rng = np.random.default_rng(1); Y = rng.standard_normal((15, 15))
        for form in ('cascade', 'parallel'):
            self._parity_nd([self._mk_nd_cost(1), self._mk_nd_cost(2)], Y, [0, 1], form)
            self._parity_nd([self._mk_nd_cost(3), self._mk_nd_cost(1)], Y, [0, 1], form)

    def test_nd_3d(self):
        rng = np.random.default_rng(2); Y = rng.standard_normal((10, 10, 10))
        for form in ('cascade', 'parallel'):
            self._parity_nd([self._mk_nd_cost(1), self._mk_nd_cost(2), self._mk_nd_cost(1)],
                            Y, [0, 1, 2], form)

    def test_nd_dim_order_permutation(self):
        rng = np.random.default_rng(3); Y = rng.standard_normal((15, 15))
        for form in ('cascade', 'parallel'):
            self._parity_nd([self._mk_nd_cost(1), self._mk_nd_cost(2)], Y, [1, 0], form)


if __name__ == '__main__':
    unittest.main()
