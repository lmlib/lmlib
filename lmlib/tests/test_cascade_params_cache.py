"""
Tests for the static cascade-parameter cache
=============================================

The lfilter cascade backend memoises the result of
``_compute_cascade_params(A, C, a, b, delta, gamma, direction)`` in a single
process-wide cache shared by every RLSAlssm instance.  The cached value depends
only on the ALSSM model and the segment window, never on the signal, so:

  * repeated ``filter()`` calls on one object reuse the entry,
  * different RLSAlssm objects with the same model+window share one entry, and
  * the numerical result is bit-for-bit identical to the uncached computation
    and to the independent numpy backend.

These tests cover the cache primitives (key construction, hit/miss, clearing)
and the end-to-end behaviour through ``RLSAlssm.filter`` (sharing + correctness).
"""

import unittest
import warnings

import numpy as np

import lmlib as lm
from lmlib.statespace.backends import rec_lfilter as rl


class TestCascadeParamsCachePrimitives(unittest.TestCase):
    """Unit tests for the cache helpers in rec_lfilter."""

    def setUp(self):
        rl.clear_lfilter_caches()
        self.A = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.C = np.array([[1.0, 0.0]])
        self.args = (0, 10, 0, 0.9, 'fw')  # a, b, delta, gamma, direction

    def test_starts_empty_after_clear(self):
        self.assertEqual(rl.lfilter_cache_info()['size'], 0)

    def test_miss_then_hit_same_object(self):
        p1 = rl._compute_cascade_params(self.A, self.C, *self.args)
        self.assertEqual(rl.lfilter_cache_info()['size'], 1)
        p2 = rl._compute_cascade_params(self.A, self.C, *self.args)
        # A second identical call must not grow the cache and must return the
        # very same (shared) dict object.
        self.assertEqual(rl.lfilter_cache_info()['size'], 1)
        self.assertIs(p1, p2)

    def test_content_equal_arrays_hit_same_entry(self):
        """Distinct array objects with identical contents map to one entry."""
        p1 = rl._compute_cascade_params(self.A, self.C, *self.args)
        p2 = rl._compute_cascade_params(self.A.copy(), self.C.copy(), *self.args)
        self.assertIs(p1, p2)
        self.assertEqual(rl.lfilter_cache_info()['size'], 1)

    def test_different_args_create_new_entries(self):
        rl._compute_cascade_params(self.A, self.C, 0, 10, 0, 0.9, 'fw')
        rl._compute_cascade_params(self.A, self.C, 0, 10, 0, 0.9, 'bw')   # direction
        rl._compute_cascade_params(self.A, self.C, 0, 10, 0, 0.8, 'fw')   # gamma
        rl._compute_cascade_params(self.A, self.C, 1, 10, 0, 0.9, 'fw')   # a
        self.assertEqual(rl.lfilter_cache_info()['size'], 4)

    def test_cached_matches_uncached(self):
        """The cached path returns the same numbers as the pure computation."""
        for direction, a, b in (('fw', 0, 10), ('bw', 0, 10),
                                 ('fw', -np.inf, 0), ('bw', 0, np.inf)):
            with self.subTest(direction=direction, a=a, b=b):
                rl.clear_lfilter_caches()
                cached = rl._compute_cascade_params(self.A, self.C, a, b, 0, 0.9, direction)
                uncached = rl._compute_cascade_params_uncached(
                    self.A, self.C, a, b, 0, 0.9, direction)
                self.assertEqual(set(cached), set(uncached))
                for k in uncached:
                    np.testing.assert_allclose(cached[k], uncached[k])

    def test_clear_empties_cache(self):
        rl._compute_cascade_params(self.A, self.C, *self.args)
        self.assertGreater(rl.lfilter_cache_info()['size'], 0)
        rl.clear_lfilter_caches()
        self.assertEqual(rl.lfilter_cache_info()['size'], 0)


class TestCascadeParamsAsteriskCachePrimitives(unittest.TestCase):
    """Unit tests for the asterisk-l cache helpers in rec_lfilter."""

    def setUp(self):
        rl.clear_lfilter_caches()
        self.A = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.C = np.array([[1.0, 0.0]])
        # a, b, delta, gamma, Nprev, direction
        self.args = (0, 10, 0, 0.9, 3, 'fw')

    def test_starts_empty_after_clear(self):
        self.assertEqual(rl.lfilter_cache_info()['asterisk_size'], 0)

    def test_miss_then_hit_same_object(self):
        p1 = rl._compute_cascade_params_asterisk(self.A, self.C, *self.args)
        self.assertEqual(rl.lfilter_cache_info()['asterisk_size'], 1)
        p2 = rl._compute_cascade_params_asterisk(self.A, self.C, *self.args)
        self.assertEqual(rl.lfilter_cache_info()['asterisk_size'], 1)
        self.assertIs(p1, p2)

    def test_content_equal_arrays_hit_same_entry(self):
        p1 = rl._compute_cascade_params_asterisk(self.A, self.C, *self.args)
        p2 = rl._compute_cascade_params_asterisk(self.A.copy(), self.C.copy(), *self.args)
        self.assertIs(p1, p2)
        self.assertEqual(rl.lfilter_cache_info()['asterisk_size'], 1)

    def test_nprev_is_part_of_the_key(self):
        """Different Nprev must not collide (the stored params depend on it)."""
        rl._compute_cascade_params_asterisk(self.A, self.C, 0, 10, 0, 0.9, 3, 'fw')
        rl._compute_cascade_params_asterisk(self.A, self.C, 0, 10, 0, 0.9, 4, 'fw')
        self.assertEqual(rl.lfilter_cache_info()['asterisk_size'], 2)

    def test_different_args_create_new_entries(self):
        rl._compute_cascade_params_asterisk(self.A, self.C, 0, 10, 0, 0.9, 3, 'fw')
        rl._compute_cascade_params_asterisk(self.A, self.C, 0, 10, 0, 0.9, 3, 'bw')  # direction
        rl._compute_cascade_params_asterisk(self.A, self.C, 0, 10, 0, 0.8, 3, 'fw')  # gamma
        rl._compute_cascade_params_asterisk(self.A, self.C, 1, 10, 0, 0.9, 3, 'fw')  # a
        self.assertEqual(rl.lfilter_cache_info()['asterisk_size'], 4)

    def test_cached_matches_uncached(self):
        for direction, a, b in (('fw', 0, 10), ('bw', 0, 10),
                                 ('fw', -np.inf, 0), ('bw', 0, np.inf)):
            with self.subTest(direction=direction, a=a, b=b):
                rl.clear_lfilter_caches()
                cached = rl._compute_cascade_params_asterisk(
                    self.A, self.C, a, b, 0, 0.9, 3, direction)
                uncached = rl._compute_cascade_params_asterisk_uncached(
                    self.A, self.C, a, b, 0, 0.9, 3, direction)
                self.assertEqual(set(cached), set(uncached))
                for k in uncached:
                    np.testing.assert_allclose(cached[k], uncached[k])

    def test_caches_are_independent(self):
        """Filling one cache must not affect the other's count."""
        rl._compute_cascade_params(self.A, self.C, 0, 10, 0, 0.9, 'fw')
        self.assertEqual(rl.lfilter_cache_info()['asterisk_size'], 0)
        rl._compute_cascade_params_asterisk(self.A, self.C, *self.args)
        info = rl.lfilter_cache_info()
        self.assertEqual(info['size'], 1)
        self.assertEqual(info['asterisk_size'], 1)

    def test_clear_empties_cache(self):
        rl._compute_cascade_params_asterisk(self.A, self.C, *self.args)
        self.assertGreater(rl.lfilter_cache_info()['asterisk_size'], 0)
        rl.clear_lfilter_caches()
        self.assertEqual(rl.lfilter_cache_info()['asterisk_size'], 0)


def _deep_equal(x, y):
    """Recursive equality for the nested list/tuple/ndarray plan structure."""
    if isinstance(x, (list, tuple)):
        return type(x) is type(y) and len(x) == len(y) and \
            all(_deep_equal(a, b) for a, b in zip(x, y))
    if isinstance(x, np.ndarray):
        return np.array_equal(x, y)
    return x == y


class TestBuildParallelNumdenomCachePrimitives(unittest.TestCase):
    """Unit tests for the parallel-plan cache helpers in rec_lfilter."""

    def setUp(self):
        rl.clear_lfilter_caches()
        # Upper-triangular A (valid for both forms); poly degree 2 -> N=3.
        from lmlib.statespace.model import AlssmSum
        alssm = lm.AlssmPoly(poly_degree=2)
        combined = AlssmSum([alssm], [1.0], force_MC=True)
        self.A = combined.A
        self.C = combined.C
        self.block_sizes = [alssm.N]
        # a, b, delta, gamma, direction, block_sizes
        self.args = (-20, -1, 0, 0.9, 'fw', self.block_sizes)

    def test_starts_empty_after_clear(self):
        self.assertEqual(rl.lfilter_cache_info()['parallel_size'], 0)

    def test_miss_then_hit_same_object(self):
        p1 = rl.build_parallel_numdenom(self.A, self.C, *self.args)
        self.assertEqual(rl.lfilter_cache_info()['parallel_size'], 1)
        p2 = rl.build_parallel_numdenom(self.A, self.C, *self.args)
        self.assertEqual(rl.lfilter_cache_info()['parallel_size'], 1)
        self.assertIs(p1, p2)

    def test_content_equal_arrays_and_list_hit_same_entry(self):
        p1 = rl.build_parallel_numdenom(self.A, self.C, *self.args)
        # distinct array copies AND a distinct (but equal) block_sizes list
        p2 = rl.build_parallel_numdenom(
            self.A.copy(), self.C.copy(),
            -20, -1, 0, 0.9, 'fw', list(self.block_sizes))
        self.assertIs(p1, p2)
        self.assertEqual(rl.lfilter_cache_info()['parallel_size'], 1)

    def test_block_sizes_is_part_of_the_key(self):
        rl.build_parallel_numdenom(self.A, self.C, -20, -1, 0, 0.9, 'fw', self.block_sizes)
        rl.build_parallel_numdenom(self.A, self.C, -20, -1, 0, 0.9, 'fw', None)
        self.assertEqual(rl.lfilter_cache_info()['parallel_size'], 2)

    def test_different_args_create_new_entries(self):
        rl.build_parallel_numdenom(self.A, self.C, -20, -1, 0, 0.9, 'fw', self.block_sizes)
        rl.build_parallel_numdenom(self.A, self.C, 0, 20, 0, 0.9, 'bw', self.block_sizes)  # dir+bounds
        rl.build_parallel_numdenom(self.A, self.C, -20, -1, 0, 0.8, 'fw', self.block_sizes)  # gamma
        self.assertEqual(rl.lfilter_cache_info()['parallel_size'], 3)

    def test_cached_matches_uncached(self):
        cached = rl.build_parallel_numdenom(self.A, self.C, *self.args)
        uncached = rl._build_parallel_numdenom_uncached(self.A, self.C, *self.args)
        self.assertTrue(_deep_equal(cached, uncached))

    def test_caches_are_independent(self):
        rl.build_parallel_numdenom(self.A, self.C, *self.args)
        info = rl.lfilter_cache_info()
        self.assertEqual(info['parallel_size'], 1)
        self.assertEqual(info['size'], 0)
        self.assertEqual(info['asterisk_size'], 0)

    def test_clear_empties_cache(self):
        rl.build_parallel_numdenom(self.A, self.C, *self.args)
        self.assertGreater(rl.lfilter_cache_info()['parallel_size'], 0)
        rl.clear_lfilter_caches()
        self.assertEqual(rl.lfilter_cache_info()['parallel_size'], 0)


class TestBuildParallelNumdenomCacheThroughFilter(unittest.TestCase):
    """End-to-end behaviour of the parallel-plan cache via RLSAlssm.filter."""

    def setUp(self):
        rl.clear_lfilter_caches()
        np.random.seed(0)
        self.y = np.random.randn(200, 4)   # multichannel exercises the per-slice loop
        alssm = lm.AlssmPoly(poly_degree=3)
        seg = lm.Segment(a=-25, b=-1, direction=lm.FORWARD, g=20)
        self.cost = lm.CostSegment(alssm, seg)

    def _new_rls(self):
        return lm.RLSAlssm(self.cost, backend='lfilter', filter_form='parallel')

    def test_plan_built_once_at_construction(self):
        self._new_rls()
        self.assertEqual(rl.lfilter_cache_info()['parallel_size'], 1)

    def test_repeated_filter_same_object_does_not_grow_parallel_cache(self):
        rls = self._new_rls()
        rls.filter(self.y)
        size = rl.lfilter_cache_info()['parallel_size']
        rls.filter(self.y)
        self.assertEqual(rl.lfilter_cache_info()['parallel_size'], size)

    def test_distinct_objects_same_model_share_parallel_cache(self):
        rls_a = self._new_rls()
        rls_a.filter(self.y)
        size = rl.lfilter_cache_info()['parallel_size']

        # A new object with the same model must reuse the plan (no growth).
        rls_b = self._new_rls()
        self.assertEqual(rl.lfilter_cache_info()['parallel_size'], size)
        rls_b.filter(self.y)
        np.testing.assert_allclose(rls_a.xi, rls_b.xi)
        np.testing.assert_allclose(rls_a.W, rls_b.W)

    def test_cache_does_not_change_results_vs_numpy(self):
        rls_p = self._new_rls()
        rls_p.filter(self.y)
        rls_np = lm.RLSAlssm(self.cost, backend='numpy')
        rls_np.filter(self.y)
        np.testing.assert_allclose(rls_p.xi, rls_np.xi)
        np.testing.assert_allclose(rls_p.W, rls_np.W)
        np.testing.assert_allclose(rls_p.kappa, rls_np.kappa)


class TestBuildParallelAstSosCachePrimitives(unittest.TestCase):
    """Unit tests for the parallel asterisk-l SOS cache helpers."""

    def setUp(self):
        rl.clear_lfilter_caches()
        from lmlib.statespace.model import AlssmSum
        alssm = lm.AlssmPoly(poly_degree=2)
        combined = AlssmSum([alssm], [1.0], force_MC=True)
        self.A = combined.A
        self.C = combined.C
        # a, b, delta, gamma, direction  (single block -> no block_sizes)
        self.args = (0, 10, 0, 0.9, 'fw')

    def test_starts_empty_after_clear(self):
        self.assertEqual(rl.lfilter_cache_info()['parallel_asterisk_size'], 0)

    def test_miss_then_hit_same_object(self):
        n1 = rl._build_parallel_ast_sos(self.A, self.C, *self.args)
        self.assertEqual(rl.lfilter_cache_info()['parallel_asterisk_size'], 1)
        n2 = rl._build_parallel_ast_sos(self.A, self.C, *self.args)
        self.assertEqual(rl.lfilter_cache_info()['parallel_asterisk_size'], 1)
        self.assertIs(n1, n2)

    def test_content_equal_arrays_hit_same_entry(self):
        n1 = rl._build_parallel_ast_sos(self.A, self.C, *self.args)
        n2 = rl._build_parallel_ast_sos(self.A.copy(), self.C.copy(), *self.args)
        self.assertIs(n1, n2)
        self.assertEqual(rl.lfilter_cache_info()['parallel_asterisk_size'], 1)

    def test_different_args_create_new_entries(self):
        rl._build_parallel_ast_sos(self.A, self.C, 0, 10, 0, 0.9, 'fw')
        rl._build_parallel_ast_sos(self.A, self.C, 0, 10, 0, 0.9, 'bw')  # direction
        rl._build_parallel_ast_sos(self.A, self.C, 0, 10, 0, 0.8, 'fw')  # gamma
        rl._build_parallel_ast_sos(self.A, self.C, 1, 10, 0, 0.9, 'fw')  # a
        self.assertEqual(rl.lfilter_cache_info()['parallel_asterisk_size'], 4)

    def test_cached_matches_uncached(self):
        cached = rl._build_parallel_ast_sos(self.A, self.C, *self.args)
        uncached = rl._build_parallel_ast_sos_uncached(self.A, self.C, *self.args)
        self.assertTrue(_deep_equal(cached, uncached))

    def test_caches_are_independent(self):
        rl._build_parallel_ast_sos(self.A, self.C, *self.args)
        info = rl.lfilter_cache_info()
        self.assertEqual(info['parallel_asterisk_size'], 1)
        self.assertEqual(info['size'], 0)
        self.assertEqual(info['asterisk_size'], 0)
        self.assertEqual(info['parallel_size'], 0)

    def test_clear_empties_cache(self):
        rl._build_parallel_ast_sos(self.A, self.C, *self.args)
        self.assertGreater(rl.lfilter_cache_info()['parallel_asterisk_size'], 0)
        rl.clear_lfilter_caches()
        self.assertEqual(rl.lfilter_cache_info()['parallel_asterisk_size'], 0)


class TestBuildParallelAstSosCacheThroughNDFilter(unittest.TestCase):
    """End-to-end behaviour of the parallel asterisk-l SOS cache via ND filter.

    The asterisk recursion runs for the 2nd+ axes of an ND cost; with the
    parallel filter form this exercises ``_build_parallel_ast_sos`` (which, unlike
    the non-asterisk parallel plan, is built on the recursion path rather than at
    construction time).
    """

    @staticmethod
    def _make_nd_cost(poly_degrees, hw=15, g=20.0):
        seg_fw = lm.Segment(a=-hw, b=-1, direction=lm.FW, g=g, delta=0)
        seg_bw = lm.Segment(a=0, b=hw, direction=lm.BW, g=g, delta=0)
        costs = [lm.CompositeCost([lm.AlssmPoly(poly_degree=pd)],
                                  [seg_fw, seg_bw], F=[[1, 1]])
                 for pd in poly_degrees]
        return lm.NDCompositeCost(costs)

    def setUp(self):
        rl.clear_lfilter_caches()
        np.random.seed(0)
        self.Y = np.random.randn(20, 30)
        self.nd = self._make_nd_cost([1, 2])

    def _new_rls(self):
        return lm.RLSAlssm(self.nd, steady_state=True,
                           backend='lfilter', filter_form='parallel')

    def _filter(self, rls):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rls.filter(self.Y, dim_order=[0, 1])

    def test_asterisk_sos_cache_populated_by_nd_filter(self):
        rls = self._new_rls()
        self._filter(rls)
        self.assertGreater(rl.lfilter_cache_info()['parallel_asterisk_size'], 0)

    def test_repeated_filter_same_object_does_not_grow_cache(self):
        """This is the key regression: before caching, the asterisk SOS was
        rebuilt on every filter() call."""
        rls = self._new_rls()
        self._filter(rls)
        size = rl.lfilter_cache_info()['parallel_asterisk_size']
        self._filter(rls)
        self.assertEqual(rl.lfilter_cache_info()['parallel_asterisk_size'], size)

    def test_distinct_objects_same_model_share_cache(self):
        rls_a = self._new_rls()
        self._filter(rls_a)
        size = rl.lfilter_cache_info()['parallel_asterisk_size']

        rls_b = self._new_rls()
        self._filter(rls_b)
        self.assertEqual(rl.lfilter_cache_info()['parallel_asterisk_size'], size)
        np.testing.assert_allclose(rls_a.xi, rls_b.xi)
        np.testing.assert_allclose(rls_a.W, rls_b.W)

    def test_cache_does_not_change_results_vs_numpy(self):
        rls_p = self._new_rls()
        self._filter(rls_p)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rls_np = lm.RLSAlssm(self.nd, steady_state=True, backend='numpy')
            rls_np.filter(self.Y, dim_order=[0, 1])
        np.testing.assert_allclose(rls_p.xi, rls_np.xi)
        np.testing.assert_allclose(rls_p.W, rls_np.W)


class TestCascadeParamsCacheThroughFilter(unittest.TestCase):
    """End-to-end behaviour of the cache via RLSAlssm.filter."""

    def setUp(self):
        rl.clear_lfilter_caches()
        np.random.seed(0)
        self.y = np.random.randn(200)
        alssm = lm.AlssmPoly(poly_degree=2)
        seg = lm.Segment(a=-20, b=-1, direction=lm.FORWARD, g=15)
        self.cost = lm.CostSegment(alssm, seg)

    def _new_rls(self):
        return lm.RLSAlssm(self.cost, backend='lfilter', filter_form='cascade')

    def test_repeated_filter_same_object_does_not_grow_cache(self):
        rls = self._new_rls()
        rls.filter(self.y)
        size_after_first = rl.lfilter_cache_info()['size']
        self.assertGreater(size_after_first, 0)
        rls.filter(self.y)
        self.assertEqual(rl.lfilter_cache_info()['size'], size_after_first)

    def test_distinct_objects_same_model_share_cache(self):
        rls_a = self._new_rls()
        rls_a.filter(self.y)
        size_after_first = rl.lfilter_cache_info()['size']

        # A brand-new object with the identical model + window must reuse the
        # existing entries (no growth) and produce identical output.
        rls_b = self._new_rls()
        rls_b.filter(self.y)
        self.assertEqual(rl.lfilter_cache_info()['size'], size_after_first)
        np.testing.assert_allclose(rls_a.xi, rls_b.xi)
        np.testing.assert_allclose(rls_a.W, rls_b.W)
        np.testing.assert_allclose(rls_a.kappa, rls_b.kappa)

    def test_cache_does_not_change_results_vs_numpy(self):
        rls_c = self._new_rls()
        rls_c.filter(self.y)
        rls_np = lm.RLSAlssm(self.cost, backend='numpy')
        rls_np.filter(self.y)
        np.testing.assert_allclose(rls_c.xi, rls_np.xi)
        np.testing.assert_allclose(rls_c.W, rls_np.W)
        np.testing.assert_allclose(rls_c.kappa, rls_np.kappa)

    def test_composite_cost_forward_and_backward(self):
        """Multi-ALSSM, mixed fw/bw segments still match the numpy reference."""
        np.random.seed(3)
        y = np.random.randn(250)
        a1 = lm.AlssmPoly(poly_degree=2)
        a2 = lm.AlssmPoly(poly_degree=1)
        seg_fw = lm.Segment(a=-25, b=-1, direction=lm.FORWARD, g=15)
        seg_bw = lm.Segment(a=0, b=25, direction=lm.BACKWARD, g=15)
        cost = lm.CompositeCost((a1, a2), (seg_fw, seg_bw), [[1, 0], [0, 1]])

        rl.clear_lfilter_caches()
        rls_c = lm.RLSAlssm(cost, backend='lfilter', filter_form='cascade')
        rls_c.filter(y)
        rls_np = lm.RLSAlssm(cost, backend='numpy')
        rls_np.filter(y)
        np.testing.assert_allclose(rls_c.xi, rls_np.xi)
        np.testing.assert_allclose(rls_c.W, rls_np.W)
        np.testing.assert_allclose(rls_c.kappa, rls_np.kappa)


class TestCascadeParamsAsteriskCacheThroughNDFilter(unittest.TestCase):
    """End-to-end behaviour of the asterisk cache via ND RLSAlssm.filter.

    The asterisk-l recursion only runs for the *second and subsequent* axes of
    an ND cost, so these tests use a 2-D NDCompositeCost with the lfilter
    cascade backend.
    """

    @staticmethod
    def _make_nd_cost(poly_degrees, hw=15, g=20.0):
        seg_fw = lm.Segment(a=-hw, b=-1, direction=lm.FW, g=g, delta=0)
        seg_bw = lm.Segment(a=0, b=hw, direction=lm.BW, g=g, delta=0)
        costs = [lm.CompositeCost([lm.AlssmPoly(poly_degree=pd)],
                                  [seg_fw, seg_bw], F=[[1, 1]])
                 for pd in poly_degrees]
        return lm.NDCompositeCost(costs)

    def setUp(self):
        rl.clear_lfilter_caches()
        np.random.seed(0)
        self.Y = np.random.randn(40, 35)
        self.nd = self._make_nd_cost([1, 2])

    def _new_rls(self):
        return lm.RLSAlssm(self.nd, steady_state=True,
                           backend='lfilter', filter_form='cascade')

    def _filter(self, rls):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')   # ignore W-conditioning warnings
            rls.filter(self.Y, dim_order=[0, 1])

    def test_asterisk_cache_populated_by_nd_filter(self):
        rls = self._new_rls()
        self._filter(rls)
        # The 2nd axis (fw + bw) feeds the asterisk recursion at q==0 and q==1.
        self.assertGreater(rl.lfilter_cache_info()['asterisk_size'], 0)

    def test_repeated_filter_same_object_does_not_grow_asterisk_cache(self):
        rls = self._new_rls()
        self._filter(rls)
        size = rl.lfilter_cache_info()['asterisk_size']
        self._filter(rls)
        self.assertEqual(rl.lfilter_cache_info()['asterisk_size'], size)

    def test_distinct_objects_same_model_share_asterisk_cache(self):
        rls_a = self._new_rls()
        self._filter(rls_a)
        size = rl.lfilter_cache_info()['asterisk_size']

        rls_b = self._new_rls()
        self._filter(rls_b)
        self.assertEqual(rl.lfilter_cache_info()['asterisk_size'], size)
        np.testing.assert_allclose(rls_a.xi, rls_b.xi)
        np.testing.assert_allclose(rls_a.W, rls_b.W)

    def test_nd_cache_does_not_change_results_vs_numpy(self):
        rls_lf = self._new_rls()
        self._filter(rls_lf)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rls_np = lm.RLSAlssm(self.nd, steady_state=True, backend='numpy')
            rls_np.filter(self.Y, dim_order=[0, 1])

        np.testing.assert_allclose(rls_lf.xi, rls_np.xi)
        np.testing.assert_allclose(rls_lf.W, rls_np.W)


if __name__ == '__main__':
    unittest.main()
