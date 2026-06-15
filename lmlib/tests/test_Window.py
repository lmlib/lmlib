import unittest
import numpy as np
import lmlib as lm
from lmlib.statespace.window import Window

__all__ = ["TestWindow1D", "TestWindowND"]


def _make_1d_composite(g=8):
    sl = lm.Segment(a=-4, b=-1, direction=lm.FW, g=g)
    sr = lm.Segment(a=0, b=4, direction=lm.BW, g=g)
    return lm.CompositeCost((lm.AlssmPoly(2),), (sl, sr), F=[[1, 1]]), sl, sr


class TestWindow1D(unittest.TestCase):
    """Baseline behaviour for CostSegment / CompositeCost (unchanged)."""

    def test_costsegment_eval(self):
        cs = lm.CostSegment(lm.AlssmPoly(2), lm.Segment(-4, -1, lm.FW, g=8))
        wins = Window.eval(cs)
        self.assertEqual(wins.shape, (1,))
        ab, w = wins[0]
        np.testing.assert_array_equal(ab, np.arange(-4, 0))
        # forward window grows toward the anchor (i = -1)
        self.assertTrue(np.all(np.diff(w) > 0))

    def test_compositecost_eval_two_segments(self):
        cc, sl, sr = _make_1d_composite()
        wins = Window.eval(cc)
        self.assertEqual(len(wins), 2)
        np.testing.assert_array_equal(wins[0][0], sl._ab_range())
        np.testing.assert_array_equal(wins[1][0], sr._ab_range())

    def test_eval_y_places_window_at_anchor(self):
        cs = lm.CostSegment(lm.AlssmPoly(2), lm.Segment(-4, -1, lm.FW, g=8))
        w = Window.eval_y(cs, ks=10, K=20)
        self.assertEqual(w.shape, (20,))
        np.testing.assert_array_equal(np.flatnonzero(w), np.array([6, 7, 8, 9]))

    def test_eval_y_out_of_bounds_clipped(self):
        cs = lm.CostSegment(lm.AlssmPoly(2), lm.Segment(-4, -1, lm.FW, g=8))
        w = Window.eval_y(cs, ks=2, K=20)   # left part falls off the edge
        np.testing.assert_array_equal(np.flatnonzero(w), np.array([0, 1]))


class TestWindowND(unittest.TestCase):
    """Window.eval / eval_y on a separable NDCompositeCost."""

    def test_nd_eval_runs_and_is_separable(self):
        cc, _, _ = _make_1d_composite()
        nd = lm.NDCompositeCost([cc, cc])
        wins = Window.eval(nd)
        # 2 segments per axis -> 2 x 2 = 4 segment combinations
        self.assertEqual(len(wins), 4)

        # canonical per-axis 1-D window weights, by segment index
        seg_w = [Window.eval(cc)[0][1], Window.eval(cc)[1][1]]
        from itertools import product
        for ci, (p0, p1) in enumerate(product([0, 1], [0, 1])):
            ab_ranges, wtensor = wins[ci]
            self.assertEqual(len(ab_ranges), 2)
            np.testing.assert_allclose(
                wtensor, np.outer(seg_w[p0], seg_w[p1]), atol=1e-12,
                err_msg=f'ND window combo {ci} is not the separable outer product',
            )

    def test_nd_eval_y_equals_outer_of_1d(self):
        """Ground truth: single FW segment per axis, single anchor ->
        the 2-D window grid is the outer product of the two 1-D window grids."""
        sl = lm.Segment(a=-4, b=-1, direction=lm.FW, g=8)
        cs = lm.CostSegment(lm.AlssmPoly(2), sl)
        nd = lm.NDCompositeCost([cs, cs])

        W_nd = Window.eval_y(nd, ks=(15, 20), K=(40, 50))
        w0 = Window.eval_y(cs, ks=15, K=40)
        w1 = Window.eval_y(cs, ks=20, K=50)
        np.testing.assert_allclose(W_nd, np.outer(w0, w1), atol=1e-12)

    def test_nd_eval_y_shapes_and_merging(self):
        cc, _, _ = _make_1d_composite()
        nd = lm.NDCompositeCost([cc, cc])

        full = Window.eval_y(nd, ks=(15, 20), K=(40, 50),
                             merged_ks=False, merged_seg=False)
        self.assertEqual(full.shape, (4, 1, 40, 50))   # (n_combos, n_anchors, *K)

        seg = Window.eval_y(nd, ks=[(15, 20), (25, 30)], K=(40, 50),
                            merged_ks=True, merged_seg=False)
        self.assertEqual(seg.shape, (4, 40, 50))        # (n_combos, *K)

        full_merge = Window.eval_y(nd, ks=(15, 20), K=(40, 50))
        self.assertEqual(full_merge.shape, (40, 50))    # K
        self.assertEqual(np.unravel_index(np.argmax(full_merge), full_merge.shape),
                         (15, 20))

    def test_nd_eval_y_multiple_anchors_merged(self):
        cc, _, _ = _make_1d_composite()
        nd = lm.NDCompositeCost([cc, cc])
        W = Window.eval_y(nd, ks=[(15, 20), (30, 35)], K=(50, 60))
        self.assertGreater(W[15, 20], 0.0)
        self.assertGreater(W[30, 35], 0.0)

    def test_nd_segment_indices_restriction(self):
        """Selecting one segment per axis yields a single combo equal to that
        quadrant's separable window."""
        cc, _, _ = _make_1d_composite()
        nd = lm.NDCompositeCost([cc, cc])
        wins = Window.eval(nd, segment_indices=[[0], [1]])  # FW on axis0, BW on axis1
        self.assertEqual(len(wins), 1)
        w_fw = Window.eval(cc)[0][1]
        w_bw = Window.eval(cc)[1][1]
        np.testing.assert_allclose(wins[0][1], np.outer(w_fw, w_bw), atol=1e-12)

    def test_nd_3d_eval_y(self):
        """A 3-D separable window must reduce to the triple outer product."""
        sl = lm.Segment(a=-3, b=-1, direction=lm.FW, g=6)
        cs = lm.CostSegment(lm.AlssmPoly(1), sl)
        nd = lm.NDCompositeCost([cs, cs, cs])
        W = Window.eval_y(nd, ks=(10, 12, 8), K=(20, 22, 18))
        self.assertEqual(W.shape, (20, 22, 18))
        w0 = Window.eval_y(cs, ks=10, K=20)
        w1 = Window.eval_y(cs, ks=12, K=22)
        w2 = Window.eval_y(cs, ks=8, K=18)
        gt = w0[:, None, None] * w1[None, :, None] * w2[None, None, :]
        np.testing.assert_allclose(W, gt, atol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
