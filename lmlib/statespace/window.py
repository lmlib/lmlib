import numpy as np
import warnings
from functools import reduce
from itertools import product as itertools_product

from lmlib.statespace.cost import NDCompositeCost

__all__ = ['Window']


def _nd_window_combos(cost, segment_indices=None, thd=1e-6):
    r"""
    Compute the separable window for every per-dimension segment combination of
    an [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost].

    The window of a separable ND cost factorises over the signal axes: the
    weight at relative offset :math:`(j_0, j_1, \ldots, j_{L-1})` equals

    .. math::
        w(j_0,\ldots,j_{L-1})
            = \prod_{l=0}^{L-1} \gamma_l^{\,j_l - \delta_l},

    i.e. the outer product of the per-axis 1-D window weights.  One such window
    tensor is produced for every combination of per-dimension segments (for a
    forward+backward window per axis in 2-D this yields the four quadrant
    tiles).

    Parameters
    ----------
    cost : NDCompositeCost
    segment_indices : list of list of int or None, optional
        Per-dimension segment indices to include, one inner list per axis
        (e.g. ``[[0], [0, 1]]`` keeps segment 0 of axis 0 and both segments of
        axis 1).  If None (default), all segments of every axis are used.
    thd : float, optional
        Window truncation threshold passed to ``Segment._ab_range``.

    Returns
    -------
    out : ndarray of dtype=object, shape (n_combos,)
        Array of ``(ab_ranges, weights)`` tuples, one per segment combination.
        ``ab_ranges`` is a list of ``L`` integer offset arrays and ``weights``
        is the corresponding window tensor of shape
        ``(len_ab_0, ..., len_ab_{L-1})``.
    """
    L = cost.L
    cost_segs_per_dim = [cost.cost_terms[l]._get_cost_segments() for l in range(L)]

    if segment_indices is None:
        idx_per_dim = [range(len(cost_segs_per_dim[l])) for l in range(L)]
    else:
        assert len(segment_indices) == L, (
            f'segment_indices must have one entry per dimension (L={L}); '
            f'got {len(segment_indices)}.'
        )
        idx_per_dim = [list(segment_indices[l]) for l in range(L)]

    all_combos = list(itertools_product(*idx_per_dim))
    out = np.empty(len(all_combos), dtype=object)
    for combo_idx, p_combo in enumerate(all_combos):
        ab_ranges = []
        weights_1d = []
        for l in range(L):
            seg = cost_segs_per_dim[l][p_combo[l]].segment
            ab = seg._ab_range(thd)
            ab_ranges.append(ab)
            weights_1d.append(seg.gamma ** (np.array(ab) - seg.delta))
        # Separable ND window = outer product of the per-axis 1-D windows.
        weights = reduce(np.multiply.outer, weights_1d)
        out[combo_idx] = (ab_ranges, weights)
    return out


class Window:
    """
    Static utility class for evaluating window functions over cost segments.

    Provides methods to compute the per-sample window weights of a cost's
    segments and to map those weights into a common output vector indexed
    by signal position.

    All methods accept [`CostSegment`][lmlib.statespace.cost.CostSegment],
    [`CompositeCost`][lmlib.statespace.cost.CompositeCost], and
    [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost].  For an
    [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost] the window is
    treated as separable (a Kronecker product over the signal axes), mirroring
    [`Trajectory`][lmlib.statespace.trajectory.Trajectory].
    """
    @staticmethod
    def eval(cost, segment_indices=None, thd=1e-6):
        r"""
        Compute the window range and weights for each segment of a cost.

        Parameters
        ----------
        cost : CostSegment, CompositeCost or NDCompositeCost
            Cost whose segment windows are evaluated.
        segment_indices : list of int, list of list of int, or None, optional
            For [`CostSegment`][lmlib.statespace.cost.CostSegment] /
            [`CompositeCost`][lmlib.statespace.cost.CompositeCost]: a flat list
            of segment indices to evaluate (None → all segments).
            For [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost]: a
            list with one inner list of segment indices **per axis**
            (None → all segments on every axis).
        thd : float, optional
            Threshold below which exponential window weights are truncated
            (passed to `_ab_range`). Default: 1e-6.

        Returns
        -------
        out : ndarray of dtype=object
            * **1-D** ([`CostSegment`][lmlib.statespace.cost.CostSegment] /
              [`CompositeCost`][lmlib.statespace.cost.CompositeCost]): shape
              ``(P,)``; one ``(ab_range, weights)`` tuple per segment.
              ``ab_range`` is a 1-D integer index range and ``weights`` the
              corresponding array of $\gamma^{i-\delta}$ values.
            * **ND** ([`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost]):
              shape ``(n_combos,)``; one ``(ab_ranges, weights)`` tuple per
              combination of per-axis segments.  ``ab_ranges`` is a list of
              ``L`` integer offset arrays and ``weights`` the separable window
              tensor of shape ``(len_ab_0, ..., len_ab_{L-1})``.

        Example
        --------
        **CostSegment** — a single forward window:

        ```python
        >>> import lmlib as lm
        >>> from lmlib.statespace.window import Window
        >>> cs = lm.CostSegment(lm.AlssmPoly(2), lm.Segment(-4, -1, lm.FW, g=8))
        >>> (ab_range, weights), = Window.eval(cs)
        >>> ab_range                    # window offsets relative to the anchor
        array([-4, -3, -2, -1])
        >>> weights.shape               # one weight gamma**(i - delta) per offset
        (4,)
        >>> weights.round(3)
        array([0.586, 0.67 , 0.766, 0.875])
        ```

        **CompositeCost** — a symmetric forward+backward window (two segments):

        ```python
        >>> import lmlib as lm
        >>> from lmlib.statespace.window import Window
        >>> sl = lm.Segment(-4, -1, lm.FW, g=8)
        >>> sr = lm.Segment(0, 4, lm.BW, g=8)
        >>> cc = lm.CompositeCost((lm.AlssmPoly(2),), (sl, sr), F=[[1, 1]])
        >>> wins = Window.eval(cc)
        >>> len(wins)                   # one (ab_range, weights) tuple per segment
        2
        >>> wins[1][0]                  # offsets of the backward (right) segment
        array([0, 1, 2, 3, 4])
        ```

        **NDCompositeCost** — a separable 2-D window (one CompositeCost per axis):

        ```python
        >>> import numpy as np
        >>> import lmlib as lm
        >>> from lmlib.statespace.window import Window
        >>> sl = lm.Segment(-4, -1, lm.FW, g=8)
        >>> sr = lm.Segment(0, 4, lm.BW, g=8)
        >>> cc = lm.CompositeCost((lm.AlssmPoly(2),), (sl, sr), F=[[1, 1]])
        >>> nd = lm.NDCompositeCost([cc, cc])
        >>> wins = Window.eval(nd)
        >>> len(wins)                   # 2 segments per axis -> 2 x 2 = 4 combos
        4
        >>> ab_ranges, w = wins[0]      # first combo: (left, left)
        >>> w.shape                     # 2-D window tensor (len_ab_0, len_ab_1)
        (4, 4)
        >>> # separable: the 2-D window is the outer product of the per-axis 1-D windows
        >>> w0 = Window.eval(cc)[0][1]  # axis-0, segment-0 1-D window weights
        >>> np.allclose(w, np.outer(w0, w0))
        True
        ```
        """
        if isinstance(cost, NDCompositeCost):
            return _nd_window_combos(cost, segment_indices, thd)

        if segment_indices is None:
            cost_segments = cost._get_cost_segments()
        else:
            cost_segments = [cost._get_sub_cost_term(seg=p) for p in segment_indices]

        out = np.empty(len(cost_segments), dtype=tuple)
        for p, cs, in enumerate(cost_segments):
            ab_range = cs.segment._ab_range(thd)
            out[p] = ab_range, cs.segment.gamma ** (np.array(ab_range) - cs.segment.delta)

        return out

    @staticmethod
    def eval_y(cost, ks, K, merged_ks=True, merged_seg=True, segment_indices=None, thd=1e-6, fill_value=0.0):
        r"""
        Map window weights for a set of anchor positions into an output array.

        For each anchor position ``k`` in ``ks`` and each cost segment, the
        window weights are placed at the corresponding absolute signal indices
        ``k + ab_range``.  Out-of-bounds indices are silently discarded.

        For an [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost] the
        separable ND window of every per-axis segment combination is placed onto
        an ``L``-dimensional output grid, mirroring
        [`Trajectory.eval_y`][lmlib.statespace.trajectory.Trajectory.eval_y].

        Parameters
        ----------
        cost : CostSegment, CompositeCost or NDCompositeCost
            Cost whose segment windows are evaluated.
        ks : array_like of int
            Anchor positions at which windows are centred.

            * **1-D**: signal indices, shape ``(n_anchors,)`` (a scalar is
              accepted for a single anchor).
            * **ND**: one ``L``-dimensional anchor ``(k_0, ..., k_{L-1})`` or an
              array of such anchors of shape ``(n_anchors, L)``.
        K : int or tuple of int
            Output size.  An ``int`` for 1-D; an ``L``-tuple
            ``(K_0, ..., K_{L-1})`` for an
            [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost].
        merged_ks : bool, optional
            If True, merge across anchor positions using element-wise maximum.
            Default: True.
        merged_seg : bool, optional
            If True, merge across segments (1-D) or segment combinations (ND)
            using element-wise maximum. Default: True.
        segment_indices : list or None, optional
            Restrict to the given segment indices (see
            [`eval`][lmlib.statespace.window.Window.eval]). Default: None.
        thd : float, optional
            Window truncation threshold. Default: 1e-6.
        fill_value : float, optional
            Value used for output positions not covered by any window.
            Default: 0.0.

        Returns
        -------
        out : ndarray
            Mapped window array.

            * **1-D**: shape ``(P, len(ks), K)`` → ``(P, K)`` → ``(K,)`` as the
              ``merged_ks`` / ``merged_seg`` flags reduce it.
            * **ND**: shape ``(n_combos, n_anchors, *K)`` →  ``(n_combos, *K)``
              → ``K`` as the flags reduce it.

        Examples
        --------
        **CostSegment** — a single forward window mapped onto a length-K vector:

        ```python
        >>> import numpy as np
        >>> import lmlib as lm
        >>> from lmlib.statespace.window import Window
        >>> cs = lm.CostSegment(lm.AlssmPoly(2), lm.Segment(-4, -1, lm.FW, g=8))
        >>> w = Window.eval_y(cs, ks=10, K=20)
        >>> w.shape
        (20,)
        >>> np.flatnonzero(w)           # window sits at k + [-4..-1] = [6..9]
        array([6, 7, 8, 9])
        ```

        **CompositeCost** — symmetric window centred at two anchors:

        ```python
        >>> import numpy as np
        >>> import lmlib as lm
        >>> from lmlib.statespace.window import Window
        >>> sl = lm.Segment(-4, -1, lm.FW, g=8)
        >>> sr = lm.Segment(0, 4, lm.BW, g=8)
        >>> cc = lm.CompositeCost((lm.AlssmPoly(2),), (sl, sr), F=[[1, 1]])
        >>> w = Window.eval_y(cc, ks=[10, 30], K=40)
        >>> w.shape                     # merged across segments and anchors
        (40,)
        >>> bool(w[10] > 0 and w[30] > 0)
        True
        ```

        **NDCompositeCost** — a separable 2-D window on an image grid:

        ```python
        >>> import numpy as np
        >>> import lmlib as lm
        >>> from lmlib.statespace.window import Window
        >>> sl = lm.Segment(-4, -1, lm.FW, g=8)
        >>> sr = lm.Segment(0, 4, lm.BW, g=8)
        >>> cc = lm.CompositeCost((lm.AlssmPoly(2),), (sl, sr), F=[[1, 1]])
        >>> nd = lm.NDCompositeCost([cc, cc])
        >>> W = Window.eval_y(nd, ks=(15, 20), K=(40, 50))
        >>> W.shape                     # 2-D output grid
        (40, 50)
        >>> # peak of the separable window is at the anchor pixel
        >>> tuple(np.unravel_index(np.argmax(W), W.shape)) == (15, 20)
        True
        ```
        """
        if isinstance(cost, NDCompositeCost):
            return Window._eval_y_nd(cost, ks, K, merged_ks=merged_ks,
                                     merged_seg=merged_seg,
                                     segment_indices=segment_indices,
                                     thd=thd, fill_value=fill_value)

        # return an empty array if ks is empty


        if np.ndim(ks) == 0:
            ks = np.array([int(ks)])

        if len(ks) == 0:
            print('Warning: ks is empty. Returned empty array of size K.')
            return np.full(K, fill_value=fill_value)

        wins = Window.eval(cost, segment_indices, thd)
        ks_dim = len(ks)
        P = len(wins)

        out = np.full((P, ks_dim, K), fill_value=fill_value)
        for p in range(P):
            ab_range, win = wins[p]
            for i, k in enumerate(ks):
                k_indexes = k + np.array(ab_range)
                mask = (k_indexes >= 0) & (k_indexes < K)
                out[p, i, k_indexes[mask]] = win[mask]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            if merged_ks:
                out = np.nanmax(out, axis=1)
            if merged_seg:
                out = np.nanmax(out, axis=0)

        return out

    @staticmethod
    def _eval_y_nd(cost, ks, K, merged_ks=True, merged_seg=True,
                   segment_indices=None, thd=1e-6, fill_value=0.0):
        r"""
        ND variant of [`eval_y`][lmlib.statespace.window.Window.eval_y] for
        [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost].

        Places the separable window of every per-axis segment combination onto
        an ``L``-dimensional output grid of shape ``K`` and merges with the
        element-wise maximum.

        Parameters
        ----------
        cost : NDCompositeCost
        ks : tuple/list of int, or array_like of shape (n_anchors, L)
            One ``L``-dimensional anchor position, or several.
        K : tuple/list of int, length L
            Output grid shape ``(K_0, ..., K_{L-1})``.
        merged_ks, merged_seg : bool
        segment_indices : list of list of int or None
        thd : float
        fill_value : scalar

        Returns
        -------
        out : ndarray
            Shape ``(n_combos, n_anchors, *K)`` reduced by ``merged_ks`` /
            ``merged_seg`` to ``(n_combos, *K)`` or ``K``.
        """
        L = cost.L
        K = tuple(K)
        assert len(K) == L, f'K must have length L={L}; got {len(K)}.'

        ks = np.asarray(ks)
        if ks.ndim == 1:                       # single anchor → (1, L)
            ks = ks[np.newaxis, :]
        assert ks.shape[1] == L, (
            f'each anchor must have length L={L}; got {ks.shape[1]}.'
        )
        n_anchors = ks.shape[0]

        if n_anchors == 0:
            print('Warning: ks is empty. Returned empty array of shape K.')
            return np.full(K, fill_value=fill_value)

        wins = _nd_window_combos(cost, segment_indices, thd)
        n_combos = len(wins)

        out = np.full((n_combos, n_anchors, *K), fill_value=fill_value)
        for combo_idx in range(n_combos):
            ab_ranges, weights = wins[combo_idx]
            grids = np.meshgrid(*[ab_ranges[l] for l in range(L)], indexing='ij')
            for ai in range(n_anchors):
                anchor = ks[ai]
                abs_coords = [grids[l] + anchor[l] for l in range(L)]
                mask = np.ones(weights.shape, dtype=bool)
                for l in range(L):
                    mask &= (abs_coords[l] >= 0) & (abs_coords[l] < K[l])
                valid_coords = tuple(abs_coords[l][mask] for l in range(L))
                dest = out[(combo_idx, ai, *valid_coords)]
                out[(combo_idx, ai, *valid_coords)] = np.fmax(dest, weights[mask])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            if merged_ks:
                out = np.nanmax(out, axis=1)
            if merged_seg:
                out = np.nanmax(out, axis=0)

        return out
