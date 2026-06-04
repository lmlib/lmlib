import numpy as np
import warnings

__all__ = ['Window']

class Window:
    """
    Static utility class for evaluating window functions over cost segments.

    Provides methods to compute the per-sample window weights of a cost's
    segments and to map those weights into a common output vector indexed
    by signal position.
    """
    @staticmethod
    def eval(cost, segment_indices=None, thd=1e-6):
        r"""
        Compute the window range and weights for each segment of a cost.

        Parameters
        ----------
        cost : CostSegment or CompositeCost
            Cost whose segment windows are evaluated.
        segment_indices : list of int or None, optional
            If given, only the segments at these indices are evaluated.
            If None (default), all segments are used.
        thd : float, optional
            Threshold below which exponential window weights are truncated
            (passed to `_ab_range`). Default: 1e-6.

        Returns
        -------
        out : ndarray of dtype=object, shape=(P,)
            Array of ``(ab_range, weights)`` tuples, one per segment.
            ``ab_range`` is the integer index range; ``weights`` is the
            corresponding array of $\gamma^{i-\delta}$ values.
        """

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
        """
        Map window weights for a set of anchor positions into a length-K output vector.

        For each anchor position ``k`` in ``ks`` and each cost segment, the
        window weights are placed at the corresponding absolute signal indices
        ``k + ab_range``.  Out-of-bounds indices are silently discarded.

        Parameters
        ----------
        cost : CostSegment or CompositeCost
            Cost whose segment windows are evaluated.
        ks : array_like of int
            Anchor positions (signal indices) at which windows are centred.
        K : int
            Length of the output signal vector.
        merged_ks : bool, optional
            If True, merge across anchor positions using element-wise maximum.
            Default: True.
        merged_seg : bool, optional
            If True, merge across segments using element-wise maximum.
            Default: True.
        segment_indices : list of int or None, optional
            Restrict to the given segment indices. Default: None (all segments).
        thd : float, optional
            Window truncation threshold. Default: 1e-6.
        fill_value : float, optional
            Value used for output indices not covered by any window. Default: 0.0.

        Returns
        -------
        out : ndarray
            Mapped window array.  Shape depends on ``merged_ks`` / ``merged_seg``:
            ``(P, len(ks), K)`` → ``(P, K)`` → ``(K,)`` after merging.
        """

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
