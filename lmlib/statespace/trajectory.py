import numpy as np
from lmlib.statespace.cost import CostSegment, CompositeCost, NDCompositeCost

import warnings

__all__ = ['Trajectory']



class Trajectory:
    """
    Static utility class for evaluating ALSSM output trajectories.

    Provides methods to compute the signal trajectories implied by a set
    of state vectors together with a cost's ALSSMs and segment windows, and
    to map those trajectories onto a common output array indexed by signal
    position.
    """

    @staticmethod
    def eval(cost, xs, F=None, thd=1e-6, merged_ks=True, merged_seg=True):
        """
        Evaluate ALSSM output trajectories for a set of state vectors.

        For each state vector in ``xs`` and each cost segment in ``cost``,
        computes the ALSSM output over the segment's window range.

        Parameters
        ----------
        cost : CostSegment or CompositeCost
            Cost whose ALSSM models and segments define the trajectory.
        xs : array_like of shape (..., N)
            State vectors defining the trajectories. The last dimension must
            equal the ALSSM model order N.
        F : array_like of shape (M, P) or None, optional
            Override mapping matrix. If ``None``, uses the mapping defined in
            ``cost``. Default: None.
        thd : float, optional
            Threshold for truncating the window range at infinite segment
            boundaries. Default: 1e-6.
        merged_ks : bool, optional
            If True, merge across the state-vector dimension by taking the
            element-wise maximum. Default: True.
        merged_seg : bool, optional
            If True, merge across segments by taking the element-wise maximum.
            Default: True.

        Returns
        -------
        out : ndarray of dtype=object, shape depends on merging flags
            Array of ``(ab_range, trajectory)`` tuples. When neither flag is
            set, shape is ``(P, *XS)`` where ``P`` is the number of segments
            and ``XS`` is the leading shape of ``xs``.  Each tuple contains:

            * ``ab_range`` — 1-D integer array of relative sample indices.
            * ``trajectory`` — ALSSM output values at those indices.
        """

        xs = np.asarray(xs)
        if xs.ndim == 1:          # shape (N,) → (1, N)
            xs = xs[np.newaxis, :]
        # else:
        #     if xs.ndim == 2 and xs.shape[1] > 1:      # is multi channel
        #         xs = xs[np.newaxis, :]

        cost_segments = cost._get_cost_segments(F)
        XS = xs.shape[:-1]

        out = np.empty((len(cost_segments), *XS), dtype=tuple)
        for p, cs in enumerate(cost_segments):
            ab_range = cs.segment._ab_range(thd)
            for idx in np.ndindex(XS):
                out[p][idx] = ab_range, cs.alssm.eval_output(xs[idx], ab_range)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            if merged_ks:
                out = np.nanmax(out, axis=1)
            if merged_seg:
                out = np.nanmax(out, axis=0)

        return out

    @staticmethod
    def eval_y(cost, xs, ks, K, F=None, thd=1e-6, merged_ks=True, merged_seg=True, fill_value=np.nan):
        """
        Evaluate trajectories at anchor positions ``ks`` and map them into a
        length-``K`` output array.

        For each anchor index ``k`` in ``ks`` and each segment in ``cost``,
        evaluates the ALSSM trajectory and places the values at the absolute
        signal positions ``k + ab_range``.  Out-of-bounds positions are
        silently discarded.

        Parameters
        ----------
        cost : CostSegment or CompositeCost
            Cost whose ALSSM models and segments define the trajectories.
        xs : array_like of shape (len(ks), N) or (K, N)
            State vectors.  Either one state vector per anchor (shape
            ``(len(ks), N)``) or one per signal sample (shape ``(K, N)``),
            in which case the state at each anchor is looked up by index.
        ks : array_like of int, shape (XS,)
            Anchor positions (absolute signal indices) at which the
            trajectories are centred.
        K : int
            Length of the output signal vector.
        F : array_like of shape (M, P) or None, optional
            Override mapping matrix. Default: None (use cost's own mapping).
        thd : float, optional
            Threshold for truncating the window at infinite boundaries.
            Default: 1e-6.
        merged_ks : bool, optional
            If True, merge across anchor positions using the element-wise
            maximum. Default: True.
        merged_seg : bool, optional
            If True, merge across segments using the element-wise maximum.
            Default: True.
        fill_value : scalar, optional
            Value used for output positions not covered by any trajectory.
            Default: ``np.nan``.

        Returns
        -------
        out : ndarray
            Mapped trajectory array of shape ``(K,)`` after both merges,
            ``(P, K)`` after only the ``ks`` merge, or ``(P, len(ks), K)``
            with no merging.

        Notes
        -----
        If ``ks`` is empty, a warning is printed and an array of shape
        ``(K,)`` filled with ``fill_value`` is returned.

        Raises
        ------
        ValueError
            If ``len(xs)`` matches neither ``len(ks)`` nor ``K``.
        NotImplementedError
            If ``cost`` is an :class:`NDCompositeCost`.
        """
        if isinstance(cost, NDCompositeCost):
            #TODO implement
            raise NotImplementedError("ND trajectory not implemented")

        else:
            # Accept a bare integer or numpy scalar for ks (single-peak convenience).
            # xs may be a single state vector of shape (N,) or a full array of
            # shape (K, N); both are handled correctly by the length checks below.
            if np.ndim(ks) == 0:
                ks = np.array([int(ks)])
                xs = np.asarray(xs)
                if xs.ndim == 1:          # shape (N,) → (1, N)
                    xs = xs[np.newaxis, :]
                else:
                    if xs.ndim == 2 and xs.shape[1] > 1:      # is multi channel
                        xs = xs[np.newaxis, :]

            # return an empty array if ks is empty
            if len(ks) == 0:
                print('Warning: ks is empty. Returned empty array of size K.')
                return np.full(K, fill_value=fill_value)

            if len(xs) == len(ks):
                trajs = Trajectory.eval(cost, xs, F, thd, merged_ks=False, merged_seg=False)
            elif len(xs) == K:
                trajs = Trajectory.eval(cost, xs[ks], F, thd, merged_ks=False, merged_seg=False)
            else:
                raise ValueError('Length of xs must match length of ks or is equal to K.')

            P, xs_dim, *multi_dim = trajs.shape

            out = np.full((P, xs_dim, K, *multi_dim), fill_value=fill_value)
            for p, xs_idx, *multi_idx in np.ndindex(trajs.shape):
                ab_range, traj = trajs[(p, xs_idx, *multi_idx)]
                ks_indexes = ks[xs_idx] + np.array(ab_range)
                mask = (ks_indexes >= 0) & (ks_indexes < K)
                out[(p, xs_idx, ...,  ks_indexes[mask], *multi_idx)] = traj[mask]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                if merged_ks:
                    out = np.nanmax(out, axis=1)
                if merged_seg:
                    out = np.nanmax(out, axis=0)

            return out
