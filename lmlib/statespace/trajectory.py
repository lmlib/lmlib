import numpy as np
from lmlib.statespace.cost import CostSegment, CompositeCost, NDCompositeCost
from numpy.linalg import matrix_power
from itertools import product as itertools_product

import warnings

__all__ = ['Trajectory']


def _eval_nd_patch(nd_cost, xs, p_combo, thd=1e-6):
    """
    Evaluate the ND ALSSM output for a single state vector ``xs`` and a
    specific combination of per-dimension segments ``p_combo``.

    Uses the separable (Kronecker-product) structure of the ND model:
    the output at relative offset ``(j_0, j_1, ..., j_{L-1})`` equals

        output = (C_0 A_0^{j_0}) ⊗ (C_1 A_1^{j_1}) ⊗ … ⊗ (C_{L-1} A_{L-1}^{j_{L-1}}) @ xs

    which is computed efficiently by reshaping ``xs`` into an L-dimensional
    tensor and contracting each mode with the corresponding propagated output
    matrix.

    Parameters
    ----------
    nd_cost : NDCompositeCost
    xs : ndarray of shape (N,)
        Combined state vector (N = prod of per-dim orders).
    p_combo : tuple of int, length L
        Segment index to use for each dimension.
    thd : float
        Threshold for truncating infinite segment boundaries.

    Returns
    -------
    ab_ranges : list of ndarray
        Per-dimension integer offset arrays.
    values : ndarray of shape (len_ab_0, len_ab_1, ..., len_ab_{L-1})
        ALSSM output values at each relative offset combination.
    """
    L = nd_cost.L
    Ns = [nd_cost.cost_terms[l].get_alssm_order() for l in range(L)]
    x_tensor = np.array(xs).reshape(Ns)

    cost_segs_per_dim = [nd_cost.cost_terms[l]._get_cost_segments() for l in range(L)]
    ab_ranges = [cost_segs_per_dim[l][p_combo[l]].segment._ab_range(thd) for l in range(L)]

    # Build per-dimension propagated output matrices: CA_powers[l] shape (len_ab_l, N_l)
    CA_powers = []
    for l in range(L):
        cs_l = cost_segs_per_dim[l][p_combo[l]]
        rows = np.array(
            [cs_l.alssm.C @ matrix_power(cs_l.alssm.A, int(j)) for j in ab_ranges[l]]
        )  # shape (len_ab_l, N_l)
        CA_powers.append(rows)

    # Contract x_tensor sequentially over dimensions l = 0, 1, ..., L-1.
    # At each step we contract axis 0 of the current result (holding the
    # remaining state components for dimension l) with the N_l axis of
    # CA_powers[l].  The new len_ab_l axis is appended at the end.
    # After all L contractions the result shape is already correct:
    #   (len_ab_0, len_ab_1, ..., len_ab_{L-1})
    result = x_tensor
    for l in range(L):
        result = np.tensordot(result, CA_powers[l], axes=([0], [1]))
    return ab_ranges, result


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

        if isinstance(cost, NDCompositeCost):
            return Trajectory._eval_nd(cost, xs, thd=thd,
                                       merged_ks=merged_ks, merged_seg=merged_seg)

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
    def _eval_nd(cost, xs, thd=1e-6, merged_ks=True, merged_seg=True):
        r"""
        ND variant of [`eval`][lmlib.statespace.trajectory.Trajectory.eval] for [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost].

        Parameters
        ----------
        cost : NDCompositeCost
        xs : array_like of shape (..., N) or (N,)
            State vector(s). The last dimension must equal the combined model
            order N = prod(N_l).
        thd : float
        merged_ks : bool
            Merge across the state-vector (anchor) dimension with nanmax.
        merged_seg : bool
            Merge across all segment-combination tuples with nanmax.

        Returns
        -------
        out : ndarray of dtype=object
            Array of ``(ab_ranges, values_tensor)`` tuples.
            Without merging: shape ``(P_0*…*P_{L-1}, *XS)`` where each
            ``values_tensor`` has shape ``(len_ab_0, …, len_ab_{L-1})``.
            After both merges: single ``(ab_ranges, values_tensor)`` tuple.
        """
        xs = np.asarray(xs)
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]       # (1, N)

        XS = xs.shape[:-1]              # anchor dimensions, e.g. (num_anchors,)
        L = cost.L
        P_per_dim = [len(cost.cost_terms[l]._get_cost_segments()) for l in range(L)]
        all_combos = list(itertools_product(*[range(P) for P in P_per_dim]))
        n_combos = len(all_combos)

        # out shape: (n_combos, *XS), dtype=object, each cell is (ab_ranges, values)
        out = np.empty((n_combos, *XS), dtype=object)
        for combo_idx, p_combo in enumerate(all_combos):
            for anchor_idx in np.ndindex(XS):
                ab_ranges, values = _eval_nd_patch(cost, xs[anchor_idx], p_combo, thd)
                out[(combo_idx, *anchor_idx)] = (ab_ranges, values)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            if merged_ks:
                # Merge across anchor dimension (axis 1) using element-wise nanmax
                merged = np.empty(n_combos, dtype=object)
                for combo_idx in range(n_combos):
                    # Collect all values tensors for this combo across anchors
                    all_vals = [out[(combo_idx, *idx)][1] for idx in np.ndindex(XS)]
                    ab_ranges = out[(combo_idx, *next(iter(np.ndindex(XS))))][0]
                    merged_vals = np.full_like(all_vals[0], np.nan)
                    for v in all_vals:
                        merged_vals = np.fmax(merged_vals, v)
                    merged[combo_idx] = (ab_ranges, merged_vals)
                out = merged

            if merged_seg:
                # Merge across all segment combos
                if merged_ks:
                    # out is 1-D array of length n_combos
                    all_ab = out[0][0]
                    all_vals = [out[i][1] for i in range(n_combos)]
                    merged_vals = np.full_like(all_vals[0], np.nan)
                    for v in all_vals:
                        merged_vals = np.fmax(merged_vals, v)
                    out = (all_ab, merged_vals)
                else:
                    # out shape (n_combos, *XS); merge along axis 0
                    merged = np.empty(XS, dtype=object)
                    for anchor_idx in np.ndindex(XS):
                        all_ab = out[(0, *anchor_idx)][0]
                        all_vals = [out[(ci, *anchor_idx)][1] for ci in range(n_combos)]
                        merged_vals = np.full_like(all_vals[0], np.nan)
                        for v in all_vals:
                            merged_vals = np.fmax(merged_vals, v)
                        merged[anchor_idx] = (all_ab, merged_vals)
                    out = merged

        return out

    @staticmethod
    def eval_y(cost, xs, ks, K, F=None, thd=1e-6, merged_ks=True, merged_seg=True, fill_value=np.nan):
        r"""
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
            If ``cost`` is an [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost].
        """
        if isinstance(cost, NDCompositeCost):
            return Trajectory._eval_y_nd(cost, xs, ks, K, thd=thd,
                                         merged_ks=merged_ks, merged_seg=merged_seg,
                                         fill_value=fill_value)

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

    @staticmethod
    def _eval_y_nd(cost, xs, ks, K, thd=1e-6, merged_ks=True, merged_seg=True,
                   fill_value=np.nan):
        r"""
        ND variant of [`eval_y`][lmlib.statespace.trajectory.Trajectory.eval_y] for [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost].

        Maps ND ALSSM trajectories centred at anchor positions ``ks`` into a
        multi-dimensional output array of shape ``K``.

        Parameters
        ----------
        cost : NDCompositeCost
        xs : array_like of shape (N,) or (K[0], K[1], …, N)
            A single combined state vector, or an array of state vectors with
            one per grid position (indexed by ``ks``).
        ks : tuple/list of int, length L
            A single L-dimensional anchor position ``(k_0, k_1, …, k_{L-1})``,
            or an array of such positions of shape ``(n_anchors, L)``.
        K : tuple/list of int, length L
            Output grid shape ``(K_0, K_1, …, K_{L-1})``.
        thd : float
        merged_ks : bool
        merged_seg : bool
        fill_value : scalar

        Returns
        -------
        out : ndarray of shape ``K`` (after both merges).
        """
        L = cost.L
        K = tuple(K)
        ks = np.asarray(ks)

        # Normalise: single anchor → shape (1, L)
        if ks.ndim == 1:
            ks = ks[np.newaxis, :]   # (1, L)

        n_anchors = ks.shape[0]
        xs = np.asarray(xs)

        # Resolve state vectors: support (N,), (n_anchors, N), or (*K, N)
        if xs.ndim == 1:
            # Single state vector → replicate for all anchors
            xs_per_anchor = np.tile(xs[np.newaxis, :], (n_anchors, 1))
        elif xs.shape == (n_anchors, cost.get_alssm_order()):
            xs_per_anchor = xs
        elif xs.shape == (*K, cost.get_alssm_order()):
            # One state per grid position; look up by anchor index
            xs_per_anchor = np.array([xs[tuple(ks[i])] for i in range(n_anchors)])
        else:
            raise ValueError(
                f"xs shape {xs.shape} is incompatible with n_anchors={n_anchors}, "
                f"N={cost.get_alssm_order()}, K={K}."
            )

        P_per_dim = [len(cost.cost_terms[l]._get_cost_segments()) for l in range(L)]
        all_combos = list(itertools_product(*[range(P) for P in P_per_dim]))
        n_combos = len(all_combos)

        # out shape: (n_combos, n_anchors, *K)
        out = np.full((n_combos, n_anchors, *K), fill_value=fill_value)

        for combo_idx, p_combo in enumerate(all_combos):
            for ai in range(n_anchors):
                anchor = ks[ai]  # shape (L,)
                ab_ranges, values = _eval_nd_patch(cost, xs_per_anchor[ai], p_combo, thd)
                # values shape: (len_ab_0, len_ab_1, ..., len_ab_{L-1})
                # Map each relative offset combo to its absolute grid position
                grids = np.meshgrid(*[ab_ranges[l] for l in range(L)], indexing='ij')
                # grids[l] has shape (len_ab_0, ..., len_ab_{L-1})
                abs_coords = [grids[l] + anchor[l] for l in range(L)]
                # Build mask: all dims in bounds
                mask = np.ones(values.shape, dtype=bool)
                for l in range(L):
                    mask &= (abs_coords[l] >= 0) & (abs_coords[l] < K[l])

                # Assign valid positions
                valid_coords = tuple(abs_coords[l][mask] for l in range(L))
                # Use nanmax merge: write only if new value is larger
                dest = out[(combo_idx, ai, *valid_coords)]
                src = values[mask]
                out[(combo_idx, ai, *valid_coords)] = np.fmax(dest, src)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            if merged_ks:
                out = np.nanmax(out, axis=1)   # (n_combos, *K)
            if merged_seg:
                out = np.nanmax(out, axis=0)   # (*K)

        return out

