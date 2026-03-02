import numpy as np
from lmlib.statespace.cost import CostSegment, CompositeCost

import warnings

__all__ = ['Trajectory']



class Trajectory:

    @staticmethod
    def eval(cost, xs, F=None, thd=1e-6):
        """
        Evaluates the trajectories for given state vectors `xs` over the specified `cost`.

        Parameters
        ----------
        cost : CostSegment, CompositeCost
            CostSegment or CompositeCost to compute trajectories for
        xs : array_like of shape (..., N)
            Array of state vectors defining the trajectories. Last Dimension must be the state dimension.
        F : None, array_like of shape (M, P)
            Mapping matrix :math:`F`, maps models to segment, where the value weights ALSSM Output
        thd : float, optional
            Threshold for window range computation, by default 1e-6

        Returns
        -------
        out : ndarray of shape (P, ...)
            For each state vector in `xs` and Cost-Segment in `cost`, returns a tuple of (window_range, trajectory)

        Examples
        --------
        TODO
        """
        xs = np.asarray(xs)
        cost_segments = cost._get_cost_segments(F)
        XS = xs.shape[:-1]

        out = np.empty((len(cost_segments), *XS), dtype=tuple)
        for p, cs in enumerate(cost_segments):
            ab_range = cs.segment._ab_range(thd)
            for idx in np.ndindex(XS):
                out[p][idx] = ab_range, cs.alssm.eval_output(xs[idx], ab_range)
        return out

    @staticmethod
    def eval_y(cost, xs, ks, K, merged_ks=True, merged_seg=True, F=None, thd=1e-6, fill_value=np.nan):
        """
        Evaluates the trajectories for given state vectors `xs` over the specified `cost` and
        maps trajectories at indices `ks` into a common target output vector of length `K`.

        Computes a mapped output array by processing input trajectories and mapping values based on
        conditions. The function allows for optional merging of the `ks` dimension and/or merging along
        the last axis of the trajectories.

        Parameters
        ----------
        cost : CostSegment, CompositeCost
            CostSegment or CompositeCost to compute trajectories for
        xs : array_like of shape (..., N)
            Array of state vectors defining the trajectories. Last Dimension must be the state dimension.
        ks : array_like of shape (XS,)
            A list of indices representing the mapping to be applied.
        K : int
            Length of the resulting mapped output array (first dimension size).
        merged_ks : bool, optional
            If True, merges the `ks` dimension using the maximum value along axis 1. Defaults to True.
        merged_seg : bool, optional
            If True, merges values along the last trajectory dimension using the maximum value.
            Defaults to True.
        F : array_like of shape (M, P), optional
            Mapping matrix :math:`F`, maps models to segment, where the value weights ALSSM Output
        thd : float, optional
            Threshold for window range computation, by default 1e-6
        fill_value : scalar, optional
            The fill value for initializing the output array or handling empty inputs.
            Defaults to `np.nan`.

        Returns
        -------
        numpy.ndarray of shape ([P,] [XS,] [...], K )
            A multi-dimensional mapped output array based on the input conditions and mapping indices. If
            `merged_ks` is enabled, the array will be reduced along the `ks` dimension (second dimension) axis using the
            maximum value. If `merged_seg` is enabled, the array is further reduced along the first axis.


        Notes
        -----
        If the input `ks` is empty, the function will issue a warning and return an empty array filled
        with the specified `fill_value` of size `(K,)`.

        Examples
        --------
        TODO

        """

        # return an empty array if ks is empty
        if isinstance(ks, int):
            ks = np.array([ks])
            xs = np.asarray([xs])

        if len(ks) == 0:
            print('Warning: ks is empty. Returned empty array of size K.')
            return np.full(K, fill_value=fill_value)

        if len(xs) == len(ks):
            trajs = Trajectory.eval(cost, xs, F, thd)
        elif len(xs) == K:
            trajs = Trajectory.eval(cost, xs[ks], F, thd)
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
    def plot(ax, trajs, **kwargs):
        pass

