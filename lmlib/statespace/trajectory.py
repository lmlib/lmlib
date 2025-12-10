import numpy as np
from lmlib.statespace.cost_v2 import CostSegment, CompositeCost
from lmlib.statespace.segment import window_range

__all__ = ['Trajectory']



class Trajectory:

    @staticmethod
    def get_local(cost, xs, F=None, thd=1e-6):
        """
        Returns local trajectories of cost segments for each sample in xs

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
        out : ndarray of shape (..., P)
            For each state vector in `xs` and Cost-Segment in `cost`, returns a tuple of (window_range, trajectory)

        Examples
        --------
        TODO
        """

        if isinstance(cost, CostSegment):
            cost_segments = [cost]
        elif isinstance(cost, CompositeCost):
            cost_segments = cost._get_cost_segments(F)
        else:
            raise ValueError('cost must be CostSegment or CompositeCost')

        out = np.empty(xs.shape[:-1] + (len(cost_segments),), dtype=tuple)
        ab_ranges = [window_range(cs.segment, thd) for cs in cost_segments]
        for *idx, p in np.ndindex(out.shape):
            out[*idx][p] = ab_ranges[p], cost_segments[p].alssm.trajectory(xs[*idx], ab_ranges[p])
        return out

    @staticmethod
    def get_mapped(cost, xs, ks, K, merged_ks=False, merged_seg=False, F=None, thd=1e-6):
        trajs = Trajectory.get_local(cost, xs, F, thd)
        return trajs

    @staticmethod
    def plot(ax, trajs, **kwargs):
        pass

