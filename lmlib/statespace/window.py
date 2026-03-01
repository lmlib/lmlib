import numpy as np
import warnings

__all__ = ['Window']

class Window:
    @staticmethod
    def eval(cost, segment_indices=None, thd=1e-6):

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
    def eval_y(cost, ks, K, merged_ks=True, merged_seg=True, segment_indices=None, thd=1e-6, fill_value=0):

        # return an empty array if ks is empty
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