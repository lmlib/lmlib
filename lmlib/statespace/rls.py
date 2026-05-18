"""
Recursive Least Square Alssm Classes to solve Alssm Cost Functions


"""

import sys
import numpy as np
from numpy.linalg import inv, cond, matrix_power, eigvals
from scipy.signal import zpk2sos

from lmlib.statespace.backend import get_backend
from lmlib.statespace.cost import CompositeCost, CostSegment, NDCompositeCost
from lmlib.statespace.model import AlssmSum
from lmlib.utils.check import *
from lmlib.statespace.backends.rec import *
import warnings


__all__ = ['RLSAlssm', 'create_rls']


def create_rls(cost, multi_channel_set=False, steady_state=False, kappa_diag=True, steady_state_method='closed_form', **kwargs):
    """Deprecated: instantiate RLS classes directly instead."""
    warnings.warn(
        "create_rls() is deprecated and will be removed in a future version. "
        "Instantiate RLSAlssm object directly with e.g. rls = lm.RLSAlssm(cost)",
        FutureWarning,
        stacklevel=2,
    )
    return RLSAlssm(cost, steady_state=steady_state, **kwargs)


def _as_composite_cost(cost):
    """
    Ensure *cost* is a CompositeCost.

    A CostSegment is a degenerate CompositeCost with M=1 ALSSM, P=1 Segment and
    F = [[1]].  Wrapping it here means the rest of the filtering code never
    needs to branch on the concrete cost type.
    """
    if isinstance(cost, CostSegment):
        return CompositeCost(
            [cost.alssm],
            [cost.segment],
            np.ones((1, 1)),
            betas=np.array([cost.beta]),
            label=cost.label,
        )
    return cost  # already a CompositeCost


class RLSAlssm:

    def __init__(self, cost_terms, steady_state=True, calc_W=True, calc_xi=True, calc_kappa=True, calc_nu=False, filter_form='cascade',
                 backend=None, numdenom=None):
        self._cost_terms = cost_terms
        assert all(isinstance(_, bool) for _ in (steady_state, calc_W, calc_xi, calc_kappa, calc_nu)), \
            'steady_state, calc_W, calc_xi, calc_kappa and calc_nu must be boolean.'

        self._steady_state = steady_state
        self._calc_W = calc_W
        self._calc_xi = calc_xi
        self._calc_kappa = calc_kappa
        self._calc_nu = calc_nu

        self._N = self._cost_terms.get_alssm_order()

        self._xi0 = None
        self._xi1 = None
        self._xi2 = None
        self._nu = None

        self._backend = backend if backend is not None else get_backend(cost_terms)
        
        self._filter_form = filter_form

        # Collect per-dimension CompositeCosts once, reused throughout __init__
        # and the recursion methods.
        #   - NDCompositeCost._get_sub_cost_term() returns a list of costs.
        #   - CompositeCost/CostSegment._get_sub_cost_term() returns the cost
        #     itself (a single object, not a list).
        # Normalise to a list of CompositeCosts in both cases.
        raw = cost_terms._get_sub_cost_term()
        if isinstance(raw, list):
            _sub_costs = [_as_composite_cost(ct) for ct in raw]
        else:
            _sub_costs = [_as_composite_cost(raw)]

        # ------------------------------------------------------------------
        # check if filter form is valid for all segments
        # ------------------------------------------------------------------
        if self._backend == 'lfilter' and self._filter_form == 'cascade':
            for ct in _sub_costs:
                for alssm in ct.alssms:
                    if not np.allclose(alssm.A, np.triu(alssm.A)):
                        self._filter_form = 'parallel'
                        print("State-Space Matrix A is not upper triangular, "
                              "cascade version can't be used. "
                              "Defaulting to filter_form='parallel'.")

        # ------------------------------------------------------------------
        # Build _numdenom as a 3-D structure: _numdenom[dim_index][seg_index][alssm_index]
        #
        # Each leaf entry is either None (numpy/jit backends, or inactive grid
        # node with f_mp==0) or [denom, num_b, num_a] (lfilter parallel backend).
        #
        # For q==1 (xi) the filtering is now done per individual ALSSM, so each
        # ALSSM gets its own transfer-function coefficients derived from its own
        # (small) A_m and C_m rather than from the combined block-diagonal matrix.
        # This avoids constructing the large AlssmSum just to decompose it again.
        #
        # For q==2 (W) the combined AlssmSum is still used (see recursion methods),
        # so _numdenom[dim][p][m] is not consumed for q==2; the combined entry is
        # built on-the-fly there instead.
        #
        # ------------------------------------------------------------------
        self._numdenom = [
            [[None] * ct.M for _ in range(ct.P)] for ct in _sub_costs
        ]

        if self._filter_form == 'parallel' and self._backend == 'lfilter' and numdenom is None:
            from lmlib.statespace.backends.statespace_tools import ss2zpk_qz
            from lmlib.statespace.backends.rec_lfilter import (
                _zpk_cancel_and_build_sos, _count_poles_in_sos)
            for dim_idx, ct in enumerate(_sub_costs):
                for p, segment in enumerate(ct.segments):
                    gamma = segment.gamma
                    a     = segment.a
                    b     = segment.b

                    for m, alssm_m in enumerate(ct.alssms):
                        f_mp = ct.F[m, p]
                        if f_mp == 0.0:
                            # Inactive grid node — leave entry as None.
                            continue

                        wrapped = AlssmSum([alssm_m], [f_mp], force_MC=True)
                        A = wrapped.A
                        C = wrapped.C
                        N_m = wrapped.N

                        if segment.direction == 'fw':
                            gAT   = (1.0 / gamma) * inv(A).T
                            Aac   = (matrix_power(A, 0 if np.isinf(a) else a - 1).T @ C.T).ravel()
                            Abc   = (matrix_power(A, b).T @ C.T).ravel()
                        elif segment.direction == 'bw':
                            gAT   = gamma * A.T
                            Aac   = (matrix_power(A, a).T @ C.T).ravel()
                            Abc   = (matrix_power(A, 0 if np.isinf(b) else b + 1).T @ C.T).ravel()
                        else:
                            raise NotImplementedError("Segment direction must be fw or bw")

                        # --- Build SOS structures ---
                        # Poles = eigenvalues of gAT (avoids polynomial expansion).
                        # A shared sos_iir is stored at index [0] for backward
                        # compatibility with user-supplied numdenom dicts.
                        poles = eigvals(gAT)
                        sos_iir_shared = zpk2sos(np.zeros(len(poles)), poles, 1.0)

                        # Zeros via QZ (Rosenbrock pencil + generalised Schur
                        # decomposition): avoids the Faddeev-LeVerrier polynomial
                        # round-trip of scipy ss2tf / ss2zpk, giving exact zeros for
                        # cancellable rows and near-MATLAB accuracy for the rest.
                        # PZ cancellation reduces each row to the minimal-order IIR.
                        # _numdenom layout:
                        #   [0] sos_iir_shared  – full-order IIR (legacy compat)
                        #   [1] sos_b_list      – per-row FIR SOS, boundary b
                        #   [2] sos_a_list      – per-row FIR SOS, boundary a
                        #   [3] db_list         – per-row FIR delay, boundary b
                        #   [4] da_list         – per-row FIR delay, boundary a
                        #   [5] sos_iir_b_list  – per-row reduced IIR SOS, boundary b
                        #   [6] sos_iir_a_list  – per-row reduced IIR SOS, boundary a
                        #   [7] n_poles_b_list  – pole count per row, boundary b
                        #   [8] n_poles_a_list  – pole count per row, boundary a
                        #   [9] advance_b_list  – backward slice advance, boundary b
                        #   [10] advance_a_list – backward slice advance, boundary a
                        #
                        # advance_*_list[n_] = 1 when the boundary vector component
                        # boundary_vec[n_] = 0, which causes the IIR slice alignment
                        # to be off by 1 sample in the backward filter.  This happens
                        # because the QZ transfer function H_n for row n_ has the
                        # correct mathematical form but the IIR slice includes one
                        # extra "warmup" sample that must be dropped.
                        # For the forward filter the analogous issue (huge spurious
                        # zero from QZ) is corrected in _zpk_cancel_and_build_sos.
                        Abc_col = Abc.reshape(N_m, 1)
                        Aac_col = Aac.reshape(N_m, 1)
                        sos_b_list = []; sos_a_list = []; db_list = []; da_list = []
                        sos_iir_b_list = []; sos_iir_a_list = []
                        n_poles_b_list = []; n_poles_a_list = []
                        advance_b_list = []; advance_a_list = []
                        for n_ in range(N_m):
                            C_row = np.zeros((1, N_m)); C_row[0, n_] = 1.0
                            z_b, _, k_b, n_inf_b = ss2zpk_qz(gAT, Abc_col, C_row)
                            z_a, _, k_a, n_inf_a = ss2zpk_qz(gAT, Aac_col, C_row)
                            sb, db, si_b = _zpk_cancel_and_build_sos(z_b, k_b, poles, n_inf_zeros=n_inf_b)
                            sa, da, si_a = _zpk_cancel_and_build_sos(z_a, k_a, poles, n_inf_zeros=n_inf_a)
                            sos_b_list.append(sb); sos_a_list.append(sa)
                            db_list.append(db);    da_list.append(da)
                            sos_iir_b_list.append(si_b); sos_iir_a_list.append(si_a)
                            n_poles_b_list.append(_count_poles_in_sos(si_b))
                            n_poles_a_list.append(_count_poles_in_sos(si_a))
                            # Advance = 1 when the boundary vector component at row
                            # n_ is zero.  Only the backward filter uses this.
                            advance_b_list.append(1 if abs(float(Abc[n_])) < 1e-10 else 0)
                            advance_a_list.append(1 if abs(float(Aac[n_])) < 1e-10 else 0)

                        self._numdenom[dim_idx][p][m] = [sos_iir_shared, sos_b_list,
                                                          sos_a_list, db_list, da_list,
                                                          sos_iir_b_list, sos_iir_a_list,
                                                          n_poles_b_list, n_poles_a_list,
                                                          advance_b_list, advance_a_list]



        if filter_form == 'parallel' and backend == 'lfilter' and numdenom is not None:
            print("Using user-supplied SOS coefficients (numdenom).")
            self._numdenom = numdenom

    # ------------------------------------------------------------------
    # Properties 
    # ------------------------------------------------------------------

    @property
    def xi2(self):
        return self.W

    @property
    def xi1(self):
        return self.xi

    @property
    def xi0(self):
        return self.kappa

    @property
    def W(self):
        """:class:`~numpy.ndarray` : Filter Parameter :math:`W`"""
        if self._xi2 is None:
            raise ValueError('xi2 has not been calculated. '
                             'Please run the filter() method with calc_W=True before calling W.')
        return self._xi2.reshape(self._xi2.shape[:-1] + (self._N, self._N))

    @property
    def xi(self):
        """:class:`~numpy.ndarray` :  Filter Parameter :math:`\\xi`"""
        if self._xi1 is None:
            raise ValueError('xi1 has not been calculated. '
                             'Please run the filter() method with calc_xi=True before calling xi.')
        return self._xi1

    @property
    def kappa(self):
        """:class:`~numpy.ndarray` : Filter Parameter :math:`\\kappa`"""
        if self._xi0 is None:
            raise ValueError('xi0 has not been calculated. '
                             'Please run the filter() method with calc_kappa=True before calling kappa.')
        return self._xi0

    @property
    def nu(self):
        """:class:`~numpy.ndarray` : Filter Parameter :math:`\\nu`"""
        # TODO nu implementation
        raise NotImplementedError("nu calculation is not yet implemented.")

    # ------------------------------------------------------------------
    # filter 
    # ------------------------------------------------------------------

    def filter(self, y, sample_weights=None, dim_order=None):

        # -------- check dimension order --------
        L = self._cost_terms.get_number_of_dimensions()
        if dim_order is None:
            dim_order = np.arange(L)
        assert len(dim_order) == L, f'dim_order has wrong length, {info_str_found_shape(dim_order)}'
        if L > 1 and self._calc_W and not self._steady_state:
            assert False, "for multidimensional ALSSMs, W requires steady_state=True"

        # -------- broadcast and check y --------
        Q = self._cost_terms.get_alssm_output_dimension()
        y = np.asarray(y)
        if isinstance(self._cost_terms, (CompositeCost, CostSegment)):
            if Q == 0:  # scalar output
                if y.ndim == 1:  # 1 dim signal
                    y = y.reshape(-1, 1)
                elif y.ndim >= 2:
                    if y.shape[1] == 1:
                        pass #already has correct dimension
                    elif y.shape[-1] != 1:  # multi dimension signal (processed in parallel)
                        y = y.reshape(*y.shape, 1)
                    else:
                        raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')    
                else:
                    raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')
            elif Q == 1:  # 1-dimensional output
                if y.ndim == 1 or y.shape[-1] != Q:
                    raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')
            else:
                if y.shape[-1] != Q:
                    raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')

        if isinstance(self._cost_terms, NDCompositeCost):
            if Q == 0:  # scalar output
                if y.ndim == L:
                    y = y.reshape(*y.shape, 1)
                else:
                    raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')
            if 1 <= Q != y.shape[-1]:
                raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')

        # -------- check sample weight --------
        if sample_weights is None:
            sample_weights = np.broadcast_to(1., y.shape[:-1])
        else:
            if np.shape(sample_weights) != y.shape[:-1]:
                raise ValueError(f'sample_weights has wrong shape, {info_str_found_shape(sample_weights)}')

        # -------- calc xi2 --------
        if self._steady_state:
            self._xi2 = self._cost_terms.get_steady_state_W(dim_order).flatten()
        elif self._calc_W and not self._steady_state:
            q = 2
            xi_prev = self._nd_xi_q_recursion(q, y, sample_weights, dim_order[0])
            for nd_dim in dim_order[1:]:
                xi_prev = self._nd_xi_q_asterisk_l_recursion(xi_prev, q, y, sample_weights, nd_dim)
            self._xi2 = xi_prev

        # -------- calc xi1 --------
        if self._calc_xi:
            q = 1
            xi_prev = self._nd_xi_q_recursion(q, y, sample_weights, dim_order[0])
            for nd_dim in dim_order[1:]:
                xi_prev = self._nd_xi_q_asterisk_l_recursion(xi_prev, q, y, sample_weights, nd_dim)
            self._xi1 = xi_prev

        # -------- calc xi0 --------
        if self._calc_kappa:
            q = 0
            xi_prev = self._nd_xi_q_recursion(q, y, sample_weights, dim_order[0])
            for nd_dim in dim_order[1:]:
                xi_prev = self._nd_xi_q_asterisk_l_recursion(xi_prev, q, y, sample_weights, nd_dim)
            self._xi0 = xi_prev[..., 0]

        # -------- calc nu --------
        # TODO

    # ------------------------------------------------------------------
    # minimize / eval 
    # ------------------------------------------------------------------

    def minimize_v(self, H=None, h=None):

        _H = np.eye(self._N) if H is None else np.asarray(H)
        _h = np.zeros(self._N) if h is None else np.asarray(h)
        assert _H.shape[0] == self._N, f'H has wrong shape, {info_str_found_shape(H)}'
        assert _h.shape[0] == self._N, f'h has wrong shape, {info_str_found_shape(h)}'

        if H is None:
            HTWH = self.W
        else:
            HTWH = _H.T @ self.W @ _H

        if h is None:
            HTxiWh = np.einsum('nm, ...m-> ...n', _H.T, self.xi)
        else:
            HTxiWh = np.einsum('nm, ...m-> ...n', _H.T, self.xi - self.W @ _h)

        v = np.full(self.xi.shape[:-1] + (_H.shape[1],), np.nan)
        msk = cond(HTWH) < 1 / sys.float_info.epsilon
        if self._steady_state:
            assert msk, 'H.T @ W @ H is not invertible.'
            np.einsum('nm, ...m-> ...n', inv(HTWH), HTxiWh, out=v)
        else:
            v[msk] = np.einsum('...nm, ...m -> ...n', inv(HTWH[msk]), HTxiWh[msk])

        return v

    def minimize_x(self, H=None, h=None):

        v = self.minimize_v(H, h)

        if H is None:
            x = v
        else:
            x = np.einsum('nm, ...m-> ...n', H, v)

        if h is not None:
            x += h

        return x

    def eval_errors(self, xs):

        if self._steady_state:
            J = np.einsum('...n, ...n', xs, np.einsum('nm, ...m->...n', self.W, xs))
        else:
            J = np.einsum('...n, ...n', xs, np.einsum('...nm, ...m->...n', self.W, xs))

        return J - 2 * np.einsum('...n, ...n', self.xi, xs) + self.kappa

    def fit(self, y, output='y_hat', sample_weights=None, dim_order=None, H=None, h=None, eval_alssm_weights=None):

        if isinstance(output, str):
            _output = (output,)
        else:
            _output = tuple(output)
        assert len(_output) != 0, 'output is empty. Must be a string or a tuple of strings.'
        assert any(_ in ('y_hat', 'x', 'v') for _ in _output), (f'output contains unknown entries: {_output}'
                                                                 f'. Allowed entries are "y_hat", "x", "v".')
        self.filter(y, sample_weights, dim_order)

        v = self.minimize_v(H, h)
        if _output == ('v',):
            return v

        out_dict = {'v': v}

        if H is None:
            x = v
        else:
            x = np.einsum('nm, ...m-> ...n', H, v)

        if h is not None:
            x += h

        out_dict['x'] = x
        if _output == ('x',):
            return x
        if 'y_hat' not in _output:
            return (out_dict[_] for _ in _output)

        alssms = self._cost_terms.get_alssms()
        weights = eval_alssm_weights if eval_alssm_weights is not None else [1.0] * len(alssms)
        out_dict['y_hat'] = AlssmSum(alssms, weights).eval_output(x)

        if _output == ('y_hat',):
            return out_dict['y_hat']
        return tuple(out_dict[_] for _ in _output)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _alssm_offsets(composite_cost):
        """
        Return the cumulative state-vector offsets for each ALSSM in a
        CompositeCost.

        Returns an array of shape (M+1,) so that ALSSM m occupies
        xi[..., offsets[m] : offsets[m+1]].
        """
        orders = [alssm.N for alssm in composite_cost.alssms]
        return np.concatenate([[0], np.cumsum(orders)])

    # ------------------------------------------------------------------
    # Recursion: _nd_xi_q_recursion
    # ------------------------------------------------------------------

    def _nd_xi_q_recursion(self, q, y, sample_weights, model_dimension):
        """
        Compute xi^(q) for one dimension of the cost.

        Key change vs. the previous implementation
        -------------------------------------------
        Previously, _get_cost_segments(force_MC=True) was called, which merged
        all M ALSSMs for each segment p into a single monolithic AlssmSum with a
        block-diagonal A of size N×N.  The recursion then operated on the full
        xi vector of length N^q.

        Now we iterate over each individual ALSSM m within every segment p and
        write the result into the appropriate sub-slice of xi_curr.  Because A
        is block-diagonal, the blocks are independent and can be computed
        separately.  Each ALSSM m is wrapped in a single-element AlssmSum with
        weight F[m,p] so that:
          - force_MC is honoured (C is guaranteed 2-D for all backends), and
          - the F-column weight is folded into C (as lambda), keeping beta
            equal to the original segment beta.
        """
        # Normalise to CompositeCost so we always have .alssms, .segments, .F
        sub_cost = _as_composite_cost(
            self._cost_terms._get_sub_cost_term(model_dimension)
        )

        # dim_index: position of this dimension inside _numdenom.
        # For NDCompositeCost each dimension has its own slot.
        # For CompositeCost/CostSegment there is exactly one slot (index 0)
        # regardless of which model_dimension axis is being processed.
        dim_index = model_dimension if isinstance(self._cost_terms, NDCompositeCost) else 0

        N = sub_cost.get_alssm_order()
        *Ks, Q = np.shape(y)
        xi_curr = np.zeros((*Ks, N ** q), order='F')  # last dimension is the nd-model-order

        # Move the model_dimension axis and flatten all other spatial dims so
        # the inner loop sees a simple (n_parallel, K, Q) array.
        _xi_curr = np.moveaxis(xi_curr, model_dimension, -2)
        _xi_curr = np.reshape(_xi_curr, (-1, *_xi_curr.shape[-2:]))
        _y = np.moveaxis(y, model_dimension, -2)
        _y = np.reshape(_y, (-1, *_y.shape[-2:]))
        _sample_weights = np.moveaxis(sample_weights, model_dimension, -1)
        _sample_weights = np.reshape(_sample_weights, (-1, *_sample_weights.shape[-1:]))

        offsets = self._alssm_offsets(sub_cost)  # shape (M+1,)

        for p, segment in enumerate(sub_cost.segments):
            beta_p = sub_cost.betas[p]

            # ------------------------------------------------------------------
            # q == 2  (W matrix)
            # The full W of a CompositeCost contains cross-terms between
            # different ALSSMs.  These sit at non-contiguous positions in the
            # flattened N^2 vector, so per-ALSSM sub-slicing is not correct
            # for M > 1 (more than 1 model). Use the combined AlssmSum (original behaviour). 
            # _numdenom is not consumed here; numdenom_p is irrelevant for q==2 with 
            # the combined path (numdenom would need to match the large A).
            # ------------------------------------------------------------------
            if q == 2:
                combined = AlssmSum(sub_cost.alssms, sub_cost.F[:, p], force_MC=True)
                for i in range(_y.shape[0]):
                    xi_q_recursion(
                        _xi_curr[i], q,
                        combined, segment,
                        _y[i], _sample_weights[i],
                        beta_p, self._backend, self._filter_form, None,
                    )
                continue  # next segment — inner m-loop not needed for q==2

            # ------------------------------------------------------------------
            # q == 0  (kappa)
            # The recursion for kappa does not use alssm.A or alssm.C at all
            # (it only accumulates y² weighted by the window).  A single pass
            # per segment is therefore sufficient regardless of how many ALSSMs
            # are present.  We pass the first ALSSM as a dummy placeholder 
            # 
            # TODO: is following wrong? It seems to contradict with the documentation after if q == 0
            # and
            # sum all F weights for this segment column into a single effective
            # beta so the overall scaling remains correct.
            # ------------------------------------------------------------------
            if q == 0:
                # kappa = integral(y^2 * window) is a property of the segment
                # alone — it is independent of the ALSSM structure and the F
                # weights.  One pass per segment with the unmodified beta_p is
                # all that is needed.  (Multiplying by f_sum, as was done
                # previously, incorrectly double-counts segments that have more
                # than one active ALSSM.)
                dummy_alssm = AlssmSum([sub_cost.alssms[0]], [1.0], force_MC=True)
                for i in range(_y.shape[0]):
                    xi_q_recursion(
                        _xi_curr[i], q,
                        dummy_alssm, segment,
                        _y[i], _sample_weights[i],
                        beta_p, self._backend, self._filter_form, None,
                    )
                continue

            # ------------------------------------------------------------------
            # q == 1  (xi): iterate per ALSSM, write into per-ALSSM sub-slice
            # ------------------------------------------------------------------
            for m, alssm_m in enumerate(sub_cost.alssms):
                f_mp = sub_cost.F[m, p]
                if f_mp == 0.0:
                    continue  # inactive grid node — skip

                # Wrap the individual ALSSM in a single-element AlssmSum.
                # This serves two purposes:
                #   1. force_MC=True ensures C is always 2-D (required by the
                #      lfilter cascade backend).
                #   2. The F-weight f_mp is absorbed into C via the lambda
                #      argument, exactly as AlssmSum(alssms, F[:,p]) did,
                #      without mutating the original ALSSM object.
                wrapped = AlssmSum([alssm_m], [f_mp], force_MC=True)

                # Per-ALSSM sub-slice — the main optimisation.
                # Block-diagonal A means ALSSM m contributes only to
                # elements [offsets[m] : offsets[m+1]] of xi.
                n0, n1 = offsets[m], offsets[m + 1]
                numdenom_pm = self._numdenom[dim_index][p][m]
                for i in range(_y.shape[0]):
                    xi_q_recursion(
                        _xi_curr[i, :, n0:n1], q,
                        wrapped, segment,
                        _y[i], _sample_weights[i],
                        beta_p, self._backend, self._filter_form, numdenom_pm,
                    )

        return xi_curr

    # ------------------------------------------------------------------
    # Recursion: _nd_xi_q_asterisk_l_recursion
    # ------------------------------------------------------------------

    def _nd_xi_q_asterisk_l_recursion(self, xi_prev, q, y, sample_weights, model_dimension):
        """
        Compute the cross-dimensional xi^(q)* for one additional dimension.

        The same per-ALSSM decomposition as _nd_xi_q_recursion is applied:
        each ALSSM m in the CompositeCost for this dimension is processed
        independently, writing into the corresponding sub-slice of xi_curr.
        """
        sub_cost = _as_composite_cost(
            self._cost_terms._get_sub_cost_term(model_dimension)
        )

        dim_index = model_dimension if isinstance(self._cost_terms, NDCompositeCost) else 0

        N = sub_cost.get_alssm_order()
        Nq_prev = xi_prev.shape[-1]
        *Ks, Q = np.shape(y)
        xi_curr = np.zeros((*Ks, Nq_prev * N ** q), order='F')

        _xi_curr = np.moveaxis(xi_curr, model_dimension, 0)
        _xi_prev = np.moveaxis(xi_prev, model_dimension, 0)
        _sample_weights = np.moveaxis(sample_weights, model_dimension, 0)

        offsets = self._alssm_offsets(sub_cost)

        for p, segment in enumerate(sub_cost.segments):
            beta_p = sub_cost.betas[p]

            # q == 2: use combined AlssmSum (old behaviour), called once per segment.
            if q == 2:
                combined = AlssmSum(sub_cost.alssms, sub_cost.F[:, p], force_MC=True)
                xi_q_asterisk_l_recursion(
                    _xi_curr, q,
                    combined, segment,
                    _xi_prev, _sample_weights,
                    beta_p, self._backend, self._filter_form, None,
                )
                continue

            # q == 0: kappa asterisk — independent of ALSSM structure and F.
            # One pass per segment with the unmodified beta_p is sufficient.
            if q == 0:
                if all(sub_cost.F[m, p] == 0.0
                       for m in range(len(sub_cost.alssms))):
                    continue  # segment fully inactive — skip
                dummy_alssm = AlssmSum([sub_cost.alssms[0]], [1.0], force_MC=True)
                xi_q_asterisk_l_recursion(
                    _xi_curr, q,
                    dummy_alssm, segment,
                    _xi_prev, _sample_weights,
                    beta_p, self._backend, self._filter_form, None,
                )
                continue

            # q == 1: per-ALSSM-pair asterisk recursion.
            #
            # The output xi_2d must follow the Kronecker layout
            # xi_2d[n_prev * N_curr + n_curr], matching W = W_dim0 ⊗ W_dim1.
            # A naive per-ALSSM split (grouping by the dim-1/current-pass ALSSM)
            # would produce the block-transposed ordering and yield J < 0.
            #
            # The correct approach exploits block-diagonality by processing each
            # (m_prev, m_curr) ALSSM pair independently:
            #   1. Slice xi_prev to only the m_prev states (last-axis sub-slice).
            #   2. Call the recursion with the m_curr wrapped ALSSM on that slice.
            #      INq = eye(N_prev_m) so the kron gives N_prev_m * N_curr_m output.
            #   3. Write the result into the correct (n_prev, n_curr) sub-block of
            #      xi_curr viewed as a (... N_total, N_total) matrix.
            #
            # This gives O(M^2) calls each operating on small (N_m × N_m) systems
            # instead of one call on the full (N_total × N_total) system, while
            # producing exactly the same result.
            for m_curr, (curr_n0, curr_n1) in enumerate(
                    zip(offsets[:-1], offsets[1:])):
                f_curr = sub_cost.F[m_curr, p]
                if f_curr == 0.0:
                    continue
                wrapped_curr = AlssmSum(
                    [sub_cost.alssms[m_curr]], [f_curr], force_MC=True)

                for m_prev, (prev_n0, prev_n1) in enumerate(
                        zip(offsets[:-1], offsets[1:])):
                    N_prev_m = prev_n1 - prev_n0
                    N_curr_m = curr_n1 - curr_n0

                    # Sub-slice of xi_prev for m_prev states (last axis, contiguous)
                    xi_prev_slice = _xi_prev[..., prev_n0:prev_n1]

                    # Temporary output: (..., N_prev_m * N_curr_m)
                    xi_tmp = np.zeros(
                        (*_xi_prev.shape[:-1], N_prev_m * N_curr_m))

                    xi_q_asterisk_l_recursion(
                        xi_tmp, q,
                        wrapped_curr, segment,
                        xi_prev_slice, _sample_weights,
                        beta_p, self._backend, self._filter_form, None,
                    )

                    # Write into the flat xi_curr at the correct strided positions.
                    # The flat layout is xi[..., n_prev * N_curr + n_curr], so
                    # the (m_prev, m_curr) block occupies a strided sub-tensor.
                    xi_tmp_mat = xi_tmp.reshape(
                        *_xi_prev.shape[:-1], N_prev_m, N_curr_m)
                    for i_prev in range(N_prev_m):
                        for i_curr in range(N_curr_m):
                            n_prev = prev_n0 + i_prev
                            n_curr = curr_n0 + i_curr
                            _xi_curr[..., n_prev * Nq_prev + n_curr] += xi_tmp_mat[..., i_prev, i_curr]

        return xi_curr
