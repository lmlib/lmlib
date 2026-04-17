"""
Recursive Least Square Alssm Classes to solve Alssm Cost Functions


"""

import sys
from typing import Union

import numpy as np
from numpy.core.numeric import moveaxis
from numpy.linalg import inv, cond

from lmlib.statespace.backend import get_backend
from lmlib.statespace.cost import CompositeCost, CostSegment, NDCompositeCost
from lmlib.statespace.model import AlssmSum
from lmlib.utils.check import *
from lmlib.statespace.backends.rec import *


__all__ = ['RLSAlssm']


class RLSAlssm:

    def __init__(self, cost_terms, steady_state=True, calc_W=True, calc_xi=True, calc_kappa=True, calc_nu=False, filter_form='cascade',
                 backend=None):
        self._cost_terms = cost_terms
        assert all(isinstance(_, bool) for _ in (steady_state, calc_W, calc_xi, calc_kappa, calc_nu)), \
            'steady_state, calc_W, calc_xi, calc_kappa and calc_nu must be boolean.'

        self._steady_state = steady_state
        self._calc_W = calc_W
        self._calc_xi = calc_xi
        self._calc_kappa = calc_kappa
        self._calc_nu = calc_nu

        self._filter_form = filter_form
        self._backend = backend if backend is not None else get_backend()

        self._N = self._cost_terms.get_alssm_order()

        self._xi0 = None
        self._xi1 = None
        self._xi2 = None
        self._nu = None

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

    def filter(self, y, sample_weights=None, dim_order=None):

        # -------- check dimension order --------
        L = self._cost_terms.get_number_of_dimensions()
        if dim_order is None:
            dim_order = np.arange(L)
        assert len(dim_order) == L, f'dim_order has wrong length, {info_str_found_shape(dim_order)}'
        if L > 1:
            assert self._steady_state, "for multidimensional ALSSMs steady_state=True has to be used"
        # -------- broadcast and check y --------
        Q = self._cost_terms.get_alssm_output_dimension()
        y = np.asarray(y)
        if isinstance(self._cost_terms, (CompositeCost, CostSegment)):
            if Q == 0: # scalar output
                if y.ndim == 1: # 1 dim signal
                    y = y.reshape(-1, 1)
                elif y.ndim >= 2 and y.shape[-1] != 1: # multi dimension signal (processed in parallel)
                    y = y.reshape(*y.shape, 1)
                else:
                    raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')
            elif Q == 1: # 1-dimensional output
                if y.ndim == 1 or y.shape[-1] != Q:
                    raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')
            else:
                if y.shape[-1] != Q:
                    raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')

        if isinstance(self._cost_terms, NDCompositeCost):
            if Q == 0: # scalar output
                if y.ndim == L:
                    y = y.reshape(*y.shape, 1)
                else:
                    raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')
            if 1 <= Q != y.shape[-1]:
                raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')

        # -------- check sample weight --------
        if sample_weights is None:
            # each element points to the same memory-location
            sample_weights = np.broadcast_to(1., y.shape[:-1])
        else:
            if np.shape(sample_weights) != y.shape[:-1]:
                raise ValueError(f'sample_weights has wrong shape, {info_str_found_shape(sample_weights)}')

        # -------- calc xi2 --------
        if self._steady_state:
            self._xi2 = self._cost_terms.get_steady_state_W(dim_order).flatten()
        elif self._calc_W and not self._steady_state:
            q = 2

            # first dimension
            xi_prev = self._nd_xi_q_recursion(q, y, sample_weights, dim_order[0])

            # n-dimensions
            for nd_dim in dim_order[1:]:
                xi_prev = self._nd_xi_q_asterisk_l_recursion(xi_prev, q, y, sample_weights, nd_dim)

            self._xi2 = xi_prev

        # -------- calc x1 --------
        if self._calc_xi:
            q = 1

            # first dimension
            xi_prev = self._nd_xi_q_recursion(q, y, sample_weights, dim_order[0])

            # n-dimensions
            for nd_dim in dim_order[1:]:
                xi_prev = self._nd_xi_q_asterisk_l_recursion(xi_prev, q, y, sample_weights, nd_dim)

            self._xi1 = xi_prev

        # -------- calc x0 --------
        if self._calc_kappa:
            q = 0

            # first dimension
            xi_prev = self._nd_xi_q_recursion(q, y, sample_weights, dim_order[0])

            # n-dimensions
            for nd_dim in dim_order[1:]:
                xi_prev = self._nd_xi_q_asterisk_l_recursion(xi_prev, q, y, sample_weights, nd_dim)

            self._xi0 = xi_prev[..., 0]  # remove the last dimension due to leftovers of nd-model-order

        # -------- calc nu --------
        # TODO

    def minimize_v(self, H=None, h=None):

        _H = np.eye(self._N) if H is None else np.asarray(H)
        _h = np.zeros(self._N) if h is None else np.asarray(h)
        assert _H.shape[0] == self._N, f'H has wrong shape, {info_str_found_shape(H)}'
        assert _h.shape[0] == self._N, f'h has wrong shape, {info_str_found_shape(h)}'

        # constrained minimization
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
            x+=h

        return x

    def eval_errors(self, xs):

        if self._steady_state:
            J = np.einsum('...n, ...n', xs, np.einsum('nm, ...m->...n', self.W, xs))
        else:
            J = np.einsum('...n, ...n', xs, np.einsum('...nm, ...m->...n', self.W, xs))

        return J - 2 * np.einsum('...n, ...n', self.xi, xs) + self.kappa

    def fit(self, y, output='y_hat', sample_weights=None, dim_order=None, H=None, h=None, eval_alssm_weights=None):

        # ----------- check output parameter -----------
        if isinstance(output, str):
            _output = (output,)
        else:
            _output = tuple(output)
        assert len(_output) != 0, 'output is empty. Must be a string or a tuple of strings.'
        assert any(_ in ('y_hat', 'x', 'v') for _ in _output), (f'output contains unknown entries: {_output}'
                                                                f'. Allowed entries are "y_hat", "x", "v".')
        # ----------- filter -----------
        self.filter(y, sample_weights, dim_order)

        # ----------- v calc -----------
        v = self.minimize_v(H, h)
        if _output == ('v',):
            return v

        out_dict = {'v': v}

        # ----------- x -----------
        if H is None:
            x = v
        else:
            x = np.einsum('nm, ...m-> ...n', H, v)

        if h is not None:
            x+=h

        out_dict['x'] = x
        if _output == ('x',):
            return x
        if 'y_hat' not in _output:
            return (out_dict[_] for _ in _output)

        # ----------- yhat -----------
        alssms = self._cost_terms.get_alssms()
        out_dict['y_hat'] = AlssmSum(alssms, eval_alssm_weights).eval_output(x)

        if _output == ('y_hat',):
            return out_dict['y_hat']
        return tuple(out_dict[_] for _ in _output)

    def _nd_xi_q_recursion(self, q, y, sample_weights, model_dimension):

        sub_cost = self._cost_terms._get_sub_cost_term(model_dimension)
        N = sub_cost.get_alssm_order()
        *Ks, Q = np.shape(y)
        xi_curr = np.zeros((*Ks, N ** q,)) # the last dimension is the nd-model-order

        # most efficient to access subarrays in a ndarray
        # 1. move subarray to last dimensions (returns a view)
        # 2. reshape by flattening dimensions to iterate over (returns a view)
        # the data is still stored in the original order
        _xi_curr = np.moveaxis(xi_curr, model_dimension, -2) # to second last dimension, last is nd-model-order
        _xi_curr = np.reshape(_xi_curr, (-1, *_xi_curr.shape[-2:]))
        _y = np.moveaxis(y, model_dimension, -2) # to the second last dimension, last is the model output dimension
        _y = np.reshape(_y, (-1, *_y.shape[-2:]))
        _sample_weights = np.moveaxis(sample_weights, model_dimension, -1) # to the last dimension, no model output dimension
        _sample_weights = np.reshape(_sample_weights, (-1, *_sample_weights.shape[-1:]))

        # iterate over CostSegments
        for cs in sub_cost._get_cost_segments(force_MC=True):
            # backend recursion
            for i in range(_y.shape[0]):
                xi_q_recursion(_xi_curr[i], q,
                               cs.alssm, cs.segment,
                               _y[i], _sample_weights[i],
                               cs.beta, self._backend, self._filter_form)

        return xi_curr

    def _nd_xi_q_asterisk_l_recursion(self, xi_prev, q, y, sample_weights, model_dimension):

        sub_cost = self._cost_terms._get_sub_cost_term(model_dimension)
        N = sub_cost.get_alssm_order()
        Nq_prev = xi_prev.shape[-1]
        *Ks, Q = np.shape(y)
        xi_curr = np.zeros((*Ks, Nq_prev * N ** q,)) # the last dimension is the nd-model-order

        # move subarray to first dimensions (returns a view)
        _xi_curr = np.moveaxis(xi_curr, model_dimension, 0)
        _xi_prev = np.moveaxis(xi_prev, model_dimension, 0)
        _sample_weights = np.moveaxis(sample_weights, model_dimension, 0)

        # cost segments
        # iterate over CostSegments
        for cs in sub_cost._get_cost_segments(force_MC=True):

            xi_q_asterisk_l_recursion(_xi_curr, q,
                                      cs.alssm, cs.segment,
                                      _xi_prev, _sample_weights,
                                      cs.beta, self._backend, self._filter_form)
        return xi_curr

