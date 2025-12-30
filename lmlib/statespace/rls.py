"""
Recursive Least Square Alssm Classes to solve Alssm Cost Functions


x

"""

import sys
from typing import Union

import numpy as np
from numpy.core.numeric import moveaxis
from numpy.linalg import inv, cond

from lmlib.statespace.backend import get_backend
from lmlib.statespace.cost import CompositeCost, CostSegment, NDCompositeCost
from lmlib.utils.check import *
from lmlib.statespace.backends.rec import *


__all__ = ['RLSAlssm']


class RLSAlssm:

    def __init__(self, cost, steady_state=True, calc_W=True, calc_xi=True, calc_kappa=True, calc_nu=False, filter_form='cascade',
                 backend=None):
        self.cost = cost
        self.steady_state = steady_state
        self.calc_W = calc_W
        self.calc_xi = calc_xi
        self.calc_kappa = calc_kappa
        self.calc_nu = calc_nu
        self.filter_form = filter_form
        self.backend = backend if backend is not None else get_backend()

        self._Ks = ()
        self._Ns = ()
        self._N = cost.get_model_order()

        self._xi0 = None
        self._xi1 = None
        self._xi2 = None
        self._nu = None

    def filter(self, y, w=None, dim_order=None):

        # -------- check dimension order --------
        L = self.cost.get_number_of_dimensions()
        if dim_order is None:
            dim_order = np.arange(L)
        assert len(dim_order) == L, f'dim_order has wrong length, {info_str_found_shape(dim_order)}'

        # -------- broadcast and check y --------
        Q = self.cost.get_model_output_dimension()
        y = np.asarray(y)
        if isinstance(self.cost, (CompositeCost, CostSegment)):
            if Q is 0: # scalar output
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

        if isinstance(self.cost, NDCompositeCost):
            if Q is 0: # scalar output
                if y.ndim == L:
                    y = y.reshape(*y.shape, 1)
                else:
                    raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')
            if 1 <= Q != y.shape[-1]:
                raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')

        # -------- check sample weight --------
        if w is None:
            # each element points to the same memory-location
            w = np.broadcast_to(1., y.shape[:-1])
        else:
            if np.shape(w) == y.shape[:-1]:
                raise ValueError(f'w has wrong shape, {info_str_found_shape(w)}')

        # -------- calc xi2 --------
        if self.steady_state:
            self._xi2 = self.cost.get_steady_state_W(dim_order).flatten()
        elif self.calc_W and not self.steady_state:
            q = 2

            # first dimension
            xi_prev = self._nd_xi_q_recursion(q, y, w, dim_order[0])

            # n-dimensions
            for nd_dim in dim_order[1:]:
                xi_prev = self._nd_xi_q_asterisk_l_recursion(xi_prev, q, y, w, nd_dim)

            self._xi2 = xi_prev

        # -------- calc x1 --------
        if self.calc_xi:
            q = 1

            # first dimension
            xi_prev = self._nd_xi_q_recursion(q, y, w, dim_order[0])

            # n-dimensions
            for nd_dim in dim_order[1:]:
                xi_prev = self._nd_xi_q_asterisk_l_recursion(xi_prev, q, y, w, nd_dim)

            self._xi1 = xi_prev

        # -------- calc x0 --------
        if self.calc_kappa:
            q = 0

            # first dimension
            xi_prev = self._nd_xi_q_recursion(q, y, w, dim_order[0])

            # n-dimensions
            for nd_dim in dim_order[1:]:
                xi_prev = self._nd_xi_q_asterisk_l_recursion(xi_prev, q, y, w, nd_dim)

            self._xi0 = xi_prev[..., 0]  # remove the last dimension  due to leftovers of nd-model-order

        # -------- calc nu --------
        # TODO


    def minimize(self, H=None, h=None, output='x'):
        pass


    def _nd_xi_q_recursion(self, q, y, w, model_dimension):

        sub_cost = self.cost._get_sub_cost(model_dimension)
        # betas = self.nd_betas[model_dimension] # todo
        N = sub_cost.get_model_order()
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
        _w = np.moveaxis(w, model_dimension, -1) # to the last dimension, no model output dimension
        _w = np.reshape(_w, (-1, *_w.shape[-1:]))

        # iterate over CostSegments
        for cs in sub_cost._get_cost_segments(force_MC=True):
            beta = 1 # todo

            # backend recursion
            for i in range(_y.shape[0]):
                xi_q_recursion(_xi_curr[i], q,
                               cs.alssm, cs.segment,
                               _y[i], _w[i],
                               beta, self.backend, self.filter_form)

        return xi_curr

    def _nd_xi_q_asterisk_l_recursion(self, xi_prev, q, y, w, model_dimension):

        sub_cost = self.cost._get_sub_cost(model_dimension)
        # betas = self.nd_betas[model_dimension] # todo
        N = sub_cost.get_model_order()
        Nq_prev = xi_prev.shape[-1]
        *Ks, Q = np.shape(y)
        xi_curr = np.zeros((*Ks, Nq_prev * N ** q,)) # the last dimension is the nd-model-order

        # move subarray to first dimensions (returns a view)
        _xi_curr = np.moveaxis(xi_curr, model_dimension, 0)
        _xi_prev = np.moveaxis(xi_prev, model_dimension, 0)
        _w = np.moveaxis(w, model_dimension, 0)

        # cost segments
        # iterate over CostSegments
        for cs in sub_cost._get_cost_segments(force_MC=True):
            beta = 1  # todo

            xi_q_asterisk_l_recursion(_xi_curr, q,
                                      cs.alssm, cs.segment,
                                      _xi_prev, _w,
                                      beta, self.backend, self.filter_form)
        return xi_curr

