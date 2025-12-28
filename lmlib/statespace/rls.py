"""
Recursive Least Square Alssm Classes to solve Alssm Cost Functions


x

"""

import sys
from typing import Union

from numpy.core.numeric import moveaxis
from numpy.linalg import inv, cond

from lmlib.statespace.backend import get_backend
from lmlib.statespace.cost import CompositeCost, CostSegment, NDCompositeCost
from lmlib.utils.check import *
from lmlib.statespace.backends import *


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

        # check dimension order
        if dim_order is None:
            dim_order = np.arange(self.cost.get_number_of_dimensions())
        assert len(dim_order) == self.cost.L, f'dim_order has wrong length, {info_str_found_shape(dim_order)}'

        # broadcast and check y
        y = np.asarray(y)
        Q = self.cost.get_model_output_dimension()
        if isinstance(self.cost, (CompositeCost, CostSegment)):
            if Q is 0: # scalar output
                if y.ndim == 1: # 1 dim signal
                    y = y.reshape(-1, 1)
                elif y.ndim >= 2: # multi dimension signal (processed in parallel)
                    y = y.reshape(*y.shape, 1)
                else:
                    raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')
            elif Q == 1: # 1-dimensional output
                if y.ndim == 1 or y.shape[-1] != Q:
                    raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')
        if isinstance(self.cost, NDCompositeCost):
            L = self.cost.get_number_of_dimensions()
            if Q is 0: # scalar output
                # todo
                pass

    def minimize(self, H=None, h=None, output='x'):
        pass




