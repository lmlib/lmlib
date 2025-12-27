"""
Recursive Least Square Alssm Classes to solve Alssm Cost Functions


x

"""

import sys
from typing import Union

from numpy.core.numeric import moveaxis
from numpy.linalg import inv, cond

from lmlib.statespace.backend import get_backend
from lmlib.statespace.cost_old import CompositeCost, NDCompositeCost
from lmlib.statespace.model import AlssmSum, Alssm
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

    def filter(self, y, v=None, dim_order=None):

        # check dimension order
        if dim_order is None:
            dim_order = np.arange(self.cost.get_number_of_dimensions())
        assert len(dim_order) == self.cost.L, f'dim_order has wrong length, {info_str_found_shape(dim_order)}'


