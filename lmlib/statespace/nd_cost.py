"""
Definition of recursively computed squared error cost functions (such as *Cost Segments* and *Composite Costs*),
all based on ALSSMs
"""

import numpy as np
from numpy.linalg import inv, cond
from abc import ABC, abstractmethod
from collections.abc import Iterable
import warnings
import copy
import sys

from lmlib.statespace.model import ModelBase, AlssmSum
from lmlib.statespace.backends import *
from lmlib.statespace.backend import get_backend, BACKEND_TYPES, available_backends
from lmlib.utils.check import *



class ndCostModel():
    pass


class ndRLSAlssm():

    def __init__(self, nd_cost_model, steady_state=True, calc_W=False, calc_xi=True, calc_kappa=True, calc_nu=False,
                 kappa_diag=True, betas=None, filter_form='auto', backend=None):

        self.nd_cost_model = nd_cost_model
        self.steady_state = steady_state
        self.calc_W = calc_W
        self.calc_xi = calc_xi
        self.calc_kappa = calc_kappa
        self.calc_nu = calc_nu
        self.kappa_diag = kappa_diag
        self.betas = betas

        self.filter_form = filter_form
        self._is_multichannel = None
        self._is_multiset = None
        self._backend = backend if backend else get_backend()

        self._xi0 = None
        self._xi1 = None
        self._xi2 = None

    @property
    def nd_cost_model(self):
        """ndCostModel : n-Dimensional Cost Model"""
        return self._nd_cost_model

    @nd_cost_model.setter
    def nd_cost_model(self, nd_cost_model):
        assert isinstance(nd_cost_model, ndCostModel), 'nd_cost_model is not a subclass of ndCostModel'
        self._nd_cost_model = nd_cost_model
        self._nd_cost_model.__class__ = ndCostModel

    @property
    def betas(self):
        """~numpy.ndarray : Segment scalars weights the cost function per segment"""
        return self._betas

    @betas.setter
    def betas(self, betas):
        if betas is None:
            self._betas = None
        else:
            raise NotImplementedError('betas is not implemented yet.')


    @property
    def filter_form(self):
        """str : Set the form of filter to be used. Options:'parallel', 'cascade' 'auto' (Default)"""
        return self._filter_form

    @filter_form.setter
    def filter_form(self, filter_form):
        assert filter_form in ('parallel', 'cascade',
                               'auto'), 'Unknown filter_form value. Options: parallel, cascade, auto.'
        self._filter_form = filter_form

    @property
    def W(self):
        """:class:`~numpy.ndarray` : Filter Parameter :math:`W`"""
        return self._xi2.reshape(self._N, self._N) if self._steady_state else self._xi2.reshape(self._K, self._N,
                                                                                                self._N)

    @property
    def xi(self):
        """:class:`~numpy.ndarray` :  Filter Parameter :math:`\\xi`"""
        return self._xi1

    @property
    def kappa(self):
        """:class:`~numpy.ndarray` : Filter Parameter :math:`\\kappa`"""
        return self._xi0

    @property
    def nu(self):
        """:class:`~numpy.ndarray` : Filter Parameter :math:`\\nu`"""
        return self._nu

    @property
    def calc_W(self):
        """bool : Do :math:`W` parameter calculation"""
        return self._calc_W

    @calc_W.setter
    def calc_W(self, calc_W):
        assert isinstance(calc_W, bool), "calc_W not of type bool"
        self._calc_W = calc_W

    @property
    def calc_xi(self):
        """bool : Do :math:`\\xi` parameter calculation"""
        return self._calc_xi

    @calc_xi.setter
    def calc_xi(self, calc_xi):
        assert isinstance(calc_xi, bool), "calc_xi not of type bool"
        self._calc_xi = calc_xi

    @property
    def calc_kappa(self):
        """bool : Do  :math:`\\kappa` parameter calculation"""
        return self._calc_kappa

    @calc_kappa.setter
    def calc_kappa(self, calc_kappa):
        assert isinstance(calc_kappa, bool), "calc_kappa not of type bool"
        self._calc_kappa = calc_kappa

    @property
    def calc_nu(self):
        """bool : Do  :math:`\\nu` parameter calculation"""
        return self._calc_nu

    @calc_nu.setter
    def calc_nu(self, calc_nu):
        assert isinstance(calc_nu, bool), "calc_nu not of type bool"
        self._calc_nu = calc_nu

    @property
    def steady_state(self):
        """bool : Use steady state Matrix :math:`W`"""
        return self._steady_state

    @steady_state.setter
    def steady_state(self, steady_state):
        assert isinstance(steady_state, bool), "steady_state not of type bool"
        self._steady_state = steady_state

    @property
    def kappa_diag(self):
        """bool : Use the diagonal of :math:`\\kappa` when :math:`y` is a set shape."""
        return self._kappa_diag

    @kappa_diag.setter
    def kappa_diag(self, kappa_diag):
        assert isinstance(kappa_diag, bool), "kappa_diag not of type bool"
        self._kappa_diag = kappa_diag


    def filter(self, Y, V=None):

        if self.calc_W:
            # allocate xi2 memory
            recursion_xi2_nd_cost_model()
        if self.calc_xi:
            # allocate xi1 memory
            self._xi1[:] = recursion_xi1_nd_cost_model(cm, Y, cm.ndim())

def recursion_xi2_nd_cost_model():
    pass

def recursion_xi1_nd_cost_model(cm, Y, n):

    if n==1:
        return recursion_xi1_cost_model(cm.cost_model_at_dim(n-1), Y)
    else:
        xi_n = recursion_xi1_nd_cost_model(cm, Y, n-1)
        Y_n = np.reshape(xi_n, )
        return recursion_xi1_cost_model(cm.cost_model_at_dim(n-1), Y)

