"""
RLSAlssm Module
===============

TODOs
-----
- Create backends for minimization methods, i.E numpy (default) and jit

"""

import sys
from typing import Union

import numpy as np
from numpy.linalg import inv, cond

from lmlib.statespace.backend import get_backend, BACKEND_TYPES, available_backends
from lmlib.statespace.cost import CompositeCost
from lmlib.statespace.model import AlssmSum
from lmlib.utils.check import *
from lmlib.statespace.backends.rec_numpy import *
from lmlib.statespace.backends.rec_lfilter import *
from lmlib.statespace.backends.rec_jit import *

#  TODO: Why was this implemented?
WARNING_NOT_STEADY_STATE = True
"""bool : If True, a warning is issued if the steady state is not used when no sample weights are provided"""

class RLSAlssm:
    """
    Filter and Data container for Recursive Least Square Alssm Filters

    :class:`RLSAlssm` computes and stores intermediate values such as covariances,
    as required to efficiently solve recursive least squares problems
    between a model-based cost function :class:`CompositeCost` or :class:`CostSegment` and given observations.
    The intermediate variables are observation dependant and therefore the memory consumption of :class:`RLSAlssm`
    scales linearly with the observation vector length.

    Main intermediate variables are the covariance `W`, weighted mean `\\xi`, signal energy `\\kappa`, weighted number
    of samples `\\nu`, see Equation (4.6) in [Wildhaber2019]_
    :download:`PDF <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=49>`


    Parameters
    ----------
    cost_model : CostSegment, CompositeCost, CostBase
        Cost Model
    steady_state : bool, optional
        If true, the RLSAlssm uses the steady state matrix of :math:`W` instead of the recursion. Default = True
    calc_W : bool, optional
        If false, RLSAlssm prohibits the calculation and memory allocation of :math:`W`. Default = True
    calc_xi : bool, optional
        If false, RLSAlssm prohibits the calculation and memory allocation of :math:`\\xi`. Default = True
    calc_kappa : bool, optional
        If false RLSAlssm prohibits the calculation and memory allocation of :math:`\\kappa`. Default = True
    calc_nu : bool, optional
        If false RLSAlssm prohibits the calculation and memory allocation of :math:`\nu`. Default = True
    kappa_diag : bool, optional
        If false RLSAlssm stores the full Outer product in case the Signal y is a multi-set. Default = True
    betas : array_like of shape=(P, ) of floats, None, optional
        Segment Scalars. Factors weighting each of the `P` cost segments.
        If `betas` is not set, the weight is for each cost segment 1.
    filter_form : str, optional
        Set the form of filter to be used. Default is 'auto' and selects based on the model the appropriate form,
        based on precision and speed.
        - `filter_form='parallel'` for parallel block form
        - `filter_form='cascade'` for cascade block form
    backend : str, None
        Sets an individual backend for the RLSAlssm.
    """

    def __init__(self, cost_model: CompositeCost,
                 steady_state:bool=True,
                 calc_W:bool=True,
                 calc_xi:bool=True,
                 calc_kappa:bool=True,
                 calc_nu:bool=True,
                 kappa_diag:bool=True,
                 betas=None,
                 filter_form:str='cascade',
                 backend:Union[None, str]=None):

        self.cost_model = cost_model
        self.steady_state = steady_state
        self.calc_W = calc_W
        self.calc_xi = calc_xi
        self.calc_kappa = calc_kappa
        self.calc_nu = calc_nu
        self.kappa_diag = kappa_diag
        self.betas = betas
        self.filter_form = filter_form

        if backend:
            assert backend in available_backends, f"{backend} is not a valid backend"
            self._backend = backend
        else:
            self._backend = get_backend()

        self._K = None
        self._N = None
        self._S = None
        self._xi0 = None
        self._xi1 = None
        self._xi2 = None


    # Properties
    @property
    def cost_model(self) -> CompositeCost:
        """CostBase : Cost Model"""
        return self._cost_model

    @cost_model.setter
    def cost_model(self, cost_model):
        assert isinstance(cost_model, CompositeCost), 'cost_model is not a subclass of CompositeCost'
        self._cost_model = cost_model

    @property
    def betas(self):
        """~numpy.ndarray : Segment scalars weights the cost function per segment"""
        return self._betas

    @betas.setter
    def betas(self, betas):
        P = len(self.cost_model.segments)
        if betas is None:
            self._betas = np.ones(P)
        else:
            assert is_array_like(betas), 'betas if not array_like'
            assert P == len(betas), f'betas has wrong length, {info_str_found_shape(betas)}'
            self._betas = np.array(betas)

    @property
    def filter_form(self) -> str:
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
    def calc_W(self) -> bool:
        """bool : Do :math:`W` parameter calculation"""
        return self._calc_W

    @calc_W.setter
    def calc_W(self, calc_W):
        assert isinstance(calc_W, bool), "calc_W not of type bool"
        self._calc_W = calc_W

    @property
    def calc_xi(self) -> bool:
        """bool : Do :math:`\\xi` parameter calculation"""
        return self._calc_xi

    @calc_xi.setter
    def calc_xi(self, calc_xi):
        assert isinstance(calc_xi, bool), "calc_xi not of type bool"
        self._calc_xi = calc_xi

    @property
    def calc_kappa(self) -> bool:
        """bool : Do  :math:`\\kappa` parameter calculation"""
        return self._calc_kappa

    @calc_kappa.setter
    def calc_kappa(self, calc_kappa):
        assert isinstance(calc_kappa, bool), "calc_kappa not of type bool"
        self._calc_kappa = calc_kappa

    @property
    def calc_nu(self) -> bool:
        """bool : Do  :math:`\\nu` parameter calculation"""
        return self._calc_nu

    @calc_nu.setter
    def calc_nu(self, calc_nu):
        assert isinstance(calc_nu, bool), "calc_nu not of type bool"
        self._calc_nu = calc_nu

    @property
    def steady_state(self) -> bool:
        """bool : Use steady state Matrix :math:`W`"""
        return self._steady_state

    @steady_state.setter
    def steady_state(self, steady_state):
        assert isinstance(steady_state, bool), "steady_state not of type bool"
        self._steady_state = steady_state

    @property
    def kappa_diag(self) -> bool:
        """bool : Use the diagonal of :math:`\\kappa` when :math:`y` is a set shape."""
        return self._kappa_diag

    @kappa_diag.setter
    def kappa_diag(self, kappa_diag):
        assert isinstance(kappa_diag, bool), "kappa_diag not of type bool"
        self._kappa_diag = kappa_diag

    def _allocate_xi2(self, K, N):
        self._xi2 = np.zeros((K, N*N))

    def _allocate_xi1(self, K, N, S):
        self._xi1 = np.zeros((K, N)) if S is None else np.zeros((K, N, S))

    def _allocate_xi0(self, K, S):
        self._xi0 = np.zeros((K,)) if S is None else np.zeros((K, S))

    def _allocate_nu(self, K):
        self._nu = np.zeros((K,))

    # Functions
    def filter(self, y, v=None) -> None:
        """
        Computes the intermediate parameters for subsequent squared error computations and minimization's.

        Computes the intermediate parameters using efficient forward- and backward recursions.
        The results are stored internally, ready to solve the least squares problem using e.g., :meth:`minimize_x`
        or :meth:`minimize_v`. The parameter allocation :meth:`allocate` is called internally,
        so a manual pre-allocation is not necessary.

        Parameters
        ----------
        y : array_like
            Input signal |br|
            - Single-channel signal is of `shape =(K,)` for |br|
            - Multi-channel signal is of `shape =(K,L)` |br|
            - Single-channel set signals is of `shape =(K,S)` for |br|
            - Multi-channel set signals is of `shape =(K,L,S)` |br|
            - Multi-channel-sets signal is of `shape =(K,L,S)`
        v : array_like, shape=(K,), optional
            Sample weights. Weights the parameters for a time step `k` and is the same for all multi-channels.
            By default, the sample weights are initialized to 1.


        |def_K|
        |def_L|
        |def_S|

        """

        segments = self.cost_model.segments
        alssms = self.cost_model.alssms
        F = self.cost_model.F
        betas = self.betas

        # get signal length
        K = len(y)
        self._K = K
        if v is not None:
            assert len(v) == K, "length of v is not equal to length of y"
        elif self._backend in ('numpy', 'jit'):
            v = np.ones(K)
        elif self._backend == 'lfilter':
            v = 1
        # get model order
        alssm = AlssmSum(alssms)
        N = alssm.N
        self._N = N
        # get signal set dimension
        if np.ndim(alssm.C) == 2:
            if np.ndim(y) == 2:
                S = None
            elif np.ndim(y) == 3:
                S = np.shape(y)[2]
            else:
                raise ValueError("y does not have a valid shape")
        else:
            if np.ndim(y) == 1:
                S = None
            elif np.ndim(y) == 2:
                S = np.shape(y)[1]
            else:
                raise ValueError("y does not have a valid shape")
        self._S = S

        # not implemented: JIT for Multi-Set
        if S is not None and self._backend == 'jit':
            raise NotImplementedError("Just In Time (jit) backend is not implemented for Multi-Set Signals.")

        if self.calc_W:
            if self.steady_state:
                self._xi2 = self.cost_model.get_steady_state_W().flatten()
            else:
                self._allocate_xi2(K, N)

                for seg, f, beta in zip(segments, F.T, betas):
                    alssm = AlssmSum(alssms, deltas=f)

                    if self._backend == 'numpy':
                        numpy_recursion_xi2(self._xi2,
                                            alssm.A, alssm.C,
                                            seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                            y, v, beta)

                    elif self._backend == 'jit':
                        jit_recursion_xi2(self._xi2,
                                            alssm.A, alssm.C,
                                            seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                            y, v, beta)

                    elif self._backend == 'lfilter':
                        if self._filter_form == 'cascade':
                            lfilter_cascade_xi2(self._xi2,
                                                alssm.A, alssm.C,
                                                seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                                y, v, beta)

                        elif self._filter_form == 'parallel':
                            lfilter_parallel_xi2(self._xi2,
                                                alssm.A, alssm.C,
                                                seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                                y, v, beta)
                        else:
                            raise ValueError("unknown filter-form: '{}'".format(self._filter_form))
                    else:
                        raise ValueError("unknown backend: '{}'".format(self._backend))

        if self.calc_xi:
            self._allocate_xi1(K, N, S)

            for seg, f, beta in zip(segments, F.T, betas):
                alssm = AlssmSum(alssms, deltas=f)

                if self._backend == 'numpy':
                    numpy_recursion_xi1(self._xi1,
                                        alssm.A, alssm.C,
                                        seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                        y, v, beta)

                elif self._backend == 'jit':
                    jit_recursion_xi1(self._xi1,
                                        alssm.A, alssm.C,
                                        seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                        y, v, beta)

                elif self._backend == 'lfilter':
                    if self._filter_form == 'cascade':
                        lfilter_cascade_xi1(self._xi1,
                                            alssm.A, alssm.C,
                                            seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                            y, v, beta)

                    elif self._filter_form == 'parallel':
                        lfilter_parallel_xi1(self._xi1,
                                            alssm.A, alssm.C,
                                            seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                            y, v, beta)
                    else:
                        raise ValueError("unknown filter-form '{}'".format(self._filter_form))
                else:
                    raise ValueError("unknown backend: '{}'".format(self._backend))

        if self.calc_kappa:
            self._allocate_xi0(K, S)

            for seg, f, beta in zip(segments, F.T, betas):
                alssm = AlssmSum(alssms, deltas=f)

                if self._backend == 'numpy':
                    numpy_recursion_xi0(self._xi0,
                                        alssm.A, alssm.C,
                                        seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                        y, v, beta)

                elif self._backend == 'jit':
                    jit_recursion_xi0(self._xi0,
                                        alssm.A, alssm.C,
                                        seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                        y, v, beta)

                elif self._backend == 'lfilter':
                    if self._filter_form == 'cascade':
                        lfilter_cascade_xi0(self._xi0,
                                            alssm.A, alssm.C,
                                            seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                            y, v, beta)
                    elif self._filter_form == 'parallel':
                        lfilter_parallel_xi0(self._xi0,
                                            alssm.A, alssm.C,
                                            seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                            y, v, beta)
                    else:
                        raise ValueError("unknown filter-form '{}'".format(self._filter_form))
                else:
                    raise ValueError("unknown backend: '{}'".format(self._backend))

        if self.calc_nu:
            self._allocate_nu(K)

            for seg, f, beta in zip(segments, F.T, betas):
                alssm = AlssmSum(alssms, deltas=f)

                if self._backend == 'numpy':
                    numpy_recursion_nu(self._nu,
                                        alssm.A, alssm.C,
                                        seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                        y, v, beta)

                elif self._backend == 'jit':
                    jit_recursion_nu(self._nu,
                                       alssm.A, alssm.C,
                                       seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                       y, v, beta)

                elif self._backend == 'lfilter':
                    if self._filter_form == 'cascade':
                        lfilter_cascade_nu(self._nu,
                                            alssm.A, alssm.C,
                                            seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                            y, v, beta)

                    elif self._filter_form == 'parallel':
                        lfilter_parallel_nu(self._nu,
                                           alssm.A, alssm.C,
                                           seg.a, seg.b, seg.direction, seg.delta, seg.gamma,
                                           y, v, beta)
                    else:
                        raise ValueError("unknown filter-form '{}'".format(self._filter_form))
                else:
                    raise ValueError("unknown backend: '{}'".format(self._backend))

    def minimize_v(self, H=None, h=None, return_constrains=False):
        r"""
        Returns the vector `v` of the squared error minimization with linear constraints

        Minimizes the squared error over the vector `v` with linear constraints with an (optional) offset
        [Wildhaber2018]_ [TABLE V].

        **Constraint:**

        - *Linear Scalar* : :math:`x=Hv,\,v\in\mathbb{R}`

          known : :math:`H \in \mathbb{R}^{N \times 1}`

          :math:`\hat{v}_k = \frac{\xi_k^{\mathsf{T}}H}{H^{\mathsf{T}}W_k H}`

        - *Linear Combination With Offset* : :math:`x=Hv +h,\,v\in\mathbb{M}`

          known : :math:`H \in \mathbb{R}^{N \times M},\,h\in\mathbb{R}^N`

          :math:`\hat{v}_k = \big(H^{\mathsf{T}}W_k H\big)^{-1} H^\mathsf{T}\big(\xi_k - W_k h\big)`


        Parameters
        ----------
        H : array_like, shape=(N, M)
            Matrix for linear constraining :math:`H`
        h : array_like, shape=(N, [S]), optional
            Offset vector for linear constraining :math:`h`
        return_constrains : bool
            If set to True, the output is extended by H and h

        Returns
        -------
        v : :class:`~numpy.ndarray`, shape = (K, M)
            Least square state vector estimate for each time index.
            The shape of one state vector `x[k]` is `(N, [S])`, where k is the time index of `K` samples,
            `N` the ALSSM order.

        |def_K|
        |def_N|

        """

        if self._xi2 is None or self._xi1 is None:
            raise ValueError("Not all Parameters are calculated to perform minimization. \n"
                             "Check. calc_W=True or steady_state=True and calc_xi=True, ")
        # check and init H
        if H is None:
            H = np.eye(self._N)
            HTWH = self.W
        else:
            H = np.asarray(H)
            if H.shape[0] != self._N:
                ValueError(f"First dimension of constrain matrix H needs to be of size {self._N} (model order), "
                           f"{info_str_found_shape(H)}.")
            HTWH = H.T @ self.W @ H

        # check and init h
        if h is None:
            h = np.zeros(self._N)
            HTxiWh = np.einsum('nm, km...-> kn...', H.T, self._xi1)
        else:
            if h.shape[0] != self._N:
                ValueError(f"First dimension of offset vector h needs to be of size {self._N} (model order), "
                           f"{info_str_found_shape(h)}.")
            HTxiWh = np.einsum('nm, km...-> kn...', H.T, self._xi1 - self.W @ h)

        # constrained minimization
        M = H.shape[1]
        v = np.full((self._K, M, self._S) if self._S else (self._K, M), np.nan)
        msk = cond(HTWH) < 1 / sys.float_info.epsilon
        if self._steady_state:
            assert msk, 'H.T @ W @ H is not invertible.'
            v[...] = np.einsum('nm, km...-> kn...', inv(HTWH), HTxiWh)
        else:
            v[msk] = np.einsum('knm, kn... -> km...', inv(HTWH[msk]), HTxiWh[msk])

        if return_constrains:
            return v, H, h
        return v

    def minimize_x(self, H=None, h=None):
        r"""
        Returns the state vector `x` of the squared error minimization with linear constraints

        Minimizes the squared error over the state vector `x`.
        If needed its possible to apply linear constraints with an (optional) offset.
        [Wildhaber2018]_ [TABLE V].

        **Constraint:**

        - *Linear Scalar* : :math:`x=Hv,\,v\in\mathbb{R}`

          known : :math:`H \in \mathbb{R}^{N \times 1}`

        - *Linear Combination With Offset* : :math:`x=Hv +h,\,v\in\mathbb{M}`

          known : :math:`H \in \mathbb{R}^{N \times M},\,h\in\mathbb{R}^N`

        See also :meth:`minimize_v`

        Parameters
        ----------
        H : array_like, shape=(N, M), optional
            Matrix for linear constraining :math:`H`
        h : array_like, shape=(N, [S]), optional
            Offset vector for linear constraining :math:`h`

        Returns
        -------
        xs : :class:`~numpy.ndarray` of shape = (K, N)
            Least square state vector estimate for each time index.
            The shape of one state vector `x[k]` is `(N,)`, where `k` is the time index of `K` samples,
            `N` the ALSSM order.


        |def_K|
        |def_N|

        """

        if self._xi2 is None or self._xi1 is None:
            raise ValueError("\nNot all Parameters are calculated to perform minimization. \n"
                             "Set: calc_W=True or steady_state=True and calc_xi=True, ")

        if H is None and h is None:
            msk = cond(self.W) < 1 / sys.float_info.epsilon
            x = np.full_like(self.xi, np.nan)
            if self._steady_state:
                assert msk, 'Steady State W Matrix is not invertible.'
                x[...] = np.einsum('nm, km...-> kn...', inv(self.W), self.xi)
            else:
                assert np.any(msk), 'All W Matrices are not invertible.'
                x[msk] = np.einsum('knm, kn... -> km...', inv(self.W[msk]), self.xi[msk])
            return x

        v, H, h = self.minimize_v(H, h, return_constrains=True)
        x = np.einsum('nm, km...->kn...', H, v)
        x += h[np.newaxis,:] if self._S is None else h[np.newaxis,:, np.newaxis]
        return x

    def eval_errors(self, xs, ks=None):
        r"""
        Evaluation of the squared error for multiple state vectors `xs`.

        The return value is the squared error

        .. math::
            J(x)  = x^{\mathsf{T}}W_kx -2*x^{\mathsf{T}}\xi_k + \kappa_k

        for each state vector :math:`x` from the list `xs`.


        Parameters
        ----------
        xs : array_like of shape=(K, N)
            List of state vectors :math:`x`
        ks : None, array_like of int of shape=(XS,)
            List of indices where to evaluate the error

        Returns
        -------
        J : :class:`np.ndarray` of shape=(XS,)
            Squared Error for each state vector


        |def_K|
        |def_XS|
        |def_N|

        """

        if self._steady_state:
            J = np.einsum('kn..., kn...->k...', xs, np.einsum('nm, km...->kn...', self.W, xs))

        if ks is None:
            if not self._steady_state:
                J = np.einsum('kn..., kn...->k...', xs, np.einsum('knm, km...->kn...', self.W, xs))
            return J - 2 * np.einsum('kn..., kn...->k...', self.xi, xs) + self.kappa

        else:
            if not self._steady_state:
                J = np.einsum('kn..., kn...->k...', xs[ks], np.einsum('knm, km...->kn...', self.W[ks], xs[ks]))
            return J - 2 * np.einsum('kn..., kn...->k...', self.xi[ks], xs[ks]) + self.kappa[ks]

    def filter_minimize_x(self, y, v=None, H=None, h=None):
        """
        Combination of :meth:`RLSAlssmBase.filter` and :meth:`RLSAlssmBase.minimize_x`.

        This method has the same output as calling the methods

        .. code::

            rls.filter(y)
            xs = rls.minimize_x()


        See Also
        --------
        :meth:`RLSAlssmBase.filter`, :meth:`RLSAlssmBase.minimize_x`

        """

        self.filter(y, v)
        return self.minimize_x(H, h)

    def filter_minimize_v(self, y, v=None, H=None, h=None, **kwargs):
        """
        Combination of :meth:`RLSAlssmBase.filter` and :meth:`RLSAlssmBase.minimize_v`.

        This method has the same output as calling the methods

        .. code::

            rls.filter(y)
            xs = rls.minimize_v()


        See Also
        --------
        :meth:`RLSAlssmBase.filter`, :meth:`RLSAlssmBase.minimize_v`

        """

        self.filter(y, v)
        return self.minimize_v(H, h, **kwargs)

    def filter_minimize_yhat(self, y, v=None, H=None, h=None, alssm_weights=None, c0s=None):
        """
        Combination of :meth:`RLSAlssmBase.filter` and :meth:`RLSAlssmBase.minimize_x` and
        :meth:`CostBase.eval_alssm_output`

        This method has the same output as calling the methods

        .. code::

            xs = rls.filter_minimize_x()
            y_hat = rls.cost_model.eval_alssm_output(xs)

        See Also
        --------
        :meth:`RLSAlssmBase.filter`, :meth:`RLSAlssmBase.minimize_x`

        """
        xs = self.filter_minimize_x(y, v=v, H=H, h=h)
        return self.cost_model.eval_alssm_output(xs, alssm_weights=alssm_weights, c0s=c0s)

    def set_backend(self, backend):
        """
        Setting the backend computations option

        Parameters
        ----------
        backend : str
            'numpy' for State-Space python backend
            'lfilter' for Transfer Function python backend
            'jit' for Just in Time backend

        """
        assert backend in BACKEND_TYPES, f'Wrong backend name {backend}'
        assert backend in available_backends, f'{backend} not available, check {backend} installation.'
        self._backend = backend
