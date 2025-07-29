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

        self._backend = backend if backend else get_backend()

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

                    if self._backend == 'jit':
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
        v = np.full((self._K, M, self._S) if self._is_multiset else (self._K, M), np.nan)
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
        x = np.einsum('nm, km...->kn...', H, v) + h
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


    def _check_output_dimensions(self, y):
        C = AlssmSum(self.cost_model.alssms).C

        self._is_multichannel = np.ndim(C) == 2
        self._is_multiset = np.ndim(y) == 2 and not self._is_multichannel or np.ndim(y) == 3

        if self._is_multichannel:
            if self._is_multiset:
                assert np.ndim(y) == 3 and np.shape(C)[0] == np.shape(y)[1], \
                    'Model output and observation shape does not match. ' \
                    'Multi-channel and -set systems expect shapes: ' \
                    'C shape (L, N) and y shape (K, L, S)'
            else:
                assert np.ndim(y) == 2 and np.shape(C)[0] == np.shape(y)[1], \
                    'Model output and observation shape does not match. ' \
                    'Multi-channel system expect shapes: ' \
                    'C shape (L, N) and y shape (K, L)'
        else:
            if self._is_multiset:
                assert np.ndim(y) == 2, \
                    'Model output and observation shape does not match. ' \
                    'Multi-set systems expect shapes: ' \
                    'C shape (N,) and y shape (K, S)'
            else:
                assert np.ndim(y) == 1, \
                    'Model output and observation shape does not match. ' \
                    'Scalar systems (non multi-channel/set) expect shapes:' \
                    'C shape (N,) and y shape (K,)'

    def _allocate_parameter_storage(self, input_shape):
        self._K = input_shape[0]
        self._S = input_shape[-1] if self._is_multiset else None

        if self._calc_W and not self._steady_state:
            self._xi2 = np.zeros((self._K, self._N ** 2))
        if self._steady_state:
            self._xi2 = np.zeros((self._N ** 2))
        if self._calc_xi:
            self._xi1 = np.zeros((self._K, self._N, self._S)) if self._is_multiset else np.zeros((self._K, self._N))
        if self._calc_kappa:
            if self._is_multiset:
                self._xi0 = np.zeros((self._K, self._S)) if self._kappa_diag else np.zeros((self._K, self._S, self._S))
            else:
                self._xi0 = np.zeros(self._K)
        if self._calc_nu:
            self._nu = np.zeros(self._K)

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



# def create_rls(cost, multi_channel_set=False, steady_state=False, kappa_diag=True, steady_state_method='closed_form'):
#     """
#     Returns a 'Recursive Least Squares Object' according to the provided parameters.
#
#     Parameters
#     ----------
#     cost : CostBase
#         cost model
#     multi_channel_set : bool
#         Set to True if a RLSAlssmSet* is desired
#     steady_state : bool
#         Set to True if a Steady State scheme is desired
#     kappa_diag : bool
#         If True a RLSAlssmSet* will perform a diagnoal kappa matrix. Only if multi_channel_set = True
#     steady_state_method : str
#         Type of Steady State method. Available: ('closed_form'). Only if steady_state == True
#
#     Returns
#     -------
#     out : RLSAlssm, RLSAlssmSet, RLSAlssmSteadyState or RLSAlssmSetSteadyState
#         Returns Recursive Least Square Object
#     """
#     warnings.warn('create_rls is deprecated. Use RLSAlssm() directly.', DeprecationWarning, 2)
#     return RLSAlssm(cost, steady_state=steady_state, kappa_diag=kappa_diag)


# class RLSAlssm(ABC):
#     """
#     Filter and Data container for Recursive Least Square Alssm Filters
#
#     :class:`RLSAlssm` computes and stores intermediate values such as covariances,
#     as required to efficiently solve recursive least squares problems
#     between a model-based cost function :class:`CompositeCost` or :class:`CostSegment` and given observations.
#     The intermediate variables are observation dependant and therefore the memory consumption of :class:`RLSAlssm`
#     scales linearly with the observation vector length.
#
#     Main intermediate variables are the covariance `W`, weighted mean `\\xi`, signal energy `\\kappa`, weighted number
#     of samples `\\nu`, see Equation (4.6) in [Wildhaber2019]_
#     :download:`PDF <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=49>`
#
#
#     Parameters
#     ----------
#     cost_model : CostSegment, CompositeCost, CostBase
#         Cost Model
#     steady_state : bool, optional
#         If true, the RLSAlssm uses the steady state matrix of :math:`W` instead the recursion. Default = True
#     calc_W : bool, optional
#         If false, RLSAlssm prohibits the calculation and memory allocation of :math:`W`. Default = True
#     calc_xi : bool, optional
#         If false, RLSAlssm prohibits the calculation and memory allocation of :math:`\\xi`. Default = True
#     calc_kappa : bool, optional
#         If false RLSAlssm prohibits the calculation and memory allocation of :math:`\\kappa`. Default = True
#     calc_nu : bool, optional
#         If false RLSAlssm prohibits the calculation and memory allocation of :math:`\nu`. Default = True
#     kappa_diag : bool, optional
#         If false RLSAlssm stores the full Outer product in case the Signal y is a multi-set. Default = True
#     betas : array_like of shape=(P, ) of floats, None, optional
#         Segment Scalars. Factors weighting each of the `P` cost segments.
#         If `betas` is not set, the weight is for each cost segment 1.
#     filter_form : str, optional
#         Set the form of filter to be used. Default is 'auto' and selects based on the model the appropriate form,
#         based on precision and speed.
#         - filter_form='parallel' for parallel block form
#         - filter_form= 'cascade' for cascade block form
#     backend : str, None
#         Sets an individual backend for the RLSAlssm.
#     """
#
#     def __init__(self, cost_model, steady_state=True, calc_W=True, calc_xi=True, calc_kappa=True, calc_nu=True,
#                  kappa_diag=True, betas=None, filter_form='auto', backend=None):
#         self.cost_model = cost_model
#         self.steady_state = steady_state
#         self.calc_W = calc_W
#         self.calc_xi = calc_xi
#         self.calc_kappa = calc_kappa
#         self.calc_nu = calc_nu
#         self.kappa_diag = kappa_diag
#         self.betas = betas
#         self.filter_form = filter_form
#         self._is_multichannel = None
#         self._is_multiset = None
#         self._backend = backend if backend else get_backend()
#         self._N = self.cost_model.get_model_order()
#         self._K = None
#         self._S = None
#         self._xi0 = None
#         self._xi1 = None
#         self._xi2 = None
#
#     @property
#     def cost_model(self):
#         """CostBase : Cost Model"""
#         return self._cost_model
#
#     @cost_model.setter
#     def cost_model(self, cost_model):
#         assert isinstance(cost_model, CostBase), 'cost_model is not a subclass of CostBase'
#         self._cost_model = cost_model
#         self._cost_model.__class__ = CostBase
#
#     @property
#     def betas(self):
#         """~numpy.ndarray : Segment scalars weights the cost function per segment"""
#         return self._betas
#
#     @betas.setter
#     def betas(self, betas):
#         if betas is None:
#             self._betas = None
#         else:
#             assert is_array_like(betas), 'betas if not array_like'
#             assert len(betas) == len(
#                 self._cost_model.segments), f'betas has wrong length, {info_str_found_shape(betas)}'
#             self._betas = betas
#
#     @property
#     def filter_form(self):
#         """str : Set the form of filter to be used. Options:'parallel', 'cascade' 'auto' (Default)"""
#         return self._filter_form
#
#     @filter_form.setter
#     def filter_form(self, filter_form):
#         assert filter_form in ('parallel', 'cascade',
#                                'auto'), 'Unknown filter_form value. Options: parallel, cascade, auto.'
#         self._filter_form = filter_form
#
#     @property
#     def W(self):
#         """:class:`~numpy.ndarray` : Filter Parameter :math:`W`"""
#         return self._xi2.reshape(self._N, self._N) if self._steady_state else self._xi2.reshape(self._K, self._N,
#                                                                                                 self._N)
#
#     @property
#     def xi(self):
#         """:class:`~numpy.ndarray` :  Filter Parameter :math:`\\xi`"""
#         return self._xi1
#
#     @property
#     def kappa(self):
#         """:class:`~numpy.ndarray` : Filter Parameter :math:`\\kappa`"""
#         return self._xi0
#
#     @property
#     def nu(self):
#         """:class:`~numpy.ndarray` : Filter Parameter :math:`\\nu`"""
#         return self._nu
#
#     @property
#     def calc_W(self):
#         """bool : Do :math:`W` parameter calculation"""
#         return self._calc_W
#
#     @calc_W.setter
#     def calc_W(self, calc_W):
#         assert isinstance(calc_W, bool), "calc_W not of type bool"
#         self._calc_W = calc_W
#
#     @property
#     def calc_xi(self):
#         """bool : Do :math:`\\xi` parameter calculation"""
#         return self._calc_xi
#
#     @calc_xi.setter
#     def calc_xi(self, calc_xi):
#         assert isinstance(calc_xi, bool), "calc_xi not of type bool"
#         self._calc_xi = calc_xi
#
#     @property
#     def calc_kappa(self):
#         """bool : Do  :math:`\\kappa` parameter calculation"""
#         return self._calc_kappa
#
#     @calc_kappa.setter
#     def calc_kappa(self, calc_kappa):
#         assert isinstance(calc_kappa, bool), "calc_kappa not of type bool"
#         self._calc_kappa = calc_kappa
#
#     @property
#     def calc_nu(self):
#         """bool : Do  :math:`\\nu` parameter calculation"""
#         return self._calc_nu
#
#     @calc_nu.setter
#     def calc_nu(self, calc_nu):
#         assert isinstance(calc_nu, bool), "calc_nu not of type bool"
#         self._calc_nu = calc_nu
#
#     @property
#     def steady_state(self):
#         """bool : Use steady state Matrix :math:`W`"""
#         return self._steady_state
#
#     @steady_state.setter
#     def steady_state(self, steady_state):
#         assert isinstance(steady_state, bool), "steady_state not of type bool"
#         self._steady_state = steady_state
#
#     @property
#     def kappa_diag(self):
#         """bool : Use the diagonal of :math:`\\kappa` when :math:`y` is a set shape."""
#         return self._kappa_diag
#
#     @kappa_diag.setter
#     def kappa_diag(self, kappa_diag):
#         assert isinstance(kappa_diag, bool), "kappa_diag not of type bool"
#         self._kappa_diag = kappa_diag
#
#     def _check_output_dimensions(self, y):
#         C = AlssmSum(self.cost_model.alssms).C
#
#         self._is_multichannel = np.ndim(C) == 2
#         self._is_multiset = np.ndim(y) == 2 and not self._is_multichannel or np.ndim(y) == 3
#
#         if self._is_multichannel:
#             if self._is_multiset:
#                 assert np.ndim(y) == 3 and np.shape(C)[0] == np.shape(y)[1], \
#                     'Model output and observation shape does not match. ' \
#                     'Multi-channel and -set systems expect shapes: ' \
#                     'C shape (L, N) and y shape (K, L, S)'
#             else:
#                 assert np.ndim(y) == 2 and np.shape(C)[0] == np.shape(y)[1], \
#                     'Model output and observation shape does not match. ' \
#                     'Multi-channel system expect shapes: ' \
#                     'C shape (L, N) and y shape (K, L)'
#         else:
#             if self._is_multiset:
#                 assert np.ndim(y) == 2, \
#                     'Model output and observation shape does not match. ' \
#                     'Multi-set systems expect shapes: ' \
#                     'C shape (N,) and y shape (K, S)'
#             else:
#                 assert np.ndim(y) == 1, \
#                     'Model output and observation shape does not match. ' \
#                     'Scalar systems (non multi-channel/set) expect shapes:' \
#                     'C shape (N,) and y shape (K,)'
#
#     def _allocate_parameter_storage(self, input_shape):
#         self._K = input_shape[0]
#         self._S = input_shape[-1] if self._is_multiset else None
#
#         if self._calc_W and not self._steady_state:
#             self._xi2 = np.zeros((self._K, self._N ** 2))
#         if self._steady_state:
#             self._xi2 = np.zeros((self._N ** 2))
#         if self._calc_xi:
#             self._xi1 = np.zeros((self._K, self._N, self._S)) if self._is_multiset else np.zeros((self._K, self._N))
#         if self._calc_kappa:
#             if self._is_multiset:
#                 self._xi0 = np.zeros((self._K, self._S)) if self._kappa_diag else np.zeros((self._K, self._S, self._S))
#             else:
#                 self._xi0 = np.zeros(self._K)
#         if self._calc_nu:
#             self._nu = np.zeros(self._K)
#
#     def _get_einsum_path_W_tf(self):
#         if self._is_multichannel:
#             einsum_path = 'kl, nl->kn'
#         else:
#             einsum_path = 'k, n->kn'
#         return einsum_path
#
#     def _get_einsum_path_xi_ss(self):
#         return 'nl..., l... ->n...' if self._is_multichannel else 'n..., ... ->n...'
#
#     def _get_einsum_path_xi_tf(self):
#         if self._is_multiset:
#             if self._is_multichannel:
#                 einsum_path = 'kls, nl->kns'
#             else:
#                 einsum_path = 'ks, n->kns'
#         else:
#             if self._is_multichannel:
#                 einsum_path = 'kl, nl->kn'
#             else:
#                 einsum_path = 'k, n->kn'
#         return einsum_path
#
#     def _get_einsum_path_kappa_ss(self):
#         if self._is_multiset:
#             if self._is_multichannel:
#                 if self._kappa_diag:
#                     einsum_path = 'm..., m...->...'
#                 else:
#                     einsum_path = 'ml..., ln... ->mn'
#             else:
#                 if self._kappa_diag:
#                     einsum_path = 'm, m->m'
#                 else:
#                     einsum_path = 'm, n->mn'
#         else:
#             if self._is_multichannel:
#                 einsum_path = 'm, m->...'
#             else:
#                 einsum_path = '..., ...'
#
#         return einsum_path
#
#     def _get_einsum_path_kappa_tf(self):
#         if self._is_multiset:
#             if self._is_multichannel:
#                 if self._kappa_diag:
#                     einsum_path = 'km..., ...->km...'
#                 else:
#                     einsum_path = 'kmn..., ... ->kmn'
#             else:
#                 if self._kappa_diag:
#                     einsum_path = 'km, ...->km'
#                 else:
#                     einsum_path = 'kmn, ...->kmn'
#         else:
#             if self._is_multichannel:
#                 einsum_path = 'k, ...->k'
#             else:
#                 einsum_path = 'k, ...->k'
#
#         return einsum_path
#
#     def _get_einsum_path_nu_tf(self):
#         return 'k..., ...->k...'
#
#     def _get_einsum_path_y_squared_tf(self):
#         if self._is_multiset:
#             if self._is_multichannel:
#                 if self._kappa_diag:
#                     einsum_path = 'klm..., klm...->km...'
#                 else:
#                     einsum_path = 'kml..., kln... ->kmn'
#             else:
#                 if self._kappa_diag:
#                     einsum_path = 'km, km->km'
#                 else:
#                     einsum_path = 'km, kn->kmn'
#         else:
#             if self._is_multichannel:
#                 einsum_path = 'kl, kl->k'
#             else:
#                 einsum_path = 'k, k->k'
#
#         return einsum_path
#
#     def _forward_recursion(self, A, C, segment, y, v, beta):
#         a, b, delta, gamma = segment.a, segment.b, segment.delta, segment.gamma
#
#         if self._steady_state:
#             self._xi2 += _covariance_matrix_closed_form(A, C, gamma, a, b, delta).reshape(self._N ** 2)
#
#         if self._backend == 'numpy':
#             v = np.ones(self._K) if v == 1 else v
#
#             if self._calc_W and not self._steady_state:
#                 forward_recursion_W_ss(self.W, a, b, delta, gamma, A, C, beta, y, v)
#
#             if self._calc_xi:
#                 forward_recursion_xi_ss(self._xi1, a, b, delta, gamma, A, C, beta, y, v, self._get_einsum_path_xi_ss())
#
#             if self._calc_kappa:
#                 forward_recursion_kappa_ss(self._xi0, a, b, delta, gamma, beta, y, v, self._get_einsum_path_kappa_ss())
#
#             if self._calc_nu:
#                 forward_recursion_nu_ss(self._nu, a, b, delta, gamma, beta, v)
#
#         if self._backend == 'lfilter' and self._filter_form == 'parallel':
#             raise NotImplemented('Parallel Block Form not implemented yet')
#
#         if self._backend == 'lfilter' and self._filter_form in ('cascade', 'auto'):
#
#             if self._calc_W and not self._steady_state:
#                 _A = np.kron(A, A)
#                 _C = np.kron(C, C)
#                 forward_cascade_xi(self._xi2, a, b, delta, gamma, _A, _C, beta, 1, v, einsum_path=self._get_einsum_path_W_tf())
#
#             if self._calc_xi:
#                 forward_cascade_xi(self._xi1, a, b, delta, gamma, A, C, beta, y, v, self._get_einsum_path_xi_tf())
#
#             if self._calc_kappa:
#                 _C = np.array(1)
#                 _A = np.array(1)
#                 _y = np.einsum(self._get_einsum_path_y_squared_tf(), y, y)
#                 forward_cascade_xi(self._xi0, a, b, delta, gamma, _A, _C, beta, _y, v, self._get_einsum_path_kappa_tf())
#
#             if self._calc_nu:
#                 _C = np.array(1)
#                 _A = np.array(1)
#                 forward_cascade_xi(self._nu, a, b, delta, gamma, _A, _C, beta, 1, v, self._get_einsum_path_nu_tf())
#
#         if self._backend == 'jit':
#             v = np.ones(self._K) if v == 1 else v
#             init_vars = forward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)
#
#             if self._is_multiset:
#                 if self.steady_state:
#                     raise NotImplemented('forward_recursion_set_xi_kappa_nu_jit not implemented yet')
#                 else:
#                     forward_recursion_set_jit(self.W, self._xi1, self._xi0, self._nu, segment.a, segment.b,
#                                               segment.delta, y, v, beta, *init_vars, self._kappa_diag)
#             else:
#                 if self.steady_state:
#                     forward_recursion_xi_kappa_nu_jit(self._xi1, self._xi0, self._nu, segment.a, segment.b,
#                                                       segment.delta, y, v, beta, *init_vars)
#                 else:
#                     forward_recursion_jit(self.W, self._xi1, self._xi0, self._nu, segment.a, segment.b, segment.delta,
#                                           y, v, beta, *init_vars)
#
#     def _backward_recursion(self, A, C, segment, y, v, beta):
#         a, b, delta, gamma = segment.a, segment.b, segment.delta, segment.gamma
#
#         if self._steady_state:
#             self._xi2 += _covariance_matrix_closed_form(A, C, gamma, a, b, delta).reshape(self._N ** 2)
#
#         if self._backend == 'numpy':
#             v = np.ones(self._K) if v == 1 else v
#             if self._calc_W and not self._steady_state:
#                 backward_recursion_W_ss(self.W, segment.a, segment.b, segment.delta, segment.gamma, A, C, beta, y, v)
#             if self._calc_xi:
#                 backward_recursion_xi_ss(self._xi1, segment.a, segment.b, segment.delta, segment.gamma, A, C, beta, y,
#                                          v, self._get_einsum_path_xi_ss())
#             if self._calc_kappa:
#                 backward_recursion_kappa_ss(self._xi0, segment.a, segment.b, segment.delta, segment.gamma, beta, y, v,
#                                             self._get_einsum_path_kappa_ss())
#             if self._calc_nu:
#                 backward_recursion_nu_ss(self._nu, segment.a, segment.b, segment.delta, segment.gamma, beta, v)
#
#         if self._backend == 'lfilter' and self._filter_form == 'parallel':
#             raise NotImplemented('Parallel Block Form not implemented yet')
#
#         if self._backend == 'lfilter' and self._filter_form in ('cascade', 'auto'):
#             if self._calc_W and not self._steady_state:
#                 backward_cascade_xi(self._xi2, a, b, delta, gamma, np.kron(A, A), np.kron(C, C), beta, 1, v,
#                                     einsum_path=self._get_einsum_path_W_tf())
#             if self._calc_xi:
#                 backward_cascade_xi(self._xi1, a, b, delta, gamma, A, C, beta, y, v, self._get_einsum_path_xi_tf())
#             if self._calc_kappa:
#                 _C = np.array(1)
#                 _A = np.array(1)
#                 _y = np.einsum(self._get_einsum_path_y_squared_tf(), y, y)
#                 backward_cascade_xi(self._xi0, a, b, delta, gamma, _A, _C, beta, _y, v,
#                                     self._get_einsum_path_kappa_tf())
#             if self._calc_nu:
#                 _C = np.array(1)
#                 _A = np.array(1)
#                 backward_cascade_xi(self._nu, a, b, delta, gamma, _A, _C, beta, 1, v, self._get_einsum_path_nu_tf())
#
#         if self._backend == 'jit':
#             v = np.ones(self._K) if v == 1 else v
#             init_vars = backward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)
#             if self._is_multiset:
#                 if self.steady_state:
#                     raise NotImplemented('backward_recursion_set_xi_kappa_nu_jit not implemented yet')
#                 else:
#                     backward_recursion_set_jit(self.W, self._xi1, self._xi0, self._nu, segment.a, segment.b,
#                                                segment.delta, y, v, beta, *init_vars, self._kappa_diag)
#             else:
#                 if self.steady_state:
#                     backward_recursion_xi_kappa_nu_jit(self._xi1, self._xi0, self._nu, segment.a, segment.b,
#                                                        segment.delta, y, v, beta, *init_vars)
#                 else:
#                     backward_recursion_jit(self.W, self._xi1, self._xi0, self._nu, segment.a, segment.b, segment.delta,
#                                            y, v, beta, *init_vars)
#
#     def set_backend(self, backend):
#         """
#         Setting the backend computations option
#
#         Parameters
#         ----------
#         backend : str
#             'numpy' for State-Space python backend
#             'lfilter' for Transfer Function python backend
#             'jit' for Just in Time backend
#
#         """
#         assert backend in BACKEND_TYPES, f'Wrong backend name {backend}'
#         assert backend in available_backends, f'{backend} not available, check {backend} installation.'
#         self._backend = backend
#
#     def filter(self, y, v=None):
#         """
#         Computes the intermediate parameters for subsequent squared error computations and minimization's.
#
#         Computes the intermediate parameters using efficient forward- and backward recursions.
#         The results are stored internally, ready to solve the least squares problem using e.g., :meth:`minimize_x`
#         or :meth:`minimize_v`. The parameter allocation :meth:`allocate` is called internally,
#         so a manual pre-allocation is not necessary.
#
#         Parameters
#         ----------
#         y : array_like
#             Input signal |br|
#             - Single-channel signal is of `shape =(K,)` for |br|
#             - Multi-channel signal is of `shape =(K,L)` |br|
#             - Single-channel set signals is of `shape =(K,S)` for |br|
#             - Multi-channel set signals is of `shape =(K,L,S)` |br|
#             - Multi-channel-sets signal is of `shape =(K,L,S)`
#         v : array_like, shape=(K,), optional
#             Sample weights. Weights the parameters for a time step `k` and is the same for all multi-channels.
#             By default, the sample weights are initialized to 1.
#
#
#         |def_K|
#         |def_L|
#         |def_S|
#
#         """
#
#         self._check_output_dimensions(y)
#         self._allocate_parameter_storage(np.shape(y))
#
#         segments = self.cost_model.segments
#         alssms = self.cost_model.alssms
#
#         A = AlssmSum(alssms).A
#
#         if v is None:
#             v = 1
#
#         betas = np.ones(len(segments)) if self.betas is None else self.betas
#
#         for i, (segment, beta) in enumerate(zip(segments, betas)):
#
#             # calculate output matrix C for the segment
#             tmp_c = []
#             for j, alssm in enumerate(alssms):
#                 tmp_c.append(self.cost_model.F[j, i] * alssm.C)
#             C = np.hstack(tmp_c)
#
#             if segment.direction == FW:
#                 self._forward_recursion(A, C, segment, y, v, beta)
#             elif segment.direction == BW:
#                 self._backward_recursion(A, C, segment, y, v, beta)
#             else:
#                 ValueError('segment.direction has wrong value.')
#
#     def minimize_v(self, H=None, h=None, return_constrains=False):
#         r"""
#         Returns the vector `v` of the squared error minimization with linear constraints
#
#         Minimizes the squared error over the vector `v` with linear constraints with an (optional) offset
#         [Wildhaber2018]_ [TABLE V].
#
#         **Constraint:**
#
#         - *Linear Scalar* : :math:`x=Hv,\,v\in\mathbb{R}`
#
#           known : :math:`H \in \mathbb{R}^{N \times 1}`
#
#           :math:`\hat{v}_k = \frac{\xi_k^{\mathsf{T}}H}{H^{\mathsf{T}}W_k H}`
#
#         - *Linear Combination With Offset* : :math:`x=Hv +h,\,v\in\mathbb{M}`
#
#           known : :math:`H \in \mathbb{R}^{N \times M},\,h\in\mathbb{R}^N`
#
#           :math:`\hat{v}_k = \big(H^{\mathsf{T}}W_k H\big)^{-1} H^\mathsf{T}\big(\xi_k - W_k h\big)`
#
#
#         Parameters
#         ----------
#         H : array_like, shape=(N, M)
#             Matrix for linear constraining :math:`H`
#         h : array_like, shape=(N, [S]), optional
#             Offset vector for linear constraining :math:`h`
#         return_constrains : bool
#             If set to True, the output is extended by H and h
#
#         Returns
#         -------
#         v : :class:`~numpy.ndarray`, shape = (K, M)
#             Least square state vector estimate for each time index.
#             The shape of one state vector `x[k]` is `(N, [S])`, where k is the time index of `K` samples,
#             `N` the ALSSM order.
#
#         |def_K|
#         |def_N|
#
#         """
#
#         # check and init H
#         if H is None:
#             H = np.eye(self._N)
#             HTWH = self.W
#         else:
#             H = np.asarray(H)
#             if H.shape[0] != self._N:
#                 ValueError(f"First dimension of constrain matrix H needs to be of size {self._N} (model order), "
#                            f"{info_str_found_shape(H)}.")
#             HTWH = H.T @ self.W @ H
#
#         # check and init h
#         if h is None:
#             h = np.zeros(self._N)
#             HTxiWh = np.einsum('nm, km...-> kn...', H.T, self._xi1)
#         else:
#             if h.shape[0] != self._N:
#                 ValueError(f"First dimension of offset vector h needs to be of size {self._N} (model order), "
#                            f"{info_str_found_shape(h)}.")
#             HTxiWh = np.einsum('nm, km...-> kn...', H.T, self._xi1 - self.W @ h)
#
#         # constrained minimization
#         M = H.shape[1]
#         v = np.full((self._K, M, self._S) if self._is_multiset else (self._K, M), np.nan)
#         msk = cond(HTWH) < 1 / sys.float_info.epsilon
#         if self._steady_state:
#             assert msk, 'H.T @ W @ H is not invertible.'
#             v[...] = np.einsum('nm, km...-> kn...', inv(HTWH), HTxiWh)
#         else:
#             v[msk] = np.einsum('knm, kn... -> km...', inv(HTWH[msk]), HTxiWh[msk])
#
#         if return_constrains:
#             return v, H, h
#         return v
#
#     def minimize_x(self, H=None, h=None):
#         r"""
#         Returns the state vector `x` of the squared error minimization with linear constraints
#
#         Minimizes the squared error over the state vector `x`.
#         If needed its possible to apply linear constraints with an (optional) offset.
#         [Wildhaber2018]_ [TABLE V].
#
#         **Constraint:**
#
#         - *Linear Scalar* : :math:`x=Hv,\,v\in\mathbb{R}`
#
#           known : :math:`H \in \mathbb{R}^{N \times 1}`
#
#         - *Linear Combination With Offset* : :math:`x=Hv +h,\,v\in\mathbb{M}`
#
#           known : :math:`H \in \mathbb{R}^{N \times M},\,h\in\mathbb{R}^N`
#
#         See also :meth:`minimize_v`
#
#         Parameters
#         ----------
#         H : array_like, shape=(N, M), optional
#             Matrix for linear constraining :math:`H`
#         h : array_like, shape=(N, [S]), optional
#             Offset vector for linear constraining :math:`h`
#
#         Returns
#         -------
#         xs : :class:`~numpy.ndarray` of shape = (K, N)
#             Least square state vector estimate for each time index.
#             The shape of one state vector `x[k]` is `(N,)`, where `k` is the time index of `K` samples,
#             `N` the ALSSM order.
#
#
#         |def_K|
#         |def_N|
#
#         """
#
#         if H is None and h is None:
#             msk = cond(self.W) < 1 / sys.float_info.epsilon
#             x = np.full_like(self.xi, np.nan)
#             if self._steady_state:
#                 assert msk, 'Steady State W Matrix is not invertible.'
#                 x[...] = np.einsum('nm, km...-> kn...', inv(self.W), self.xi)
#             else:
#                 assert np.any(msk), 'All W Matrices are not invertible.'
#                 x[msk] = np.einsum('knm, kn... -> km...', inv(self.W[msk]), self.xi[msk])
#             return x
#
#         v, H, h = self.minimize_v(H, h, return_constrains=True)
#         x = np.einsum('nm, km...->kn...', H, v) + h
#         return x
#
#     def eval_errors(self, xs, ks=None):
#         r"""
#         Evaluation of the squared error for multiple state vectors `xs`.
#
#         The return value is the squared error
#
#         .. math::
#             J(x)  = x^{\mathsf{T}}W_kx -2*x^{\mathsf{T}}\xi_k + \kappa_k
#
#         for each state vector :math:`x` from the list `xs`.
#
#
#         Parameters
#         ----------
#         xs : array_like of shape=(K, N)
#             List of state vectors :math:`x`
#         ks : None, array_like of int of shape=(XS,)
#             List of indices where to evaluate the error
#
#         Returns
#         -------
#         J : :class:`np.ndarray` of shape=(XS,)
#             Squared Error for each state vector
#
#
#         |def_K|
#         |def_XS|
#         |def_N|
#
#         """
#
#         if self._steady_state:
#             J = np.einsum('kn..., kn...->k...', xs, np.einsum('nm, km...->kn...', self.W, xs))
#
#         if ks is None:
#             if not self._steady_state:
#                 J = np.einsum('kn..., kn...->k...', xs, np.einsum('knm, km...->kn...', self.W, xs))
#             return J - 2 * np.einsum('kn..., kn...->k...', self.xi, xs) + self.kappa
#
#         else:
#             if not self._steady_state:
#                 J = np.einsum('kn..., kn...->k...', xs[ks], np.einsum('knm, km...->kn...', self.W[ks], xs[ks]))
#             return J - 2 * np.einsum('kn..., kn...->k...', self.xi[ks], xs[ks]) + self.kappa[ks]
#
#     def filter_minimize_x(self, y, v=None, H=None, h=None):
#         """
#         Combination of :meth:`RLSAlssmBase.filter` and :meth:`RLSAlssmBase.minimize_x`.
#
#         This method has the same output as calling the methods
#
#         .. code::
#
#             rls.filter(y)
#             xs = rls.minimize_x()
#
#
#         See Also
#         --------
#         :meth:`RLSAlssmBase.filter`, :meth:`RLSAlssmBase.minimize_x`
#
#         """
#
#         self.filter(y, v)
#         return self.minimize_x(H, h)
#
#     def filter_minimize_v(self, y, v=None, H=None, h=None, **kwargs):
#         """
#         Combination of :meth:`RLSAlssmBase.filter` and :meth:`RLSAlssmBase.minimize_v`.
#
#         This method has the same output as calling the methods
#
#         .. code::
#
#             rls.filter(y)
#             xs = rls.minimize_v()
#
#
#         See Also
#         --------
#         :meth:`RLSAlssmBase.filter`, :meth:`RLSAlssmBase.minimize_v`
#
#         """
#
#         self.filter(y, v)
#         return self.minimize_v(H, h, **kwargs)
#
#     def filter_minimize_yhat(self, y, v=None, H=None, h=None, alssm_weights=None, c0s=None):
#         """
#         Combination of :meth:`RLSAlssmBase.filter` and :meth:`RLSAlssmBase.minimize_x` and
#         :meth:`CostBase.eval_alssm_output`
#
#         This method has the same output as calling the methods
#
#         .. code::
#
#             xs = rls.filter_minimize_x()
#             y_hat = rls.cost_model.eval_alssm_output(xs)
#
#         See Also
#         --------
#         :meth:`RLSAlssmBase.filter`, :meth:`RLSAlssmBase.minimize_x`
#
#         """
#         xs = self.filter_minimize_x(y, v=v, H=H, h=h)
#         return self.cost_model.eval_alssm_output(xs, alssm_weights=alssm_weights, c0s=c0s)
#
#
# class RLSAlssmSet(RLSAlssm):
#     """
#     Filter and Data container for Recursive Least Square Alssm Filters using Sets (multichannel parallel processing)
#
#     This class is the same as :class:`RLSAlssm` except that the signal `y` has an additional last dimension.
#     The signals in these dimensions are processed simultaneously, as in a normal :class:`RLSAlssm` called multiple times.
#
#     Parameters
#     ----------
#     cost_model : CostSegment, CompositeCost
#         Cost Model
#     kappa_diag : bool
#         If set to False, kappa will be computed as a matrix (outer product of each signal energy) else
#         its diagonal will be saved
#     **kwargs
#         Forwarded to :class:`.RLSAlssmBase`
#
#     """
#
#     def __init__(self, cost_model, kappa_diag=True, **kwargs):
#         warnings.warn('RLSAlssmSet is deprecated. Use RLSAlssm instead.', DeprecationWarning, 2)
#         super().__init__(cost_model, kappa_diag=kappa_diag, **kwargs)
#
#
# class RLSAlssmSteadyState(RLSAlssm):
#     """
#     Filter and Data container for Recursive Least Square Alssm Filters in Steady State Mode
#
#     With :class:`RLSAlssmSteadyState` a common :math:`W_k = W_{steady}` is used for all samples (faster computation).
#     Note that using a common :math:`W_k` potentially leads to border missmatch effects and to completely invalid results
#     when samples have individual sample weights.
#
#     See Also
#     --------
#     :class:`RLSAlssm`
#
#     """
#
#     def __init__(self, cost_model, steady_state_method='closed_form', **kwargs):
#         warnings.warn('RLSAlssmSteadyState is deprecated. Use RLSAlssm(..., steady_state=True) instead.',
#                       DeprecationWarning, 2)
#         super().__init__(cost_model, steady_state=True, **kwargs)
#
#
# class RLSAlssmSetSteadyState(RLSAlssm):
#     """
#     Filter and Data container for Recursive Least Square Alssm Filters using Sets in Steady State Mode
#
#     With :class:`RLSAlssmSteadyState` a common :math:`W_k = W_{steady}` is used for all samples (faster computation).
#     Note that using a common :math:`W_k` potentially leads to border missmatch effects and to completely invalid results
#     when samples have individual sample weights.
#
#     See Also
#     --------
#     :class:`RLSAlssmSet`
#
#     """
#
#     def __init__(self, cost_model, steady_state_method='closed_form', kappa_diag=True, **kwargs):
#         warnings.warn('RLSAlssmSetSteadyState is deprecated. Use RLSAlssm(..., steady_state=True) instead.',
#                       DeprecationWarning, 2)
#         super().__init__(cost_model, kappa_diag=kappa_diag, steady_state=True, **kwargs)


# class RLSAlssmBase(ABC):
#     """
#     Base Class for Recursive Least Square Alssm Classes
#
#
#     Parameters
#     ----------
#     betas : array_like of shape=(P,) of floats, None, optional
#         Segment Scalars. Factors weighting each of the `P` cost segments.
#         If `betas` is not set, the weight is for each cost segment 1.
#     """
#
#     def __init__(self, betas=None):
#         self._cost_model = None
#         self._W = None
#         self._xi = None
#         self._kappa = None
#         self._nu = None
#         self.betas = betas
#         self._backend = get_backend()
#         self._kappa_diag = True
#
#     @property
#     def cost_model(self):
#         """CostBase : Cost Model"""
#         return self._cost_model
#
#     @cost_model.setter
#     def cost_model(self, cost_model):
#         assert isinstance(cost_model, CostBase), 'cost_model is not a subclass of CostBase'
#         self._cost_model = cost_model
#         self._cost_model.__class__ = CostBase
#
#     @property
#     def betas(self):
#         """~numpy.ndarray : Segment scalars weights the cost function per segment"""
#         return self._betas
#
#     @betas.setter
#     def betas(self, betas):
#         if betas is None:
#             self._betas = None
#         else:
#             assert is_array_like(betas), 'betas if not array_like'
#             assert len(betas) == len(
#                 self._cost_model.segments), f'betas has wrong length, {info_str_found_shape(betas)}'
#             self._betas = betas
#
#     @property
#     def W(self):
#         """:class:`~numpy.ndarray` : Filter Parameter :math:`W`"""
#         return self._W
#
#     @property
#     def xi(self):
#         """:class:`~numpy.ndarray` :  Filter Parameter :math:`\\xi`"""
#         return self._xi
#
#     @property
#     def kappa(self):
#         """:class:`~numpy.ndarray`  : Filter Parameter :math:`\\kappa`"""
#         return self._kappa
#
#     @property
#     def nu(self):
#         """:class:`~numpy.ndarray`  : Filter Parameter :math:`\\nu`"""
#         return self._nu
#
#     def _check_output_dimensions(self, y):
#         C = AlssmSum(self.cost_model.alssms).C
#
#         is_multichannel = np.ndim(C) == 2
#         is_multiset = isinstance(self, (RLSAlssmSet, RLSAlssmSetSteadyState))
#
#         if is_multichannel:
#             if is_multiset:
#                 assert np.ndim(y) == 3 and np.shape(C)[0] == np.shape(y)[1], \
#                     'Model output and observation shape does not match. ' \
#                     'Multi-channel and -set systems expect shapes: ' \
#                     'C shape (L, N) and y shape (K, L, S)'
#             else:
#                 assert np.ndim(y) == 2 and np.shape(C)[0] == np.shape(y)[1], \
#                     'Model output and observation shape does not match. ' \
#                     'Multi-channel system expect shapes: ' \
#                     'C shape (L, N) and y shape (K, L)'
#         else:
#             if is_multiset:
#                 assert np.ndim(y) == 2, \
#                     'Model output and observation shape does not match. ' \
#                     'Multi-set systems expect shapes: ' \
#                     'C shape (N,) and y shape (K, S)'
#             else:
#                 assert np.ndim(y) == 1, \
#                     'Model output and observation shape does not match. ' \
#                     'Scalar systems (non multi-channel/set) expect shapes:' \
#                     'C shape (N,) and y shape (K,)'
#
#     @abstractmethod
#     def _allocate_parameter_storage(self, input_shape):
#         pass
#
#     def set_backend(self, backend):
#         """
#         Setting the backend computations option
#
#         Parameters
#         ----------
#         backend : str
#             'numpy' for State-Space python backend
#             'lfilter' for Transfer Function python backend
#             'jit' for Just in Time backend
#
#         """
#         assert backend in BACKEND_TYPES, f'Wrong backend name {backend}'
#         assert backend in available_backends, f'{backend} not available, check {backend} installation.'
#         self._backend = backend
#
#     @abstractmethod
#     def _forward_recursion(self, segment, A, C, y, v, beta):
#         pass
#
#     @abstractmethod
#     def _backward_recursion(self, segment, A, C, y, v, beta):
#         pass
#
#     def filter(self, y, v=None):
#         """
#         Computes the intermediate parameters for subsequent squared error computations and minimization's.
#
#         Computes the intermediate parameters using efficient forward- and backward recursions.
#         The results are stored internally, ready to solve the least squares problem using e.g., :meth:`minimize_x`
#         or :meth:`minimize_v`. The parameter allocation :meth:`allocate` is called internally,
#         so a manual pre-allocation is not necessary.
#
#         Parameters
#         ----------
#         y : array_like
#             Input signal |br|
#             :class:`RLSAlssm` or :class:`RLSAlssmSteadyState`
#             - Single-channel signal is of `shape =(K,)` for |br|
#             - Multi-channel signal is of `shape =(K,L)` |br|
#             :class:`RLSAlssmSet` or :class:`RLSAlssmSetSteadyState`
#             - Single-channel set signals is of `shape =(K,S)` for |br|
#             - Multi-channel set signals is of `shape =(K,L,S)` |br|
#             - Multi-channel-sets signal is of `shape =(K,L,S)`
#         v : array_like, shape=(K,), optional
#             Sample weights. Weights the parameters for a time step `k` and is the same for all multi-channels.
#             By default, the sample weights are initialized to 1.
#
#
#         |def_K|
#         |def_L|
#         |def_S|
#
#         """
#
#         self._check_output_dimensions(y)
#         self._allocate_parameter_storage(np.shape(y))
#
#         segments = self.cost_model.segments
#         alssms = self.cost_model.alssms
#
#         A = AlssmSum(alssms).A
#
#         if v is None:
#             v = np.ones(np.shape(y)[0])
#
#         betas = np.ones(len(segments)) if self.betas is None else self.betas
#
#         for i, (segment, beta) in enumerate(zip(segments, betas)):
#
#             # calculate output matrix C for the segment
#             tmp_c = []
#             for j, alssm in enumerate(alssms):
#                 tmp_c.append(self.cost_model.F[j, i] * alssm.C)
#             C = np.hstack(tmp_c)
#
#             if segment.direction == FW:
#                 self._forward_recursion(A, C, segment, y, v, beta)
#             elif segment.direction == BW:
#                 self._backward_recursion(A, C, segment, y, v, beta)
#             else:
#                 ValueError('segment.direction has wrong value.')
#
#     @abstractmethod
#     def minimize_x(self, *args, **kwargs):
#         pass
#
#     @abstractmethod
#     def minimize_v(self, *args, **kwargs):
#         pass
#
#     def filter_minimize_x(self, y, v=None, H=None, h=None):
#         """
#         Combination of :meth:`RLSAlssmBase.filter` and :meth:`RLSAlssmBase.minimize_x`.
#
#         This method has the same output as calling the methods
#
#         .. code::
#
#             rls.filter(y)
#             xs = rls.minimize_x()
#
#
#         See Also
#         --------
#         :meth:`RLSAlssmBase.filter`, :meth:`RLSAlssmBase.minimize_x`
#
#         """
#
#         self.filter(y, v)
#         return self.minimize_x(H, h)
#
#     def filter_minimize_v(self, y, v=None, H=None, h=None, **kwargs):
#         """
#         Combination of :meth:`RLSAlssmBase.filter` and :meth:`RLSAlssmBase.minimize_v`.
#
#         This method has the same output as calling the methods
#
#         .. code::
#
#             rls.filter(y)
#             xs = rls.minimize_v()
#
#
#         See Also
#         --------
#         :meth:`RLSAlssmBase.filter`, :meth:`RLSAlssmBase.minimize_v`
#
#         """
#
#         self.filter(y, v)
#         return self.minimize_v(H, h, **kwargs)
#
#     def filter_minimize_yhat(self, y, v=None, H=None, h=None, alssm_weights=None, c0s=None):
#         """
#         Combination of :meth:`RLSAlssmBase.filter` and :meth:`RLSAlssmBase.minimize_x` and
#         :meth:`CostBase.eval_alssm_output`
#
#         This method has the same output as calling the methods
#
#         .. code::
#
#             xs = rls.filter_minimize_x()
#             y_hat = rls.cost_model.eval_alssm_output(xs)
#
#         See Also
#         --------
#         :meth:`RLSAlssmBase.filter`, :meth:`RLSAlssmBase.minimize_x`
#
#         """
#         xs = self.filter_minimize_x(y, v=v, H=H, h=h)
#         return self.cost_model.eval_alssm_output(xs, alssm_weights=alssm_weights, c0s=c0s)
#
#
# class RLSAlssm(RLSAlssmBase):
#     """
#     Filter and Data container for Recursive Least Square Alssm Filters
#
#     :class:`RLSAlssm` computes and stores intermediate values such as covariances,
#     as required to efficiently solve recursive least squares problems
#     between a model-based cost function :class:`CompositeCost` or :class:`CostSegment` and given observations.
#     The intermediate variables are observation dependant and therefore the memory consumption of :class:`RLSAlssm`
#     scales linearly with the observation vector length.
#
#     Main intermediate variables are the covariance `W`, weighted mean `\\xi`, signal energy `\\kappa`, weighted number
#     of samples `\\nu`, see Equation (4.6) in [Wildhaber2019]_
#     :download:`PDF <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=49>`
#
#
#     Parameters
#     ----------
#     cost_model : CostSegment, CompositeCost, CostBase
#         Cost Model
#     **kwargs
#             Forwarded to :class:`.RLSAlssmBase`
#
#     """
#
#     def __init__(self, cost_model, **kwargs):
#         super().__init__(**kwargs)
#         self.cost_model = cost_model
#
#     def _allocate_parameter_storage(self, input_shape):
#         K = input_shape[0]
#
#         N = self.cost_model.get_model_order()
#         self._W = np.zeros((K, N, N))
#         self._xi = np.zeros((K, N))
#         self._kappa = np.zeros(K)
#         self._nu = np.zeros(K)
#
#     def _forward_recursion(self, A, C, segment, y, v, beta):
#         init_vars = forward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)
#
#         if self._backend == 'numpy':
#             forward_recursion_W_ss(self._W, segment.a, segment.b, segment.delta, segment.gamma, A, C, beta, y, v)
#             einsum_path = 'n..., ... ->n...' if np.ndim(C) == 1 else 'nl..., l... ->n...'
#             forward_recursion_xi_ss(self._xi, segment.a, segment.b, segment.delta, segment.gamma, A, C, beta, y, v, einsum_path)
#             if np.ndim(C) == 2:
#                 if self._kappa_diag:
#                     einsum_path = 'm..., m...->...'
#                 else:
#                     einsum_path = 'ml..., ln... ->mn'
#             else:
#                 if self._kappa_diag:
#                     einsum_path = 'm, m->m'
#                 else:
#                     einsum_path = 'm, n->mn'
#             if np.ndim(y) == 1:
#                 einsum_path = '..., ...'
#
#             forward_recursion_kappa_ss(self._kappa, segment.a, segment.b, segment.delta, segment.gamma, beta, y, v, einsum_path)
#             forward_recursion_nu_ss(self._nu, segment.a, segment.b, segment.delta, segment.gamma, beta, v)
#
#             #self._allocate_parameter_storage(np.shape(y))
#             #forward_recursion_py_ss(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v, beta, *init_vars)
#
#         if self._backend == 'lfilter':
#             forward_recursion_py_tf(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
#                                  beta, *init_vars)
#         if self._backend == 'jit':
#             forward_recursion_jit(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
#                                   beta, *init_vars)
#
#     def _backward_recursion(self, A, C, segment, y, v, beta):
#         init_vars = backward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)
#
#         if self._backend == 'numpy':
#             if True:
#                 backward_recursion_W_ss(self._W, segment.a, segment.b, segment.delta, segment.gamma, A, C, beta, y, v)
#                 einsum_path = 'n..., ... ->n...' if np.ndim(C) == 1 else 'nl..., l... ->n...'
#                 backward_recursion_xi_ss(self._xi, segment.a, segment.b, segment.delta, segment.gamma, A, C, beta, y, v, einsum_path)
#                 if np.ndim(C) == 2:
#                     if self._kappa_diag:
#                         einsum_path = 'm..., m...->...'
#                     else:
#                         einsum_path = 'ml..., ln... ->mn'
#                 else:
#                     if self._kappa_diag:
#                         einsum_path = 'm, m->m'
#                     else:
#                         einsum_path = 'm, n->mn'
#                 if np.ndim(y) == 1:
#                     einsum_path = '..., ...'
#
#                 backward_recursion_kappa_ss(self._kappa, segment.a, segment.b, segment.delta, segment.gamma, beta, y, v, einsum_path)
#                 backward_recursion_nu_ss(self._nu, segment.a, segment.b, segment.delta, segment.gamma, beta, v)
#             else:
#                 backward_recursion_py_ss(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v, beta, *init_vars)
#         if self._backend == 'lfilter':
#             backward_recursion_py_tf(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
#                                   beta, *init_vars)
#         if self._backend == 'jit':
#             backward_recursion_jit(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
#                                    beta, *init_vars)
#
#     def minimize_v(self, H=None, h=None, return_constrains=False):
#         r"""
#         Returns the vector `v` of the squared error minimization with linear constraints
#
#         Minimizes the squared error over the vector `v` with linear constraints with an (optional) offset
#         [Wildhaber2018]_ [TABLE V].
#
#         **Constraint:**
#
#         - *Linear Scalar* : :math:`x=Hv,\,v\in\mathbb{R}`
#
#           known : :math:`H \in \mathbb{R}^{N \times 1}`
#
#           :math:`\hat{v}_k = \frac{\xi_k^{\mathsf{T}}H}{H^{\mathsf{T}}W_k H}`
#
#         - *Linear Combination With Offset* : :math:`x=Hv +h,\,v\in\mathbb{M}`
#
#           known : :math:`H \in \mathbb{R}^{N \times M},\,h\in\mathbb{R}^N`
#
#           :math:`\hat{v}_k = \big(H^{\mathsf{T}}W_k H\big)^{-1} H^\mathsf{T}\big(\xi_k - W_k h\big)`
#
#
#         Parameters
#         ----------
#         H : array_like, shape=(N, M)
#             Matrix for linear constraining :math:`H`
#         h : array_like, shape=(N, [S]), optional
#             Offset vector for linear constraining :math:`h`
#         return_constrains : bool
#             If set to True, the output is extened by H and h
#
#         Returns
#         -------
#         v : :class:`~numpy.ndarray`, shape = (K, M)
#             Least square state vector estimate for each time index.
#             The shape of one state vector `x[k]` is `(N, [S])`, where k is the time index of `K` samples,
#             `N` the ALSSM order.
#
#         |def_K|
#         |def_N|
#
#         """
#
#         N = self.cost_model.get_model_order()
#
#         # check and init H
#         H = np.eye(N) if H is None else np.asarray(H)
#         assert H.shape[
#                    0] == N, f"first dimension of constrain matrix H needs to be of size {N} (model order), found size {H.shape[0]}"
#         M = H.shape[1]
#
#         # check and init h
#         h = np.zeros(N) if h is None else np.asarray(h)
#         assert h.shape == (N,), f"offset vector h needs to be of shape ({N},) (model order), {info_str_found_shape(h)}"
#
#         # allocate v and minimize
#         v = np.full((len(self.W), M), np.nan)
#         minimize_v_py_ss(v, self.W, self.xi, H, h)
#
#         if return_constrains:
#             return v, H, h
#         return v
#
#     def minimize_x(self, H=None, h=None):
#         r"""
#         Returns the state vector `x` of the squared error minimization with linear constraints
#
#         Minimizes the squared error over the state vector `x`.
#         If needed its possible to apply linear constraints with an (optional) offset.
#         [Wildhaber2018]_ [TABLE V].
#
#         **Constraint:**
#
#         - *Linear Scalar* : :math:`x=Hv,\,v\in\mathbb{R}`
#
#           known : :math:`H \in \mathbb{R}^{N \times 1}`
#
#         - *Linear Combination With Offset* : :math:`x=Hv +h,\,v\in\mathbb{M}`
#
#           known : :math:`H \in \mathbb{R}^{N \times M},\,h\in\mathbb{R}^N`
#
#         See also :meth:`minimize_v`
#
#         Parameters
#         ----------
#         H : array_like, shape=(N, M), optional
#             Matrix for linear constraining :math:`H`
#         h : array_like, shape=(N, [S]), optional
#             Offset vector for linear constraining :math:`h`
#
#         Returns
#         -------
#         xs : :class:`~numpy.ndarray` of shape = (K, N)
#             Least square state vector estimate for each time index.
#             The shape of one state vector `x[k]` is `(N,)`, where `k` is the time index of `K` samples,
#             `N` the ALSSM order.
#
#
#         |def_K|
#         |def_N|
#
#         """
#
#         if H is None and h is None:
#             # allocate x and minimize
#             x = np.full_like(self.xi, np.nan)
#             minimize_x_py_ss(x, self.W, self.xi)
#         else:
#             v, H, h = self.minimize_v(H, h, return_constrains=True)
#             x = np.einsum('nm, km->kn', H, v) + h
#
#         return x
#
#     def eval_errors(self, xs, ks=None):
#         r"""
#         Evaluation of the squared error for multiple state vectors `xs`.
#
#         The return value is the squared error
#
#         .. math::
#             J(x)  = x^{\mathsf{T}}W_kx -2*x^{\mathsf{T}}\xi_k + \kappa_k
#
#         for each state vector :math:`x` from the list `xs`.
#
#
#         Parameters
#         ----------
#         xs : array_like of shape=(K, N)
#             List of state vectors :math:`x`
#         ks : None, array_like of int of shape=(XS,)
#             List of indices where to evaluate the error
#
#         Returns
#         -------
#         J : :class:`np.ndarray` of shape=(XS,)
#             Squared Error for each state vector
#
#
#         |def_K|
#         |def_XS|
#         |def_N|
#
#         """
#         if ks is None:
#             return (np.einsum('kn, kn->k', xs, np.einsum('knm, km->kn', self.W, xs))
#                     - 2 * np.einsum('kn, kn->k', self.xi, xs)
#                     + self.kappa)
#         else:
#             return (np.einsum('kn, kn->k', xs[ks], np.einsum('knm, km->kn', self.W[ks], xs[ks]))
#                     - 2 * np.einsum('kn, kn->k', self.xi[ks], xs[ks])
#                     + self.kappa[ks])
#
#
# class RLSAlssmSet(RLSAlssmBase):
#     """
#     Filter and Data container for Recursive Least Square Alssm Filters using Sets (multichannel parallel processing)
#
#     This class is the same as :class:`RLSAlssm` except that the signal `y` has an additional last dimension.
#     The signals in these dimensions are processed simultaneously, as in a normal :class:`RLSAlssm` called multiple times.
#
#     Parameters
#     ----------
#     cost_model : CostSegment, CompositeCost
#         Cost Model
#     kappa_diag : bool
#         If set to False, kappa will be computed as a matrix (outer product of each signal energy) else
#         its diagonal will be saved
#     **kwargs
#         Forwarded to :class:`.RLSAlssmBase`
#
#     """
#
#     def __init__(self, cost_model, kappa_diag=True, **kwargs):
#         super().__init__(**kwargs)
#         self._kappa_diag = None
#         self.cost_model = cost_model
#         self.set_kappa_diag(kappa_diag)
#
#     def set_kappa_diag(self, b):
#         assert isinstance(b, bool), 'kappa_diag is not of type bool'
#         self._kappa_diag = b
#
#     def _allocate_parameter_storage(self, input_shape):
#
#         K = input_shape[0]
#         S = input_shape[-1]
#         N = self.cost_model.get_model_order()
#         self._W = np.zeros((K, N, N))
#         self._nu = np.zeros(K)
#         self._xi = np.zeros((K, N, S))
#         if self._kappa_diag:
#             self._kappa = np.zeros((K, S))
#         else:
#             self._kappa = np.zeros((K, S, S))
#
#     def _forward_recursion(self, A, C, segment, y, v, beta):
#         init_vars = forward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)
#
#         if self._backend == 'numpy':
#             forward_recursion_W_ss(self._W, segment.a, segment.b, segment.delta, segment.gamma, A, C, beta, y, v)
#
#             einsum_path = 'n..., ... ->n...' if np.ndim(C) == 1 else 'nl..., l... ->n...'
#             forward_recursion_xi_ss(self._xi, segment.a, segment.b, segment.delta, segment.gamma, A, C, beta, y, v, einsum_path)
#
#             if np.ndim(C) == 2:
#                 if self._kappa_diag:
#                     einsum_path = 'm..., m...->...'
#                 else:
#                     einsum_path = 'ml..., ln... ->mn'
#             else:
#                 if self._kappa_diag:
#                     einsum_path = 'm, m->m'
#                 else:
#                     einsum_path = 'm, n->mn'
#             if np.ndim(y) == 1:
#                 einsum_path = '..., ...'
#
#             forward_recursion_kappa_ss(self._kappa, segment.a, segment.b, segment.delta, segment.gamma, beta, y, v, einsum_path)
#             forward_recursion_nu_ss(self._nu, segment.a, segment.b, segment.delta, segment.gamma, beta, v)
#         if self._backend == 'lfilter':
#             forward_recursion_set_py_tf(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y,
#                                      v,
#                                      beta, *init_vars, self._kappa_diag)
#         if self._backend == 'jit':
#             forward_recursion_set_jit(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y,
#                                       v,
#                                       beta, *init_vars, self._kappa_diag)
#
#     def _backward_recursion(self, A, C, segment, y, v, beta):
#         init_vars = backward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)
#
#         if self._backend == 'numpy':
#             if True:
#                 backward_recursion_W_ss(self._W, segment.a, segment.b, segment.delta, segment.gamma, A, C, beta, y, v)
#                 einsum_path = 'n..., ... ->n...' if np.ndim(C) == 1 else 'nl..., l... ->n...'
#                 backward_recursion_xi_ss(self._xi, segment.a, segment.b, segment.delta, segment.gamma, A, C, beta, y, v, einsum_path)
#                 if np.ndim(C) == 2:
#                     if self._kappa_diag:
#                         einsum_path = 'm..., m...->...'
#                     else:
#                         einsum_path = 'ml..., ln... ->mn'
#                 else:
#                     if self._kappa_diag:
#                         einsum_path = 'm, m->m'
#                     else:
#                         einsum_path = 'm, n->mn'
#                 if np.ndim(y) == 1:
#                     einsum_path = '..., ...'
#
#                 backward_recursion_kappa_ss(self._kappa, segment.a, segment.b, segment.delta, segment.gamma, beta, y, v, einsum_path)
#                 backward_recursion_nu_ss(self._nu, segment.a, segment.b, segment.delta, segment.gamma, beta, v)
#             else:
#                 backward_recursion_set_py_ss(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v, beta, *init_vars, self._kappa_diag)
#         if self._backend == 'lfilter':
#             backward_recursion_set_py_tf(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y,
#                                       v, beta, *init_vars, self._kappa_diag)
#         if self._backend == 'jit':
#             backward_recursion_set_jit(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y,
#                                        v, beta, *init_vars, self._kappa_diag)
#
#     def minimize_v(self, H=None, h=None, broadcast_h=True, return_constrains=False):
#         r"""
#         Returns the vector `v` of the squared error minimization with linear constraints
#
#         Minimizes the squared error over the vector `v` with linear constraints with an (optional) offset
#         [Wildhaber2018]_ [TABLE V].
#
#         **Constraint:**
#
#         - *Linear Scalar* : :math:`x=Hv,\,v\in\mathbb{R}`
#
#           known : :math:`H \in \mathbb{R}^{N \times 1}`
#
#           :math:`\hat{v}_k = \frac{\xi_k^{\mathsf{T}}H}{H^{\mathsf{T}}W_k H}`
#
#         - *Linear Combination With Offset* : :math:`x=Hv +h,\,v\in\mathbb{M}`
#
#           known : :math:`H \in \mathbb{R}^{N \times M},\,h\in\mathbb{R}^N`
#
#           :math:`\hat{v}_k = \big(H^{\mathsf{T}}W_k H\big)^{-1} H^\mathsf{T}\big(\xi_k - W_k h\big)`
#
#
#         Parameters
#         ----------
#         H : array_like, shape=(N, M)
#             Matrix for linear constraining :math:`H`
#         h : array_like, shape=(N, [S]), optional
#             Offset vector for linear constraining :math:`h`
#         broadcast_h : bool
#             if True each channel has the same h vector else h needs same shape as `x`.
#         return_constrains : bool
#             If set to True, the output is extended by H and h
#
#         Returns
#         -------
#         v : :class:`~numpy.ndarray`, shape = (K, M, S),
#             Least square state vector estimate for each time index.
#             The shape of one state vector `x[k]` is `(N, [S])`, where k is the time index of `K` samples,
#             `N` the ALSSM order.
#
#         |def_K|
#         |def_S|
#         |def_N|
#
#         """
#         N = self.cost_model.get_model_order()
#         S = np.shape(self.xi)[-1]
#
#         # check and init H
#         H = np.eye(N) if H is None else np.asarray(H)
#         assert H.shape[
#                    0] == N, f"first dimension of constrain matrix H needs to be of size {N} (model order), found size {H.shape[0]}"
#         M = H.shape[1]
#
#         # check and init h
#         if h is None:
#             h = np.zeros((N, S))
#         else:
#             if broadcast_h:
#                 h = np.repeat(h, S, axis=1)
#             else:
#                 h = np.asarray(h)
#         assert h.shape == (N,
#                            S), f"offset vector h needs to be of shape ({N}, {S}) (model order,  multi-channel set size), {info_str_found_shape(h)}"
#
#         # allocate v and minimize
#         v = np.full((len(self.W), M, S), np.nan)
#         minimize_v_py_ss(v, self.W, self.xi, H, h)
#
#         if return_constrains:
#             return v, H, h
#         return v
#
#     def minimize_x(self, H=None, h=None, broadcast_h=True):
#         r"""
#         Returns the state vector `x` of the squared error minimization with linear constraints
#
#         Minimizes the squared error over the state vector `x`.
#         If needed its possible to apply linear constraints with an (optional) offset.
#         [Wildhaber2018]_ [TABLE V].
#
#         **Constraint:**
#
#         - *Linear Scalar* : :math:`x=Hv,\,v\in\mathbb{R}`
#
#           known : :math:`H \in \mathbb{R}^{N \times 1}`
#
#         - *Linear Combination With Offset* : :math:`x=Hv +h,\,v\in\mathbb{M}`
#
#           known : :math:`H \in \mathbb{R}^{N \times M},\,h\in\mathbb{R}^N`
#
#         See also :meth:`minimize_v`
#
#         Parameters
#         ----------
#         H : array_like, shape=(N, M), optional
#             Matrix for linear constraining :math:`H`
#         h : array_like, shape=(N, [S]), optional
#             Offset vector for linear constraining :math:`h`
#         broadcast_h : bool
#             if True each channel has the same h vectore else h needs same shape as `x`.
#
#         Returns
#         -------
#         xs : :class:`~numpy.ndarray` of shape = (K, N, S)
#             Least square state vector estimate for each time index.
#             The shape of one state vector `x[k]` is `(N, S)`, where `k` is the time index of `K` samples,
#             `N` the ALSSM order.
#
#
#         |def_K|
#         |def_S|
#         |def_N|
#
#         """
#         if H is None and h is None:
#             # allocate x and minimize
#             x = np.full_like(self.xi, np.nan)
#             minimize_x_py_ss(x, self.W, self.xi)
#         else:
#             v, H, h = self.minimize_v(H, h, broadcast_h, return_constrains=True)
#             x = np.einsum('nm, kms->kns', H, v) + h
#
#         return x
#
#     def eval_errors(self, xs, ks=None):
#         r"""
#         Evaluation of the squared error for multiple state vectors `xs`.
#
#         The return value is the squared error
#
#         .. math::
#             J(x)  = x^{\mathsf{T}}W_kx -2*x^{\mathsf{T}}\xi_k + \kappa_k
#
#         for each state vector :math:`x` from the list `xs`.
#
#
#         Parameters
#         ----------
#         xs : array_like of shape=(K, N, S)
#             List of state vectors :math:`x`
#         ks : None, array_like of int of shape=(XS,)
#             List of indices where to evaluate the error
#
#         Returns
#         -------
#         J : :class:`np.ndarray` of shape=(XS, S [,S])
#             Squared Error for each state vector
#
#
#         |def_K|
#         |def_XS|
#         |def_N|
#
#         """
#         if ks is None:
#             if self._kappa_diag:
#                 return (np.einsum('kns, kns->ks', xs, np.einsum('knm, kmt->knt', self.W, xs))
#                         - 2 * np.einsum('kns, kns->ks', self.xi, xs)
#                         + self.kappa)
#             else:
#                 return (np.einsum('kns, knt->kst', xs, np.einsum('knm, kmt->knt', self.W, xs))
#                         - 2 * np.einsum('kns, knt->kst', self.xi, xs)
#                         + self.kappa)
#         else:
#             if self._kappa_diag:
#                 return (np.einsum('kns, kns->ks', xs[ks], np.einsum('knm, kmt->knt', self.W[ks], xs[ks]))
#                         - 2 * np.einsum('kns, kns->ks', self.xi[ks], xs[ks])
#                         + self.kappa[ks])
#             else:
#                 return (np.einsum('kns, knt->kst', xs[ks], np.einsum('knm, kmt->knt', self.W[ks], xs[ks]))
#                         - 2 * np.einsum('kns, knt->kst', self.xi[ks], xs[ks])
#                         + self.kappa[ks])
#
#
# class RLSAlssmSteadyState(RLSAlssmBase):
#     """
#     Filter and Data container for Recursive Least Square Alssm Filters in Steady State Mode
#
#     With :class:`RLSAlssmSteadyState` a common :math:`W_k = W_{steady}` is used for all samples (faster computation).
#     Note that using a common :math:`W_k` potentially leads to border missmatch effects and to completely invalid results
#     when samples have individual sample weights.
#
#     See Also
#     --------
#     :class:`RLSAlssm`
#
#     """
#
#     def __init__(self, cost_model, steady_state_method='closed_form', **kwargs):
#         super().__init__(**kwargs)
#         self.cost_model = cost_model
#         self._W = self.cost_model.get_steady_state_W(method=steady_state_method)
#
#     def _allocate_parameter_storage(self, input_shape):
#         K = input_shape[0]
#
#         N = self.cost_model.get_model_order()
#         self._xi = np.zeros((K, N))
#         self._kappa = np.zeros(K)
#         self._nu = np.zeros(K)
#
#     def _forward_recursion(self, A, C, segment, y, v, beta):
#         init_vars = forward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)
#
#         if self._backend == 'numpy':
#             forward_recursion_xi_kappa_nu_py_ss(self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
#                                              beta, *init_vars)
#         if self._backend == 'lfilter':
#             forward_recursion_xi_kappa_nu_py_tf(self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
#                                              beta, *init_vars)
#         if self._backend == 'jit':
#             forward_recursion_xi_kappa_nu_jit(self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y,
#                                               v,
#                                               beta, *init_vars)
#
#     def _backward_recursion(self, A, C, segment, y, v, beta):
#         init_vars = backward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)
#
#         if self._backend == 'numpy':
#             backward_recursion_xi_kappa_nu_py_ss(self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y,
#                                               v,
#                                               beta, *init_vars)
#         if self._backend == 'lfilter':
#             backward_recursion_xi_kappa_nu_py_tf(self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y,
#                                               v,
#                                               beta, *init_vars)
#         if self._backend == 'jit':
#             backward_recursion_xi_kappa_nu_jit(self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y,
#                                                v,
#                                                beta, *init_vars)
#
#     def minimize_v(self, H=None, h=None, return_constrains=False):
#         """
#         Returns the state vector `v` of the squared error minimization with linear constraints
#
#         See Also
#         --------
#         :class:`RLSAlssm.minimize_v`
#
#         """
#         N = self.cost_model.get_model_order()
#
#         # check and init H
#         H = np.eye(N) if H is None else np.asarray(H)
#         assert H.shape[0] == N, ""
#         M = H.shape[1]
#
#         # check and init h
#         h = np.zeros(N) if h is None else np.asarray(h)
#         assert h.shape == (N,), ""
#
#         # allocate v and minimize
#         v = np.full((len(self.xi), M), np.nan)
#         minimize_v_steady_state_py_ss(v, self.W, self.xi, H, h)
#
#         if return_constrains:
#             return v, H, h
#         return v
#
#     def minimize_x(self, H=None, h=None):
#         """
#         Returns the state vector `x` of the squared error minimization with linear constraints
#
#         See Also
#         --------
#         :class:`RLSAlssm.minimize_x`
#
#         """
#         if H is None and h is None:
#             # allocate x and minimize
#             x = np.full_like(self.xi, np.nan)
#             minimize_x_steady_state_py_ss(x, self.W, self.xi)
#         else:
#             v, H, h = self.minimize_v(H, h, return_constrains=True)
#             x = np.einsum('nm, km->kn', H, v) + h
#
#         return x
#
#     def eval_errors(self, xs, ks=None):
#         r"""
#         Evaluation of the squared error for multiple state vectors `xs`.
#
#         See Also
#         --------
#         :class:`RLSAlssm.eval_error`
#
#         """
#         if ks is None:
#             return (np.einsum('kn, kn->k', xs, np.einsum('nm, km->kn', self.W, xs))
#                     - 2 * np.einsum('kn, kn->k', self.xi, xs)
#                     + self.kappa)
#         else:
#             return (np.einsum('kn, kn->k', xs[ks], np.einsum('nm, km->kn', self.W, xs[ks]))
#                     - 2 * np.einsum('kn, kn->k', self.xi[ks], xs[ks])
#                     + self.kappa[ks])
#
#
# class RLSAlssmSetSteadyState(RLSAlssmBase):
#     """
#     Filter and Data container for Recursive Least Square Alssm Filters using Sets in Steady State Mode
#
#     With :class:`RLSAlssmSteadyState` a common :math:`W_k = W_{steady}` is used for all samples (faster computation).
#     Note that using a common :math:`W_k` potentially leads to border missmatch effects and to completely invalid results
#     when samples have individual sample weights.
#
#     See Also
#     --------
#     :class:`RLSAlssmSet`
#
#     """
#
#     def __init__(self, cost_model, steady_state_method='closed_form', kappa_diag=True, **kwargs):
#         super().__init__(**kwargs)
#         self._kappa_diag = None
#         self.cost_model = cost_model
#         self.set_kappa_diag(kappa_diag)
#         self._W = self.cost_model.get_steady_state_W(method=steady_state_method)
#
#     def set_kappa_diag(self, b):
#         assert isinstance(b, bool), 'kappa_diag is not of type bool'
#         self._kappa_diag = b
#
#     def _allocate_parameter_storage(self, input_shape):
#
#         K = input_shape[0]
#         S = input_shape[-1]
#         N = self.cost_model.get_model_order()
#         self._nu = np.zeros(K)
#         self._xi = np.zeros((K, N, S))
#         if self._kappa_diag:
#             self._kappa = np.zeros((K, S))
#         else:
#             self._kappa = np.zeros((K, S, S))
#
#     def _forward_recursion(self, A, C, segment, y, v, beta):
#         init_vars = forward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)
#
#         if self._backend == 'numpy':
#             forward_recursion_set_xi_kappa_nu_py_ss(self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta,
#                                                  y, v,
#                                                  beta, *init_vars, self._kappa_diag)
#         if self._backend == 'lfilter':
#             forward_recursion_set_xi_kappa_nu_py_tf(self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta,
#                                                  y, v,
#                                                  beta, *init_vars, self._kappa_diag)
#         if self._backend == 'jit':
#             forward_recursion_set_jit(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y,
#                                       v,
#                                       beta, *init_vars, self._kappa_diag)
#
#     def _backward_recursion(self, A, C, segment, y, v, beta):
#         init_vars = backward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)
#
#         if self._backend == 'numpy':
#             backward_recursion_set_xi_kappa_nu_py_ss(self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta,
#                                                   y,
#                                                   v, beta, *init_vars, self._kappa_diag)
#         if self._backend == 'lfilter':
#             backward_recursion_set_xi_kappa_nu_py_tf(self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta,
#                                                   y,
#                                                   v, beta, *init_vars, self._kappa_diag)
#         if self._backend == 'jit':
#             backward_recursion_set_jit(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y,
#                                        v, beta, *init_vars, self._kappa_diag)
#
#     def minimize_v(self, H=None, h=None, broadcast_h=True, return_constrains=False):
#         """
#         Returns the state vector `v` of the squared error minimization with linear constraints
#
#         See Also
#         --------
#         :class:`RLSAlssmSet.minimize_v`
#
#         """
#         N = self.cost_model.get_model_order()
#         S = np.shape(self.xi)[-1]
#
#         # check and init H
#         H = np.eye(N) if H is None else np.asarray(H)
#         assert H.shape[0] == N, ""
#         M = H.shape[1]
#
#         # check and init h
#         if h is None:
#             h = np.zeros((N, S))
#         else:
#             if broadcast_h:
#                 h = np.repeat(h, S, axis=1)
#             else:
#                 h = np.asarray(h)
#         assert h.shape == (N, S), ""
#
#         # allocate v and minimize
#         v = np.full((len(self.xi), M, S), np.nan)
#         minimize_v_steady_state_py_ss(v, self.W, self.xi, H, h)
#
#         if return_constrains:
#             return v, H, h
#         return v
#
#     def minimize_x(self, H=None, h=None, broadcast_h=True):
#         """
#         Returns the state vector `x` of the squared error minimization with linear constraints
#
#         See Also
#         --------
#         :class:`RLSAlssmSet.minimize_x`
#
#         """
#         if H is None and h is None:
#             # allocate x and minimize
#             x = np.full_like(self.xi, np.nan)
#             minimize_x_steady_state_py_ss(x, self.W, self.xi)
#         else:
#             v, H, h = self.minimize_v(H, h, broadcast_h, return_constrains=True)
#             x = np.einsum('nm, kms->kns', H, v) + h
#
#         return x
#
#     def eval_errors(self, xs, ks=None):
#         r"""
#         Evaluation of the squared error for multiple state vectors `xs`.
#
#         See Also
#         --------
#         :class:`RLSAlssm.eval_error`
#
#         """
#         if ks is None:
#             if self._kappa_diag:
#                 return (np.einsum('kns, kns->ks', xs, np.einsum('nm, kmt->knt', self.W, xs))
#                         - 2 * np.einsum('kns, kns->ks', self.xi, xs)
#                         + self.kappa)
#             else:
#                 return (np.einsum('kns, knt->kst', xs, np.einsum('nm, kmt->knt', self.W, xs))
#                         - 2 * np.einsum('kns, knt->kst', self.xi, xs)
#                         + self.kappa)
#         else:
#             if self._kappa_diag:
#                 return (np.einsum('kns, kns->ks', xs[ks], np.einsum('nm, kmt->knt', self.W, xs[ks]))
#                         - 2 * np.einsum('kns, kns->ks', self.xi[ks], xs[ks])
#                         + self.kappa[ks])
#             else:
#                 return (np.einsum('kns, knt->kst', xs[ks], np.einsum('nm, kmt->knt', self.W, xs[ks]))
#                         - 2 * np.einsum('kns, knt->kst', self.xi[ks], xs[ks])
#                         + self.kappa[ks])

