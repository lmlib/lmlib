"""
Recursive Least Square Alssm Classes to solve Alssm Cost Functions


x

"""

import sys
from typing import Union

from numpy.core.numeric import moveaxis
from numpy.linalg import inv, cond

from lmlib.statespace.backend import get_backend
from lmlib.statespace.cost import CompositeCost, NDCompositeCost
from lmlib.statespace.model import AlssmSum, Alssm
from lmlib.utils.check import *
from lmlib.statespace.backends import *


__all__ = ['RLSAlssm', 'NDRLSAlssm']


#
# class RLSAlssm:
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
#         If true, the RLSAlssm uses the steady state matrix of :math:`W` instead of the recursion. Default = True
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
#         - `filter_form='parallel'` for parallel block form
#         - `filter_form='cascade'` for cascade block form
#     backend : str, None
#         Sets an individual backend for the RLSAlssm.
#     """
#
#     def __init__(self, cost_model: CompositeCost,
#                  steady_state:bool=True,
#                  calc_W:bool=True,
#                  calc_xi:bool=True,
#                  calc_kappa:bool=True,
#                  calc_nu:bool=True,
#                  kappa_diag:bool=True,
#                  betas=None,
#                  filter_form:str='cascade',
#                  backend:Union[None, str]=None):
#
#         self.cost_model = cost_model
#         self.steady_state = steady_state
#         self.calc_W = calc_W
#         self.calc_xi = calc_xi
#         self.calc_kappa = calc_kappa
#         self.calc_nu = calc_nu
#         self.kappa_diag = kappa_diag
#         self.betas = betas
#         self.filter_form = filter_form
#
#         if backend:
#             assert backend in available_backends, f"{backend} is not a valid backend"
#             self._backend = backend
#         else:
#             self._backend = get_backend()
#
#         self._K = None
#         self._N = None
#         self._Ss = ()
#         self._xi0 = None
#         self._xi1 = None
#         self._xi2 = None
#
#
#     # Properties
#     @property
#     def cost_model(self) -> CompositeCost:
#         """CompositeCost : Cost Model"""
#         return self._cost_model
#
#     @cost_model.setter
#     def cost_model(self, cost_model):
#         assert isinstance(cost_model, CompositeCost), 'cost_model is not a subclass of CompositeCost'
#         self._cost_model = cost_model
#
#     @property
#     def betas(self):
#         """~numpy.ndarray : Segment scalars weights the cost function per segment"""
#         return self._betas
#
#     @betas.setter
#     def betas(self, betas):
#         P = len(self.cost_model.segments)
#         if betas is None:
#             self._betas = np.ones(P)
#         else:
#             assert is_array_like(betas), 'betas if not array_like'
#             assert P == len(betas), f'betas has wrong length, {info_str_found_shape(betas)}'
#             self._betas = np.array(betas)
#
#     @property
#     def filter_form(self) -> str:
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
#     def backend(self) -> str:
#         """str : backend used"""
#         return self._backend
#
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
#     def calc_W(self) -> bool:
#         """bool : Do :math:`W` parameter calculation"""
#         return self._calc_W
#
#     @calc_W.setter
#     def calc_W(self, calc_W):
#         assert isinstance(calc_W, bool), "calc_W not of type bool"
#         self._calc_W = calc_W
#
#     @property
#     def calc_xi(self) -> bool:
#         """bool : Do :math:`\\xi` parameter calculation"""
#         return self._calc_xi
#
#     @calc_xi.setter
#     def calc_xi(self, calc_xi):
#         assert isinstance(calc_xi, bool), "calc_xi not of type bool"
#         self._calc_xi = calc_xi
#
#     @property
#     def calc_kappa(self) -> bool:
#         """bool : Do  :math:`\\kappa` parameter calculation"""
#         return self._calc_kappa
#
#     @calc_kappa.setter
#     def calc_kappa(self, calc_kappa):
#         assert isinstance(calc_kappa, bool), "calc_kappa not of type bool"
#         self._calc_kappa = calc_kappa
#
#     @property
#     def calc_nu(self) -> bool:
#         """bool : Do  :math:`\\nu` parameter calculation"""
#         return self._calc_nu
#
#     @calc_nu.setter
#     def calc_nu(self, calc_nu):
#         assert isinstance(calc_nu, bool), "calc_nu not of type bool"
#         self._calc_nu = calc_nu
#
#     @property
#     def steady_state(self) -> bool:
#         """bool : Use steady state Matrix :math:`W`"""
#         return self._steady_state
#
#     @steady_state.setter
#     def steady_state(self, steady_state):
#         assert isinstance(steady_state, bool), "steady_state not of type bool"
#         self._steady_state = steady_state
#
#     @property
#     def kappa_diag(self) -> bool:
#         """bool : Use the diagonal of :math:`\\kappa` when :math:`y` is a set shape."""
#         return self._kappa_diag
#
#     @kappa_diag.setter
#     def kappa_diag(self, kappa_diag):
#         assert isinstance(kappa_diag, bool), "kappa_diag not of type bool"
#         self._kappa_diag = kappa_diag
#
#     def _allocate_xi2(self, K: int, N: int):
#         self._xi2 = np.zeros((K, N*N))
#
#     def _allocate_xi1(self, K: int, N: int, Ss: tuple):
#         self._xi1 = np.zeros((K, N) + Ss)
#
#     def _allocate_xi0(self, K: int, Ss: tuple):
#         self._xi0 = np.zeros((K, 1)+Ss)
#
#     def _allocate_nu(self, K):
#         self._nu = np.zeros((K,))
#
#     # Functions
#     def filter(self, y, v=None) -> None:
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
#         segments = self.cost_model.segments
#         alssms = self.cost_model.alssms
#         F = self.cost_model.F
#         betas = self.betas
#
#         # get signal length
#         K = len(y)
#         self._K = K
#         if v is not None:
#             assert len(v) == K, "length of v is not equal to length of y"
#         elif self._backend in ('numpy', 'jit'):
#             v = np.ones(K)
#         elif self._backend == 'lfilter':
#             v = 1
#
#         # get model order
#         alssm = AlssmSum(alssms)
#         N = alssm.N
#         self._N = N
#
#         # get signal set dimension
#         Ss = ()
#         if alssm.is_MC:
#             if np.ndim(y) < 2:
#                 raise ValueError("y does not have a valid shape")
#             if np.ndim(y) > 2:
#                 Ss = np.shape(y)[2:]
#             _y = y
#         else:
#             if np.ndim(y) > 1:
#                 Ss = np.shape(y)[1:]
#             _y = y.reshape(K, 1, *Ss)  # extend y for 2dim C (backend supports only 2dim C on purpose)
#
#         self._Ss = Ss
#
#         # not implemented: JIT for Multi-Set
#         if len(Ss) > 1 and self._backend == 'jit':
#             raise NotImplementedError("Just In Time (jit) backend is not implemented for Multi-Set Signals.")
#
#         # allocate necessary memory
#         if self._steady_state:
#             self._xi2 = self.cost_model.get_steady_state_W().flatten()
#         elif self.calc_W:
#             self._allocate_xi2(K, N)
#         else:
#             pass
#
#         if self.calc_xi:
#             self._allocate_xi1(K, N, Ss)
#
#         if self.calc_kappa:
#             self._allocate_xi0(K, Ss)
#
#         if self.calc_nu:
#             self._allocate_nu(K)
#
#
#         # run recursions
#
#         for segment, f, beta in zip(segments, F.T, betas):
#             alssm = AlssmSum(alssms, deltas=f, force_MC=True)  # create 2dim C for backend
#
#             if self.calc_W and not self._steady_state:
#                 if self._backend == 'numpy':
#                     numpy_recursion_xi2(self._xi2,
#                                         alssm.A, alssm.C,
#                                         segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                         _y, v, beta)
#
#                 elif self._backend == 'jit':
#                     jit_recursion_xi2(self._xi2,
#                                       alssm.A, alssm.C,
#                                       segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                       _y, v, beta)
#
#                 elif self._backend == 'lfilter':
#                     if self._filter_form == 'cascade':
#                         lfilter_cascade_xi2(self._xi2,
#                                             alssm.A, alssm.C,
#                                             segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                             _y, v, beta)
#
#                     elif self._filter_form == 'parallel':
#                         lfilter_parallel_xi2(self._xi2,
#                                              alssm.A, alssm.C,
#                                              segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                              _y, v, beta)
#                     else:
#                         raise ValueError("unknown filter-form: '{}'".format(self._filter_form))
#                 else:
#                     raise ValueError("unknown backend: '{}'".format(self._backend))
#
#             if self.calc_xi:
#
#                 if self._backend == 'numpy':
#                     numpy_recursion_xi1(self._xi1,
#                                         alssm.A, alssm.C,
#                                         segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                         _y, v, beta)
#
#                 elif self._backend == 'jit':
#                     jit_recursion_xi1(self._xi1,
#                                       alssm.A, alssm.C,
#                                       segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                       _y, v, beta)
#
#                 elif self._backend == 'lfilter':
#                     if self._filter_form == 'cascade':
#                         lfilter_cascade_xi1(self._xi1,
#                                             alssm.A, alssm.C,
#                                             segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                             _y, v, beta)
#
#                     elif self._filter_form == 'parallel':
#                         lfilter_parallel_xi1(self._xi1,
#                                              alssm.A, alssm.C,
#                                              segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                              _y, v, beta)
#                     else:
#                         raise ValueError("unknown filter-form '{}'".format(self._filter_form))
#                 else:
#                     raise ValueError("unknown backend: '{}'".format(self._backend))
#
#             if self.calc_kappa:
#
#                 if self._backend == 'numpy':
#                     numpy_recursion_xi0(self._xi0,
#                                         alssm.A, alssm.C,
#                                         segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                         _y, v, beta)
#
#                 elif self._backend == 'jit':
#                     jit_recursion_xi0(self._xi0,
#                                       alssm.A, alssm.C,
#                                       segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                       _y, v, beta)
#
#                 elif self._backend == 'lfilter':
#                     if self._filter_form == 'cascade':
#                         lfilter_cascade_xi0(self._xi0,
#                                             alssm.A, alssm.C,
#                                             segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                             _y, v, beta)
#                     elif self._filter_form == 'parallel':
#                         lfilter_parallel_xi0(self._xi0,
#                                              alssm.A, alssm.C,
#                                              segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                              _y, v, beta)
#                     else:
#                         raise ValueError("unknown filter-form '{}'".format(self._filter_form))
#                 else:
#                     raise ValueError("unknown backend: '{}'".format(self._backend))
#
#             if self.calc_nu:
#
#                 if self._backend == 'numpy':
#                     numpy_recursion_nu(self._nu,
#                                        alssm.A, alssm.C,
#                                        segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                        _y, v, beta)
#
#                 elif self._backend == 'jit':
#                     jit_recursion_nu(self._nu,
#                                      alssm.A, alssm.C,
#                                      segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                      _y, v, beta)
#
#                 elif self._backend == 'lfilter':
#                     if self._filter_form == 'cascade':
#                         lfilter_cascade_nu(self._nu,
#                                            alssm.A, alssm.C,
#                                            segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                            _y, v, beta)
#
#                     elif self._filter_form == 'parallel':
#                         lfilter_parallel_nu(self._nu,
#                                             alssm.A, alssm.C,
#                                             segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
#                                             _y, v, beta)
#                     else:
#                         raise ValueError("unknown filter-form '{}'".format(self._filter_form))
#                 else:
#                     raise ValueError("unknown backend: '{}'".format(self._backend))
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
#         if self._xi2 is None or self._xi1 is None:
#             raise ValueError("Not all Parameters are calculated to perform minimization. \n"
#                              "Check. calc_W=True or steady_state=True and calc_xi=True, ")
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
#         v = np.full((self._K, M, *self._Ss), np.nan)
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
#         if self._xi2 is None or self._xi1 is None:
#             raise ValueError("\nNot all Parameters are calculated to perform minimization. \n"
#                              "Set: calc_W=True or steady_state=True and calc_xi=True, ")
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
#         x = np.einsum('nm, km...->kn...', H, v)
#         x += h.reshape(-1, *[1]*len(self._Ss)) # todo: only of h or and H are given
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
#             return J - 2 * np.einsum('kn..., kn...->k...', self.xi, xs) + self.kappa[:, 0]
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


class NDRLSAlssm:
    r"""
    N-Dimensional Filter and Data container for Recursive Least Square Alssm Filters

    :class:`NDRLSAlssm` computes and stores intermediate values such as covariances,
    as required to efficiently solve n-dimensional recursive least squares problems
    of a model-based cost function :class:`NDCompositeCost` and given observations.

    Main intermediate variables are the covariance `W`, weighted mean `xi`, signal energy `kappa`, weighted number
    of samples `nu`, see [Baeriswyl2025]_

    Parameters
    ----------
    nd_cost_model : NDCompositeCost
       n-dimensional Cost Model
    nd_betas : array_like of shape=(L,P,) of floats, None, optional
       Segment Scalars. Factors weighting each of the `P` cost segments.
       If `betas` is not set, the weight is for each cost segment 1.
    steady_state : bool, optional
       If true, the RLSAlssm uses the steady state matrix of :math:`W` instead of the recursion. Default = True
    calc_W : bool, optional
       If false, RLSAlssm prohibits the calculation and memory allocation of :math:`W`. Default = True
    calc_xi : bool, optional
       If false, RLSAlssm prohibits the calculation and memory allocation of :math:`\\xi`. Default = True
    calc_kappa : bool, optional
       If false RLSAlssm prohibits the calculation and memory allocation of :math:`\\kappa`. Default = True
    calc_nu : bool, optional
       If false RLSAlssm prohibits the calculation and memory allocation of :math:`\nu`. Default = False
    filter_form: str, optional
        selects the filter form if lfilter is selected as backend. default = 'cascade'
        Other options are not yet supported.
    backend : str, optional
        selects the backend. Default = 'numpy'
        Other options are 'lfilter' and 'jit' see :ref:`backend`
    """

    def __init__(self, nd_cost_model, nd_betas=None,
                 steady_state: bool = True,
                 calc_W: bool = True,
                 calc_xi: bool = True,
                 calc_kappa: bool = True,
                 calc_nu: bool = False,
                 filter_form: str = 'cascade',
                 backend: Union[None, str] = None
                 ):

        self.nd_cost_model = nd_cost_model
        self.nd_betas = nd_betas

        self.steady_state = steady_state
        self.calc_W = calc_W
        self.calc_xi = calc_xi
        self.calc_kappa = calc_kappa
        self.calc_nu = calc_nu
        self.filter_form = filter_form

        if backend:
            self.backend = backend
        else:
            self.backend = get_backend()

        self._Ks = ()
        self._Ns = ()
        self._N = nd_cost_model.get_model_order()

        self._xi0 = None
        self._xi1 = None
        self._xi2 = None
        self._nu = None


    # Properties
    @property
    def nd_cost_model(self) -> NDCompositeCost:
        """NDCompositeCost : Cost Model"""
        return self._nd_cost_model

    @nd_cost_model.setter
    def nd_cost_model(self, nd_cost_model):
        assert isinstance(nd_cost_model, NDCompositeCost), 'nd_cost_model is not a subclass of NDCompositeCost'
        self._nd_cost_model = nd_cost_model
        self.L = self._nd_cost_model.L

    @property
    def nd_betas(self):
        """~numpy.ndarray : Scalar weight for each dimension and each cost segment"""
        return self._nd_betas

    @nd_betas.setter
    def nd_betas(self, nd_betas):
        self._nd_betas = []
        if nd_betas is None:
            for cost_model in self.nd_cost_model.composite_costs:
                P = len(cost_model.segments)
                self._nd_betas.append(np.ones(P))
        else:
            assert len(nd_betas) == self.L, f'nd_betas has wrong length, {info_str_found_shape(nd_betas)}'
            for betas, cost_model in zip(nd_betas, self.nd_cost_model.composite_costs):
                P = len(cost_model.segments)
                assert is_array_like(betas), 'betas if not array_like'
                assert P == len(betas), f'betas has wrong length, {info_str_found_shape(betas)}'
                self._nd_betas.append(betas)

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
    def backend(self) -> str:
        """str : backend used"""
        return self._backend

    @backend.setter
    def backend(self, backend):
        assert backend in available_backends, f"{backend} is not a valid backend"
        self._backend = backend

    @property
    def W(self):
        """:class:`~numpy.ndarray` : Filter Parameter :math:`W`"""
        N = self._N
        return self._xi2.reshape(N, N) if self._steady_state else self._xi2.reshape(*self._Ks, N, N)

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


    # functions
    def filter(self, y, v=None, dim_order=None)-> None:
        r"""
        Computes the intermediate parameters for subsequent squared error computations and minimization's.

        Computes the intermediate parameters using efficient forward- and backward recursions.
        The results are stored internally, ready to solve the least squares problem using e.g., :meth:`minimize_x`
        or :meth:`minimize_v`. The parameter allocation :meth:`allocate` is called internally,
        so a manual pre-allocation is not necessary.

        Parameters
        ----------
        y : array_like
            Input signal |br|
            - Single-channel signal is of `shape =(K1, ..., KL)` for |br|
            - Multi-channel signal is of `shape =(K1, ..., KL,L)` |br|
        v : array_like, optional
            Sample weights needs to be the same shape as `y`.
            By default, the sample weights are initialized to 1.
        dim_order : list, optional
            List of indices sorting the dimensions. Default refers to increasing order.
            i.e. `dim_order=[0,1]` means :math:`\xi_{1, 2}`
            i.e. `dim_order=[2,0,1]` means :math:`\xi_{3, 1, 2}`


        Notes
        -----
        |def_K|
        |def_Q|
        |def_L|

        """

        # check dimension order
        if dim_order is None:
            dim_order = np.arange(self.L)
        else:
            assert len(dim_order) == self.L, f'dim_order has wrong length, {info_str_found_shape(dim_order)}'

        # check input dimensions
        Q = self.nd_cost_model.get_model_output_dimension()
        if Q == ():
            Ks = np.shape(y)
            y_MC = y[..., np.newaxis]
        else:
            assert np.shape(y)[-1] == Q, 'input signal mising ALSSM output dimension Q'
            Ks = np.shape(y)[:-1]
            y_MC = y
        self._Ks = Ks

        # check sample weight
        if v is None:
            v = np.broadcast_to(1., Ks) # create an array of shape Ks, but contains only a single 1.0 in memory
        else:
            assert np.shape(v) == Ks, f'v has wrong shape, {info_str_found_shape(v)}'


        # calc xi2
        if self.steady_state:
            self._xi2 = self.nd_cost_model.get_steady_state_W(dim_order).flatten()
        elif self._calc_W and not self.steady_state:
            q = 2

            # first dimension
            xi_prev = self._rls_nd_xi_q_recursion(q, y_MC, v, dim_order[0])

            # n-dimensions
            for nd_dim in dim_order[1:]:
                xi_prev = self._rls_nd_xi_q_asterisk_l_recursion(xi_prev, q, y_MC, v, nd_dim)

            self._xi2 = xi_prev

        # calc x1
        if self._calc_xi:
            q = 1

            # first dimension
            xi_prev = self._rls_nd_xi_q_recursion(q, y_MC, v, dim_order[0])

            # n-dimensions
            for nd_dim in dim_order[1:]:
                xi_prev = self._rls_nd_xi_q_asterisk_l_recursion(xi_prev, q, y_MC, v, nd_dim)

            self._xi1 = xi_prev

        # calc x0
        if self._calc_kappa:
            q = 0

            # first dimension
            xi_prev = self._rls_nd_xi_q_recursion(q, y_MC, v, dim_order[0])

            # n-dimensions
            for nd_dim in dim_order[1:]:
                xi_prev = self._rls_nd_xi_q_asterisk_l_recursion(xi_prev, q, y_MC, v, nd_dim)

            self._xi0 = xi_prev[..., 0] # remove last dimension (=1)

        # calc nu
            # steady state
            # first dimension
                # segments
                    # to backends
            # nd dimensions
                # segments

    def _rls_nd_xi_q_recursion(self, q, y, v, model_dimension):

        cost_model = self.nd_cost_model.composite_costs[model_dimension]
        betas = self.nd_betas[model_dimension]
        N = cost_model.get_model_order()
        Ks = np.shape(y)[:-1]

        xi_curr = np.zeros(Ks + (N ** q,))
        xi_curr = np.moveaxis(xi_curr, model_dimension, -2)
        _y = np.moveaxis(y, model_dimension, -2)
        _v = np.moveaxis(v, model_dimension, -1)

        signal_dimensions = _y.shape[:-2]

        # cost segments
        for segment, f, beta in zip(cost_model.segments, cost_model.F.T, betas):
            alssm = AlssmSum(cost_model.alssms, deltas=f, force_MC=True)

            # backend recursion
            for ij in np.ndindex(signal_dimensions):
                _xi_q_recursion(xi_curr[ij], q,
                                alssm, segment,
                                _y[ij], _v[ij],
                                beta, self.backend, self.filter_form)

        return np.moveaxis(xi_curr, -2, model_dimension)

    def _rls_nd_xi_q_asterisk_l_recursion(self, xi_prev, q, y, v, model_dimension):
        cost_model = self.nd_cost_model.composite_costs[model_dimension]
        betas = self.nd_betas[model_dimension]
        N = cost_model.get_model_order()
        Nq_prev = xi_prev.shape[-1]
        Ks = np.shape(y)[:-1]

        xi_curr = np.zeros(Ks + (Nq_prev * N ** q,))
        xi_curr = np.moveaxis(xi_curr, model_dimension, 0)
        xi_prev = np.moveaxis(xi_prev, model_dimension, 0)

        # cost segments
        for segment, f, beta in zip(cost_model.segments, cost_model.F.T, betas):
            alssm = AlssmSum(cost_model.alssms, deltas=f, force_MC=True)
            _xi_q_asterisk_l_recursion(xi_curr, q, alssm, segment, xi_prev, v, beta, self.backend, self.filter_form)
        return np.moveaxis(xi_curr, 0, model_dimension)

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
            HTxiWh = np.einsum('nm, ...m-> ...n', H.T, self._xi1)
        else:
            if h.shape[0] != self._N:
                ValueError(f"First dimension of offset vector h needs to be of size {self._N} (model order), "
                           f"{info_str_found_shape(h)}.")
            HTxiWh = np.einsum('nm, ...m-> ...n', H.T, self._xi1 - self.W @ h)

        # constrained minimization
        M = H.shape[1]
        v = np.full(self._Ks + (M,), np.nan)
        msk = cond(HTWH) < 1 / sys.float_info.epsilon
        if self._steady_state:
            assert msk, 'H.T @ W @ H is not invertible.'
            v[...] = np.einsum('nm, ...m-> ...n', inv(HTWH), HTxiWh)
        else:
            v[msk] = np.einsum('...nm, ...m -> ...n', inv(HTWH[msk]), HTxiWh[msk])

        if return_constrains:
            return v, H, h
        return v

    def minimize_x(self, H=None, h=None):
        r"""
        Returns the state vector `x` of the squared error minimization with linear constraints

        Minimizes the squared error over the state vector `x`.
        If needed, it is possible to apply linear constraints with an (optional) offset.
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
            x = np.full_like(self.xi, np.nan)
            if self._steady_state:
                msk = cond(self.W) < (1 / sys.float_info.epsilon)
                # TODO: Condition is mostly not true
                # assert msk, 'Steady State W Matrix is not invertible.'
                x[...] = np.einsum('nm, ...m-> ...n', inv(self.W), self.xi)
            else:
                msk = cond(self.W) < (1 / sys.float_info.epsilon)
                assert np.any(msk), 'All W Matrices are not invertible.'
                x[msk] = np.einsum('...nm, ...m -> ...n', inv(self.W[msk]), self.xi[msk])
            return x

        v, H, h = self.minimize_v(H, h, return_constrains=True)
        x = np.einsum('nm, ...m->...n', H, v)
        x += h
        return x

    def filter_minimize_x(self, y, v=None, dim_order=None, H=None, h=None):
        """
        Combination of :meth:`filter` and :meth:`minimize_x`.

        This method has the same output as calling the methods

        .. code::

            rls.filter(y)
            xs = rls.minimize_x()


        See Also
        --------
        :meth:`filter`, :meth:`minimize_x`

        """

        self.filter(y, v, dim_order)
        return self.minimize_x(H, h)

    def filter_minimize_v(self, y, v=None, dim_order=None, H=None, h=None, **kwargs):
        """
        Combination of :meth:`filter` and :meth:`minimize_v`.

        This method has the same output as calling the methods

        .. code::

            rls.filter(y)
            xs = rls.minimize_v()


        See Also
        --------
        :meth:`filter`, :meth:`minimize_v`

        """

        self.filter(y, v, dim_order)
        return self.minimize_v(H, h, **kwargs)

    def filter_minimize_yhat(self, y, v=None, dim_order=None, H=None, h=None, alssm_weights=None, c0s=None):
        """
        Combination of :meth:`filter` and :meth:`minimize_x` and
        :meth:`eval_alssm_output`

        This method has the same output as calling the methods

        .. code::

            xs = rls.filter_minimize_x()
            y_hat = rls.cost_model.eval_alssm_output(xs)

        See Also
        --------
        :meth:`filter`, :meth:`minimize_x`

        """
        xs = self.filter_minimize_x(y, v=v, dim_order=dim_order, H=H, h=h)
        return self.nd_cost_model.eval_nd_alssm_output(xs, dim_order=dim_order, alssm_weights=alssm_weights)

    def eval_errors(self, xs, ks=None):
        r"""
        Evaluation of the squared error for multiple state vectors `xs`.

        The return value is the squared error

        .. math::
            J(x)  = x^{\mathsf{T}}W_kx -2*x^{\mathsf{T}}\xi_k + \kappa_k

        for each state vector :math:`x` from the list `xs`.


        Parameters
        ----------
        xs : array_like of shape=(K1, ..., KL, N)
            List of state vectors :math:`x`
        ks : None, array_like of int of shape=(XS,)
            List of indices where to evaluate the error
            TODO: for nd

        Returns
        -------
        J : :class:`np.ndarray` of shape=(XS,)
            Squared Error for each state vector


        |def_K|
        |def_XS|
        |def_N|

        """

        if self._steady_state:
            J = np.einsum('...n, ...n', xs, np.einsum('nm, ...m->...n', self.W, xs))

        if ks is None:
            if not self._steady_state:
                J = np.einsum('...n, ...n', xs, np.einsum('...nm, ...m->...n', self.W, xs))
            return J - 2 * np.einsum('...n, ...n', self.xi, xs) + self.kappa

        else:
            if not self._steady_state:
                J = np.einsum('...n, ...n', xs[ks], np.einsum('...nm, ...m->...n', self.W[ks], xs[ks]))
            return J - 2 * np.einsum('...n, ...n', self.xi[ks], xs[ks]) + self.kappa[ks]

    def eval_nd_alssm_output(self, xs, dim_order=None, alssm_weights=None):
        r"""
        Evaluation of the n-Dimensioanl ALSSM output for multiple state vectors `xs`.

        See Also
        --------
        :meth:`Alssm.eval_states`

        Parameters
        ----------
        xs : array_like of shape=(K1, ..., KL, N)
           List of state vectors :math:`x`
        dim_order : None, array_like of int
            Definees the dimension order used in the state vectors
        alssm_weights : array_like of float
            List of weights for each Alssm output matrix

        Returns
        -------
        y_hat : array_like of shape=(K1, ..., KL,[Q,])
           ND-Alssm Output


        |def_K|
        |def_L|
        |def_N|
        |def_Q|

        """


        nd_alssm = self.nd_cost_model.get_nd_alssm(dim_order=dim_order, alssm_weights=alssm_weights)
        return nd_alssm.eval_states(xs)



class RLSAlssm(NDRLSAlssm):
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
    betas : array_like of shape=(P, ) of floats, None, optional
        Segment Scalars. Factors weighting each of the `P` cost segments.
        If `betas` is not set, the weight is for each cost segment 1.
    steady_state : bool, optional
        If true, the RLSAlssm uses the steady state matrix of :math:`W` instead of the recursion. Default = True
    calc_W : bool, optional
        If false, RLSAlssm prohibits the calculation and memory allocation of :math:`W`. Default = True
    calc_xi : bool, optional
        If false, RLSAlssm prohibits the calculation and memory allocation of :math:`\\xi`. Default = True
    calc_kappa : bool, optional
        If false RLSAlssm prohibits the calculation and memory allocation of :math:`\\kappa`. Default = True
    calc_nu : bool, optional
        If false RLSAlssm prohibits the calculation and memory allocation of :math:`\nu`. Default = False
    """

    def __init__(self, cost_model: CompositeCost, betas=None, *args, **kwargs):
        nd_cost_model = NDCompositeCost([cost_model])
        nd_betas = None if betas is None else [betas]
        super().__init__(nd_cost_model=nd_cost_model,
                         nd_betas=nd_betas,
                         *args, **kwargs)

    def filter(self, y, v=None, dim_order=None):
        dim_order = [0] if dim_order is None else dim_order
        super().filter(y, v, dim_order=[0])


def _xi_q_asterisk_l_recursion(xi_curr, q, alssm, segment, xi_prev, v, beta, backend, filter_form):
    Nq_prev = xi_prev.shape[-1]
    INq = np.eye(Nq_prev)
    A = kron_q(alssm.A, q)
    C = kron_q(alssm.C, q)

    if backend in ('numpy', 'jit', 'lfilter'):
        numpy_xi_asterisk_l_recursion(xi_curr, A, C,
                                      segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                      INq, xi_prev,
                                      v, beta)
    else:
        raise ValueError("unknown backend: '{}'".format(backend))


def _xi_q_recursion(xi, q, alssm, segment, y, v, beta, backend, filter_form):

    if backend == 'numpy':
        if q == 2:
            numpy_recursion_xi2(xi,
                                alssm.A, alssm.C,
                                segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                y, v, beta)
        elif q == 1:
            numpy_recursion_xi1(xi,
                                alssm.A, alssm.C,
                                segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                y, v, beta)
        elif q == 0:
            numpy_recursion_xi0(xi,
                                alssm.A, alssm.C,
                                segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                y, v, beta)
        else:
            raise ValueError("q value not supported: '{}'".format(q))

    elif backend == 'jit':
        if q == 2:
            jit_recursion_xi2(xi,
                              alssm.A, alssm.C,
                              segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                              y, v, beta)
        elif q == 1:

            jit_recursion_xi1(xi,
                              alssm.A, alssm.C,
                              segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                              y, v, beta)
        elif q == 0:
            jit_recursion_xi0(xi,
                                alssm.A, alssm.C,
                                segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                y, v, beta)
        else:
            raise ValueError("q value not supported: '{}'".format(q))

    elif backend == 'lfilter':
        if filter_form == 'cascade':
            if q == 2:
                lfilter_cascade_xi2(xi,
                                    alssm.A, alssm.C,
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta)
            elif q == 1:
                lfilter_cascade_xi1(xi,
                                    alssm.A, alssm.C,
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta)
            elif q == 0:
                lfilter_cascade_xi0(xi,
                                    alssm.A, alssm.C,
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta)
            else:
                raise ValueError("q value not supported: '{}'".format(q))



        elif filter_form == 'parallel':
            if q == 2:
                lfilter_parallel_xi2(xi,
                                    alssm.A, alssm.C,
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta)
            elif q == 1:
                lfilter_parallel_xi1(xi,
                                    alssm.A, alssm.C,
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta)
            elif q == 0:
                lfilter_parallel_xi0(xi,
                                    alssm.A, alssm.C,
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta)
            else:
                raise ValueError("q value not supported: '{}'".format(q))
        else:
            raise ValueError("unknown filter-form: '{}'".format(filter_form))
    else:
        raise ValueError("unknown backend: '{}'".format(backend))



