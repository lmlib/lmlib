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
    r"""
    Recursive Least Square Alssm Class to solve Alssm Cost Functions.

    This class uses either a :class:`CostSegment`, :class:`CompositeCost` or :class:`NDCompositeCost` and defines the functions to solve it recursively. 
    :math:`W_k`, :math:`\xi_k`, :math:`\kappa_k` and :math:`\nu_k` can be computed either as a forward or as a backward recursion as defined in Eq. (22-25) [Wildhaber2018]_.
    :math:`W_k` is the Gram matrix defined by the ALSSM (:math:`A`, :math:`c`, :math:`\alpha` and :math:`w`). :math:`\mathrm{vec}(W_k)=\xi^{(2)}(k,y)=\xi^{(2)}(k,\mathbf{1})` (with :math:`\mathbf{1}` the all ones vector) Eq. (21) [Baeriswyl2025]_.
    :math:`\xi_k` (:math:`=\xi^{(1)}(k,y)` Eq. (20) [Baeriswyl2025]_) is the projection of the signal to the ALSSM subspace.
    :math:`\kappa_k` gives the energy of a signal weighted under :math:`\alpha` and :math:`w`. :math:`\kappa_k=\xi^{(0)}(k,y)` Eq. (19) [Baeriswyl2025]_.
    Additionally, :math:`\nu_k` is introduced which is the number of weighted samples in the window.

    The cost function is (Eq. (20) [Wildhaber2018]_):
    .. math::
        J_k(x) = \sum_{i=k+a}^{k+b} \alpha^{k+\delta}(i) w_i \big(y_i - CA^{i-k}x\big)^2

    .. seealso:: 
        [Wildhaber2018]_ [Wildhaber2019]_ 
        For the definition of the :math:`\xi^{(q)}(k,y)` terms see Eq. (19-21) [Baeriswyl2025]_

    Parameters
    ----------
    cost_terms : CostSegment, CompositeCost, NDCompositeCost
        Cost function to be minimized recursively. See :class:`CostSegment`, :class:`CompositeCost` or :class:`NDCompositeCost`.
    steady_state : bool, optional
        Defines if the ALSSM is steady state (not time-varying, e.g. LTI). If so, :math:`W_k` reduces to :math:`W`. Default: True.
        This happens in case :math:`w_k = w`, :math:`\gamma_k = \gamma` (see Sec. III-I.2 [Wildhaber2018]_).
        Setting this incorrectly may produce silently wrong results.
    calc_W : bool, optional
        If True, computes the Gram matrix :math:`W_k` (:math:`\xi^{(2)}`).
        Required for :meth:`minimize_v`, :meth:`minimize_x`, and :meth:`eval_errors`. Default: True.
    calc_xi : bool, optional
        If True, computes the signal projection :math:`\xi_k = \xi^{(1)}(k, y)`.
        Required for :meth:`minimize_v`, :meth:`minimize_x`, and :meth:`eval_errors`. Default: True.
    calc_kappa : bool, optional
        If True, computes the signal energy :math:`\kappa_k = \xi^{(0)}(k, y)`.
        Required for :meth:`eval_errors`. Can be set to False when only the minimizer is needed. Default: True.
    calc_nu : bool, optional
        If True, computes :math:`\nu_k`, the number of weighted samples in the window.
        Not yet implemented. Default: False.
    filter_form : str, optional
        Controls the internal block structure of the recursive filter.

        - ``'cascade'`` : cascade block form (default)
        - ``'parallel'`` : parallel block form

    backend : str, optional
        Selects the computational backend for the state-space recursions.
        If ``None``, the globally configured backend is used (see :func:`set_backend`).

        - ``'numpy'`` : pure NumPy implementation (default)
        - ``'lfilter'`` : transfer-function backend using :func:`scipy.signal.lfilter`
        - ``'jit'`` : Numba JIT-compiled backend (requires ``numba`` package)

    Notes
    -----
    Setting ``steady_state=True`` (default) assumes the window defined by the cost segments
    does not change over time (LTI case), which allows :math:`W_k` to be precomputed once
    as a constant :math:`W`. Set ``steady_state=False`` for time-varying windows
    (e.g. near signal boundaries or with sample-dependent weights), at the cost of
    a full :math:`W_k` recursion at every sample.

    Examples
    --------
    Fit a polynomial of degree 2 to a signal using a left-sided window:

    >>> import numpy as np
    >>> import lmlib as lm
    >>> from lmlib.utils.generator import gen_rect
    >>> K = 500
    >>> y = gen_rect(K, 200, 100)
    >>> alssm = lm.AlssmPoly(poly_degree=2)
    >>> seg = lm.Segment(a=-20, b=0, direction=lm.FORWARD, g=10)
    >>> rls = lm.RLSAlssm(lm.CostSegment(alssm, seg))
    >>> y_hat = rls.fit(y)

    To save computation when only the signal estimate is needed, disable :math:`\kappa`:

    >>> rls = lm.RLSAlssm(lm.CostSegment(alssm, seg), calc_kappa=False)
    >>> y_hat = rls.fit(y)
    """

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
        """
        Calculates the ALSSM cost parameters :math:`\xi^{(q)}(k,y)` for :math:`q \in \{0,1,2\}` and :math:`\nu_k` based on an input signal :math:`y`.

        The cost parameters :math:`\xi^{(q)}(k,y)` for :math:`q \in \{0,1,2\}` are equivalent to :math:`\kappa_k`, :math:`\xi_k` and :math:`W_k`. They are calculated through the recursive equations (22-25) [Wildhaber2018]_ for each cost segment defined by the different backends.
        This function validates if every parameter has the right dimensions and redirects to each :math:`q`. It also coordinates the recursion steps across dimensions: :meth:`_nd_xi_q_recursion()` for the first dimension, :meth:`_nd_xi_q_asterisk_l_recursion()` for the subsequent ones.
        TODO: calculate :math:`\nu_k`.

        Parameters
        ----------
        y : array_like of shape (K, [Q])
            Input signal. The Q dimension is the ALSSM output dimension, and for scalar ALSSMs (Q=0) a 1D array is also accepted.
        sample_weights : array_like of shape (K,), optional
            Per-sample weights :math:`w_i \in \[0,1\]`. Default: all ones.
        dim_order : array_like of int, optional
            Has no effect for :class:`CostSegment` and :class:`CompositeCost`. The order in which ND dimensions are reduced in the recursion for :class:`NDCompositeCost`. Default: np.arange(L) with L number of dimensions.

        Notes
        -----
        When `steady_state=True`, :math:`W_k` is precomputed once as a constant :math:`W` regardless of `calc_W`. This is valid when all window parameters are independent of :math:k (i.e., :math:`w_k = w` and :math:\gamma_k = \gamma), in which case the :math:`W_k` recursion converges to a steady state satisfying a Lyapunov equation (see Section III-I.2 in [Wildhaber2018]_).
        The code uses :math:`Q=0` to mean a :math:`1\times1` scalar, not zero-dimensional.

        Returns
        -------
        None
            Results are stored in-place and accessed via :attr:`W`, :attr:`xi`, and :attr:`kappa`.

        Raises
        ------
        ValueError
            If `y` has wrong shape.
            If `sample_weights` has wrong shape.
        AssertionError
            If `dim_order` has wrong length.
        """

        # -------- check dimension order --------
        L = self._cost_terms.get_number_of_dimensions()
        if dim_order is None:
            dim_order = np.arange(L)
        assert len(dim_order) == L, f'dim_order has wrong length, {info_str_found_shape(dim_order)}'

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

        # -------- calc xi1 --------
        if self._calc_xi:
            q = 1

            # first dimension
            xi_prev = self._nd_xi_q_recursion(q, y, sample_weights, dim_order[0])

            # n-dimensions
            for nd_dim in dim_order[1:]:
                xi_prev = self._nd_xi_q_asterisk_l_recursion(xi_prev, q, y, sample_weights, nd_dim)

            self._xi1 = xi_prev

        # -------- calc xi0 --------
        if self._calc_kappa:
            q = 0

            # first dimension
            xi_prev = self._nd_xi_q_recursion(q, y, sample_weights, dim_order[0])

            # n-dimensions
            for nd_dim in dim_order[1:]:
                xi_prev = self._nd_xi_q_asterisk_l_recursion(xi_prev, q, y, sample_weights, nd_dim)

            self._xi0 = xi_prev[..., 0]  # remove the last dimension  due to leftovers of nd-model-order

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
        """
        Defines the recursion to calculate the ALSSM cost parameters :math:`\xi^{(q)}(k,y)` for a given :math:`q \in \{0,1,2\}` based on an input signal :math:`y`.

        The cost parameters :math:`\xi^{(q)}(k,y)` for :math:`q \in \{0,1,2\}` are equivalent to :math:`\kappa_k`, :math:`\xi_k` and :math:`W_k`. They are calculated through the recursive equations (22-25) [Wildhaber2018]_ for each cost segment defined by the different backends.
        This function subdivides the calculation into cost segments (for-loop).

        Parameters
        ----------
        q : int
            Order of the cost parameter. Must be 0, 1, or 2, corresponding to
            :math:`\kappa_k` (signal energy), :math:`\xi_k` (signal projection), and
            :math:`W_k` (Gram matrix), respectively.
            Determines the trailing size of the output: :math:`N_l^q`, where :math:`N_l`
            is the model order of the sub-cost associated with ``model_dimension``.
        y : array_like of shape (\*Ks, Q)
            Input signal, where ``Ks`` are the signal length dimensions (one per ND axis)
            and ``Q`` is the ALSSM output dimension. For scalar ALSSMs ``Q=0`` after
            preprocessing in :meth:`filter`.
        sample_weights : array_like of shape (\*Ks,)
            Per-sample weights :math:`w_i \in [0,1]`.
        model_dimension : int
            Index of the signal axis along which the 1-D state-space recursion is applied.
            All other signal axes are flattened into a batch and iterated over
            sequentially (one slice at a time via an inner loop).
            For a 1-D signal (``CostSegment`` / ``CompositeCost``) this is always 0.
            For an ND signal (``NDCompositeCost``) each axis is processed in a separate call.

        Notes
        -----
        The difference between this function and :meth:`_nd_xi_q_asterisk_l_recursion()` is
        that this function is the **first** recursion step: it reads directly from the input
        signal `y` and initialises :math:`\xi^{(q)}` from scratch with shape
        ``(*Ks, N_l**q)``.

        :meth:`_nd_xi_q_asterisk_l_recursion()` is used for every **subsequent** dimension:
        it takes the accumulated ``xi_prev`` (shape ``(*Ks, Nq_prev)``) from the previous step
        and extends the trailing axis to ``Nq_prev * N_l**q``, realising the tensor-product
        (Kronecker) structure over ND dimensions. After processing all L dimensions the
        trailing axis has size :math:`(N_0 \cdots N_{L-1})^q = N_\mathrm{total}^q`.

        Returns
        -------
        xi_curr : :class:`~numpy.ndarray` of shape (\*Ks, N_l**q)
            :math:`\xi^{(q)}(k,y)` cost parameter with the same leading shape as `y`
            (all signal dimensions) and a trailing axis of size :math:`N_l^q`, where
            :math:`N_l` is the model order of the sub-cost for ``model_dimension``.
        """

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
        """
        Defines the recursion to calculate the ALSSM cost parameters :math:`\xi^{(q)}(k,y)` for a given :math:`q \in \{0,1,2\}` based on an input signal :math:`y`.

        The cost parameters :math:`\xi^{(q)}(k,y)` for :math:`q \in \{0,1,2\}` are equivalent to :math:`\kappa_k`, :math:`\xi_k` and :math:`W_k`. They are calculated through the recursive equations (22-25) [Wildhaber2018]_ for each cost segment defined by the different backends.
        This function subdivides the calculation into cost segments (for-loop).

        Parameters
        ----------
        xi_prev : :class:`~numpy.ndarray` of shape (\*Ks, Nq_prev)
            Accumulated cost parameter from the previous dimension step, where
            ``Nq_prev`` is the trailing size built up so far (e.g. ``N_0**q`` after
            the first step, ``N_0**q * N_1**q`` after the second, etc.).
        q : int
            Order of the cost parameter. Must be 0, 1, or 2, corresponding to
            :math:`\kappa_k` (signal energy), :math:`\xi_k` (signal projection), and
            :math:`W_k` (Gram matrix), respectively.
            The trailing axis of the output grows by a factor :math:`N_l^q`, where
            :math:`N_l` is the model order of the sub-cost for ``model_dimension``.
        y : array_like of shape (\*Ks, Q)
            Input signal. Only used to determine the leading shape ``Ks``; the
            actual values are not read (the signal information enters via ``xi_prev``).
        sample_weights : array_like of shape (\*Ks,)
            Per-sample weights :math:`w_i \in [0,1]`.
        model_dimension : int
            Index of the signal axis along which the 1-D state-space recursion is applied.
            All other signal axes are moved to the front and iterated over
            sequentially (one slice at a time via the backend call).
            For an ND signal (``NDCompositeCost``) each axis is processed in a separate call.

        Notes
        -----
        The difference between this function and :meth:`_nd_xi_q_recursion()` is
        that this function defines every **subsequent** recursion step after the first:
        it reads from the accumulated ``xi_prev`` (shape ``(*Ks, Nq_prev)``) and
        extends the trailing axis to ``Nq_prev * N_l**q``, realising the tensor-product
        (Kronecker) structure over ND dimensions. After processing all L dimensions the
        trailing axis has size :math:`(N_0 \cdots N_{L-1})^q = N_\mathrm{total}^q`.

        :meth:`_nd_xi_q_recursion()` is used for the **first** dimension:
        it reads directly from the input signal `y` and initialises :math:`\xi^{(q)}`
        from scratch with shape ``(*Ks, N_0**q)``.

        Returns
        -------
        xi_curr : :class:`~numpy.ndarray` of shape (\*Ks, Nq_prev \* N_l**q)
            Extended cost parameter. The leading shape matches `y`; the trailing axis
            is the product of all accumulated sub-cost orders raised to :math:`q`.
        """

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

