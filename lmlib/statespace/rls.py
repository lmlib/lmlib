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

    num_denom : TODO

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
                 backend=None, numdenom=None, steady_state_method='schur'):
        self._cost_terms = cost_terms
        assert all(isinstance(_, bool) for _ in (steady_state, calc_W, calc_xi, calc_kappa, calc_nu)), \
            'steady_state, calc_W, calc_xi, calc_kappa and calc_nu must be boolean.'

        self._steady_state = steady_state
        self._steady_state_method = steady_state_method
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

        self._build_cascade_params(_sub_costs)

        self._build_numdenom(_sub_costs, numdenom)

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
    
    def _build_cascade_params(self, _sub_costs):
        r"""
        Build _cascade_params as a 3-D structure: _cascade_params[dim_index][seg_index][alssm_index]
        
        Each leaf entry is either None (numpy/jit backends, parallel filter form,
        or inactive grid node with f_mp==0) or a dict of precomputed scalars and
        matrices for the lfilter cascade backend. The dict keys depend on the
        segment direction:
          fw: gamma_inv, gamma_a, gamma_b, gAinvT, Aac, Abc, N
          bw: gamma,     gamma_a, gamma_b, gAT,    Aac, Abc, N
        
        For q==1 (xi) the filtering is done per individual ALSSM, so each
        ALSSM gets its own pre-calculated parameters derived from its own
        (small) A_m and C_m rather than from the combined block-diagonal matrix.
        This avoids constructing the large AlssmSum just to decompose it again.
        
        For q==2 (W) the combined AlssmSum is still used (see recursion methods),
        so _cascade_params[dim][p][m] is not consumed for q==2; the combined entry is
        built on-the-fly there instead.
        """
        self._cascade_params = [
            [[None] * ct.M for _ in range(ct.P)] for ct in _sub_costs
            ]

        if self._filter_form == 'cascade' and self._backend == 'lfilter':
            from lmlib.statespace.backends.rec_lfilter import _compute_cascade_params
            for dim_idx, ct in enumerate(_sub_costs):
                for p, segment in enumerate(ct.segments):
                    a, b, delta, gamma = segment.a, segment.b, segment.delta, segment.gamma
                    for m, alssm_m in enumerate(ct.alssms):
                        if ct.F[m, p] == 0.0:
                            continue
                        wrapped = AlssmSum([alssm_m], [ct.F[m, p]], force_MC=True)
                        A, C = wrapped.A, wrapped.C
                        self._cascade_params[dim_idx][p][m] = _compute_cascade_params(
                            A, C, a, b, delta, gamma, segment.direction
                        )

    
    def _build_numdenom(self, _sub_costs, numdenom):
        r"""
        Build _numdenom as a 3-D structure: _numdenom[dim_index][seg_index][alssm_index]
        
        Each leaf entry is either None (numpy/jit backends, or inactive grid
        node with f_mp==0) or [denom, num_b, num_a] (lfilter parallel backend).
        
        For q==1 (xi) the filtering is now done per individual ALSSM, so each
        ALSSM gets its own transfer-function coefficients derived from its own
        (small) A_m and C_m rather than from the combined block-diagonal matrix.
        This avoids constructing the large AlssmSum just to decompose it again.
        
        For q==2 (W) the combined AlssmSum is still used (see recursion methods),
        so _numdenom[dim][p][m] is not consumed for q==2; the combined entry is
        built on-the-fly there instead.
        """
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



        if self._filter_form == 'parallel' and self._backend == 'lfilter' and numdenom is not None:
            print("Using user-supplied SOS coefficients (numdenom).")
            self._numdenom = numdenom


    # ------------------------------------------------------------------
    # filter 
    # ------------------------------------------------------------------

    def filter(self, y, sample_weights=None, dim_order=None):
        r"""
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
            self._xi2 = self._cost_terms.get_steady_state_W(dim_order,method=self._steady_state_method).flatten()
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

    def minimize_v(self, H=None, h=None, solver='lstsq'):
        r"""
        Minimizes the cost :math:`J_k(x)` subject to a linear constraint on the state vector,
        returning the reduced-dimensional free parameter :math:`v`.

        The state vector :math:`x` is constrained to the affine subspace

        .. math::
            x = Hv + h,

        with known :math:`H \in \mathbb{R}^{N \times M}` and :math:`h \in \mathbb{R}^N`, and
        unknown :math:`v \in \mathbb{R}^M`. Substituting into the cost function (Eq. (21)
        [Wildhaber2018]_) and minimizing over :math:`v` yields the closed-form solution
        (Eq. (69) [Wildhaber2018]_):

        .. math::
            \hat{v}_k = (H^T W_k H)^{-1} H^T (\xi_k - W_k h).

        When ``H=None`` and ``h=None`` (defaults), the constraint reduces to :math:`x = v`
        (unconstrained minimization), so :math:`\hat{v}_k = W_k^{-1} \xi_k`.

        Parameters
        ----------
        H : array_like of shape (N, M), optional
            Constraint matrix mapping the free parameter :math:`v \in \mathbb{R}^M` to the
            state space :math:`\mathbb{R}^N`. If ``None``, defaults to the identity matrix
            :math:`I_N` (unconstrained, :math:`M = N`).
        h : array_like of shape (N,), optional
            Constraint offset vector :math:`h \in \mathbb{R}^N`. If ``None``, defaults to
            the zero vector (no offset).
        solver : {'lstsq', 'solve', 'inv'}, optional
            Linear-system solver strategy (default: ``'lstsq'``).

        Notes
        -----
        The distinction between :meth:`minimize_v` and :meth:`minimize_x` is:

        - :meth:`minimize_v` returns :math:`v \in \mathbb{R}^M` (the free parameter, dimension M).
        - :meth:`minimize_x` returns :math:`x = Hv + h \in \mathbb{R}^N` (the full state vector,
          dimension N).

        When ``H=None``, both methods return the same result since :math:`x = v`.

        When ``steady_state=True``, :math:`W_k = W` is constant, so :math:`H^T W H` is
        inverted once. When ``steady_state=False``, samples where :math:`H^T W_k H` is
        ill-conditioned (singular to machine precision) are left as ``NaN``.

        Returns
        -------
        v : :class:`~numpy.ndarray` of shape (..., M)
            Free parameter :math:`\hat{v}_k` minimizing :math:`J_k(Hv + h)`. The leading
            dimensions match the signal length and any parallel dimensions of the input.
            Entries are ``NaN`` where :math:`H^T W_k H` is ill-conditioned
            (only possible when ``steady_state=False``).

        Raises
        ------
        AssertionError
            If ``H`` does not have ``N`` rows.
            If ``h`` does not have length ``N``.
            If ``H.T @ W @ H`` is not invertible (only when ``steady_state=True``).


        Solver strategies
        -----------------
        Three strategies are available via the ``solver`` keyword.  All are
        **closed-form** (no iterative steps).

        ``'lstsq'`` *(default)*
            **Steady-state** — Computes the economy SVD of :math:`H^\top W H`
            *once* and stores the resulting pseudoinverse as an attribute
            ``_pinv_HTWH``.  Subsequent calls with the same ``H`` and ``h``
            reuse this cached pseudoinverse, reducing the cost to a single
            matrix–vector multiply per time step.  This is typically **4–22×
            faster** than calling ``numpy.linalg.lstsq`` with all right-hand
            sides at once, because that function recomputes the SVD on every
            call even when the matrix has not changed.

            **Non-steady-state** — Attempts a batched LU solve
            (``numpy.linalg.solve``) on the full :math:`(K, N, N)` stack of
            matrices.  If any matrix is singular, falls back to a fully
            vectorised batched SVD pseudoinverse (``numpy.linalg.svd`` over
            the batch dimension) — no Python loop required.  This gives a
            **5–70× speedup** over the previous per-sample ``lstsq`` loop.

            Handles rank-deficient :math:`H^\top W H` gracefully in both
            modes by returning the minimum-norm least-squares solution
            (Moore–Penrose pseudoinverse).

        ``'solve'``
            Uses ``numpy.linalg.solve`` (LU factorisation) on the invertible
            subset and falls back to ``'lstsq'`` for rank-deficient blocks.
            Slightly faster than ``'lstsq'`` when :math:`H^\top W H` is always
            full-rank (the common case), but equally robust otherwise.  Does
            **not** cache a pseudoinverse.

        ``'inv'`` *(legacy)*
            Explicit matrix inversion — the original behaviour.  Raises an
            ``AssertionError`` (steady-state) or leaves entries as ``nan``
            (non-steady-state) when :math:`H^\top W H` is not invertible.
            Kept for backward compatibility and benchmarking.



        """

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
        if self._steady_state and not np.all(msk):
            if isinstance(self._cost_terms, CostSegment):
                print (f'condition number of W from {type(self._cost_terms.alssm)} too high; results of minimization may be meaningless. Try using AlssmPolyLegendre.')
            if isinstance(self._cost_terms, CompositeCost):
                print (f'condition number of W from {([type(item) for item in self._cost_terms._alssms])} too high; results of minimization may be meaningless. Try using AlssmPolyLegendre.')
        #     assert msk, 'H.T @ W @ H is not invertible.'
        #     np.einsum('nm, ...m-> ...n', inv(HTWH), HTxiWh, out=v)
        # else:
        #     v[msk] = np.einsum('...nm, ...m -> ...n', inv(HTWH[msk]), HTxiWh[msk])

        if solver == 'lstsq':
            if self._steady_state:
                # ----------------------------------------------------------
                # Steady-state: HTWH is a fixed (P, P) matrix.
                # Compute its SVD-based pseudoinverse once and cache it on
                # self.  The cached key encodes H and h so that a change in
                # constraint resets the cache automatically.
                #
                # The pseudoinverse is more accurate than explicit inversion
                # (error ~ u * kappa, vs u * kappa^2 for inv), and applying
                # it to all K right-hand sides via a single matrix multiply
                # is 4–22x faster than np.linalg.lstsq(HTWH, rhs) which
                # recomputes the SVD on every call.
                # ----------------------------------------------------------
                # The cache key encodes H, h, AND the current W (via self._xi2).
                # Including id(self._xi2) is essential: filter() always assigns
                # self._xi2 = cost.get_steady_state_W(...).flatten(), creating a
                # NEW array object on every call.  If W changes between filter()
                # calls (e.g. cost parameters were mutated), the new id forces a
                # recompute.  Without this, stale pinv(old_W) silently yields wrong xs.
                cache_key = (id(H), id(h), id(self._xi2))
                if getattr(self, '_pinv_cache_key', None) != cache_key:
                    U, s, Vt = np.linalg.svd(HTWH)
                    P = HTWH.shape[0]
                    rcond = np.finfo(float).eps * P * s[0]
                    s_inv = np.where(s > rcond, 1.0 / s, 0.0)
                    # pinv(HTWH) = V diag(s_inv) U^T
                    self._pinv_HTWH = (Vt.T * s_inv) @ U.T   # (P, P)
                    self._pinv_cache_key = cache_key

                # Apply: v[...] = pinv_HTWH @ HTxiWh  (batched over leading dims)
                v[...] = np.einsum('nm,...m->...n', self._pinv_HTWH, HTxiWh)

            else:
                # ----------------------------------------------------------
                # Non-steady-state: HTWH is batched, shape (..., P, P).
                # Fast path: batched LU solve (np.linalg.solve).  This is
                # 70x faster than a Python loop over samples and handles the
                # common well-conditioned case.
                # Fallback: if any matrix in the batch is singular, numpy
                # raises LinAlgError.  We catch that and switch to a fully
                # vectorised batched SVD pseudoinverse — still no Python loop.
                # ----------------------------------------------------------
                orig_shape = HTxiWh.shape
                P = orig_shape[-1]
                rhs_flat = HTxiWh.reshape(-1, P)           # (K, P)
                HTWH_flat = HTWH.reshape(-1, P, P)         # (K, P, P)

                try:
                    # batched solve: (K, P, P) @ (K, P, 1) -> (K, P)
                    sol = np.linalg.solve(
                        HTWH_flat, rhs_flat[..., np.newaxis])[..., 0]
                    v[...] = sol.reshape(orig_shape)
                except np.linalg.LinAlgError:
                    # Singular matrices in batch — use batched SVD pseudoinverse.
                    # np.linalg.svd supports (..., M, N) natively (no loop).
                    U, s, Vt = np.linalg.svd(HTWH_flat)
                    rcond = np.finfo(float).eps * P * s[..., :1]  # (K, 1)
                    # errstate suppresses the benign divide-by-zero that numpy
                    # raises when evaluating 1/s for the s==0 entries before
                    # np.where discards them.
                    with np.errstate(divide='ignore', invalid='ignore'):
                        s_inv = np.where(s > rcond, 1.0 / s, 0.0)    # (K, P)
                    # pinv[k] = Vt[k].T @ diag(s_inv[k]) @ U[k].T
                    pinv_batch = np.einsum(
                        '...ij,...j->...ij', Vt.transpose(0, 2, 1), s_inv
                    ) @ U.transpose(0, 2, 1)                      # (K, P, P)
                    sol = np.einsum('...nm,...m->...n', pinv_batch, rhs_flat)
                    v[...] = sol.reshape(orig_shape)

        elif solver == 'solve':
            # ----------------------------------------------------------------
            # LU-based solver.  Fast for full-rank systems; falls back to the
            # lstsq path for rank-deficient blocks.
            # ----------------------------------------------------------------
            msk = cond(HTWH) < 1 / sys.float_info.epsilon
            if self._steady_state:
                if msk:
                    rhs = HTxiWh.reshape(-1, HTxiWh.shape[-1]).T   # (P, K)
                    v[...] = np.linalg.solve(HTWH, rhs).T.reshape(HTxiWh.shape)
                else:
                    # Rank-deficient: fall through to lstsq path
                    return self.minimize_v(H=H, h=h, solver='lstsq')
            else:
                orig_shape = HTxiWh.shape
                P = orig_shape[-1]
                rhs_flat = HTxiWh.reshape(-1, P)
                HTWH_flat = HTWH.reshape(-1, P, P)
                v_flat = v.reshape(-1, P)
                if np.any(msk):
                    v_flat[msk] = np.linalg.solve(
                        HTWH_flat[msk],
                        rhs_flat[msk, np.newaxis].transpose(0, 2, 1)
                    )[:, :, 0]
                if np.any(~msk):
                    U, s, Vt = np.linalg.svd(HTWH_flat[~msk])
                    rcond = np.finfo(float).eps * P * s[..., :1]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        s_inv = np.where(s > rcond, 1.0 / s, 0.0)
                    pinv_batch = np.einsum(
                        '...ij,...j->...ij', Vt.transpose(0, 2, 1), s_inv
                    ) @ U.transpose(0, 2, 1)
                    v_flat[~msk] = np.einsum(
                        '...nm,...m->...n', pinv_batch, rhs_flat[~msk])
                v[...] = v_flat.reshape(orig_shape)

        elif solver == 'inv':
            # ----------------------------------------------------------------
            # Original explicit-inversion path (legacy).
            # ----------------------------------------------------------------
            msk = cond(HTWH) < 1 / sys.float_info.epsilon
            if self._steady_state:
                assert msk, 'condition number too high; H.T @ W @ H is not invertible. Try using AlssmPolyLegendre.'
                np.einsum('nm, ...m-> ...n', inv(HTWH), HTxiWh, out=v)
            else:
                v[msk] = np.einsum('...nm, ...m -> ...n', inv(HTWH[msk]), HTxiWh[msk])

        else:
            raise ValueError(
                f"Unknown solver '{solver}'. Choose from 'lstsq', 'solve', or 'inv'.")

        return v

    def minimize_x(self, H=None, h=None, solver='lstsq'):
        r"""
        Minimizes the cost :math:`J_k(x)` subject to a linear constraint on the state vector,
        returning the full N-dimensional state vector :math:`x`.

        Internally calls :meth:`minimize_v` to obtain :math:`\hat{v}_k`, then reconstructs
        the state vector via the constraint (Eq. (66) [Wildhaber2018]_):

        .. math::
            \hat{x}_k = H \hat{v}_k + h.

        When ``H=None`` and ``h=None`` (defaults), the constraint reduces to :math:`x = v`,
        so :math:`\hat{x}_k = W_k^{-1} \xi_k` (Table V row 1, [Wildhaber2018]_).

        Parameters
        ----------
        H : array_like of shape (N, M), optional
            Constraint matrix mapping the free parameter :math:`v \in \mathbb{R}^M` to the
            state space :math:`\mathbb{R}^N`. If ``None``, defaults to the identity matrix
            :math:`I_N` (unconstrained, :math:`M = N`).
        h : array_like of shape (N,), optional
            Constraint offset vector :math:`h \in \mathbb{R}^N`. If ``None``, defaults to
            the zero vector (no offset).

        Notes
        -----
        The distinction between :meth:`minimize_v` and :meth:`minimize_x` is:

        - :meth:`minimize_v` returns :math:`v \in \mathbb{R}^M` (the free parameter, dimension M).
        - :meth:`minimize_x` returns :math:`x = Hv + h \in \mathbb{R}^N` (the full state vector,
          dimension N).

        When ``H=None``, both methods return the same result since :math:`x = v`.

        Returns
        -------
        x : :class:`~numpy.ndarray` of shape (..., N)
            Optimal state vector :math:`\hat{x}_k = H\hat{v}_k + h`. The leading dimensions
            match the signal length and any parallel dimensions of the input.
            Entries are ``NaN`` where :math:`H^T W_k H` is ill-conditioned
            (only possible when ``steady_state=False``).

        Raises
        ------
        AssertionError
            Propagated from :meth:`minimize_v`: 
            If ``H`` does not have ``N`` rows.
            If ``h`` does not have length ``N``.
            If ``H.T @ W @ H`` is not invertible (only when ``steady_state=True``).
        """

        v = self.minimize_v(H, h, solver=solver)

        if H is None:
            x = v
        else:
            x = np.einsum('nm, ...m-> ...n', H, v)

        if h is not None:
            x += h

        return x

    def eval_errors(self, xs):
        r"""
        Evaluates the cost function :math:`J_k(x)` at given state vectors `xs`.

        Using the expanded form of the cost (Eq. (21) [Wildhaber2018]_):

        .. math::
            J_k(x) = x^T W_k x - 2 x^T \xi_k + \kappa_k,

        this method computes the scalar cost for each provided state vector without
        performing any minimization. It is useful for comparing the fit quality of
        different candidate state vectors, e.g. for computing error ratios or
        log-cost ratios (LCR) as in Section III.F [Wildhaber2018]_.

        Requires :attr:`W`, :attr:`xi`, and :attr:`kappa` to be available, i.e.
        :meth:`filter` must have been called with ``calc_W=True``, ``calc_xi=True``,
        and ``calc_kappa=True`` (all defaults).

        Parameters
        ----------
        xs : array_like of shape (..., N)
            State vector(s) at which to evaluate the cost. The last dimension must
            be the state dimension N. Leading dimensions are broadcast over the
            signal samples.

        Returns
        -------
        J : :class:`~numpy.ndarray` of shape (...)
            Cost :math:`J_k(x)` evaluated at each state vector in `xs`. Same shape
            as `xs` with the last axis removed.
        """

        if self._steady_state:
            J = np.einsum('...n, ...n', xs, np.einsum('nm, ...m->...n', self.W, xs))
        else:
            J = np.einsum('...n, ...n', xs, np.einsum('...nm, ...m->...n', self.W, xs))

        return J - 2 * np.einsum('...n, ...n', self.xi, xs) + self.kappa

    def fit(self, y, output='y_hat', sample_weights=None, dim_order=None, H=None, h=None, eval_alssm_weights=None, solver='lstsq'):
        r"""
        Method that chains :meth:`filter`, :meth:`minimize_v`, and signal
        reconstruction into a single call.

        Executes the following steps in order:

        1. :meth:`filter` — computes the recursive filter quantities :math:`W_k`,
           :math:`\xi_k`, :math:`\kappa_k` from the input signal `y`.
        2. :meth:`minimize_v` — solves the constrained minimization
           :math:`\hat{v}_k = (H^T W_k H)^{-1} H^T (\xi_k - W_k h)`.
        3. Reconstruction — builds :math:`\hat{x}_k = H\hat{v}_k + h` and/or the
           signal estimate :math:`\hat{y}_k = C A^j \hat{x}_k` depending on `output`.

        Parameters
        ----------
        y : array_like of shape (K, [Q])
            Input signal. See :meth:`filter` for shape details.
        output : str or tuple of str, optional
            Selects what is returned. One or more of:

            - ``'y_hat'`` *(default)* — signal estimate :math:`\hat{y}_k = CA^j\hat{x}_k`
              evaluated via :class:`~lmlib.statespace.model.AlssmSum`.
            - ``'x'`` — full state vector :math:`\hat{x}_k = H\hat{v}_k + h`,
              shape ``(..., N)``.
            - ``'v'`` — free parameter :math:`\hat{v}_k`, shape ``(..., M)``.

            Pass a tuple to return multiple outputs, e.g. ``output=('x', 'y_hat')``.
        sample_weights : array_like of shape (K,), optional
            Per-sample weights :math:`w_i \in [0,1]`. Passed to :meth:`filter`.
            Default: all ones.
        dim_order : array_like of int, optional
            Dimension processing order for :class:`~lmlib.statespace.cost.NDCompositeCost`.
            Passed to :meth:`filter`. Default: ``np.arange(L)``.
        H : array_like of shape (N, M), optional
            Constraint matrix. Passed to :meth:`minimize_v`. Default: identity (unconstrained).
        h : array_like of shape (N,), optional
            Constraint offset. Passed to :meth:`minimize_v`. Default: zero vector.
        eval_alssm_weights : array_like, optional
            Per-ALSSM output weights used when evaluating :math:`\hat{y}` from a
            :class:`~lmlib.statespace.cost.CompositeCost` with multiple models.
            Passed to :class:`~lmlib.statespace.model.AlssmSum`. If ``None``, all
            models contribute equally.

        Returns
        -------
        result : ndarray or tuple of ndarray
            If `output` is a single string, returns the corresponding array directly.
            If `output` is a tuple, returns a tuple of arrays in the same order.

            - ``'y_hat'``: shape ``(K, [Q])`` — signal estimate at every time step.
            - ``'x'``: shape ``(..., N)`` — optimal state vector.
            - ``'v'``: shape ``(..., M)`` — optimal free parameter.

        Raises
        ------
        AssertionError
            If `output` is empty or contains unknown entries.
            Propagated from :meth:`minimize_v` if ``H`` / ``h`` have wrong shape or
            ``H.T @ W @ H`` is not invertible (when ``steady_state=True``).

        Examples
        --------
        Fit a degree-2 polynomial and return the signal estimate:

        >>> import numpy as np
        >>> import lmlib as lm
        >>> K = 200
        >>> y = np.random.randn(K)
        >>> alssm = lm.AlssmPoly(poly_degree=2)
        >>> seg = lm.Segment(a=-20, b=0, direction=lm.FORWARD, g=10)
        >>> rls = lm.RLSAlssm(lm.CostSegment(alssm, seg))
        >>> y_hat = rls.fit(y)

        Return both the state vector and the signal estimate:

        >>> x, y_hat = rls.fit(y, output=('x', 'y_hat'))
        """

        # ----------- check output parameter -----------
        if isinstance(output, str):
            _output = (output,)
        else:
            _output = tuple(output)
        assert len(_output) != 0, 'output is empty. Must be a string or a tuple of strings.'
        assert any(_ in ('y_hat', 'x', 'v') for _ in _output), (f'output contains unknown entries: {_output}'
                                                                 f'. Allowed entries are "y_hat", "x", "v".')
        self.filter(y, sample_weights, dim_order)

        v = self.minimize_v(H, h, solver=solver)
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
        r"""
        Defines the recursion to calculate the ALSSM cost parameters :math:`\xi^{(q)}(k,y)` for a given :math:`q \in \{0,1,2\}` based on an input signal :math:`y`.

        The cost parameters :math:`\xi^{(q)}(k,y)` for :math:`q \in \{0,1,2\}` are equivalent to :math:`\kappa_k`, :math:`\xi_k` and :math:`W_k`. 
        They are calculated through the recursive equations (22-25) [Wildhaber2018]_ for each cost segment defined by the different backends.
        We iterate over each individual ALSSM m within every segment p and write the result into the appropriate sub-slice of xi_curr.  
        Because A is block-diagonal, the blocks are independent and can be computed separately. 
        Each ALSSM m is wrapped in a single-element AlssmSum with weight F[m,p] so that:
          - force_MC is honoured (C is guaranteed 2-D for all backends), and
          - the F-column weight is folded into C (as lambda), keeping beta equal to the original segment beta.

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

        # iterate over CostSegments
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
            # are present.  We pass the first ALSSM as a dummy placeholder and
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
                        beta_p, self._backend, self._filter_form, numdenom_pm, self._cascade_params[dim_index][p][m]
                    )

        return xi_curr

    # ------------------------------------------------------------------
    # Recursion: _nd_xi_q_asterisk_l_recursion
    # ------------------------------------------------------------------

    def _nd_xi_q_asterisk_l_recursion(self, xi_prev, q, y, sample_weights, model_dimension):
        r"""
        Defines the recursion for one additional dimension to calculate the ALSSM cost parameters :math:`\xi^{(q)*}(k,y)` for a given :math:`q \in \{0,1,2\}` based on an input signal :math:`y`.

        The cost parameters :math:`\xi^{(q)}(k,y)` for :math:`q \in \{0,1,2\}` are equivalent to :math:`\kappa_k`, :math:`\xi_k` and :math:`W_k`. 
        They are calculated through the recursive equations (22-25) [Wildhaber2018]_ for each cost segment defined by the different backends.
        This function subdivides the calculation into cost segments (for-loop). The same per-ALSSM decomposition as _nd_xi_q_recursion is applied:
        each ALSSM m in the CompositeCost for this dimension is processed independently, writing into the corresponding sub-slice of xi_curr.

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
        sub_cost = _as_composite_cost(
            self._cost_terms._get_sub_cost_term(model_dimension)
        )

        dim_index = model_dimension if isinstance(self._cost_terms, NDCompositeCost) else 0

        N = sub_cost.get_alssm_order()
        Nq_prev = xi_prev.shape[-1]
        *Ks, Q = np.shape(y)
        xi_curr = np.zeros((*Ks, Nq_prev * N ** q), order='F')

        # move subarray to first dimensions (returns a view)
        _xi_curr = np.moveaxis(xi_curr, model_dimension, 0)
        _xi_prev = np.moveaxis(xi_prev, model_dimension, 0)
        _sample_weights = np.moveaxis(sample_weights, model_dimension, 0)

        offsets = self._alssm_offsets(sub_cost)

        # iterate over CostSegments
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
                        beta_p, self._backend, self._filter_form, None, None
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
