import sys
import numpy as np
from numpy.linalg import inv, cond

from lmlib.statespace.backend import get_backend
from lmlib.statespace.cost import CompositeCost, CostSegment, NDCompositeCost
from lmlib.statespace.model import AlssmSum
from lmlib.utils.check import *
from lmlib.statespace.backends.rec import *
from lmlib._warnings import WConditionNumberWarning
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


def _cost_parts(cost): #TODO remove this function and directly insert the code at the corresponding calls.
    """
    Return ``(segments, alssms, F, betas)`` for a CostSegment or CompositeCost.

    A CostSegment is the degenerate single-ALSSM / single-segment case; it is
    unpacked here directly (``F = [[1]]``) so the recursion drivers never branch
    on the concrete cost type and never have to build a throw-away CompositeCost.
    """
    if isinstance(cost, CostSegment):
        return [cost.segment], [cost.alssm], np.ones((1, 1)), np.array([cost.beta])
    return list(cost.segments), list(cost.alssms), cost.F, cost.betas


class RLSAlssm:
    r"""
    Recursive Least Square Alssm Class to solve Alssm Cost Functions.

    This class uses either a :class:`CostSegment`, :class:`CompositeCost` or :class:`NDCompositeCost` and defines the functions to solve it recursively.
    :math:`W_k`, :math:`\xi_k`, :math:`\kappa_k` and :math:`\nu_k` can be computed either as a forward or as a backward recursion as defined in Eq. (22-25) [Wildhaber2018]_.
    :math:`W_k` is the Gram matrix defined by the ALSSM (:math:`A`, :math:`c`, :math:`\alpha` and :math:`w`). :math:`\mathrm{vec}(W_k)=\xi^{(2)}(k,y)=\xi^{(2)}(k,\mathbf{1})` (with :math:`\mathbf{1}` the all ones vector) Eq. (21) [Baeriswyl2025]_.
    :math:`\xi_k` (:math:`=\xi^{(1)}(k,y)` Eq. (20) [Baeriswyl2025]_) is the cross correlation of the signal with the ALSSM basis.
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
        If True, computes the signal cross correlation :math:`\xi_k = \xi^{(1)}(k, y)`.
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
                 backend=None, steady_state_method='schur'):
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

        # Per-dimension cost terms (CostSegment or CompositeCost), one per ND axis.
        #   - NDCompositeCost._get_sub_cost_term() returns a list of costs.
        #   - CompositeCost/CostSegment._get_sub_cost_term() returns the cost itself.
        raw = cost_terms._get_sub_cost_term()
        _sub_costs = raw if isinstance(raw, list) else [raw]

        # ------------------------------------------------------------------
        # check if filter form is valid for all segments
        # ------------------------------------------------------------------
        if self._backend == 'lfilter' and self._filter_form == 'cascade':
            for ct in _sub_costs:
                _, _alssms, _, _ = _cost_parts(ct)
                for alssm in _alssms:
                    if not np.allclose(alssm.A, np.triu(alssm.A)):
                        self._filter_form = 'parallel'
                        print("State-Space Matrix A is not upper triangular, "
                              "cascade version can't be used. "
                              "Defaulting to filter_form='parallel'.")

        # The parallel xi^(1) realization needs per-ALSSM transfer functions.
        # Build them once here (the heavy SOS/QZ construction lives in the
        # backend, build_parallel_numdenom); cascade/numpy/jit need nothing.
        # Layout: _parallel_plan[dim_index][seg_index] -> list of (n0, n1, numdenom).
        self._parallel_plan = self._build_parallel_plan(_sub_costs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def xi2(self):
        """ndarray : Alias for :attr:`W` (:math:`\\xi^{(2)}`, the Gram matrix)."""
        return self.W

    @property
    def xi1(self):
        """ndarray : Alias for :attr:`xi` (:math:`\\xi^{(1)}`, the signal cross correlation)."""
        return self.xi

    @property
    def xi0(self):
        """ndarray : Alias for :attr:`kappa` (:math:`\\xi^{(0)}`, the signal energy)."""
        return self.kappa

    @property
    def W(self):
        """:class:`~numpy.ndarray` : Gram matrix :math:`W_k` (constant :math:`W` in steady-state mode). Shape ``(..., N, N)``."""
        if self._xi2 is None:
            raise ValueError('xi2 has not been calculated. '
                             'Please run the filter() method with calc_W=True before calling W.')
        return self._xi2.reshape(self._xi2.shape[:-1] + (self._N, self._N))

    @property
    def xi(self):
        """:class:`~numpy.ndarray` : Signal cross-correlation :math:`\\xi_k = \\xi^{(1)}(k,y)`. Shape ``(..., N)``."""
        if self._xi1 is None:
            raise ValueError('xi1 has not been calculated. '
                             'Please run the filter() method with calc_xi=True before calling xi.')
        return self._xi1

    @property
    def kappa(self):
        """:class:`~numpy.ndarray` : Signal energy :math:`\\kappa_k = \\xi^{(0)}(k,y)`. Shape ``(...,)``."""
        if self._xi0 is None:
            raise ValueError('xi0 has not been calculated. '
                             'Please run the filter() method with calc_kappa=True before calling kappa.')
        return self._xi0

    @property
    def nu(self):
        """:class:`~numpy.ndarray` : Effective number of weighted samples :math:`\\nu_k` in the window. Not yet implemented."""
        # TODO nu implementation
        raise NotImplementedError("nu calculation is not yet implemented.")

    def _build_parallel_plan(self, _sub_costs):
        r"""
        Pre-build per-ALSSM transfer-function coefficients for the parallel
        :math:`\xi^{(1)}` realization (lfilter parallel backend only).

        Returns a 3-D structure ``plan[dim_index][seg_index]`` where each leaf is
        the list of ``(n0, n1, numdenom)`` tuples produced by
        :func:`~lmlib.statespace.backends.rec_lfilter.build_parallel_numdenom`
        (one entry per active ALSSM block).  Returns ``None`` for every other
        backend/filter-form combination (nothing to pre-build).

        The combined block-diagonal ``A`` / block-partitioned ``C`` are built
        once per segment here; the backend slices and decomposes each block.
        """
        if not (self._filter_form == 'parallel' and self._backend == 'lfilter'):
            return None

        from lmlib.statespace.backends.rec_lfilter import build_parallel_numdenom

        plan = []
        for ct in _sub_costs:
            segments, alssms, F, _ = _cost_parts(ct)
            block_sizes = [al.N for al in alssms]
            seg_plans = []
            for p, segment in enumerate(segments):
                combined = AlssmSum(alssms, F[:, p], force_MC=True)
                seg_plans.append(build_parallel_numdenom(
                    combined.A, combined.C,
                    segment.a, segment.b, segment.delta, segment.gamma,
                    segment.direction, block_sizes))
            plan.append(seg_plans)
        return plan



    # ------------------------------------------------------------------
    # filter
    # ------------------------------------------------------------------

    def filter(self, y, sample_weights=None, dim_order=None):
        r"""
        Compute the three ALSSM cost parameters from the input signal :math:`y`.

        The cost function at signal index :math:`k` is:

        .. math::

            J_k(x) = x^\top W_k x - 2\,x^\top \xi_k + \kappa_k

        where :math:`W_k`, :math:`\xi_k`, and :math:`\kappa_k` are computed here as
        the three :math:`\xi^{(q)}` terms for :math:`q \in \{0, 1, 2\}`:

        .. list-table::
            :header-rows: 1
            :widths: 8 20 55

            * - :math:`q`
              - Symbol
              - Meaning
            * - 0
              - :math:`\kappa_k = \xi^{(0)}_k`
              - Weighted signal energy :math:`\sum_i \alpha^{k+\delta}(i)\,w_i\,\|y_i\|^2`
            * - 1
              - :math:`\xi_k = \xi^{(1)}_k`
              - Cross-correlation of :math:`y` with the ALSSM basis; the right-hand
                side of the normal equations :math:`W_k x = \xi_k`
            * - 2
              - :math:`W_k = \xi^{(2)}_k`
              - Gram matrix (independent of :math:`y` in the LTI / steady-state case)

        For :class:`CostSegment` and :class:`CompositeCost` (1-D) each quantity is
        computed via the recursive equations (22–25) in [Wildhaber2018]_ by the
        selected backend.

        For :class:`NDCompositeCost` (multi-dimensional) the ND cost separates as a
        Kronecker product over the per-dimension sub-costs.  The computation is
        therefore chained over dimensions:

        1. :meth:`_nd_xi_q_recursion` processes the first axis in ``dim_order``,
           reading directly from ``y`` and producing ``xi_prev`` of shape
           ``(*Ks, N_0**q)``.
        2. For each subsequent axis ``dim_order[l]``,
           :meth:`_nd_xi_q_asterisk_l_recursion` extends the trailing axis of
           ``xi_prev`` from ``Nq_prev`` to ``Nq_prev * N_l**q``, accumulating the
           Kronecker structure dimension by dimension.
        3. After all ``L`` axes are processed the trailing axis has size
           :math:`(N_0 \cdots N_{L-1})^q = N_\text{total}^q`.

        The final results are stored in :attr:`W` (:math:`\xi^{(2)}`),
        :attr:`xi` (:math:`\xi^{(1)}`), and :attr:`kappa` (:math:`\xi^{(0)}`).

        Parameters
        ----------
        y : array_like
            Input signal.

            * **1-D / CompositeCost / CostSegment** — shape ``(K,)`` or ``(K, Q)``.
              ``K`` is the signal length; ``Q`` is the ALSSM output dimension.
              For scalar ALSSMs (``Q = 0`` internally) a plain 1-D array ``(K,)``
              is accepted and silently reshaped to ``(K, 1)``.
            * **ND / NDCompositeCost** — shape ``(K_0, K_1, ..., K_{L-1})`` for
              scalar ALSSMs (``Q = 0``), or ``(K_0, ..., K_{L-1}, Q)`` for
              vector-output ALSSMs.  There must be exactly ``L`` spatial axes,
              one per sub-cost, regardless of the processing order given by
              ``dim_order``.

        sample_weights : array_like of shape matching ``y`` without the trailing
            ``Q`` axis, optional.
            Per-sample scalar weights :math:`w_i \in [0, 1]` applied to each
            observation before accumulation.  Must broadcast to the spatial shape
            of ``y``.  Default: all ones (uniform weighting).

        dim_order : array_like of int of length ``L``, optional
            Processing order of the signal axes for :class:`NDCompositeCost`.

            ``dim_order[0]`` is processed first by :meth:`_nd_xi_q_recursion`;
            ``dim_order[1], dim_order[2], ...`` are each processed in turn by
            :meth:`_nd_xi_q_asterisk_l_recursion`.

            The Kronecker structure of the output state vector follows this order:
            with ``dim_order = [d_0, d_1, ..., d_{L-1}]``, the trailing axis of
            the minimiser result ``xs`` (see :meth:`minimize_x`) is arranged as:

            .. math::

                x_{d_0} \otimes x_{d_1} \otimes \cdots \otimes x_{d_{L-1}}

            where :math:`x_{d_l}` is the per-dimension state vector for axis
            :math:`d_l`.  Changing ``dim_order`` therefore permutes the Kronecker
            blocks in ``xs``, but does not change the scalar cost :math:`J_k(x)`.

            Has no effect for :class:`CostSegment` or :class:`CompositeCost`
            (those are inherently 1-D).  Default: ``np.arange(L)`` (axes in
            natural order).

        Notes
        -----
        **Steady-state mode** (``steady_state=True``, the default):
            :math:`W_k` is constant and precomputed once via
            :meth:`~lmlib.CompositeCost.get_steady_state_W` before the signal
            recursion starts.  This is valid whenever the window is
            time-invariant (:math:`w_k = w`, :math:`\gamma_k = \gamma`), which
            holds for all standard :class:`Segment` definitions without
            sample-dependent weights.  See Section III-I.2 in [Wildhaber2018]_.
            The ``calc_W`` flag is ignored in this mode.

        **Time-varying mode** (``steady_state=False``):
            :math:`W_k` is computed sample-by-sample via the :math:`q = 2`
            recursion, which carries the same computational cost as the
            :math:`q = 1` recursion.  Required near signal boundaries or when
            ``sample_weights`` varies over time.  Only supported for 1-D costs;
            multi-dimensional costs require ``steady_state=True`` (the ND
            :math:`W` recursion is not yet implemented).

        **ND Kronecker product structure**:
            For an :class:`NDCompositeCost` with sub-costs of orders
            :math:`N_0, N_1, \ldots, N_{L-1}`:

            * ``xi`` (shape ``(*Ks, N_0 \cdots N_{L-1})``) is the cross-correlation
              vector arranged as a Kronecker-product of the per-dimension basis
              vectors, consistent with a separable cost function over all dimensions.
            * ``W`` (steady-state, shape ``(N_\text{total}^2,)`` flattened then
              reshaped to ``(N_\text{total}, N_\text{total})`` by :attr:`W`) is the
              Kronecker product of the per-dimension Gram matrices.
            * The minimiser ``xs = W^{-1} \xi`` therefore recovers the full
              multi-dimensional Kronecker-product state at every sample location.

        Returns
        -------
        None
            Results are stored in-place.  Access via:

            * :attr:`W`     — Gram matrix :math:`W_k` or :math:`W`
              (shape ``(N, N)`` steady-state, or ``(*Ks, N, N)`` time-varying)
            * :attr:`xi`    — cross-correlation :math:`\xi_k`
              (shape ``(*Ks, N)``)
            * :attr:`kappa` — signal energy :math:`\kappa_k`
              (shape ``(*Ks,)``)

            where :math:`N = N_0 \cdots N_{L-1}` is the total model order.

        Raises
        ------
        ValueError
            If ``y`` does not have the expected number of spatial axes or output
            dimension ``Q``.
            If ``sample_weights`` does not match the spatial shape of ``y``.
        AssertionError
            If ``dim_order`` does not have exactly ``L`` elements.
            If ``steady_state=False`` and ``L > 1`` (ND time-varying W not supported).
        """

        # ── Resolve and validate dim_order ────────────────────────────────────
        # L == 1 for CostSegment / CompositeCost; L >= 2 for NDCompositeCost.
        # dim_order controls which signal axis is processed at each chain step
        # and therefore the Kronecker ordering of the output state vector.
        L = self._cost_terms.get_number_of_dimensions()
        if dim_order is None:
            dim_order = np.arange(L)
        assert len(dim_order) == L, f'dim_order has wrong length, {info_str_found_shape(dim_order)}'

        # Time-varying W for ND costs is not implemented: the asterisk recursion
        # for q=2 would need to handle xi_prev correctly across dimensions, which
        # requires the same Kronecker chain used for steady-state W.
        if L > 1 and self._calc_W and not self._steady_state:
            assert False, "for multidimensional ALSSMs, W requires steady_state=True"

        # ── Normalise y shape ─────────────────────────────────────────────────
        # All backends expect y with an explicit trailing output dimension Q,
        # even for scalar ALSSMs (Q=0 → Q treated as 1 after reshape).
        # The spatial shape (*Ks,) is preserved; sample_weights must match it.
        Q = self._cost_terms.get_alssm_output_dimension()
        y = np.asarray(y)
        if isinstance(self._cost_terms, (CompositeCost, CostSegment)):
            if Q == 0:  # scalar output: ensure trailing singleton axis
                if y.ndim == 1:  # 1-D signal (K,) → (K, 1)
                    y = y.reshape(-1, 1)
                elif y.ndim >= 2:
                    if y.shape[1] == 1:
                        pass  # already (K, 1)
                    elif y.shape[-1] != 1:  # parallel signals (K, ...) → (K, ..., 1)
                        y = y.reshape(*y.shape, 1)
                    else:
                        raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')
                else:
                    raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')
            elif Q == 1:  # explicit 1-D output: must already have trailing axis
                if y.ndim == 1 or y.shape[-1] != Q:
                    raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')
            else:  # vector output: trailing axis must match Q
                if y.shape[-1] != Q:
                    raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')

        if isinstance(self._cost_terms, NDCompositeCost):
            # NDCompositeCost: y must have exactly L spatial axes.
            # For scalar output (Q=0) y has shape (K_0, ..., K_{L-1}); append
            # a trailing singleton so all backends see a uniform (..., Q) shape.
            if Q == 0:
                if y.ndim == L:
                    y = y.reshape(*y.shape, 1)
                else:
                    raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')
            if 1 <= Q != y.shape[-1]:
                raise ValueError(f'y has wrong dimension, {info_str_found_shape(y)}')

        # ── Validate and broadcast sample_weights ────────────────────────────
        # sample_weights shape must equal the spatial shape of y, i.e. y.shape[:-1].
        # Using broadcast_to for the default (all-ones) avoids allocating a
        # full array when weights are not needed.
        if sample_weights is None:
            sample_weights = np.broadcast_to(1., y.shape[:-1])
        else:
            if np.shape(sample_weights) != y.shape[:-1]:
                raise ValueError(f'sample_weights has wrong shape, {info_str_found_shape(sample_weights)}')

        # ── Compute W (xi^(2), Gram matrix) ──────────────────────────────────
        # Steady-state: W is time-invariant, precomputed once from the segment
        # parameters via a Stein/Lyapunov equation.  Result is stored flattened
        # as a 1-D array; the W property reshapes it to (N, N) on access.
        #
        # Time-varying: W_k is computed sample-by-sample via the q=2 recursion
        # chain, using the same _nd_xi_q_recursion / _nd_xi_q_asterisk_l_recursion
        # machinery as q=1.  Only supported for L=1.
        if self._steady_state:
            self._xi2 = self._cost_terms.get_steady_state_W(dim_order, method=self._steady_state_method).flatten()
        elif self._calc_W and not self._steady_state:
            q = 2
            xi_prev = self._nd_xi_q_recursion(q, y, sample_weights, dim_order[0])
            for nd_dim in dim_order[1:]:
                xi_prev = self._nd_xi_q_asterisk_l_recursion(xi_prev, q, y, sample_weights, nd_dim)
            self._xi2 = xi_prev

        # ── Compute xi (xi^(1), cross-correlation vector) ────────────────────
        # xi_k = sum_i alpha(i) w_i (C A^{i-k})^T y_i  (vector, shape (*Ks, N))
        #
        # For ND costs with L dimensions the computation is chained:
        #   Step 0 (dim_order[0]):   xi_prev shape  (*Ks, N_0)
        #   Step 1 (dim_order[1]):   xi_prev shape  (*Ks, N_0 * N_1)
        #   ...
        #   Step L-1 (dim_order[L-1]): xi_prev shape (*Ks, N_0 * ... * N_{L-1})
        #
        # The resulting Kronecker ordering of the trailing axis matches dim_order:
        #   xi[k] = xi_{d_0}(k) ⊗ xi_{d_1}(k) ⊗ ... ⊗ xi_{d_{L-1}}(k)
        if self._calc_xi:
            q = 1
            xi_prev = self._nd_xi_q_recursion(q, y, sample_weights, dim_order[0])
            for nd_dim in dim_order[1:]:
                xi_prev = self._nd_xi_q_asterisk_l_recursion(xi_prev, q, y, sample_weights, nd_dim)
            self._xi1 = xi_prev

        # ── Compute kappa (xi^(0), signal energy) ────────────────────────────
        # kappa_k = sum_i alpha(i) w_i ||y_i||^2  (scalar per sample, shape (*Ks,))
        #
        # kappa is independent of the ALSSM model (no A or C involved); it is
        # the same chain call as for q=1 but the ALSSM is used only as a
        # structural placeholder.  The final [..., 0] index extracts the scalar
        # from the trailing singleton axis that the q=0 recursion produces.
        if self._calc_kappa:
            q = 0
            xi_prev = self._nd_xi_q_recursion(q, y, sample_weights, dim_order[0])
            for nd_dim in dim_order[1:]:
                xi_prev = self._nd_xi_q_asterisk_l_recursion(xi_prev, q, y, sample_weights, nd_dim)
            self._xi0 = xi_prev[..., 0]

        # ── nu (number of weighted samples) ──────────────────────────────────
        # TODO: not yet implemented.

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
                warnings.warn(
                    f'condition number of W from {type(self._cost_terms.alssm)} too high; results of minimization may be meaningless.',
                    WConditionNumberWarning, stacklevel=3,
                )
            if isinstance(self._cost_terms, CompositeCost):
                warnings.warn(
                    f'condition number of W from {([type(item) for item in self._cost_terms._alssms])} too high; results of minimization may be meaningless.',
                    WConditionNumberWarning, stacklevel=3,
                )
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

    # ------------------------------------------------------------------
    # Recursion: _nd_xi_q_recursion (first dimension)
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
        segments, alssms, F, betas = _cost_parts(
            self._cost_terms._get_sub_cost_term(model_dimension))
        dim_index = model_dimension if isinstance(self._cost_terms, NDCompositeCost) else 0
        N = sum(al.N for al in alssms)
        block_sizes = [al.N for al in alssms]

        *Ks, Q = np.shape(y)
        xi_curr = np.zeros((*Ks, N ** q))  # C-order; last dimension is the nd-model-order

        # Move model_dimension next to the trailing axis and flatten the other
        # signal axes into a batch.  moveaxis can yield a non-contiguous view for
        # which reshape silently returns a copy, so work on an explicit copy and
        # write the result back through the view.
        _xi_view = np.moveaxis(xi_curr, model_dimension, -2)
        _xi_work = np.ascontiguousarray(_xi_view)
        _xi_curr = np.reshape(_xi_work, (-1, *_xi_work.shape[-2:]))
        _ym = np.ascontiguousarray(np.moveaxis(y, model_dimension, -2))
        _y = np.reshape(_ym, (-1, *_ym.shape[-2:]))
        _swm = np.ascontiguousarray(np.moveaxis(sample_weights, model_dimension, -1))
        _sw = np.reshape(_swm, (-1, *_swm.shape[-1:]))

        for p, segment in enumerate(segments):
            combined = AlssmSum(alssms, F[:, p], force_MC=True)
            plan = (self._parallel_plan[dim_index][p]
                    if self._parallel_plan is not None else None)
            for i in range(_y.shape[0]):
                xi_q_recursion(
                    _xi_curr[i], q, combined, segment,
                    _y[i], _sw[i], betas[p],
                    self._backend, self._filter_form, block_sizes, plan)

        _xi_view[:] = np.reshape(_xi_curr, _xi_view.shape)
        return xi_curr

    # ------------------------------------------------------------------
    # Recursion: _nd_xi_q_asterisk_l_recursion (subsequent dimensions)
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
            Input signal (only its shape is used; values enter via ``xi_prev``).
        sample_weights : array_like of shape (\*Ks,)
            Per-sample weights :math:`w_i \in [0,1]`.
        model_dimension : int
            Signal axis processed in this step.

        Returns
        -------
        xi_curr : :class:`~numpy.ndarray` of shape (\*Ks, Nq_prev \* N_l**q)
            Extended cost parameter. The leading shape matches `y`; the trailing axis
            is the product of all accumulated sub-cost orders raised to :math:`q`.
        """
        segments, alssms, F, betas = _cost_parts(
            self._cost_terms._get_sub_cost_term(model_dimension))
        N = sum(al.N for al in alssms)
        block_sizes = [al.N for al in alssms]
        Nq_prev = xi_prev.shape[-1]

        *Ks, Q = np.shape(y)
        xi_curr = np.zeros((*Ks, Nq_prev * N ** q), order='F')

        # move the processed axis to the front (views); the backend handles the
        # remaining leading axes as a batch.
        _xi_curr = np.moveaxis(xi_curr, model_dimension, 0)
        _xi_prev = np.moveaxis(xi_prev, model_dimension, 0)
        _sample_weights = np.moveaxis(sample_weights, model_dimension, 0)

        for p, segment in enumerate(segments):
            combined = AlssmSum(alssms, F[:, p], force_MC=True)
            xi_q_asterisk_l_recursion(
                _xi_curr, q, combined, segment,
                _xi_prev, _sample_weights, betas[p],
                self._backend, self._filter_form, block_sizes)

        return xi_curr
