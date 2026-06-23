import sys
import numpy as np
from numpy.linalg import inv, cond

from lmlib.statespace.backend import get_backend, available_backends
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


class RLSAlssm:
    r"""
    Recursive Least Square Alssm Class to compute and minimize Alssm Cost Functions.

    This class uses either a [`CostSegment`][lmlib.statespace.cost.CostSegment], [`CompositeCost`][lmlib.statespace.cost.CompositeCost] or [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost].

    The cost function is (Eq. (20) [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018)):

    $$\begin{aligned}
    J_k(x) &= \sum_{i=k+a}^{k+b} \alpha^{k+\delta}(i) w_i \big(y_i - CA^{i-k}x\big)^2 \\
           &= x^\top W_k x - 2\,x^\top \xi_k + \kappa_k
    \end{aligned}
    $$

    with $W_k \in \mathbb{R}^{N \times N}$ the Gram Matrix defined by the ALSMS ($A$, $c$, $\alpha$ and $w$),
    $\xi_k \in \mathbb{R}^{N}$ the cross correlation of the signal with the ALSSM basis, and
    $\kappa_k \in \mathbb{R}$ the signal energy weighted under $\alpha$ and $w$.
    Additionally, $\nu_k$ is the number of weighted samples in the window.

    The quantities $W_k$, $\xi_k$, $\kappa_k$ and $\nu_k$ can be computed either as a forward or as a backward recursion as defined in Eq. (22-25) [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018).

    !!! info
        See also [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018) [\[Wildhaber2019\]](../../bibliography.md#wildhaber2019).
        Eq. (19)-(21) in [\[Baeriswyl2025\]](../../bibliography.md#baeriswyl2025) define the quantities $\xi^{(q)}(k,y), q\in\{0,1,2\}$, with the following correspondence:
        $\mathrm{vec}(W_k) \triangleq \xi^{(2)}(k,\mathbf{1})$ (with $\mathbf{1}$ the all ones vector),
        $\xi_k \triangleq \xi^{(1)}(k,y)$, and
        $\kappa_k \triangleq \xi^{(0)}(k,y)$.

    Parameters
    ----------
    cost_terms : CostSegment, CompositeCost, NDCompositeCost
        Cost function to be minimized recursively. See [`CostSegment`][lmlib.statespace.cost.CostSegment], [`CompositeCost`][lmlib.statespace.cost.CompositeCost] or [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost].
    steady_state : bool, optional
        Defines if the ALSSM is steady state (not time-varying, e.g. LTI). If so, $W_k$ reduces to $W$. Default: True.
        This happens in case $w_k = w$, $\gamma_k = \gamma$ (see Sec. III-I.2 [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018)).
        Setting this incorrectly may produce silently wrong results.
    calc_W : bool, optional
        If True, compute [`W`][lmlib.statespace.rls.RLSAlssm.W].
        Required for [`minimize_v`][lmlib.statespace.rls.RLSAlssm.minimize_v], [`minimize_x`][lmlib.statespace.rls.RLSAlssm.minimize_x], and [`eval_errors`][lmlib.statespace.rls.RLSAlssm.eval_errors]. Default: True.
    calc_xi : bool, optional
        If True, compute [`xi`][lmlib.statespace.rls.RLSAlssm.xi]
        Required for [`minimize_v`][lmlib.statespace.rls.RLSAlssm.minimize_v], [`minimize_x`][lmlib.statespace.rls.RLSAlssm.minimize_x], and [`eval_errors`][lmlib.statespace.rls.RLSAlssm.eval_errors]. Default: True.
    calc_kappa : bool, optional
        If True, computes [`kappa`][lmlib.statespace.rls.RLSAlssm.kappa]
        Required for [`eval_errors`][lmlib.statespace.rls.RLSAlssm.eval_errors]. Can be set to False when only the minimizer is needed. Default: True.
    calc_nu : bool, optional
        If True, computes $\nu_k$, the number of weighted samples in the window.
        Not yet implemented. Default: False.
    filter_form : str, optional
        Controls the internal block structure of the recursive filter.

        - ``'cascade'`` : cascade block form (default)
        - ``'parallel'`` : parallel block form

    backend : str, optional
        Selects the computational backend for the state-space recursions.
        If ``None``, the globally configured backend is used (see [`set_backend`][lmlib.statespace.backend.set_backend]).

        - ``'numpy'`` : pure NumPy implementation (default)
        - ``'lfilter'`` : transfer-function backend using [`lfilter`][scipy.signal.lfilter]
        - ``'jit'`` : Numba JIT-compiled backend (requires ``numba`` package)

    Notes
    -----
    Setting ``steady_state=True`` (default) assumes the window defined by the cost segments
    does not change over time (LTI case), which allows $W_k$ to be precomputed once
    as a constant $W$. Set ``steady_state=False`` for time-varying windows
    (e.g. near signal boundaries or with sample-dependent weights), at the cost of
    a full $W_k$ calculation at every sample.

    Example
    --------
    Fit a polynomial of degree 2 to a signal using a left-sided window:
    ```python
    >>> import numpy as np
    >>> import lmlib as lm
    >>> from lmlib.utils.generator import gen_rect
    >>> K = 500
    >>> y = gen_rect(K, 200, 100)
    >>> alssm = lm.AlssmPoly(poly_degree=2)
    >>> seg = lm.Segment(a=-20, b=0, direction=lm.FORWARD, g=10)
    >>> rls = lm.RLSAlssm(lm.CostSegment(alssm, seg))
    >>> y_hat = rls.fit(y)
    ```

    To save computation when only the signal estimate is needed, disable $\kappa$:
    ```python
    >>> rls = lm.RLSAlssm(lm.CostSegment(alssm, seg), calc_kappa=False)
    >>> y_hat = rls.fit(y)
    ```
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
        assert self._backend in available_backends, (
            f"backend '{self._backend}' is not available on this system. "
            f"Available backends: {available_backends}. "
            f"(The 'cupy' backend requires the cupy package and a visible CUDA device; "
            f"'jit' requires numba.)")

        self._filter_form = filter_form

        # Per-dimension cost terms (CostSegment or CompositeCost), one per ND axis.
        #   - NDCompositeCost._get_sub_cost_term() returns a list of costs.
        #   - CompositeCost/CostSegment._get_sub_cost_term() returns the cost itself.
        raw = cost_terms._get_sub_cost_term()
        _sub_costs = raw if isinstance(raw, list) else [raw]

        # ------------------------------------------------------------------
        # check if filter form is valid for all segments
        # ------------------------------------------------------------------
        if self._backend in ('lfilter', 'cupy') and self._filter_form == 'cascade':
            for ct in _sub_costs:
                _alssms = [ct.alssm] if isinstance(ct, CostSegment) else list(ct.alssms)
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
        r"""[`ndarray`][numpy.ndarray] : $\xi^{(2)}(k,\mathbf{1})$, Alias for [`W`][lmlib.statespace.rls.RLSAlssm.W]."""
        return self.W

    @property
    def xi1(self):
        r"""[`ndarray`][numpy.ndarray] : $\xi^{(1)}(k,y)$, Alias for [`xi`][lmlib.statespace.rls.RLSAlssm.xi]."""
        return self.xi

    @property
    def xi0(self):
        r"""[`ndarray`][numpy.ndarray] : $\xi^{(0)}(k,y)$, Alias for [`kappa`][lmlib.statespace.rls.RLSAlssm.kappa]."""
        return self.kappa

    @property
    def W(self):
        r"""[`ndarray`][numpy.ndarray] : Gram matrix $W_k$ (const. $W$ in steady-state mode). Shape ``(..., N, N)``."""
        if self._xi2 is None:
            raise ValueError('xi2 has not been calculated. '
                             'Please run the filter() method with calc_W=True before calling W.')
        return self._xi2.reshape(self._xi2.shape[:-1] + (self._N, self._N))

    @property
    def xi(self):
        r"""[`ndarray`][numpy.ndarray] : Signal cross-correlation $\xi_k$. Shape ``(..., N)``."""
        if self._xi1 is None:
            raise ValueError('xi1 has not been calculated. '
                             'Please run the filter() method with calc_xi=True before calling xi.')
        return self._xi1

    @property
    def kappa(self):
        r"""[`ndarray`][numpy.ndarray] : Signal energy $\kappa_k$. Shape ``(...,)``."""
        if self._xi0 is None:
            raise ValueError('xi0 has not been calculated. '
                             'Please run the filter() method with calc_kappa=True before calling kappa.')
        return self._xi0

    @property
    def nu(self):
        r"""[`ndarray`][numpy.ndarray] : Effective number of weighted samples $\nu_k$ in the window. Not yet implemented."""
        # TODO nu implementation
        raise NotImplementedError("nu calculation is not yet implemented.")

    def _build_parallel_plan(self, _sub_costs):
        r"""
        Pre-build per-ALSSM transfer-function coefficients for the parallel
        $\xi^{(1)}$ realization (lfilter parallel backend only).

        Returns a 3-D structure ``plan[dim_index][seg_index]`` where each leaf is
        the list of ``(n0, n1, numdenom)`` tuples produced by
        [`build_parallel_numdenom`][lmlib.statespace.backends.rec_lfilter.build_parallel_numdenom]
        (one entry per active ALSSM block).  Returns ``None`` for every other
        backend/filter-form combination (nothing to pre-build).

        The combined block-diagonal ``A`` / block-partitioned ``C`` are built
        once per segment here; the backend slices and decomposes each block.
        """
        if not (self._filter_form == 'parallel' and self._backend in ('lfilter', 'cupy')):
            return None

        from lmlib.statespace.backends.rec_lfilter import build_parallel_numdenom

        plan = []
        for ct in _sub_costs:
            if isinstance(ct, CostSegment):
                segments, alssms, F = [ct.segment], [ct.alssm], np.ones((1, 1))
            else:
                segments, alssms, F = list(ct.segments), list(ct.alssms), ct.F
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
        Compute the ALSSM cost parameters $\xi^{(q)}(k,y)$ for $q \in \{0,1,2\}$ and $\nu_k$ for the input signal $y$.

        For [`CostSegment`][lmlib.statespace.cost.CostSegment] and [`CompositeCost`][lmlib.statespace.cost.CompositeCost] (1-D) each quantity is
        computed via the recursive equations (22–25) in [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018) for each cost segment by the
        selected backend.

        For [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost] (multi-dimensional) the ND cost separates as a
        Kronecker product over the per-dimension sub-costs, see Table III in [\[Baeriswyl2025\]](../../bibliography.md#baeriswyl2025).
        The computation is therefore chained over dimensions:

        1. `_nd_xi_q_recursion` processes the first axis in ``dim_order``,
           reading directly from ``y`` and producing ``xi_prev`` of shape
           ``(*Ks, N_0**q)``.
        2. For each subsequent axis ``dim_order[l]``,
           `_nd_xi_q_asterisk_l_recursion` extends the trailing axis of
           ``xi_prev`` from ``Nq_prev`` to ``Nq_prev * N_l**q``, accumulating the
           Kronecker structure dimension by dimension.
        3. After all ``L`` axes are processed the trailing axis has size
           $(N_0 \cdots N_{L-1})^q = N_\text{total}^q$.

        The final results are stored in [`W`][lmlib.statespace.rls.RLSAlssm.W],
        [`xi`][lmlib.statespace.rls.RLSAlssm.xi], and [`kappa`][lmlib.statespace.rls.RLSAlssm.kappa].

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
            Per-sample scalar weights $w_i \in [0, 1]$ applied to each
            observation before accumulation.  Must broadcast to the spatial shape
            of ``y``.  Default: all ones (uniform weighting).

        dim_order : array_like of int of length ``L``, optional
            Processing order of the signal axes for [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost].

            ``dim_order[0]`` is processed first by `_nd_xi_q_recursion`;
            ``dim_order[1], dim_order[2], ...`` are each processed in turn by
            `_nd_xi_q_asterisk_l_recursion`.

            The Kronecker structure of the output state vector follows this order:
            with ``dim_order = [d_0, d_1, ..., d_{L-1}]``, the trailing axis of
            the minimiser result ``xs`` (see [`minimize_x`][lmlib.statespace.rls.RLSAlssm.minimize_x]) is arranged as:

            $$
            x_{d_0} \otimes x_{d_1} \otimes \cdots \otimes x_{d_{L-1}}
            $$

            where $x_{d_l}$ is the per-dimension state vector for axis
            $d_l$.  Changing ``dim_order`` therefore permutes the Kronecker
            blocks in ``xs``, but does not change the scalar cost $J_k(x)$.

            Has no effect for [`CostSegment`][lmlib.statespace.cost.CostSegment] or [`CompositeCost`][lmlib.statespace.cost.CompositeCost]
            (those are inherently 1-D).  Default: ``np.arange(L)`` (axes in
            natural order).

        Notes
        -----
        **Steady-state mode** (``steady_state=True``, the default):
            $W_k$ is constant and precomputed once via
            [`get_steady_state_W`][lmlib.statespace.cost.CompositeCost.get_steady_state_W] before the other
            recursions start.  This is valid whenever the window is
            time-invariant ($w_k = w$, $\gamma_k = \gamma$), which
            holds for all standard [`Segment`][lmlib.statespace.segment.Segment] definitions without
            sample-dependent weights.  See Section III-I.2 in [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018).
            The ``calc_W`` flag is ignored in this mode.

        **Time-varying mode** (``steady_state=False``):
            $W_k$ is computed sample-by-sample via the $q = 2$
            recursion, which carries the same computational cost as the
            $q = 1$ recursion.  Required near signal boundaries or when
            ``sample_weights`` varies over time.  Only supported for 1-D costs;
            multi-dimensional costs require ``steady_state=True`` (the ND
            $W$ recursion is not yet implemented).

        Returns
        -------
        None
            Results are stored in-place.  Access via [`W`][lmlib.statespace.rls.RLSAlssm.W] (shape ``(N, N)`` steady-state, or ``(*Ks, N, N)`` time-varying),
            [`xi`][lmlib.statespace.rls.RLSAlssm.xi] (shape ``(*Ks, N)``), and [`kappa`][lmlib.statespace.rls.RLSAlssm.kappa] (shape ``(*Ks,)``),

            where $N = N_0 \cdots N_{L-1}$ is the total model order.

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

    def convolve(self, y, xref, sample_weights=None, dim_order=None):
        r"""
        Convolve (or correlate) a signal with a reference state vector in the ALSSM feature space.

        This is a thin convenience wrapper around [`filter`][lmlib.statespace.rls.RLSAlssm.filter]:
        it projects the signal $y$ into the low-dimensional ALSSM feature space
        and then contracts the per-sample cross-correlation state $\xi_k$ with a
        fixed reference state vector $x_\mathrm{ref}$,

        $$
            \mathrm{out}_k \;=\; \langle \xi_k,\, x_\mathrm{ref} \rangle ,
        $$

        which is equivalent to a sliding-window linear filter whose effective
        impulse response lives in the span of the ALSSM.
        Because $\xi_k$ is obtained by a recursive filter, the cost is
        $\mathcal{O}(K \cdot N)$ and **independent of the effective filter
        length** (the segment window), unlike a direct sample-domain
        convolution.

        Whether the operation realises a convolution or a correlation depends on
        $x_\mathrm{ref}$ and on the segment direction of the underlying cost
        (a time-reversed/inverted model gives the convolution, a forward model
        the correlation); see the convolution examples in the ``50-convolution``
        folder.

        Internally this is ``self.filter(y, ...)`` followed by
        ``numpy.tensordot(self.xi, xref, axes=xref.ndim)``.  For a 1-D ``xref``
        (shape ``(N,)``) this reduces to ``self.xi @ xref``; a higher-rank
        ``xref`` (e.g. shape ``(channels, N)`` for multichannel signals)
        additionally sums over the leading reference axes, reproducing the
        channel-wise accumulation $\sum_j \xi_{k,j} \, x_{\mathrm{ref},j}$.

        Parameters
        ----------
        y : array_like
            Input signal, in the same shape as accepted by
            [`filter`][lmlib.statespace.rls.RLSAlssm.filter].
        xref : array_like
            Reference state vector in the ALSSM feature space. Its shape must
            match the trailing feature axes of [`xi`][lmlib.statespace.rls.RLSAlssm.xi]:

            * 1-D ``(N,)`` for a single-channel [`CostSegment`][lmlib.statespace.cost.CostSegment] /
              [`CompositeCost`][lmlib.statespace.cost.CompositeCost], or the Kronecker
              state ``(N,)`` of an [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost];
            * ``(channels, N)`` for a multichannel signal, in which case the
              channel axis is summed over.
        sample_weights : array_like, optional
            Per-sample weights forwarded to [`filter`][lmlib.statespace.rls.RLSAlssm.filter].
        dim_order : array_like, optional
            Processing order of the signal axes for
            [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost], forwarded to
            [`filter`][lmlib.statespace.rls.RLSAlssm.filter].

        Returns
        -------
        out : ndarray
            Convolution/correlation output with the spatial shape of $y$
            (the trailing feature axes are contracted away). Shape ``(K,)`` for
            a 1-D signal and ``(K_0, ..., K_{L-1})`` for an ND signal.

        Notes
        -----
        Only $\xi$ is needed, so if this ``RLSAlssm`` was constructed with
        ``calc_W=True`` and/or ``calc_kappa=True`` those flags are turned off
        (with a warning) on the first call to avoid computing $W$ and $\kappa$.
        Construct the ``RLSAlssm`` with ``calc_W=False, calc_kappa=False`` to
        silence the warning.

        Examples
        --------
        ```python
        >>> import numpy as np
        >>> import lmlib as lm
        >>> alssm = lm.AlssmPoly(poly_degree=2)
        >>> segment = lm.Segment(a=-10, b=10, direction=lm.BACKWARD, g=200)
        >>> cost = lm.CostSegment(alssm, segment)
        >>> rls = lm.RLSAlssm(cost, steady_state=True)
        >>> y = np.zeros(100); y[40:61] = 1.0          # a rectangular pulse
        >>> xref = np.array([1.0, 0.0, 0.0])           # match the local mean (N = 3)
        >>> out = rls.convolve(y, xref)
        >>> out.shape
        (100,)
        ```
        """
        # convolve() only consumes xi; W (xi^(2)) and kappa (xi^(0)) are not
        # needed and would only add per-sample cost. Disable them (once) and
        # warn so the caller can silence it by constructing the RLSAlssm with
        # calc_W=False, calc_kappa=False.
        _enabled = [name for name, flag in
                    (('calc_W', self._calc_W), ('calc_kappa', self._calc_kappa)) if flag]
        if _enabled:
            self._calc_W = False
            self._calc_kappa = False
            warnings.warn(
                f"convolve() only requires xi; disabling {' and '.join(_enabled)} "
                f"for this RLSAlssm. Construct it with calc_W=False, calc_kappa=False "
                f"to avoid this warning.",
                stacklevel=2,
            )

        self.filter(y, sample_weights=sample_weights, dim_order=dim_order)
        xref = np.asarray(xref)
        return np.tensordot(self.xi, xref, axes=xref.ndim)

    # ------------------------------------------------------------------
    # minimize / eval
    # ------------------------------------------------------------------

    def minimize_v(self, H=None, h=None, solver='lstsq'):
        r"""
        Minimizes the cost $J_k(x)$ subject to a linear constraint on the state vector, returning the reduced-dimensional free parameter $v$.

        The state vector $x$ is constrained to the affine subspace

        $$
        x = Hv + h,
        $$

        with known $H \in \mathbb{R}^{N \times M}$ and $h \in \mathbb{R}^N$, and
        unknown $v \in \mathbb{R}^M$. Substituting into the cost function (Eq. (21)
        [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018)) and minimizing over $v$ yields the closed-form solution
        (Eq. (69) [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018)):

        $$
        \hat{v}_k = (H^T W_k H)^{-1} H^T (\xi_k - W_k h).
        $$

        When ``H=None`` and ``h=None`` (defaults), the constraint reduces to $x = v$
        (unconstrained minimization), so $\hat{v}_k = W_k^{-1} \xi_k$.

        Parameters
        ----------
        H : array_like of shape (N, M), optional
            Constraint matrix mapping the free parameter $v \in \mathbb{R}^M$ to the
            state space $\mathbb{R}^N$. If ``None``, defaults to the identity matrix
            $I_N$ (unconstrained, $M = N$).
        h : array_like of shape (N,), optional
            Constraint offset vector $h \in \mathbb{R}^N$. If ``None``, defaults to
            the zero vector (no offset).
        solver : {'lstsq', 'solve', 'inv'}, optional
            Linear-system solver strategy (default: ``'lstsq'``).

        Notes
        -----
        The distinction between [`minimize_v`][lmlib.statespace.rls.RLSAlssm.minimize_v] and [`minimize_x`][lmlib.statespace.rls.RLSAlssm.minimize_x] is:

        - [`minimize_v`][lmlib.statespace.rls.RLSAlssm.minimize_v] returns $v \in \mathbb{R}^M$ (the free parameter, dimension M).
        - [`minimize_x`][lmlib.statespace.rls.RLSAlssm.minimize_x] returns $x = Hv + h \in \mathbb{R}^N$ (the full state vector,
          dimension N).

        When ``H=None``, both methods return the same result since $x = v$.

        When ``steady_state=True``, $W_k = W$ is constant, so $H^T W H$ is
        inverted once. When ``steady_state=False``, samples where $H^T W_k H$ is
        ill-conditioned (singular to machine precision) are left as ``NaN``.

        Returns
        -------
        v : ndarray of shape (..., M)
            Free parameter $\hat{v}_k$ minimizing $J_k(Hv + h)$. The leading
            dimensions match the signal length and any parallel dimensions of the input.
            Entries are ``NaN`` where $H^T W_k H$ is ill-conditioned
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

        **``'lstsq'`` (default)**

        **Steady-state** — Computes the economy SVD of $H^\top W H$
        *once* and stores the resulting pseudoinverse as an attribute
        ``_pinv_HTWH``.  Subsequent calls with the same ``H`` and ``h``
        reuse this cached pseudoinverse, reducing the cost to a single
        matrix–vector multiply per time step.  This is typically **4–22×
        faster** than calling ``numpy.linalg.lstsq`` with all right-hand
        sides at once, because that function recomputes the SVD on every
        call even when the matrix has not changed.

        **Non-steady-state** — Attempts a batched LU solve
        (``numpy.linalg.solve``) on the full $(K, N, N)$ stack of
        matrices.  If any matrix is singular, falls back to a fully
        vectorised batched SVD pseudoinverse (``numpy.linalg.svd`` over
        the batch dimension) — no Python loop required.  This gives a
        **5–70× speedup** over the previous per-sample ``lstsq`` loop.

        Handles rank-deficient $H^\top W H$ gracefully in both
        modes by returning the minimum-norm least-squares solution
        (Moore–Penrose pseudoinverse).

        **``'solve'``**

        Uses ``numpy.linalg.solve`` (LU factorisation) on the invertible
        subset and falls back to ``'lstsq'`` for rank-deficient blocks.
        Slightly faster than ``'lstsq'`` when $H^\top W H$ is always
        full-rank (the common case), but equally robust otherwise.  Does
        **not** cache a pseudoinverse.

        **``'inv'`` (legacy)**

        Explicit matrix inversion — the original behaviour.  Raises an
        ``AssertionError`` (steady-state) or leaves entries as ``nan``
        (non-steady-state) when $H^\top W H$ is not invertible.
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
        Minimizes the cost $J_k(x)$ subject to a linear constraint on the state vector, returning the full N-dimensional state vector $x$.

        Internally calls [`minimize_v`][lmlib.statespace.rls.RLSAlssm.minimize_v] to obtain $\hat{v}_k$, then reconstructs
        the state vector via the constraint (Eq. (66) [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018)):

        $$
        \hat{x}_k = H \hat{v}_k + h.
        $$

        When ``H=None`` and ``h=None`` (defaults), the constraint reduces to $x = v$,
        so $\hat{x}_k = W_k^{-1} \xi_k$ (Table V row 1, [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018)).

        Parameters
        ----------
        H : array_like of shape (N, M), optional
            Constraint matrix mapping the free parameter $v \in \mathbb{R}^M$ to the
            state space $\mathbb{R}^N$. If ``None``, defaults to the identity matrix
            $I_N$ (unconstrained, $M = N$).
        h : array_like of shape (N,), optional
            Constraint offset vector $h \in \mathbb{R}^N$. If ``None``, defaults to
            the zero vector (no offset).

        Notes
        -----
        The distinction between [`minimize_v`][lmlib.statespace.rls.RLSAlssm.minimize_v] and [`minimize_x`][lmlib.statespace.rls.RLSAlssm.minimize_x] is:

        - [`minimize_v`][lmlib.statespace.rls.RLSAlssm.minimize_v] returns $v \in \mathbb{R}^M$ (the free parameter, dimension M).
        - [`minimize_x`][lmlib.statespace.rls.RLSAlssm.minimize_x] returns $x = Hv + h \in \mathbb{R}^N$ (the full state vector,
          dimension N).

        When ``H=None``, both methods return the same result since $x = v$.

        Returns
        -------
        x : ndarray of shape (..., N)
            Optimal state vector $\hat{x}_k = H\hat{v}_k + h$. The leading dimensions
            match the signal length and any parallel dimensions of the input.
            Entries are ``NaN`` where $H^T W_k H$ is ill-conditioned
            (only possible when ``steady_state=False``).

        Raises
        ------
        AssertionError
            Propagated from [`minimize_v`][lmlib.statespace.rls.RLSAlssm.minimize_v]:
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
        Evaluates the cost function $J_k(x)$ at given state vectors `xs`.

        Using the expanded form of the cost (Eq. (21) [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018)):

        $$
        J_k(x) = x^T W_k x - 2 x^T \xi_k + \kappa_k,
        $$

        this method computes the scalar cost for each provided state vector without
        performing any minimization. It is useful for comparing the fit quality of
        different candidate state vectors, e.g. for computing error ratios or
        log-cost ratios (LCR) as in Section III.F [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018).

        Requires [`W`][lmlib.statespace.rls.RLSAlssm.W], [`xi`][lmlib.statespace.rls.RLSAlssm.xi], and [`kappa`][lmlib.statespace.rls.RLSAlssm.kappa] to be available, i.e.
        [`filter`][lmlib.statespace.rls.RLSAlssm.filter] must have been called with ``calc_W=True``, ``calc_xi=True``,
        and ``calc_kappa=True`` (all defaults).

        Parameters
        ----------
        xs : array_like of shape (..., N)
            State vector(s) at which to evaluate the cost. The last dimension must
            be the state dimension N. Leading dimensions are broadcast over the
            signal samples.

        Returns
        -------
        J : ndarray of shape (...)
            Cost $J_k(x)$ evaluated at each state vector in `xs`. Same shape
            as `xs` with the last axis removed.
        """

        if self._steady_state:
            J = np.einsum('...n, ...n', xs, np.einsum('nm, ...m->...n', self.W, xs))
        else:
            J = np.einsum('...n, ...n', xs, np.einsum('...nm, ...m->...n', self.W, xs))

        return J - 2 * np.einsum('...n, ...n', self.xi, xs) + self.kappa

    def fit(self, y, output='y_hat', sample_weights=None, dim_order=None, H=None, h=None, eval_alssm_weights=None, solver='lstsq'):
        r"""
        Method that chains [`filter`][lmlib.statespace.rls.RLSAlssm.filter], [`minimize_v`][lmlib.statespace.rls.RLSAlssm.minimize_v], and signal reconstruction into a single call.

        Executes the following steps in order:

        1. [`filter`][lmlib.statespace.rls.RLSAlssm.filter] — computes the recursive filter quantities $W_k$,
           $\xi_k$, $\kappa_k$ from the input signal `y`.
        2. [`minimize_v`][lmlib.statespace.rls.RLSAlssm.minimize_v] — solves the constrained minimization
           $\hat{v}_k = (H^T W_k H)^{-1} H^T (\xi_k - W_k h)$.
        3. Reconstruction — builds $\hat{x}_k = H\hat{v}_k + h$ and/or the
           signal estimate $\hat{y}_k = C A^j \hat{x}_k$ depending on `output`.

        Parameters
        ----------
        y : array_like of shape (K, [Q])
            Input signal. See [`filter`][lmlib.statespace.rls.RLSAlssm.filter] for shape details.
        output : str or tuple of str, optional
            Selects what is returned. One or more of:

            - ``'y_hat'`` *(default)* — signal estimate $\hat{y}_k = CA^j\hat{x}_k$
              evaluated via the cost term's
              [`eval_alssm_output`][lmlib.statespace.cost.CompositeCost.eval_alssm_output].
            - ``'x'`` — full state vector $\hat{x}_k = H\hat{v}_k + h$,
              shape ``(..., N)``.
            - ``'v'`` — free parameter $\hat{v}_k$, shape ``(..., M)``.

            Pass a tuple to return multiple outputs, e.g. ``output=('x', 'y_hat')``.
        sample_weights : array_like of shape (K,), optional
            Per-sample weights $w_i \in [0,1]$. Passed to [`filter`][lmlib.statespace.rls.RLSAlssm.filter].
            Default: all ones.
        dim_order : array_like of int, optional
            Dimension processing order for [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost].
            Passed to [`filter`][lmlib.statespace.rls.RLSAlssm.filter]. Default: ``np.arange(L)``.
        H : array_like of shape (N, M), optional
            Constraint matrix. Passed to [`minimize_v`][lmlib.statespace.rls.RLSAlssm.minimize_v]. Default: identity (unconstrained).
        h : array_like of shape (N,), optional
            Constraint offset. Passed to [`minimize_v`][lmlib.statespace.rls.RLSAlssm.minimize_v]. Default: zero vector.
        eval_alssm_weights : array_like, optional
            Per-ALSSM output weights used when evaluating $\hat{y}$.

            - For [`CompositeCost`][lmlib.statespace.cost.CompositeCost] / [`CostSegment`][lmlib.statespace.cost.CostSegment]:
              shape ``(M,)`` (or scalar), forwarded as ``alssm_weights`` to the cost
              term's [`eval_alssm_output`][lmlib.statespace.cost.CompositeCost.eval_alssm_output]
              (which passes them to [`AlssmSum`][lmlib.statespace.model.AlssmSum]).
            - For [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost]:
              shape ``(L, M)``, forwarded as ``nd_alssm_weights`` to
              [`eval_alssm_output`][lmlib.statespace.cost.NDCompositeCost.eval_alssm_output],
              where element ``[l, m]`` scales ALSSM ``m`` in signal dimension ``l``.

            If ``None``, all models contribute equally.

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
            Propagated from [`minimize_v`][lmlib.statespace.rls.RLSAlssm.minimize_v] if ``H`` / ``h`` have wrong shape or
            ``H.T @ W @ H`` is not invertible (when ``steady_state=True``).

        Example
        --------
        Fit a degree-2 polynomial and return the signal estimate:
        ```python
        >>> import numpy as np
        >>> import lmlib as lm
        >>> K = 200
        >>> y = np.random.randn(K)
        >>> alssm = lm.AlssmPoly(poly_degree=2)
        >>> seg = lm.Segment(a=-20, b=0, direction=lm.FORWARD, g=10)
        >>> rls = lm.RLSAlssm(lm.CostSegment(alssm, seg))
        >>> y_hat = rls.fit(y)
        ```
        Return both the state vector and the signal estimate:
        ```python
        >>> x, y_hat = rls.fit(y, output=('x', 'y_hat'))
        ```
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

        if isinstance(self._cost_terms, NDCompositeCost):
            # NDCompositeCost separates per signal dimension; its eval_alssm_output
            # takes `nd_alssm_weights` of shape (L, M). `None` weights all ALSSMs
            # equally, which matches the default behaviour of the 1-D branch below.
            out_dict['y_hat'] = self._cost_terms.eval_alssm_output(x, nd_alssm_weights=eval_alssm_weights)
        else:
            out_dict['y_hat'] = self._cost_terms.eval_alssm_output(x, alssm_weights= eval_alssm_weights if eval_alssm_weights is not None else [1.0] * len(self._cost_terms.get_alssms()))

        if _output == ('y_hat',):
            return out_dict['y_hat']
        return tuple(out_dict[_] for _ in _output)

    # ------------------------------------------------------------------
    # Recursion: _nd_xi_q_recursion (first dimension)
    # ------------------------------------------------------------------

    def _nd_xi_q_recursion(self, q, y, sample_weights, model_dimension):
        r"""
        Defines the recursion to calculate the ALSSM cost parameters $\xi^{(q)}(k,y)$ for a given $q \in \{0,1,2\}$ based on an input signal $y$.

        The cost parameters $\xi^{(q)}(k,y)$ for $q \in \{0,1,2\}$ are equivalent to $\kappa_k$, $\xi_k$ and $W_k$.
        They are calculated through the recursive equations (22-25) [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018) for each cost segment defined by the different backends.
        We iterate over each individual ALSSM m within every segment p and write the result into the appropriate sub-slice of xi_curr.
        Because A is block-diagonal, the blocks are independent and can be computed separately.
        Each ALSSM m is wrapped in a single-element AlssmSum with weight F[m,p] so that:
          - force_MC is honoured (C is guaranteed 2-D for all backends), and
          - the F-column weight is folded into C (as lambda), keeping beta equal to the original segment beta.

        Parameters
        ----------
        q : int
            Order of the cost parameter. Must be 0, 1, or 2, corresponding to
            $\kappa_k$ (signal energy), $\xi_k$ (signal projection), and
            $W_k$ (Gram matrix), respectively.
            Determines the trailing size of the output: $N_l^q$, where $N_l$
            is the model order of the sub-cost associated with ``model_dimension``.
        y : array_like of shape (\*Ks, Q)
            Input signal, where ``Ks`` are the signal length dimensions (one per ND axis)
            and ``Q`` is the ALSSM output dimension. For scalar ALSSMs ``Q=0`` after
            preprocessing in [`filter`][lmlib.statespace.rls.RLSAlssm.filter].
        sample_weights : array_like of shape (\*Ks,)
            Per-sample weights $w_i \in [0,1]$.
        model_dimension : int
            Index of the signal axis along which the 1-D state-space recursion is applied.
            All other signal axes are flattened into a batch and iterated over
            sequentially (one slice at a time via an inner loop).
            For a 1-D signal (``CostSegment`` / ``CompositeCost``) this is always 0.
            For an ND signal (``NDCompositeCost``) each axis is processed in a separate call.

        Notes
        -----
        The difference between this function and [`_nd_xi_q_asterisk_l_recursion()`][lmlib.statespace.rls._nd_xi_q_asterisk_l_recursion()] is
        that this function is the **first** recursion step: it reads directly from the input
        signal `y` and initialises $\xi^{(q)}$ from scratch with shape
        ``(*Ks, N_l**q)``.

        [`_nd_xi_q_asterisk_l_recursion()`][lmlib.statespace.rls._nd_xi_q_asterisk_l_recursion()] is used for every **subsequent** dimension:
        it takes the accumulated ``xi_prev`` (shape ``(*Ks, Nq_prev)``) from the previous step
        and extends the trailing axis to ``Nq_prev * N_l**q``, realising the tensor-product
        (Kronecker) structure over ND dimensions. After processing all L dimensions the
        trailing axis has size $(N_0 \cdots N_{L-1})^q = N_\mathrm{total}^q$.

        Returns
        -------
        xi_curr : ndarray of shape (\*Ks, N_l**q)
            $\xi^{(q)}(k,y)$ cost parameter with the same leading shape as `y`
            (all signal dimensions) and a trailing axis of size $N_l^q$, where
            $N_l$ is the model order of the sub-cost for ``model_dimension``.
        """
        ct = self._cost_terms._get_sub_cost_term(model_dimension)
        if isinstance(ct, CostSegment):
            segments, alssms, F, betas = [ct.segment], [ct.alssm], np.ones((1, 1)), np.array([ct.beta])
        else:
            segments, alssms, F, betas = list(ct.segments), list(ct.alssms), ct.F, ct.betas
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
        if self._backend == 'cupy':
            # The GPU backend writes its result with ``xi += asnumpy(...)``
            # directly into the output view, so the two largest host-side
            # transposes are pure waste: ``ascontiguousarray(_xi_view)`` copies a
            # freshly-zeroed ~output-sized array, and the final write-back copies
            # it back -- together ~2x the output through single-threaded numpy,
            # which dominates the multichannel wall-clock.  ``reshape`` of the
            # moveaxis view returns a *view* when only one batch axis is flattened
            # (the common 1-D multichannel case), so the backend accumulates
            # straight into ``xi_curr`` and no write-back is needed.  When reshape
            # must copy (several ND batch axes) ``_writeback`` restores it.  The
            # input copies are kept: ``cp.asarray`` needs a contiguous host buffer
            # anyway, so contiguifying here is no more work and avoids surprises.
            _xi_curr = np.reshape(_xi_view, (-1, *_xi_view.shape[-2:]))
            _writeback = not np.shares_memory(_xi_curr, xi_curr)
        else:
            _xi_work = np.ascontiguousarray(_xi_view)
            _xi_curr = np.reshape(_xi_work, (-1, *_xi_work.shape[-2:]))
            _writeback = True
        _ym = np.ascontiguousarray(np.moveaxis(y, model_dimension, -2))
        _y = np.reshape(_ym, (-1, *_ym.shape[-2:]))
        _swm = np.ascontiguousarray(np.moveaxis(sample_weights, model_dimension, -1))
        _sw = np.reshape(_swm, (-1, *_swm.shape[-1:]))

        for p, segment in enumerate(segments):
            combined = AlssmSum(alssms, F[:, p], force_MC=True)
            plan = (self._parallel_plan[dim_index][p]
                    if self._parallel_plan is not None else None)
            if self._backend == 'cupy' and self._filter_form == 'cascade':
                # Process the whole channel batch in a single GPU sweep instead
                # of one backend call per channel (see rec_cupy batched section).
                from lmlib.statespace.backends.rec_cupy import cupy_xi_q_recursion_batch
                cupy_xi_q_recursion_batch(
                    _xi_curr, q, combined, segment,
                    _y, _sw, betas[p], block_sizes)
            elif self._backend == 'cupy' and self._filter_form == 'parallel':
                # Batched parallel form: all channels in one GPU sweep.
                from lmlib.statespace.backends.rec_cupy import cupy_xi_q_recursion_parallel_batch
                cupy_xi_q_recursion_parallel_batch(
                    _xi_curr, q, combined, segment,
                    _y, _sw, betas[p], plan, block_sizes)
            else:
                for i in range(_y.shape[0]):
                    xi_q_recursion(
                        _xi_curr[i], q, combined, segment,
                        _y[i], _sw[i], betas[p],
                        self._backend, self._filter_form, block_sizes, plan)

        if _writeback:
            _xi_view[:] = np.reshape(_xi_curr, _xi_view.shape)
        return xi_curr

    # ------------------------------------------------------------------
    # Recursion: _nd_xi_q_asterisk_l_recursion (subsequent dimensions)
    # ------------------------------------------------------------------

    def _nd_xi_q_asterisk_l_recursion(self, xi_prev, q, y, sample_weights, model_dimension):
        r"""
        Defines the recursion for one additional dimension to calculate the ALSSM cost parameters $\xi^{(q)*}(k,y)$ for a given $q \in \{0,1,2\}$ based on an input signal $y$.

        The cost parameters $\xi^{(q)}(k,y)$ for $q \in \{0,1,2\}$ are equivalent to $\kappa_k$, $\xi_k$ and $W_k$.
        They are calculated through the recursive equations (22-25) [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018) for each cost segment defined by the different backends.
        This function subdivides the calculation into cost segments (for-loop). The same per-ALSSM decomposition as _nd_xi_q_recursion is applied:
        each ALSSM m in the CompositeCost for this dimension is processed independently, writing into the corresponding sub-slice of xi_curr.

        Parameters
        ----------
        xi_prev : ndarray of shape (\*Ks, Nq_prev)
            Accumulated cost parameter from the previous dimension step, where
            ``Nq_prev`` is the trailing size built up so far (e.g. ``N_0**q`` after
            the first step, ``N_0**q * N_1**q`` after the second, etc.).
        q : int
            Order of the cost parameter. Must be 0, 1, or 2, corresponding to
            $\kappa_k$ (signal energy), $\xi_k$ (signal projection), and
            $W_k$ (Gram matrix), respectively.
            The trailing axis of the output grows by a factor $N_l^q$, where
            $N_l$ is the model order of the sub-cost for ``model_dimension``.
        y : array_like of shape (\*Ks, Q)
            Input signal (only its shape is used; values enter via ``xi_prev``).
        sample_weights : array_like of shape (\*Ks,)
            Per-sample weights $w_i \in [0,1]$.
        model_dimension : int
            Signal axis processed in this step.

        Returns
        -------
        xi_curr : ndarray of shape (\*Ks, Nq_prev \* N_l**q)
            Extended cost parameter. The leading shape matches `y`; the trailing axis
            is the product of all accumulated sub-cost orders raised to $q$.
        """
        ct = self._cost_terms._get_sub_cost_term(model_dimension)
        if isinstance(ct, CostSegment):
            segments, alssms, F, betas = [ct.segment], [ct.alssm], np.ones((1, 1)), np.array([ct.beta])
        else:
            segments, alssms, F, betas = list(ct.segments), list(ct.alssms), ct.F, ct.betas
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
