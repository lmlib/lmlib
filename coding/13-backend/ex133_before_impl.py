"""
Baseline ("before") implementations used by the ex133 benchmark notebooks.

**``nd_xi_q_recursion``** — before state for the RLS pipeline:
    Production code from Christof's develop branch with the *only* difference
    being that ``xi_curr`` is allocated in default C-order instead of F-order.
    Used in ex133.3 (full-pipeline before/after benchmark) by monkey-patching
    it onto ``lm.RLSAlssm._nd_xi_q_recursion``.

**``lfilter_forward_cascade_xi`` / ``lfilter_backward_cascade_xi``** — before
    state for the cascade inner loop:
    Same logic as the production versions in ``rec_lfilter.py`` but with
    ``xi0 = np.zeros_like(xi)`` (C-order, same shape as ``xi``) instead of the
    F-order ``(K + K_append, N)`` buffer introduced by Christof.  Used by:

    * ex133.0 — line-by-line profiling baseline (``inspect.getsource()`` must work)
    * ex133.1 — N=2 before/after cascade benchmark (called via ``variant_before``)
    * ex133.2 — multi-N before/after cascade benchmark (called via ``make_variants``)

    In ex133.3 only ``nd_xi_q_recursion`` is monkey-patched; the cascade functions
    there always use the current production versions.

Kept as a standalone module so that ``inspect.getsource()`` works correctly
when functions are registered with ``line_profiler.LineProfiler``.
"""

import numpy as np
from numpy.linalg import inv as _inv, matrix_power as _mpow
from scipy.signal import lfilter as _lfilter
from lmlib.statespace.backends.rec import xi_q_recursion as _xi_q_recursion
from lmlib.statespace.rls import _as_composite_cost
from lmlib.statespace.cost import NDCompositeCost
from lmlib.statespace.model import AlssmSum


def lfilter_forward_cascade_xi(xi, A, C, a, b, delta, gamma, y, v, beta):
    """Before: C-order xi0 scratch buffer; xi += xi0 accumulation copy."""
    if not (a < 0 and b <= 0):
        raise NotImplementedError('BACKEND: a and b has to be lower then zero for forward calculated segments.')
    gamma_inv = 1 / gamma
    gamma_a   = gamma ** (a - 1 - delta)
    gamma_b   = gamma ** (b - delta)
    gAinvT    = gamma_inv * _inv(A).T
    Aac       = np.dot(_mpow(A, 0 if np.isinf(a) else a - 1).T, C.T)
    Abc       = np.dot(_mpow(A, b).T, C.T)
    N         = np.shape(A)[1]
    if not np.allclose(gAinvT, np.tril(gAinvT)):
        raise ValueError('State-Space Matrix A needs to be upper triangular for cascaded version')
    vy = y * v[:, None]
    y_delayed_b = np.empty_like(vy)
    y_delayed_b[:-b] = 0
    y_delayed_b[-b:] = vy[:b]
    y_diff = np.einsum('kl,nl->kn', y_delayed_b, gamma_b * Abc)
    if not np.isinf(a):
        y_delayed_a = np.empty_like(vy)
        y_delayed_a[:-a + 1] = 0
        y_delayed_a[-a + 1:] = vy[:a - 1]
        y_diff -= np.einsum('kl,nl->kn', y_delayed_a, gamma_a * Aac)
    # iterating through dimensions
    y_diff = np.swapaxes(y_diff, 0, 1)  # convenient for later indexing
    xi0 = np.zeros_like(xi)
    n_ = 0
    xi0[:, n_] = _lfilter([1, 0], [1, -gamma_inv], y_diff[n_].T).T
    for n_ in range(1, N):
        y_diff[n_, 1:] += np.einsum('kn..., n->k...', xi0[:-1], gAinvT[n_])
        xi0[:, n_] = _lfilter([1, 0], [1, -gamma_inv], y_diff[n_].T).T
    xi += xi0                         # <- accumulation copy (before)
    if beta != 1:
        xi *= beta


def lfilter_backward_cascade_xi(xi, A, C, a, b, delta, gamma, y, v, beta):
    """Before: C-order xi0 scratch buffer; xi += xi0[::-1] accumulation copy."""
    if not (a >= 0 and b > 0):
        raise NotImplementedError('BACKEND: a and b has to be higher then zero for backward calculated segments.')
    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)
    gAT     = gamma * A.T
    Aac     = np.dot(_mpow(A, a).T, C.T)
    Abc     = np.dot(_mpow(A, 0 if np.isinf(b) else b + 1).T, C.T)
    N       = np.shape(A)[1]
    K       = len(xi)
    if not np.allclose(gAT, np.tril(gAT)):
        raise ValueError('State-Space Matrix A needs to be upper triangular for cascaded version')
    vy = y * v[:, None]
    y_delayed_a = np.empty_like(vy)
    y_delayed_a[-a:] = 0
    y_delayed_a[:K - a] = vy[a:]
    y_diff = np.einsum('kl,nl->kn', y_delayed_a, gamma_a * Aac)
    if not np.isinf(b):
        y_delayed_b = np.empty_like(vy)
        y_delayed_b[-b - 1:] = 0
        y_delayed_b[:K - b - 1] = vy[b + 1:]
        y_diff -= np.einsum('kl,nl->kn', y_delayed_b, gamma_b * Abc)
    y_diff_flipped = np.swapaxes(y_diff[::-1], 0, 1)
    xi0 = np.zeros(xi.shape, order='C')       # <- C-order scratch (before)
    xi0[:, 0] = _lfilter([1, 0], [1, -gamma], y_diff_flipped[0].T).T
    for n_ in range(1, N):
        y_diff_flipped[n_, 1:] += np.einsum('kn...,n->k...', xi0[:-1], gAT[n_])
        xi0[:, n_] = _lfilter([1, 0], [1, -gamma], y_diff_flipped[n_].T).T
    xi += xi0[::-1]                            # <- accumulation copy (before)
    if beta != 1:
        xi *= beta


def nd_xi_q_recursion(self, q, y, sample_weights, model_dimension):
    """Before: xi_curr allocated in default C-order (no order='F').
    This is the production code from Christof's develop branch minus the
    order='F' change — used to isolate that change's effect in ex133.3.
    """
    sub_cost = _as_composite_cost(
        self._cost_terms._get_sub_cost_term(model_dimension)
    )
    dim_index = model_dimension if isinstance(self._cost_terms, NDCompositeCost) else 0

    N = sub_cost.get_alssm_order()
    *Ks, Q = np.shape(y)
    xi_curr = np.zeros((*Ks, N ** q))  # <- C-order (before Christof's order='F' change)

    _xi_curr = np.moveaxis(xi_curr, model_dimension, -2)
    _xi_curr = np.reshape(_xi_curr, (-1, *_xi_curr.shape[-2:]))
    _y = np.moveaxis(y, model_dimension, -2)
    _y = np.reshape(_y, (-1, *_y.shape[-2:]))
    _sample_weights = np.moveaxis(sample_weights, model_dimension, -1)
    _sample_weights = np.reshape(_sample_weights, (-1, *_sample_weights.shape[-1:]))

    offsets = self._alssm_offsets(sub_cost)

    for p, segment in enumerate(sub_cost.segments):
        beta_p = sub_cost.betas[p]

        if q == 2:
            combined = AlssmSum(sub_cost.alssms, sub_cost.F[:, p], force_MC=True)
            for i in range(_y.shape[0]):
                _xi_q_recursion(_xi_curr[i], q, combined, segment,
                                _y[i], _sample_weights[i],
                                beta_p, self._backend, self._filter_form, None)
            continue

        if q == 0:
            dummy_alssm = AlssmSum([sub_cost.alssms[0]], [1.0], force_MC=True)
            for i in range(_y.shape[0]):
                _xi_q_recursion(_xi_curr[i], q, dummy_alssm, segment,
                                _y[i], _sample_weights[i],
                                beta_p, self._backend, self._filter_form, None)
            continue

        for m, alssm_m in enumerate(sub_cost.alssms):
            f_mp = sub_cost.F[m, p]
            if f_mp == 0.0:
                continue
            wrapped = AlssmSum([alssm_m], [f_mp], force_MC=True)
            n0, n1 = offsets[m], offsets[m + 1]
            numdenom_pm = self._numdenom[dim_index][p][m]
            for i in range(_y.shape[0]):
                _xi_q_recursion(_xi_curr[i, :, n0:n1], q, wrapped, segment,
                                _y[i], _sample_weights[i],
                                beta_p, self._backend, self._filter_form, numdenom_pm)

    return xi_curr
