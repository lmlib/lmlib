"""
Pre-optimization implementations of the lfilter cascade and RLS recursion.

These replicate the behaviour of lmlib *before* the F-order / in-place
optimisation was applied:

  - C-order scratch buffer ``xi0``       (instead of pre-allocated F-order)
  - Accumulation via ``xi += xi0`` copy  (instead of in-place fill)
  - Shared ``xi_curr`` across segments   (instead of per-segment ``_xi_seg``)

Kept as a standalone module so that ``inspect.getsource()`` works correctly
when the functions are registered with ``line_profiler.LineProfiler``.
"""

import numpy as np
from numpy.linalg import inv as _inv, matrix_power as _mpow
from scipy.signal import lfilter as _lfilter
from lmlib.statespace.backends.rec import xi_q_recursion as _xi_q_recursion


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
            _xi_q_recursion(_xi_curr[i], q,
                            cs.alssm, cs.segment,
                            _y[i], _sample_weights[i],
                            cs.beta, self._backend, self._filter_form)

    return xi_curr
