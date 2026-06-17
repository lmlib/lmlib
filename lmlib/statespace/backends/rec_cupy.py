r"""
GPU Backend for Recursive Least Squares Calculation using CuPy / cupyx
======================================================================

This backend is the GPU counterpart of
[`rec_lfilter`][lmlib.statespace.backends.rec_lfilter].  It mirrors the
**cascade** realization of the transfer-function backend one-to-one, but moves
the heavy per-sample work onto the GPU:

* the signal-shaping ``einsum`` contractions (``y_diff`` construction), and
* the per-state-dimension first-order IIR recursions, which are evaluated with
  [`cupyx.scipy.signal.lfilter`][cupyx.scipy.signal.lfilter] instead of
  [`scipy.signal.lfilter`][scipy.signal.lfilter].

The small, signal-independent state-space/window algebra
(``inv`` / ``matrix_power`` / ``kron`` on the ``N x N`` model matrices) is reused
unchanged from the CPU lfilter backend — those matrices are tiny and stay on the
host; only the per-sample arrays live on the device.

Scope
-----
This implementation targets **one-dimensional signals** with the **cascade**
filter form (upper-triangular ``A``) for ``q in {0, 1, 2}`` — i.e. exactly the
path exercised by ``RLSAlssm.filter`` for a 1-D ``CompositeCost`` /
``CostSegment``.  The N-dimensional asterisk-l recursion and the ``parallel``
filter form are intentionally **not** reimplemented on the GPU here; the
dispatcher in [`rec`][lmlib.statespace.backends.rec] routes those cases to the
NumPy backend so that any call remains correct.

Authors: GPU backend derived from the lfilter backend by
Christof Baeriswyl, Frederic Waldmann, Alexander Bertrand, Reto Wildhaber.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Optional GPU import.  The module must be importable on machines without CuPy
# or without a CUDA device (so that `import lmlib` never fails); the dispatcher
# only routes work here when CUPY_AVAILABLE is True.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    from cupyx.scipy.signal import lfilter as _cp_lfilter
    from cupyx.scipy.signal import sosfilt as _cp_sosfilt
    # A CuPy install with no visible device must NOT advertise itself.
    CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:                      # pragma: no cover - depends on hardware
    cp = None
    _cp_lfilter = None
    _cp_sosfilt = None
    CUPY_AVAILABLE = False

# Reuse the host-side cascade-parameter algebra + cache and the block-range
# helper from the CPU lfilter backend.  These operate on N x N model matrices
# only, so there is nothing to gain by moving them to the GPU, and sharing the
# cache means a model+window pair is factorised once for *both* backends.
from .rec_lfilter import _compute_cascade_params, _block_ranges, _poles_are_real

__all__ = [
    'cupy_cascade_xi0',
    'cupy_cascade_xi1',
    'cupy_cascade_xi2',
    'cupy_forward_cascade_xi',
    'cupy_backward_cascade_xi',
    'cupy_xi_q_recursion_batch',
    'cupy_parallel_xi1_split',
    'cupy_parallel_xi1',
    'cupy_parallel_xi0',
    'cupy_forward_parallel_xi',
    'cupy_backward_parallel_xi',
    'set_gpu_dtype',
    'get_gpu_dtype',
    'CUPY_AVAILABLE',
]


# ─────────────────────────────────────────────────────────────────────────────
# Device compute precision (lever #1)
# ─────────────────────────────────────────────────────────────────────────────
# Only the on-device arithmetic uses this dtype; the host accumulation buffers
# (xi / kappa / W) remain float64 and the float32 result is up-cast back into
# them on the final ``xi += asnumpy(...)``.  float32 is dramatically faster on
# consumer / laptop GPUs (whose FP64 throughput is heavily reduced) at the cost
# of ~1e-6 relative accuracy instead of ~1e-13.  Default stays float64 (exact
# parity with the lfilter backend).
_GPU_DTYPE = np.float64


def set_gpu_dtype(dtype):
    r"""
    Select the cupy backend's on-device compute precision.

    Parameters
    ----------
    dtype : {'float32', 'float64'} or numpy dtype
        ``'float64'`` (default) matches the lfilter backend to ~1e-13.
        ``'float32'`` is much faster on GPUs with reduced FP64 throughput
        (most consumer/laptop cards) at ~1e-6 relative accuracy.

    Notes
    -----
    Host buffers stay float64; only device math changes. Returns the active
    numpy scalar type.
    """
    global _GPU_DTYPE
    _GPU_DTYPE = np.dtype(dtype).type
    return _GPU_DTYPE


def get_gpu_dtype():
    r"""Return the active cupy-backend device compute dtype (numpy scalar type)."""
    return _GPU_DTYPE


def _dt():
    return _GPU_DTYPE


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _first_order_lfilter(pole, x):
    r"""
    Evaluate the first-order IIR recursion ``y[k] = x[k] + pole * y[k-1]`` on the
    GPU, i.e. ``lfilter([1, 0], [1, -pole], x)`` along the last axis.

    Parameters
    ----------
    pole : 0-d cupy.ndarray or float
        Recursion pole (diagonal entry of the cascade recursion matrix).
    x : cupy.ndarray
        Driving signal; filtered along ``axis=-1``.

    Returns
    -------
    cupy.ndarray
        Filtered signal, same shape as ``x``.
    """
    b = cp.asarray([1.0, 0.0], dtype=x.dtype)
    a = cp.empty(2, dtype=x.dtype)
    a[0] = 1.0
    a[1] = -pole
    return _cp_lfilter(b, a, x, axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# General forward cascade (GPU)
# ─────────────────────────────────────────────────────────────────────────────
def cupy_forward_cascade_xi(xi, cascade_params, a, b, y, sample_weights, beta,
                            block_sizes=None):
    r"""
    GPU forward calculation of :math:`\xi` (cascade of first-order IIR filters).

    Faithful port of
    [`lfilter_forward_cascade_xi`][lmlib.statespace.backends.rec_lfilter.lfilter_forward_cascade_xi]:
    same math, same indexing, same boundary handling — but the per-sample arrays
    and the IIR cascade run on the GPU.

    ``xi`` is the host (NumPy) accumulation buffer allocated by ``RLSAlssm``; it
    is updated **in place**.  Inputs are copied to the device on entry and the
    contribution of this segment is copied back and added into ``xi`` once, at
    the end (a single device-to-host transfer per call).

    Parameters
    ----------
    xi : numpy.ndarray, shape (K, N, [S])
        Accumulation target (host), updated in place.
    cascade_params : dict
        Output of ``_compute_cascade_params(..., direction='fw')`` (host arrays).
    a, b : int or inf
        Segment boundaries.
    y : numpy.ndarray or cupy.ndarray, shape (K, [L], [S])
        Weighted observations (or a broadcast 1 for the W recursion).
    sample_weights : numpy.ndarray, shape (K,)
        Per-sample weights.
    beta : float
        Segment weight.
    block_sizes : list or None
        Per-ALSSM state orders of the (block-diagonal) ``A``; cross-block
        feed-forward terms are skipped when provided.
    """
    gamma_a = cascade_params['gamma_a']
    gamma_b = cascade_params['gamma_b']
    gAinvT = cp.asarray(cascade_params['gAinvT'], dtype=_dt())
    Aac = cp.asarray(cascade_params['Aac'], dtype=_dt())
    Abc = cp.asarray(cascade_params['Abc'], dtype=_dt())
    N = cascade_params['N']

    if not cp.allclose(gAinvT, cp.tril(gAinvT)):
        raise ValueError("State-Space Matrix A needs to be upper triangular "
                         "for cascaded version")

    # Per-dimension IIR pole = diagonal of gamma^{-1} A^{-T}; remove the diagonal
    # from the (strictly lower-triangular) feed-forward coupling.
    poles = cp.diag(gAinvT).copy()
    gAinvT = gAinvT - cp.diagflat(cp.diag(gAinvT))

    y = cp.asarray(y, dtype=_dt())
    sample_weights = cp.asarray(sample_weights, dtype=_dt())
    y_weighted = y * sample_weights[:, None]
    K = y_weighted.shape[0]

    # ── boundary padding (identical logic to the CPU backend) ────────────────
    if not np.isinf(a):
        window_width = b - a + 1
        K_append = max(window_width, b + 1) if b >= 0 else window_width
    else:
        K_append = 0

    y_delayed_b = cp.zeros((K + K_append, *y_weighted.shape[1:]), dtype=_dt())
    y_delayed_b[0:K] = y_weighted
    y_diff = cp.einsum('kl, nl->kn', y_delayed_b, gamma_b * Abc)

    if not np.isinf(a):
        a_offset = b - a + 1
        y_delayed_a = cp.zeros((K + K_append, *y_weighted.shape[1:]), dtype=_dt())
        y_delayed_a[a_offset:a_offset + K] = y_weighted
        y_diff -= cp.einsum('kl, nl->kn', y_delayed_a, gamma_a * Aac)

    # ── cascade IIR loop (one lfilter per state dim n) ───────────────────────
    y_diff = cp.swapaxes(y_diff, 0, 1)                # (N, K+K_append, [S])
    xi_add = cp.zeros((K + K_append, *xi.shape[1:]), dtype=_dt(), order='F')
    for s, e in _block_ranges(block_sizes, N):
        xi_add[:, s] = _first_order_lfilter(poles[s], y_diff[s].T).T
        for n_ in range(s + 1, e):
            y_diff[n_, 1:] += cp.einsum('kn..., n->k...',
                                        xi_add[:-1, s:e], gAinvT[n_, s:e])
            xi_add[:, n_] = _first_order_lfilter(poles[n_], y_diff[n_].T).T

    if beta != 1:
        xi_add *= beta

    # ── add this segment's contribution back into the host buffer ────────────
    if b >= 0:
        xi += cp.asnumpy(xi_add[b:b + K])
    else:  # b < 0
        xi[-b:] += cp.asnumpy(xi_add[0:K + b])


# ─────────────────────────────────────────────────────────────────────────────
# General backward cascade (GPU)
# ─────────────────────────────────────────────────────────────────────────────
def cupy_backward_cascade_xi(xi, cascade_params, a, b, y, sample_weights, beta,
                             block_sizes=None):
    r"""
    GPU backward calculation of :math:`\xi`.

    Faithful port of
    [`lfilter_backward_cascade_xi`][lmlib.statespace.backends.rec_lfilter.lfilter_backward_cascade_xi]
    (see [`cupy_forward_cascade_xi`][lmlib.statespace.backends.rec_cupy.cupy_forward_cascade_xi]
    for the host/device conventions).
    """
    gamma_a = cascade_params['gamma_a']
    gamma_b = cascade_params['gamma_b']
    gAT = cp.asarray(cascade_params['gAT'], dtype=_dt())
    Aac = cp.asarray(cascade_params['Aac'], dtype=_dt())
    Abc = cp.asarray(cascade_params['Abc'], dtype=_dt())
    N = cascade_params['N']

    if not cp.allclose(gAT, cp.tril(gAT)):
        raise ValueError("State-Space Matrix A needs to be upper triangular "
                         "for cascaded version")

    poles = cp.diag(gAT).copy()
    gAT = gAT - cp.diagflat(cp.diag(gAT))

    K = len(xi)
    y = cp.asarray(y, dtype=_dt())
    sample_weights = cp.asarray(sample_weights, dtype=_dt())
    y_weighted = y * sample_weights[:, None]

    # time-reverse observation for backward recursion
    y_weighted_flipped = y_weighted[::-1]

    if not np.isinf(b):
        K_append = b - a + 1
    else:
        K_append = 0

    y_delayed_a = cp.zeros((K + K_append, *y_weighted_flipped.shape[1:]), dtype=_dt())
    y_delayed_a[0:K] = y_weighted_flipped
    y_diff = cp.einsum('kl, nl->kn', y_delayed_a, gamma_a * Aac)

    if not np.isinf(b):
        y_delayed_b = cp.zeros((K + K_append, *y_weighted_flipped.shape[1:]), dtype=_dt())
        y_delayed_b[K_append:] = y_weighted_flipped
        y_diff -= cp.einsum('kl, nl->kn', y_delayed_b, gamma_b * Abc)

    y_diff = cp.swapaxes(y_diff, 0, 1)
    xi_add = cp.zeros((K + K_append, *xi.shape[1:]), dtype=_dt(), order='F')
    for s, e in _block_ranges(block_sizes, N):
        xi_add[:, s] = _first_order_lfilter(poles[s], y_diff[s].T).T
        for n_ in range(s + 1, e):
            y_diff[n_, 1:] += cp.einsum('kn..., n->k...',
                                        xi_add[:-1, s:e], gAT[n_, s:e])
            xi_add[:, n_] = _first_order_lfilter(poles[n_], y_diff[n_].T).T

    if beta != 1:
        xi_add *= beta

    xi0_flipped = xi_add[::-1]
    if a >= 0:
        xi[0:K - a] += cp.asnumpy(xi0_flipped[-(K - a):])
    else:  # a < 0
        xi += cp.asnumpy(xi0_flipped[b + 1:K + b + 1])


# ─────────────────────────────────────────────────────────────────────────────
# xi^(q) cascade entry points (GPU)  — mirror lfilter_cascade_xi{0,1,2}
# ─────────────────────────────────────────────────────────────────────────────
def cupy_cascade_xi2(xi2, A, C, a, b, direction, delta, gamma, y,
                     sample_weights, beta):
    r"""
    GPU port of
    [`lfilter_cascade_xi2`][lmlib.statespace.backends.rec_lfilter.lfilter_cascade_xi2]:
    the :math:`W_k = \xi^{(2)}` recursion via the Kronecker substitution
    :math:`A \to A \otimes A`, :math:`C \to C \otimes C`, :math:`y \to 1`.

    Note: in steady-state mode (the default) ``RLSAlssm`` computes ``W``
    analytically and never reaches this function; it is provided for the
    time-varying ``W`` case (``steady_state=False``).
    """
    _A = np.kron(A, A)
    _C = np.kron(C, C)
    # the W recursion is signal-independent: feed a broadcast 1 (device-side).
    _y = cp.broadcast_to(cp.asarray(1.0, dtype=_dt()), np.shape(y))
    cascade_params = _compute_cascade_params(_A, _C, a, b, delta, gamma, direction)
    if direction == 'fw':
        cupy_forward_cascade_xi(xi2, cascade_params, a, b, _y, sample_weights, beta)
    elif direction == 'bw':
        cupy_backward_cascade_xi(xi2, cascade_params, a, b, _y, sample_weights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


def cupy_cascade_xi1(xi1, A, C, a, b, direction, delta, gamma, y,
                     sample_weights, beta, block_sizes=None):
    r"""
    GPU port of
    [`lfilter_cascade_xi1`][lmlib.statespace.backends.rec_lfilter.lfilter_cascade_xi1]:
    the signal-projection vector :math:`\xi_k = \xi^{(1)}(k, y)`.
    """
    cascade_params = _compute_cascade_params(A, C, a, b, delta, gamma, direction)
    if direction == 'fw':
        cupy_forward_cascade_xi(xi1, cascade_params, a, b, y, sample_weights,
                                beta, block_sizes)
    elif direction == 'bw':
        cupy_backward_cascade_xi(xi1, cascade_params, a, b, y, sample_weights,
                                 beta, block_sizes)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


def cupy_cascade_xi0(xi0, A, C, a, b, direction, delta, gamma, y,
                     sample_weights, beta):
    r"""
    GPU port of
    [`lfilter_cascade_xi0`][lmlib.statespace.backends.rec_lfilter.lfilter_cascade_xi0]:
    the weighted signal energy :math:`\kappa_k = \xi^{(0)}(k, y)` via the scalar
    substitution :math:`A \to [[1]]`, :math:`C \to [[1]]`, :math:`y \to y^2`.
    """
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = cp.asarray(y, dtype=_dt()) ** 2
    cascade_params = _compute_cascade_params(_A, _C, a, b, delta, gamma, direction)
    if direction == 'fw':
        cupy_forward_cascade_xi(xi0, cascade_params, a, b, _y, sample_weights, beta)
    elif direction == 'bw':
        cupy_backward_cascade_xi(xi0, cascade_params, a, b, _y, sample_weights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# ═════════════════════════════════════════════════════════════════════════════
# Batched (multi-channel) GPU cascade
# ═════════════════════════════════════════════════════════════════════════════
#
# The per-channel work in ``RLSAlssm._nd_xi_q_recursion`` is, for every other
# backend, a Python ``for`` loop over the flattened batch of signals — one
# backend call per channel.  On the GPU that is the worst case: B tiny kernel
# launches plus B device<->host transfers.
#
# These functions instead process the **whole batch in a single GPU sweep** by
# carrying the batch axis B as the trailing dimension: every ``lfilter`` runs
# along time for all B channels at once, and the signal-shaping einsums gain one
# batched index.  The cascade math is otherwise identical to the single-signal
# functions above (and is validated to match them and the lfilter backend).
#
# Buffer layout coming from rls.py (model_dimension already moved to -2 and the
# remaining signal axes flattened into B):
#     xi  : (B, K, Nx)   host (numpy) accumulation buffer, updated in place
#     y   : (B, K, Q)
#     sw  : (B, K)
# Internally we move B to the last axis -> (K, Q, B) / (K, Nx, B) so that
# cupyx.scipy.signal.lfilter (axis=-1 = time) and the einsums vectorise over B.
# ─────────────────────────────────────────────────────────────────────────────

GPU_MAX_BATCH = None
r"""int or None : hard cap on the number of channels processed per GPU sweep.
``None`` => auto-size from free GPU memory (with adaptive halving on
out-of-memory). Set an int to force a fixed maximum chunk."""

# Capture the cupy OOM exception type once (a harmless dummy under the test shim,
# which never raises it).
try:
    from cupy.cuda.memory import OutOfMemoryError as _OOM
except Exception:                                      # pragma: no cover
    class _OOM(Exception):
        pass


def _free_gpu_pool():
    """Return cached device blocks to the driver (best-effort)."""
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:                                  # pragma: no cover
        pass


def _auto_chunk(B, K_eff, Nx, Q):
    """
    Initial channels-per-sweep estimate from free GPU memory.

    Uses a deliberately conservative per-channel byte estimate for the
    forward/backward cascade intermediates (signal copies, ``y_diff``,
    ``xi_add``) plus head-room for cupyx ``lfilter``'s internal copy, and spends
    at most ~half of currently-free memory.  The adaptive halving in
    ``cupy_xi_q_recursion_batch`` corrects any underestimate.
    """
    bytes_per_chan = K_eff * np.dtype(_dt()).itemsize * (3 * Q + 4 * Nx + 6)
    try:
        free, _total = cp.cuda.Device().mem_info
    except Exception:                                  # pragma: no cover (shim)
        free = 2 << 30
    chunk = max(1, int(free * 0.5) // max(1, bytes_per_chan))
    if GPU_MAX_BATCH:
        chunk = min(chunk, int(GPU_MAX_BATCH))
    return max(1, min(chunk, B))


def _dispatch_q_batch(xi, q, alssm, segment, y, sample_weights, beta, block_sizes):
    """Route one batch chunk to the q-specific batched cascade."""
    A, C = alssm.A, alssm.C
    a, b = segment.a, segment.b
    direction, delta, gamma = segment.direction, segment.delta, segment.gamma
    if q == 2:
        _cupy_cascade_xi2_batch(xi, A, C, a, b, direction, delta, gamma,
                                y, sample_weights, beta)
    elif q == 1:
        _cupy_cascade_xi1_batch(xi, A, C, a, b, direction, delta, gamma,
                                y, sample_weights, beta, block_sizes)
    elif q == 0:
        _cupy_cascade_xi0_batch(xi, A, C, a, b, direction, delta, gamma,
                                y, sample_weights, beta)
    else:
        raise ValueError("q value not supported: '{}'".format(q))

def cupy_forward_cascade_xi_batch(xi, cascade_params, a, b, y, sample_weights,
                                  beta, block_sizes=None):
    r"""Batched GPU forward cascade. See module section header for layout."""
    gamma_a = cascade_params['gamma_a']
    gamma_b = cascade_params['gamma_b']
    gAinvT = cp.asarray(cascade_params['gAinvT'], dtype=_dt())
    Aac = cp.asarray(cascade_params['Aac'], dtype=_dt())
    Abc = cp.asarray(cascade_params['Abc'], dtype=_dt())
    N = cascade_params['N']

    if not cp.allclose(gAinvT, cp.tril(gAinvT)):
        raise ValueError("State-Space Matrix A needs to be upper triangular "
                         "for cascaded version")

    poles = cp.diag(gAinvT).copy()
    gAinvT = gAinvT - cp.diagflat(cp.diag(gAinvT))

    Nx = xi.shape[-1]
    # (B, K, Q) -> (K, Q, B)   ;   (B, K) -> (K, B)
    y = cp.moveaxis(cp.asarray(y, dtype=_dt()), 0, -1)
    sw = cp.moveaxis(cp.asarray(sample_weights, dtype=_dt()), 0, -1)
    y_weighted = y * sw[:, None, :]
    K = y_weighted.shape[0]
    B = y_weighted.shape[-1]

    if not np.isinf(a):
        window_width = b - a + 1
        K_append = max(window_width, b + 1) if b >= 0 else window_width
    else:
        K_append = 0

    y_delayed_b = cp.zeros((K + K_append, *y_weighted.shape[1:]), dtype=_dt())
    y_delayed_b[0:K] = y_weighted
    y_diff = cp.einsum('klB, nl->knB', y_delayed_b, gamma_b * Abc)

    if not np.isinf(a):
        a_offset = b - a + 1
        y_delayed_a = cp.zeros((K + K_append, *y_weighted.shape[1:]), dtype=_dt())
        y_delayed_a[a_offset:a_offset + K] = y_weighted
        y_diff -= cp.einsum('klB, nl->knB', y_delayed_a, gamma_a * Aac)

    y_diff = cp.swapaxes(y_diff, 0, 1)                 # (N, K+K_append, B)
    xi_add = cp.zeros((K + K_append, Nx, B), dtype=_dt(), order='F')
    for s, e in _block_ranges(block_sizes, N):
        xi_add[:, s] = _first_order_lfilter(poles[s], y_diff[s].T).T
        for n_ in range(s + 1, e):
            y_diff[n_, 1:] += cp.einsum('knB, n->kB',
                                        xi_add[:-1, s:e], gAinvT[n_, s:e])
            xi_add[:, n_] = _first_order_lfilter(poles[n_], y_diff[n_].T).T

    if beta != 1:
        xi_add *= beta

    # assemble the (K, Nx, B) contribution, then move batch back to front.
    if b >= 0:
        res = xi_add[b:b + K]
    else:
        res = cp.zeros((K, Nx, B), dtype=_dt())
        res[-b:] = xi_add[0:K + b]
    xi += cp.asnumpy(cp.moveaxis(res, -1, 0))          # (B, K, Nx)


def cupy_backward_cascade_xi_batch(xi, cascade_params, a, b, y, sample_weights,
                                   beta, block_sizes=None):
    r"""Batched GPU backward cascade. See module section header for layout."""
    gamma_a = cascade_params['gamma_a']
    gamma_b = cascade_params['gamma_b']
    gAT = cp.asarray(cascade_params['gAT'], dtype=_dt())
    Aac = cp.asarray(cascade_params['Aac'], dtype=_dt())
    Abc = cp.asarray(cascade_params['Abc'], dtype=_dt())
    N = cascade_params['N']

    if not cp.allclose(gAT, cp.tril(gAT)):
        raise ValueError("State-Space Matrix A needs to be upper triangular "
                         "for cascaded version")

    poles = cp.diag(gAT).copy()
    gAT = gAT - cp.diagflat(cp.diag(gAT))

    Nx = xi.shape[-1]
    K = xi.shape[1]
    y = cp.moveaxis(cp.asarray(y, dtype=_dt()), 0, -1)        # (K, Q, B)
    sw = cp.moveaxis(cp.asarray(sample_weights, dtype=_dt()), 0, -1)  # (K, B)
    y_weighted = y * sw[:, None, :]
    B = y_weighted.shape[-1]

    # time-reverse along the time axis (axis 0)
    y_weighted_flipped = y_weighted[::-1]

    if not np.isinf(b):
        K_append = b - a + 1
    else:
        K_append = 0

    y_delayed_a = cp.zeros((K + K_append, *y_weighted_flipped.shape[1:]), dtype=_dt())
    y_delayed_a[0:K] = y_weighted_flipped
    y_diff = cp.einsum('klB, nl->knB', y_delayed_a, gamma_a * Aac)

    if not np.isinf(b):
        y_delayed_b = cp.zeros((K + K_append, *y_weighted_flipped.shape[1:]), dtype=_dt())
        y_delayed_b[K_append:] = y_weighted_flipped
        y_diff -= cp.einsum('klB, nl->knB', y_delayed_b, gamma_b * Abc)

    y_diff = cp.swapaxes(y_diff, 0, 1)
    xi_add = cp.zeros((K + K_append, Nx, B), dtype=_dt(), order='F')
    for s, e in _block_ranges(block_sizes, N):
        xi_add[:, s] = _first_order_lfilter(poles[s], y_diff[s].T).T
        for n_ in range(s + 1, e):
            y_diff[n_, 1:] += cp.einsum('knB, n->kB',
                                        xi_add[:-1, s:e], gAT[n_, s:e])
            xi_add[:, n_] = _first_order_lfilter(poles[n_], y_diff[n_].T).T

    if beta != 1:
        xi_add *= beta

    xi0_flipped = xi_add[::-1]
    res = cp.zeros((K, Nx, B), dtype=_dt())
    if a >= 0:
        res[0:K - a] = xi0_flipped[-(K - a):]
    else:
        res = xi0_flipped[b + 1:K + b + 1]
    xi += cp.asnumpy(cp.moveaxis(res, -1, 0))


def _cupy_cascade_xi2_batch(xi2, A, C, a, b, direction, delta, gamma, y,
                            sample_weights, beta):
    _A = np.kron(A, A)
    _C = np.kron(C, C)
    B, K, _ = np.shape(y)
    _y = cp.ones((B, K, 1), dtype=_dt())
    cp_ = _compute_cascade_params(_A, _C, a, b, delta, gamma, direction)
    if direction == 'fw':
        cupy_forward_cascade_xi_batch(xi2, cp_, a, b, _y, sample_weights, beta)
    else:
        cupy_backward_cascade_xi_batch(xi2, cp_, a, b, _y, sample_weights, beta)


def _cupy_cascade_xi1_batch(xi1, A, C, a, b, direction, delta, gamma, y,
                            sample_weights, beta, block_sizes=None):
    cp_ = _compute_cascade_params(A, C, a, b, delta, gamma, direction)
    if direction == 'fw':
        cupy_forward_cascade_xi_batch(xi1, cp_, a, b, y, sample_weights, beta, block_sizes)
    else:
        cupy_backward_cascade_xi_batch(xi1, cp_, a, b, y, sample_weights, beta, block_sizes)


def _cupy_cascade_xi0_batch(xi0, A, C, a, b, direction, delta, gamma, y,
                            sample_weights, beta):
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = cp.asarray(y, dtype=_dt()) ** 2
    cp_ = _compute_cascade_params(_A, _C, a, b, delta, gamma, direction)
    if direction == 'fw':
        cupy_forward_cascade_xi_batch(xi0, cp_, a, b, _y, sample_weights, beta)
    else:
        cupy_backward_cascade_xi_batch(xi0, cp_, a, b, _y, sample_weights, beta)


def cupy_xi_q_recursion_batch(xi, q, alssm, segment, y, sample_weights, beta,
                              block_sizes=None):
    r"""
    Batched GPU :math:`\xi^{(q)}` recursion over a whole channel batch, with
    automatic memory-bounded chunking.

    Drop-in replacement for the per-channel ``for`` loop in
    ``RLSAlssm._nd_xi_q_recursion`` when ``backend='cupy'`` and
    ``filter_form='cascade'``: processes the ``B`` channels in **chunks**, each
    chunk handled in a single GPU sweep.  The chunk size is auto-sized from free
    GPU memory and halved adaptively if a chunk still runs out of memory, so a
    large ``K * B`` workload degrades gracefully instead of raising
    ``OutOfMemoryError``.  Set the module global ``GPU_MAX_BATCH`` to cap the
    channels-per-sweep explicitly.

    Each chunk is independent (the only host write is a single ``xi[chunk] +=``
    at the very end of the cascade), so a chunk that OOMs mid-computation can be
    safely retried at a smaller size without double-counting.

    Parameters
    ----------
    xi : numpy.ndarray, shape (B, K, N**q)
        Host accumulation buffer (updated in place).
    q : int
        Recursion order (0, 1, or 2).
    alssm : ModelBase
        Combined ALSSM (provides A, C).
    segment : Segment
        Window parameters and direction.
    y : numpy.ndarray, shape (B, K, Q)
        Batched input signal.
    sample_weights : numpy.ndarray, shape (B, K)
        Per-sample weights.
    beta : float
        Cost scaling factor.
    block_sizes : list or None
        Per-ALSSM state orders (q==1 cascade).
    """
    B = xi.shape[0]
    K = xi.shape[1]
    Nx = xi.shape[-1]
    Q = np.shape(y)[-1]
    chunk = _auto_chunk(B, K, Nx, Q)

    lo = 0
    while lo < B:
        hi = min(B, lo + chunk)
        try:
            _dispatch_q_batch(xi[lo:hi], q, alssm, segment,
                              y[lo:hi], sample_weights[lo:hi], beta, block_sizes)
            lo = hi
        except _OOM:
            _free_gpu_pool()
            if chunk == 1:
                raise  # a single channel does not fit — nothing more we can do
            chunk = max(1, chunk // 2)  # retry the same range with a smaller chunk


# ═════════════════════════════════════════════════════════════════════════════
# Parallel filter form (GPU)
# ═════════════════════════════════════════════════════════════════════════════
#
# GPU port of the 1-D ``parallel`` realization of rec_lfilter
# (``lfilter_forward_parallel_xi`` / ``lfilter_backward_parallel_xi`` and the
# ``lfilter_parallel_xi{0,1}`` entry points).  The per-ALSSM transfer-function
# *plan* (the per-row SOS / FIR / pole structure) is built once on the host by
# ``build_parallel_numdenom`` and cached; this module only moves the per-row
# filtering — the FIR numerator passes, the IIR ``sosfilt``/gamma-shift, and the
# branch subtraction — onto the GPU.
#
# Scope matches the CPU parallel backend: q==1 is the genuine parallel
# realization; q==0 delegates to the (cascade) energy recursion; q==2 (parallel
# W) is not implemented (steady-state mode avoids it).  Per-row pole/zero
# classification (``_poles_are_real``) stays on the host on the tiny SOS arrays.
# ─────────────────────────────────────────────────────────────────────────────

# A custom sequential biquad-cascade kernel. cupyx's lfilter AND sosfilt both
# use an internal *parallel scan* (apply_iir) that forms cumulative pole-products
# and overflows / loses precision on the stiff, high-gain, near-unit/complex-pole
# filters of the parallel form (where the CPU's sequential recursion stays
# bounded). This kernel recurses sequentially in time — exactly like scipy — and
# parallelizes across the *batch* of independent series (channels) instead, which
# is the numerically stable formulation.
GPU_PARALLEL_STABLE_IIR = False
r"""bool : Parallel-form IIR strategy. ``False`` (default) uses
``cupyx.scipy.signal.sosfilt`` (fast, parallel scan, accurate for stable
filters, bit-compatible with the CPU path). ``True`` uses the sequential biquad
``RawKernel`` (slower per series but numerically bounded for stiff / marginally
unstable filters where the scan overflows)."""
_SOS_KERNEL_SRC = r'''
extern "C" __global__
void sos_cascade(const {ctype}* __restrict__ x, {ctype}* __restrict__ y,
                 const long L, const long M, const int nsec,
                 const {ctype}* __restrict__ sos) {{
    long m = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    {ctype} w1[8]; {ctype} w2[8];          // Direct-Form-II-transposed states
    for (int s = 0; s < nsec; ++s) {{ w1[s] = ({ctype})0; w2[s] = ({ctype})0; }}
    for (long k = 0; k < L; ++k) {{
        {ctype} v = x[k * M + m];
        for (int s = 0; s < nsec; ++s) {{
            const {ctype}* c = sos + s * 6;     // [b0,b1,b2,a0(=1),a1,a2]
            {ctype} out = c[0] * v + w1[s];
            w1[s] = c[1] * v - c[4] * out + w2[s];
            w2[s] = c[2] * v - c[5] * out;
            v = out;
        }}
        y[k * M + m] = v;
    }}
}}
'''

_SOS_KERNELS = {}
# RawKernel only exists with a real cupy; under the numpy test-shim we fall back
# to a sequential scipy lfilter-per-biquad (also stable) so logic stays testable.
_HAS_RAWKERNEL = (cp is not None) and hasattr(cp, 'RawKernel')


def _get_sos_kernel():
    key = np.dtype(_dt()).name
    if key not in _SOS_KERNELS:
        ctype = 'float' if key == 'float32' else 'double'
        _SOS_KERNELS[key] = cp.RawKernel(_SOS_KERNEL_SRC.format(ctype=ctype), 'sos_cascade')
    return _SOS_KERNELS[key]


def _cupy_sos_apply(sos, x):
    r"""
    Apply a Second-Order-Sections filter along time (axis 0) to ``x`` (shape
    ``(L,)`` or ``(L, M)``).

    Default (``GPU_PARALLEL_STABLE_IIR == False``): ``cupyx.scipy.signal.sosfilt``
    — high throughput and bit-compatible with the CPU ``sosfilt`` path. This is
    accurate for stable filters (the parallel form's intended use).

    Opt-in (``GPU_PARALLEL_STABLE_IIR == True``): a custom sequential biquad
    kernel (Direct-Form-II-transposed, ``cp.RawKernel``) that recurses in time
    like scipy instead of using ``cupyx``'s internal parallel scan. Slower for a
    single series, but bounded for stiff / marginally-unstable filters where the
    scan can overflow. Under the numpy test-shim it falls back to a sequential
    scipy ``lfilter``-per-biquad.
    """
    sos = np.asarray(sos, dtype=np.float64)

    if not GPU_PARALLEL_STABLE_IIR:
        return _cp_sosfilt(cp.asarray(sos[:, :6], dtype=_dt()), x, axis=0)

    # ---- stable sequential path (opt-in) ----
    nsec = int(sos.shape[0])
    if nsec > 8:                       # kernel holds up to 8 sections in registers
        out = x
        for i in range(0, nsec, 8):
            out = _cupy_sos_apply(sos[i:i + 8], out)
        return out

    if not _HAS_RAWKERNEL:
        # test-shim / no real GPU: sequential lfilter-per-biquad (scipy) — stable.
        out = x
        for sec in sos:
            out = _cp_lfilter(cp.asarray(sec[0:3], dtype=_dt()),
                              cp.asarray(sec[3:6], dtype=_dt()), out)
        return out

    one_d = (x.ndim == 1)
    x2 = cp.ascontiguousarray(x[:, None] if one_d else x, dtype=_dt())
    L, M = int(x2.shape[0]), int(x2.shape[1])
    y = cp.empty_like(x2)
    sos_d = cp.ascontiguousarray(cp.asarray(sos[:, :6], dtype=_dt()).ravel())
    kern = _get_sos_kernel()
    threads = 128
    blocks = (M + threads - 1) // threads
    kern((blocks,), (threads,),
         (x2, y, np.int64(L), np.int64(M), np.int32(nsec), sos_d))
    return y[:, 0] if one_d else y


def _cupy_apply_fir(sos, extra_delay, y_sig, Lout):
    r"""GPU analogue of ``_apply_fir``: numerator-only SOS pass + integer delay."""
    result = cp.zeros(Lout, dtype=_dt())
    if sos is None:
        return result
    filtered = _cupy_sos_apply(sos, y_sig)
    end = min(extra_delay + len(filtered), Lout)
    result[extra_delay:end] = filtered[:end - extra_delay]
    return result


def cupy_forward_parallel_xi(xi, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                             a, b, delta, gamma, y, sample_weights, beta,
                             sos_iir_b_list=None, sos_iir_a_list=None,
                             n_poles_b_list=None, n_poles_a_list=None):
    r"""GPU forward parallel xi filter. Faithful port of ``lfilter_forward_parallel_xi``."""
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)
    y = cp.asarray(y, dtype=_dt())
    sample_weights = cp.asarray(sample_weights, dtype=_dt())
    y_weighted = (y * sample_weights[:, None]).ravel()
    K = int(y_weighted.shape[0])
    N = xi.shape[1]
    K_append = (b - a + 1) if not np.isinf(a) else 0
    L = K + K_append

    y_db = cp.zeros(L, dtype=_dt())
    y_db[:K] = y_weighted * gamma_b
    y_da = cp.zeros(L, dtype=_dt())
    if not np.isinf(a):
        y_da[K_append:K + K_append] = y_weighted * gamma_a

    use_per_row_iir = (sos_iir_b_list is not None and sos_iir_a_list is not None)

    for n_ in range(N):
        Lout = L + max(db_list[n_], da_list[n_]) + 1
        fb = _cupy_apply_fir(sos_b_list[n_], db_list[n_], y_db, Lout)
        fa = _cupy_apply_fir(sos_a_list[n_], da_list[n_], y_da, Lout)
        if use_per_row_iir:
            # per-row reduced IIR (from QZ-based PZ cancellation), each branch
            # filtered with its own SOS via the stable biquad-lfilter path.
            ib = _cupy_sos_apply(sos_iir_b_list[n_], fb)
            ia = _cupy_sos_apply(sos_iir_a_list[n_], fa)
            iir = ib - ia
        else:
            iir = _cupy_sos_apply(sos_iir, fb - fa)

        if beta != 1:
            iir = iir * beta

        if b >= 0:
            xi[:, n_] += cp.asnumpy(iir[b:b + K])
        else:
            xi[-b:, n_] += cp.asnumpy(iir[0:K + b])


def cupy_backward_parallel_xi(xi, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                              a, b, delta, gamma, y, sample_weights, beta,
                              sos_iir_b_list=None, sos_iir_a_list=None,
                              n_poles_b_list=None, n_poles_a_list=None,
                              advance_b_list=None, advance_a_list=None):
    r"""GPU backward parallel xi filter. Faithful port of ``lfilter_backward_parallel_xi``."""
    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)
    y = cp.asarray(y, dtype=_dt())
    sample_weights = cp.asarray(sample_weights, dtype=_dt())
    y_weighted = (y * sample_weights[:, None]).ravel()
    K = int(y_weighted.shape[0])
    N = xi.shape[1]
    K_append = (b - a + 1) if not np.isinf(a) else 0
    L = K + K_append
    y_weighted_flipped = y_weighted[::-1]

    y_da = cp.zeros(L, dtype=_dt())
    y_da[:K] = y_weighted_flipped * gamma_a
    y_db = cp.zeros(L, dtype=_dt())
    if not np.isinf(b):
        y_db[K_append:K + K_append] = y_weighted_flipped * gamma_b

    use_per_row_iir = (sos_iir_b_list is not None and sos_iir_a_list is not None)

    for n_ in range(N):
        Lout = L + max(db_list[n_], da_list[n_]) + 1
        fa = _cupy_apply_fir(sos_a_list[n_], da_list[n_], y_da, Lout)
        fb = _cupy_apply_fir(sos_b_list[n_], db_list[n_], y_db, Lout)
        if use_per_row_iir:
            ia = _cupy_sos_apply(sos_iir_a_list[n_], fa)
            ib = _cupy_sos_apply(sos_iir_b_list[n_], fb)
            iir = ia - ib
        else:
            iir = _cupy_sos_apply(sos_iir, fa - fb)

        if beta != 1:
            iir = iir * beta

        if a >= 0:
            end = K - a
            if end > 0:
                xi[:end, n_] += cp.asnumpy(iir[0:end][::-1])
        else:
            xi[:, n_] += cp.asnumpy(iir[-a:K - a][::-1])


def cupy_parallel_xi1(xi1, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                      a, b, direction, delta, gamma, y, sample_weights, beta,
                      sos_iir_b_list=None, sos_iir_a_list=None,
                      n_poles_b_list=None, n_poles_a_list=None,
                      advance_b_list=None, advance_a_list=None):
    r"""GPU parallel :math:`\xi^{(1)}` for one ALSSM block. Port of ``lfilter_parallel_xi1``."""
    if direction == 'fw':
        cupy_forward_parallel_xi(xi1, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                                 a, b, delta, gamma, y, sample_weights, beta,
                                 sos_iir_b_list, sos_iir_a_list,
                                 n_poles_b_list, n_poles_a_list)
    elif direction == 'bw':
        cupy_backward_parallel_xi(xi1, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                                  a, b, delta, gamma, y, sample_weights, beta,
                                  sos_iir_b_list, sos_iir_a_list,
                                  n_poles_b_list, n_poles_a_list,
                                  advance_b_list, advance_a_list)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


def cupy_parallel_xi1_split(xi1, plan, a, b, direction, delta, gamma, y, sample_weights, beta):
    r"""
    Per-ALSSM parallel :math:`\xi^{(1)}` recursion on the GPU.

    Port of ``lfilter_parallel_xi1_split``: ``plan`` is the list produced by
    ``build_parallel_numdenom`` (host, cached); each active ALSSM block is
    filtered independently into its sub-slice ``xi1[..., n0:n1]``.
    """
    for n0, n1, nd in plan:
        _iir_b = nd[5] if len(nd) > 5 else None
        _iir_a = nd[6] if len(nd) > 6 else None
        _np_b = nd[7] if len(nd) > 7 else None
        _np_a = nd[8] if len(nd) > 8 else None
        _adv_b = nd[9] if len(nd) > 9 else None
        _adv_a = nd[10] if len(nd) > 10 else None
        cupy_parallel_xi1(xi1[..., n0:n1], nd[0], nd[1], nd[2], nd[3], nd[4],
                          a, b, direction, delta, gamma, y, sample_weights, beta,
                          _iir_b, _iir_a, _np_b, _np_a, _adv_b, _adv_a)


def cupy_parallel_xi0(xi0, denom, num_b, num_a, a, b, direction, delta, gamma,
                      y, sample_weights, beta, kappa_diag=True):
    r"""
    GPU parallel :math:`\xi^{(0)}` (signal energy). Port of ``lfilter_parallel_xi0``:
    the ALSSM is not used, so this delegates to the GPU cascade energy recursion.
    """
    cupy_cascade_xi0(xi0, np.ones((1, 1)), np.ones((1, 1)),
                     a, b, direction, delta, gamma, y, sample_weights, beta)


# ═════════════════════════════════════════════════════════════════════════════
# N-dimensional (asterisk-l) recursions on GPU  —  xi^{(1)*l}, xi^{(0)*l}
# ═════════════════════════════════════════════════════════════════════════════
#
# GPU ports of the lfilter asterisk recursions. They realize the
# cross-dimensional Kronecker recursion (A_ast = I_{Nprev} (x) A) and are driven
# by xi_N (the accumulated cost vector from the lower-dimensional recursion)
# instead of the raw signal. The (tiny, model-only) parameter builders
# (_compute_cascade_params_asterisk, _build_parallel_ast_sos) are reused from the
# CPU backend on the host; only the per-sample filtering moves to the GPU.
#
# The cascade asterisk uses first-order lfilter (the validated-stable cupyx
# path); the parallel asterisk reuses _cupy_sos_apply. xi is the host (numpy)
# accumulation buffer, updated in place with a single device->host copy.
# ─────────────────────────────────────────────────────────────────────────────

from .rec_lfilter import _compute_cascade_params_asterisk, _build_parallel_ast_sos


def cupy_xi_asterisk_l_forward_cascade(xi, cascade_params_ast, a, b, xi_N, v, beta):
    r"""GPU forward cascade for xi^{(1)*l}. Port of lfilter_xi_asterisk_l_forward_cascade_recursion."""
    gamma_a = cascade_params_ast['gamma_a']
    gamma_b = cascade_params_ast['gamma_b']
    gAinvT = cp.asarray(cascade_params_ast['gAinvT'], dtype=_dt())
    Aac = cp.asarray(cascade_params_ast['Aac'], dtype=_dt())
    Abc = cp.asarray(cascade_params_ast['Abc'], dtype=_dt())
    N = cascade_params_ast['N']
    Nprev = cascade_params_ast['Nprev']
    if not cp.allclose(gAinvT, cp.tril(gAinvT)):
        raise ValueError("State-Space Matrix A needs to be upper triangular for cascaded version")
    poles = cp.diag(gAinvT)

    xi_N = cp.asarray(xi_N, dtype=_dt())
    K = xi_N.shape[0]
    S_shape = xi_N.shape[1:-1]
    if not np.isinf(a):
        window_width = b - a + 1
        K_append = max(window_width, b + 1) if b >= 0 else window_width
    else:
        K_append = 0
    L = K + K_append

    xi_N_b = cp.zeros((L, *S_shape, Nprev), dtype=_dt())
    xi_N_b[:K] = xi_N
    y_diff = xi_N_b[..., :, None] * (gamma_b * Abc)          # (L, *S, Nprev, N)
    if not np.isinf(a):
        a_offset = b - a + 1
        xi_N_a = cp.zeros((L, *S_shape, Nprev), dtype=_dt())
        xi_N_a[a_offset:a_offset + K] = xi_N
        y_diff = y_diff - xi_N_a[..., :, None] * (gamma_a * Aac)

    xi_add = cp.zeros((L, *S_shape, Nprev, N), dtype=_dt())
    for n in range(N):
        if n > 0:
            y_diff[1:, ..., n] += cp.einsum('...m,m->...', xi_add[:-1, ..., :n], gAinvT[n, :n])
        inp = y_diff[..., n].reshape(L, -1)
        out = _first_order_lfilter(poles[n], inp.T).T
        xi_add[..., n] = out.reshape(L, *S_shape, Nprev)

    if beta != 1:
        xi_add = xi_add * beta
    xi_add = xi_add.reshape(L, *S_shape, Nprev * N)          # trailing = n_prev*N + n
    if b >= 0:
        xi += cp.asnumpy(xi_add[b:b + K])
    else:
        xi[-b:] += cp.asnumpy(xi_add[0:K + b])


def cupy_xi_asterisk_l_backward_cascade(xi, cascade_params_ast, a, b, xi_N, v, beta):
    r"""GPU backward cascade for xi^{(1)*l}. Port of lfilter_xi_asterisk_l_backward_cascade_recursion."""
    gamma_a = cascade_params_ast['gamma_a']
    gamma_b = cascade_params_ast['gamma_b']
    gAT = cp.asarray(cascade_params_ast['gAT'], dtype=_dt())
    Aac = cp.asarray(cascade_params_ast['Aac'], dtype=_dt())
    Abc = cp.asarray(cascade_params_ast['Abc'], dtype=_dt())
    N = cascade_params_ast['N']
    Nprev = cascade_params_ast['Nprev']
    if not cp.allclose(gAT, cp.tril(gAT)):
        raise ValueError("State-Space Matrix A needs to be upper triangular for cascaded version")
    poles = cp.diag(gAT)

    xi_N = cp.asarray(xi_N, dtype=_dt())
    K = xi_N.shape[0]
    S_shape = xi_N.shape[1:-1]
    K_append = (b - a + 1) if not np.isinf(b) else 0
    L = K + K_append
    xi_N_flipped = xi_N[::-1]

    xi_N_a = cp.zeros((L, *S_shape, Nprev), dtype=_dt())
    xi_N_a[:K] = xi_N_flipped
    y_diff = xi_N_a[..., :, None] * (gamma_a * Aac)
    if not np.isinf(b):
        xi_N_b = cp.zeros((L, *S_shape, Nprev), dtype=_dt())
        xi_N_b[K_append:] = xi_N_flipped
        y_diff = y_diff - xi_N_b[..., :, None] * (gamma_b * Abc)

    xi_add = cp.zeros((L, *S_shape, Nprev, N), dtype=_dt())
    for n in range(N):
        if n > 0:
            y_diff[1:, ..., n] += cp.einsum('...m,m->...', xi_add[:-1, ..., :n], gAT[n, :n])
        inp = y_diff[..., n].reshape(L, -1)
        out = _first_order_lfilter(poles[n], inp.T).T
        xi_add[..., n] = out.reshape(L, *S_shape, Nprev)

    if beta != 1:
        xi_add = xi_add * beta
    xi_add = xi_add.reshape(L, *S_shape, Nprev * N)
    xi_add_flipped = xi_add[::-1]
    if a >= 0:
        end = K - a
        if end > 0:
            xi[:end] += cp.asnumpy(xi_add_flipped[-end:])
    else:
        xi += cp.asnumpy(xi_add_flipped[b + 1:K + b + 1])


def cupy_xi_asterisk_l_forward_parallel(xi, nd, a, b, delta, gamma, xi_N, v, beta):
    r"""GPU forward parallel for xi^{(1)*l}. Port of lfilter_xi_asterisk_l_forward_parallel_recursion."""
    K = xi_N.shape[0]
    Nprev = xi_N.shape[-1]
    S_shape = xi_N.shape[1:-1]
    S_flat = int(np.prod(S_shape)) if S_shape else 1
    N = xi.shape[-1] // Nprev

    sos_iir = nd[0]; sos_b_list = nd[1]; sos_a_list = nd[2]
    db_list = nd[3]; da_list = nd[4]
    sos_iir_b_list = nd[5] if len(nd) > 5 else None
    sos_iir_a_list = nd[6] if len(nd) > 6 else None

    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)
    K_append = (max(b - a + 1, b + 1) if (not np.isinf(a) and b >= 0)
                else (b - a + 1 if not np.isinf(a) else 0))
    L = K + K_append

    xi_N = cp.asarray(xi_N, dtype=_dt())
    xi_N_2d = xi_N.reshape(K, S_flat * Nprev)
    y_db = cp.zeros((L, S_flat * Nprev), dtype=_dt())
    y_db[:K] = xi_N_2d * gamma_b
    y_da = cp.zeros((L, S_flat * Nprev), dtype=_dt())
    if not np.isinf(a):
        y_da[K_append:K + K_append] = xi_N_2d * gamma_a

    use_per_row_iir = (sos_iir_b_list is not None and sos_iir_a_list is not None)
    out_acc = cp.zeros((K, S_flat * Nprev, N), dtype=_dt())
    for n_ in range(N):
        Lout = L + max(db_list[n_], da_list[n_]) + 1
        fb = _cupy_apply_fir(sos_b_list[n_], db_list[n_], y_db, Lout)
        fa = _cupy_apply_fir(sos_a_list[n_], da_list[n_], y_da, Lout)
        if use_per_row_iir:
            iir = _cupy_sos_apply(sos_iir_b_list[n_], fb) - _cupy_sos_apply(sos_iir_a_list[n_], fa)
        else:
            iir = _cupy_sos_apply(sos_iir, fb - fa)
        if beta != 1:
            iir = iir * beta
        if b >= 0:
            out_acc[:, :, n_] = iir[b:b + K]
        else:
            out_acc[-b:, :, n_] = iir[0:K + b]

    out_full = out_acc.reshape(K, S_flat, Nprev, N).reshape(K, *S_shape, Nprev * N)
    xi += cp.asnumpy(out_full)


def cupy_xi_asterisk_l_backward_parallel(xi, nd, a, b, delta, gamma, xi_N, v, beta):
    r"""GPU backward parallel for xi^{(1)*l}. Port of lfilter_xi_asterisk_l_backward_parallel_recursion."""
    K = xi_N.shape[0]
    Nprev = xi_N.shape[-1]
    S_shape = xi_N.shape[1:-1]
    S_flat = int(np.prod(S_shape)) if S_shape else 1
    N = xi.shape[-1] // Nprev

    sos_iir = nd[0]; sos_b_list = nd[1]; sos_a_list = nd[2]
    db_list = nd[3]; da_list = nd[4]
    sos_iir_b_list = nd[5] if len(nd) > 5 else None
    sos_iir_a_list = nd[6] if len(nd) > 6 else None

    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)
    K_append = (b - a + 1) if not np.isinf(b) else 0
    L = K + K_append

    xi_N = cp.asarray(xi_N, dtype=_dt())
    xi_N_2d = xi_N.reshape(K, S_flat * Nprev)
    xi_N_flipped = xi_N_2d[::-1]

    y_da = cp.zeros((L, S_flat * Nprev), dtype=_dt())
    y_da[:K] = xi_N_flipped * gamma_a
    y_db = cp.zeros((L, S_flat * Nprev), dtype=_dt())
    if not np.isinf(b):
        y_db[K_append:] = xi_N_flipped * gamma_b

    use_per_row_iir = (sos_iir_b_list is not None and sos_iir_a_list is not None)
    out_acc = cp.zeros((K, S_flat * Nprev, N), dtype=_dt())
    for n_ in range(N):
        Lout = L + max(db_list[n_], da_list[n_]) + 1
        fa = _cupy_apply_fir(sos_a_list[n_], da_list[n_], y_da, Lout)
        fb = _cupy_apply_fir(sos_b_list[n_], db_list[n_], y_db, Lout)
        if use_per_row_iir:
            iir = _cupy_sos_apply(sos_iir_a_list[n_], fa) - _cupy_sos_apply(sos_iir_b_list[n_], fb)
        else:
            iir = _cupy_sos_apply(sos_iir, fa - fb)
        if beta != 1:
            iir = iir * beta
        iir_flipped = iir[::-1]
        if a >= 0:
            end = K - a
            if end > 0:
                out_acc[:end, :, n_] = iir_flipped[-end:]
        else:
            out_acc[:, :, n_] = iir_flipped[b + 1:K + b + 1]

    out_full = out_acc.reshape(K, S_flat, Nprev, N).reshape(K, *S_shape, Nprev * N)
    xi += cp.asnumpy(out_full)


def cupy_parallel_xi_asterisk_split(xi_curr, A, C, a, b, delta, gamma, direction,
                                    xi_prev, v, beta, block_sizes):
    r"""
    Per-ALSSM-block parallel asterisk-l recursion on GPU. Port of
    ``lfilter_parallel_xi_asterisk_split``: each ALSSM block's transfer function
    is built (host) and filtered (device) independently, then scattered into the
    Kronecker layout of ``xi_curr``.
    """
    Nq_prev = xi_prev.shape[-1]
    N = A.shape[0]
    for n0, n1 in _block_ranges(block_sizes, N):
        C_m = C[:, n0:n1]
        if not np.any(C_m):
            continue
        A_m = A[n0:n1, n0:n1]
        N_m = n1 - n0
        nd = _build_parallel_ast_sos(A_m, C_m, a, b, delta, gamma, direction)
        xi_tmp = np.zeros((*xi_prev.shape[:-1], Nq_prev * N_m))
        if direction == 'fw':
            cupy_xi_asterisk_l_forward_parallel(xi_tmp, nd, a, b, delta, gamma, xi_prev, v, beta)
        else:
            cupy_xi_asterisk_l_backward_parallel(xi_tmp, nd, a, b, delta, gamma, xi_prev, v, beta)
        for n_prev in range(Nq_prev):
            xi_curr[..., n_prev * N + n0: n_prev * N + n1] += \
                xi_tmp[..., n_prev * N_m: (n_prev + 1) * N_m]


def cupy_xi_q_asterisk_l_recursion(xi_curr, q, A, C, a, b, direction, delta, gamma,
                                   xi_prev, v, beta, filter_form, block_sizes=None):
    r"""
    GPU dispatcher for the ND asterisk-l recursion, mirroring the lfilter path:
    parallel q==1 uses the per-block parallel realization; cascade (and parallel
    q==0) use the cascade asterisk (upper-triangular A). q==2 is not handled here
    (caller falls back to numpy).
    """
    Nprev = xi_prev.shape[-1]
    if filter_form == 'parallel' and q == 1:
        cupy_parallel_xi_asterisk_split(xi_curr, A, C, a, b, delta, gamma,
                                        direction, xi_prev, v, beta, block_sizes)
        return
    # cascade realization (q in {0, 1}, and parallel q==0)
    cp_ast = _compute_cascade_params_asterisk(A, C, a, b, delta, gamma, Nprev, direction)
    if direction == 'fw':
        cupy_xi_asterisk_l_forward_cascade(xi_curr, cp_ast, a, b, xi_prev, v, beta)
    else:
        cupy_xi_asterisk_l_backward_cascade(xi_curr, cp_ast, a, b, xi_prev, v, beta)
