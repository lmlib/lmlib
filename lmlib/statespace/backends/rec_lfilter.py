"""
Backend for Recursive Least Squares Calculation using lfilter
===========================================================

Resources
---------
Authors: Christof Baeriswyl, Frédéric Waldmann, Alexander Bertrand, Reto Wildhaber

"""


import numpy as np
from numpy.linalg import inv, matrix_power, eigvals
from scipy.signal import lfilter, convolve, zpk2sos, sosfilt, ss2tf, sos2tf
from lmlib.utils.profiling import profile


# ─────────────────────────────────────────────────────────────────────────────
# Cascade-parameter cache (module-level / static)
# ─────────────────────────────────────────────────────────────────────────────
#
# ``_compute_cascade_params`` turns a ``(A, C, a, b, delta, gamma, direction)``
# tuple into the precomputed ``inv``/``matrix_power``/``np.dot`` scalars that the
# cascade IIR filters consume.  Those inputs depend only on the ALSSM model and
# the segment window — never on the signal ``y`` or the sample weights — so the
# result is identical:
#
#   * across the inner per-signal-slice loop of one ``filter()`` call,
#   * across repeated ``filter()`` calls on the same RLSAlssm object, and
#   * across *different* RLSAlssm objects that happen to share the same model
#     and window (same A, C, a, b, delta, gamma, direction).
#
# Recomputing it every time (the previous behaviour) wasted ``inv`` /
# ``matrix_power`` calls.  The old design cached the dict per RLSAlssm instance
# (``self._cascade_params[dim][seg][alssm]``); because the cache key is fully
# determined by the model+window — not by which object asked — a *single static
# cache shared by the whole lfilter cascade backend* is both simpler and strictly
# more sharing: two RLSAlssm objects with identical models now compute the params
# once between them.  The cache is therefore deliberately NOT an instance
# attribute; it lives here, next to the function that fills it.
#
# The cached dicts are treated as read-only by every consumer
# (``lfilter_forward_cascade_xi`` / ``lfilter_backward_cascade_xi`` only read the
# entries), so handing out the shared object is safe.
#
# Lifetime: unbounded for the process lifetime.  Entries are tiny (a handful of
# small arrays + scalars) and the number of distinct (model, window) keys in any
# real program is small and bounded, so this does not grow without limit in
# practice.  ``clear_lfilter_caches()`` is provided for tests and for the
# rare caller that wants to reclaim the memory explicitly.

_CASCADE_PARAMS_CACHE = {}

# Companion cache for the asterisk-l recursion (Eq. 47).  Its result is computed
# by ``_compute_cascade_params_asterisk`` and depends on the same model+window
# inputs *plus* ``Nprev`` (the trailing state dimension of ``xi_prev``), so it
# needs its own key shape and is kept in a separate dict.  Same rationale,
# lifetime and read-only-sharing guarantees as ``_CASCADE_PARAMS_CACHE`` above.
_CASCADE_PARAMS_ASTERISK_CACHE = {}

# Companion cache for the parallel xi^(1) realization.  ``build_parallel_numdenom``
# turns the same model+window inputs (plus ``block_sizes``) into per-ALSSM
# transfer-function (SOS) coefficients via QZ + pole-zero cancellation, which is
# comparatively expensive.  Unlike the cascade params it is already built only
# once per RLSAlssm object (in ``RLSAlssm._build_parallel_plan`` at construction
# time, then reused across every ``filter()`` call), so this cache does NOT
# remove a per-slice/per-filter hot loop — its sole benefit is sharing the plan
# across *different* RLSAlssm objects that have the same model + window +
# block layout (e.g. parameter sweeps that re-instantiate the same cost).  Same
# lifetime and read-only-sharing guarantees as the caches above (the returned
# plan is only read by ``lfilter_parallel_xi1_split`` / the parallel xi filters).
_BUILD_PARALLEL_NUMDENOM_CACHE = {}

# Companion cache for the *parallel asterisk-l* realization.  ``_build_parallel_ast_sos``
# is the asterisk analogue of ``build_parallel_numdenom`` (same QZ + pole-zero
# cancellation per output row) but for a single ALSSM block, so its key omits
# ``block_sizes``.  Unlike the non-asterisk parallel plan, this one is built
# inside ``lfilter_parallel_xi_asterisk_split`` on the asterisk recursion path
# and is NOT pre-built at construction time — without this cache it is recomputed
# on every ``filter()`` call (and per object).  Caching therefore removes both a
# per-filter redundancy and the cross-object recompute.  Same lifetime and
# read-only-sharing guarantees as the caches above.
_BUILD_PARALLEL_AST_SOS_CACHE = {}


def _array_key(arr):
    """
    Build a hashable, content-based key fragment for an ndarray.

    Combines shape, dtype and raw bytes so that two arrays with identical
    contents (regardless of identity) map to the same key.  ``np.ascontiguousarray``
    guarantees a well-defined byte layout for non-contiguous inputs (e.g. ``.T``
    views), and the dtype is included so e.g. an int and float array with the
    same nominal values do not collide.
    """
    a = np.ascontiguousarray(arr)
    return (a.shape, a.dtype.str, a.tobytes())


def _cascade_params_key(A, C, a, b, delta, gamma, direction):
    """
    Hashable cache key fully identifying a ``_compute_cascade_params`` result.

    The result depends only on these seven inputs; ``a``/``b`` may be ``±inf``
    (valid, hashable float keys) and the array contents are folded in via
    [`_array_key`][lmlib.statespace.backends.rec_lfilter._array_key].
    """
    return (_array_key(A), _array_key(C),
            a, b, delta, gamma, direction)


def _cascade_params_asterisk_key(A, C, a, b, delta, gamma, Nprev, direction):
    """
    Hashable cache key fully identifying a ``_compute_cascade_params_asterisk``
    result.

    Identical to [`_cascade_params_key`][lmlib.statespace.backends.rec_lfilter._cascade_params_key]
    but with ``Nprev`` folded in, since the asterisk result (and its stored
    ``Nprev`` field, used by the consumers to reshape) depends on it.
    """
    return (_array_key(A), _array_key(C),
            a, b, delta, gamma, Nprev, direction)


def _build_parallel_numdenom_key(A, C, a, b, delta, gamma, direction, block_sizes):
    """
    Hashable cache key fully identifying a ``build_parallel_numdenom`` result.

    Same model+window fields as
    [`_cascade_params_key`][lmlib.statespace.backends.rec_lfilter._cascade_params_key]
    plus ``direction`` and ``block_sizes``: the plan's per-block decomposition
    depends on how the combined state vector is partitioned, so the block layout
    is part of the identity.  ``block_sizes`` may be ``None`` (single block) —
    normalised to a tuple so both forms hash consistently.
    """
    bs = None if not block_sizes else tuple(int(x) for x in block_sizes)
    return (_array_key(A), _array_key(C),
            a, b, delta, gamma, direction, bs)


def _build_parallel_ast_sos_key(A, C, a, b, delta, gamma, direction):
    """
    Hashable cache key fully identifying a ``_build_parallel_ast_sos`` result.

    Operates on a single ALSSM block, so — unlike
    [`_build_parallel_numdenom_key`][lmlib.statespace.backends.rec_lfilter._build_parallel_numdenom_key]
    — there is no ``block_sizes`` field.
    """
    return (_array_key(A), _array_key(C),
            a, b, delta, gamma, direction)


def clear_lfilter_caches():
    """
    Empty all static lfilter-backend coefficient caches.

    Drops every entry from the cascade $\\xi^{(q)}$ cache
    ([`_compute_cascade_params`][lmlib.statespace.backends.rec_lfilter._compute_cascade_params]),
    the asterisk-l cascade cache
    ([`_compute_cascade_params_asterisk`][lmlib.statespace.backends.rec_lfilter._compute_cascade_params_asterisk]),
    the parallel $\\xi^{(1)}$ plan cache
    ([`build_parallel_numdenom`][lmlib.statespace.backends.rec_lfilter.build_parallel_numdenom]),
    and the parallel asterisk-l plan cache
    ([`_build_parallel_ast_sos`][lmlib.statespace.backends.rec_lfilter._build_parallel_ast_sos]).
    All persist for the process lifetime and are shared by every lfilter
    consumer.  This helper is mainly useful in tests (to measure cold-vs-warm
    behaviour) and to reclaim memory explicitly.
    """
    _CASCADE_PARAMS_CACHE.clear()
    _CASCADE_PARAMS_ASTERISK_CACHE.clear()
    _BUILD_PARALLEL_NUMDENOM_CACHE.clear()
    _BUILD_PARALLEL_AST_SOS_CACHE.clear()


def lfilter_cache_info():
    """
    Return entry-count diagnostics for the static lfilter-backend caches.

    Returns
    -------
    dict
        ``{'size': <entries in the xi^(q) cascade cache>,
        'asterisk_size': <entries in the asterisk-l cascade cache>,
        'parallel_size': <entries in the parallel xi^(1) plan cache>,
        'parallel_asterisk_size': <entries in the parallel asterisk-l plan cache>}``.
    """
    return {'size': len(_CASCADE_PARAMS_CACHE),
            'asterisk_size': len(_CASCADE_PARAMS_ASTERISK_CACHE),
            'parallel_size': len(_BUILD_PARALLEL_NUMDENOM_CACHE),
            'parallel_asterisk_size': len(_BUILD_PARALLEL_AST_SOS_CACHE)}


def _block_ranges(block_sizes, N):
    """
    Yield ``(start, stop)`` index ranges for each ALSSM block in a combined,
    block-diagonal state vector of length ``N``.

    ``block_sizes`` is the list of per-ALSSM state orders.  When it is ``None``
    or describes a single block the whole vector ``[0, N)`` is returned, so
    callers reduce to the original full-width (non-block-aware) behaviour.
    """
    if not block_sizes or len(block_sizes) <= 1:
        return [(0, N)]
    offsets = np.concatenate([[0], np.cumsum(block_sizes)])
    return [(int(offsets[m]), int(offsets[m + 1])) for m in range(len(block_sizes))]


def _compute_cascade_params(A, C, a, b, delta, gamma, direction):
    r"""
    Precompute (and cache) the state-space/gamma scalars for the cascade IIR filters.

    The returned dict is passed directly to ``lfilter_forward_cascade_xi`` /
    ``lfilter_backward_cascade_xi``, avoiding repeated ``inv``, ``matrix_power``
    and ``np.dot`` calls inside the filter loop.

    The result is fully determined by ``(A, C, a, b, delta, gamma, direction)``
    — it does not depend on the signal or the sample weights — so it is memoised
    in a process-wide static cache
    (see [`clear_lfilter_caches`][lmlib.statespace.backends.rec_lfilter.clear_lfilter_caches]).
    Two calls with the same model and window (whether from the same RLSAlssm
    object, repeated ``filter()`` calls, or *different* RLSAlssm objects) return
    the same cached dict.  Consumers only read the dict, so sharing it is safe.

    The actual computation lives in
    [`_compute_cascade_params_uncached`][lmlib.statespace.backends.rec_lfilter._compute_cascade_params_uncached];
    this wrapper handles the cache lookup/insert.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
    C : np.ndarray, shape ([L,] N)
    a, b : int or ±inf  — segment boundaries
    delta : int          — segment offset
    gamma : float        — decay factor
    direction : str      — 'fw' or 'bw'

    Returns
    -------
    dict with keys:
        fw: gamma_inv, gamma_a, gamma_b, gAinvT, Aac, Abc, N
        bw: gamma, gamma_a, gamma_b, gAT, Aac, Abc, N
    """
    key = _cascade_params_key(A, C, a, b, delta, gamma, direction)
    params = _CASCADE_PARAMS_CACHE.get(key)
    if params is None:
        params = _compute_cascade_params_uncached(A, C, a, b, delta, gamma, direction)
        _CASCADE_PARAMS_CACHE[key] = params
    return params


def _compute_cascade_params_uncached(A, C, a, b, delta, gamma, direction):
    r"""
    Compute the state-space and gamma scalars for the cascade IIR filters.

    Pure function with no caching; see
    [`_compute_cascade_params`][lmlib.statespace.backends.rec_lfilter._compute_cascade_params]
    for the cached entry point and parameter/return documentation.
    """
    N = A.shape[1]
    if direction == 'fw':
        gamma_inv = 1.0 / gamma
        A_inv = inv(A)
        return {
            'gamma_inv': gamma_inv,
            'gamma_a':   gamma ** (a - 1 - delta),
            'gamma_b':   gamma ** (b - delta),
            'gAinvT':    gamma_inv * A_inv.T,
            'Aac':       np.dot(matrix_power(A, 0 if np.isinf(a) else a - 1).T, C.T),
            'Abc':       np.dot(matrix_power(A, b).T, C.T),
            'N':         N,
        }
    else:  # bw
        return {
            'gamma': gamma,
            'gamma_a': gamma ** (a - delta),
            'gamma_b': gamma ** (b - delta + 1),
            'gAT':     gamma * A.T,
            'Aac':     np.dot(matrix_power(A, a).T, C.T),
            'Abc':     np.dot(matrix_power(A, 0 if np.isinf(b) else b + 1).T, C.T),
            'N':       N,
        }


# xi2 lfilter cascade
def lfilter_cascade_xi2(xi2, A, C, a, b, direction, delta, gamma, y, sample_weights, beta):
    r"""
    Computes the second-order cost parameter $\xi^{(2)}(k, \mathbf{1})$ in-place,
    which equals the vectorized Gram matrix $\mathrm{vec}(W_k)$.

    $W_k \in \mathbb{R}^{N \times N}$ is independent of the signal `y` (it depends
    only on the model and window), so `y` is replaced by an all-ones array internally.
    The Kronecker product identity

    $$
    \mathrm{vec}(A^T c^T c A) = (A \otimes A)^T \mathrm{vec}(c^T c)
    $$

    allows the $W_k$ recursion to be recast as a standard $\xi^{(1)}$ recursion
    with substitutions $A \to A \otimes A$ and $C \to C \otimes C$.
    The result is stored in `xi2` as a flat vector of length $N^2$
    (i.e. $\mathrm{vec}(W_k)$).

    See also [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018) Eq. (22) and [\[Baeriswyl2025\]](../../bibliography.md#baeriswyl2025) Table I.

    Parameters
    ----------
    xi2 : np.ndarray, shape=(K, N**2, [S])
        Output array, modified in-place. Stores $\mathrm{vec}(W_k)$ for each
        time step k. Reshaped to ``(K, N, N)`` by the caller to recover $W_k$.
    A : np.ndarray, shape=(N, N)
        State-transition matrix of the ALSSM.
    C : np.ndarray, shape=([L,] N)
        Output matrix of the ALSSM.
    a : int or np.inf
        Left boundary of the segment interval.
    b : int or np.inf
        Right boundary of the segment interval.
    direction : str
        Recursion direction: ``'fw'`` for forward, ``'bw'`` for backward.
    delta : int
        Window normalization shift (window equals 1 at relative index ``delta``).
    gamma : float
        Window decay factor $\gamma$.
    y : np.ndarray, shape=(K, [L], [S]) or scalar
        Input signal. Only the shape is used (values are replaced by 1); `y` is
        passed solely to determine the number of time steps K.
    sample_weights : np.ndarray, shape=(K,) or scalar
        Per-sample weights $w_i$.
    beta : float
        Cost segment weight $\beta$.

    Notes
    -----
    The underlying [`lfilter_forward_cascade_xi`][lmlib.statespace.backends.rec_lfilter.lfilter_forward_cascade_xi] /
    [`lfilter_backward_cascade_xi`][lmlib.statespace.backends.rec_lfilter.lfilter_backward_cascade_xi] implement the recursion as a **cascade of
    1-D IIR filters**: each state dimension ``n`` is solved by one [`lfilter`][scipy.signal.lfilter]
    call, and its output is fed forward into the next dimension ``n+1``. This is possible
    because ``A`` must be upper-triangular, so the state equations are lower-dimensional and
    can be solved in order.

    Raises
    ------
    ValueError
        If `direction` is not ``'fw'`` or ``'bw'``.
    """
    _A = np.kron(A, A)
    _C = np.kron(C, C)
    _y = np.broadcast_to(1., np.shape(y))  # create an array of shape Ks, but contains only a single 1.0 in memory
    cascade_params = _compute_cascade_params(_A, _C, a, b, delta, gamma, direction)
    if direction == 'fw':
        lfilter_forward_cascade_xi(xi2, cascade_params, a, b, _y, sample_weights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi2, cascade_params, a, b, _y, sample_weights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi1 lfilter cascade
def lfilter_cascade_xi1(xi1, A, C, a, b, direction, delta, gamma, y, sample_weights, beta, block_sizes=None):
    r"""
    Computes the first-order cost parameter $\xi^{(1)}(k, y)$ in-place,
    which equals the signal projection vector $\xi_k$.

    $\xi_k \in \mathbb{R}^{N}$ depends on the signal `y` and is computed
    directly using the ALSSM matrices ``A`` and ``C`` without any substitution.
    The result is stored in `xi1` as a vector of length $N$.

    See also [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018) Eq. (23) and [\[Baeriswyl2025\]](../../bibliography.md#baeriswyl2025) Table I.

    Parameters
    ----------
    xi1 : np.ndarray, shape=(K, N, [S])
        Output array, modified in-place. Stores $\xi_k$ for each time step k.
    A : np.ndarray, shape=(N, N)
        State-transition matrix of the ALSSM.
    C : np.ndarray, shape=([L,] N)
        Output matrix of the ALSSM.
    a : int or np.inf
        Left boundary of the segment interval.
    b : int or np.inf
        Right boundary of the segment interval.
    direction : str
        Recursion direction: ``'fw'`` for forward, ``'bw'`` for backward.
    delta : int
        Window normalization shift (window equals 1 at relative index ``delta``).
    gamma : float
        Window decay factor $\gamma$.
    y : np.ndarray, shape=(K, [L], [S]) or scalar
        Input signal. Signal values are used directly in the recursion.
    sample_weights : np.ndarray, shape=(K,) or scalar
        Per-sample weights $w_i$.
    beta : float
        Cost segment weight $\beta$.

    Notes
    -----
    The underlying [`lfilter_forward_cascade_xi`][lmlib.statespace.backends.rec_lfilter.lfilter_forward_cascade_xi] /
    [`lfilter_backward_cascade_xi`][lmlib.statespace.backends.rec_lfilter.lfilter_backward_cascade_xi] implement the recursion as a **cascade of
    1-D IIR filters**: each state dimension ``n`` is solved by one [`lfilter`][scipy.signal.lfilter]
    call, and its output is fed forward into the next dimension ``n+1``. This is possible
    because ``A`` must be upper-triangular, so the state equations are lower-dimensional and
    can be solved in order.

    ``block_sizes`` is the list of per-ALSSM state orders making up the (combined,
    block-diagonal) ``A``.  When given, the cross-block feed-forward terms — which
    are structurally zero for a block-diagonal ``A`` — are skipped, so the work
    drops from $O(K N^2)$ to $O(K \\sum_m N_m^2)$ in a single pass
    (no per-block function-call overhead).  ``cascade_params`` are computed here
    from ``A``/``C`` (mirroring [`lfilter_cascade_xi2`][lmlib.statespace.backends.rec_lfilter.lfilter_cascade_xi2]).

    Raises
    ------
    ValueError
        If `direction` is not ``'fw'`` or ``'bw'``.
    """
    cascade_params = _compute_cascade_params(A, C, a, b, delta, gamma, direction)
    if direction == 'fw':
        lfilter_forward_cascade_xi(xi1, cascade_params, a, b, y, sample_weights, beta, block_sizes)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi1, cascade_params, a, b, y, sample_weights, beta, block_sizes)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi0 lfilter cascade
def lfilter_cascade_xi0(xi0, A, C, a, b, direction, delta, gamma, y, sample_weights, beta):
    r"""
    Computes the zeroth-order cost parameter $\xi^{(0)}(k, y)$ in-place,
    which equals the weighted signal energy $\kappa_k$.

    $\kappa_k \in \mathbb{R}$ is a scalar representing the accumulated weighted
    energy of the signal `y` within the window. It is computed by reducing the recursion
    to a scalar IIR filter via the substitutions

    $$
    A \to [[1]], \quad C \to [[1]], \quad y \to y^2
    $$

    so that the standard $\xi^{(1)}$ recursion accumulates
    $\kappa_k = \sum_i w_i \, y_i^2$.
    The result is stored in `xi0` with shape ``(K, 1, [S])``.

    Parameters ``A`` and ``C`` are accepted for interface consistency but are not used.

    See also [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018) Eq. (24) and [\[Baeriswyl2025\]](../../bibliography.md#baeriswyl2025) Table I.

    Parameters
    ----------
    xi0 : np.ndarray, shape=(K, 1, [S])
        Output array, modified in-place. Stores $\kappa_k$ for each time step k.
    A : np.ndarray, shape=(N, N)
        State-transition matrix of the ALSSM. Not used; accepted for interface consistency.
    C : np.ndarray, shape=([L,] N)
        Output matrix of the ALSSM. Not used; accepted for interface consistency.
    a : int or np.inf
        Left boundary of the segment interval.
    b : int or np.inf
        Right boundary of the segment interval.
    direction : str
        Recursion direction: ``'fw'`` for forward, ``'bw'`` for backward.
    delta : int
        Window normalization shift (window equals 1 at relative index ``delta``).
    gamma : float
        Window decay factor $\gamma$.
    y : np.ndarray, shape=(K, [L], [S]) or scalar
        Input signal. Values are squared internally (``_y = y**2``) before the recursion.
    sample_weights : np.ndarray, shape=(K,) or scalar
        Per-sample weights $w_i$.
    beta : float
        Cost segment weight $\beta$.

    Raises
    ------
    ValueError
        If `direction` is not ``'fw'`` or ``'bw'``.
    """
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = y**2
    cascade_params = _compute_cascade_params(_A, _C, a, b, delta, gamma, direction)
    if direction == 'fw':
        lfilter_forward_cascade_xi(xi0, cascade_params, a, b, _y, sample_weights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi0, cascade_params, a, b, _y, sample_weights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# nu lfilter cascade
def lfilter_cascade_nu(nu, A, C, a, b, direction, delta, gamma, y, sample_weights, beta):
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = np.broadcast_to(1., np.shape(y))  # create an array of shape Ks, but contains only a single 1.0 in memory
    cascade_params = _compute_cascade_params(_A, _C, a, b, delta, gamma, direction)
    if direction == 'fw':
        lfilter_forward_cascade_xi(nu, cascade_params, a, b, _y, sample_weights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(nu, cascade_params, a, b, _y, sample_weights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# general forward cascade
# @profile is intentional on this production function: the decorator is a
# transparent pass-through when lm.profiling.enable() has not been called
# (overhead is a single bool check per call). See lmlib/utils/profiling.py.
@profile
def lfilter_forward_cascade_xi(xi, cascade_params, a, b, y, sample_weights, beta, block_sizes=None):
    """
    IIR forward calculation of xi.

    Precomputed state-space and gamma scalars are passed in via *cascade_params*
    (built by ``_compute_cascade_params`` with ``direction='fw'``), so no matrix
    inversion or power computation occurs inside this function.

    Parameters
    ----------
    xi : np.ndarray
        shape=(K, N, [S]) — accumulation target, updated in-place.
    cascade_params : dict
        Precomputed parameters from ``_compute_cascade_params``.
        Required keys: ``gamma_inv``, ``gamma_a``, ``gamma_b``, ``gAinvT``,
        ``Aac``, ``Abc``, ``N``.
    a : int or inf
        Left segment boundary.
    b : int or inf
        Right segment boundary.
    y : np.ndarray
        shape=(K, [L], [S]) or scalar 1 — weighted observations.
    sample_weights : np.ndarray
        shape=(K,) or scalar 1.
    beta : float
        Segment weight (SE beta).
    """

    gamma_inv = cascade_params['gamma_inv']
    gamma_a = cascade_params['gamma_a']
    gamma_b = cascade_params['gamma_b']
    gAinvT = cascade_params['gAinvT']
    Aac = cascade_params['Aac']
    Abc = cascade_params['Abc']
    N = cascade_params['N']

    if not np.allclose(gAinvT, np.tril(gAinvT)):
        raise ValueError("State-Space Matrix A needs to be upper triangular for cascaded version")

    # Per-dimension IIR pole = diagonal of the recursion matrix gamma^{-1} A^{-T}.
    # For unit-diagonal (polynomial) A this is just gamma_inv; for exponential
    # models (AlssmExp / AlssmProd) the model eigenvalue folds into the pole
    # (gamma_inv / diag(A)).  The diagonal is then removed from the feed-forward
    # coupling, leaving it strictly lower-triangular.
    poles = np.diag(gAinvT).copy()
    gAinvT = gAinvT - np.diagflat(np.diag(gAinvT))

    y_weighted = y*sample_weights[:, None]
    K = y_weighted.shape[0]

    # shift signal
    # insert the shifted signal: since b > a (by definition), the recursion starts with signal b only.
    if not np.isinf(a):
        # K_append must satisfy two constraints:
        #   1. The window width (b-a+1) sets the delay between the a- and b-boundary contributions.
        #   2. The output extraction xi_add[b:b+K] requires K_append >= b  (so that xi_add has b+K rows).
        # When both a and b are positive (a > 0), constraint 2 is tighter than constraint 1.
        window_width = b - a + 1
        K_append = max(window_width, b + 1) if b >= 0 else window_width
    else:
        K_append = 0
    y_delayed_b = np.zeros((K + K_append, *y_weighted.shape[1:]))
    y_delayed_b[0:K] = y_weighted
    y_diff = np.einsum('kl, nl->kn', y_delayed_b, gamma_b * Abc)

    if not np.isinf(a):
        # The a-boundary signal is delayed by (b-a+1) positions relative to the b-boundary.
        # This offset is always b-a+1 regardless of K_append (which may be larger).
        a_offset = b - a + 1
        y_delayed_a = np.zeros((K + K_append, *y_weighted.shape[1:]))
        y_delayed_a[a_offset:a_offset + K] = y_weighted
        y_diff -= np.einsum('kl, nl->kn', y_delayed_a, gamma_a * Aac)

    # iterating through ALSSM (xi) elements
    y_diff = np.swapaxes(y_diff, 0, 1)  # convenient for later indexing
    xi_add = np.zeros((K + K_append, *xi.shape[1:]), order='F')
    # gAinvT is block-diagonal when A is (each ALSSM is one block); the
    # feed-forward einsum is therefore sliced to each block, skipping the
    # structurally-zero cross-block terms.  block_sizes=None (or a single
    # block) reproduces the original full-width cascade.
    for s, e in _block_ranges(block_sizes, N):
        xi_add[:, s] = lfilter([1, 0], [1, -poles[s]], y_diff[s].T).T
        for n_ in range(s + 1, e):
            y_diff[n_, 1:] += np.einsum('kn..., n->k...', xi_add[:-1, s:e], gAinvT[n_, s:e])
            xi_add[:, n_] = lfilter([1, 0], [1, -poles[n_]], y_diff[n_].T).T

    # SE weight for this cost segment
    if beta != 1:
        xi_add *= beta

    #xi needs to be correctly inserted. since the signal y_delayed_b had an actual delay of 0,
    #we need to shift xi0 by b.
    if b >= 0:
        xi += xi_add[b:b+K]
    #  if b < 0, first few elements of xi need to be 0 (both boundaries negative)
    if b < 0:
        xi[-b:] += xi_add[0:K+b]



# general backward cascade
# @profile is intentional on this production function (see forward cascade comment above).
@profile
def lfilter_backward_cascade_xi(xi, cascade_params, a, b, y, sample_weights, beta, block_sizes=None):
    """
    IIR backward calculation of xi.

    Precomputed state-space and gamma scalars are passed in via *cascade_params*
    (built by ``_compute_cascade_params`` with ``direction='bw'``), so no matrix
    power computation occurs inside this function.

    Parameters
    ----------
    xi : np.ndarray
        shape=(K, N, [S]) — accumulation target, updated in-place.
    cascade_params : dict
        Precomputed parameters from ``_compute_cascade_params``.
        Required keys: ``gamma``, ``gamma_a``, ``gamma_b``, ``gAT``,
        ``Aac``, ``Abc``, ``N``.
    a : int or inf
        Left segment boundary.
    b : int or inf
        Right segment boundary.
    y : np.ndarray
        shape=(K, [L], [S]) or scalar 1 — weighted observations.
    sample_weights : np.ndarray
        shape=(K,) or scalar 1.
    beta : float
        Segment weight (SE beta).
    """

    gamma = cascade_params['gamma']
    gamma_a = cascade_params['gamma_a']
    gamma_b = cascade_params['gamma_b']
    gAT = cascade_params['gAT']
    Aac = cascade_params['Aac']
    Abc = cascade_params['Abc']
    N = cascade_params['N']

    if not np.allclose(gAT, np.tril(gAT)):
        raise ValueError("State-Space Matrix A needs to be upper triangular for cascaded version")

    # Per-dimension IIR pole = diagonal of the backward recursion matrix gamma A^T
    # (gamma * diag(A)); the diagonal is removed from the feed-forward coupling.
    poles = np.diag(gAT).copy()
    gAT = gAT - np.diagflat(np.diag(gAT))

    K = len(xi)
    y_weighted = y*sample_weights[:, None]

    #time-reverse observation for backward recursion
    y_weighted_flipped = y_weighted[::-1]

    # shift signal
    # insert the shifted signal: since a < b (by definition), the backward recursion starts with signal a only.
    if not np.isinf(b):
        K_append  = b-a+1 #this is the length of the window
    else:
        K_append = 0
    y_delayed_a = np.zeros((K + K_append, *y_weighted_flipped.shape[1:]))
    y_delayed_a[0:K] = y_weighted_flipped
    y_diff = np.einsum('kl, nl->kn', y_delayed_a, gamma_a * Aac)

    if not np.isinf(b):
        # insert the shifted signal: b is inserted after K_append (length of the window).
        y_delayed_b = np.zeros((K + K_append, *y_weighted_flipped.shape[1:]))
        y_delayed_b[K_append:] = y_weighted_flipped
        y_diff -= np.einsum('kl, nl->kn', y_delayed_b, gamma_b * Abc)

    # iterating through dimensions
    y_diff = np.swapaxes(y_diff, 0, 1)  # convenient for later indexing
    xi_add = np.zeros((K + K_append, *xi.shape[1:]), order='F')
    # gAT is block-diagonal when A is; slice the feed-forward to each block
    # (see lfilter_forward_cascade_xi for details).
    for s, e in _block_ranges(block_sizes, N):
        xi_add[:, s] = lfilter([1, 0], [1, -poles[s]], y_diff[s].T).T
        for n_ in range(s + 1, e):
            y_diff[n_, 1:] += np.einsum('kn..., n->k...', xi_add[:-1, s:e], gAT[n_, s:e])
            xi_add[:, n_] = lfilter([1, 0], [1, -poles[n_]], y_diff[n_].T).T

    # SE weight for this cost segment
    if beta != 1:
        xi_add *= beta

    #xi needs to be correctly inserted. since the signal y_delayed_a had an actual delay of 0,
    #we need to shift xi0 by a.
    xi0_flipped = xi_add[::-1]
    if a >= 0:
        xi[0:K-a] += xi0_flipped[-(K-a):]
    if a < 0:
        xi += xi0_flipped[b+1:K+b+1]




# ─── asterisk-l cascade helpers ──────────────────────────────────────────────

def _compute_cascade_params_asterisk(A, C, a, b, delta, gamma, Nprev, direction):
    r"""
    Precompute (and cache) state-space scalars for the asterisk-l cascade IIR filters.

    The returned dict is passed directly to
    [`lfilter_xi_asterisk_l_forward_cascade_recursion`][lmlib.statespace.backends.rec_lfilter.lfilter_xi_asterisk_l_forward_cascade_recursion] or
    [`lfilter_xi_asterisk_l_backward_cascade_recursion`][lmlib.statespace.backends.rec_lfilter.lfilter_xi_asterisk_l_backward_cascade_recursion].

    The effective system for the asterisk recursion uses
    $A_\mathrm{ast} = I_{N_\mathrm{prev}} \otimes A$, which is
    block-diagonal with $N_\mathrm{prev}$ copies of ``A`` on the
    diagonal.  Because ``A`` is upper triangular, ``A_ast`` is also upper
    triangular, so the cascade structure (lower-triangular
    $\gamma^{-1} A_\mathrm{ast}^{-\mathrm{T}}$) is preserved.

    The ``Aac`` / ``Abc`` entries are stored as ravelled 1-D vectors so that
    the driving-input computation

    ```python
    y_diff = (xi_N_b[..., :, None] * Abc_1d).reshape(L, *S_shape, Nprev*N)
    ```

    works for both 1-D ``C`` (shape ``(N,)``) and 2-D ``C`` (shape ``(1, N)``
    as produced by [`AlssmSum`][lmlib.statespace.model.AlssmSum]).

    The result is fully determined by ``(A, C, a, b, delta, gamma, Nprev,
    direction)`` — it does not depend on the signal or the sample weights — so it
    is memoised in a process-wide static cache (see
    [`clear_lfilter_caches`][lmlib.statespace.backends.rec_lfilter.clear_lfilter_caches]),
    shared across the per-slice asterisk loop, repeated ``filter()`` calls, and
    different RLSAlssm objects with the same model+window+``Nprev``.  Consumers
    only read the dict, so sharing it is safe.  The actual computation lives in
    [`_compute_cascade_params_asterisk_uncached`][lmlib.statespace.backends.rec_lfilter._compute_cascade_params_asterisk_uncached].

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Upper-triangular state-transition matrix of the ALSSM.
    C : np.ndarray, shape ([L,] N)
        Output matrix of the ALSSM (1-D or 2-D).
    a, b : int or ±inf
        Segment boundaries.
    delta : int
        Segment normalisation shift.
    gamma : float
        Window decay factor.
    Nprev : int
        Trailing state dimension of ``xi_prev`` (i.e. $N_\mathrm{prev}$).
    direction : str
        ``'fw'`` or ``'bw'``.

    Returns
    -------
    dict
        Keys for ``'fw'``:
            ``gamma_inv``, ``gamma_a``, ``gamma_b``, ``gAinvT``,
            ``Aac``, ``Abc``, ``N``, ``Nprev``.
        Keys for ``'bw'``:
            ``gamma``, ``gamma_a``, ``gamma_b``, ``gAT``,
            ``Aac``, ``Abc``, ``N``, ``Nprev``.
    """
    key = _cascade_params_asterisk_key(A, C, a, b, delta, gamma, Nprev, direction)
    params = _CASCADE_PARAMS_ASTERISK_CACHE.get(key)
    if params is None:
        params = _compute_cascade_params_asterisk_uncached(
            A, C, a, b, delta, gamma, Nprev, direction)
        _CASCADE_PARAMS_ASTERISK_CACHE[key] = params
    return params


def _compute_cascade_params_asterisk_uncached(A, C, a, b, delta, gamma, Nprev, direction):
    r"""
    Compute state-space scalars for the asterisk-l cascade IIR filters.

    Pure function with no caching; see
    [`_compute_cascade_params_asterisk`][lmlib.statespace.backends.rec_lfilter._compute_cascade_params_asterisk]
    for the cached entry point and parameter/return documentation.
    """
    N = A.shape[0]
    A_inv = inv(A)
    gamma_inv = 1.0 / gamma
    if direction == 'fw':
        Aa = matrix_power(A, 0 if np.isinf(a) else a - 1)
        Ab = matrix_power(A, b)
        return {
            'gamma_inv': gamma_inv,
            'gamma_a':   gamma ** (a - 1 - delta),
            'gamma_b':   gamma ** (b - delta),
            'gAinvT':    gamma_inv * A_inv.T,
            'Aac':       np.dot(Aa.T, C.T).ravel(),   # always (N,)
            'Abc':       np.dot(Ab.T, C.T).ravel(),    # always (N,)
            'N':         N,
            'Nprev':     Nprev,
        }
    else:  # bw
        Aa = matrix_power(A, a)
        Ab = matrix_power(A, 0 if np.isinf(b) else b + 1)
        return {
            'gamma':   gamma,
            'gamma_a': gamma ** (a - delta),
            'gamma_b': gamma ** (b - delta + 1),
            'gAT':     gamma * A.T,
            'Aac':     np.dot(Aa.T, C.T).ravel(),     # always (N,)
            'Abc':     np.dot(Ab.T, C.T).ravel(),     # always (N,)
            'N':       N,
            'Nprev':   Nprev,
        }


@profile
def lfilter_xi_asterisk_l_forward_cascade_recursion(xi, cascade_params_ast, a, b, xi_N, v, beta):
    r"""
    IIR forward cascade for the asterisk-l recursion $\xi^{(1)*l}$.

    Computes the cross-dimensional xi term

    $$
    \xi^{(1)*l}(k) \mathrel{+}=
    \text{(boundary-b)} - \text{(boundary-a)}
    $$

    using the same cascade-of-1-D-IIR structure as
    [`lfilter_forward_cascade_xi`][lmlib.statespace.backends.rec_lfilter.lfilter_forward_cascade_xi], but driven by the previously
    accumulated ``xi_N`` (a state vector from the lower-dimensional
    recursion) rather than the raw signal ``y``.

    The effective system matrix is $A_\mathrm{ast} = I_{N_p} \otimes A$
    (block-diagonal, upper triangular), so the cascade can process all
    $N_p$ blocks simultaneously.  For each model state dimension
    ``n``, one [`lfilter`][scipy.signal.lfilter] call handles all
    $N_p \times \prod(S)$ channels in a single pass.

    See [\[Baeriswyl2025\]](../../bibliography.md#baeriswyl2025) Eq. (47).

    Parameters
    ----------
    xi : np.ndarray, shape (K, \*S, Nprev \* N)
        Output buffer, updated in-place.
    cascade_params_ast : dict
        Precomputed parameters from
        [`_compute_cascade_params_asterisk`][lmlib.statespace.backends.rec_lfilter._compute_cascade_params_asterisk] with ``direction='fw'``.
        Required keys: ``gamma_inv``, ``gamma_a``, ``gamma_b``,
        ``gAinvT``, ``Aac``, ``Abc``, ``N``, ``Nprev``.
    a : int or inf
        Left segment boundary.
    b : int or inf
        Right segment boundary.
    xi_N : np.ndarray, shape (K, \*S, Nprev)
        Previously accumulated xi from the lower-dimensional recursion.
        The first axis is the active filter axis; ``\*S`` are extra
        spatial dimensions; the last axis is the Nprev state components.
    v : np.ndarray
        Sample weights (unused in the asterisk recursion; accepted for
        interface consistency with the numpy backend).
    beta : float
        Segment cost weight.

    Notes
    -----
    The driving input for each output state ``n_`` = ``j * N + n`` is

    $$
    d[k,\,j,\,n] =
      \gamma_b \cdot \mathrm{Abc}[n] \cdot \xi_N[k+b,\,j]
      - \gamma_a \cdot \mathrm{Aac}[n] \cdot \xi_N[k+a-1,\,j]
    $$

    where ``j`` indexes the $N_p$ blocks and ``n`` indexes the
    model state dimension within each block.  After computing ``y_diff``
    with shape ``(L, \*S, Nprev, N)``, the inner cascade loop adds the
    lower-triangular coupling ``gAinvT[n, :n]`` from previously computed
    state dims and applies one ``lfilter`` call per model dimension ``n``.

    Raises
    ------
    ValueError
        If the state-transition matrix ``A`` is not upper triangular.
    """
    gamma_inv = cascade_params_ast['gamma_inv']
    gamma_a   = cascade_params_ast['gamma_a']
    gamma_b   = cascade_params_ast['gamma_b']
    gAinvT    = cascade_params_ast['gAinvT']   # (N, N) lower-triangular
    Aac       = cascade_params_ast['Aac']       # (N,) ravelled
    Abc       = cascade_params_ast['Abc']       # (N,) ravelled
    N         = cascade_params_ast['N']
    Nprev     = cascade_params_ast['Nprev']

    if not np.allclose(gAinvT, np.tril(gAinvT)):
        raise ValueError("State-Space Matrix A needs to be upper triangular for cascaded version")

    # Per-dimension IIR pole = diagonal of gamma^{-1} A^{-T} (folds in the model
    # eigenvalue for exponential models).  The coupling gAinvT[n, :n] used below is
    # already strictly lower-triangular, so only the pole needs the diagonal.
    poles = np.diag(gAinvT)

    K       = xi_N.shape[0]
    S_shape = xi_N.shape[1:-1]   # extra spatial dims between K and Nprev

    # ── boundary padding (same logic as lfilter_forward_cascade_xi) ──────────
    if not np.isinf(a):
        window_width = b - a + 1
        K_append = max(window_width, b + 1) if b >= 0 else window_width
    else:
        K_append = 0
    L = K + K_append

    # ── b-boundary driving input ──────────────────────────────────────────────
    # xi_N_b shape: (L, *S, Nprev); delayed_b = xi_N[:K], zeros after.
    xi_N_b = np.zeros((L, *S_shape, Nprev))
    xi_N_b[:K] = xi_N

    # y_diff[l, *s, j*N + n] = gamma_b * Abc[n] * xi_N_b[l, *s, j]
    # (..., Nprev, 1) * (N,) -> (..., Nprev, N) -> reshape (..., Nprev*N)
    y_diff = (xi_N_b[..., :, None] * (gamma_b * Abc)).reshape(L, *S_shape, Nprev * N)

    # ── a-boundary subtraction ────────────────────────────────────────────────
    if not np.isinf(a):
        a_offset = b - a + 1     # delay between b- and a-boundary signals
        xi_N_a = np.zeros((L, *S_shape, Nprev))
        xi_N_a[a_offset:a_offset + K] = xi_N
        y_diff -= (xi_N_a[..., :, None] * (gamma_a * Aac)).reshape(L, *S_shape, Nprev * N)

    # ── cascade IIR loop (one lfilter call per model state dim n) ─────────────
    # Reshape to (L, *S, Nprev, N) for convenient per-n block access.
    xi_add = np.zeros((L, *S_shape, Nprev * N), order='F')
    yd_r   = y_diff.reshape(L, *S_shape, Nprev, N)   # view into y_diff
    xa_r   = xi_add.reshape(L, *S_shape, Nprev, N)   # view into xi_add

    for n in range(N):
        if n > 0:
            # Lower-triangular coupling: add contributions from dims 0..n-1
            # within each block (gAinvT[n, :n] is the coupling row).
            yd_r[1:, ..., n] += np.einsum('...m,m->...', xa_r[:-1, ..., :n], gAinvT[n, :n])
        # Filter all (*S, Nprev) channels simultaneously along the time axis.
        inp    = yd_r[..., n].reshape(L, -1)           # (L, prod(S)*Nprev)
        out    = lfilter([1, 0], [1, -poles[n]], inp.T).T
        xa_r[..., n] = out.reshape(L, *S_shape, Nprev)

    if beta != 1:
        xi_add *= beta

    # ── accumulate into output (same slicing as lfilter_forward_cascade_xi) ───
    if b >= 0:
        xi += xi_add[b:b + K]
    if b < 0:
        xi[-b:] += xi_add[0:K + b]


@profile
def lfilter_xi_asterisk_l_backward_cascade_recursion(xi, cascade_params_ast, a, b, xi_N, v, beta):
    r"""
    IIR backward cascade for the asterisk-l recursion $\xi^{(1)*l}$.

    Mirror of [`lfilter_xi_asterisk_l_forward_cascade_recursion`][lmlib.statespace.backends.rec_lfilter.lfilter_xi_asterisk_l_forward_cascade_recursion] for
    backward segments (direction ``'bw'``).

    The time axis of ``xi_N`` is reversed before the cascade, and the
    accumulated result is reversed again before being written back, exactly
    mirroring how [`lfilter_backward_cascade_xi`][lmlib.statespace.backends.rec_lfilter.lfilter_backward_cascade_xi] handles the regular
    xi recursion.

    Parameters
    ----------
    xi : np.ndarray, shape (K, \*S, Nprev \* N)
        Output buffer, updated in-place.
    cascade_params_ast : dict
        Precomputed parameters from
        [`_compute_cascade_params_asterisk`][lmlib.statespace.backends.rec_lfilter._compute_cascade_params_asterisk] with ``direction='bw'``.
        Required keys: ``gamma``, ``gamma_a``, ``gamma_b``,
        ``gAT``, ``Aac``, ``Abc``, ``N``, ``Nprev``.
    a : int or inf
        Left segment boundary.
    b : int or inf
        Right segment boundary.
    xi_N : np.ndarray, shape (K, \*S, Nprev)
        Previously accumulated xi from the lower-dimensional recursion.
    v : np.ndarray
        Sample weights (unused; accepted for interface consistency).
    beta : float
        Segment cost weight.

    Raises
    ------
    ValueError
        If the state-transition matrix ``A`` is not upper triangular.
    """
    gamma   = cascade_params_ast['gamma']
    gamma_a = cascade_params_ast['gamma_a']
    gamma_b = cascade_params_ast['gamma_b']
    gAT     = cascade_params_ast['gAT']     # (N, N) lower-triangular
    Aac     = cascade_params_ast['Aac']     # (N,) ravelled
    Abc     = cascade_params_ast['Abc']     # (N,) ravelled
    N       = cascade_params_ast['N']
    Nprev   = cascade_params_ast['Nprev']

    if not np.allclose(gAT, np.tril(gAT)):
        raise ValueError("State-Space Matrix A needs to be upper triangular for cascaded version")

    # Per-dimension IIR pole = diagonal of gamma A^T (folds in the model eigenvalue).
    poles = np.diag(gAT)

    K       = xi_N.shape[0]
    S_shape = xi_N.shape[1:-1]

    # ── boundary padding ──────────────────────────────────────────────────────
    K_append = (b - a + 1) if not np.isinf(b) else 0
    L = K + K_append

    # Time-reverse xi_N for the backward recursion (mirrors lfilter_backward_cascade_xi).
    xi_N_flipped = xi_N[::-1]

    # ── a-boundary driving input (leading contribution in backward pass) ──────
    xi_N_a = np.zeros((L, *S_shape, Nprev))
    xi_N_a[:K] = xi_N_flipped
    y_diff = (xi_N_a[..., :, None] * (gamma_a * Aac)).reshape(L, *S_shape, Nprev * N)

    # ── b-boundary subtraction (delayed by window width) ─────────────────────
    if not np.isinf(b):
        xi_N_b = np.zeros((L, *S_shape, Nprev))
        xi_N_b[K_append:] = xi_N_flipped
        y_diff -= (xi_N_b[..., :, None] * (gamma_b * Abc)).reshape(L, *S_shape, Nprev * N)

    # ── cascade IIR loop ──────────────────────────────────────────────────────
    xi_add = np.zeros((L, *S_shape, Nprev * N), order='F')
    yd_r   = y_diff.reshape(L, *S_shape, Nprev, N)
    xa_r   = xi_add.reshape(L, *S_shape, Nprev, N)

    for n in range(N):
        if n > 0:
            yd_r[1:, ..., n] += np.einsum('...m,m->...', xa_r[:-1, ..., :n], gAT[n, :n])
        inp  = yd_r[..., n].reshape(L, -1)
        out  = lfilter([1, 0], [1, -poles[n]], inp.T).T
        xa_r[..., n] = out.reshape(L, *S_shape, Nprev)

    if beta != 1:
        xi_add *= beta

    # ── accumulate (same slicing as lfilter_backward_cascade_xi) ─────────────
    xi_add_flipped = xi_add[::-1]
    if a >= 0:
        end = K - a
        if end > 0:
            xi[:end] += xi_add_flipped[-end:]
    if a < 0:
        xi += xi_add_flipped[b + 1:K + b + 1]


def _apply_fir_batched(sos, extra_delay, y_sig_2d, Lout):
    r"""
    Batched variant of [`_apply_fir`][lmlib.statespace.backends.rec_lfilter._apply_fir] for 2-D input.

    Parameters
    ----------
    sos : ndarray or None
        Numerator SOS filter coefficients.
    extra_delay : int
        Integer delay to prepend (zero-padding at the front).
    y_sig_2d : ndarray, shape (L, C)
        Multi-channel input; each column is one independent signal.
    Lout : int
        Desired output length.

    Returns
    -------
    result : ndarray, shape (Lout, C)
        Filtered and delayed output.  Columns beyond the natural filter
        output are zero-padded; samples before index 0 are dropped.
    """
    C = y_sig_2d.shape[1]
    result = np.zeros((Lout, C))
    if sos is None:
        return result
    filtered = _fir_taps(_sos_to_fir_taps(sos), y_sig_2d)   # (L, C)
    end = min(extra_delay + filtered.shape[0], Lout)
    result[extra_delay:end] = filtered[:end - extra_delay]
    return result


# ─── parallel asterisk-l recursion  ───────────────────────────────────────────
#
# `xi_q_asterisk_l_recursion` (backends/rec.py) routes the # q==1 asterisk step 
#  for filter_form='parallel' through `lfilter_parallel_xi_asterisk_split`, 
# which builds a per-ALSSM-block transfer
# function via `_build_parallel_ast_sos` and scatters each block result into the
# Kronecker layout of xi_curr.  (q==0 kappa and q==2 W use the cascade / numpy
# realizations.)
#
# The previously-broken backward recursion (`...backward_parallel_recursion`)
# applied a spurious per-row `advance` offset to its output slice that the
# forward path and the first-pass backward filter never used; this is fixed and
# both directions now match the numpy reference to ~1e-12.
# ──────────────────────────────────────────────────────────────────────────────


def _build_parallel_ast_sos(A, C, a, b, delta, gamma, direction):
    r"""
    Build (and cache) the per-row SOS structure for the parallel asterisk-l recursion.

    Produces the same 11-element list as ``_numdenom[dim][p][m]`` (see
    [`_build_numdenom`][lmlib.statespace.rls.RLSAlssm._build_numdenom]), but for a
    single ALSSM block and without the ``AlssmSum`` wrapper.

    The result is fully determined by ``(A, C, a, b, delta, gamma, direction)``
    — it does not depend on the signal or the sample weights — and the
    decomposition (QZ + pole-zero cancellation + per-row SOS) is comparatively
    expensive.  It is therefore memoised in a process-wide static cache (see
    [`clear_lfilter_caches`][lmlib.statespace.backends.rec_lfilter.clear_lfilter_caches]).
    Unlike the non-asterisk parallel plan, this structure is built on the
    asterisk recursion path (inside
    [`lfilter_parallel_xi_asterisk_split`][lmlib.statespace.backends.rec_lfilter.lfilter_parallel_xi_asterisk_split])
    rather than pre-built at construction, so the cache removes a per-filter
    recompute in addition to sharing across objects.  Consumers only read the
    returned list, so sharing it is safe.  The actual construction lives in
    [`_build_parallel_ast_sos_uncached`][lmlib.statespace.backends.rec_lfilter._build_parallel_ast_sos_uncached].

    Parameters
    ----------
    A : ndarray, shape (N, N)
        State-transition matrix.
    C : ndarray, shape ([L,] N)
        Output matrix (1-D or 2-D).
    a, b : int or ±inf
        Segment boundaries.
    delta : int
        Window normalisation shift.
    gamma : float
        Window decay factor.
    direction : str
        ``'fw'`` or ``'bw'``.

    Returns
    -------
    list of length 11
        ``[sos_iir_shared, sos_b_list, sos_a_list, db_list, da_list,``
        ``sos_iir_b_list, sos_iir_a_list, n_poles_b_list, n_poles_a_list,``
        ``advance_b_list, advance_a_list]``
    """
    key = _build_parallel_ast_sos_key(A, C, a, b, delta, gamma, direction)
    nd = _BUILD_PARALLEL_AST_SOS_CACHE.get(key)
    if nd is None:
        nd = _build_parallel_ast_sos_uncached(A, C, a, b, delta, gamma, direction)
        _BUILD_PARALLEL_AST_SOS_CACHE[key] = nd
    return nd


def _build_parallel_ast_sos_uncached(A, C, a, b, delta, gamma, direction):
    r"""
    Build the per-row SOS structure for the parallel asterisk-l recursion.

    Pure function with no caching; see
    [`_build_parallel_ast_sos`][lmlib.statespace.backends.rec_lfilter._build_parallel_ast_sos]
    for the cached entry point and full parameter/return documentation.
    """
    from lmlib.statespace.backends.statespace_tools import ss2zpk_qz
    N_m = A.shape[0]
    if direction == 'fw':
        gAT = (1.0 / gamma) * inv(A).T
        Aac = (matrix_power(A, 0 if np.isinf(a) else a - 1).T @ C.T).ravel()
        Abc = (matrix_power(A, b).T @ C.T).ravel()
    else:  # bw
        gAT = gamma * A.T
        Aac = (matrix_power(A, a).T @ C.T).ravel()
        Abc = (matrix_power(A, 0 if np.isinf(b) else b + 1).T @ C.T).ravel()

    poles = eigvals(gAT)
    sos_iir_shared = zpk2sos(np.zeros(len(poles)), poles, 1.0)

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
        advance_b_list.append(1 if abs(float(Abc[n_])) < 1e-10 else 0)
        advance_a_list.append(1 if abs(float(Aac[n_])) < 1e-10 else 0)

    return [sos_iir_shared, sos_b_list, sos_a_list, db_list, da_list,
            sos_iir_b_list, sos_iir_a_list, n_poles_b_list, n_poles_a_list,
            advance_b_list, advance_a_list]


@profile
def lfilter_xi_asterisk_l_forward_parallel_recursion(xi, nd, a, b, delta, gamma, xi_N, v, beta):
    r"""
    SOS-based forward parallel filter for the asterisk-l recursion
    $\xi^{(1)*l}$.

    Applies the same per-row SOS structure as
    [`lfilter_forward_parallel_xi`][lmlib.statespace.backends.rec_lfilter.lfilter_forward_parallel_xi], but driven by ``xi_N`` (the
    accumulated cost vector from the previous dimension) rather than by the
    raw signal ``y``.

    Because the effective system matrix for the asterisk recursion is
    $A_\mathrm{ast} = I_{N_p} \otimes A$ (block-diagonal), each of the
    $N_p$ blocks shares exactly the same transfer function as the
    base-model xi recursion.  This function therefore applies the **same**
    ``nd`` SOS structure once per input channel ``j`` of ``xi_N``, writing
    the ``N``-dimensional result into output rows ``j \cdot N \ldots (j+1)N-1``.

    All $N_p$ input channels (and any extra spatial dimensions ``\*S``)
    are batched into a single 2-D array so that each ``sosfilt`` / FIR call
    processes all channels simultaneously.

    Parameters
    ----------
    xi : ndarray, shape (K, \*S, Nprev \* N)
        Output buffer, updated in-place.
    nd : list
        11-element SOS list from [`_build_parallel_ast_sos`][lmlib.statespace.backends.rec_lfilter._build_parallel_ast_sos] with
        ``direction='fw'``.
    a : int or inf
        Left segment boundary.
    b : int or inf
        Right segment boundary.
    delta : int
        Window normalisation shift.
    gamma : float
        Window decay factor.
    xi_N : ndarray, shape (K, \*S, Nprev)
        Input from the lower-dimensional recursion.
    v : ndarray
        Sample weights (unused; accepted for interface consistency).
    beta : float
        Segment cost weight.
    """
    K      = xi_N.shape[0]
    Nprev  = xi_N.shape[-1]
    S_shape = xi_N.shape[1:-1]
    S_flat = int(np.prod(S_shape)) if S_shape else 1
    N      = xi.shape[-1] // Nprev       # N_model

    sos_iir      = nd[0]
    sos_b_list   = nd[1]; sos_a_list   = nd[2]
    db_list      = nd[3]; da_list      = nd[4]
    sos_iir_b_list = nd[5] if len(nd) > 5 else None
    sos_iir_a_list = nd[6] if len(nd) > 6 else None
    n_poles_b_list = nd[7] if len(nd) > 7 else None
    n_poles_a_list = nd[8] if len(nd) > 8 else None

    gamma_a   = gamma ** (a - 1 - delta)
    gamma_b   = gamma ** (b - delta)
    gamma_inv = 1.0 / gamma

    K_append = max(b - a + 1, b + 1) if (not np.isinf(a) and b >= 0) else \
               (b - a + 1 if not np.isinf(a) else 0)
    L = K + K_append

    # ── flatten (K, *S, Nprev) -> (K, S_flat * Nprev) ────────────────────────
    # The kron(I_Nprev, A) block-diagonal structure means channel j maps
    # to output rows j*N .. (j+1)*N-1, independent of all other j.
    # We can therefore treat (S_flat * Nprev) as an independent channel batch.
    xi_N_2d = xi_N.reshape(K, S_flat * Nprev)   # (K, S_flat*Nprev)

    # ── boundary driving signals ──────────────────────────────────────────────
    y_db = np.zeros((L, S_flat * Nprev))
    y_db[:K] = xi_N_2d * gamma_b

    y_da = np.zeros((L, S_flat * Nprev))
    if not np.isinf(a):
        y_da[K_append:K + K_append] = xi_N_2d * gamma_a

    use_per_row_iir = (sos_iir_b_list is not None and sos_iir_a_list is not None)
    use_gamma_shift = (n_poles_b_list  is not None and n_poles_a_list  is not None)

    # ── reshape xi for in-place accumulation ──────────────────────────────────
    # (K, *S, Nprev*N) -> (K, S_flat, Nprev, N) -> (K, S_flat*Nprev, N)
    xi_r = xi.reshape(K, S_flat, Nprev, N).reshape(K, S_flat * Nprev, N)

    for n_ in range(N):
        Lout = L + max(db_list[n_], da_list[n_]) + 1
        fb = _apply_fir_batched(sos_b_list[n_], db_list[n_], y_db, Lout)  # (Lout, S*Nprev)
        fa = _apply_fir_batched(sos_a_list[n_], da_list[n_], y_da, Lout)  # (Lout, S*Nprev)

        if use_per_row_iir:
            np_b = n_poles_b_list[n_] if use_gamma_shift else None
            np_a = n_poles_a_list[n_] if use_gamma_shift else None
            iir = _parallel_iir_diff(sos_iir_b_list[n_], sos_iir_a_list[n_], fb, fa,
                                     gamma_inv, np_b, np_a, use_gamma_shift, batched=True)
        else:
            iir = sosfilt(sos_iir, (fb - fa).T).T          # (Lout, S*Nprev)

        if beta != 1:
            iir *= beta

        if b >= 0:
            xi_r[:, :, n_] += iir[b:b + K]
        else:
            xi_r[-b:, :, n_] += iir[0:K + b]

    # Write back (xi_r may be a copy if reshape broke contiguity)
    xi[:] = xi_r.reshape(K, *S_shape, Nprev * N)


@profile
def lfilter_xi_asterisk_l_backward_parallel_recursion(xi, nd, a, b, delta, gamma, xi_N, v, beta):
    r"""
    SOS-based backward parallel filter for the asterisk-l recursion
    $\xi^{(1)*l}$.

    Mirror of [`lfilter_xi_asterisk_l_forward_parallel_recursion`][lmlib.statespace.backends.rec_lfilter.lfilter_xi_asterisk_l_forward_parallel_recursion] for
    backward segments.  The time axis of ``xi_N`` is reversed before the SOS
    filter pass, and the output is reversed again before accumulation, exactly
    mirroring [`lfilter_backward_parallel_xi`][lmlib.statespace.backends.rec_lfilter.lfilter_backward_parallel_xi].

    Parameters
    ----------
    xi : ndarray, shape (K, \*S, Nprev \* N)
        Output buffer, updated in-place.
    nd : list
        11-element SOS list from [`_build_parallel_ast_sos`][lmlib.statespace.backends.rec_lfilter._build_parallel_ast_sos] with
        ``direction='bw'``.
    a : int or inf
        Left segment boundary.
    b : int or inf
        Right segment boundary.
    delta : int
        Window normalisation shift.
    gamma : float
        Window decay factor.
    xi_N : ndarray, shape (K, \*S, Nprev)
        Input from the lower-dimensional recursion.
    v : ndarray
        Sample weights (unused; accepted for interface consistency).
    beta : float
        Segment cost weight.
    """
    K       = xi_N.shape[0]
    Nprev   = xi_N.shape[-1]
    S_shape = xi_N.shape[1:-1]
    S_flat  = int(np.prod(S_shape)) if S_shape else 1
    N       = xi.shape[-1] // Nprev

    sos_iir      = nd[0]
    sos_b_list   = nd[1]; sos_a_list   = nd[2]
    db_list      = nd[3]; da_list      = nd[4]
    sos_iir_b_list = nd[5]  if len(nd) > 5  else None
    sos_iir_a_list = nd[6]  if len(nd) > 6  else None
    n_poles_b_list = nd[7]  if len(nd) > 7  else None
    n_poles_a_list = nd[8]  if len(nd) > 8  else None
    advance_b_list = nd[9]  if len(nd) > 9  else None
    advance_a_list = nd[10] if len(nd) > 10 else None

    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)

    K_append = (b - a + 1) if not np.isinf(b) else 0
    L = K + K_append

    xi_N_2d      = xi_N.reshape(K, S_flat * Nprev)
    xi_N_flipped = xi_N_2d[::-1]                           # time-reverse

    y_da = np.zeros((L, S_flat * Nprev))
    y_da[:K] = xi_N_flipped * gamma_a

    y_db = np.zeros((L, S_flat * Nprev))
    if not np.isinf(b):
        y_db[K_append:K + K_append] = xi_N_flipped * gamma_b

    use_per_row_iir = (sos_iir_b_list is not None and sos_iir_a_list is not None)
    use_gamma_shift = (n_poles_b_list  is not None and n_poles_a_list  is not None)

    xi_r = xi.reshape(K, S_flat, Nprev, N).reshape(K, S_flat * Nprev, N)

    for n_ in range(N):
        Lout = L + max(db_list[n_], da_list[n_]) + 1
        fa = _apply_fir_batched(sos_a_list[n_], da_list[n_], y_da, Lout)
        fb = _apply_fir_batched(sos_b_list[n_], db_list[n_], y_db, Lout)

        if use_per_row_iir:
            np_a = n_poles_a_list[n_] if use_gamma_shift else None
            np_b = n_poles_b_list[n_] if use_gamma_shift else None
            iir = _parallel_iir_diff(sos_iir_a_list[n_], sos_iir_b_list[n_], fa, fb,
                                     gamma, np_a, np_b, use_gamma_shift, batched=True)
        else:
            iir = sosfilt(sos_iir, (fa - fb).T).T

        if beta != 1:
            iir *= beta

        if a >= 0:
            end = K - a
            if end > 0:
                xi_r[:end, :, n_] += iir[0:end][::-1]
        else:
            xi_r[:, :, n_] += iir[-a:K - a][::-1]

    xi[:] = xi_r.reshape(K, *S_shape, Nprev * N)


def lfilter_parallel_xi_asterisk_split(xi_curr, A, C, a, b, delta, gamma, direction,
                                       xi_prev, v, beta, block_sizes):
    r"""
    Per-ALSSM-block parallel asterisk-l recursion ($\xi^{(1)*l}$, Option A).

    For each ALSSM block of the current dimension the (small) block transfer
    function is built and filtered independently — avoiding the ill-conditioned
    state-space → transfer-function conversion that the full block-diagonal
    matrix would incur — and the block result is scattered into the combined
    Kronecker layout of ``xi_curr``.

    A single-ALSSM current dimension reduces to one block, so the scatter is the
    identity and this matches [`lfilter_xi_asterisk_l_forward_parallel_recursion`][lmlib.statespace.backends.rec_lfilter.lfilter_xi_asterisk_l_forward_parallel_recursion]
    applied directly.

    Layout
    ------
    Block ``m`` (order ``N_m`` at offset ``[n0, n1)``) produces ``xi_tmp`` with
    trailing layout ``n_prev * N_m + i``.  It is scattered into ``xi_curr`` whose
    trailing layout is ``n_prev * N + n_curr`` (``N`` = total current-dim order,
    ``n_curr = n0 + i``).  The scatter is done per ``n_prev`` with contiguous
    slices so it is robust to the (Fortran-order / moved-axis) memory layout of
    ``xi_curr``.
    """
    Nq_prev = xi_prev.shape[-1]
    N = A.shape[0]
    for n0, n1 in _block_ranges(block_sizes, N):
        C_m = C[:, n0:n1]
        if not np.any(C_m):
            continue  # inactive block (F[m, p] == 0)
        A_m = A[n0:n1, n0:n1]
        N_m = n1 - n0
        nd = _build_parallel_ast_sos(A_m, C_m, a, b, delta, gamma, direction)
        xi_tmp = np.zeros((*xi_prev.shape[:-1], Nq_prev * N_m), order='F')
        if direction == 'fw':
            lfilter_xi_asterisk_l_forward_parallel_recursion(
                xi_tmp, nd, a, b, delta, gamma, xi_prev, v, beta)
        else:
            lfilter_xi_asterisk_l_backward_parallel_recursion(
                xi_tmp, nd, a, b, delta, gamma, xi_prev, v, beta)
        # scatter block result into the Kronecker layout of xi_curr
        for n_prev in range(Nq_prev):
            xi_curr[..., n_prev * N + n0: n_prev * N + n1] += \
                xi_tmp[..., n_prev * N_m: (n_prev + 1) * N_m]


def build_parallel_numdenom(A, C, a, b, delta, gamma, direction, block_sizes):
    r"""
    Build (and cache) the per-ALSSM transfer-function (SOS) coefficients consumed
    by the parallel $\xi^{(1)}$ filter.

    The combined ``A`` is block-diagonal and ``C`` is block-partitioned
    (``C[:, n0:n1] == f_m C_m`` for ALSSM ``m``).  Each block is decomposed
    independently — the parallel form converts state-space to transfer function
    per output row, which is ill-conditioned on the full block-diagonal matrix
    (clustered poles) but well-behaved per block.

    The result is fully determined by ``(A, C, a, b, delta, gamma, direction,
    block_sizes)`` — it does not depend on the signal or the sample weights — and
    the decomposition (QZ + pole-zero cancellation + per-row SOS) is comparatively
    expensive, so it is memoised in a process-wide static cache (see
    [`clear_lfilter_caches`][lmlib.statespace.backends.rec_lfilter.clear_lfilter_caches]).

    Unlike the cascade parameters, the parallel plan is already built only once
    per RLSAlssm object (in ``RLSAlssm._build_parallel_plan`` at construction
    time) and reused across ``filter()`` calls, so this cache does not eliminate a
    per-slice/per-filter hot loop.  Its benefit is sharing the plan across
    *different* RLSAlssm objects with the same model + window + block layout.
    Consumers only read the returned plan, so sharing it is safe.  The actual
    construction lives in
    [`_build_parallel_numdenom_uncached`][lmlib.statespace.backends.rec_lfilter._build_parallel_numdenom_uncached].

    Returns
    -------
    list of (n0, n1, numdenom_m)
        One entry per *active* ALSSM block (a block is inactive when its ``C``
        slice is all-zero, i.e. ``F[m, p] == 0``).  ``numdenom_m`` is the
        11-element coefficient list expected by [`lfilter_parallel_xi1`][lmlib.statespace.backends.rec_lfilter.lfilter_parallel_xi1].

    Notes
    -----
    Relocated here (was ``RLSAlssm._build_numdenom``) so the parallel realization
    lives entirely in the backend.  The ``numdenom_m`` layout is documented in
    [`lfilter_forward_parallel_xi`][lmlib.statespace.backends.rec_lfilter.lfilter_forward_parallel_xi].
    """
    key = _build_parallel_numdenom_key(A, C, a, b, delta, gamma, direction, block_sizes)
    plan = _BUILD_PARALLEL_NUMDENOM_CACHE.get(key)
    if plan is None:
        plan = _build_parallel_numdenom_uncached(
            A, C, a, b, delta, gamma, direction, block_sizes)
        _BUILD_PARALLEL_NUMDENOM_CACHE[key] = plan
    return plan


def _build_parallel_numdenom_uncached(A, C, a, b, delta, gamma, direction, block_sizes):
    r"""
    Build the per-ALSSM transfer-function (SOS) coefficients for the parallel
    $\xi^{(1)}$ filter.

    Pure function with no caching; see
    [`build_parallel_numdenom`][lmlib.statespace.backends.rec_lfilter.build_parallel_numdenom]
    for the cached entry point and full parameter/return documentation.
    """
    from lmlib.statespace.backends.statespace_tools import ss2zpk_qz

    plan = []
    for n0, n1 in _block_ranges(block_sizes, A.shape[0]):
        A_m = A[n0:n1, n0:n1]
        C_m = C[:, n0:n1]
        if not np.any(C_m):
            continue  # inactive block (F[m, p] == 0)
        N_m = n1 - n0

        if direction == 'fw':
            gAT = (1.0 / gamma) * inv(A_m).T
            Aac = (matrix_power(A_m, 0 if np.isinf(a) else a - 1).T @ C_m.T).ravel()
            Abc = (matrix_power(A_m, b).T @ C_m.T).ravel()
        elif direction == 'bw':
            gAT = gamma * A_m.T
            Aac = (matrix_power(A_m, a).T @ C_m.T).ravel()
            Abc = (matrix_power(A_m, 0 if np.isinf(b) else b + 1).T @ C_m.T).ravel()
        else:
            raise NotImplementedError("Segment direction must be fw or bw")

        poles = eigvals(gAT)
        sos_iir_shared = zpk2sos(np.zeros(len(poles)), poles, 1.0)

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
            advance_b_list.append(1 if abs(float(Abc[n_])) < 1e-10 else 0)
            advance_a_list.append(1 if abs(float(Aac[n_])) < 1e-10 else 0)

        numdenom_m = [sos_iir_shared, sos_b_list, sos_a_list, db_list, da_list,
                      sos_iir_b_list, sos_iir_a_list, n_poles_b_list, n_poles_a_list,
                      advance_b_list, advance_a_list]
        plan.append((n0, n1, numdenom_m))
    return plan


def lfilter_parallel_xi1_split(xi1, plan, a, b, direction, delta, gamma, y, sample_weights, beta):
    r"""
    Per-ALSSM parallel $\xi^{(1)}$ recursion.

    ``plan`` is the list produced by [`build_parallel_numdenom`][lmlib.statespace.backends.rec_lfilter.build_parallel_numdenom]; each active
    ALSSM block is filtered independently and written into its sub-slice
    ``xi1[..., n0:n1]`` (block-diagonal ``A`` ⇒ blocks are independent for
    $q = 1$).
    """
    for n0, n1, nd in plan:
        _iir_b = nd[5] if len(nd) > 5 else None
        _iir_a = nd[6] if len(nd) > 6 else None
        _np_b = nd[7] if len(nd) > 7 else None
        _np_a = nd[8] if len(nd) > 8 else None
        _adv_b = nd[9] if len(nd) > 9 else None
        _adv_a = nd[10] if len(nd) > 10 else None
        lfilter_parallel_xi1(xi1[..., n0:n1], nd[0], nd[1], nd[2], nd[3], nd[4],
                             a, b, direction, delta, gamma, y, sample_weights, beta,
                             _iir_b, _iir_a, _np_b, _np_a, _adv_b, _adv_a)


# xi2 lfilter parallel
def lfilter_parallel_xi2(xi2, denom, num_b, num_a, a, b, direction, delta, gamma, y, sample_weights, beta):
    r"""Compute $\xi^{(2)}$ using the parallel lfilter backend (transfer-function form)."""
    raise NotImplementedError("lfilter_parallel_xi2 not implemented yet.")


# xi1 lfilter parallel
def lfilter_parallel_xi1(xi1, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                          a, b, direction, delta, gamma, y, sample_weights, beta,
                          sos_iir_b_list=None, sos_iir_a_list=None,
                          n_poles_b_list=None, n_poles_a_list=None,
                          advance_b_list=None, advance_a_list=None):
    r"""Compute $\xi^{(1)}$ using the parallel lfilter backend (transfer-function form)."""
    if direction == 'fw':
        lfilter_forward_parallel_xi(xi1, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                                     a, b, delta, gamma, y, sample_weights, beta,
                                     sos_iir_b_list, sos_iir_a_list,
                                     n_poles_b_list, n_poles_a_list)
    elif direction == 'bw':
        lfilter_backward_parallel_xi(xi1, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                                      a, b, delta, gamma, y, sample_weights, beta,
                                      sos_iir_b_list, sos_iir_a_list,
                                      n_poles_b_list, n_poles_a_list,
                                      advance_b_list, advance_a_list)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi0 lfilter parallel (delegates to cascade implementation, since the ALSSM is not used for xi0 calculation)
def lfilter_parallel_xi0(xi0, denom, num_b, num_a, a, b, direction, delta, gamma, y, sample_weights, beta, kappa_diag=True):
    r"""Compute $\xi^{(0)}$ using the parallel lfilter backend (delegates to cascade internally)."""
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    lfilter_cascade_xi0(xi0,_A,_C,a,b,direction,delta,gamma,y,sample_weights,beta)


# nu lfilter parallel
def lfilter_parallel_nu(nu, A, C, a, b, direction, delta, gamma, y, sample_weights, beta):
    r"""Compute $\nu$ using the parallel lfilter backend (delegates to cascade internally)."""
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = np.broadcast_to(1., np.shape(y))  # create an array of shape Ks, but contains only a single 1.0 in memory
    _params = _compute_cascade_params(_A, _C, a, b, delta, gamma, direction)
    if direction == 'fw':
        lfilter_forward_cascade_xi(nu, _A, _C, a, b, delta, gamma, _y, sample_weights, beta, _params)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(nu, _A, _C, a, b, delta, gamma, _y, sample_weights, beta, _params)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


def _make_num_sos(num_row):
    """Build a numerator-only SOS from a single row of ss2tf output.

    ss2tf always inserts a leading zero in each numerator row (the z^{-1}
    normalisation).  This helper strips that leading zero, finds any further
    leading zeros (extra delay factors), extracts the finite zeros with
    np.roots and returns a zpk2sos filter together with the extra delay count.

    Returns
    -------
    sos : ndarray, shape (n_sections, 6)  or  None  (all-zero numerator)
    extra_delay : int  (number of additional z^{-1} factors beyond the one
                        already stripped)
    """
    poly = num_row[1:]          # strip the ss2tf z^{-1}
    nz = np.argmax(np.abs(poly) > 1e-300)
    if np.abs(poly[nz]) < 1e-300:
        return None, 0          # numerically zero – contributes nothing
    extra_delay = nz
    poly_trimmed = poly[nz:]
    gain = poly_trimmed[0]
    zeros_finite = np.roots(poly_trimmed / gain) if len(poly_trimmed) > 1 else np.array([])
    zeros_at_zero = np.zeros(extra_delay)
    zeros = np.concatenate([zeros_finite, zeros_at_zero])
    if len(zeros) > 0:
        sos = zpk2sos(zeros, np.zeros(len(zeros)), gain)
    else:
        sos = np.array([[gain, 0., 0., 1., 0., 0.]])
    return sos, extra_delay


def _zpk_cancel_and_build_sos(zeros, gain, iir_poles, tol=1e-3, n_inf_zeros=0):
    """Build FIR SOS and reduced IIR SOS from QZ-computed zeros with PZ cancellation.

    Takes zeros from ``ss2zpk_qz`` (computed via the QZ algorithm without
    polynomial expansion) and performs explicit pole-zero cancellation against
    the IIR poles.  Returns both the (reduced) FIR SOS and the matching
    (reduced) IIR SOS so that the caller can apply them as paired per-row
    filters rather than using a shared full-order IIR.

    The benefit: for polynomial ALSSMs where all IIR poles equal ``gamma_inv``,
    the FIR numerator for row ``n_`` has ``N-1-n_`` zeros also equal to
    ``gamma_inv`` (exact when using QZ zeros).  Cancelling them reduces the
    IIR from order N to order ``n_+1``, dramatically improving precision for
    low-index rows.

    Parameters
    ----------
    zeros : ndarray
        FIR zeros from ``ss2zpk_qz``.
    gain : float
        System gain from ``ss2zpk_qz``.
    iir_poles : ndarray
        IIR poles (eigenvalues of gAT).
    tol : float, optional
        Cancellation matching tolerance. Default 1e-3.

    Returns
    -------
    sos_fir : ndarray, shape (n_sections, 6)
        Numerator-only SOS for the reduced FIR stage.
    extra_delay : int
        Number of z^{-1} delay factors (zeros at the origin), same convention
        as ``_apply_fir``.
    sos_iir_reduced : ndarray, shape (n_sections, 6)
        Reduced-order IIR SOS (poles remaining after cancellation).
    """
    zeros_rem = list(np.asarray(zeros).ravel())
    poles_rem = list(np.asarray(iir_poles).ravel())

    # -- Greedy pole-zero cancellation ----------------------------------------
    for z in list(zeros_rem):
        if not poles_rem:
            break
        dists = [abs(z - p) for p in poles_rem]
        idx = int(np.argmin(dists))
        if dists[idx] < tol:
            zeros_rem.remove(z)
            poles_rem.pop(idx)

    zeros_rem = np.asarray(zeros_rem)
    n_rem = len(poles_rem)

    # -- Reduced IIR SOS ------------------------------------------------------
    sos_iir_red = (np.array([[1., 0., 0., 1., 0., 0.]])
                   if n_rem == 0
                   else zpk2sos(np.zeros(n_rem), np.asarray(poles_rem), 1.0))

    # -- FIR SOS from remaining zeros -----------------------------------------
    # extra_delay accounts for two sources of output-sample shift:
    #
    # 1. Dropped infinite QZ eigenvalues (n_inf_zeros): the Rosenbrock pencil
    #    always has exactly 1 structural infinite eigenvalue (from the rank-N
    #    lead matrix E), so the effective count is (n_inf_zeros - 1).  Each
    #    additional dropped eigenvalue represents one z^{-1} delay from a
    #    numerator degree reduction.
    #
    # 2. HUGE finite zeros (|z| >> pole scale): QZ sometimes returns a zero at
    #    very large |z| when the true zero is at infinity.  Each contributes
    #    one z^{-1} delay.  We absorb it into the gain: eff_gain *= -z_huge,
    #    and add 1 to extra_delay per such zero.

    pole_scale = float(np.max(np.abs(iir_poles))) if len(iir_poles) else 1.0
    huge_tol   = 1e6 * (pole_scale + 1e-12)

    # Absorb huge zeros: each one becomes a z^{-1} delay + gain factor.
    finite_zeros = []
    eff_gain = float(gain)
    n_huge = 0
    for zi in zeros_rem:
        if abs(zi) > huge_tol:
            eff_gain *= -float(zi.real if zi.imag == 0 else abs(zi))
            n_huge += 1
        else:
            finite_zeros.append(zi)
    zeros_rem = np.asarray(finite_zeros)

    # The Rosenbrock pencil always produces exactly 2 "free" infinite
    # eigenvalues for any system (1 structural from E's rank deficiency,
    # 1 from the ss2tf z^{-1} normalisation convention).  Every additional
    # dropped eigenvalue beyond these 2 represents a genuine extra z^{-1}
    # delay from the numerator degree reduction.  This formula is correct
    # for both forward and backward filters regardless of boundary sign.
    extra_delay = max(0, n_inf_zeros - 2) + n_huge

    if len(zeros_rem) == 0:
        sos_fir = np.array([[eff_gain, 0., 0., 1., 0., 0.]])
    else:
        # Snap near-real zeros and enforce conjugate symmetry before zpk2sos.
        from lmlib.statespace.backends.statespace_tools import _sanitize_zeros
        zeros_rem = _sanitize_zeros(zeros_rem)
        sos_fir = zpk2sos(zeros_rem, np.zeros(len(zeros_rem)), eff_gain)

    return sos_fir, extra_delay, sos_iir_red


_FIR_TAP_CACHE = {}


def _sos_to_fir_taps(sos):
    r"""Convert a numerator-only SOS (poles at 0) to its FIR tap vector, memoized.

    The parallel-form numerators are short (1-3 taps after pole-zero
    cancellation). Applying them with ``sosfilt`` runs the full sequential biquad
    recurrence — the expensive K-long scan — just to compute a few-tap (often
    single-tap, i.e. a scalar multiply) convolution. We instead apply them as a
    direct causal convolution (``_fir_taps``), which is cheap and, on the GPU,
    fully parallel. Shared by the lfilter and cupy backends so both stay
    bit-for-bit identical.
    """
    key = np.asarray(sos, dtype=np.float64).tobytes()
    taps = _FIR_TAP_CACHE.get(key)
    if taps is None:
        b, _a = sos2tf(np.asarray(sos, dtype=np.float64))
        b = np.trim_zeros(np.asarray(b, dtype=np.float64).ravel(), 'b')
        if b.size == 0:
            b = np.zeros(1)
        taps = b
        _FIR_TAP_CACHE[key] = taps
    return taps


def _fir_taps(taps, x):
    r"""Causal FIR ``out[k] = sum_j taps[j]*x[k-j]`` (zero IC) via shift-and-add.

    ``x`` is ``(L,)`` or ``(L, C)``. Equivalent to ``sosfilt`` of the
    numerator-only SOS but without the sequential recurrence.
    """
    out = taps[0] * x
    for j in range(1, len(taps)):
        if taps[j] != 0.0:
            out[j:] += taps[j] * x[:-j]
    return out


def _apply_fir(sos, extra_delay, y_sig, Lout):
    """Apply a numerator SOS filter (as a cheap tap convolution) with an
    additional integer delay.

    The output array always has length *Lout*.  Any samples beyond the
    filter's natural output are zero-padded; samples that would fall before
    index 0 are silently dropped.
    """
    result = np.zeros(Lout)
    if sos is None:
        return result
    filtered = _fir_taps(_sos_to_fir_taps(sos), y_sig)   # length == len(y_sig)
    end = min(extra_delay + len(filtered), Lout)
    result[extra_delay:end] = filtered[:end - extra_delay]
    return result

def _poles_are_real(sos_iir_red):
    """Return True if the IIR SOS has only real poles (a2 ≈ 0 everywhere).

    The gamma-shift IIR is only valid when all poles are real and equal.
    If any section has ``a2 != 0``, the poles form complex conjugate pairs
    (e.g. AlssmSin) and ``_gamma_shift_iir`` must not be used.
    """
    return all(abs(s[5]) < 1e-10 for s in sos_iir_red)


def _count_poles_in_sos(sos_iir_red):
    """Return the number of poles encoded in a reduced per-row IIR SOS.

    An all-pass section ``[1, 0, 0, 1, 0, 0]`` means zero poles.
    Each SOS section contributes 2 poles if ``a2 != 0``, otherwise 1 pole
    (first-order section).
    """
    if sos_iir_red.shape == (1, 6) and np.allclose(sos_iir_red[0], [1, 0, 0, 1, 0, 0]):
        return 0
    return sum(2 if abs(s[5]) > 1e-15 else 1 for s in sos_iir_red)


def _gamma_shift_iir(x, n_poles, gamma_inv):
    """Apply an n-pole IIR (all poles at *gamma_inv*) via frequency-shift + cumsums.

    Replaces ``sosfilt`` for ``n_poles >= 2``.  ``sosfilt`` with poles near
    z = 1 accumulates O(K · g^{n_poles} · eps) error because the running state
    is large (~dc_gain · signal) and each step multiplies by the near-1
    coefficient.  The gamma-shift reformulation avoids this:

      1. u[k]  = x[k] · (1/gamma_inv)^k    (shift poles from gamma_inv to z = 1)
      2. v     = cumsum^{n_poles}(u)         (integrate at z = 1 — no coefficient)
      3. y[k]  = v[k] · gamma_inv^k         (shift back)

    The IIR algorithm error is O(K^{n_poles + 0.5} · eps), but in practice the
    total filter error is dominated by the float64 FIR coefficient precision,
    which plateaus for K >> g independently of which IIR implementation is used.
    Gamma-shift is still 95–440× more accurate than sosfilt for rows 1–3 because
    it avoids the O(K · g^{n_poles} · eps) sosfilt growth that dominates for large K.

    For ``n_poles == 1`` use plain ``sosfilt`` — it is already near machine
    precision for a single-pole section and avoids the overhead.
    """
    k = np.arange(len(x), dtype=np.float64)
    u = x * (1.0 / gamma_inv) ** k
    for _ in range(n_poles):
        u = np.cumsum(u)
    return u * (gamma_inv ** k)


def _parallel_iir_diff(sos_b, sos_a, sig_b, sig_a, gamma_pole,
                       np_b=None, np_a=None, use_gamma_shift=False, batched=False):
    r"""
    Per-row parallel IIR applied to the two boundary FIR branches, returning the
    difference ``IIR_b(sig_b) - IIR_a(sig_a)``.

    When the two branch denominators are identical — the common case, as both
    branches share the poles ``eigvals(gAT)`` and the QZ pole-zero cancellation
    removes the same poles per row — this fuses via linearity to a **single** IIR
    pass on the pre-differenced signal (``IIR(sig_b - sig_a)``). That halves the
    IIR passes and keeps the running state bounded (``sig_b - sig_a`` is the
    windowed difference) instead of subtracting two separately-grown near-unit-
    pole outputs, improving conditioning. Both realizations (``sosfilt`` and the
    linear ``_gamma_shift_iir``) fuse; it falls back to the exact two-pass form
    only when the denominators differ (unequal cancellation between branches).

    ``batched`` selects the 2-D ``(L, C)`` path (channel axis as columns).
    """
    fuse = (np_b == np_a) and np.array_equal(np.asarray(sos_b), np.asarray(sos_a))

    def _iir(sos, np_, sig):
        if use_gamma_shift and np_ is not None and np_ >= 2 and _poles_are_real(sos):
            if batched:
                return np.column_stack([_gamma_shift_iir(sig[:, ch], np_, gamma_pole)
                                        for ch in range(sig.shape[1])])
            return _gamma_shift_iir(sig, np_, gamma_pole)
        return sosfilt(sos, sig.T).T if batched else sosfilt(sos, sig)

    if fuse:
        return _iir(sos_b, np_b, sig_b - sig_a)
    return _iir(sos_b, np_b, sig_b) - _iir(sos_a, np_a, sig_a)




def lfilter_forward_parallel_xi(xi, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                                 a, b, delta, gamma, y, sample_weights, beta,
                                 sos_iir_b_list=None, sos_iir_a_list=None,
                                 n_poles_b_list=None, n_poles_a_list=None):
    """SOS-based forward parallel xi filter – supports all boundary combinations.

    Parameters are the SOS structures built once in RLSAlssm._numdenom.

    Signal construction (cascade style, length K+Ka):
      y_db[:K]         = y * gamma_b   (boundary-b contribution)
      y_da[Ka:K+Ka]    = y * gamma_a   (boundary-a contribution, zero if a==-inf)

    Output slice (replicates cascade forward slicing):
      b >= 0:  iir[b : b+K]
      b  < 0:  iir[0 : K+b]  (with leading zeros at xi[-b:])

    When *sos_iir_b_list* and *sos_iir_a_list* are provided (per-row reduced IIR
    SOS from QZ-based PZ cancellation), each branch is filtered with its own
    per-row IIR and the outputs are subtracted after.

    When *n_poles_b_list* / *n_poles_a_list* are also provided, ``sosfilt`` is
    replaced by ``_gamma_shift_iir`` for any row with 2+ remaining poles.
    Gamma-shift gives 95–440× lower error than sosfilt for those rows because
    it avoids the O(K · g^{n_poles} · eps) coefficient-rounding accumulation.
    """
    gamma_a   = gamma ** (a - 1 - delta)
    gamma_b   = gamma ** (b - delta)
    gamma_inv = 1.0 / gamma
    y_weighted = (y * sample_weights[:, None]).ravel()
    K = len(y_weighted)
    N = xi.shape[1]
    if not np.isinf(a):
        K_append  = b-a+1 #this is the length of the window
    else:
        K_append = 0
    L = K + K_append

    y_db = np.zeros(L)
    y_db[:K] = y_weighted * gamma_b

    y_da = np.zeros(L)
    if not np.isinf(a):
        y_da[K_append:K + K_append] = y_weighted * gamma_a

    use_per_row_iir  = (sos_iir_b_list is not None and sos_iir_a_list is not None)
    use_gamma_shift  = (n_poles_b_list  is not None and n_poles_a_list  is not None)

    for n_ in range(N):
        Lout = L + max(db_list[n_], da_list[n_]) + 1
        fb = _apply_fir(sos_b_list[n_], db_list[n_], y_db, Lout)
        fa = _apply_fir(sos_a_list[n_], da_list[n_], y_da, Lout)
        if use_per_row_iir:
            np_b = n_poles_b_list[n_] if use_gamma_shift else None
            np_a = n_poles_a_list[n_] if use_gamma_shift else None
            iir = _parallel_iir_diff(sos_iir_b_list[n_], sos_iir_a_list[n_], fb, fa,
                                     gamma_inv, np_b, np_a, use_gamma_shift)
        else:
            iir = sosfilt(sos_iir, fb - fa)

        # SE weight for this cost segment
        if beta != 1:
            iir *= beta

        if b >= 0:
            xi[:, n_] += iir[b:b + K]
        else:
            xi[-b:, n_] += iir[0:K + b]




def lfilter_backward_parallel_xi(xi, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                                  a, b, delta, gamma, y, sample_weights, beta,
                                  sos_iir_b_list=None, sos_iir_a_list=None,
                                  n_poles_b_list=None, n_poles_a_list=None,
                                  advance_b_list=None, advance_a_list=None):
    """SOS-based backward parallel xi filter – supports all boundary combinations.

    Signal construction (cascade style, length K+Ka, time-reversed):
      y_da[:K]         = y[::-1] * gamma_a   (boundary-a contribution)
      y_db[Ka:K+Ka]    = y[::-1] * gamma_b   (boundary-b contribution, zero if b==+inf)

    Output accumulation (replicates cascade backward slicing, no explicit flip):
      a >= 0:  xi[0:K-a] += iir[0:K-a][::-1]
      a  < 0:  xi[:]     += iir[-a:K-a][::-1]

    When *sos_iir_b_list* / *sos_iir_a_list* are provided, each branch is filtered
    with its own per-row reduced IIR (from QZ-based PZ cancellation) and subtracted
    after.  When *n_poles_b_list* / *n_poles_a_list* are also provided, the
    gamma-shift IIR replaces sosfilt for rows with 2+ remaining poles, matching
    the forward filter's accuracy improvement.
    """
    gamma_a   = gamma ** (a - delta)
    gamma_b   = gamma ** (b - delta + 1)
    gamma_inv = 1.0 / gamma
    y_weighted = (y * sample_weights[:, None]).ravel()
    K = len(y_weighted)
    N = xi.shape[1]
    if not np.isinf(a):
        K_append  = b-a+1 #this is the length of the window
    else:
        K_append = 0
    L = K + K_append
    y_weighted_flipped = y_weighted[::-1]

    y_da = np.zeros(L)
    y_da[:K] = y_weighted_flipped * gamma_a

    y_db = np.zeros(L)
    if not np.isinf(b):
        y_db[K_append:K + K_append] = y_weighted_flipped * gamma_b

    use_per_row_iir  = (sos_iir_b_list is not None and sos_iir_a_list is not None)
    use_gamma_shift  = (n_poles_b_list  is not None and n_poles_a_list  is not None)

    for n_ in range(N):
        Lout = L + max(db_list[n_], da_list[n_]) + 1
        fa = _apply_fir(sos_a_list[n_], da_list[n_], y_da, Lout)
        fb = _apply_fir(sos_b_list[n_], db_list[n_], y_db, Lout)
        if use_per_row_iir:
            # backward IIR poles equal gamma (gAT = gamma * A.T), not gamma_inv.
            np_a = n_poles_a_list[n_] if use_gamma_shift else None
            np_b = n_poles_b_list[n_] if use_gamma_shift else None
            iir = _parallel_iir_diff(sos_iir_a_list[n_], sos_iir_b_list[n_], fa, fb,
                                     gamma, np_a, np_b, use_gamma_shift)
        else:
            iir = sosfilt(sos_iir, fa - fb)

        # SE weight for this cost segment
        if beta != 1:
            iir *= beta

        if a >= 0:
            end = K - a
            if end > 0:
                xi[:end, n_] += iir[0:end][::-1]
        else:
            xi[:, n_] += iir[-a:K - a][::-1]
