import numpy as np
from numpy.linalg import inv, matrix_power, eigvals
from scipy.signal import lfilter, convolve, zpk2sos, sosfilt, ss2tf
from lmlib.utils.profiling import profile


def _compute_cascade_params(A, C, a, b, delta, gamma, direction):
    r"""
    Precompute all state-space and gamma scalars needed by the cascade IIR filters.

    Called once per (ALSSM, segment) pair at RLSAlssm construction time and
    stored in ``_cascade_params[dim][p][m]``.  The returned dict is then passed
    directly to ``lfilter_forward_cascade_xi`` / ``lfilter_backward_cascade_xi``,
    avoiding repeated ``inv``, ``matrix_power`` and ``np.dot`` calls inside the
    filter loop.

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
        bw: gamma_a, gamma_b, gAT, Aac, Abc, N
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
    Computes the second-order cost parameter :math:`\xi^{(2)}(k, \mathbf{1})` in-place,
    which equals the vectorized Gram matrix :math:`\mathrm{vec}(W_k)`.

    :math:`W_k \in \mathbb{R}^{N \times N}` is independent of the signal `y` (it depends
    only on the model and window), so `y` is replaced by an all-ones array internally.
    The Kronecker product identity

    .. math::
        \mathrm{vec}(A^T c^T c A) = (A \otimes A)^T \mathrm{vec}(c^T c)

    allows the :math:`W_k` recursion to be recast as a standard :math:`\xi^{(1)}` recursion
    with substitutions :math:`A \to A \otimes A` and :math:`C \to C \otimes C`.
    The result is stored in `xi2` as a flat vector of length :math:`N^2`
    (i.e. :math:`\mathrm{vec}(W_k)`).

    See also [Wildhaber2018]_ Eq. (22) and [Baeriswyl2025]_ Table I.

    Parameters
    ----------
    xi2 : np.ndarray, shape=(K, N**2, [S])
        Output array, modified in-place. Stores :math:`\mathrm{vec}(W_k)` for each
        time step k. Reshaped to ``(K, N, N)`` by the caller to recover :math:`W_k`.
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
        Window decay factor :math:`\gamma`.
    y : np.ndarray, shape=(K, [L], [S]) or scalar
        Input signal. Only the shape is used (values are replaced by 1); `y` is
        passed solely to determine the number of time steps K.
    sample_weights : np.ndarray, shape=(K,) or scalar
        Per-sample weights :math:`w_i`.
    beta : float
        Cost segment weight :math:`\beta`.

    Notes
    -----
    The underlying :func:`lfilter_forward_cascade_xi` /
    :func:`lfilter_backward_cascade_xi` implement the recursion as a **cascade of
    1-D IIR filters**: each state dimension ``n`` is solved by one :func:`scipy.signal.lfilter`
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
def lfilter_cascade_xi1(xi1, A, C, a, b, direction, delta, gamma, y, sample_weights, beta):
    r"""
    Computes the first-order cost parameter :math:`\xi^{(1)}(k, y)` in-place,
    which equals the signal projection vector :math:`\xi_k`.

    :math:`\xi_k \in \mathbb{R}^{N}` depends on the signal `y` and is computed
    directly using the ALSSM matrices ``A`` and ``C`` without any substitution.
    The result is stored in `xi1` as a vector of length :math:`N`.

    See also [Wildhaber2018]_ Eq. (23) and [Baeriswyl2025]_ Table I.

    Parameters
    ----------
    xi1 : np.ndarray, shape=(K, N, [S])
        Output array, modified in-place. Stores :math:`\xi_k` for each time step k.
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
        Window decay factor :math:`\gamma`.
    y : np.ndarray, shape=(K, [L], [S]) or scalar
        Input signal. Signal values are used directly in the recursion.
    sample_weights : np.ndarray, shape=(K,) or scalar
        Per-sample weights :math:`w_i`.
    beta : float
        Cost segment weight :math:`\beta`.

    Notes
    -----
    The underlying :func:`lfilter_forward_cascade_xi` /
    :func:`lfilter_backward_cascade_xi` implement the recursion as a **cascade of
    1-D IIR filters**: each state dimension ``n`` is solved by one :func:`scipy.signal.lfilter`
    call, and its output is fed forward into the next dimension ``n+1``. This is possible
    because ``A`` must be upper-triangular, so the state equations are lower-dimensional and
    can be solved in order.

    Raises
    ------
    ValueError
        If `direction` is not ``'fw'`` or ``'bw'``.
    """
    if direction == 'fw':
        lfilter_forward_cascade_xi(xi1, cascade_params, a, b, y, sample_weights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi1, cascade_params, a, b, y, sample_weights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi0 lfilter cascade
def lfilter_cascade_xi0(xi0, A, C, a, b, direction, delta, gamma, y, sample_weights, beta):
    r"""
    Computes the zeroth-order cost parameter :math:`\xi^{(0)}(k, y)` in-place,
    which equals the weighted signal energy :math:`\kappa_k`.

    :math:`\kappa_k \in \mathbb{R}` is a scalar representing the accumulated weighted
    energy of the signal `y` within the window. It is computed by reducing the recursion
    to a scalar IIR filter via the substitutions

    .. math::
        A \to [[1]], \quad C \to [[1]], \quad y \to y^2

    so that the standard :math:`\xi^{(1)}` recursion accumulates
    :math:`\kappa_k = \sum_i w_i \, y_i^2`.
    The result is stored in `xi0` with shape ``(K, 1, [S])``.

    Parameters ``A`` and ``C`` are accepted for interface consistency but are not used.

    See also [Wildhaber2018]_ Eq. (24) and [Baeriswyl2025]_ Table I.

    Parameters
    ----------
    xi0 : np.ndarray, shape=(K, 1, [S])
        Output array, modified in-place. Stores :math:`\kappa_k` for each time step k.
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
        Window decay factor :math:`\gamma`.
    y : np.ndarray, shape=(K, [L], [S]) or scalar
        Input signal. Values are squared internally (``_y = y**2``) before the recursion.
    sample_weights : np.ndarray, shape=(K,) or scalar
        Per-sample weights :math:`w_i`.
    beta : float
        Cost segment weight :math:`\beta`.

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
def lfilter_forward_cascade_xi(xi, cascade_params, a, b, y, sample_weights, beta):
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
    n_ = 0
    xi_add[:, n_] = lfilter([1, 0], [1, -gamma_inv], y_diff[n_].T).T
    for n_ in range(1, N):
        y_diff[n_, 1:] += np.einsum('kn..., n->k...', xi_add[:-1], gAinvT[n_])
        xi_add[:, n_] = lfilter([1, 0], [1, -gamma_inv], y_diff[n_].T).T
        
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
def lfilter_backward_cascade_xi(xi, cascade_params, a, b, y, sample_weights, beta):
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
    n_ = 0
    xi_add[:, n_] = lfilter([1, 0], [1, -gamma], y_diff[n_].T).T
    for n_ in range(1, N):
        y_diff[n_, 1:] += np.einsum('kn..., n->k...', xi_add[:-1], gAT[n_])
        xi_add[:, n_] = lfilter([1, 0], [1, -gamma], y_diff[n_].T).T
    
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




# xi2 lfilter parallel
def lfilter_parallel_xi2(xi2, denom, num_b, num_a, a, b, direction, delta, gamma, y, sample_weights, beta):
    raise NotImplementedError("lfilter_parallel_xi2 not implemented yet.")


# xi1 lfilter parallel
def lfilter_parallel_xi1(xi1, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                          a, b, direction, delta, gamma, y, sample_weights, beta,
                          sos_iir_b_list=None, sos_iir_a_list=None,
                          n_poles_b_list=None, n_poles_a_list=None,
                          advance_b_list=None, advance_a_list=None):
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


# xi0 lfilter parallel
def lfilter_parallel_xi0(xi0, denom, num_b, num_a, a, b, direction, delta, gamma, y, sample_weights, beta, kappa_diag=True):
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = y**2
    _params = _compute_cascade_params(_A, _C, a, b, delta, gamma, direction)
    if direction == 'fw':
        lfilter_forward_cascade_xi(xi0, _A, _C, a, b, delta, gamma, _y, sample_weights, beta, _params)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi0, _A, _C, a, b, delta, gamma, _y, sample_weights, beta, _params)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

# nu lfilter parallel
def lfilter_parallel_nu(nu, A, C, a, b, direction, delta, gamma, y, sample_weights, beta):
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


def _apply_fir(sos, extra_delay, y_sig, Lout):
    """Apply a numerator SOS filter with an additional integer delay.

    The output array always has length *Lout*.  Any samples beyond the
    filter's natural output are zero-padded; samples that would fall before
    index 0 are silently dropped.
    """
    result = np.zeros(Lout)
    if sos is None:
        return result
    filtered = sosfilt(sos, y_sig)          # length == len(y_sig)
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
            if use_gamma_shift:
                np_b = n_poles_b_list[n_]; np_a = n_poles_a_list[n_]
                ib = (_gamma_shift_iir(fb, np_b, gamma_inv)
                      if np_b >= 2 and _poles_are_real(sos_iir_b_list[n_])
                      else sosfilt(sos_iir_b_list[n_], fb))
                ia = (_gamma_shift_iir(fa, np_a, gamma_inv)
                      if np_a >= 2 and _poles_are_real(sos_iir_a_list[n_])
                      else sosfilt(sos_iir_a_list[n_], fa))
            else:
                ib = sosfilt(sos_iir_b_list[n_], fb)
                ia = sosfilt(sos_iir_a_list[n_], fa)
            iir = ib - ia
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
            if use_gamma_shift:
                np_b = n_poles_b_list[n_]; np_a = n_poles_a_list[n_]
                # For the backward filter, gAT = gamma * A.T so the IIR poles
                # equal gamma (not gamma_inv = 1/gamma as in the forward case).
                # _gamma_shift_iir must receive the actual pole value.
                # Only use gamma-shift for real poles; complex poles (e.g.
                # AlssmSin) must fall back to sosfilt.
                ia = (_gamma_shift_iir(fa, np_a, gamma)
                      if np_a >= 2 and _poles_are_real(sos_iir_a_list[n_])
                      else sosfilt(sos_iir_a_list[n_], fa))
                ib = (_gamma_shift_iir(fb, np_b, gamma)
                      if np_b >= 2 and _poles_are_real(sos_iir_b_list[n_])
                      else sosfilt(sos_iir_b_list[n_], fb))
            else:
                ia = sosfilt(sos_iir_a_list[n_], fa)
                ib = sosfilt(sos_iir_b_list[n_], fb)
            iir = ia - ib
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
