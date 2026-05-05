import numpy as np
from numpy.linalg import inv, matrix_power
from scipy.signal import lfilter
from warnings import warn


# xi2 lfilter cascade
def lfilter_cascade_xi2(xi2, A, C, a, b, direction, delta, gamma, y, v, beta):
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
    v : np.ndarray, shape=(K,) or scalar
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

    if direction == 'fw':
        lfilter_forward_cascade_xi(xi2, _A, _C, a, b, delta, gamma, _y, v, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi2, _A, _C, a, b, delta, gamma, _y, v, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi1 lfilter cascade
def lfilter_cascade_xi1(xi1, A, C, a, b, direction, delta, gamma, y, v, beta):
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
    v : np.ndarray, shape=(K,) or scalar
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
        lfilter_forward_cascade_xi(xi1, A, C, a, b, delta, gamma, y, v, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi1, A, C, a, b, delta, gamma, y, v, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi0 lfilter cascade
def lfilter_cascade_xi0(xi0, A, C, a, b, direction, delta, gamma, y, v, beta):
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
    v : np.ndarray, shape=(K,) or scalar
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

    if direction == 'fw':
        lfilter_forward_cascade_xi(xi0, _A, _C, a, b, delta, gamma, _y, v, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi0, _A, _C, a, b, delta, gamma, _y, v, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# nu lfilter cascade
def lfilter_cascade_nu(nu, A, C, a, b, direction, delta, gamma, y, v, beta):
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = np.broadcast_to(1., np.shape(y))  # create an array of shape Ks, but contains only a single 1.0 in memory

    if direction == 'fw':
        lfilter_forward_cascade_xi(nu, _A, _C, a, b, delta, gamma, _y, v, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(nu, _A, _C, a, b, delta, gamma, _y, v, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# general forward cascade
def lfilter_forward_cascade_xi(xi, A, C,  a, b, delta, gamma, y, v, beta):
    """
    IIR forward calculation of xi

    Due to generalization, different input parameter shapes are possible.
    The input parameter shapes are used to enhance the performance of the function (avoidance of matrix multiplication and memory allocation).
    Therefore, A, C, y, v can be scalar or nd-arrays.

    Parameters
    ----------
    xi : np.ndarray
        shape=(K, N, [S])
    A : np.ndarray, scalar
        shape=(N, N)
    C : np.ndarray, scalar
        shape=([L,] N)
    a : int, inf
    b : int, inf
    delta : int
    gamma : float
    y : np.ndarray, scalar
        shape=(K, [L], [S]) or 1
    v : np.ndarray
        shape=(K,) or 1
    beta : float, SE Segment weight
    einsum_path : str (see RLSALssm)
    """

    if not (a < 0 and b <= 0):
        NotImplemented('BACKEND: a and b has to be lower then zero for forward calculated segments.')

    # gamma pre-calculation
    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)

    # state space pre-calculation separated into matrix
    A_inv = inv(A)
    gAinvT = gamma_inv * A_inv.T
    Aa = matrix_power(A, 0 if np.isinf(a) else a - 1)
    Aac = np.dot(Aa.T, C.T)
    Ab = matrix_power(A, b)
    Abc = np.dot(Ab.T, C.T)
    N = np.shape(A)[1]

    if not np.allclose(gAinvT, np.tril(gAinvT)):
        raise "State-Space Matrix A needs to be upper triangular for cascaded version"

    vy = y*v[:, None]

    # shift signal
    y_delayed_b = np.empty_like(vy)
    y_delayed_b[:-b] = 0
    y_delayed_b[-b:] = vy[:b]#vy[:K + b]
    y_diff = np.einsum('kl, nl->kn', y_delayed_b, gamma_b * Abc)

    if not np.isinf(a):
        y_delayed_a = np.empty_like(vy)
        y_delayed_a[:-a + 1] = 0
        y_delayed_a[-a + 1:] = vy[:a - 1] # vy[:K + a - 1]
        y_diff -= np.einsum('kl, nl->kn', y_delayed_a, gamma_a * Aac)

    # iterating through dimensions
    y_diff = np.swapaxes(y_diff, 0, 1)  # convenient for later indexing
    xi0 = np.zeros_like(xi)
    n_ = 0
    xi0[:, n_] = lfilter([1, 0], [1, -gamma_inv], y_diff[n_].T).T
    for n_ in range(1, N):
        y_diff[n_, 1:] += np.einsum('kn..., n->k...', xi0[:-1], gAinvT[n_])
        xi0[:, n_] = lfilter([1, 0], [1, -gamma_inv], y_diff[n_].T).T
    xi += xi0


    # SE weight for this cost segment
    if beta != 1:
        xi *= beta

# general backward cascade
def lfilter_backward_cascade_xi(xi, A, C,  a, b, delta, gamma, y, v, beta):
    """
    IIR backward calculation of xi

    Due to generalization, different input parameter shapes are possible.
    The input parameter shapes are used to enhance the performance of the function (avoidance of matrix multiplication and memory allocation).
    Therefore, A, C, y, v can be scalar or nd-arrays.

    Parameters
    ----------
    xi : np.ndarray
        shape=(K, N, [S])
    A : np.ndarray, scalar
        shape=(N, N)
    C : np.ndarray, scalar
        shape=([L,] N)
    a : int, inf
    b : int, inf
    delta : int
    gamma : float
    y : np.ndarray, scalar
        shape=(K, [L], [S]) or 1
    v : np.ndarray
        shape=(K,) or 1
    beta : float, SE Segment weight
    einsum_path : str (see RLSALssm)
    """
    if not (a >= 0 and b > 0):
        NotImplemented('BACKEND: a and b has to be higher then zero for backward calculated segments.')


    # gamma pre-calculation
    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)

    # state space pre-calculation separated into scalar and matrix
    gAT = gamma * A.T
    Aa = matrix_power(A, a)
    Aac = np.dot(Aa.T, C.T)
    Ab = matrix_power(A, 0 if np.isinf(b) else b + 1)
    Abc = np.dot(Ab.T, C.T)
    N = np.shape(A)[1]

    if not np.allclose(gAT, np.tril(gAT)):
        raise "State-Space Matrix A needs to be upper triangular for cascaded version"

    K = len(xi)
    vy = y*v[:, None]

    # shift signal
    y_delayed_a = np.empty_like(vy)
    y_delayed_a[-a:] = 0
    y_delayed_a[:K-a] = vy[a:]
    y_diff = np.einsum('kl, nl->kn', y_delayed_a, gamma_a * Aac)

    if not np.isinf(b):
        y_delayed_b = np.empty_like(vy)
        y_delayed_b[-b-1:] = 0
        y_delayed_b[:K-b-1] = vy[b+1:]
        y_diff -= np.einsum('kl, nl->kn', y_delayed_b, gamma_b * Abc)

    y_diff_flipped = y_diff[::-1]
    # "axis_reverse" numpy version ? TODO

    # iterating through dimensions
    y_diff_flipped = np.swapaxes(y_diff_flipped, 0, 1)  # convenient for later indexing
    xi0 = np.zeros_like(xi)
    n_ = 0
    xi0[:, n_] = lfilter([1, 0], [1, -gamma], y_diff_flipped[n_].T).T
    for n_ in range(1, N):
        y_diff_flipped[n_, 1:] += np.einsum('kn..., n->k...', xi0[:-1], gAT[n_])
        xi0[:, n_] = lfilter([1, 0], [1, -gamma], y_diff_flipped[n_].T).T
    xi += xi0[::-1]


    # SE weight for this cost segment
    if beta != 1:
        xi *= beta
        
        
# xi2 lfilter parallel
def lfilter_parallel_xi2(xi2, A, C, a, b, direction, delta, gamma, y, v, beta):
    raise NotImplemented("lfilter_parallel_xi2 not implemented yet.")


# xi1 lfilter parallel
def lfilter_parallel_xi1(xi1, A, C, a, b, direction, delta, gamma, y, v, beta):
    raise NotImplemented("lfilter_parallel_xi1 not implemented yet.")


# xi0 lfilter parallel
def lfilter_parallel_xi0(xi0, A, C, a, b, direction, delta, gamma, y, v, beta, kappa_diag=True):
    raise NotImplemented("lfilter_parallel_xi0 not implemented yet.")


# nu lfilter parallel
def lfilter_parallel_nu(nu, A, C, a, b, direction, delta, gamma, y, v, beta):
    raise NotImplemented("lfilter_parallel_nu not implemented yet.")