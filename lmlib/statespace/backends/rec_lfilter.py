import numpy as np
from numpy.linalg import inv, matrix_power
from scipy.signal import lfilter
from warnings import warn


# xi2 lfilter cascade
def lfilter_cascade_xi2(xi2, segment, A, C, y, v, beta):
    _A = np.kron(A, A)
    _C = np.kron(C, C)
    _y = np.broadcast_to(1., np.shape(y))  # create an array of shape Ks, but contains only a single 1.0 in memory

    if segment.direction == 'fw':
        lfilter_forward_cascade_xi(xi2, segment, _A, _C, _y, v, beta)
    elif segment.direction == 'bw':
        lfilter_backward_cascade_xi(xi2, segment, _A, _C, _y, v, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi1 lfilter cascade
def lfilter_cascade_xi1(xi1, segment, A, C, y, v, beta):
    if segment.direction == 'fw':
        lfilter_forward_cascade_xi(xi1, segment, A, C, y, v, beta)
    elif segment.direction == 'bw':
        lfilter_backward_cascade_xi(xi1, segment, A, C, y, v, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi0 lfilter cascade
def lfilter_cascade_xi0(xi0, segment, A, C, y, v, beta):
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = y**2

    if segment.direction == 'fw':
        lfilter_forward_cascade_xi(xi0, segment, _A, _C, _y, v, beta)
    elif segment.direction == 'bw':
        lfilter_backward_cascade_xi(xi0, segment, _A, _C, _y, v, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# nu lfilter cascade
def lfilter_cascade_nu(nu, segment, A, C, y, v, beta):
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = np.broadcast_to(1., np.shape(y))  # create an array of shape Ks, but contains only a single 1.0 in memory

    if segment.direction == 'fw':
        lfilter_forward_cascade_xi(nu, segment, _A, _C, _y, v, beta)
    elif segment.direction == 'bw':
        lfilter_backward_cascade_xi(nu, segment, _A, _C, _y, v, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# general forward cascade
def lfilter_forward_cascade_xi(xi, segment, A, C, y, v, beta):
    """
    IIR forward calculation of xi

    Due to generalization, different input parameter shapes are possible.
    The input parameter shapes are used to enhance the performance of the function (avoidance of matrix multiplication and memory allocation).
    Therefore, A, C, y, v can be scalar or nd-arrays.

    Parameters
    ----------
    xi : np.ndarray
        shape=(K, N, [S])
    segment : Segment
        Segment object carrying a, b, delta, gamma and the precompute cache.
    A : np.ndarray, scalar
        shape=(N, N)
    C : np.ndarray, scalar
        shape=([L,] N)
    y : np.ndarray, scalar
        shape=(K, [L], [S]) or 1
    v : np.ndarray
        shape=(K,) or 1
    beta : float, SE Segment weight
    """

    if not (segment.a < 0 and segment.b <= 0):
        NotImplemented('BACKEND: a and b has to be lower then zero for forward calculated segments.')

    # fetch (or compute once) all signal-independent quantities from segment cache
    gamma_inv, gamma_a, gamma_b, gAinvT, Aac, Abc, N = segment.fw_params(A, C)

    if not np.allclose(gAinvT, np.tril(gAinvT)):
        raise "State-Space Matrix A needs to be upper triangular for cascaded version"

    vy = y*v[:, None]

    # shift signal
    y_delayed_b = np.empty_like(vy)
    y_delayed_b[:-segment.b] = 0
    y_delayed_b[-segment.b:] = vy[:segment.b]#vy[:K + b]
    y_diff = np.einsum('kl, nl->kn', y_delayed_b, gamma_b * Abc)

    if not np.isinf(segment.a):
        y_delayed_a = np.empty_like(vy)
        y_delayed_a[:-segment.a + 1] = 0
        y_delayed_a[-segment.a + 1:] = vy[:segment.a - 1] # vy[:K + a - 1]
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
def lfilter_backward_cascade_xi(xi, segment, A, C, y, v, beta):
    """-
    IIR backward calculation of xi

    Due to generalization, different input parameter shapes are possible.
    The input parameter shapes are used to enhance the performance of the function (avoidance of matrix multiplication and memory allocation).
    Therefore, A, C, y, v can be scalar or nd-arrays.

    Parameters
    ----------
    xi : np.ndarray
        shape=(K, N, [S])
    segment : Segment
        Segment object carrying a, b, delta, gamma and the precompute cache.
    A : np.ndarray, scalar
        shape=(N, N)
    C : np.ndarray, scalar
        shape=([L,] N)
    y : np.ndarray, scalar
        shape=(K, [L], [S]) or 1
    v : np.ndarray
        shape=(K,) or 1
    beta : float, SE Segment weight
    """
    if not (segment.a >= 0 and segment.b > 0):
        NotImplemented('BACKEND: a and b has to be higher then zero for backward calculated segments.')

    # fetch (or compute once) all signal-independent quantities from segment cache
    gamma_a, gamma_b, gAT, Aac, Abc, N = segment.bw_params(A, C)

    if not np.allclose(gAT, np.tril(gAT)):
        raise "State-Space Matrix A needs to be upper triangular for cascaded version"

    K = len(xi)
    vy = y*v[:, None]

    # shift signal
    y_delayed_a = np.empty_like(vy)
    y_delayed_a[-segment.a:] = 0
    y_delayed_a[:K-segment.a] = vy[segment.a:]
    y_diff = np.einsum('kl, nl->kn', y_delayed_a, gamma_a * Aac)

    if not np.isinf(segment.b):
        y_delayed_b = np.empty_like(vy)
        y_delayed_b[-segment.b-1:] = 0
        y_delayed_b[:K-segment.b-1] = vy[segment.b+1:]
        y_diff -= np.einsum('kl, nl->kn', y_delayed_b, gamma_b * Abc)

    y_diff_flipped = y_diff[::-1]
    # "axis_reverse" numpy version ? TODO

    # iterating through dimensions
    y_diff_flipped = np.swapaxes(y_diff_flipped, 0, 1)  # convenient for later indexing
    xi0 = np.zeros_like(xi)
    n_ = 0
    xi0[:, n_] = lfilter([1, 0], [1, -segment.gamma], y_diff_flipped[n_].T).T
    for n_ in range(1, N):
        y_diff_flipped[n_, 1:] += np.einsum('kn..., n->k...', xi0[:-1], gAT[n_])
        xi0[:, n_] = lfilter([1, 0], [1, -segment.gamma], y_diff_flipped[n_].T).T
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