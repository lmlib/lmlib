import numpy as np
from numpy.linalg import inv, matrix_power
from scipy.signal import lfilter
from warnings import warn

# helper function for lfilter method
def _lfilter_vy(K, L, y, v):
    # returns the weighted signal based on different inputs.
    if np.isscalar(y) and np.isscalar(v):
        if False:
            warn('For Speed up use RLSAlssm(..., steady_state=True) or supress warnings with lm.WARNING_NOT_STEADY_STATE = False.')
        vy = np.ones((K, L)) if L is not None else np.ones(K)
    elif np.isscalar(y) and not np.isscalar(v):
        assert len(v) == K, 'len(v) != K'
        vy = np.einsum('k, ...->k...', v, y)
    elif not np.isscalar(y) and np.isscalar(v):
        assert len(y) == K, 'len(y) != K'
        vy = np.einsum('..., k...->k...', v, y)
    elif not np.isscalar(y) and not np.isscalar(v):
        assert len(y) == K, 'len(y) != K'
        assert len(v) == K, 'len(v) != K'
        vy = np.einsum('k, k...->k...', v, y)
    return vy

def _lfilter_einsum_path_xi2(is_multichannel):
    if is_multichannel:
        einsum_path = 'kl, nl->kn'
    else:
        einsum_path = 'k, n->kn'
    return einsum_path

def _lfilter_einsum_path_xi1(is_multichannel, is_multiset):
    if is_multiset:
        if is_multichannel:
            einsum_path = 'kls, nl->kns'
        else:
            einsum_path = 'ks, n->kns'
    else:
        if is_multichannel:
            einsum_path = 'kl, nl->kn'
        else:
            einsum_path = 'k, n->kn'
    return einsum_path

def _lfilter_einsum_path_xi0(is_multichannel, is_multiset, kappa_diag):
    if is_multiset:
        if is_multichannel:
            if kappa_diag:
                einsum_path = 'km..., ...->km...'
            else:
                einsum_path = 'kmn..., ... ->kmn'
        else:
            if kappa_diag:
                einsum_path = 'km, ...->km'
            else:
                einsum_path = 'kmn, ...->kmn'
    else:
        if is_multichannel:
            einsum_path = 'k, ...->k'
        else:
            einsum_path = 'k, ...->k'

    return einsum_path

def _lfilter_einsum_path_nu_tf():
    return 'k..., ...->k...'

def _lfilter_einsum_path_y_squared(is_multichannel, is_multiset, kappa_diag):
    if is_multiset:
        if is_multichannel:
            if kappa_diag:
                einsum_path = 'klm..., klm...->km...'
            else:
                einsum_path = 'kml..., kln... ->kmn'
        else:
            if kappa_diag:
                einsum_path = 'km, km->km'
            else:
                einsum_path = 'km, kn->kmn'
    else:
        if is_multichannel:
            einsum_path = 'kl, kl->k'
        else:
            einsum_path = 'k, k->k'

    return einsum_path


# xi2 lfilter cascade
def lfilter_cascade_xi2(xi2, A, C, a, b, direction, delta, gamma, y, v, beta):
    einsum_path = _lfilter_einsum_path_xi2(np.ndim(C)==2)
    _A = np.kron(A, A)
    _C = np.kron(C, C)
    _y = 1
    if direction == 'fw':
        lfilter_forward_cascade_xi(xi2, _A, _C, a, b, delta, gamma, _y, v, beta, einsum_path)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi2, _A, _C, a, b, delta, gamma, _y, v, beta, einsum_path)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi1 lfilter cascade
def lfilter_cascade_xi1(xi1, A, C, a, b, direction, delta, gamma, y, v, beta):
    is_multichannel = np.ndim(C) == 2
    is_multiset = np.ndim(y) == 2 and not is_multichannel or np.ndim(y) > 2
    einsum_path = _lfilter_einsum_path_xi1(is_multichannel, is_multiset)
    if direction == 'fw':
        lfilter_forward_cascade_xi(xi1, A, C, a, b, delta, gamma, y, v, beta, einsum_path)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi1, A, C, a, b, delta, gamma, y, v, beta, einsum_path)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi0 lfilter cascade
def lfilter_cascade_xi0(xi0, A, C, a, b, direction, delta, gamma, y, v, beta, kappa_diag=True):
    is_multichannel = np.ndim(C) == 2
    is_multiset = np.ndim(y) == 2 and not is_multichannel or np.ndim(y) > 2
    einsum_path = _lfilter_einsum_path_xi0(is_multichannel, is_multiset, kappa_diag)
    _A = np.array(1)
    _C = np.array(1)
    _y = np.einsum(_lfilter_einsum_path_y_squared(is_multichannel, is_multiset, kappa_diag), y, y)

    if direction == 'fw':
        lfilter_forward_cascade_xi(xi0, _A, _C, a, b, delta, gamma, _y, v, beta, einsum_path)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi0, _A, _C, a, b, delta, gamma, _y, v, beta, einsum_path)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# nu lfilter cascade
def lfilter_cascade_nu(nu, A, C, a, b, direction, delta, gamma, y, v, beta):
    einsum_path = _lfilter_einsum_path_nu_tf()
    _A = np.array(1)
    _C = np.array(1)
    _y = 1
    if direction == 'fw':
        lfilter_forward_cascade_xi(nu, _A, _C, a, b, delta, gamma, _y, v, beta, einsum_path)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(nu, _A, _C, a, b, delta, gamma, _y, v, beta, einsum_path)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# general forward cascade
def lfilter_forward_cascade_xi(xi, A, C,  a, b, delta, gamma, y, v, beta, einsum_path):
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

    # state space pre-calculation separated into scalar and matrix
    if np.ndim(A) == 0:
        A_inv = 1 / A
        gAinvT = gamma_inv * A_inv
        Aa = A
        Aac = np.dot(Aa.T, C.T)
        Ab = A
        Abc = np.dot(Ab.T, C.T)
        N = 0
    else:
        A_inv = inv(A)
        gAinvT = gamma_inv * A_inv.T
        Aa = matrix_power(A, 0 if np.isinf(a) else a - 1)
        Aac = np.dot(Aa.T, C.T)
        Ab = matrix_power(A, b)
        Abc = np.dot(Ab.T, C.T)
        N = np.shape(A)[1]

        if not np.allclose(gAinvT, np.tril(gAinvT)):
            raise "State-Space Matrix A needs to be upper triangular for cascaded version"

    K = len(xi)
    L = None if np.ndim(C) in (0, 1) else np.shape(C)[0]
    vy = _lfilter_vy(K, L, y, v) # signal and sample weight calculation

    # shift signal
    y_delayed_b = np.empty_like(vy)
    y_delayed_b[:-b] = 0
    y_delayed_b[-b:] = vy[:b]#vy[:K + b]
    y_diff = np.einsum(einsum_path, y_delayed_b, gamma_b * Abc)

    if not np.isinf(a):
        y_delayed_a = np.empty_like(vy)
        y_delayed_a[:-a + 1] = 0
        y_delayed_a[-a + 1:] = vy[:a - 1] # vy[:K + a - 1]
        y_diff -= np.einsum(einsum_path, y_delayed_a, gamma_a * Aac)

    # system as matrix / scalar
    if N != 0:
        # iterating through dimensions
        y_diff = np.swapaxes(y_diff, 0, 1)  # convenient for later indexing
        xi0 = np.zeros_like(xi)
        n_ = 0
        xi0[:, n_] = lfilter([1, 0], [1, -gamma_inv], y_diff[n_].T).T
        for n_ in range(1, N):
            y_diff[n_, 1:] += np.einsum('kn..., n->k...', xi0[:-1], gAinvT[n_])
            xi0[:, n_] = lfilter([1, 0], [1, -gamma_inv], y_diff[n_].T).T
        xi += xi0
    else:
        # as a system is scalar no iteration through dimensions possible
        xi += lfilter([1, 0], [1, -gamma_inv], y_diff.T).T

    # SE weight for this cost segment
    if beta != 1:
        xi *= beta

# general backward cascade
def lfilter_backward_cascade_xi(xi, A, C,  a, b, delta, gamma, y, v, beta, einsum_path):
    """-
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
        NotImplemented('BACKEND: a and b has to be higher then zero for  backward calculated segments.')


    # gamma pre-calculation
    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)

    # state space pre-calculation separated into scalar and matrix
    gAT = gamma * A.T
    if np.ndim(A) == 0:
        Aa = A
        Aac = np.dot(Aa.T, C.T)
        Ab = A
        Abc = np.dot(Ab.T, C.T)
        N = 0
    else:
        Aa = matrix_power(A, a)
        Aac = np.dot(Aa.T, C.T)
        Ab = matrix_power(A, 0 if np.isinf(b) else b + 1)
        Abc = np.dot(Ab.T, C.T)
        N = np.shape(A)[1]

        if not np.allclose(gAT, np.tril(gAT)):
            raise "State-Space Matrix A needs to be upper triangular for cascaded version"

    K = len(xi)
    L = None if np.ndim(C) in (0, 1) else np.shape(C)[0]
    vy = _lfilter_vy(K, L, y, v) # signal and sample weight calculation

    # shift signal
    y_delayed_a = np.empty_like(vy)
    y_delayed_a[-a:] = 0
    y_delayed_a[:K-a] = vy[a:]
    y_diff = np.einsum(einsum_path, y_delayed_a, gamma_a * Aac)

    if not np.isinf(b):
        y_delayed_b = np.empty_like(vy)
        y_delayed_b[-b-1:] = 0
        y_delayed_b[:K-b-1] = vy[b+1:]
        y_diff -= np.einsum(einsum_path, y_delayed_b, gamma_b * Abc)

    y_diff_flipped = y_diff[::-1]
    # "axis_reverse" numpy version ? TODO

    # system as matrix / scalar
    if N != 0:
        # iterating through dimensions
        y_diff_flipped = np.swapaxes(y_diff_flipped, 0, 1)  # convenient for later indexing
        xi0 = np.zeros_like(xi)
        n_ = 0
        xi0[:, n_] = lfilter([1, 0], [1, -gamma], y_diff_flipped[n_].T).T
        for n_ in range(1, N):
            y_diff_flipped[n_, 1:] += np.einsum('kn..., n->k...', xi0[:-1], gAT[n_])
            xi0[:, n_] = lfilter([1, 0], [1, -gamma], y_diff_flipped[n_].T).T
        xi += xi0[::-1]
    else:
        # as a system is scalar no iteration through dimensions possible
        xi += lfilter([1, 0], [1, -gamma], y_diff_flipped.T).T[::-1]

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