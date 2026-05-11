import numpy as np
from numpy.linalg import inv, matrix_power, eigvals
from scipy.signal import lfilter, convolve, zpk2sos, sosfilt, ss2tf


# xi2 lfilter cascade
def lfilter_cascade_xi2(xi2, A, C, a, b, direction, delta, gamma, y, sampleweights, beta):
    _A = np.kron(A, A)
    _C = np.kron(C, C)
    _y = np.broadcast_to(1., np.shape(y))  # create an array of shape Ks, but contains only a single 1.0 in memory

    if direction == 'fw':
        lfilter_forward_cascade_xi(xi2, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi2, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi1 lfilter cascade
def lfilter_cascade_xi1(xi1, A, C, a, b, direction, delta, gamma, y, sampleweights, beta):
    if direction == 'fw':
        lfilter_forward_cascade_xi(xi1, A, C, a, b, delta, gamma, y, sampleweights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi1, A, C, a, b, delta, gamma, y, sampleweights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi0 lfilter cascade
def lfilter_cascade_xi0(xi0, A, C, a, b, direction, delta, gamma, y, sampleweights, beta):
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = y**2

    if direction == 'fw':
        lfilter_forward_cascade_xi(xi0, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi0, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# nu lfilter cascade
def lfilter_cascade_nu(nu, A, C, a, b, direction, delta, gamma, y, sampleweights, beta):
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = np.broadcast_to(1., np.shape(y))  # create an array of shape Ks, but contains only a single 1.0 in memory

    if direction == 'fw':
        lfilter_forward_cascade_xi(nu, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(nu, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# general forward cascade
def lfilter_forward_cascade_xi(xi, A, C,  a, b, delta, gamma, y, sampleweights, beta):
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
    sampleweights : np.ndarray
        shape=(K,) or 1
    beta : float, SE Segment weight
    einsum_path : str (see RLSALssm)
    """

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

    yweighted = y*sampleweights[:, None]
    K = yweighted.shape[0]

    # shift signal
    # insert the shifted signal: since b > a (by definition), the recursion starts with signal b only.
    if not np.isinf(a):
        K_append  = b-a+1 #this is the length of the window
    else:
        K_append = 0
    y_delayed_b = np.zeros((K + K_append, *yweighted.shape[1:]))
    y_delayed_b[0:K] = yweighted
    y_diff = np.einsum('kl, nl->kn', y_delayed_b, gamma_b * Abc)

    if not np.isinf(a):
        # insert the shifted signal: a is inserted after K_append (length of the window).
        y_delayed_a = np.zeros((K + K_append, *yweighted.shape[1:]))
        y_delayed_a[K_append:] = yweighted
        y_diff -= np.einsum('kl, nl->kn', y_delayed_a, gamma_a * Aac)

    # iterating through ALSSM (xi) elements
    y_diff = np.swapaxes(y_diff, 0, 1)  # convenient for later indexing
    xi0 = np.zeros((K + K_append, *xi.shape[1:]))
    n_ = 0
    xi0[:, n_] = lfilter([1, 0], [1, -gamma_inv], y_diff[n_].T).T
    for n_ in range(1, N):
        y_diff[n_, 1:] += np.einsum('kn..., n->k...', xi0[:-1], gAinvT[n_])
        xi0[:, n_] = lfilter([1, 0], [1, -gamma_inv], y_diff[n_].T).T
    #xi needs to be correctly inserted. since the signal y_delayed_b had an actual delay of 0, 
    #we need to shift xi0 by b.
    if b >= 0:
        xi += xi0[b:b+K]
    #  if b < 0, first few elements of xi need to be 0 (both boundaries negative)
    if b < 0: 
        xi[-b:] += xi0[0:K+b] 


    # SE weight for this cost segment
    if beta != 1:
        xi *= beta

# general backward cascade
def lfilter_backward_cascade_xi(xi, A, C,  a, b, delta, gamma, y, sampleweights, beta):
    """-
    IIR backward calculation of xi

    Due to generalization, different input parameter shapes are possible.
    The input parameter shapes are used to enhance the performance of the function (avoidance of matrix multiplication and memory allocation).
    Therefore, A, C, y, sampleweights can be scalar or nd-arrays.

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
    sampleweights : np.ndarray
        shape=(K,) or 1
    beta : float, SE Segment weight
    einsum_path : str (see RLSALssm)
    """
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
    yweighted = y*sampleweights[:, None]

    #time-reverse observation for backward recursion
    yweighted_flipped = yweighted[::-1]
    
    # shift signal
    # insert the shifted signal: since a < b (by definition), the backward recursion starts with signal a only.
    if not np.isinf(b):
        K_append  = b-a+1 #this is the length of the window
    else:
        K_append = 0
    y_delayed_a = np.zeros((K + K_append, *yweighted_flipped.shape[1:]))
    y_delayed_a[0:K] = yweighted_flipped
    y_diff = np.einsum('kl, nl->kn', y_delayed_a, gamma_a * Aac)

    if not np.isinf(b):
        # insert the shifted signal: b is inserted after K_append (length of the window).
        y_delayed_b = np.zeros((K + K_append, *yweighted_flipped.shape[1:]))
        y_delayed_b[K_append:] = yweighted_flipped
        y_diff -= np.einsum('kl, nl->kn', y_delayed_b, gamma_b * Abc)

    # iterating through dimensions
    y_diff = np.swapaxes(y_diff, 0, 1)  # convenient for later indexing
    xi0 = np.zeros((K + K_append, *xi.shape[1:]))
    n_ = 0
    xi0[:, n_] = lfilter([1, 0], [1, -gamma], y_diff[n_].T).T
    for n_ in range(1, N):
        y_diff[n_, 1:] += np.einsum('kn..., n->k...', xi0[:-1], gAT[n_])
        xi0[:, n_] = lfilter([1, 0], [1, -gamma], y_diff[n_].T).T
    
    #xi needs to be correctly inserted. since the signal y_delayed_a had an actual delay of 0, 
    #we need to shift xi0 by a.
    xi0_flipped=xi0[::-1]
    xi_add=np.zeros_like(xi)
    #  if a >= 0, last elements of xi need to be 0 (both boundaries positive)
    if a >= 0:
        xi_add[0:K-a] += xi0_flipped[-(K-a):] 
    if a < 0: 
        xi_add += xi0_flipped[b+1:K+b+1] 
        
    xi += xi_add

    # SE weight for this cost segment
    if beta != 1:
        xi *= beta
        
        
# xi2 lfilter parallel
def lfilter_parallel_xi2(xi2, denom, num_b, num_a, a, b, direction, delta, gamma, y, sampleweights, beta):
    raise NotImplementedError("lfilter_parallel_xi2 not implemented yet.")


# xi1 lfilter parallel
def lfilter_parallel_xi1(xi1, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                          a, b, direction, delta, gamma, y, sampleweights, beta):
    if direction == 'fw':
        lfilter_forward_parallel_xi(xi1, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                                     a, b, delta, gamma, y, sampleweights, beta)
    elif direction == 'bw':
        lfilter_backward_parallel_xi(xi1, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                                      a, b, delta, gamma, y, sampleweights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi0 lfilter parallel
def lfilter_parallel_xi0(xi0, denom, num_b, num_a, a, b, direction, delta, gamma, y, sampleweights, beta, kappa_diag=True):
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = y**2

    if direction == 'fw':
        lfilter_forward_cascade_xi(xi0, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi0, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

# nu lfilter parallel
def lfilter_parallel_nu(nu, A, C, a, b, direction, delta, gamma, y, sampleweights, beta):
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = np.broadcast_to(1., np.shape(y))  # create an array of shape Ks, but contains only a single 1.0 in memory

    if direction == 'fw':
        lfilter_forward_cascade_xi(nu, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(nu, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
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


def lfilter_forward_parallel_xi(xi, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                                 a, b, delta, gamma, y, sampleweights, beta):
    """SOS-based forward parallel xi filter – supports all boundary combinations.

    Parameters are the SOS structures built once in RLSAlssm._numdenom.

    Signal construction (cascade style, length K+Ka):
      y_db[:K]         = y * gamma_b   (boundary-b contribution)
      y_da[Ka:K+Ka]    = y * gamma_a   (boundary-a contribution, zero if a==-inf)

    Output slice (replicates cascade forward slicing):
      b >= 0:  iir[b : b+K]
      b  < 0:  iir[0 : K+b]  (with leading zeros at xi[-b:])
    """
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)
    yw = (y * sampleweights[:, None]).ravel()
    K = len(yw)
    N = xi.shape[1]
    if not np.isinf(a):
        K_append  = b-a+1 #this is the length of the window
    else:
        K_append = 0
    L = K + K_append

    y_db = np.zeros(L)
    y_db[:K] = yw * gamma_b

    y_da = np.zeros(L)
    if not np.isinf(a):
        y_da[K_append:K + K_append] = yw * gamma_a

    for n_ in range(N):
        Lout = L + max(db_list[n_], da_list[n_]) + 1
        fb = _apply_fir(sos_b_list[n_], db_list[n_], y_db, Lout)
        fa = _apply_fir(sos_a_list[n_], da_list[n_], y_da, Lout)
        iir = sosfilt(sos_iir, fb - fa)
        if b >= 0:
            xi[:, n_] += iir[b:b + K]
        else:
            xi[-b:, n_] += iir[0:K + b]

    if beta != 1:
        xi *= beta


def lfilter_backward_parallel_xi(xi, sos_iir, sos_b_list, sos_a_list, db_list, da_list,
                                  a, b, delta, gamma, y, sampleweights, beta):
    """SOS-based backward parallel xi filter – supports all boundary combinations.

    Signal construction (cascade style, length K+Ka, time-reversed):
      y_da[:K]         = y[::-1] * gamma_a   (boundary-a contribution)
      y_db[Ka:K+Ka]    = y[::-1] * gamma_b   (boundary-b contribution, zero if b==+inf)

    Output accumulation (replicates cascade backward slicing, no explicit flip):
      a >= 0:  xi[0:K-a] += iir[0:K-a][::-1]
      a  < 0:  xi[:]     += iir[-a:K-a][::-1]
    """
    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)
    yw = (y * sampleweights[:, None]).ravel()
    K = len(yw)
    N = xi.shape[1]
    if not np.isinf(a):
        K_append  = b-a+1 #this is the length of the window
    else:
        K_append = 0
    L = K + K_append
    yw_r = yw[::-1]

    y_da = np.zeros(L)
    y_da[:K] = yw_r * gamma_a

    y_db = np.zeros(L)
    if not np.isinf(b):
        y_db[K_append:K + K_append] = yw_r * gamma_b

    for n_ in range(N):
        Lout = L + max(db_list[n_], da_list[n_]) + 1
        fa = _apply_fir(sos_a_list[n_], da_list[n_], y_da, Lout)
        fb = _apply_fir(sos_b_list[n_], db_list[n_], y_db, Lout)
        iir = sosfilt(sos_iir, fa - fb)
        if a >= 0:
            xi[:K - a, n_] += iir[0:K - a][::-1]
        else:
            xi[:, n_] += iir[-a:K - a][::-1]

    if beta != 1:
        xi *= beta
        
