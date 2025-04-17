import numpy as np
from numpy.linalg import inv, matrix_power
from scipy.signal import lfilter

__all__ = ['forward_recursion_xi_tf', 'backward_recursion_xi_tf',
           'forward_recursion_kappa_tf', 'backward_recursion_kappa_tf',
           'forward_recursion_nu_tf', 'backward_recursion_nu_tf',
           ]


def forward_recursion_xi_tf(xi, a, b, delta, gamma, A, C, beta, y, v, einsum_path):

    if not (a < 0 and b <= 0):
        NotImplemented('BACKEND: a and b has to be lower then zero for forward calculated segments.')

    is_multichannel = np.ndim(C) == 2
    is_multiset = np.ndim(y) == 3 and is_multichannel or np.ndim(y) == 2 and not is_multichannel

    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)
    A_inv = inv(A)
    Aa = matrix_power(A, 0 if np.isinf(a) else a - 1)
    Aac = np.dot(Aa.T, C.T)
    Ab = matrix_power(A, b)
    Abc = np.dot(Ab.T, C.T)

    K = len(y)
    N = np.shape(A)[1]

    gAinvT = gamma_inv * A_inv.T
    if not np.allclose(gAinvT, np.tril(gAinvT)):
        raise "State-Space Matrix A needs to be upper triangular for cascaded version"

    vy =  np.einsum('k, k...->k...', v, y)

    y_delayed_a = np.zeros_like(y)
    y_delayed_a[-a + 1:] = vy[:K + a - 1]  # first few samples stay at 0, rest is signal, tail is cut out

    y_delayed_b = np.zeros_like(y)
    y_delayed_b[-b:] = vy[:K + b] # first few samples stay at 0, rest is signal, tail is cut out

    y_diff = np.einsum(einsum_path, y_delayed_b, gamma_b * Abc) - np.einsum(einsum_path, y_delayed_a,  gamma_a * Aac)  # take difference of new and old sample

    # filter with TF
    n_ = 0
    xi[:, n_] += lfilter([1, 0], [1, -gamma_inv], y_diff[n_]).T  # recursion: iir part
    #in_signal = np.zeros_like(y)
    for n_ in range(1, N):
        #in_signal[0] = y_diff[0, n_]
        #in_signal[1:] = np.einsum('kn..., n->k...', xi[:-1], gAinvT[n_]) + y_diff[1:, n_]  # recursion: multiply with corresponding row of A matrix (weighted sum)
        y_diff[n_,..., 1:] += np.einsum('kn..., n->...k', xi[:-1], gAinvT[n_])
        xi[:, n_] += lfilter([1, 0], [1, -gamma_inv], y_diff[n_]).T  # recursion: iir part

    xi *= beta

def backward_recursion_xi_tf(xi, a, b, delta, gamma, A, C, beta, y, v, einsum_path):
    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)
    Aa = matrix_power(A, a)
    Aac = np.dot(Aa.T, C.T)
    Ab = matrix_power(A, 0 if np.isinf(b) else b + 1)
    Abc = np.dot(Ab.T, C.T)

    K = len(y)
    N = np.shape(A)[1]

    gAT = gamma * A.T
    if not np.allclose(gAT, np.tril(gAT)):
        raise "State-Space Matrix A needs to be upper triangular for cascaded version"

    vy =  np.einsum('k, k...->k...', v, y)

    # Calculate the inner loop for all time indexes (because y is an image)
    y_delayed_a = np.zeros_like(y)
    y_delayed_a[:K - a] = vy[a:]  # begin is cut out, rest is signal, last few samples stay at 0

    y_delayed_b = np.zeros_like(y)
    y_delayed_b[:K - b - 1] = vy[b + 1:]  # begin is cut out, rest is signal, last few samples stay at 0

    y_diff = np.einsum(einsum_path, y_delayed_a, gamma_a * Aac) - np.einsum(einsum_path, y_delayed_b,  gamma_b * Abc)
    y_diff_flipped = y_diff[..., ::-1]

    xi0= np.zeros_like(xi) # TODO remove and add directly to xi

    for n_ in range(N):
        if n_ == 0:
            in_signal = y_diff[n_][..., ::-1]
        else:
            in_signal[..., 0] = y_diff[n_, ..., -1]
            in_signal[..., 1:] = gAT[n_, :] @ xi0[::-1][:-1, :].T + y_diff[n_, ..., :-1][::-1] # recursion: multiply with corresponding row of A matrix (weighted sum)
        xi0[:, n_] = lfilter([1, 0], [1, -gamma], in_signal).T[::-1]# recursion: iir part

    xi[:] += xi0 # flip xi back to correct direction #TODO: check whether it is faster to omit xi_flipped array.

    # n_ = 0
    # xi[:, n_] += lfilter([1, 0], [1, -gamma], y_diff[n_, ::-1]).T[::-1]  # recursion: iir part
    #
    # for n_ in range(1, N):
    #     y_diff[n_, ..., :-1] += np.einsum('kn..., n->...k', xi[1:], gAT[n_])  # recursion: multiply with corresponding row of A matrix (weighted sum)
    #     xi[:, n_] += lfilter([1, 0], [1, -gamma], y_diff[n_, ::-1])[..., ::-1] .T # recursion: iir part
    #
    # y_diff_flipped = y_diff[..., ::-1]
    #
    # # IIR part
    # xi_flipped = np.zeros_like(xi) # TODO remove and add directly to xi
    #
    # n_ = 0
    # xi_flipped[:, n_] = lfilter([1, 0], [1, -gamma], y_diff_flipped[n_]).T # recursion: iir part
    #
    # in_signal = np.zeros_like(y_diff[0])
    # for n_ in range(1, N):
    #     in_signal[..., 0] = y_diff_flipped[n_, ..., 0]
    #     in_signal[..., 1:] = np.einsum('kn..., n->...k', xi_flipped[:-1], gAT[n_]) + y_diff_flipped[n_,...,1:]  # recursion: multiply with corresponding row of A matrix (weighted sum)
    #     xi_flipped[:, n_] = lfilter([1, 0], [1, -gamma], in_signal).T  # recursion: iir part
    # xi[:] = xi_flipped[::-1]  # flip xi back to correct direction #TODO: check whether it is faster to omit xi_flipped array.
    #

    #IIR part
    # xi_flipped = np.zeros_like(xi) # TODO remove and add directly to xi
    #
    # n_ = 0
    # # xi[:, n_] += lfilter([1, 0], [1, -gamma], y_diff[n_, ..., ::-1]).T[::-1] # recursion: iir part
    # xi_flipped[:, n_] = lfilter([1, 0], [1, -gamma], y_diff[n_,...,::-1]).T # recursion: iir part
    # in_signal2 = np.zeros_like(y_diff[0])
    # for n_ in range(1, N):
    #     in_signal2[..., 0] = y_diff[n_, ..., -1]
    #     in_signal2[..., 1:] = np.einsum('kn..., n->...k', xi[::-1][:-1], gAT[n_]) + y_diff[n_,...,::-1][..., 1:]  # recursion: multiply with corresponding row of A matrix (weighted sum)
    #     xi_flipped[:, n_] = lfilter([1, 0], [1, -gamma], in_signal2).T  # recursion: iir part
    # xi[:] = xi_flipped[::-1]  # flip xi back to correct direction #TODO: check whether it is faster to omit xi_flipped array.

        # xi[:, n_] += lfilter([1, 0], [1, -gamma], in_signal).T[::-1]  # recursion: iir part

    # y_diff_flipped = y_diff[..., ::-1]
    # xi_flipped = np.zeros_like(xi) # TODO remove and add directly to xi
    #
    # n_ = 0
    # xi_flipped[:, n_] = lfilter([1, 0], [1, -gamma], y_diff_flipped[n_]).T # recursion: iir part
    #
    # in_signal = np.zeros_like(y_diff[0])
    # for n_ in range(1, N):
    #     in_signal[..., 0] = y_diff_flipped[n_, ..., 0]
    #     in_signal[..., 1:] = np.einsum('kn..., n->...k', xi_flipped[:-1], gAT[n_]) + y_diff_flipped[n_,...,1:]  # recursion: multiply with corresponding row of A matrix (weighted sum)
    #     xi_flipped[:, n_] = lfilter([1, 0], [1, -gamma], in_signal).T  # recursion: iir part
    # xi[:] = xi_flipped[::-1]  # flip xi back to correct direction #TODO: check whether it is faster to omit xi_flipped array.


    xi *= beta


def forward_recursion_kappa_tf(kappa, a, b, delta, gamma, beta, y, v, einsum_path):
    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)

    K = len(y)
    yTy = np.einsum(einsum_path, y, y)
    y2 = np.einsum('k, k...->k...', v, yTy)

    y_delayed_a = np.zeros_like(y2)
    y_delayed_a[-a + 1:] = y2[:K + a - 1]  # first few samples stay at 0, rest is signal, tail is cut out

    y_delayed_b = np.zeros_like(y2)
    y_delayed_b[-b:] = y2[:K + b]  # first few samples stay at 0, rest is signal, tail is cut out

    y_diff = np.dot(gamma_b, y_delayed_b) - np.dot(gamma_a, y_delayed_a)  # take difference of new and old sample

    # IIR part
    kappa[:] += lfilter([1, 0], [1, -gamma_inv], y_diff.T).T
    kappa *= beta

def backward_recursion_kappa_tf(kappa, a, b, delta, gamma, beta, y, v, einsum_path):
    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)

    K = len(y)

    yTy = np.einsum(einsum_path, y, y)
    y2 = np.einsum('k, k...->k...', v, yTy)

    y_delayed_a = np.zeros_like(y2)
    y_delayed_a[:K-a] = y2[a:]  # begin is cut out, rest is signal, last few

    y_delayed_b = np.zeros_like(y2)
    y_delayed_b[:K-b-1] = y2[b+1:]  # begin is cut out, rest is signal, last few

    y_diff = np.dot(gamma_a, y_delayed_a) - np.dot(gamma_b, y_delayed_b)  # take difference of new and old sample

    # IIR part
    kappa[::-1] += lfilter([1, 0], [1, -gamma], y_diff[::-1].T).T
    kappa *= beta


def forward_recursion_nu_tf(nu, a, b, delta, gamma, beta, v):
    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)

    K = len(v)

    v_delayed_a = np.zeros(K)
    v_delayed_a[-a + 1:] = v[:K + a - 1]  # first few samples stay at 0, rest is signal, tail is cut out

    v_delayed_b = np.zeros(K)
    v_delayed_b[-b:] = v[:K + b]  # first few samples stay at 0, rest is signal, tail is cut out

    v_diff = np.dot(gamma_b, v_delayed_b) - np.dot(gamma_a, v_delayed_a)  # take difference of new and old sample

    # IIR part
    nu[:] += lfilter([1, 0], [1, -gamma_inv], v_diff)
    nu *= beta


def backward_recursion_nu_tf(nu, a, b, delta, gamma, beta, v):
    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)

    K = len(v)

    v_delayed_a = np.zeros(K)
    v_delayed_a[:K - a] = v[a:]  # begin is cut out, rest is signal, last few

    v_delayed_b = np.zeros(K)
    v_delayed_b[:K - b - 1] = v[b + 1:]  # begin is cut out, rest is signal, last few

    ydiff = np.dot(gamma_a, v_delayed_a) - np.dot(gamma_b, v_delayed_b)  # take difference of new and old sample

    # IIR part
    nu[::-1] += lfilter([1, 0], [1, -gamma], ydiff[::-1])
    nu *= beta
