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

    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)
    A_inv = inv(A)
    gAinvT = gamma_inv * A_inv.T
    Aa = matrix_power(A, 0 if np.isinf(a) else a - 1)
    Aac = np.dot(Aa.T, C.T)
    Ab = matrix_power(A, b)
    Abc = np.dot(Ab.T, C.T)

    if not np.allclose(gAinvT, np.tril(gAinvT)):
        raise "State-Space Matrix A needs to be upper triangular for cascaded version"

    vy = np.einsum('k, k...->k...', v, y)

    K = len(y)
    N = np.shape(A)[1]

    y_delayed_a = np.zeros_like(y)
    y_delayed_a[-a + 1:] = vy[:K + a - 1]  # first few samples stay at 0, rest is signal, tail is cut out

    y_delayed_b = np.zeros_like(y)
    y_delayed_b[-b:] = vy[:K + b]  # first few samples stay at 0, rest is signal, tail is cut out

    y_diff = (np.einsum(einsum_path, y_delayed_b, gamma_b * Abc)
              - np.einsum(einsum_path, y_delayed_a, gamma_a * Aac))  # take difference of new and old sample

    y_diff = np.swapaxes(y_diff, 0, 1)  # convenient for later indexing

    xi0 = np.zeros_like(xi)
    n_ = 0
    xi0[:, n_] = lfilter([1, 0], [1, -gamma_inv], y_diff[n_].T).T  # recursion: iir part
    for n_ in range(1, N):
        y_diff[n_, 1:] += np.einsum('kn..., n->k...', xi0[:-1], gAinvT[n_])
        xi0[:, n_] = lfilter([1, 0], [1, -gamma_inv], y_diff[n_].T).T  # recursion: iir part

    xi += xi0
    xi *= beta


def backward_recursion_xi_tf(xi, a, b, delta, gamma, A, C, beta, y, v, einsum_path):
    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)
    gAT = gamma * A.T
    Aa = matrix_power(A, a)
    Aac = np.dot(Aa.T, C.T)
    Ab = matrix_power(A, 0 if np.isinf(b) else b + 1)
    Abc = np.dot(Ab.T, C.T)

    if not np.allclose(gAT, np.tril(gAT)):
        raise "State-Space Matrix A needs to be upper triangular for cascaded version"

    vy = np.einsum('k, k...->k...', v, y)

    K = len(y)
    N = np.shape(A)[1]

    # Calculate the inner loop for all time indexes (because y is an image)
    y_delayed_a = np.zeros_like(y)
    y_delayed_a[:K - a] = vy[a:]  # begin is cut out, rest is signal, last few samples stay at 0

    y_delayed_b = np.zeros_like(y)
    y_delayed_b[:K - b - 1] = vy[b + 1:]  # begin is cut out, rest is signal, last few samples stay at 0

    y_diff = (np.einsum(einsum_path, y_delayed_a, gamma_a * Aac)
              - np.einsum(einsum_path, y_delayed_b, gamma_b * Abc))

    y_diff_flipped = y_diff[::-1]
    y_diff_flipped = np.swapaxes(y_diff_flipped, 0, 1)  # convenient for later indexing

    xi0 = np.zeros_like(xi)
    n_ = 0
    xi0[:, n_] = lfilter([1, 0], [1, -gamma], y_diff_flipped[n_].T).T  # recursion: iir part
    for n_ in range(1, N):
        y_diff_flipped[n_, 1:] += np.einsum('kn..., n->k...', xi0[:-1], gAT[n_])
        xi0[:, n_] = lfilter([1, 0], [1, -gamma], y_diff_flipped[n_].T).T  # recursion: iir part

    xi += xi0[::-1]
    xi *= beta

def forward_recursion_kappa_tf(kappa, a, b, delta, gamma, beta, y, v, einsum_path):
    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)

    yTy = np.einsum(einsum_path, y, y)
    vy = np.einsum('k, k...->k...', v, yTy)

    K = len(y)
    y_delayed_a = np.zeros_like(vy)
    y_delayed_a[-a + 1:] = vy[:K + a - 1]  # first few samples stay at 0, rest is signal, tail is cut out

    y_delayed_b = np.zeros_like(vy)
    y_delayed_b[-b:] = vy[:K + b]  # first few samples stay at 0, rest is signal, tail is cut out

    y_diff = np.dot(gamma_b, y_delayed_b) - np.dot(gamma_a, y_delayed_a)  # take difference of new and old sample

    kappa += lfilter([1, 0], [1, -gamma_inv], y_diff.T).T
    kappa *= beta


def backward_recursion_kappa_tf(kappa, a, b, delta, gamma, beta, y, v, einsum_path):
    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)

    yTy = np.einsum(einsum_path, y, y)
    vy = np.einsum('k, k...->k...', v, yTy)

    K = len(y)
    y_delayed_a = np.zeros_like(vy)
    y_delayed_a[:K - a] = vy[a:]  # begin is cut out, rest is signal, last few

    y_delayed_b = np.zeros_like(vy)
    y_delayed_b[:K - b - 1] = vy[b + 1:]  # begin is cut out, rest is signal, last few

    y_diff = np.dot(gamma_a, y_delayed_a) - np.dot(gamma_b, y_delayed_b)  # take difference of new and old sample

    kappa += lfilter([1, 0], [1, -gamma], y_diff[::-1].T).T[::-1]
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

    nu += lfilter([1, 0], [1, -gamma_inv], v_diff)
    nu *= beta


def backward_recursion_nu_tf(nu, a, b, delta, gamma, beta, v):
    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)

    K = len(v)

    v_delayed_a = np.zeros(K)
    v_delayed_a[:K - a] = v[a:]  # begin is cut out, rest is signal, last few

    v_delayed_b = np.zeros(K)
    v_delayed_b[:K - b - 1] = v[b + 1:]  # begin is cut out, rest is signal, last few

    y_diff = np.dot(gamma_a, v_delayed_a) - np.dot(gamma_b, v_delayed_b)  # take difference of new and old sample
    y_diff_flipped = y_diff[::-1]

    nu += lfilter([1, 0], [1, -gamma], y_diff_flipped.T).T[::-1]
    nu *= beta
