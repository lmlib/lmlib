import numpy as np
from numpy.linalg import inv, matrix_power

# helper function for numpy method
def _numpy_einsum_path_xi(is_multichannel):
    return 'nl..., l... ->n...' if is_multichannel else 'n..., ... ->n...'

def _numpy_einsum_path_kappa(is_multichannel, is_multiset, kappa_diag):
        if is_multiset:
            if is_multichannel:
                if kappa_diag:
                    einsum_path = 'm..., m...->...'
                else:
                    einsum_path = 'ml..., ln... ->mn'
            else:
                if kappa_diag:
                    einsum_path = 'm, m->m'
                else:
                    einsum_path = 'm, n->mn'
        else:
            if is_multichannel:
                einsum_path = 'm, m->...'
            else:
                einsum_path = '..., ...'

        return einsum_path


# xi2 recursions
def numpy_recursion_xi2(xi2, A, C, a, b, direction, delta, gamma, y, v, beta):
    W = xi2.reshape(-1, *A.shape)
    if direction == 'fw':
        numpy_forward_recursion_W(W, A, C, a, b, delta, gamma, y, v, beta)
    elif direction == 'bw':
        numpy_backward_recursion_W(W, A, C, a, b, delta, gamma, y, v, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

def numpy_forward_recursion_W(W, A, C,  a, b, delta, gamma, y, v, beta):

    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)
    A_inv = inv(A)
    Aa = matrix_power(A, 0 if np.isinf(a) else a - 1)
    AaccAa = np.outer(np.dot(Aa.T, C.T), np.dot(C, Aa))
    Ab = matrix_power(A, b)
    AbccAb = np.outer(np.dot(Ab.T, C.T), np.dot(C, Ab))

    W0 = np.zeros_like(W[0])
    K = len(y)

    for k in range(min(0, -b), K):
        W0[:] = gamma_inv * A_inv.T.dot(W0).dot(A_inv)

        if 1 - a <= k <= K - a:
            W0 -= gamma_a * v[k + a - 1] * AaccAa

        if -b <= k <= K - b - 1:
            W0 += gamma_b * v[k + b] * AbccAb

        if 0 <= k <= K - 1:
            W[k] += W0

    if beta != 1:
        W *= beta

def numpy_backward_recursion_W(W, A, C,  a, b, delta, gamma, y, v, beta):

    gamma_a = gamma**(a - delta)
    gamma_b = gamma**(b - delta + 1)
    Aa = matrix_power(A, a)
    AaccAa = np.outer(np.dot(Aa.T, C.T), np.dot(C, Aa))
    Ab = matrix_power(A, 0 if np.isinf(b) else b + 1)
    AbccAb = np.outer(np.dot(Ab.T, C.T), np.dot(C, Ab))

    W0 = np.zeros_like(W[0])
    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        W0[:] = gamma * (A.T.dot(W0).dot(A))

        if -(a - 1) + 1 <= k <= K -(a - 1):
            W0 += gamma_a * v[k + a - 1 + -1] * AaccAa

        if -b + 1 <= k <= K - b:
            W0 -= gamma_b * v[k + b + -1] * AbccAb

        if 2 <= k <= K + 1:
            W[k-2] += W0

    if beta != 1:
        W *= beta


# xi1 recursions
def numpy_recursion_xi1(xi1, A, C, a, b, direction, delta, gamma, y, v, beta):
    einsum_path = _numpy_einsum_path_xi(is_multichannel = np.ndim(C) == 2)
    if direction == 'fw':
        numpy_forward_recursion_xi(xi1, A, C, a, b, delta, gamma, y, v, beta, einsum_path)
    elif direction == 'bw':
        numpy_backward_recursion_xi(xi1, A, C, a, b, delta, gamma, y, v, beta, einsum_path)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

def numpy_forward_recursion_xi(xi, A, C,  a, b, delta, gamma, y, v, beta, einsum_path):

    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)
    A_inv = inv(A)
    Aa = matrix_power(A, 0 if np.isinf(a) else a - 1)
    Aac = np.dot(Aa.T, C.T)
    Ab = matrix_power(A, b)
    Abc = np.dot(Ab.T, C.T)

    xi0 = np.zeros_like(xi[0])
    K = len(y)
    for k in range(min(0, -b), K):
        xi0[:] = gamma_inv * A_inv.T.dot(xi0)

        if 1 - a <= k <= K - a:
            xi0 -= gamma_a * v[k + a - 1] * np.einsum(einsum_path, Aac, y[k + a - 1])

        if -b <= k <= K - b - 1:
            xi0 += gamma_b * v[k + b] * np.einsum(einsum_path, Abc, y[k + b])

        if 0 <= k <= K - 1:
            xi[k] += xi0

    if beta != 1:
        xi *= beta

def numpy_backward_recursion_xi(xi, A, C,  a, b, delta, gamma, y, v, beta, einsum_path):

    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)
    Aa = matrix_power(A, a)
    Aac = np.dot(Aa.T, C.T)
    Ab = matrix_power(A, 0 if np.isinf(b) else b + 1)
    Abc = np.dot(Ab.T, C.T)

    # intermediate vars
    xi0 = np.zeros_like(xi[0])
    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        xi0[:] = gamma * (A.T.dot(xi0))

        if -(a - 1) + 1 <= k <= K - (a - 1):
            xi0 += gamma_a * v[k + a - 1 + -1] * np.einsum(einsum_path, Aac, y[k + a - 1 + -1])

        if -b + 1 <= k <= K - b:
            xi0 -= gamma_b * v[k + b + -1] * np.einsum(einsum_path, Abc, y[k + b + -1])

        if 2 <= k <= K + 1:
            xi[k - 2] += xi0

    if beta != 1:
        xi *= beta


# xi0 recursions
def numpy_recursion_xi0(xi0, A, C, a, b, direction, delta, gamma, y, v, beta, kappa_diag=True):
    is_multichannel = np.ndim(C) == 2
    is_multiset = np.ndim(y) == 2 and not is_multichannel or np.ndim(y) > 2
    einsum_path = _numpy_einsum_path_kappa(is_multichannel, is_multiset, kappa_diag)

    if direction == 'fw':
        numpy_forward_recursion_kappa(xi0, a, b, delta, gamma, y, v, beta, einsum_path)
    elif direction == 'bw':
        numpy_backward_recursion_kappa(xi0, a, b, delta, gamma, y, v, beta, einsum_path)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

def numpy_forward_recursion_kappa(kappa, a, b, delta, gamma, y, v, beta, einsum_path):
    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)

    kappa0 = np.zeros_like(kappa[0])
    K = len(y)
    for k in range(min(0, -b), K):
        kappa0 *= gamma_inv

        if 1 - a <= k <= K - a:
            kappa0 -= gamma_a * v[k + a - 1] * np.einsum(einsum_path, y[k + a - 1], y[k + a - 1])

        if -b <= k <= K - b - 1:
            kappa0 += gamma_b * v[k + b] * np.einsum(einsum_path, y[k + b], y[k + b])

        if 0 <= k <= K - 1:
            kappa[k] += kappa0

    if beta != 1:
        kappa *= beta

def numpy_backward_recursion_kappa(kappa, a, b, delta, gamma, y, v, beta, einsum_path):

    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)

    kappa0 = np.zeros_like(kappa[0])
    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        kappa0 *= gamma

        if -(a - 1) + 1 <= k <= K - (a - 1):
            kappa0 += gamma_a * v[k + a - 1 + -1] * np.einsum(einsum_path, y[k + a - 1 + -1], y[k + a - 1 + -1])

        if -b + 1 <= k <= K - b:
            kappa0 -= gamma_b * v[k + b + -1] * np.einsum(einsum_path, y[k + b + -1], y[k + b + -1])

        if 2 <= k <= K + 1:
            kappa[k - 2] += kappa0

    if beta != 1:
        kappa *= beta


# nu recursions
def numpy_recursion_nu(nu, A, C, a, b, direction, delta, gamma, y, v, beta, kappa_diag=True):
    is_multichannel = np.ndim(C) == 2
    is_multiset = np.ndim(y) == 2 and not is_multichannel or np.ndim(y) > 2
    einsum_path = _numpy_einsum_path_kappa(is_multichannel, is_multiset, kappa_diag)

    if direction == 'fw':
        numpy_forward_recursion_nu(nu, a, b, delta, gamma, v, beta)
    elif direction == 'bw':
        numpy_backward_recursion_nu(nu, a, b, delta, gamma, v, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

def numpy_forward_recursion_nu(nu, a, b, delta, gamma, v, beta):
    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)

    nu0 = np.zeros_like(nu[0])
    K = len(v)
    for k in range(min(0, -b), K):
        nu0 *= gamma_inv

        if 1 - a <= k <= K - a:
            nu0 -= gamma_a * v[k + a - 1]

        if -b <= k <= K - b - 1:
            nu0 += gamma_b * v[k + b]

        if 0 <= k <= K - 1:
            nu[k] += nu0

    if beta != 1:
        nu *= beta

def numpy_backward_recursion_nu(nu, a, b, delta, gamma, v, beta):

    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)

    nu0 = np.zeros_like(nu[0])
    K = len(v)

    for k in range(max(K - a, K) + 1, 1, -1):
        nu0 *= gamma

        if -(a - 1) + 1 <= k <= K - (a - 1):
            nu0 += gamma_a * v[k + a - 1 + -1]

        if -b + 1 <= k <= K - b:
            nu0 -= gamma_b * v[k + b + -1]

        if 2 <= k <= K + 1:
            nu[k - 2] += nu0

    nu *= beta
    if beta != 1:
        nu *= beta


