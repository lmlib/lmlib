"""
Backend for Recursive Least Squares Calculation using Numpy
===========================================================

Resources
---------
Authors: Reto A. Wildhaber , Nour Zalmai , Marcel Jacomet , and Hans-Andrea Loeliger
Windowed State-Space Filters for Signal Detection and Separation
DOI: 10.1109/TSP.2018.2833804

"""

import numpy as np
from numpy.linalg import inv, matrix_power


# xi2 recursions
def numpy_recursion_xi2(xi2, A, C, a, b, direction, delta, gamma, y, v, beta):
    Ks = np.shape(xi2)[:-1]
    W = xi2.reshape(*Ks, *A.shape)
    if direction == 'fw':
        numpy_forward_recursion_W(W, A, C, a, b, delta, gamma, y, v, beta)
    elif direction == 'bw':
        numpy_backward_recursion_W(W, A, C, a, b, delta, gamma, y, v, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

def numpy_forward_recursion_W(W, A, C,  a, b, delta, gamma, y, v, beta):
    """Equation  (27)  in Wildhaber 2018"""

    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)
    A_inv = inv(A)
    Aa = matrix_power(A, 0 if np.isinf(a) else a - 1)
    AaccAa = np.dot( np.dot(Aa.T, C.T), np.dot(C, Aa))
    Ab = matrix_power(A, b)
    AbccAb = np.dot(np.dot(Ab.T, C.T), np.dot(C, Ab))

    W0 = np.zeros_like(W[0])
    K = len(y)

    for k in range(min(0, -b), K):
        W0[:] = gamma_inv * np.einsum('...ij, jk->...ik', np.einsum('ij, ...jk->...ik', A_inv.T, W0), A_inv)

        if 1 - a <= k <= K - a:
            W0 -= gamma_a * np.einsum('..., mn->...mn', v[k+a-1], AaccAa)

        if -b <= k <= K - b - 1:
            W0 += gamma_b * np.einsum('..., mn->...mn', v[k+b], AbccAb)

        if 0 <= k <= K - 1:
            W[k] += W0

    if beta != 1:
        W *= beta

def numpy_backward_recursion_W(W, A, C,  a, b, delta, gamma, y, v, beta):
    """Equation  (31)  in Wildhaber 2018"""

    gamma_a = gamma**(a - delta)
    gamma_b = gamma**(b - delta + 1)
    Aa = matrix_power(A, a)
    AaccAa = np.dot(np.dot(Aa.T, C.T), np.dot(C, Aa))
    Ab = matrix_power(A, 0 if np.isinf(b) else b + 1)
    AbccAb = np.dot(np.dot(Ab.T, C.T), np.dot(C, Ab))

    W0 = np.zeros_like(W[0])
    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        W0[:] = gamma * np.einsum('...ij, jk->...ik', np.einsum('ij, ...jk->...ik', A.T, W0), A)

        if -(a - 1) + 1 <= k <= K -(a - 1):
            W0 += gamma_a * np.einsum('..., mn->...mn', v[k+a-1 +-1], AaccAa)

        if -b + 1 <= k <= K - b:
            W0 -= gamma_b * np.einsum('..., mn->...mn', v[k+b+ -1], AbccAb)

        if 2 <= k <= K + 1:
            W[k-2] += W0

    if beta != 1:
        W *= beta


# xi1 recursions
def numpy_recursion_xi1(xi1, A, C, a, b, direction, delta, gamma, y, v, beta):
    if direction == 'fw':
        numpy_forward_recursion_xi(xi1, A, C, a, b, delta, gamma, y, v, beta)
    elif direction == 'bw':
        numpy_backward_recursion_xi(xi1, A, C, a, b, delta, gamma, y, v, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

def numpy_forward_recursion_xi(xi, A, C,  a, b, delta, gamma, y, v, beta):
    """Equation (28)  in Wildhaber 2018"""
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
        # xi0[:] = gamma_inv * A_inv.T.dot(xi0)
        xi0[:] = gamma_inv * np.einsum( 'mn, ...n ->...m', A_inv.T, xi0)

        if 1 - a <= k <= K - a:
            xi0 -= gamma_a * np.einsum('..., ...n->...n',
                                       v[k + a - 1],
                                       np.einsum( 'nl, ...l ->...n', Aac, y[k + a - 1])
                                       )

        if -b <= k <= K - b - 1:
            xi0 += gamma_b * np.einsum('..., ...n->...n',
                                       v[k + b],
                                       np.einsum( 'nl, ...l ->...n', Abc, y[k + b])
                                       )

        if 0 <= k <= K - 1:
            xi[k] += xi0

    if beta != 1:
        xi *= beta

def numpy_backward_recursion_xi(xi, A, C,  a, b, delta, gamma, y, v, beta):
    """Equation  (32)  in Wildhaber 2018"""

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
        xi0[:] = gamma * np.einsum( 'mn, ...n ->...m', A.T, xi0)

        if -(a - 1) + 1 <= k <= K - (a - 1):
            xi0 += gamma_a * np.einsum('..., ...n->...n',
                                       v[k + a - 1 + -1],
                                       np.einsum( 'nl, ...l ->...n', Aac, y[k + a - 1 + -1])
                                       )

        if -b + 1 <= k <= K - b:
            xi0 -= gamma_b *  np.einsum('..., ...n->...n',
                                        v[k + b + -1],
                                        np.einsum( 'nl, ...l ->...n', Abc, y[k + b + -1])
                                        )

        if 2 <= k <= K + 1:
            xi[k - 2] += xi0

    if beta != 1:
        xi *= beta


# xi0 recursions
def numpy_recursion_xi0(xi0, A, C, a, b, direction, delta, gamma, y, v, beta):
    # y**2
    if direction == 'fw':
        numpy_forward_recursion_kappa(xi0, a, b, delta, gamma, y, v, beta)
    elif direction == 'bw':
        numpy_backward_recursion_kappa(xi0, a, b, delta, gamma, y, v, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

def numpy_forward_recursion_kappa(kappa, a, b, delta, gamma, y, v, beta):
    """Equation  (29)  in Wildhaber 2018"""

    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)

    kappa0 = np.zeros_like(kappa[0])
    K = len(y)
    for k in range(min(0, -b), K):
        kappa0 *= gamma_inv

        if 1 - a <= k <= K - a:
            kappa0 -= gamma_a * np.einsum('..., ...n->...n',
                                          v[k + a - 1],
                                          np.einsum('...m, ...m', y[k + a - 1], y[k + a - 1])[..., np.newaxis]
                                          )

        if -b <= k <= K - b - 1:
            kappa0 += gamma_b *  np.einsum('..., ...n->...n',
                                           v[k + b],
                                           np.einsum('...m, ...m', y[k + b], y[k + b])[..., np.newaxis]
                                           )

        if 0 <= k <= K - 1:
            kappa[k] += kappa0

    if beta != 1:
        kappa *= beta

def numpy_backward_recursion_kappa(kappa, a, b, delta, gamma, y, v, beta):
    """Equation  (33)  in Wildhaber 2018"""

    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)

    kappa0 = np.zeros_like(kappa[0])
    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        kappa0 *= gamma

        if -(a - 1) + 1 <= k <= K - (a - 1):
            kappa0 += gamma_a * np.einsum('..., ...n->...n',
                                          v[k + a - 1 + -1],
                                          np.einsum('...m, ...m', y[k + a - 1 + -1], y[k + a - 1 + -1])[..., np.newaxis]
                                          )

        if -b + 1 <= k <= K - b:
            kappa0 -= gamma_b * np.einsum('..., ...n->...n',
                                          v[k + b + -1],
                                          np.einsum('...m, ...m', y[k + b + -1], y[k + b + -1])[..., np.newaxis]
                                          )

        if 2 <= k <= K + 1:
            kappa[k - 2] += kappa0

    if beta != 1:
        kappa *= beta


# nu recursions
def numpy_recursion_nu(nu, A, C, a, b, direction, delta, gamma, y, v, beta):
    if direction == 'fw':
        numpy_forward_recursion_nu(nu, a, b, delta, gamma, v, beta)
    elif direction == 'bw':
        numpy_backward_recursion_nu(nu, a, b, delta, gamma, v, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

def numpy_forward_recursion_nu(nu, a, b, delta, gamma, v, beta):
    """Equation  (30)  in Wildhaber 2018"""

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
    """Equation  (34)  in Wildhaber 2018"""

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


# xi asterisk l recursions
def numpy_xi_asterisk_l_recursion(xi, A, C, a, b, direction, delta, gamma, I_N, xi_N, v, beta):
    if direction == 'fw':
        numpy_xi_asterisk_l_forward_recursion(xi, A, C, a, b, delta, gamma, I_N, xi_N, v, beta)
    elif direction == 'bw':
        numpy_xi_asterisk_l_backward_recursion(xi, A, C, a, b, delta, gamma, I_N, xi_N, v, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

def numpy_xi_asterisk_l_forward_recursion(xi, A, C, a, b, delta, gamma, I_N, xi_N, v, beta):
    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)

    A_inv = inv(A)
    gIkAinvT = gamma_inv * np.kron(I_N, A_inv).T

    Aa = matrix_power(A, 0 if np.isinf(a) else a - 1)
    gIkcAaT = gamma_a * np.kron(I_N, (C@Aa).T)

    Ab = matrix_power(A, b)
    gIkcAbT = gamma_b * np.kron(I_N, (C@Ab).T)

    _xi = np.zeros_like(xi[0])
    K = len(xi_N)

    for k in range(min(0, -b), K):
        _xi[:] = np.einsum('mn,...n->...m', gIkAinvT, _xi)

        if 1 - a <= k <= K - a:
            _xi -= np.einsum('mn,...n->...m', gIkcAaT, xi_N[k + a - 1])

        if -b <= k <= K - b - 1:
            _xi += np.einsum('mn,...n->...m', gIkcAbT, xi_N[k + b])

        if 0 <= k <= K - 1:
            xi[k] += _xi

    if beta != 1:
        xi *= beta

def numpy_xi_asterisk_l_backward_recursion(xi, A, C,  a, b, delta, gamma, I_N, xi_N, v, beta):
    gamma_a = gamma**(a - delta)
    gamma_b = gamma**(b - delta + 1)

    gIkAT = gamma*np.kron(I_N, A).T

    Aa = matrix_power(A, a)
    gIkcAaT = gamma_a * np.kron(I_N, (C @Aa).T)

    Ab = matrix_power(A, 0 if np.isinf(b) else b + 1)
    gIkcAbT = gamma_b * np.kron(I_N, (C@Ab).T)

    _xi = np.zeros_like(xi[0])
    K = len(xi_N)

    for k in range(max(K - a, K) + 1, 1, -1):
        _xi[:] = np.einsum('mn,...n->...m', gIkAT , _xi)

        if -(a - 1) + 1 <= k <= K -(a - 1):
            _xi += np.einsum('mn,...n->...m', gIkcAaT , xi_N[k + a - 1 + -1])

        if -b + 1 <= k <= K - b:
            _xi -= np.einsum('mn,...n->...m', gIkcAbT , xi_N[k + b + -1])

        if 2 <= k <= K + 1:
            xi[k-2] += _xi

    if beta != 1:
        xi *= beta
