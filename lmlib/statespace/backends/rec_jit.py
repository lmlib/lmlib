"""
Backend for Recursive Least Squares Calculation using Just in Time Compiler from Numba
======================================================================================

Resources
---------
Authors: Reto A. Wildhaber , Nour Zalmai , Marcel Jacomet , and Hans-Andrea Loeliger
Windowed State-Space Filters for Signal Detection and Separation
DOI: 10.1109/TSP.2018.2833804

"""


from numba import jit
import numpy as np
from numpy.linalg import inv, matrix_power


# xi2 recursions
def jit_recursion_xi2(xi2, A, C, a, b, direction, delta, gamma, y, v, beta):
    _A = A.astype(float)
    _C = C.astype(float)
    W = xi2.reshape(-1, *A.shape)

    if direction == 'fw':
        gamma_a = gamma ** (a - 1 - delta)
        _a = -2**31 if np.isinf(a) else a
        Aa = matrix_power(_A, 0 if np.isnan(a) else _a - 1)
        Ab = matrix_power(A, b)
        AaccAa = np.dot(np.dot(Aa.T, C.T), np.dot(C, Aa))
        AbccAb = np.dot(np.dot(Ab.T, C.T), np.dot(C, Ab))

        jit_forward_recursion_W(W, _A, _C, _a, b, delta, gamma, y, v, beta, AaccAa, AbccAb, gamma_a)
    elif direction == 'bw':
        Ab = matrix_power(_A, 0 if np.isinf(b) else b + 1)
        gamma_b = gamma ** (b - delta + 1)
        _b = 2**31 if np.isinf(b) else b
        Aa = matrix_power(A, a)
        AaccAa = np.dot(np.dot(Aa.T, C.T), np.dot(C, Aa))
        AbccAb = np.dot(np.dot(Ab.T, C.T), np.dot(C, Ab))

        jit_backward_recursion_W(W, _A, _C, a, _b, delta, gamma, y, v, beta,  AaccAa, AbccAb, gamma_b)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

@jit(nopython=True)
def jit_forward_recursion_W(W, A, C,  a, b, delta, gamma, y, v, beta, AaccAa, AbccAb, gamma_a):
    """Equation  (27)  in Wildhaber 2018"""

    gamma_inv = 1/gamma
    gamma_b = gamma**(b-delta)

    A_inv = inv(A)
    W0 = np.zeros_like(W[0])
    K = len(y)

    for k in range(min(0, -b), K):
        W0[:] = gamma_inv * A_inv.T.dot(W0).dot(A_inv)

        if 1-a <= k <= K - a:
            W0 -= gamma_a * v[k + a - 1] * AaccAa

        if -b <= k <= K - b - 1:
            W0 += gamma_b * v[k + b] * AbccAb

        if 0 <= k <= K - 1:
            W[k] += W0

    if beta != 1:
        W *= beta

@jit(nopython=True)
def jit_backward_recursion_W(W, A, C,  a, b, delta, gamma, y, v, beta, AaccAa, AbccAb, gamma_b):
    """Equation  (31)  in Wildhaber 2018"""

    gamma_a = gamma**(a - delta)


    W0 = np.zeros_like(W[0])
    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        W0[:] = gamma * A.T.dot(W0).dot(A)

        if -(a - 1) + 1 <= k <= K -(a - 1):
            W0 += gamma_a * v[k + a - 1 + -1] * AaccAa

        if -b + 1 <= k <= K - b:
            W0 -= gamma_b * v[k + b + -1] * AbccAb

        if 2 <= k <= K + 1:
            W[k-2] += W0

    if beta != 1:
        W *= beta

# xi1 recursions
def jit_recursion_xi1(xi1, A, C, a, b, direction, delta, gamma, y, v, beta):
    _A = A.astype(float)
    _C = C.astype(float)
    if direction == 'fw':
        gamma_a = gamma ** (a - 1 - delta)
        _a = -2**31 if np.isinf(a) else a
        Aa = matrix_power(_A, 0 if np.isinf(a) else _a - 1)
        jit_forward_recursion_xi(xi1, _A, _C, _a, b, delta, gamma, y, v, beta, Aa, gamma_a)
    elif direction == 'bw':
        gamma_b = gamma ** (b - delta + 1)
        _b = 2**31 if np.isinf(b) else b
        Ab = matrix_power(_A, 0 if np.isinf(b) else _b + 1)
        jit_backward_recursion_xi(xi1, _A, _C, a, _b, delta, gamma, y, v, beta, Ab, gamma_b)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

# @jit(nopython=True)
def jit_forward_recursion_xi(xi, A, C,  a, b, delta, gamma, y, v, beta, Aa, gamma_a) :
    """Equation  (28)  in Wildhaber 2018"""

    gamma_inv = 1 / gamma
    gamma_b = gamma ** (b - delta)
    A_inv = inv(A)
    Ab = matrix_power(A, b)
    Aac = np.dot(Aa.T, C.T)
    Abc = np.dot(Ab.T, C.T)

    xi0 = np.zeros_like(xi[0])
    K = len(y)
    for k in range(min(0, -b), K):
        xi0[:] = gamma_inv * A_inv.T.dot(xi0)

        if 1 - a <= k <= K - a:
            xi0 -= gamma_a * v[k + a - 1] * np.dot(Aac, y[k + a - 1])

        if -b <= k <= K - b - 1:
            xi0 += gamma_b * v[k + b] * np.dot(Abc, y[k + b])

        if 0 <= k <= K - 1:
            xi[k] += xi0

    if beta != 1:
        xi *= beta

@jit(nopython=True)
def jit_backward_recursion_xi(xi, A, C,  a, b, delta, gamma, y, v, beta, Ab, gamma_b):
    """Equation  (32)  in Wildhaber 2018"""

    gamma_a = gamma ** (a - delta)
    Aa = matrix_power(A, a)
    Aac = np.dot(Aa.T, C.T)
    Abc = np.dot(Ab.T, C.T)

    # intermediate vars
    xi0 = np.zeros_like(xi[0])
    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        xi0[:] = gamma * (A.T.dot(xi0))

        if -(a - 1) + 1 <= k <= K - (a - 1):
            xi0 += gamma_a * v[k + a - 1 + -1] * np.dot(Aac, y[k + a - 1 + -1])

        if -b + 1 <= k <= K - b:
            xi0 -= gamma_b * v[k + b + -1] * np.dot(Abc, y[k + b + -1])

        if 2 <= k <= K + 1:
            xi[k - 2] += xi0

    if beta != 1:
        xi *= beta


# xi0 recursions
def jit_recursion_xi0(xi0, A, C, a, b, direction, delta, gamma, y, v, beta):
    if direction == 'fw':
        gamma_a = gamma ** (a - 1 - delta)
        _a = -2 ** 31 if np.isinf(a) else a
        jit_forward_recursion_kappa(xi0, _a, b, delta, gamma, y, v, beta, gamma_a)
    elif direction == 'bw':
        gamma_b = gamma ** (b - delta + 1)
        _b = 2 ** 31 if np.isinf(b) else b
        jit_backward_recursion_kappa(xi0, a, _b, delta, gamma, y, v, beta, gamma_b)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

@jit(nopython=True)
def jit_forward_recursion_kappa(kappa, a, b, delta, gamma, y, v, beta, gamma_a):
    """Equation  (29)  in Wildhaber 2018"""

    gamma_inv = 1 / gamma
    gamma_b = gamma ** (b - delta)

    kappa0 = 0.0
    K = len(y)

    for k in range(min(0, -b), K):
        kappa0 *= gamma_inv

        if 1 - a <= k <= K - a:
            kappa0 -= gamma_a * v[k + a - 1] * np.dot(y[k + a - 1], y[k + a - 1])

        if -b <= k <= K - b - 1:
            kappa0 += gamma_b * v[k + b] * np.dot(y[k + b], y[k + b])

        if 0 <= k <= K - 1:
            kappa[k] += kappa0

    if beta != 1:
        kappa *= beta

@jit(nopython=True)
def jit_backward_recursion_kappa(kappa, a, b, delta, gamma, y, v, beta, gamma_b):
    """Equation  (33)  in Wildhaber 2018"""

    gamma_a = gamma ** (a - delta)

    kappa0 = 0.0
    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        kappa0 *= gamma

        if -(a - 1) + 1 <= k <= K - (a - 1):
            kappa0 += gamma_a * v[k + a - 1 + -1] * np.dot(y[k + a - 1 + -1], y[k + a - 1 + -1])

        if -b + 1 <= k <= K - b:
            kappa0 -= gamma_b * v[k + b + -1] * np.dot(y[k + b + -1], y[k + b + -1])

        if 2 <= k <= K + 1:
            kappa[k - 2] += kappa0

    if beta != 1:
        kappa *= beta


# nu recursions
def jit_recursion_nu(nu, A, C, a, b, direction, delta, gamma, y, v, beta):
    if direction == 'fw':
        gamma_a = gamma ** (a - 1 - delta)
        _a = -2 ** 31 if np.isinf(a) else a
        jit_forward_recursion_nu(nu, _a, b, delta, gamma, v, beta, gamma_a)
    elif direction == 'bw':
        gamma_b = gamma ** (b - delta + 1)
        _b = 2 ** 31 if np.isinf(b) else b
        jit_backward_recursion_nu(nu, a, _b, delta, gamma, v, beta, gamma_b)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

@jit(nopython=True)
def jit_forward_recursion_nu(nu, a, b, delta, gamma, v, beta, gamma_a):
    """Equation  (30)  in Wildhaber 2018"""

    gamma_inv = 1 / gamma
    gamma_b = gamma ** (b - delta)

    nu0 = 0.0
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

@jit(nopython=True)
def jit_backward_recursion_nu(nu, a, b, delta, gamma, v, beta, gamma_b):
    """Equation  (34)  in Wildhaber 2018"""

    gamma_a = gamma ** (a - delta)

    nu0 = 0.0
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


