from numba import jit
import numpy as np
from numpy.linalg import inv, matrix_power
__all__ = ['forward_recursion_jit', 'backward_recursion_jit',
           'forward_recursion_xi_kappa_nu_jit','backward_recursion_xi_kappa_nu_jit',
           'forward_recursion_set_jit', 'backward_recursion_set_jit']


@jit(nopython=True)
def forward_recursion_W_jit(W, a, b, delta, gamma, A, C, beta, y, v):
    gamma_inv = 1/gamma
    gamma_a = gamma**(a-1-delta)
    gamma_b = gamma**(b-delta)
    A_inv = inv(A)
    Aa = matrix_power(A, 0 if np.isnan(a) else a-1)
    AaccAa = np.outer(np.dot(Aa.T, C.T), np.dot(C, Aa))
    Ab = matrix_power(A, b)
    AbccAb = np.outer(np.dot(Ab.T, C.T), np.dot(C, Ab))

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

    W *= beta

@jit(nopython=True)
def forward_recursion_xi_jit(xi, a, b, delta, gamma, A, C, beta, y, v, einsum_path):
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

    xi *= beta

@jit(nopython=True)
def forward_recursion_kappa_ss(kappa, a, b, delta, gamma, beta, y, v, einsum_path):
    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)

    kappa0 = np.zeros_like(kappa[0])
    K = len(y)
    for k in range(min(0, -b), K):
        kappa0 *= gamma_inv

        if 1-a <= k <= K - a:
            kappa0 -= gamma_a * v[k + a - 1] * np.einsum(einsum_path, y[k + a - 1].T, y[k + a - 1])

        if -b <= k <= K - b - 1:
            kappa0 += gamma_b * v[k + b] * np.einsum(einsum_path, y[k + b].T, y[k + b])

        if 0 <= k <= K - 1:
            kappa[k] += kappa0

    kappa *= beta


def forward_recursion_nu_ss(nu, a, b, delta, gamma, beta, v):
    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)

    nu0 = np.zeros_like(nu[0])
    K = len(v)
    for k in range(min(0, -b), K):
        nu0 *= gamma_inv

        if 1-a <= k <= K - a:
            nu0 -= gamma_a * v[k + a - 1]

        if -b <= k <= K - b - 1:
            nu0 += gamma_b * v[k + b]

        if 0 <= k <= K - 1:
            nu[k] += nu0

    nu *= beta


def backward_recursion_W_ss(W, a, b, delta, gamma, A, C, beta, y, v):
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

    W *= beta


def backward_recursion_xi_ss(xi, a, b, delta, gamma, A, C, beta, y, v, einsum_path):
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

    xi *= beta


def backward_recursion_kappa_ss(kappa, a, b, delta, gamma, beta, y, v, einsum_path):
    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)

    kappa0 = np.zeros_like(kappa[0])
    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        kappa0 *= gamma

        if -(a - 1) + 1 <= k <= K - (a - 1):
            kappa0 += gamma_a * v[k + a - 1 + -1] * np.einsum(einsum_path, y[k + a - 1 + -1].T, y[k + a - 1 + -1])

        if -b + 1 <= k <= K - b:
            kappa0 -= gamma_b * v[k + b + -1] * np.einsum(einsum_path, y[k + b + -1].T, y[k + b + -1])

        if 2 <= k <= K + 1:
            kappa[k - 2] += kappa0

    kappa *= beta


def backward_recursion_nu_ss(nu, a, b, delta, gamma, beta, v):
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

def forward_recursion_jit(*args):
    if np.ndim(args[13]) == 1:  # args[13] : Aac
        forward_recursion_jit_1dim(*args)
    else:
        forward_recursion_jit_2dim(*args)

def forward_recursion_xi_kappa_nu_jit(*args):
    if np.ndim(args[12]) == 1:  # args[12] : Aac
        forward_recursion_xi_kappa_nu_jit_1dim(*args)
    else:
        forward_recursion_xi_kappa_nu_jit_2dim(*args)

def backward_recursion_jit(*args):
    if np.ndim(args[16]) == 1:  # args[13] : Abc
        backward_recursion_jit_1dim(*args)
    else:
        backward_recursion_jit_2dim(*args)


def backward_recursion_xi_kappa_nu_jit(*args):
    if np.ndim(args[15]) == 1:  # args[15] : Abc
        backward_recursion_xi_kappa_nu_jit_1dim(*args)
    else:
        backward_recursion_xi_kappa_nu_jit_2dim(*args)


def forward_recursion_set_jit(*args):
    if np.ndim(args[13]) == 1:  # args[13] : Aac
        if args[-1]: # kappa_diag
            forward_recursion_set_jit_1dim_kappa_diag(*args)
        else:
            forward_recursion_set_jit_1dim_not_kappa_diag(*args)
    else:
        raise NotImplementedError('JIT backend with 2 dimensional Alssm Output and Multi-Set is not implemented yet. Please use python backend.')
        if args[-1]:  # kappa_diag
            forward_recursion_set_jit_2dim_kappa_diag(*args)
        else:
            forward_recursion_set_jit_2dim_not_kappa_diag(*args)


def backward_recursion_set_jit(*args):
    if np.ndim(args[16]) == 1:  # args[13] : Abc
        if args[-1]: # kappa_diag
            backward_recursion_set_jit_1dim_kappa_diag(*args)
        else:
            backward_recursion_set_jit_1dim_not_kappa_diag(*args)
    else:
        raise NotImplementedError('JIT backend with 2 dimensional Alssm Output and Multi-Set is not implemented yet. Please use python backend.')
        if args[-1]:  # kappa_diag
            backward_recursion_set_jit_2dim_kappa_diag(*args)
        else:
            backward_recursion_set_jit_2dim_not_kappa_diag(*args)


@jit(nopython=True)
def forward_recursion_jit_1dim(W, xi, kappa, nu, a, b, delta, y, v, beta, gamma_inv, A_inv, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb):
    W0 = np.zeros_like(W[0])
    xi0 = np.zeros_like(xi[0])
    kappa0 = 0.0
    nu0 = 0.0

    K = len(y)
    for k in range(min(0, -b), K):
        W0[:] = gamma_inv * (A_inv.T.dot(W0).dot(A_inv))
        xi0[:] = gamma_inv * (A_inv.T.dot(xi0))
        kappa0 *= gamma_inv
        nu0 *= gamma_inv

        if 1-a <= k <= K - a:
            gav = gamma_a * v[k + a - 1]
            W0 -= gav * AaccAa
            xi0 -= gav * Aac * y[k + a - 1]
            kappa0 -= gav * y[k + a - 1] * y[k + a - 1]
            nu0 -= gav

        if -b <= k <= K - b - 1:
            gbv = gamma_b * v[k + b]
            W0 += gbv * AbccAb
            xi0 += gbv * Abc * y[k + b]
            kappa0 += gbv * y[k + b]*y[k + b]
            nu0 += gbv

        if 0 <= k <= K - 1:
            W[k] += W0
            xi[k] += xi0
            kappa[k] += kappa0
            nu[k] += nu0

    W *= beta
    xi *= beta
    kappa *= beta
    nu *= beta

@jit(nopython=True)
def forward_recursion_xi_kappa_nu_jit_1dim(xi, kappa, nu, a, b, delta, y, v, beta, gamma_inv, A_inv, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb):
    xi0 = np.zeros_like(xi[0])
    kappa0 = 0.0
    nu0 = 0.0

    K = len(y)
    for k in range(min(0, -b), K):
        xi0[:] = gamma_inv * (A_inv.T.dot(xi0))
        kappa0 *= gamma_inv
        nu0 *= gamma_inv

        if  1-a <= k <= K - a:
            gav = gamma_a * v[k + a - 1]
            xi0 -= gav * Aac * y[k + a - 1]
            kappa0 -= gav * y[k + a - 1] * y[k + a - 1]
            nu0 -= gav

        if -b <= k <= K - b - 1:
            gbv = gamma_b * v[k + b]
            xi0 += gbv * Abc * y[k + b]
            kappa0 += gbv * y[k + b]*y[k + b]
            nu0 += gbv

        if 0 <= k <= K - 1:
            xi[k] += xi0
            kappa[k] += kappa0
            nu[k] += nu0

    xi *= beta
    kappa *= beta
    nu *= beta


@jit(nopython=True)
def forward_recursion_jit_2dim(W, xi, kappa, nu, a, b, delta, y, v, beta, gamma_inv, A_inv, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb):
    W0 = np.zeros_like(W[0])
    xi0 = np.zeros_like(xi[0])
    kappa0 = 0.0
    nu0 = 0.0

    K = len(y)
    for k in range(min(0, -b), K):
        W0[:] = gamma_inv * (A_inv.T.dot(W0).dot(A_inv))
        xi0[:] = gamma_inv * (A_inv.T.dot(xi0))
        kappa0 *= gamma_inv
        nu0 *= gamma_inv

        if 1-a <= k <= K - a:
            gav = gamma_a * v[k + a - 1]
            W0 -= gav * AaccAa
            xi0 -= gav * Aac.dot(y[k + a - 1])
            kappa0 -= gav * y[k + a - 1].dot(y[k + a - 1])
            nu0 -= gav

        if -b <= k <= K - b - 1:
            gbv = gamma_b * v[k + b]
            W0 += gbv * AbccAb
            xi0 += gbv * Abc.dot(y[k + b])
            kappa0 += gbv * y[k + b].dot(y[k + b])
            nu0 += gbv

        if 0 <= k <= K - 1:
            W[k] += W0
            xi[k] += xi0
            kappa[k] += kappa0
            nu[k] += nu0

    W *= beta
    xi *= beta
    kappa *= beta
    nu *= beta

@jit(nopython=True)
def forward_recursion_xi_kappa_nu_jit_2dim(xi, kappa, nu, a, b, delta, y, v, beta, gamma_inv, A_inv, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb):
    xi0 = np.zeros_like(xi[0])
    kappa0 = 0.0
    nu0 = 0.0

    K = len(y)
    for k in range(min(0, -b), K):
        xi0[:] = gamma_inv * (A_inv.T.dot(xi0))
        kappa0 *= gamma_inv
        nu0 *= gamma_inv

        if 1-a <= k <= K - a:
            gav = gamma_a * v[k + a - 1]
            xi0 -= gav * Aac.dot(y[k + a - 1])
            kappa0 -= gav * y[k + a - 1].dot(y[k + a - 1])
            nu0 -= gav

        if -b <= k <= K - b - 1:
            gbv = gamma_b * v[k + b]
            xi0 += gbv * Abc.dot(y[k + b])
            kappa0 += gbv * y[k + b].dot(y[k + b])
            nu0 += gbv

        if 0 <= k <= K - 1:
            xi[k] += xi0
            kappa[k] += kappa0
            nu[k] += nu0

    xi *= beta
    kappa *= beta
    nu *= beta

@jit(nopython=True)
def backward_recursion_jit_1dim(W, xi, kappa, nu, a, b, delta, y, v, beta, gamma, A, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb):
    W0 = np.zeros_like(W[0])
    xi0 = np.zeros_like(xi[0])
    kappa0 = 0.0
    nu0 = 0.0

    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        W0[:] = gamma * (A.T.dot(W0).dot(A))
        xi0[:] = gamma * (A.T.dot(xi0))
        kappa0 *= gamma
        nu0 *= gamma

        if -(a - 1) + 1 <= k <= K -(a - 1):
            gav = gamma_a * v[k + a - 1 + -1]
            W0 += gav * AaccAa
            xi0 += gav * Aac * y[k + a - 1 + -1]
            kappa0 += gav * y[k + a - 1 + -1] * y[k + a - 1 + -1]
            nu0 += gav

        if -b + 1 <= k <= K - b:
            gbv = gamma_b * v[k + b + -1]
            W0 -= gbv * AbccAb
            xi0 -= gbv * Abc * y[k + b + -1]
            kappa0 -= gbv * y[k + b + -1] * y[k + b + -1]
            nu0 -= gbv

        if 2 <= k <= K + 1:
            W[k-2] += W0
            xi[k-2] += xi0
            kappa[k-2] += kappa0
            nu[k-2] += nu0

    W *= beta
    xi *= beta
    kappa *= beta
    nu *= beta

@jit(nopython=True)
def backward_recursion_xi_kappa_nu_jit_1dim(xi, kappa, nu, a, b, delta, y, v, beta, gamma, A, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb):
    xi0 = np.zeros_like(xi[0])
    kappa0 = 0.0
    nu0 = 0.0

    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        xi0[:] = gamma * (A.T.dot(xi0))
        kappa0 *= gamma
        nu0 *= gamma

        if -(a - 1) + 1 <= k <= K -(a - 1):
            gav = gamma_a * v[k + a - 1 + -1]
            xi0 += gav * Aac * y[k + a - 1 + -1]
            kappa0 += gav * y[k + a - 1 + -1] * y[k + a - 1 + -1]
            nu0 += gav

        if -b + 1 <= k <= K - b:
            gbv = gamma_b * v[k + b + -1]
            xi0 -= gbv * Abc * y[k + b + -1]
            kappa0 -= gbv * y[k + b + -1] * y[k + b + -1]
            nu0 -= gbv

        if 2 <= k <= K + 1:
            xi[k-2] += xi0
            kappa[k-2] += kappa0
            nu[k-2] += nu0

    xi *= beta
    kappa *= beta
    nu *= beta

@jit(nopython=True)
def backward_recursion_jit_2dim(W, xi, kappa, nu, a, b, delta, y, v, beta, gamma, A, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb):
    W0 = np.zeros_like(W[0])
    xi0 = np.zeros_like(xi[0])
    kappa0 = np.zeros_like(kappa[0])
    nu0 = np.zeros_like(nu[0])

    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        W0[:] = gamma * (A.T.dot(W0).dot(A))
        xi0[:] = gamma * (A.T.dot(xi0))
        kappa0 *= gamma
        nu0 *= gamma

        if -(a - 1) + 1 <= k <= K -(a - 1):
            gav = gamma_a * v[k + a - 1 + -1]
            W0 += gav * AaccAa
            xi0 += gav * np.dot(Aac, y[k + a - 1 + -1])
            kappa0 += gav * y[k + a - 1 + -1].dot(y[k + a - 1 + -1])
            nu0 += gav

        if -b + 1 <= k <= K - b:
            gbv = gamma_b * v[k + b + -1]
            W -= gbv * AbccAb
            xi -= gbv * np.dot(Abc, y[k + b + -1])
            kappa -= gbv * y[k + b + -1].dot(y[k + b + -1])
            nu -= gbv

        if 0 <= k <= K - 1:
            W[k] += W0
            xi[k] += xi0
            kappa[k] += kappa0
            nu[k] += nu0

    W *= beta
    xi *= beta
    kappa *= beta
    nu *= beta

@jit(nopython=True)
def backward_recursion_xi_kappa_nu_jit_2dim(xi, kappa, nu, a, b, delta, y, v, beta, gamma, A, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb):
    xi0 = np.zeros_like(xi[0])
    kappa0 = np.zeros_like(kappa[0])
    nu0 = np.zeros_like(nu[0])

    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        xi0[:] = gamma * (A.T.dot(xi0))
        kappa0 *= gamma
        nu0 *= gamma

        if -(a - 1) + 1 <= k <= K -(a - 1):
            gav = gamma_a * v[k + a - 1 + -1]
            xi0 += gav * np.dot(Aac, y[k + a - 1 + -1])
            kappa0 += gav * y[k + a - 1 + -1].dot(y[k + a - 1 + -1])
            nu0 += gav

        if -b + 1 <= k <= K - b:
            gbv = gamma_b * v[k + b + -1]
            xi -= gbv * np.dot(Abc, y[k + b + -1])
            kappa -= gbv * y[k + b + -1].dot(y[k + b + -1])
            nu -= gbv

        if 0 <= k <= K - 1:
            xi[k] += xi0
            kappa[k] += kappa0
            nu[k] += nu0

    xi *= beta
    kappa *= beta
    nu *= beta

@jit(nopython=True)
def forward_recursion_set_jit_1dim_kappa_diag(W, xi, kappa, nu, a, b, delta, y, v, beta, gamma_inv, A_inv, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb, kappa_diag=True):
    W0 = np.zeros_like(W[0])
    xi0 = np.zeros_like(xi[0])
    kappa0 = np.zeros_like(kappa[0])
    nu0 = 0.0

    K = len(y)
    for k in range(min(0, -b), K):
        W0[:] = gamma_inv * (A_inv.T.dot(W0).dot(A_inv))
        xi0[:] = gamma_inv * (A_inv.T.dot(xi0))
        kappa0 *= gamma_inv
        nu0 *= gamma_inv

        if 1 - a <= k <= K - a:
            gav = gamma_a * v[k + a - 1]
            W0 -= gav * AaccAa
            xi0 -= gav * np.outer(Aac, y[k + a - 1])
            kappa0 -= gav * np.diag(np.outer(y[k + a - 1], y[k + a - 1]))

        if -b <= k <= K - b - 1:
            gbv = gamma_b * v[k + b]
            W0 += gbv * AbccAb
            xi0 += gbv * np.outer(Abc, y[k + b])
            kappa0 += gbv * np.diag(np.outer(y[k + b], y[k + b]))
            nu0 += gbv

        if 0 <= k <= K - 1:
            W[k] += W0
            xi[k] += xi0
            kappa[k] += kappa0
            nu[k] += nu0

    W *= beta
    xi *= beta
    kappa *= beta
    nu *= beta

@jit(nopython=True)
def forward_recursion_set_jit_1dim_not_kappa_diag(W, xi, kappa, nu, a, b, delta, y, v, beta, gamma_inv, A_inv, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb, kappa_diag=True):
    W0 = np.zeros_like(W[0])
    xi0 = np.zeros_like(xi[0])
    kappa0 = np.zeros_like(kappa[0])
    nu0 = 0.0

    K = len(y)
    for k in range(min(0, -b), K):
        W0[:] = gamma_inv * (A_inv.T.dot(W0).dot(A_inv))
        xi0[:] = gamma_inv * (A_inv.T.dot(xi0))
        kappa0 *= gamma_inv
        nu0 *= gamma_inv

        if 1 - a <= k <= K - a:
            gav = gamma_a * v[k + a - 1]
            W0 -= gav * AaccAa
            xi0 -= gav * np.outer(Aac, y[k + a - 1])
            kappa0 -= gav * np.outer(y[k + a - 1], y[k + a - 1])

        if -b <= k <= K - b - 1:
            gbv = gamma_b * v[k + b]
            W0 += gbv * AbccAb
            xi0 += gbv * np.outer(Abc, y[k + b])
            kappa0 += gbv * np.outer(y[k + b], y[k + b])
            nu0 += gbv

        if 0 <= k <= K - 1:
            W[k] += W0
            xi[k] += xi0
            kappa[k] += kappa0
            nu[k] += nu0

    W *= beta
    xi *= beta
    kappa *= beta
    nu *= beta

@jit(nopython=True)
def backward_recursion_set_jit_1dim_kappa_diag(W, xi, kappa, nu, a, b, delta, y, v, beta, gamma, A, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb, kappa_diag=True):
    W0 = np.zeros_like(W[0])
    xi0 = np.zeros_like(xi[0])
    kappa0 = np.zeros_like(kappa[0])
    nu0 = 0.0

    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        W0[:] = gamma * (A.T.dot(W0).dot(A))
        xi0[:] = gamma * (A.T.dot(xi0))
        kappa0 *= gamma
        nu0 *= gamma

        if -(a - 1) + 1 <= k <= K -(a - 1):
            gav = gamma_a * v[k + a - 1 + -1]
            W0 += gav * AaccAa
            xi0 += gav * np.outer(Aac, y[k + a - 1 + -1])
            kappa0 += gav * np.diag(np.outer(y[k + a - 1 + -1], y[k + a - 1 + -1]))
            nu0 += gav

        if -b + 1 <= k <= K - b:
            gbv = gamma_b * v[k + b + -1]
            W0 -= gbv * AbccAb
            xi0 -= gbv * np.outer(Abc, y[k + b + -1])
            kappa0 -= gbv * np.diag(np.outer(y[k + b + -1], y[k + b + -1]))
            nu0 -= gbv

        if 2 <= k <= K + 1:
            W[k-2] += W0
            xi[k-2] += xi0
            kappa[k-2] += kappa0
            nu[k-2] += nu0

    W *= beta
    xi *= beta
    kappa *= beta
    nu *= beta

@jit(nopython=True)
def backward_recursion_set_jit_1dim_not_kappa_diag(W, xi, kappa, nu, a, b, delta, y, v, beta, gamma, A, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb, kappa_diag=True):
    W0 = np.zeros_like(W[0])
    xi0 = np.zeros_like(xi[0])
    kappa0 = np.zeros_like(kappa[0])
    nu0 = 0.0

    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        W0[:] = gamma * (A.T.dot(W0).dot(A))
        xi0[:] = gamma * (A.T.dot(xi0))
        kappa0 *= gamma
        nu0 *= gamma

        if -(a - 1) + 1 <= k <= K -(a - 1):
            gav = gamma_a * v[k + a - 1 + -1]
            W0 += gav * AaccAa
            xi0 += gav * np.outer(Aac, y[k + a - 1 + -1])
            kappa0 += gav * np.outer(y[k + a - 1 + -1], y[k + a - 1 + -1])
            nu0 += gav

        if -b + 1 <= k <= K - b:
            gbv = gamma_b * v[k + b + -1]
            W0 -= gbv * AbccAb
            xi0 -= gbv * np.outer(Abc, y[k + b + -1])
            kappa0 -= gbv * np.outer(y[k + b + -1], y[k + b + -1])
            nu0 -= gbv

        if 2 <= k <= K + 1:
            W[k-2] += W0
            xi[k-2] += xi0
            kappa[k-2] += kappa0
            nu[k-2] += nu0

    W *= beta
    xi *= beta
    kappa *= beta
    nu *= beta
