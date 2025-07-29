import numpy as np
from numpy.linalg import inv, matrix_rank

__all__ = ['forward_recursion_py', 'backward_recursion_py',
           'forward_recursion_xi_kappa_nu_py', 'backward_recursion_xi_kappa_nu_py',
           'forward_recursion_set_py', 'backward_recursion_set_py',
           'forward_recursion_set_xi_kappa_nu_py', 'backward_recursion_set_xi_kappa_nu_py',
           'minimize_v_py', 'minimize_x_py',
           'minimize_v_steady_state_py', 'minimize_x_steady_state_py']


def forward_recursion_py(W, xi, kappa, nu, a, b, delta, y, v, beta, gamma_inv, A_inv, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb):
    W0 = np.zeros_like(W[0])
    xi0 = np.zeros_like(xi[0])
    kappa0 = np.zeros_like(kappa[0])
    nu0 = np.zeros_like(nu[0])

    K = len(y)
    for k in range(min(0, -b), K):
        W0 = gamma_inv * (A_inv.T.dot(W0).dot(A_inv))
        xi0 = gamma_inv * (A_inv.T.dot(xi0))
        kappa0 *= gamma_inv
        nu0 *= gamma_inv

        if 1-a <= k <= K - a:
            gav = gamma_a * v[k + a - 1]
            W0 -= gav * AaccAa
            xi0 -= gav * np.dot(Aac, y[k + a - 1])
            kappa0 -= gav * np.dot(y[k + a - 1].T, y[k + a - 1])
            nu0 -= gav

        if -b <= k <= K - b - 1:
            gbv = gamma_b * v[k + b]
            W0 += gbv * AbccAb
            xi0 += gbv * np.dot(Abc, y[k + b])
            kappa0 += gbv * np.dot(y[k + b].T, y[k + b])
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


def backward_recursion_py(W, xi, kappa, nu, a, b, delta, y, v, beta, gamma, A, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb):
    W0 = np.zeros_like(W[0])
    xi0 = np.zeros_like(xi[0])
    kappa0 = np.zeros_like(kappa[0])
    nu0 = np.zeros_like(nu[0])

    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        W0 = gamma * (A.T.dot(W0).dot(A))
        xi0 = gamma * (A.T.dot(xi0))
        kappa0 *= gamma
        nu0 *= gamma

        if -(a - 1) + 1 <= k <= K -(a - 1):
            gav = gamma_a * v[k + a - 1 + -1]
            W0 += gav * AaccAa
            xi0 += gav * np.dot(Aac, y[k + a - 1 + -1])
            kappa0 += gav * np.dot(y[k + a - 1 + -1].T, y[k + a - 1 + -1])
            nu0 += gav

        if -b + 1 <= k <= K - b:
            gbv = gamma_b * v[k + b + -1]
            W0 -= gbv * AbccAb
            xi0 -= gbv * np.dot(Abc, y[k + b + -1])
            kappa0 -= gbv * np.dot(y[k + b + -1].T, y[k + b + -1])
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


def forward_recursion_xi_kappa_nu_py(xi, kappa, nu, a, b, delta, y, v, beta, gamma_inv, A_inv, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb):
    xi0 = np.zeros_like(xi[0])
    kappa0 = np.zeros_like(kappa[0])
    nu0 = np.zeros_like(nu[0])

    K = len(y)
    for k in range(min(0, -b), K):
        xi0 = gamma_inv * (A_inv.T.dot(xi0))
        kappa0 *= gamma_inv
        nu0 *= gamma_inv

        if 1 - a <= k <= K - a:
            gav = gamma_a * v[k + a - 1]
            xi0 -= gav * np.dot(Aac, y[k + a - 1])
            kappa0 -= gav * np.dot(y[k + a - 1].T, y[k + a - 1])
            nu0 -= gav

        if -b <= k <= K - b - 1:
            gbv = gamma_b * v[k + b]
            xi0 += gbv * np.dot(Abc, y[k + b])
            kappa0 += gbv * np.dot(y[k + b].T, y[k + b])
            nu0 += gbv

        if 0 <= k <= K - 1:
            xi[k] += xi0
            kappa[k] += kappa0
            nu[k] += nu0

    xi *= beta
    kappa *= beta
    nu *= beta


def backward_recursion_xi_kappa_nu_py(xi, kappa, nu, a, b, delta, y, v, beta, gamma, A, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb):
    xi0 = np.zeros_like(xi[0])
    kappa0 = np.zeros_like(kappa[0])
    nu0 = np.zeros_like(nu[0])

    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        xi0 = gamma * (A.T.dot(xi0))
        kappa0 *= gamma
        nu0 *= gamma

        if -(a - 1) + 1 <= k <= K -(a - 1):
            gav = gamma_a * v[k + a - 1 + -1]
            xi0 += gav * np.dot(Aac, y[k + a - 1 + -1])
            kappa0 += gav * np.dot(y[k + a - 1 + -1].T, y[k + a - 1 + -1])
            nu0 += gav

        if -b + 1 <= k <= K - b:
            gbv = gamma_b * v[k + b + -1]
            xi0 -= gbv * np.dot(Abc, y[k + b + -1])
            kappa0 -= gbv * np.dot(y[k + b + -1].T, y[k + b + -1])
            nu0 -= gbv

        if 2 <= k <= K + 1:
            xi[k - 2] += xi0
            kappa[k - 2] += kappa0
            nu[k - 2] += nu0

    xi *= beta
    kappa *= beta
    nu *= beta


def forward_recursion_set_py(W, xi, kappa, nu, a, b, delta, y, v, beta, gamma_inv, A_inv, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb, kappa_diag=True):
    W0 = np.zeros_like(W[0])
    xi0 = np.zeros_like(xi[0])
    kappa0 = np.zeros_like(kappa[0])
    nu0 = np.zeros_like(nu[0])

    K = len(y)

    for k in range(min(0, -b), K):
        W0 = gamma_inv * (A_inv.T.dot(W0).dot(A_inv))
        xi0 = gamma_inv * (A_inv.T.dot(xi0))
        kappa0 *= gamma_inv
        nu0 *= gamma_inv

        if 1 - a <= k <= K - a:
            gav = gamma_a * v[k + a - 1]
            W0 -= gav * AaccAa
            xi0 -= gav * np.outer(Aac, y[k + a - 1])
            if kappa_diag:
                kappa0 -= gav * np.diag(np.outer(y[k + a - 1], y[k + a - 1]))
            else:
                kappa0 -= gav * np.outer(y[k + a - 1], y[k + a - 1])
            nu0 -= gav

        if -b <= k <= K - b - 1:
            gbv = gamma_b * v[k + b]
            W0 += gbv * AbccAb
            xi0 += gbv * np.outer(Abc, y[k + b])
            if kappa_diag:
                kappa0 += gbv * np.diag(np.outer(y[k + b], y[k + b]))
            else:
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


def backward_recursion_set_py(W, xi, kappa, nu, a, b, delta, y, v, beta, gamma, A, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb, kappa_diag=True):
    W0 = np.zeros_like(W[0])
    xi0 = np.zeros_like(xi[0])
    kappa0 = np.zeros_like(kappa[0])
    nu0 = np.zeros_like(nu[0])

    K = len(y)
    for k in range(max(K - a, K) + 1, 1, -1):
        W0 = gamma * (A.T.dot(W0).dot(A))
        xi0 = gamma * (A.T.dot(xi0))
        kappa0 *= gamma
        nu0 *= gamma

        if -(a - 1) + 1 <= k <= K - (a - 1):
            gav = gamma_a * v[k + a - 1 + -1]
            W0 += gav * AaccAa
            xi0 += gav * np.outer(Aac, y[k + a - 1 + -1])
            if kappa_diag:
                kappa0 += gav * np.diag(np.outer(y[k + a - 1 + -1], y[k + a - 1 + -1]))
            else:
                kappa0 += gav * np.outer(y[k + a - 1 + -1], y[k + a - 1 + -1])
            nu0 += gav

        if -b + 1 <= k <= K - b:
            gbv = gamma_b * v[k + b + -1]
            W0 -= gbv * AbccAb
            xi0 -= gbv * np.outer(Abc, y[k + b + -1])
            if kappa_diag:
                kappa0 -= gbv * np.diag(np.outer(y[k + b + -1], y[k + b + -1]))
            else:
                kappa0 -= gbv * np.outer(y[k + b + -1], y[k + b + -1])

            nu0 -= gbv

        if 2 <= k <= K + 1:
            W[k - 2] += W0
            xi[k - 2] += xi0
            kappa[k - 2] += kappa0
            nu[k - 2] += nu0

    W *= beta
    xi *= beta
    kappa *= beta
    nu *= beta


def forward_recursion_set_xi_kappa_nu_py(xi, kappa, nu, a, b, delta, y, v, beta, gamma_inv, A_inv, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb, kappa_diag=True):
    xi0 = np.zeros_like(xi[0])
    kappa0 = np.zeros_like(kappa[0])
    nu0 = np.zeros_like(nu[0])

    K = len(y)

    for k in range(min(0, -b), K):
        xi0 = gamma_inv * (A_inv.T.dot(xi0))
        kappa0 *= gamma_inv
        nu0 *= gamma_inv

        if 1 - a <= k <= K - a:
            gav = gamma_a * v[k + a - 1]
            xi0 -= gav * np.outer(Aac, y[k + a - 1])
            if kappa_diag:
                kappa0 -= gav * np.diag(np.outer(y[k + a - 1], y[k + a - 1]))
            else:
                kappa0 -= gav * np.outer(y[k + a - 1], y[k + a - 1])
            nu0 -= gav

        if -b <= k <= K - b - 1:
            gbv = gamma_b * v[k + b]
            xi0 += gbv * np.outer(Abc, y[k + b])
            if kappa_diag:
                kappa0 += gbv * np.diag(np.outer(y[k + b], y[k + b]))
            else:
                kappa0 += gbv * np.outer(y[k + b], y[k + b])
            nu0 += gbv

        if 0 <= k <= K - 1:
            xi[k] += xi0
            kappa[k] += kappa0
            nu[k] += nu0

    xi *= beta
    kappa *= beta
    nu *= beta


def backward_recursion_set_xi_kappa_nu_py(xi, kappa, nu, a, b, delta, y, v, beta, gamma, A, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb, kappa_diag=True):
    xi0 = np.zeros_like(xi[0])
    kappa0 = np.zeros_like(kappa[0])
    nu0 = np.zeros_like(nu[0])

    K = len(y)
    for k in range(max(K - a, K) + 1, 1, -1):
        xi0 = gamma * (A.T.dot(xi0))
        kappa0 *= gamma
        nu0 *= gamma

        if -(a - 1) + 1 <= k <= K - (a - 1):
            gav = gamma_a * v[k + a - 1 + -1]
            xi0 += gav * np.outer(Aac, y[k + a - 1 + -1])
            if kappa_diag:
                kappa0 += gav * np.diag(np.outer(y[k + a - 1 + -1], y[k + a - 1 + -1]))
            else:
                kappa0 += gav * np.outer(y[k + a - 1 + -1], y[k + a - 1 + -1])
            nu0 += gav

        if -b + 1 <= k <= K - b:
            gbv = gamma_b * v[k + b + -1]
            xi0 -= gbv * np.outer(Abc, y[k + b + -1])
            if kappa_diag:
                kappa0 -= gbv * np.diag(np.outer(y[k + b + -1], y[k + b + -1]))
            else:
                kappa0 -= gbv * np.outer(y[k + b + -1], y[k + b + -1])

            nu0 -= gbv

        if 2 <= k <= K + 1:
            xi[k - 2] += xi0
            kappa[k - 2] += kappa0
            nu[k - 2] += nu0

    xi *= beta
    kappa *= beta
    nu *= beta


def minimize_v_py(v, W, xi, H, h):

    HTWH = H.T @ W @ H
    HTxiWh = np.einsum('mn, kn...->km...', H.T, (xi - W @ h))

    mask_is_invertible = matrix_rank(HTWH) == H.shape[1]
    v[mask_is_invertible] = np.einsum('knm, kn... -> km...', inv(HTWH[mask_is_invertible]), HTxiWh[mask_is_invertible])


def minimize_x_py(x, W, xi):

    mask_is_invertible = matrix_rank(W) == x.shape[1]
    x[mask_is_invertible] = np.einsum('knm, kn... -> km...', inv(W[mask_is_invertible]), xi[mask_is_invertible])


def minimize_v_steady_state_py(v, W, xi, H, h):

    HTWH = H.T @ W @ H
    HTxiWh = np.einsum('mn, kn...->km...', H.T, (xi - W @ h))

    # TODO: get direct inverse steady state W
    v[...] = np.einsum('nm, kn... -> km...', inv(HTWH), HTxiWh)



def minimize_x_steady_state_py(x, W, xi):
    # TODO: get direct inverse steady state W
    x[...] = np.einsum('nm, km...-> kn...', inv(W), xi)