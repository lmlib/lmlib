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
            if beta == 1:
                W[k] += W0
            else:
                W[k] += W0 * beta
    

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
            if beta == 1:
                W[k-2] += W0
            else:
                W[k-2] += W0 * beta


# xi1 recursions
def jit_recursion_xi1(xi1, A, C, a, b, direction, delta, gamma, y, v, beta):
    _A = A.astype(float)
    _C = C.astype(float)
    if direction == 'fw':
        gamma_a = gamma ** (a - 1 - delta)
        _a = -2**31 if np.isinf(a) else a
        Aa = matrix_power(_A, 0 if np.isinf(a) else _a - 1)
        Ab = matrix_power(_A, b)
        jit_forward_recursion_xi(xi1, _A, _C, _a, b, delta, gamma, y, v, beta, Aa, gamma_a, Ab)
    elif direction == 'bw':
        gamma_b = gamma ** (b - delta + 1)
        _b = 2**31 if np.isinf(b) else b
        Ab = matrix_power(_A, 0 if np.isinf(b) else _b + 1)
        Aa = matrix_power(_A, a)
        jit_backward_recursion_xi(xi1, _A, _C, a, _b, delta, gamma, y, v, beta, Ab, gamma_b, Aa)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

@jit(nopython=True)
def jit_forward_recursion_xi(xi, A, C,  a, b, delta, gamma, y, v, beta, Aa, gamma_a, Ab) :
    """Equation  (28)  in Wildhaber 2018"""

    gamma_inv = 1 / gamma
    gamma_b = gamma ** (b - delta)
    A_inv = inv(A)
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
            if beta == 1:
                xi[k] += xi0
            else:
                xi[k] += xi0 * beta


@jit(nopython=True)
def jit_backward_recursion_xi(xi, A, C,  a, b, delta, gamma, y, v, beta, Ab, gamma_b, Aa):
    """Equation  (32)  in Wildhaber 2018"""

    gamma_a = gamma ** (a - delta)
    Aac = np.dot(Aa.T, C.T)
    Abc = np.dot(Ab.T, C.T)

    # intermediate vars
    xi0 = np.zeros_like(xi[0])
    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        xi0[:] = gamma * (A.T.dot(xi0))

        if -(a - 1) + 1 <= k <= K - (a - 1):
            y_slice = np.ascontiguousarray(y[k + a - 1 + -1])
            xi0 += gamma_a * v[k + a - 1 + -1] * np.dot(Aac, y_slice)

        if -b + 1 <= k <= K - b:
            y_slice = np.ascontiguousarray(y[k + b + -1])
            xi0 -= gamma_b * v[k + b + -1] * np.dot(Abc, y_slice)

        if 2 <= k <= K + 1:
            if beta == 1:
                xi[k - 2] += xi0
            else:
                xi[k - 2] += xi0 * beta


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
            y_slice = np.ascontiguousarray(y[k + a - 1])
            kappa0 -= gamma_a * v[k + a - 1] * np.dot(y_slice, y_slice)

        if -b <= k <= K - b - 1:
            y_slice = np.ascontiguousarray(y[k + b])
            kappa0 += gamma_b * v[k + b] * np.dot(y_slice, y_slice)

        if 0 <= k <= K - 1:
            if beta == 1:
                kappa[k] += kappa0
            else:
                kappa[k] += kappa0 * beta



@jit(nopython=True)
def jit_backward_recursion_kappa(kappa, a, b, delta, gamma, y, v, beta, gamma_b):
    """Equation  (33)  in Wildhaber 2018"""

    gamma_a = gamma ** (a - delta)

    kappa0 = 0.0
    K = len(y)

    for k in range(max(K - a, K) + 1, 1, -1):
        kappa0 *= gamma

        if -(a - 1) + 1 <= k <= K - (a - 1):
            y_slice = np.ascontiguousarray(y[k + a - 1 + -1])
            kappa0 += gamma_a * v[k + a - 1 + -1] * np.dot(y_slice, y_slice)

        if -b + 1 <= k <= K - b:
            y_slice = np.ascontiguousarray(y[k + b + -1])
            kappa0 -= gamma_b * v[k + b + -1] * np.dot(y_slice, y_slice)

        if 2 <= k <= K + 1:
            if beta == 1:
                kappa[k - 2] += kappa0
            else:
                kappa[k - 2] += kappa0 * beta




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
            if beta == 1:
                nu[k] += nu0
            else:
                nu[k] += nu0 * beta


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
            if beta == 1:
                nu[k - 2] += nu0
            else:
                nu[k - 2] += nu0 * beta




# ---------------------------------------------------------------------------
# xi asterisk l recursions (ND / multi-dimensional costs)
# ---------------------------------------------------------------------------

def jit_xi_asterisk_l_recursion(xi, A, C, a, b, direction, delta, gamma, I_N, xi_N, v, beta):
    """Dispatcher for the JIT asterisk-l recursion (forward / backward)."""
    if direction == 'fw':
        _jit_xi_asterisk_l_forward_recursion(xi, A, C, a, b, delta, gamma, I_N, xi_N, v, beta)
    elif direction == 'bw':
        _jit_xi_asterisk_l_backward_recursion(xi, A, C, a, b, delta, gamma, I_N, xi_N, v, beta)
    else:
        raise ValueError('direction must be either "fw" or "bw"')


def _jit_xi_asterisk_l_forward_recursion(xi, A, C, a, b, delta, gamma, I_N, xi_N, v, beta):
    """JIT forward asterisk-l recursion: precompute matrices then call nopython kernel."""
    gamma_inv = 1.0 / gamma
    gamma_a   = gamma ** (a - 1 - delta)
    gamma_b   = gamma ** (b - delta)

    A_f = np.atleast_2d(A).astype(np.float64)
    # C may be 1-D (shape (N,)) from AlssmPolyLegendre — ensure 2-D (Q_C, N_curr)
    C_f = np.atleast_2d(C).astype(np.float64)
    A_inv = inv(A_f)

    Nq_prev = int(I_N.shape[0])
    N_curr  = int(A_f.shape[0])
    Q_C     = int(C_f.shape[0])

    Aa  = matrix_power(A_f, 0 if np.isinf(a) else int(a - 1))
    CAa = np.ascontiguousarray((C_f @ Aa).astype(np.float64))   # (Q_C, N_curr)
    Ab  = matrix_power(A_f, int(b))
    CAb = np.ascontiguousarray((C_f @ Ab).astype(np.float64))   # (Q_C, N_curr)

    K      = int(xi.shape[0])
    batch  = int(np.prod(xi.shape[1:-1])) if xi.ndim > 2 else 1
    Nq_out = int(xi.shape[-1])

    xi_flat   = xi.reshape(K, batch, Nq_out)
    xi_N_flat = xi_N.reshape(K, batch, int(xi_N.shape[-1]))

    _jit_ast_forward_kernel(
        xi_flat,
        np.ascontiguousarray(A_inv.astype(np.float64)),
        np.ascontiguousarray(CAa),
        np.ascontiguousarray(CAb),
        int(a), int(b), gamma_inv, gamma_a, gamma_b,
        Nq_prev, N_curr, Q_C,
        np.ascontiguousarray(xi_N_flat.astype(np.float64)),
        float(beta),
    )


def _jit_xi_asterisk_l_backward_recursion(xi, A, C, a, b, delta, gamma, I_N, xi_N, v, beta):
    """JIT backward asterisk-l recursion: precompute matrices then call nopython kernel."""
    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)

    A_f = np.atleast_2d(A).astype(np.float64)
    C_f = np.atleast_2d(C).astype(np.float64)

    Nq_prev = int(I_N.shape[0])
    N_curr  = int(A_f.shape[0])
    Q_C     = int(C_f.shape[0])

    Aa  = matrix_power(A_f, int(a))
    CAa = np.ascontiguousarray((C_f @ Aa).astype(np.float64))
    Ab  = matrix_power(A_f, 0 if np.isinf(b) else int(b + 1))
    CAb = np.ascontiguousarray((C_f @ Ab).astype(np.float64))

    K      = int(xi.shape[0])
    batch  = int(np.prod(xi.shape[1:-1])) if xi.ndim > 2 else 1
    Nq_out = int(xi.shape[-1])

    xi_flat   = xi.reshape(K, batch, Nq_out)
    xi_N_flat = xi_N.reshape(K, batch, int(xi_N.shape[-1]))

    _jit_ast_backward_kernel(
        xi_flat,
        np.ascontiguousarray(A_f),
        np.ascontiguousarray(CAa),
        np.ascontiguousarray(CAb),
        int(a), int(b), gamma, gamma_a, gamma_b,
        Nq_prev, N_curr, Q_C,
        np.ascontiguousarray(xi_N_flat.astype(np.float64)),
        float(beta),
    )


@jit(nopython=True)
def _kron_blockdiag_mv(A_T, vec_in, Nq_prev, N, vec_out):
    """
    Compute  vec_out = kron(I_{Nq_prev}, A).T @ vec_in  in-place (overwrites vec_out).

    Equivalent to: reshape vec_in -> (Nq_prev, N), apply A.T to each row, reshape back.
    Operates without forming the (Nq_prev*N) x (Nq_prev*N) Kronecker matrix.

    Parameters
    ----------
    A_T    : (N, N) — transpose of A
    vec_in : (Nq_prev * N,)
    Nq_prev, N : ints
    vec_out: (Nq_prev * N,) — overwritten
    """
    for i in range(Nq_prev):
        off = i * N
        for m in range(N):
            s = 0.0
            for n in range(N):
                s += A_T[m, n] * vec_in[off + n]
            vec_out[off + m] = s


@jit(nopython=True)
def _kron_CA_mv_acc(CA_T, xi_n, Nq_prev, N_out, Q_C, result):
    """
    Accumulate  result += kron(I_{Nq_prev}, CA).T @ xi_n  in-place.

    CA_T   : (N_out, Q_C) — (C @ A^power).T
    xi_n   : (Nq_prev * Q_C,)
    result : (Nq_prev * N_out,) — accumulated in-place (caller must zero before call)
    """
    for i in range(Nq_prev):
        off_in  = i * Q_C
        off_out = i * N_out
        for j in range(N_out):
            s = 0.0
            for q in range(Q_C):
                s += CA_T[j, q] * xi_n[off_in + q]
            result[off_out + j] += s


@jit(nopython=True)
def _jit_ast_forward_kernel(xi, A_inv, CAa, CAb, a, b,
                             gamma_inv, gamma_a, gamma_b,
                             Nq_prev, N_curr, Q_C, xi_N, beta):
    """
    Numba nopython forward xi-asterisk-l recursion kernel.

    Mirrors numpy_xi_asterisk_l_forward_recursion exactly.
    Uses explicit loops over batch and state dimensions instead of einsum/kron.

    Parameters
    ----------
    xi     : (K, batch, Nq_out)  float64  — output buffer, updated in-place
    A_inv  : (N_curr, N_curr)    float64
    CAa    : (Q_C, N_curr)       float64  C @ A^(a-1)
    CAb    : (Q_C, N_curr)       float64  C @ A^b
    a, b   : int  — segment endpoints
    gamma_* : float  — pre-computed gamma powers
    Nq_prev, N_curr, Q_C : int  — dimension bookkeeping
    xi_N   : (K, batch, Nq_in)  float64  — input from previous dimension
    beta   : float
    """
    K      = xi.shape[0]
    batch  = xi.shape[1]
    Nq_out = xi.shape[2]

    A_inv_T = A_inv.T.copy()
    CAa_T   = CAa.T.copy()   # (N_curr, Q_C)
    CAb_T   = CAb.T.copy()

    _xi  = np.zeros(Nq_out)
    _tmp = np.zeros(Nq_out)

    for b_idx in range(batch):
        # Reset running state
        for n in range(Nq_out):
            _xi[n] = 0.0

        for k in range(min(0, -b), K):
            # _xi = gamma_inv * kron(I, A_inv).T @ _xi
            _kron_blockdiag_mv(A_inv_T, _xi, Nq_prev, N_curr, _tmp)
            for n in range(Nq_out):
                _xi[n] = gamma_inv * _tmp[n]

            if 1 - a <= k <= K - a:
                xi_n = xi_N[k + a - 1, b_idx]
                for n in range(Nq_out):
                    _tmp[n] = 0.0
                _kron_CA_mv_acc(CAa_T, xi_n, Nq_prev, N_curr, Q_C, _tmp)
                for n in range(Nq_out):
                    _xi[n] -= gamma_a * _tmp[n]

            if -b <= k <= K - b - 1:
                xi_n = xi_N[k + b, b_idx]
                for n in range(Nq_out):
                    _tmp[n] = 0.0
                _kron_CA_mv_acc(CAb_T, xi_n, Nq_prev, N_curr, Q_C, _tmp)
                for n in range(Nq_out):
                    _xi[n] += gamma_b * _tmp[n]

            if 0 <= k <= K - 1:
                if beta == 1.0:
                    for n in range(Nq_out):
                        xi[k, b_idx, n] += _xi[n]
                else:
                    for n in range(Nq_out):
                        xi[k, b_idx, n] += _xi[n] * beta


@jit(nopython=True)
def _jit_ast_backward_kernel(xi, A, CAa, CAb, a, b,
                              gamma, gamma_a, gamma_b,
                              Nq_prev, N_curr, Q_C, xi_N, beta):
    """
    Numba nopython backward xi-asterisk-l recursion kernel.

    Mirrors numpy_xi_asterisk_l_backward_recursion exactly.

    Parameters
    ----------
    xi     : (K, batch, Nq_out)  float64
    A      : (N_curr, N_curr)    float64
    CAa    : (Q_C, N_curr)       float64  C @ A^a
    CAb    : (Q_C, N_curr)       float64  C @ A^(b+1)
    a, b   : int
    gamma* : float
    Nq_prev, N_curr, Q_C : int
    xi_N   : (K, batch, Nq_in)  float64
    beta   : float
    """
    K      = xi.shape[0]
    batch  = xi.shape[1]
    Nq_out = xi.shape[2]

    A_T   = A.T.copy()
    CAa_T = CAa.T.copy()
    CAb_T = CAb.T.copy()

    _xi  = np.zeros(Nq_out)
    _tmp = np.zeros(Nq_out)

    for b_idx in range(batch):
        for n in range(Nq_out):
            _xi[n] = 0.0

        for k in range(max(K - a, K) + 1, 1, -1):
            # _xi = gamma * kron(I, A).T @ _xi
            _kron_blockdiag_mv(A_T, _xi, Nq_prev, N_curr, _tmp)
            for n in range(Nq_out):
                _xi[n] = gamma * _tmp[n]

            if -(a - 1) + 1 <= k <= K - (a - 1):
                xi_n = xi_N[k + a - 1 - 1, b_idx]
                for n in range(Nq_out):
                    _tmp[n] = 0.0
                _kron_CA_mv_acc(CAa_T, xi_n, Nq_prev, N_curr, Q_C, _tmp)
                for n in range(Nq_out):
                    _xi[n] += gamma_a * _tmp[n]

            if -b + 1 <= k <= K - b:
                xi_n = xi_N[k + b - 1, b_idx]
                for n in range(Nq_out):
                    _tmp[n] = 0.0
                _kron_CA_mv_acc(CAb_T, xi_n, Nq_prev, N_curr, Q_C, _tmp)
                for n in range(Nq_out):
                    _xi[n] -= gamma_b * _tmp[n]

            if 2 <= k <= K + 1:
                if beta == 1.0:
                    for n in range(Nq_out):
                        xi[k - 2, b_idx, n] += _xi[n]
                else:
                    for n in range(Nq_out):
                        xi[k - 2, b_idx, n] += _xi[n] * beta
