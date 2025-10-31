"""
State-Space Module Tools
========================

Helper functions for State-Space Module

"""


import numpy as np
from numpy.linalg import matrix_power

import warnings

__all__ = ['_trajectory_output', '_window_range', '_window_output',
           '_merge_ks_seg',
           '_covariance_matrix_closed_form', '_covariance_matrix_limited_sum',
           'kron_q', 'common_C_dim']

def _transform_ALSSM_matrices(A, C, P):
    P_inv = np.linalg.inv(P)
    At = P@A@P_inv
    Ct = C@P_inv
    return At, Ct

def _transform_x(xs, P):
    return np.einsum('mn, kn...->km...', P, xs)

def _trajectory_output(A, C, x, js):
    return np.asarray([np.tensordot(C @ matrix_power(A, j), x, axes=(-1, 0)) for j in js])

def _window_range(a, b, direction, gamma, delta, thd):
    if direction == 'fw':
        if gamma > 1:
            a_lim = max(np.log(thd) / np.log(gamma) - 1 + delta, a)
        else:
            a_lim = max(np.log(thd) / np.log(1 / gamma) - 1 + delta, a)
        b_lim = b
    elif direction == 'bw':
        a_lim = a
        if gamma < 1:
            b_lim = min(np.log(thd) / np.log(gamma) + 1 + delta, b)
        else:
            b_lim = min(np.log(thd) / np.log(1 / gamma) + 1 + delta, b)
    else:
        raise ValueError(f'direction {direction} not supported. Must be \'forward\' or \'backward\'.')

    return range(int(a_lim), int(b_lim) + 1)

def _window_output(a, b, direction, gamma, delta, thd):
    ab_range = _window_range(a, b, direction, gamma, delta, thd)
    return ab_range, gamma ** (np.array(ab_range) - delta)

def _merge_ks_seg(arr, merge_ks, merge_seg):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if merge_seg:
            if merge_ks:
                return np.nanmax(arr, axis=(0, 1))
            else:
                return np.nanmax(arr, axis=1)
        else:
            if merge_ks:
                return np.nanmax(arr, axis=0)
            else:
                return arr

def _covariance_matrix_closed_form(A, C, gamma, a, b, delta):
    if np.shape(np.atleast_2d(C))[0] != 1:
        # TODO:_covariance_matrix_limited_sum() as alternative
        raise ValueError('Multi-channel ALSSM Steady State not implemented.')

    N = np.shape(A)[0]
    gATA = gamma * np.kron(np.transpose(A), A)

    if gamma > 1:
        gATA_a = np.linalg.matrix_power(gATA, a - 1) if ~(np.isinf(a)) else np.zeros_like(gATA, dtype=float)
        gATA_b = np.linalg.matrix_power(gATA, b) if ~(np.isinf(b)) else np.zeros_like(gATA, dtype=float)
        if np.linalg.cond(np.linalg.inv(gATA) - np.eye(N * N)) > 1e15:
            print(Warning('Badly Conditioned Steady State Matrix W: Use larger boundaries or lower g.'))

        return np.dot(gamma ** (-delta),
                      np.kron(np.eye(N), np.atleast_2d(C)) @
                      (np.linalg.inv(np.linalg.inv(gATA) - np.eye(N * N)) @ (gATA_a - gATA_b)) @
                      np.kron(np.atleast_2d(C).T, np.eye(N))
                      )
    else:
        gATA_a = np.linalg.matrix_power(gATA, a) if ~(np.isinf(a)) else np.zeros_like(gATA)
        gATA_b = np.linalg.matrix_power(gATA, b + 1) if ~(np.isinf(b)) else np.zeros_like(gATA)
        if np.linalg.cond(np.eye(N * N) - gATA) > 1e15:
            print(Warning('Badly Conditioned Steady State Matrix W: Use larger boundaries or lower g.'))
        return np.dot(gamma ** (-delta),
                      np.kron(np.eye(N), np.atleast_2d(C)) @
                      (np.linalg.inv(np.eye(N * N) - gATA) @ (gATA_a - gATA_b)) @
                      np.kron(np.atleast_2d(C).T, np.eye(N))
                      )

def _covariance_matrix_limited_sum(A, C, gamma, a, b, delta):
    # TODO: Implementation
    raise NotImplementedError('limited_sum is not implemented yet.')

def kron_q(x, q):
    """
    Kronecker Power for a vector x by as non-negative integer q

    For reference, see  [Baeriswyl2025]_ [Eq. 6].

    Parameters
    ----------
    x : array_like
        Base array/matrix
    q : int
        Kronecker exponent

    Returns
    -------
    out : array_like
        array/matrix raised by the kronecker exponent

    """
    if q == 0:
        return np.eye(1)
    elif q == 1:
        return x
    else:
        out = x
        for _ in range(q-1):
            out = np.kron(out, out)
    return out

def common_C_dim(alssms):
    C_ndim = [alssm.C.ndim for alssm in alssms]
    C_L = [np.atleast_2d(alssm.C).shape[0] for alssm in alssms]
    return sum(np.diff(C_ndim)) == 0 and sum(np.diff(C_L)) == 0
