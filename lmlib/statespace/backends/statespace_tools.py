"""
State-Space Module Tools
========================

Helper functions for State-Space Module

"""
import numpy as np


__all__ = ['kron_q', 'common_C_dim']

def _transform_ALSSM_matrices(A, C, P):
    P_inv = np.linalg.inv(P)
    At = P@A@P_inv
    Ct = C@P_inv
    return At, Ct

def _transform_x(xs, P):
    return np.einsum('mn, kn...->km...', P, xs)


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
    """
    Checks if all Alssm have the same C output dimension

    Parameters
    ----------
    alssms : list of Alssm
        Alssms to check for common C dimension

    Returns
    -------
    bool : True if all Alssms have the same C dimension else False

    """
    C_ndim = [alssm.C.ndim for alssm in alssms]
    C_L = [np.atleast_2d(alssm.C).shape[0] for alssm in alssms]
    return sum(np.diff(C_ndim)) == 0 and sum(np.diff(C_L)) == 0
