"""Collection of experimental functions (beta)"""

import os
import numpy as np
import csv as csv
import lmlib as lm
import itertools
import datetime as datetime


__all__ = ['mpoly_fit_subspace1',
           'constrain_matrix_for_exponents'
           ]


"""
 This file hosts helper functions or classes which are provided as beta and will later be transferred to the library 
 when having reached a stable state. 
"""




def constrain_matrix_for_exponents(expos, threshold):
    """
    Creates a matrix which reduces exponents which are summed higher than the threshold

    Parameters
    ----------
    expos : tuple of array_like
        exponent vectors
    threshold : int
        Threshold

    Returns
    -------
    H : :class:`np.ndarray`
        H matrix
    """
    prod = list(itertools.product(*expos))
    mask = np.sum(prod, axis=1) > threshold
    H = np.eye(len(prod))
    return np.delete(H, np.argwhere(mask), 1)


def mpoly_fit_subspace1(betas, beta_expo, boundaries, coords, coord_expos, H=None, return_cost=False):
    """
    Fits a multivariate polynomial to polynomial subspace with corresponding coordinates

    Parameters
    ----------
    betas : array_like of shape (N, Q, ...)
        polynomial coefficients
    beta_expo : array_like of shape (Q,)
        exponent vector of beta
    boundaries : array_like of shape (2,)
        integral boundaries (lower, upper)
    coords : array_like of shape (N, M)
        coordinates of shape (M,) for each beta in betas
    coord_expos : tuple of shape (M,) of 1D- array_like
        exponent vector for each variable in coordinates
    H : None, array_like, optional
        output exponent reduction matrix ( max limit )
    return_cost : bool, optional
        returns additionally the cost for each sample

    Returns
    -------
    alphas : array_like
        coefficients of multivariate polynomial fit
    cost : array_like, optional
        Cost if flag `return_cost = True`

    """
    beta_expo = np.asarray(beta_expo)
    betas = np.asarray(betas)
    if betas.shape[1] != len(beta_expo):
        raise ValueError('beta is betas doesn\'t match with beta_expo. Wrong length.')
    N = betas.shape[0]
    if len(boundaries) != 2:
        raise ValueError('boundaries has not a length of 2')
    a = boundaries[0]
    b = boundaries[1]
    if a > b:
        raise ValueError('Condition not fulfilled: a <= b')
    coord_expos = np.asarray(coord_expos)
    if coord_expos.ndim != 2:
        raise ValueError('coord_expos has wrong number of dimensions. '
                         'Needs to be tuple(array_like)/tuple(array_like, array_like, ...)')
    coords = np.asarray(coords)
    if coords.shape != (N, len(coord_expos)):
        raise ValueError('coords  doesn\'t match with length of betas or with length of coord_expos.')

    J = max(beta_expo) + 1
    M = lm.poly_square_expo_M(beta_expo)
    L = lm.poly_int_coef_L(M @ beta_expo)
    c = np.power(b, M @ beta_expo + 1) - np.power(a, M @ beta_expo + 1)
    C = np.reshape(L.T @ c, (J, J))
    As = np.zeros((N, len(beta_expo), np.prod([len(expo) for expo in coord_expos])*len(beta_expo)))
    for n, coord in enumerate(coords):
        A = lm.kron_sequence([np.power(value, expo) for value, expo in zip(coord, coord_expos)])
        As[n] = np.kron(np.eye(J), A)

    if H is None:
        ACA = np.sum([A.T @ C @ A for A in As], axis=0)
        ACbeta = np.sum([np.einsum('an, n...->a...', A.T.dot(C), beta) for A, beta in zip(As, betas)], axis=0)
    else:
        ACA = np.sum([H.T @ A.T @ C @ A @ H for A in As], axis=0)
        ACbeta = np.sum([np.einsum('an, n...->a...', H.T @ A.T.dot(C), beta) for A, beta in zip(As, betas)], axis=0)
    alphas = np.einsum('ba, a...->b...', np.linalg.pinv(ACA), ACbeta)
    if return_cost:
        Js = np.einsum('a..., a...->...', np.einsum('b..., ba->a...', alphas, ACA), alphas) \
             - 2*np.einsum('b..., b...->...', ACbeta, alphas)\
             + np.einsum('ja..., ja... ->...', np.einsum('jb..., ba ->ja...', betas, C), betas)
        return alphas, Js
    return alphas

