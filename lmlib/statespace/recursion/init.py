import numpy as np
from numpy.linalg import inv

__all__ = ['forward_initialize', 'backward_initialize']


def forward_initialize(A, C, gamma, a, b, delta):
    A = A.astype(np.float64)
    C = C.astype(np.float64)
    gamma_inv = gamma ** -1
    A_inv = inv(A)

    gamma_a = np.float_power(gamma, a - 1 - delta)
    if np.isinf(a):
        Aac = C.T
        AaccAa = np.dot(np.atleast_2d(C).T, np.atleast_2d(C))
    else:
        Aa = np.linalg.matrix_power(A, a - 1)
        Aac = np.dot(Aa.T, C.T)
        AaccAa = np.linalg.multi_dot((Aa.T, np.atleast_2d(C).T, np.atleast_2d(C), Aa))

    gamma_b = np.float_power(gamma, b - delta)
    Ab = np.linalg.matrix_power(A, b)
    Abc = np.dot(Ab.T, C.T)
    AbccAb = np.linalg.multi_dot((Ab.T, np.atleast_2d(C).T, np.atleast_2d(C), Ab))

    return gamma_inv, A_inv, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb


def backward_initialize(A, C, gamma, a, b, delta):
    A = A.astype(np.float64)
    C = C.astype(np.float64)

    gamma_a = np.float_power(gamma, a - delta)
    Aa = np.linalg.matrix_power(A, a)
    Aac = np.dot(Aa.T, C.T)
    AaccAa = np.linalg.multi_dot((Aa.T, np.atleast_2d(C).T, np.atleast_2d(C), Aa))

    gamma_b = np.float_power(gamma, b - delta + 1)
    if np.isinf(b):
        Abc = C.T
        AbccAb = np.dot(C.T, C)
    else:
        Ab = np.linalg.matrix_power(A, b + 1)
        Abc = np.dot(Ab.T, C.T)
        AbccAb = np.linalg.multi_dot((Ab.T, np.atleast_2d(C).T, np.atleast_2d(C), Ab))

    return gamma, A, gamma_a, Aac, AaccAa, gamma_b, Abc, AbccAb
