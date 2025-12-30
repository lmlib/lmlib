import numpy as np
from numpy.linalg import matrix_power, cond, inv

__all__ = ['covariance_matrix_closed_form', 'covariance_matrix_limited_sum']

def covariance_matrix_closed_form(A, C, gamma, a, b, delta):
    N = np.shape(A)[0]
    gATA = gamma * np.kron(np.transpose(A), A)

    if gamma > 1:
        gATA_a = matrix_power(gATA, a - 1) if ~(np.isinf(a)) else np.zeros_like(gATA, dtype=float)
        gATA_b = matrix_power(gATA, b) if ~(np.isinf(b)) else np.zeros_like(gATA, dtype=float)
        if cond(inv(gATA) - np.eye(N * N)) > 1e15:
            print(Warning('Badly Conditioned Steady State Matrix W: Use larger boundaries or lower g.'))

        return np.dot(gamma ** (-delta),
                      np.kron(np.eye(N), np.atleast_2d(C)) @
                      (inv(inv(gATA) - np.eye(N * N)) @ (gATA_a - gATA_b)) @
                      np.kron(np.atleast_2d(C).T, np.eye(N))
                      )
    else:
        gATA_a = matrix_power(gATA, a) if ~(np.isinf(a)) else np.zeros_like(gATA)
        gATA_b = matrix_power(gATA, b + 1) if ~(np.isinf(b)) else np.zeros_like(gATA)
        if cond(np.eye(N * N) - gATA) > 1e15:
            print(Warning('Badly Conditioned Steady State Matrix W: Use larger boundaries or lower g.'))
        return np.dot(gamma ** (-delta),
                      np.kron(np.eye(N), np.atleast_2d(C)) @
                      (inv(np.eye(N * N) - gATA) @ (gATA_a - gATA_b)) @
                      np.kron(np.atleast_2d(C).T, np.eye(N))
                      )



def covariance_matrix_limited_sum(A, C, gamma, a, b, delta):
    raise NotImplementedError("limited_sum is not implemented yet.")