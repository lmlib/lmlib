"""
ALSSMs Transformation [ex126.0]
===============================

Transformed ALSSM removes the need for a minimization, which is shown in the example.
"""

import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from lmlib.utils.generator import gen_rect
from lmlib.statespace.backends.statespace_tools import _transform_ALSSM_matrices, _transform_x

# --- Generating test signal ---
K = 2000
k = np.arange(K)
y = gen_rect(K, 500,250)
# y= y[:, None]

# Polynomial ALSSM
alssm_poly = lm.AlssmPoly(poly_degree=0, force_MC=False)

# --- Plot I: ALSSM Filtering - Finite Support Moving Average Filter ---
# Segments
segment_left = lm.Segment(a=-50, b=-1, direction=lm.FORWARD, g=1000)
segment_right = lm.Segment(a=0, b=49, direction=lm.BACKWARD, g=1000)

# Composite Cost
costs = lm.CompositeCost((alssm_poly,), (segment_left, segment_right), F=[[1, 1]])

P = np.linalg.cholesky(costs.get_steady_state_W()).T
_A,_C = _transform_ALSSM_matrices(alssm_poly.A, alssm_poly.C, P)
alssm_trans = lm.Alssm(_A, _C)

costs_trans = lm.CompositeCost((alssm_trans,), (segment_left, segment_right), F=[[1, 1]])


# filter signal and take the approximation
rls = lm.RLSAlssm(costs, backend='numpy')
y_hat = rls.filter_minimize_yhat(y)

rls_trans = lm.RLSAlssm(costs_trans)
rls_trans.filter(y)
y_hat_trans = costs_trans.eval_alssm_output(rls_trans.xi)

# --- Plotting ----
fig, ax = plt.subplots(1, sharex='all', figsize=(8,3))

ax.plot(k, y, lw=0.6, c='gray', label=rf'$y$')
ax.plot(k, y_hat, lw=1, c='g', label=r'$\hat{y}_n$')
ax.plot(k, y_hat_trans, lw=1, ls = '--', c='r', label=r'$\hat{y}_t$')
ax.legend(loc='upper right')
ax.set_title('Calculation with and without Transformation/Minimization')

plt.show()



