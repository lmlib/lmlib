"""
Symmetric Moving Average Filters with ALSSMs [ex121.0]
======================================================

Applies a Composite Cost with a two-sided, symmetric window as a symmetric moving average filter of length L=100.

"""

import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from lmlib.utils.generator import gen_rect
import time

# --- Generating test signal ---
K = 2000
k = np.arange(K)
y = gen_rect(K, 500,250)


# Polynomial ALSSM
alssm_poly = lm.AlssmPoly(poly_degree=0)

# --- Plot I: ALSSM Filtering - Finite Support Moving Average Filter ---
# Segments
segment_left = lm.Segment(a=-50, b=-1, direction=lm.FORWARD, g=1000)
segment_right = lm.Segment(a=0, b=49, direction=lm.BACKWARD, g=1000)

# CompsiteCost
costs = lm.CompositeCost((alssm_poly,), (segment_left, segment_right), F=[[1, 1]])

# filter signal and take the approximation
rls = lm.create_rls(costs, steady_state=True)
xs = rls.filter_minimize_x(y)

# extracts filtered signal
y_hat_1 = costs.eval_alssm_output(xs, alssm_weights=[1])



# --- Plot II: ALSSM Filtering - Symmetric Infinte Support Moving Average Filter ---
# Segments
segment_left = lm.Segment(a=-np.Infinity, b=-1, direction=lm.FORWARD, g=20)
segment_right = lm.Segment(a=0, b=np.Infinity, direction=lm.BACKWARD, g=20)

# CompsiteCost
costs = lm.CompositeCost((alssm_poly,), (segment_left, segment_right), F=[[1, 1]])

# filter signal and take the approximation
rls = lm.create_rls(costs, steady_state=True)
xs = rls.filter_minimize_x(y)

# extracts filtered signal
y_hat_2 = costs.eval_alssm_output(xs, alssm_weights=[1])



# --- Plotting ----
fig, ax = plt.subplots(2, 1, sharex='all', figsize=(10,6))

ax[0].plot(k, y, lw=0.6, c='gray', label=rf'$y$')
ax[0].plot(k, y_hat_1, lw=1, c='b', label=r'$\hat{y}$')
ax[0].legend(loc='upper right')
ax[0].set_title('Symmetric, Finite Support')


ax[1].plot(k, y, lw=0.6, c='gray', label=rf'$y$')
ax[1].plot(k, y_hat_2, lw=1, c='b', label=r'$\hat{y}$')
ax[1].legend(loc='upper right')
ax[1].set_xlabel('k')
ax[1].set_title('Symmetric, Infinite Support')


plt.show()
