"""
Multi-Channel Symmetric Signal Filter [ex123.0]
===============================================

Applies a Composite Cost with a symmetric window as a symmetric, linear filter to a multi-channel signal.


"""

import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from lmlib.utils.generator import gen_rand_walk

lm.set_backend('lfilter')
# --- Generating test signal ---
K = 1000
seeds = [130, 150, 200]
y = np.column_stack([gen_rand_walk(K, seed=s) for s in seeds])


# --- ALSSM Filtering ---
# Polynomial ALSSM
alssm_poly = lm.AlssmPoly(poly_degree=5)

# Segments
segment_left = lm.Segment(a=-50, b=-1, direction=lm.FW, g=10)
segment_right = lm.Segment(a=0, b=50, direction=lm.BW, g=10)

# Composite Cost
costs = lm.CompositeCost((alssm_poly,), (segment_left, segment_right), F=[[1, 1]])

# filter signal and take the approximation
rls = lm.RLSAlssm(costs, steady_state=False)

# extracts filtered signals
y_hat = rls.fit(y)

# --- Plotting ----
fig, axs = plt.subplots(len(seeds), 1, sharex='all')
for m, ax in enumerate(axs):
    ax.plot(y[:, m], lw=0.6, c='gray', label=r'$y_{}$'.format(m))
    ax.plot(y_hat[:,  m], lw=1, label=r'$\hat{{y}}_{}$'.format(m))
    ax.legend(loc='upper right')

axs[-1].set_xlabel('k')

plt.show()
