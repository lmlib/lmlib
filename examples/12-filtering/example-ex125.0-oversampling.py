"""
Oversampling [ex125.0]
======================

Oversamples a signal using an ALSSM-based polynomial interpolation.

A sparse input signal (every ``os_rate``-th sample is non-zero, set to the
original ECG sample value) is filtered with a [`CompositeCost`][lmlib.statespace.cost.CompositeCost] whose
forward and backward segments span the oversampling interval.  The ALSSM
fits a degree-2 polynomial locally around each original sample, and the
dense output is read at all oversampled indices, effectively performing
polynomial interpolation at the oversampling rate.

Signal source: ``EECG_BASELINE_1CH_10S_FS2400HZ.csv`` (bundled library data).
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
import lmlib as lm
from lmlib.utils import load_lib_csv

K = 800
y = load_lib_csv('EECG_BASELINE_1CH_10S_FS2400HZ.csv', K, k_start=1200)

os_rate = 20  # oversampling rate
K_os = K * os_rate
k_os = np.arange(0, K_os)
k_sparse = np.arange(0, K_os, os_rate)

v = np.zeros(K_os)
v[k_sparse] = 1
y_sparse = np.zeros(K_os)
y_sparse[k_sparse] = y

alssm = lm.AlssmPoly(poly_degree=2)
segments = lm.Segment(a=-20, b=-1, direction=lm.FW, g=100), lm.Segment(a=0, b=20, direction=lm.BW, g=100)
cost = lm.CompositeCost([alssm], segments, F=np.ones((1, 2)))

rls = lm.RLSAlssm(cost, steady_state=False)
y_os = rls.fit(y_sparse, sample_weights=v)

plt.scatter(k_sparse, y, s=20, edgecolors='k', marker='o', linewidths=0.3, facecolor='none', label='original signal')
plt.plot(k_os, y_os, 'b-', label='oversampled signal')
plt.legend()
plt.xlabel('k')
plt.title('Oversampling (Rate={})'.format(os_rate))
plt.show()
