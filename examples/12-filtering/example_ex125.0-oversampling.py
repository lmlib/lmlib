"""
Oversampling [ex125.0]
======================

Oversampling Signals

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
segment = lm.Segment(a=-20, b=20, direction=lm.FW, g=1000, delta=20)
cost = lm.CostSegment(alssm, segment)

rls = lm.RLSAlssm(cost)
xs = rls.filter_minimize_x(y_sparse, v)

y_os = cost.eval_alssm_output(xs)

plt.scatter(k_sparse, y, s=20, edgecolors='k', marker='o', linewidths=0.3, facecolor='none', label='original signal')
plt.plot(k_os, y_os, 'b-', label='oversampled signal')
plt.legend()
plt.xlabel('k')
plt.title('Oversampling')
plt.show()
