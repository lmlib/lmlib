# -*- coding: utf-8 -*-
# Author: Waldmann Frédéric, Wildhaber Reto
r"""
Rectangular Pulse Detection with Baseline [ex111.0]
===================================================

This example demonstrates the detection of a rectangular pulses of known duration in a noisy signal with baseline
interferences.

"""
import matplotlib.pyplot as plt
import numpy as np
from lmlib.utils.generator import gen_rand_pulse, gen_wgn, gen_rand_walk
from scipy.signal import find_peaks

import lmlib as lm

# --------------- parameters of example -----------------------
K = 4000  # number of samples (length of test signal)
k = np.arange(K)
len_pulse = 20  # [samples] number of samples of the pulse width
y_rpulse = 0.03*gen_rand_pulse(K, n_pulses=6, length=len_pulse, seed=1000)
y = y_rpulse + gen_wgn(K, sigma=0.01, seed=1000) + 1e-3*gen_rand_walk(K)

LCR_THD = 0.2  # minimum log-cost ratio to detect a pulse in noise

g_sp = 15000  # pulse window weight, effective sample number under the window # (larger value lead to a more rectangular-like windows while too large values might lead to nummerical instabilities in the recursive computations.)
g_bl = 50  # baseline window weight, effective sample number under the window (larger value leads to a wider window)

# --------------- main -----------------------

# Defining ALSSM models
alssm_pulse = lm.AlssmPoly(poly_degree=0, label="line-model-pulse")
alssm_baseline = lm.AlssmPoly(poly_degree=2, label="offset-model-baseline")

# Defining segments with a left- resp. right-sided decaying window and a center segment with nearly rectangular window
segmentL = lm.Segment(a=-np.inf, b=-1, direction=lm.FORWARD, g=g_bl)
segmentC = lm.Segment(a=0, b=len_pulse, direction=lm.FORWARD, g=g_sp)
segmentR = lm.Segment(a=len_pulse+1, b=np.inf, direction=lm.BACKWARD, g=g_bl, delta=len_pulse)

# Defining the final cost function (a so called composite cost = CCost)
# mapping matrix between models and segments (rows = models, columns = segments)
F = [[0, 1, 0],
     [1, 1, 1]]
costs = lm.CompositeCost((alssm_pulse, alssm_baseline), (segmentL, segmentC, segmentR), F)

# filter signal
rls = lm.RLSAlssm(costs)
xs_1 = rls.filter_minimize_x(y)
y_hat = costs.eval_alssm_output(xs_1, alssm_weights=[1, 0])

xs_0 = np.copy(xs_1)
xs_0[:, costs.get_state_var_indices('line-model-pulse.x')] = 0

J1 = rls.eval_errors(xs_1)  # get SE (squared error) for hypothesis 1 (baseline + pulse)
J0 = rls.eval_errors(xs_0)  # get SE (squared error)  for hypothesis 0 (baseline only)

lcr = -0.5 * np.log(J1 / J0)

# find peaks
peaks, _ = find_peaks(lcr, height=LCR_THD, distance=30)


# --------------- plotting of results -----------------------

fig, axs = plt.subplots(5, 1, sharex='all', figsize=(9,6))

fig.subplots_adjust(hspace=0.1)

if peaks.size != 0:
    wins = lm.map_windows(costs.windows(segment_indices=[0,1,2], thd=0.001), peaks, K, merge_ks=True)
    axs[0].plot(k, wins[0], color='k', lw=0.75, ls='--', label=r"$\alpha_k(.)$")
    axs[0].plot(k, wins[1], color='k', lw=0.75, ls='-', label=r"$\alpha_k(.)$")
    axs[0].plot(k, wins[2], color='k', lw=0.75, ls=':', label=r"$\alpha_k(.)$")
axs[0].set(ylabel='windows')
axs[0].legend(loc='upper right')

axs[1].plot(k, y_rpulse, color="k", lw=1.5, linestyle="-", label='true pulses')
axs[1].set_ylim(bottom=-.1,top=.1)
axs[1].legend(loc='upper right')

axs[2].plot(k, y, color="grey", lw=0.25, label='$y$')
axs[2].plot(peaks, y[peaks], "s", color='b', fillstyle='none', markersize=15, markeredgewidth=1.0, lw=1.0, label='detected pulses')
axs[2].plot(k, xs_1[:,costs.get_state_var_indices('offset-model-baseline.x')[0]],
               '-', lw=1.0, color='k', label=r'(baseline)')
axs[2].legend(loc='upper right')

axs[3].plot(k, lcr, lw=1.0, color='k', label=r"$LCR = J(\hat{\lambda}_k) / J(0)$")
axs[3].scatter(peaks, lcr[peaks], marker=7, c='b')
axs[3].axhline(LCR_THD, color="black", linestyle="--", lw=1.0)
axs[3].legend(loc='upper right')

axs[4].plot(k, xs_1[:,costs.get_state_var_indices('line-model-pulse.x')], '-', lw=.5, color='gray', label=r'$\hat{\lambda}_{k}$ (pulse amplitudes estimates)')
axs[4].plot(peaks, xs_1[peaks,  costs.get_state_var_indices('line-model-pulse.x')], "o", color='b', markersize=4, markeredgewidth=1.0)
axs[4].axhline(0, color="black",  lw=0.5)
axs[4].legend(loc='upper right')
axs[4].set(xlabel='time index $k$')

plt.show()
