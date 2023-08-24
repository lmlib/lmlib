"""
Multi-Channel Spike Detection [ex112.0]
=======================================

This example shows a spike detection algorithm that uses autonomous linear state space models together with
exponentially decaying windows. Given is a multi-channel signal containing multiple spikes
(sinusoidal cycle with decaying amplitude) with additive white Gaussian noise and a baseline.

"""

import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from scipy.linalg import block_diag
from scipy.signal import find_peaks

from lmlib.utils.generator import gen_conv, gen_sine, gen_exp, gen_pulse, gen_wgn, k_period_to_omega

# signal generation
K = 550
L = 3  # number of channels
spike_length = 20
spike_decay = 0.88
spike_locations = [100, 240, 370]
spike = gen_sine(spike_length, spike_length) * gen_exp(spike_length, spike_decay)
y_sp = gen_conv(gen_pulse(K, spike_locations), spike)
y = np.column_stack([0.8*y_sp + gen_wgn(K, sigma=0.2, seed=10000-l) for l in range(L)]).reshape(K, L)

# Model
alssm_sp = lm.AlssmSin(k_period_to_omega(spike_length), spike_decay)
alssm_bl = lm.AlssmPoly(poly_degree=3)

# Segments
g_bl = 500
g_sp = 5000
len_sp = spike_length
len_bl = int(1.5*spike_length)
segment_left = lm.Segment(a=-len_bl, b=-1, direction=lm.FORWARD, g=g_bl, delta=-1)
segment_middle = lm.Segment(a=0, b=len_sp, direction=lm.BACKWARD, g=g_sp)
segment_right = lm.Segment(a=len_sp+1, b=len_sp+1+len_bl, direction=lm.BACKWARD, g=g_bl, delta=len_sp)

# Cost
F = [[0, 1, 0],
     [1, 1, 1]]
cost = lm.CompositeCost((alssm_sp, alssm_bl), (segment_left, segment_middle, segment_right), F)

rls = lm.RLSAlssmSet(cost)
rls.filter(y)
H_sp = block_diag([[0], [1]], np.eye(alssm_bl.N))
xs_sp = rls.minimize_x(H_sp)
H_bl = np.vstack([np.zeros((alssm_sp.N, alssm_bl.N)), np.eye(alssm_bl.N)])
xs_bl = rls.minimize_x(H_bl)

# Error
J = rls.eval_errors(xs_sp)
J_bl = rls.eval_errors(xs_bl)
J_sum = np.sum(J, axis=-1)
J_bl_sum = np.sum(J_bl, axis=-1)

lcr = -0.5 * np.log(J_sum / J_bl_sum)

peaks, _ = find_peaks(lcr, height=0.041, distance=30)

# Plot
fig, axs = plt.subplots(5, 1, figsize=(9, 8), gridspec_kw={'height_ratios': [1, 1, 3, 1, 1]}, sharex='all')

# Window
wins = lm.map_windows(cost.windows(segment_indices=[0,1,2]), peaks, K, merge_ks=True)

# Trajectories
trajs_baseline = lm.map_trajectories(cost.trajectories(xs_sp[peaks], F=[[0, 0, 0], [1, 1, 1]], thd=0.01), peaks, K,
                                     merge_ks=True, merge_seg=True)
trajs_pulse = lm.map_trajectories(cost.trajectories(xs_sp[peaks], F=[[0, 1, 0], [1, 1, 1]], thd=0.01), peaks, K,
                                  merge_ks=True, merge_seg=True)

axs[0].set(ylabel='$w_k$')
axs[0].plot(range(K), wins[0], lw=1, c='k', ls='--')
axs[0].plot(range(K), wins[1], lw=1, c='k', ls='-')
axs[0].plot(range(K), wins[2], lw=1, c='k', ls=':')
axs[0].legend(('segm. 1 ("baseline")','segm. 2 ("pulse"+"baseline")', 'segm. 3 ("baseline")'), loc=1, fontsize='small')

# True Signals
axs[1].plot(range(K), y_sp, c='b', lw=1.0)
axs[1].set_ylim(-0.5, 2)
axs[1].legend(('true spikes',), loc=1)

# Signals
OFFSETS = [0, 2, 4]
axs[2].set(ylabel='$y_k$')
axs[2].plot(range(K), y + OFFSETS, c='tab:gray', lw=1)
axs[2].plot(range(K), trajs_pulse + OFFSETS, color='b', lw=1.5, linestyle="-")
axs[2].plot(range(K), trajs_baseline + OFFSETS , color='k', lw=1.5, linestyle="-")
axs[2].legend(('$y$','_','_','"pulses"','_','_','"baseline"','_','_',), loc=1)

# LCR
axs[3].set(ylabel='LCR', ylim=[0, 0.15])
axs[3].plot(range(K), lcr, c='k', lw=1.0, label='LCR')
axs[3].scatter(peaks, lcr[peaks], marker=7, c='b')
axs[3].legend(loc=1)

# Error
axs[4].set(ylabel='$J_k$', xlabel='$k$')
axs[4].plot(range(K), J_sum, c='b', lw=1.0, label='$SE($'+'"baseline"'+'$)$')
axs[4].plot(range(K), J_bl_sum, c='k', lw=1.0, label='$SE($'+'"pulse"+"baseline"'+'$)$')
axs[4].legend(loc=1)

plt.show()
