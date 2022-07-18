"""
ECG Shape Detection [ex113.0]
=======================================


"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.linalg import block_diag

import lmlib as lm
from lmlib.utils.generator import load_lib_csv

# --------------- loading test signal -----------------------
file_name = 'EECG_BASELINE_1CH_10S_FS2400HZ.csv'
K = 10000  # number of samples to process
y = load_lib_csv(file_name, K)

# --------------- parameters of example -----------------------
LCR_THD = 0.15  # minimum log-cost ratio to detect a pulse in noise

K_REF = 1865  # time index of reference shape

SHAPE_LEN_2 = 100  # 1/2 length of shape to be found
MIN_DIST = 1500  # minimal number of samples between two pulses

g_sp = 4000  # pulse window weight, effective sample number under the window # (larger value lead to a more rectangular-like windows while too large values might lead to nummerical instabilities in the recursive computations.)
g_bl = 250  # baseline window weight, effective sample number under the window (larger value leads to a wider window)

N1 = 3  # number of polynomial coefficient for baseline  Note: N>4 might leads to nummerical instabilities in the recursions
N2 = 5  # number of polynomial coefficient for spike. Note: N>4 might leads to  nummerical instabilities in the recursions.

# --------------- main -----------------------

# Defining ALSSM models
alssm_baseline = lm.AlssmPoly(poly_degree=N1 - 1, label="alssm-baseline")
alssm_pulse = lm.AlssmPolyJordan(poly_degree=N2 - 1, label="alssm-pulse")

# Defining segments with a left- resp. right-sided decaying window and a center segment with nearly rectangular window
segmentL = lm.Segment(a=-np.infty, b=-1 - SHAPE_LEN_2, direction=lm.FORWARD, g=g_bl, delta=-1 - SHAPE_LEN_2,
                      label='left-segment')
segmentC = lm.Segment(a=-SHAPE_LEN_2, b=SHAPE_LEN_2, direction=lm.FORWARD, g=g_sp, label='center-segment')
segmentR = lm.Segment(a=SHAPE_LEN_2 + 1, b=np.infty, direction=lm.BACKWARD, g=g_bl, delta=SHAPE_LEN_2 + 1,
                      label='right-segment')

# Defining the final cost function (a so called composite cost = CCost)
# mapping matrix between models and segments (rows = models, columns = segments)
F = [[0, 1, 0],
     [1, 1, 1]]
ccost = lm.CompositeCost((alssm_pulse, alssm_baseline), (segmentL, segmentC, segmentR), F)
print(ccost)

# filter signal
se_param = lm.RLSAlssm(ccost)
se_param.filter(y)  # run recursions

xs = se_param.minimize_x()  # unconstrained minimization
xs_ref = xs[K_REF]  # store state variables as reference pulse shape

H_A = np.transpose(block_diag([xs_ref[0:N2]], np.eye(N1)))  # constrain matrix to find pulses of same shape as the reference pulse
H_0 = np.transpose(np.hstack([np.zeros((N1, N2)), np.eye(N1)]))  # constrain matrix to test for no pulse (baseline only)

print("H_A : ", H_A)
print("H_0 : ", H_0)

xs_A = se_param.minimize_x(H_A)
xs_0 = se_param.minimize_x(H_0)

J_A = se_param.eval_errors(xs_A)  # get SE (squared error) for hypothesis 1 (baseline + pulse)
J_0 = se_param.eval_errors(xs_0)  # get SE (squared error)  for hypothesis 0 (baseline only) --> J0 should be a vector not a matrice

lcr = -0.5 * np.log(J_A / J_0)  # log-cost ratio computation

vs = se_param.minimize_v(H_A)
amp = vs[:, 0]

# find peaks
peaks, _ = find_peaks(lcr, height=LCR_THD, distance=MIN_DIST)

# --------------- plotting of results -----------------------
k = np.arange(K)

# Trajectories
trajs_baseline = lm.map_trajectories(ccost.trajectories(xs_A[peaks], F=[[0, 0, 0], [1, 1, 1]], thd=0.01), peaks, K,
                                     merge_ks=True, merge_seg=True)
trajs_pulse = lm.map_trajectories(ccost.trajectories(xs_A[peaks], F=[[0, 1, 0], [1, 1, 1]], thd=0.01), peaks, K,
                                  merge_ks=True, merge_seg=True)

fig, axs = plt.subplots(6, 1, sharex='all', figsize=(8, 6))

# Remove horizontal space between axes, maximize use of plotting pane
fig.tight_layout()
fig.subplots_adjust(hspace=0.0, left=0.08, bottom=0.05)

if peaks.size != 0:
    wins = lm.map_windows(ccost.windows(segment_indices=[0, 1, 2], thd=0.001), peaks, K, merge_ks=True,
                         fill_value=0)
    axs[0].plot(k, wins[0], color='k', lw=1.0, ls='--', label=segmentL.label)
    axs[0].plot(k, wins[1], color='k', lw=1.0, ls='-', label=segmentC.label)
    axs[0].plot(k, wins[2], color='k', lw=1.0, ls=':', label=segmentR.label)
    
axs[0].set(ylabel='windows')
axs[0].legend(loc='upper right')

axs[1].plot(k, y, color="gray", lw=1.0, label='y')
axs[1].axvline(x=K_REF, color='black', ls='--')
axs[1].text(K_REF, y[K_REF], ' K_REF', horizontalalignment='right', rotation=90)
axs[1].legend(loc='upper right')

axs[2].plot(k, y, color="gray", lw=1.0, label='y')
axs[2].plot(range(K), trajs_pulse, color='b', lw=1.0, linestyle="-", label='"pulses"')
axs[2].plot(range(K),trajs_baseline , color='k', lw=1.0, linestyle="-", label='"baseline"')
axs[2].legend(loc='upper right')

axs[3].plot(k, J_A, lw=1.0, color='blue', label=r"$J(x_A)$")
axs[3].plot(k, J_0, lw=1.0, color='black', label=r"$J(x_0)$")
axs[3].legend(loc='upper right')

axs[4].plot(k, lcr, lw=1.0, color='black', label=r"$LCR = -.5 ln(J(\hat{x}_A) / J(\hat{x}_0))$")
axs[4].scatter(peaks, lcr[peaks], marker=7, c='b')
axs[4].axhline(LCR_THD, color="black", linestyle="--", lw=1.0)
axs[4].set_ylim(0.0, 0.4)
axs[4].legend(loc='upper right')

axs[5].plot(k, amp, lw=1.0, color='gray', label=r'$\hat{\lambda}_{k}$')
_, stemlines, _ = axs[5].stem(peaks, amp[peaks], markerfmt="bo", basefmt=" ", use_line_collection=True)
plt.setp(stemlines, 'linewidth', 2, 'color', 'blue')
axs[5].axhline(1.0, color="black", linestyle="--", lw=1.0)
axs[5].axhline(0, color="black", lw=0.5)
axs[5].legend(loc='upper right')
axs[5].set_ylim(0.5, 1.4)
axs[5].set(xlabel='time index $k$')

plt.show()
