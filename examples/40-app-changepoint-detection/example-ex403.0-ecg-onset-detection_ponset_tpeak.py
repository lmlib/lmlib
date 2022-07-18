"""
ECG P,T Wave Onset and Peak Detection [ex403.0]
====================================================
Example 3 published in [Waldmann2022]_.

Another common task in cardiology is to accurately measure onsets and peaks of P and T waves in the electrocardiography (ECG) signals.
We here demonstrate a Two-Sided Line Model (TSLM) to extract such features. 

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import lmlib as lm
from lmlib.utils.generator import gen_wgn, load_csv


# Linear Constraints
H_Free = np.array([[1, 0, 0, 0],  # x_A,left : offset of left line
                   [0, 1, 0, 0],  # x_B,left : slope of left line
                   [0, 0, 1, 0],  # x_A,right : offset of right line
                   [0, 0, 0, 1]]) # x_B,right : slope of right line

H_Continuous = np.array(
                  [[1, 0, 0],  # x_A,left : offset of left line
                   [0, 1, 0],  # x_B,left : slope of left line
                   [1, 0, 0],  # x_A,right : offset of right line
                   [0, 0, 1]])  # x_B,right : slope of right line


H_Straight = np.array(
                  [[1, 0],  # x_A,left : offset of left line
                   [0, 1],  # x_B,left : slope of left line
                   [1, 0],  # x_A,right : offset of right line
                   [0, 1]])  # x_B,right : slope of right line

H_Horizontal = np.array(
                  [[1],  # x_A,left : offset of left line
                   [0],  # x_B,left : slope of left line
                   [1],  # x_A,right : offset of right line
                   [0]])  # x_B,right : slope of right line

H_Left_Horizontal = np.array(
                  [[1, 0],  # x_A,left : offset of left line
                   [0, 0],  # x_B,left : slope of left line
                   [1, 0],  # x_A,right : offset of right line
                   [0, 1]])  # x_B,right : slope of right line

H_Right_Horizontal = np.array(
                  [[1, 0],  # x_A,left : offset of left line
                   [0, 1],  # x_B,left : slope of left line
                   [1, 0],  # x_A,right : offset of right line
                   [0, 0]])  # x_B,right : slope of right line

H_Peak = np.array([[1, 0],  # x_A,left : offset of left line
                   [0, 1],  # x_B,left : slope of left line
                   [1, 0],  # x_A,right : offset of right line
                   [0, -1]])  # x_B,right : slope of right line

H_Step = np.array([[1, 0],  # x_A,left : offset of left line
                   [0, 0],  # x_B,left : slope of left line
                   [0, 1],  # x_A,right : offset of right line
                   [0, 0]])  # x_B,right : slope of right line



# Implementation of Two-Sided Line Model (TSLM)
# y: input signal vector
# a,b: left and right sided interval border (in number of samples)
# gl, gr: left and right sided exponential window weight (defines exponential decay factor gamma)

def J_TSLM(y, a, b, gl, gr):

    # Set up Composite Cost Model using two ALSSMs and exponentially decaying windows
    #
    #         ---->     <----
    #      --------------------
    #  A_L |   c_L  |    0    |
    #      --------------------
    #  A_R |   0    |    c_R  |
    #      --------------------
    #      :        :         :
    #     a=-80     0        b=20

    alssm_left = lm.AlssmPoly(poly_degree=1)  # A_L, c_L
    alssm_right = lm.AlssmPoly(poly_degree=1)  # A_R, c_R
    segment_left = lm.Segment(a=a, b=-1, direction=lm.FORWARD, g=gl)
    segment_right = lm.Segment(a=0, b=b, direction=lm.BACKWARD, g=gr)
    F = [[1, 0], [0, 1]]  # mixing matrix, turning on and off models per segment (1=on, 0=off)
    costs = lm.CompositeCost((alssm_left, alssm_right), (segment_left, segment_right), F)

    return costs




# Constants
K = 2200        # number of samples
sigma = 0.015   # Adding Gaussian Noise
fs = 500        # Sampling Frequency [Hz]
k = range(K)


# load ECG signal as a test signal from library
# y1_mc = load_multi_channel('EECG_FILT_9CH_A0S_FS600HZ.csv', K=K)
# y = y1_mc[0:,1] # select single channel

# load alternative ECG signal
y = load_csv('ECG_001-nohead.csv', K, channel=1)

y = y + gen_wgn(K, sigma, seed=233453)*np.concatenate((np.ones(K//2),3*np.ones(K-K//2)))





# (A) -- P wave onset Detection -----
costs_A = J_TSLM(y, -80, 40, gl=80, gr=80)

# Filter
separam_A = lm.RLSAlssm(costs_A)
separam_A.filter(y)

# constraint minimization
x_hat_H1_A = separam_A.minimize_x(H_Left_Horizontal)
x_hat_H0_A = separam_A.minimize_x(H_Horizontal)

# Square Error and LCR
error_edge_A = separam_A.eval_errors(x_hat_H1_A)
error_line_A = separam_A.eval_errors(x_hat_H0_A)
lcr_A = -1 / 2 * np.log(np.divide(error_edge_A, error_line_A))

# Find LCR peaks with minimal distance and height
# peaks_A, _ = find_peaks(lcr[0:300], height=0.3, distance=50)
peaks_A, _ = find_peaks(lcr_A, height=.25, distance=300)

# Evaluate trajectories (for plotting only)
trajs_edge_A = lm.map_trajectories(costs_A.trajectories(x_hat_H1_A[peaks_A]), peaks_A, K, merge_ks=True)
trajs_line_A = lm.map_trajectories(costs_A.trajectories(x_hat_H0_A[peaks_A]), peaks_A, K, merge_ks=True)



# (B) -- T wave peak Detection -----
costs_B = J_TSLM(y, -45, 45, gl=80, gr=80)

# Filter
separam_B = lm.RLSAlssm(costs_B)
separam_B.filter(y)

# constraint minimization
x_hat_H1_B = separam_B.minimize_x(H_Peak)
x_hat_H0_B = separam_B.minimize_x(H_Horizontal)

# Square Error and LCR
error_edge_B = separam_B.eval_errors(x_hat_H1_B)
error_line_B = separam_B.eval_errors(x_hat_H0_B)
lcr = -1 / 2 * np.log(np.divide(error_edge_B, error_line_B))

# Find LCR peaks with minimal distance and height
peaks_B, _ = find_peaks(lcr, height=.25, distance=150)

# Evaluate trajectories (for plotting only)
trajs_edge = lm.map_trajectories(costs_B.trajectories(x_hat_H1_B[peaks_B]), peaks_B, K, merge_ks=True)
trajs_line = lm.map_trajectories(costs_B.trajectories(x_hat_H0_B[peaks_B]), peaks_B, K, merge_ks=True)

wins = lm.map_windows(costs_B.windows(segment_indices=[1, 1]), peaks_B, K, merge_ks=True)




# -- PLOTTING --
_, axs = plt.subplots(5, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [1.5, 1, 0.1, 1.5, 1]}, sharex='all')
nax = 0


# ----- p onset ----- 
t = np.array(list(k))/fs

axs[nax].plot(t, y, lw=1.0, c='gray', label='$y$', zorder=0)
if True:
   axs[nax].plot(t, trajs_edge_A[0, :], c='k', lw=.75, ls='-', zorder=1, label='$\overrightarrow{s}_{i-k}(\hat x_\ell)$')
   axs[nax].plot(t, trajs_edge_A[1, :], c='b', lw=.75, ls='-', zorder=1, label='$\overleftarrow{s}_{i-k}(\hat x_r)$')
   axs[nax].scatter(peaks_A[0]/fs, x_hat_H1_A[peaks_A[0], 0], marker='.', c='k', s=20.0)

#axs[nax].axhline(x=peaks_A, ymin=plt.ylim()[0], ymax=plt.ylim()[1])
for xp in peaks_A/fs:
   axs[nax].axvline(x=xp, ls='--', c='b', lw=0.5)

#axs[nax].scatter(peaks_A, y[peaks_A], marker=7, c='b')
axs[nax].legend(loc='upper right', labelspacing = -0.0)
axs[nax].set_ylim(bottom=min(y),top=max(y))
nax+=1

axs[nax].plot(t, lcr_A, lw=1.0, c='k', label='LCR')
axs[nax].scatter(peaks_A/fs, lcr_A[peaks_A], marker=7, c='b')
axs[nax].legend(loc=1)
axs[nax].set_ylim(bottom=-0,top=2.2)


# axs[0].set(title='Edge Detection in ECG Signals')


nax+=1
axs[nax].set_visible(False)
nax+=1

t = np.array(list(k))/fs

axs[nax].plot(t, y, lw=1.0, c='gray', label='$y$', zorder=0)
if True:
   axs[nax].plot(t, trajs_edge[0, :], c='k', lw=.75, ls='-', zorder=1, label='$\overrightarrow{s}_{i-k}(\hat x_\ell)$')
   axs[nax].plot(t, trajs_edge[1, :], c='b', lw=.75, ls='-', zorder=1, label='$\overleftarrow{s}_{i-k}(\hat x_r)$')
   axs[nax].scatter(peaks_B[1]/fs, x_hat_H1_B[peaks_B[1], 0], marker='.', c='k', s=20.0)

#axs[nax].axhline(x=peaks_B, ymin=plt.ylim()[0], ymax=plt.ylim()[1])
for xp in peaks_B/fs:
   axs[nax].axvline(x=xp, ls='--', c='b', lw=0.5)

#axs[nax].scatter(peaks_B, y[peaks_B], marker=7, c='b')
axs[nax].legend(loc='upper right', labelspacing = -0.0)
axs[nax].set_ylim(bottom=min(y),top=max(y))
nax+=1

axs[nax].plot(t, lcr, lw=1.0, c='k', label='LCR')
axs[nax].scatter(peaks_B/fs, lcr[peaks_B], marker=7, c='b')
axs[nax].legend(loc=1)
axs[nax].set_ylim(bottom=0,top=2.2)

axs[nax].set_xlim(left=.4, right=4.2)

nax+=1
plt.gcf().text(0.883, 0.164, '(s)')
plt.subplots_adjust(bottom=0.21)

plt.show()