"""
Time and Amplitude Scaling [ex603.0]
====================================

This example is published in [Wildhaber2020]_ .

"""

import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from scipy.linalg import cholesky
from scipy.signal import find_peaks
from lmlib.utils import load_csv_mc

def time_amplitude_scaling(Q, a, b):
    q = np.arange(Q)
    Mq = lm.poly_square_expo(q)
    Mqp1 = lm.poly_int_expo(Mq)
    L_int = lm.poly_int_coef_L(Mq)
    L_eta = lm.mpoly_dilate_ind_coef_L(q)  # matrix operation of vdiag
    R_QQ = lm.permutation_matrix_square(Q, Q)

    I_Q = np.eye(Q)
    I_QQ = np.eye(Q ** 2)

    c1 = L_int
    c2 = np.kron(I_Q, L_int).dot(np.kron(L_eta, I_Q))
    c3 = np.kron(I_QQ, L_int).dot(R_QQ).dot(np.kron(L_eta, L_eta))

    bmaT = (np.power(b, Mqp1)-np.power(a, Mqp1)).T
    A = bmaT@c1
    B = np.kron(I_Q, bmaT)@c2
    C = np.kron(I_QQ, bmaT)@c3

    return A, B, C, q, Mq


y = load_csv_mc('time_amplitude_scalung_data.csv')
fs = 2000
K = len(y)
Q = 4
sp_len = int((40e-3*fs)/2)
ENERGY_THD = 0.6
k_alpha = 2200 # fixing template
etas =  np.concatenate([np.flip(np.logspace(np.log10(1), np.log10(0.5), 100)), np.logspace(np.log10(1), np.log10(2), 100)])
alssm = lm.AlssmPoly(poly_degree=Q-1)
segment_right = lm.Segment(a=0, b=sp_len, direction=lm.BW, g=50)
cost = lm.CostSegment(alssm, segment_right)
rls = lm.RLSAlssmSetSteadyState(cost)
xs = rls.filter_minimize_x(y)

A, B, C, q, Mq = time_amplitude_scaling(Q, a=int(0.0*sp_len), b=int(1*sp_len))

alphas = xs[k_alpha]
J = np.full(K, np.inf)
time_scaling_hat = np.full(K, np.nan)
amplitude_hat = np.full(K, np.nan)
for k in range(K):
    beta_beta = np.kron(xs[k, :, 0],  xs[k, :, 0]) + np.kron(xs[k, :, 1],  xs[k, :, 1])
    alpha_beta = np.kron(alphas[..., 0], xs[k, :, 0]) + np.kron(alphas[..., 1], xs[k, :, 1])
    alpha_alpha = np.kron(alphas[..., 0], alphas[..., 0]) + np.kron(alphas[..., 1],alphas[..., 1])

    a1 = A@beta_beta
    for eta in etas:
        J_ = a1 -(((B@alpha_beta).T@np.power(eta, q))**2) / ((C@alpha_alpha).T@np.power(eta, Mq))
        if J_ < J[k]:
            J[k] = J_
            time_scaling_hat[k] = eta
            amplitude_hat[k] = ((B@alpha_beta).T@np.power(eta, q)) / ((C@alpha_alpha).T@np.power(eta, Mq))


trajs = lm.map_trajectories(cost.trajectories(xs[[k_alpha]]), [k_alpha], K, True, True)

fig, axs = plt.subplots(4, sharex='all')
axs[0].plot(y+2*np.arange(2))
axs[0].plot(trajs+2*np.arange(2))
axs[1].plot(time_scaling_hat)
axs[2].plot(amplitude_hat)
axs[3].plot(J)
plt.show()