"""
Audio Signal Shift Estimation [ex602.0]
=======================================
Example 2 published in [Wildhaber2020]_.

Top: two-channel acoustic signal from left (L, solid line) and right (R, dashed line) ear with interaural time delay. Bottom: sˆ shows the local time delay estimate of corresponding local polynomial fits, while s is the averaged version of sˆ using a rectangular window according to (42). The true delay (dashed black line), calculated according to head geometry and speed of sound for an azimuth of 45°, is approximately 0.52 ms.
"""

import numpy as np
import matplotlib.pyplot as plt

import lmlib as lm
from lmlib.utils import load_csv_mc


def const_shift_estimations(Q, a, b):
    q = np.arange(Q)
    L_shift = lm.mpoly_shift_coef_L(q)
    qt, _ = lm.mpoly_shift_expos(q)
    Lt = np.diag(np.kron(1 ** qt, -0.5 ** qt)) @ L_shift
    Bt = np.diag(np.kron(1 ** qt, 0.5 ** qt)) @ L_shift
    qp, _ = lm.mpoly_square_expos((qt, qt))
    Rqr = lm.mpoly_square_coef_L((qt, qt))
    Ctp = lm.mpoly_int_coef_L((qp, qp), position=1) @ Rqr
    Ct = np.kron(b ** qp - a ** qp, np.eye(len(qp))) @ Ctp
    m = Bt.shape[1]
    n = Lt.shape[1]
    K = lm.commutation_matrix(m ** 2, n ** 2)
    Kt = np.eye(len(K)) + K
    A = Ct @ np.kron(Lt, Lt)
    B = Ct @ Kt @ np.kron(Lt, Bt)
    C = Ct @ np.kron(Bt, Bt)
    return A, B, C, qp


def poly_newton(alphaD, qD, alphaDD, qDD, x0, min_step):
    cur_x = np.array(x0).astype('float').copy()
    step = float('inf')
    iter= 0
    while step >= min_step:
        iter +=1
        prev_x = cur_x.copy()
        delta_x = 1e-3 * (alphaD.T @ (prev_x ** qD)) / (alphaDD.T @ (prev_x ** qDD))
        step = (alphaD.T @ prev_x ** qD) * delta_x
        cur_x = prev_x - delta_x
    # print(iter)
    return cur_x


K = 4000
k_start = 44000
y = load_csv_mc('Female_CIPIC003_Az_30_El_0.csv', K, k_start)
fs = 44100

alssm = lm.AlssmPoly(poly_degree=4)
segment_left = lm.Segment(a=-120, b=-1, direction=lm.FW, g=800)
segment_right = lm.Segment(a=0, b=120, direction=lm.BW, g=800)
cost = lm.CompositeCost([alssm], [segment_left, segment_right], F=[[1, 1]])
rls = lm.RLSAlssmSetSteadyState(cost)
xs = rls.filter_minimize_x(y)

a = -120 * 0.8
b = 120 * 0.8
Q = 5
k_span = 100
energy_thd = 0.01

A, B, C, q = const_shift_estimations(Q, a, b)
# K_red = lm.mpoly_remove_redundancy((q,))
Ld = lm.poly_diff_coef_L(q)
qd = lm.poly_diff_expo(q)
Ldd = lm.poly_diff_coef_L(qd)@Ld
qdd = lm.poly_diff_expo(qd)


shifts_hat = np.zeros(K)
shifts_mov_ave = np.zeros(K)
for k0 in range(K):
    if np.all(rls.kappa[k0] > energy_thd):
        alphas = np.zeros(A.shape[0])
        for k in range(max(0, k0-k_span), min(k0+k_span, K)):
            alpha = xs[k, :, 0]
            beta = xs[k, :, 1]
            alphas += A@np.kron(alpha, alpha) - B@np.kron(alpha, beta) + C@A@np.kron(beta, beta)
        shifts_hat[k0] = poly_newton(Ld@alphas, qd, Ldd@alphas, qdd, 0.0, min_step=1e-12)
        shifts_mov_ave[k0] = shifts_hat[k0]*0.05 + shifts_mov_ave[k0-1]*0.95
fig, (ax1, ax2) = plt.subplots(2, sharex='all')
ax1.plot(y)
# ax2.plot(rls.kappa)
ax2.plot(shifts_hat*-1/fs*1000)
ax2.plot(shifts_mov_ave*-1/fs*1000)
plt.show()
