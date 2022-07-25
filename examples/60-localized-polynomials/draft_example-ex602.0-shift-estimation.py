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
    # the multiplication on the A, B, C matrices creates numerical issues , so in this from it's not applicable for high order systems
    # assert False
    q = np.arange(Q)
    L_shift = lm.mpoly_shift_coef_L(q)  # Eq. 6.52
    q_shift, _ = lm.mpoly_shift_expos(q)  # Eq, 6.51
    Lt = np.diag(np.kron(1 ** q_shift, (-0.5) ** q_shift)) @ L_shift  # Eq. 6.54
    Bt = np.diag(np.kron(1 ** q_shift, 0.5 ** q_shift)) @ L_shift  # Eq. 6.54

    R = lm.mpoly_square_coef_L((q_shift, q_shift))  # Eq. 6.103
    qp, _ = lm.mpoly_square_expos((q_shift, q_shift))
    L_def_int = lm.mpoly_def_int_coef_L((qp, qp), 0, a, b)  # Eq. 6.61, Eq. 6.62
    Ct = L_def_int @ R  #
    Kt = np.eye(Bt.shape[1] ** 4) + lm.permutation_matrix_square(Bt.shape[1], Lt.shape[1])  # Eq. 6.115
    A = Ct @ np.kron(Lt, Lt)
    B = Ct @ Kt @ np.kron(Lt, Bt)
    C = Ct @ np.kron(Bt, Bt)
    return A, B, C, qp


def const_shift_estimations2(Q, a, b):
    q = np.arange(Q)
    L_shift = lm.mpoly_shift_coef_L(q)  # Eq. 6.52
    q_shift, _ = lm.mpoly_shift_expos(q)  # Eq, 6.51
    Lt = np.diag(np.kron(1 ** q_shift, (-0.5) ** q_shift)) @ L_shift  # Eq. 6.54
    Bt = np.diag(np.kron(1 ** q_shift, 0.5 ** q_shift)) @ L_shift  # Eq. 6.54

    R = lm.mpoly_square_coef_L((q_shift, q_shift))  # Eq. 6.103
    qp, _ = lm.mpoly_square_expos((q_shift, q_shift))
    L_def_int = lm.mpoly_def_int_coef_L((qp, qp), 0, a, b)  # Eq. 6.61, Eq. 6.62
    Ct = L_def_int @ R  #
    Kt = np.eye(Bt.shape[1] ** 4) + lm.permutation_matrix_square(Bt.shape[1], Lt.shape[1])  # Eq. 6.115
    A = Ct @ np.kron(Lt, Lt)
    B = Ct @ Kt @ np.kron(Lt, Bt)
    C = Ct @ np.kron(Bt, Bt)
    return Lt, Bt, Ct, qp


def poly_newton(alphaD, qD, alphaDD, qDD, x0, min_step):
    cur_x = np.array(x0).astype('float').copy()
    step = float('inf')
    iter= 0
    while step >= min_step and iter < 100:
        iter +=1
        prev_x = cur_x.copy()
        delta_x = (alphaD.T @ (prev_x ** qD)) / (alphaDD.T @ (prev_x ** qDD))
        step = (alphaD.T @ prev_x ** qD) * delta_x
        cur_x = prev_x - delta_x
    # print(iter)
    return cur_x


y = load_csv_mc('shift_estimation_data.csv')
true_shift = .52e-3 # seconds
# true_shift = .1e-3 # seconds
# true_shift = 1
fs = 44100

K = len(y)
t = np.arange(K)/fs
f = 250
# y = np.column_stack([np.sin(2*np.pi*f*(t-true_shift/2)), np.sin(2*np.pi*f*(t+true_shift/2))])
# y *= np.column_stack([np.linspace(0, 2, K), np.linspace(0, 2, K)])
# y +=  0.05*np.random.randn(K, 2)


method_new = True # numerical stable
# plt.plot(y)
# plt.show()

alssm = lm.AlssmPoly(poly_degree=3)
segment_left = lm.Segment(a=-80, b=-1, direction=lm.FW, g=600)
segment_right = lm.Segment(a=0, b=80-1, direction=lm.BW, g=600)
cost = lm.CompositeCost([alssm], [segment_left, segment_right], F=[[1, 1]])
rls = lm.RLSAlssmSetSteadyState(cost)
xs = rls.filter_minimize_x(y)

a = segment_left.a * 0.8
b = segment_right.b * 0.8
k_span = np.arange(-100, 101, 1)
# k_span = np.array([0])

A, B, C, q = const_shift_estimations(alssm.N, a, b)
Lt, Bt, Ct, q = const_shift_estimations2(alssm.N, a, b)
# K_red = lm.mpoly_remove_redundancy((q,))
Ld = lm.poly_diff_coef_L(q)
qd = lm.poly_diff_expo(q)
Ldd = lm.poly_diff_coef_L(qd)@Ld
qdd = lm.poly_diff_expo(qd)


shifts_hat = np.zeros(K)
shifts_mov_ave = np.zeros(K)
shift_range = np.linspace(-1.2*true_shift, 1.2*true_shift, 1000)
# shift_range = np.linspace(-0.2*true_shift, 0.2*true_shift, 21)
# shift_range = [0]
Js = np.full(K, np.nan)
Js_mov = np.full(K, np.nan)
for k0 in range(K):
    print(k0)
    alpha = xs[k0, :, 0]
    # beta = lm.poly_shift_coef_L(np.arange(alssm.N), true_shift)@alpha
    # beta = alpha
    beta = xs[k0, :, 1]
    if method_new:
        alphas_mov = (Ct @ np.kron(Lt @ alpha - Bt @ beta, Lt @ alpha - Bt @ beta))
    else:
        alphas_mov = (A @ np.kron(alpha, alpha) - B @ np.kron(alpha, beta) + C @ np.kron(beta, beta))

    # traj_J2mov = np.array([alphas_mov.T @ s ** q for s in shift_range])
    # shifts_hat[k0] = shift_range[np.nanargmin(traj_J2)]
    shifts_mov_ave[k0] = poly_newton(Ld @ alphas_mov, qd, Ldd @ alphas_mov, qdd, shifts_mov_ave[k0 - 1], min_step=1e-12)
    Js_mov[k0] = alphas_mov.T @ shifts_mov_ave[k0] ** q

    alphas = np.zeros(Ct.shape[0])

    for k in np.unique(np.clip(k_span + k0, 0, K-1)):
        alpha = xs[k, :, 0]
        # beta = lm.poly_shift_coef_L(np.arange(alssm.N), true_shift)@alpha
        # beta = alpha
        beta = xs[k, :, 1]
        if method_new:
            alphas += (Ct@np.kron(Lt@alpha-Bt@beta, Lt@alpha-Bt@beta))
        else:
            alphas += (A @ np.kron(alpha, alpha) - B @ np.kron(alpha, beta) + C @ np.kron(beta, beta))

        # traj_J2 = np.array([alphas.T @ s ** q for s in shift_range])
        # shifts_hat[k0] = shift_range[np.nanargmin(traj_J2)]
        shifts_hat[k0] = poly_newton(Ld@alphas, qd, Ldd@alphas, qdd, shifts_hat[k0-1], min_step=1e-12)
        Js[k0] = alphas.T @ shifts_hat[k0] ** q

        # shifts_mov_ave[k0] = shifts_hat[k0]*0.05 + shifts_mov_ave[k0-1]*0.95


# plot
ks = [2747, 2997]
trajs = lm.map_trajectories(cost.trajectories(xs[ks]), ks, K, True, True)
fig, (ax1, ax11, ax2, ax3) = plt.subplots(4, sharex='all')
ax1.plot(y[:, 0],c=(0.5, 0.5, 0.5), lw=1, label='channel 1')
ax1.plot(y[:, 1],c=(0.1, 0.1, 0.1), lw=1, label='channel 2')
ax1.plot(trajs, label='trajectories')
ax1.legend(loc=1, fontsize=8)

print(np.median(shifts_hat), np.median(shifts_hat)/fs)
k_corr_ch1 = np.clip(np.arange(K)-int(np.median(shifts_hat)/2), 0, K-1)
k_corr_ch2 = np.clip(np.arange(K)+int(np.median(shifts_hat)/2), 0, K-1)
ax11.plot(y[k_corr_ch1, 0], c='b', ls='--',lw=1, label='shifted channel 1')
ax11.plot(y[k_corr_ch2, 1], c='r', ls='--',lw=1, label='shifted channel 2')
ax11.legend(loc=1, fontsize=8)

# ax2.axhline(true_shift, c='k', ls='--', lw=0.8)
ax2.axhline(-true_shift, c='k', ls='--', lw=0.8, label='expected shift')
ax2.plot(shifts_hat/fs, label=r'estimated shift $\hat{s}_k$')
ax2.plot(shifts_mov_ave/fs, label=r'estimated shift $\bar{s}_k$')
ax2.legend(loc=1, fontsize=8)

# ax2.plot(-2*shifts_mov_ave/fs)
# ax2.set_ylim([-2*true_shift, true_shift*2])

ax3.plot(Js, label=r'$J(\hat{s}_k)=$')
ax3.plot(Js_mov, label=r'$J(\bar{s}_k)=$')
ax3.legend(loc=1, fontsize=8)

plt.show()
