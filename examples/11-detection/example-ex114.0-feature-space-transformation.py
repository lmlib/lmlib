"""
ECG Feature Space Transformation [ex114.0]
==========================================


"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_power
from scipy.signal import find_peaks
from scipy.linalg import block_diag

import lmlib as lm
from lmlib.utils.generator import load_lib_csv
from lmlib.utils.generator import gen_wgn, gen_sine, gen_rand_walk, gen_exp

np.random.seed(0)

# Feature space parameters
use_offset = False
draw_trajs = True
draw_clusters = True
show_dimensions_3d = [1, 2, 3]
use_pca = True

# Signal
K = 2000
ks = [40, 120, 310, 480, 610, 850, 920, 1100, 1250, 1320, 1550, 1700, 1910]
spikes_len = [60, 45, 50, 60, 45, 80, 60, 50, 81, 45, 60, 50, 45]
k0_sin = [0, -2, 5, 0, -2, 1, 0, 4, 1, -2, 0, 6, -2]
matches = [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 0, 2, 1]
if (draw_clusters):
    colors = ['b', 'g', 'r', 'orange']
else:
    colors = ['b', 'b', 'b', 'b']

NN = [1, 1.2, 1.6, 1.9]

# Model Parameters
spike_width = 30
g_spike = 50
g_baseline = 10
order_baseline = 3
order_spike = 4

y_baseline = 0.2 * gen_sine(K, K // 2) * gen_sine(K, K // 3) + 0.01 * gen_rand_walk(K, seed=0)
y_signal = np.zeros(K)
for ik, (k, s_len, k0, match) in enumerate(zip(ks, spikes_len, k0_sin, matches)):
    k0 = 0
    s_len = 60
    nn = NN[match]
    y_signal[k:k + s_len] = gen_sine(s_len, s_len / nn, 1, k0) * gen_exp(s_len, ((s_len - 1) / s_len) ** 5)
y = y_signal + 0.5 * y_baseline + gen_wgn(K, sigma=0.01, seed=0)

# Defining ALSSM models
alssm_baseline = lm.AlssmPoly(poly_degree=order_baseline - 1, label="alssm-baseline")
alssm_spike = lm.AlssmPolyJordan(poly_degree=order_spike - 1, label="alssm-pulse")

# Defining segments with a left- resp. right-sided decaying window and a center segment with nearly rectangular window
segmentL = lm.Segment(a=-np.infty, b=-1, direction=lm.FORWARD, g=g_baseline, delta=0, label='left-segment')
segmentC = lm.Segment(a=0, b=spike_width, direction=lm.FORWARD, g=g_spike, label='center-segment')
segmentR = lm.Segment(a=spike_width + 1, b=np.infty, direction=lm.BACKWARD, g=g_baseline, delta=spike_width + 1, label='right-segment')

# Defining the final cost function (a so called composite cost = CCost)
# mapping matrix between models and segments (rows = models, columns = segments)
F = [[0, 1, 0],
     [1, 1, 1]]
cost = lm.CompositeCost((alssm_spike, alssm_baseline), (segmentL, segmentC, segmentR), F)

# Filter
rls = lm.RLSAlssmSteadyState(cost)
xs_hat = rls.filter_minimize_x(y)
V = np.linalg.cholesky(rls.W).T[:order_spike, :order_spike]

xs_pulse = xs_hat[:, 0:order_spike]
zs = np.einsum('mn, kn->km', V, xs_pulse)

# Trajectories
trajs_pulse = lm.map_trajectories(cost.trajectories(xs_hat[ks], F=[[0, 1, 0], [0, 1, 0]], thd=0.001), ks, K, merge_ks=False, merge_seg=False)
trajs_baseline = lm.map_trajectories(cost.trajectories(xs_hat[ks], F=[[0, 0, 0], [1, 1, 1]], thd=0.005), ks, K, merge_ks=True, merge_seg=True)

# Plot
fig = plt.figure(figsize=(9, 5), constrained_layout=True)
plt.subplots_adjust(left=0.06, right=0.96, top=0.96, bottom=0.06)

spec = fig.add_gridspec(3, 2)
ax0 = fig.add_subplot(spec[0, :])
ax1 = fig.add_subplot(spec[1:3, 0], projection='3d')
ax2 = fig.add_subplot(spec[1:3, 1], projection='3d')

ax1.view_init(50, -70)
ax2.view_init(50, -70)

# plot signal
ax0.plot(range(K), y, lw=0.5, c='gray', label='observation')
if (draw_trajs):
    for traj, m in zip(trajs_pulse, matches):
        ax0.plot(range(K), traj[1, :], lw=1.0, c=colors[m], label='Pulse')
ax0.plot(range(K), trajs_baseline, lw=1.0, ls=':', c='black', label='Baseline')
ax0.scatter(ks, y[ks], marker=7, c='k')
ax0.set(xlabel='k', ylabel=r'$y$')

# plot 3d of x
if use_pca:
    data_x = xs_pulse[ks]
    data_x = data_x - data_x.mean(axis=0)
    cov_x = np.cov(data_x.T) / data_x.shape[0]
    v_x, w_x = np.linalg.eig(cov_x)
    idx = v_x.argsort()[::-1]
    v_x = v_x[idx]
    w_x = w_x[:, idx]
    data_x_2 = data_x.dot(w_x[:, :4])
else:
    data_x_2 = xs_pulse[ks]

for i, m in enumerate(matches):
    ax1.scatter(data_x_2[i, show_dimensions_3d[0]], data_x_2[i, show_dimensions_3d[1]],
                data_x_2[i, show_dimensions_3d[2]], facecolor=colors[m])

ax1.set(xlabel='$a_1$', ylabel=r'$a_2$', zlabel=r'$a_3$')
ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax1.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))

# plot 3d of z

if use_pca:
    data_z = zs[ks]
    data_z = data_z - data_z.mean(axis=0)
    cov_z = np.cov(data_z.T) / data_z.shape[0]
    v_z, w_z = np.linalg.eig(cov_z)
    idx = v_z.argsort()[::-1]
    v_z = v_z[idx]
    w_z = w_z[:, idx]
    data_z_2 = data_z.dot(w_z[:, :4])
else:
    data_z_2 = zs[ks]

for i, m in enumerate(matches):
    ax2.scatter(data_z_2[i, 0], data_z_2[i, 1], data_z_2[i, 2], facecolor=colors[m])

ax2.set(xlabel='$z_1$', ylabel=r'$z_2$', zlabel=r'$z_3$')

A_MIN = -0.8
A_MAX = +0.8

ax2.set_xlim3d(A_MIN, A_MAX)
ax2.set_ylim3d(A_MIN, A_MAX)
ax2.set_zlim3d(A_MIN, A_MAX)

plt.show()
