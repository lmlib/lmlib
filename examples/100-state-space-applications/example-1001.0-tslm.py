"""
Example of the Application for TSLM Object
==========================================

Demonstration of different Hypothesis for TSLM
"""

import lmlib as lm
import numpy as np
import matplotlib.pyplot as plt
from lmlib.utils import gen_slopes, gen_wgn

K = 800
ks = [100, 250, 400, 500, 501, 600, 700, 701, 799]
deltas = [0, 7, -7, 0, 6, 0, 3, -5, 2]
y = gen_slopes(K, ks, deltas) + gen_wgn(K, 4e-1, seed=3141592)

names0 = ['cont.', 'left hori.', 'peak', 'right hori.', 'step', 'free']
names1 = ['straight.', 'straight', 'hori.', 'hori.', 'hori.', 'straight']
H0s = [lm.TSLM.H_Continuous, lm.TSLM.H_Left_Horizontal, lm.TSLM.H_Peak, lm.TSLM.H_Right_Horizontal, lm.TSLM.H_Step, lm.TSLM.H_Free]
H1s = [lm.TSLM.H_Straight, lm.TSLM.H_Straight, lm.TSLM.H_Horizontal, lm.TSLM.H_Horizontal, lm.TSLM.H_Horizontal, lm.TSLM.H_Straight]

ks_ranges_plt = [range(K), range(50, 150), range(150, 350), range(350, 450), range(450, 550), range(650, 750)]
colors0 = plt.cm.get_cmap('tab20').colors[::2]
colors1 = plt.cm.get_cmap('tab20').colors[1::2]
linestyles = ['--'] + ['-'] * 5

ks_cont = np.arange(550, 650)
a = -40
b = 40
cost = lm.TSLM.create_cost(ab=(a, b), gs=(50, 50))
rls = lm.RLSAlssm(cost)

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='all', figsize=(15, 12))
ax1.plot(y, c=(0.3,)*3, lw=0.5, label='y')

for H0, H1, ks_range, c0, c1, ls, name0, name1 in zip(H0s, H1s, ks_ranges_plt, colors0, colors1, linestyles, names0, names1):
    xs_0 = rls.filter_minimize_x(y, H=H0)
    xs_1 = rls.filter_minimize_x(y, H=H1)

    J_0 = rls.eval_errors(xs_0)
    J_1 = rls.eval_errors(xs_1)

    CR = -0.5 * np.log10(J_0 / J_1)

    ks_range = np.array(list(ks_range))
    if name0 == 'cont.':
        ks_max = ks_cont[[np.nanargmax(CR[ks_cont])]]
    else:
        ks_max = ks_range[[np.nanargmax(CR[ks_range])]]

    trajs_0 = lm.map_trajectories(cost.trajectories(xs_0[ks_max]), ks_max, K, True, True)
    trajs_1 = lm.map_trajectories(cost.trajectories(xs_1[ks_max]), ks_max, K, True, True)
    ax1.plot(trajs_0, c=c0, label=name0 + r' $s_k(x_0)$')
    ax1.plot(trajs_1, c=c1, label=name1 + r' $s_k(x_1)$')

    ax1.legend(loc=1, fontsize=7)
    ax2.plot(ks_range, J_0[ks_range], c=c0, lw=1, label=name0 + r' $J(x_0)$')
    ax2.plot(ks_range, J_1[ks_range], c=c1, lw=1, label=name1 + r' $J(x_1)$')
    ax2.legend(loc=1, fontsize=7)
    ax3.plot(ks_range, CR[ks_range], c=c0, ls=ls, lw=1, label= name0 + '/' + name1 + ' cost ratio')
    ax3.legend(loc=1, fontsize=7)
    ax3.set_xlabel('k')

plt.show()
