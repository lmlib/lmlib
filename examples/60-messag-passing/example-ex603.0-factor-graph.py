"""
Approximation with straight-line segments using NUV Priors [ex603.0]
====================================================================

Approximation with straight-line segments, which can be represented by a state space model of order n = 2.
The input block has two separate sparse scalar input signals [Loeliger2016]_ .

"""

from matplotlib import pyplot as plt

import lmlib as lm
import numpy as np
from lmlib.utils import gen_slopes, gen_steps

# signal
K = 900
ks = [0, 349, 350, 620, 750, 900]
y_diff = [0, 3, -1, 2, -6, 1]
y = gen_slopes(K, ks, y_diff) + 0.1 * np.random.randn(K)

# system
A = [[1, 1], [0, 1]]
B1 = [[1], [0]]
B2 = [[0], [1]]
C = [[1, 0]]

blk_system = lm.BlockSystem(A)
blk_input_offset = lm.BlockInputNUV(B1, sigma2_init=1.0, estimate_input=True, save_deployed_sigma2=True)
blk_input_slope = lm.BlockInputNUV(B2, sigma2_init=1.0, estimate_input=True, save_deployed_sigma2=True)
blk_output = lm.BlockOutput(C, sigma2_init=1.0, y=y, estimate_output=True)
blk = lm.BlockContainer(blocks=[blk_system, blk_input_offset, blk_input_slope, blk_output], save_marginals=True)

# message passing
fg = lm.FactorGraph(blk, lm.MBF, K)

# set initial states & optimize
init_msg_bw = lm.MBF.get_backward_initial_state(fg.N, xi=0, W=1e-3)
init_msg_fw = lm.MBF.get_forward_initial_state(fg.N, m=0, V=1000)
fg.optimize(iterations=100, init_msg_fw=init_msg_fw, init_msg_bw=init_msg_bw)

# get variables of fg
Yt = fg.get_mp_block(blk_output).memory['Yt']
X = fg.get_mp_block(blk).get_marginals()
U_offset = fg.get_mp_block(blk_input_offset).get_U()
U_slope = fg.get_mp_block(blk_input_slope).get_U()

# plot
fig, axs = plt.subplots(5, 1, sharex='all')
axs[0].plot(y, lw=.7, c='grey', label='y')
axs[0].plot(Yt.m, lw=1.0, c='b', label=r'$m_\tilde{y}$')
axs[1].plot(U_offset.V[:, 0, 0], lw=1.0, c='k', label=r'$V_{U_1}$')
axs[2].plot(U_slope.V[:, 0, 0], lw=1.0, c='k', label=r'$V_{U_2}$')
axs[3].plot(X.m[:, 0], lw=1.0, c='k', label=r'$m_X^{(0)}$ (offset)')
axs[4].plot(X.m[:, 1], lw=1.0, c='k', label=r'$m_X^{(1)}$ (slope)')

axs[-1].set_xlabel(r'$k$')
for ax in axs:
    ax.legend(loc=1)
    ax.grid('minor')
plt.show()
