"""
Outlier- insensitive Kalman smoothing and marginal message passing [ex604.0]
============================================================================

Outlier- insensitive Kalman smoothing, with both a Modified Bryson-Frasier smoother and
Backward Information Filter Forward Marginal smoother, neither of which requires matrix inversions.
A simple and effective method to detect and to remove outliers from the scalar output signal of a state space model.
[Wadehn2016]_,  [Loeliger2016]_ .

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
blk_input = lm.BlockInputNUV(B1, sigma2_init=1.0, estimate_input=True, save_deployed_sigma2=True)
blk_output = lm.BlockOutputOutlier(C, sigma2_init=1.0, y=y, estimate_output=True)
blk = lm.BlockContainer(blocks=[blk_system, blk_input, blk_output], save_marginals=True)

# message passing
fg = lm.FactorGraph(blk, lm.MBF, K)

# set initial states & optimize
init_msg_bw = lm.MBF.get_backward_initial_state(fg.N, xi=0, W=1e-3)
init_msg_fw = lm.MBF.get_forward_initial_state(fg.N, m=0, V=1000)
fg.optimize(iterations=100, init_msg_fw=init_msg_fw, init_msg_bw=init_msg_bw)

# get variables of fg
Yt = fg.get_mp_block(blk_output).memory['Yt']
X = fg.get_mp_block(blk).get_marginals()
U_offset = fg.get_mp_block(blk_input).get_U()
U_slope = fg.get_mp_block(blk_output).get_U()

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
