"""
Estimation a Random Walk with Occasional Jumps using NUV Priors [ex602.0]
=========================================================================

Approximation with a state space model of order n = 1 with a primary white-noise
input with a secondary sparse input [Loeliger2016]_ .

"""

from matplotlib import pyplot as plt

import lmlib as lm
import numpy as np

# signal
np.random.seed(6)
K = 700
ks = [150, 500, 620]
y_diff = [-2, 2.3, -3]
y = np.zeros(K)
y[ks] = y_diff
y = np.cumsum(y + 0.1 * np.random.randn(K))

# system
A = [[1]]
B = [[1]]
C = [[1]]

blk_system = lm.BlockSystem(A)
blk_input_noise = lm.BlockInput(B,sigma2_init=0.05, estimate_input=True)
blk_input_jump = lm.BlockInputNUV(B, sigma2_init=10.0, estimate_input=True, save_deployed_sigma2=True)
blk_output = lm.BlockOutput(C, sigma2_init=1.0, y=y, estimate_output=True)
blk = lm.BlockContainer(blocks=[blk_system, blk_input_noise, blk_input_jump, blk_output], save_marginals=True)

# message passing
fg = lm.FactorGraph(blk, lm.MBF, K)
fg.optimize(iterations=40)

# get variables of fg
Yt = fg.get_mp_block(blk_output).memory['Yt']
X = fg.get_mp_block(blk).get_marginals()
U_noise = fg.get_mp_block(blk_input_noise).get_U()
U_jump = fg.get_mp_block(blk_input_jump).get_deployed_sigma2()

# plot
fig, axs = plt.subplots(3, 1, sharex='all')
axs[0].plot(y, lw=.7, c='grey', label='y')
axs[0].plot(Yt.m, lw=1.0, c='b', label=r'$m_\tilde{y}$')
axs[1].plot(U_noise.V[:, 0, 0], lw=1.0, c='k', label=r'$V_{U_B}$')
axs[1].set_ylim(bottom=0)
axs[2].plot(U_jump.V[:, 0, 0], lw=1.0, c='k', label=r'$V_{U_J}$')
axs[2].set_ylim(bottom=0)

axs[-1].set_xlabel(r'$k$')
for ax in axs:
    ax.legend(loc=1)
    ax.grid('minor')
plt.show()
