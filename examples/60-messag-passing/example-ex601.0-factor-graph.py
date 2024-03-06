"""
Estimating (or fitting) a piecewise constant signal using NUV Priors [ex601.0]
==============================================================================

Constant segments can be represented by the simplest possible state space model
with n = 1, A = C = (1), and no input. For the occasional jumps between the constant segments,
a sparse input signal with a NUV prior is used. [Loeliger2016]_ .

"""


from matplotlib import pyplot as plt

import lmlib as lm
import numpy as np

# signal
K = 700
ks = [110, 160, 340, 510, 610]
y_diff = [3, 1, -3.5, -1, -1.2]
y = np.zeros(K)
y[ks] = y_diff
y = np.cumsum(y) + np.random.randn(K)*0.3

# system
A = [[1]]
B = [[1]]
C = [[1]]

blk_system = lm.BlockSystem(A, label="system")
blk_input = lm.BlockInputNUV(B, sigma2_init=1.0, estimate_input=True, save_deployed_sigma2=True)
blk_output = lm.BlockOutput(C, sigma2_init=1.0, y=y, estimate_output=True)
blk = lm.BlockContainer(blocks=[blk_system, blk_input, blk_output], save_marginals=True)

# message passing
fg = lm.FactorGraph(blk, lm.MBF, K)
fg.optimize(iterations=100)


# get variables of fg
Yt = fg.get_mp_block(blk_output).memory['Yt']
X = fg.get_mp_block(blk).get_marginals()
U = fg.get_mp_block(blk_input).get_U()
U_fw = fg.get_mp_block(blk_input).get_deployed_sigma2()


# plot
fig, axs = plt.subplots(2, 1, sharex='all')
axs[0].plot(y, lw=.7, c='grey', label='y')
axs[0].plot(Yt.m,  lw=1.0, c='b',label=r'$m_\tilde{y}$')
axs[1].plot(U.V[:, 0, 0],  lw=1.0, c='k',label=r'$V_U$')
axs[1].plot(U_fw.V[:, 0, 0], lw=1.0, c='grey', label=r'$\overrightarrow{V}_U$')

axs[-1].set_xlabel(r'$k$')
for ax in axs:
    ax.legend(loc=1)
    ax.grid('minor')
plt.show()



