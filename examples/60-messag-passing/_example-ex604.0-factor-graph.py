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
from lmlib.utils import gen_sine, k_period_to_omega

# signal
K = 1000

ks = [0, 349, 350, 620, 750, 900]
y_diff = [0, 3, -1, 2, -6, 1]
k_period_1 = 200
k_period_2 = 40
y = gen_sine(K, k_period_1) + gen_sine(K, k_period_2, 0.5)
y[600:800] = 0
y += 0.1*np.random.randn(K)
y[np.random.randint(0, K, 50)] = 5*np.random.randn(50)

# system

A = lm.AlssmSum(
    [lm.AlssmSin(k_period_to_omega(k_period_1)),
     lm.AlssmSin(k_period_to_omega(k_period_2))]
).A
B1 = [[1], [0], [0], [0]]
C = [[1, 0, 0, 0]]

blk_system = lm.BlockSystem(A)
blk_input = lm.BlockInput(B1, sigma2_init=0.1, estimate_input=True)
blk_output = lm.BlockOutputOutlier(C, sigma2_init=0.1, y=y, estimate_output=True, save_outlier_estimate=True)
blk = lm.BlockContainer(blocks=[blk_system, blk_input, blk_output], save_marginals=True)

# message passing
fg = lm.FactorGraph(blk, lm.MBF, K)

# set initial states & optimize
init_msg_bw = lm.MBF.get_backward_initial_state(fg.N, xi=0, W=1e-3)
init_msg_fw = lm.MBF.get_forward_initial_state(fg.N, m=0, V=1000)
fg.optimize(iterations=10, init_msg_fw=init_msg_fw, init_msg_bw=init_msg_bw)

# get variables of fg
Yt = fg.get_mp_block(blk_output).memory['Yt']
X = fg.get_mp_block(blk).get_marginals()
S = fg.get_mp_block(blk_output).get_S()

# plot
fig, axs = plt.subplots(3, 1, sharex='all')
axs[0].plot(y, lw=.7, c='grey', label='y')
axs[0].plot(Yt.m, lw=1.0, c='b', label=r'$m_\tilde{y}$')
axs[1].plot(np.sqrt(S.V[:, 0, 0]), lw=1.0, c='k', label=r'$\sqrt{V_{S}}$')
axs[2].plot(Yt.V[:, 0, 0], lw=1.0, c='k', label=r'$V_{\tilde{Y}}$')

axs[-1].set_xlabel(r'$k$')
for ax in axs:
    ax.legend(loc=1)
    ax.grid('minor')
plt.show()
