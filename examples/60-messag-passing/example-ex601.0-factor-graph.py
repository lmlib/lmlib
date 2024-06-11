"""
Estimating (or fitting) a piecewise constant signal using NUV Priors [ex601.0]
==============================================================================

Constant segments can be represented by the simplest possible state space model
with n = 1, A = C = (1), and no input. For the occasional jumps between the constant segments,
a sparse input signal with a NUV prior is used. [Loeliger2016]_ .


.. image:: /static/examples/irrls-ex601.0.svg
    :width: 800
    :align: center

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

sc_system = lm.SectionSystem(A, label="system")
sc_input = lm.SectionInput(B, save_input_marginal=True, label="input")
sc_output = lm.SectionOutput(C, y, save_output_marginal=True, label="output")
sc = lm.SectionContainer([sc_system, sc_input, sc_output])

# message passing
fg = lm.FactorGraph(sc)
fg.initialize_mp(lm.MBF, K)
fg.optimize(iterations=1)

# get variables of fg
Yt = sc_output.get_output_marginal()
U = sc_input.get_input_marginal()


# plot
fig, axs = plt.subplots(2, 1, sharex='all')
axs[0].plot(y, lw=.7, c='grey', label='y')
axs[0].plot(Yt.m,  lw=1.0, c='b',label=r'$m_\tilde{y}$')
axs[1].plot(U.V[:, 0, 0],  lw=1.0, c='k',label=r'$V_U$')

axs[-1].set_xlabel(r'$k$')
for ax in axs:
    ax.legend(loc=1)
    ax.grid('minor')
plt.show()



