"""
Digital-to-analog conversion with target waveform [ex604.0]
===========================================================

Digital-to-analog conversion with target waveform using binary prior [Keusch2022]_ .

"""


import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm
from lmlib.utils import gen_sine, gen_saw, gen_wgn

y_sine = gen_sine(K=220, k_periods=110, amplitudes=0.5, k0s=(110 * 3) // 4) + 0.5
y_saw = gen_saw(K=250, k_period=110) + 0.5
y = np.concatenate((y_sine, y_saw))
K = len(y)

# system
A = [[0.7967, -6.3978, -94.2123],
     [0.0027, 0.9902, -0.1467],
     [0, 0.003, 0.9999]]
B = [[0.0027], [0], [0]]
C = [[0, 0, 35037.9]]

sc_system = lm.SectionSystem(A, label="system")
sc_input = lm.SectionInput_Binary(B, a=0, b=1, sigma2a_init=1., sigma2b_init=1., update_method='EM', save_input_marginal=True, label="binary-input")
sc_output = lm.SectionOutput(C, y, sigma2=0.045, save_output_marginal=True, label="output")
sc = lm.SectionContainer([sc_system, sc_input, sc_output])

# message passing
fg = lm.FactorGraph(sc, left_side_prior=(0, 1e3))
fg.initialize_mp(lm.MBF, K)
fg.optimize(iterations=10)

# get variables of fg
Yt = sc_output.get_output_marginal()
U = sc_input.get_input_marginal()

# plot
fig, axs = plt.subplots(2, sharex='all', gridspec_kw={'height_ratios': [2, 1]})
axs[0].plot(y, c='k', ls='--', lw=0.9, label=r'$y$')
axs[0].plot(Yt.m, c='b', ls='-', lw=0.9, label=r'$\tilde{y}$')
axs[1].plot(U.m, c='k', ls='-')
axs[1].set(xlabel='k', ylabel='u', xlim=(0, K))
for ax in axs:
    ax.grid('minor', lw=0.4)
    ax.legend()
plt.show()
