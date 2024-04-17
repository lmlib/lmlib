"""
Different Types of NUV Priors [ex605.0]
=======================================

[Keusch2022]_ .

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
A = [[1]]
B = [[1]]
C = [[1]]



fig, axs = plt.subplots(5, 1, sharex='all')

line_styles = [dict(ls='-', lw=1, c='b'), dict(ls='-', lw=1, c='g')]

nuv_priors = [dict(prior_type='trivial', constraint=None, update_algo='EM'), dict(prior_type='trivial', constraint=None, update_algo='AM')]
nuv_priors = [dict(prior_type='binary', constraint=(-5, 5), update_algo='EM'), dict(prior_type='binary', constraint=(-5, 5), update_algo='AM')]
nuv_priors = [dict(prior_type='box', constraint=(-2, 2, 1e-2), update_algo='AM')]
# nuv_priors = [dict(prior_type='half-space', constraint=(-0.5, 1e-2), update_algo='AM')]

for i, nuv_prior in enumerate(nuv_priors):

    sc_system = lm.SectionSystem(A)
    sc_input = lm.SectionInputNUV(B, **nuv_prior, sigma2_init=0.0, estimate_input=True, save_deployed_sigma2=True)
    sc_output = lm.SectionOutput(C, sigma2_init=1.0, y=y, estimate_output=True)
    sc = lm.SectionContainer(sections=[sc_system, sc_input, sc_output], save_marginal=True)

    # message passing & set initial states
    fg = lm.FactorGraph(sc, left_side_prior=(0, 1e3), right_side_prior=(0, 1e-3))
    fg.initialize_mp(lm.MBF, K)

    # optimize
    fg.optimize(iterations=10)

    # get variables of fg
    mp_sc = fg.get_mp_section()
    X = mp_sc.get_marginal()

    Yt = mp_sc.get_mp_subsection(sc_output).get_Y_tilde()
    U = mp_sc.get_mp_subsection(sc_input).get_U()

    # plot
    axs[0].plot(y, lw=.7, c='grey', label='y')
    axs[0].plot(Yt.m, **line_styles[i], label=r'$m_\tilde{y}$')
    axs[1].plot(U.V[:, 0, 0], **line_styles[i], label=r'$V_{U}$')
    axs[2].plot(U.m[:, 0], **line_styles[i], label=r'$m_{U}$')
    axs[3].plot(X.m[:, 0], **line_styles[i], label=r'$m_X$')

axs[-1].set_xlabel(r'$k$')
for ax in axs:
    ax.legend(loc=1)
    ax.grid('minor')
plt.show()
