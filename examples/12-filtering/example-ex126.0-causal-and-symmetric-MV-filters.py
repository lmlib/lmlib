"""
Causal and Symmetric Moving Average Filters with ALSSMs [ex126.0]
=================================================================

Implementation of the causal and symmetric non-Causal filters using different ALSSMs.

Four different filters are presented:
    - Simple moving average (causal) of length L+1 using a constant polynomial ALSSM of order=1
    - Symmetric moving average (non-causal) of length 2L+1 using a constant polynomial ALSSM of order=1
    - Causal exponential moving average (causal) with decay factor gamma using an exponential ALSSM of order=1
    - Symmetric exponential moving average (non-causal) with decay factor gamma using an exponential ALSSM of order=1

The corresponding LS costs are depicted below each filter plot.

"""

import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from lmlib.utils.generator import gen_rect, gen_wgn, load_lib_csv

# --- Generating test signal ---
K_rect = 1500
y_rect = 0.2*(1-gen_rect(K_rect, 500, 250, k0=100))
K_ecg = 3000
ds_rate = 2
y_ecg = load_lib_csv('EECG_FILT_1CH_10S_FS2400HZ.csv', K_ecg, k_start=500, ds_rate=ds_rate) + gen_wgn(K_ecg//ds_rate, 0.01)

y = np.concatenate([y_rect, y_ecg], axis=0)
K = len(y)
k = np.arange(K)


# --- ALSSM Filtering ---
filter_lengths_L = [25, 50, 80]
gammas = [1.2,  1.075, 1.03]


y_hats_causal_MA = []
y_hats_non_causal_MA = []
y_hats_causal_EXP = []
y_hats_non_causal_EXP = []

J_causal_MA = []
J_non_causal_MA = []
J_causal_EXP = []
J_non_causal_EXP = []

for L in filter_lengths_L:

    # Polynomial ALSSM
    alssm_MA = lm.AlssmPoly(poly_degree=0)

    # Segments
    segment_causal = lm.Segment(a=-L, b=0, direction=lm.FORWARD, g=1e4)
    segments_non_causal = [lm.Segment(a=-L, b=0, direction=lm.FORWARD, g=1e4), lm.Segment(a=1, b=L+1, direction=lm.BACKWARD, g=1e4)]

    # Cost Models
    costs_causal_MA = lm.CostSegment(alssm_MA, segment_causal)
    costs_non_causal_MA = lm.CompositeCost([alssm_MA], segments_non_causal, F=[[1, 1]])

    # filter signal causal MA and take the approximation
    rls = lm.RLSAlssm(costs_causal_MA)
    xs = rls.filter_minimize_x(y)
    y_hats_causal_MA.append(costs_causal_MA.eval_alssm_output(xs))
    J_causal_MA.append(rls.eval_errors(xs))

    # filter signal non-causal MA and take the approximation
    rls = lm.RLSAlssm(costs_non_causal_MA)
    xs = rls.filter_minimize_x(y)
    y_hats_non_causal_MA.append(costs_non_causal_MA.eval_alssm_output(xs))
    J_non_causal_MA.append(rls.eval_errors(xs))

for gamma in gammas:

    # Polynomial ALSSM
    alssm_MA = lm.AlssmPoly(poly_degree=0)

    # Segments
    segment_causal_EXP = lm.Segment(a=-np.inf, b=0, direction=lm.FORWARD, g=None, gamma=gamma)
    segments_non_causal_EXP = [lm.Segment(a=-np.inf, b=0, direction=lm.FORWARD, g=None, gamma=gamma), lm.Segment(a=1, b=np.inf, direction=lm.BACKWARD, g=None, gamma=1/gamma)]

    costs_causal_EXP = lm.CostSegment(alssm_MA, segment_causal_EXP)
    costs_non_causal_EXP = lm.CompositeCost([alssm_MA], segments_non_causal_EXP, F=[1, 1])

    # filter signal causal EXP and take the approximation
    rls = lm.RLSAlssm(costs_causal_EXP)
    xs = rls.filter_minimize_x(y)
    y_hats_causal_EXP.append(costs_causal_EXP.eval_alssm_output(xs))
    J_causal_EXP.append(rls.eval_errors(xs))

    # filter signal non-causal EXP and take the approximation
    rls = lm.RLSAlssm(costs_non_causal_EXP)
    xs = rls.filter_minimize_x(y)
    y_hats_non_causal_EXP.append(costs_non_causal_EXP.eval_alssm_output(xs))
    J_non_causal_EXP.append(rls.eval_errors(xs))

# --- Plotting ----

if False:  # do not show costs
    STYLES = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    LW = 1.5
    LW_IN = 1.0

    fig, axs = plt.subplots(4, sharex='all', figsize=(8, 10))

    axs[0].plot(k, y, lw=LW_IN, c=np.ones(3) * 0.7, label=rf'$y$')
    for i, (y_hat, J) in enumerate(zip(y_hats_causal_MA, J_causal_MA)):
        axs[0].plot(k, y_hat, STYLES[i], lw=LW, label=r'$L=' + str(filter_lengths_L[i]) + '$')
    axs[0].set_title('Simple Moving Average (Causal)')

    axs[1].plot(k, y, lw=LW_IN, c=np.ones(3) * 0.7, label=rf'$y$')
    for i, (y_hat, J) in enumerate(zip(y_hats_non_causal_MA, J_non_causal_MA)):
        axs[1].plot(k, y_hat, STYLES[i], lw=LW, label=r'$L=' + str(filter_lengths_L[i]) + '$')
    axs[1].set_title('Symmetric Moving Average  (Non-Causal)')

    axs[2].plot(k, y, lw=LW_IN, c=np.ones(3) * 0.7, label=rf'$y$')
    for i, (y_hat, J) in enumerate(zip(y_hats_causal_EXP, J_causal_EXP)):
        axs[2].plot(k, y_hat, STYLES[i], lw=LW, label=r'$\gamma =' + str(gammas[i]) + r'$')
    axs[2].set_title('Exponential Moving Average (Causal)')

    axs[3].plot(k, y, lw=LW_IN, c=np.ones(3) * 0.7, label=rf'$y$')
    for i, (y_hat, J) in enumerate(zip(y_hats_non_causal_EXP, J_non_causal_EXP)):
        axs[3].plot(k, y_hat, STYLES[i], lw=LW, label=r'$\gamma =' + str(gammas[i]) + r'$')
    axs[3].set_title('Symmetric Exponential Moving Average (Non-Causal)')

    axs[-1].set_xlabel('k')
    for ax in axs:
        ax.legend(loc='upper right')
        ax.set_xlim([0, K])
        ax.set_ylim([-0.2, 0.32])
    plt.tight_layout(pad=0.25, w_pad=0.2, h_pad=0.02)
    plt.show()

else:  # do show costs
    STYLES = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    LW = 1.5
    LW_C = 1.0
    LW_IN = 1.0

    fig, axs = plt.subplots(11, sharex='all', figsize=(10, 10),
                            gridspec_kw={'height_ratios': [3, 1, .2, 3, 1, .2, 3, 1, .2, 3, 1]})

    axs[0].plot(k, y, lw=LW_IN, c=np.ones(3) * 0.7, label=rf'$y$')
    for i, (y_hat, J) in enumerate(zip(y_hats_causal_MA, J_causal_MA)):
        axs[0].plot(k, y_hat, STYLES[i], lw=LW, label=r'$L=' + str(filter_lengths_L[i]) + '$')
        axs[1].plot(k, J, STYLES[i], lw=LW_C, label=r'$L=' + str(filter_lengths_L[i]) + '$')
    axs[0].set_title(r"Simple Moving Average (Causal), Window L+1")

    axs[2].set_visible(False)

    axs[3].plot(k, y, lw=LW_IN, c=np.ones(3) * 0.7, label=rf'$y$')
    for i, (y_hat, J) in enumerate(zip(y_hats_non_causal_MA, J_non_causal_MA)):
        axs[3].plot(k, y_hat, STYLES[i], lw=LW, label=r'$L=' + str(filter_lengths_L[i]) + '$')
        axs[4].plot(k, J, STYLES[i], lw=LW_C, label=r'$L=' + str(filter_lengths_L[i]) + '$')
    axs[3].set_title('Symmetric Moving Average  (Non-Causal), Window 2L+1')

    axs[5].set_visible(False)

    axs[6].plot(k, y, lw=LW_IN, c=np.ones(3) * 0.7, label=rf'$y$')
    for i, (y_hat, J) in enumerate(zip(y_hats_causal_EXP, J_causal_EXP)):
        axs[6].plot(k, y_hat, STYLES[i], lw=LW, label=r'$\gamma =' + str(gammas[i]) + r'$')
        axs[7].plot(k, J, STYLES[i], lw=LW_C, label=r'$\gamma =' + str(gammas[i]) + r'$')
    axs[6].set_title(r'Exponential Moving Average (Causal), Decay $\gamma$')

    axs[8].set_visible(False)

    axs[9].plot(k, y, lw=LW_IN, c=np.ones(3) * 0.7, label=rf'$y$')
    for i, (y_hat, J) in enumerate(zip(y_hats_non_causal_EXP, J_non_causal_EXP)):
        axs[9].plot(k, y_hat, STYLES[i], lw=LW, label=r'$\gamma =' + str(gammas[i]) + r'$')
        axs[10].plot(k, J, STYLES[i], lw=LW_C, label=r'$\gamma =' + str(gammas[i]) + r'$')
    axs[9].set_title(r'Symmetric Exponential Moving Average (Non-Causal), Decay $\gamma$')

    axs[-1].set_xlabel('k')
    for (i, ax) in enumerate(axs):
        if i % 3 == 0:
            ax.legend(loc='upper right')
        else:
            ax.set_ylabel('cost')

    plt.tight_layout(pad=0.25, w_pad=0.2, h_pad=0.02)
    plt.show()