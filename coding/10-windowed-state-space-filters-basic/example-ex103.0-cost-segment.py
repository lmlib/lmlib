"""
Single-Segment (CostSegment) Models: Trajectories and Windows [ex103.0]
=======================================================================

Defines a cost segment which consists of an ALSSM and a left-sided,
exponentially decaying window.

See also:
:ref:`Cost Function Classes <lmlib.statespace>`,
:class:`~lmlib.statespace.cost.CostSegment`,
:class:`~lmlib.statespace.cost.Segment`
"""
import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm

# Defining a second order polynomial ALSSM
alssm_poly = lm.AlssmPoly(poly_degree=3, label="alssm-polynomial")

# Defining a segment with a left-sided, exponentially decaying window
a = -10  # left boundary
b = 5  # right boundary
g = 8  # effective weighted number of sample under the window (controlling the window size)
left_seg = lm.Segment(a, b, lm.FORWARD, g, label="left-decaying")

# creating the cost segment, combining window (segment) and model (ALSSM).
costs = lm.CostSegment(alssm_poly, left_seg, label="costs segment for polynomial model")

# print internal structure
print('-- Print --')
print(costs)

# get trajectory from initial state
xs = [[-1, 2, 0, 0],  # polynomial coefficients of trajectories
      [-1, 2, .6, 0],
      [-1, -1, .4, -.08]]

# --------- Upper Plots ---------
# get window weights
windows = costs.windows()
js, window = windows[0]

trajectories = costs.trajectories(xs)
print(xs[0])

# plot
fig, axs = plt.subplots(5, sharex='all', gridspec_kw={'hspace': 0.1}, figsize=(6, 6))
axs[0].plot(js, window, '-', c='k', lw=1.5, label='winodw weights $w_j = \gamma^j$')
axs[0].set_title('costs.trajectories(xs)')
axs[0].axvline(0, color="black", linestyle="--", lw=1.0)
axs[0].axhline(1, color="black", linestyle="--", lw=0.5)
axs[0].axvline(a, color="gray", linestyle="-", lw=0.5)
axs[0].axvline(b, color="gray", linestyle="-", lw=0.5)
axs[0].set(ylim=[0, 2.1])

for n in range(len(trajectories)):
    print(trajectories[n][0])
    js, trajectory = trajectories[n][0]
    axs[1].plot(js, trajectory, lw=1.5, label='trajectory $s_j(x_' + str(n) + ') = cA^jx_' + str(n) + '$')
axs[1].set_xlabel('Evaluation Index $j$')
axs[1].axvline(0, color="black", linestyle="--", lw=0.5)
axs[1].axvline(a, color="gray", linestyle="-", lw=0.5)
axs[1].axvline(b, color="gray", linestyle="-", lw=0.5)
axs[1].set_ylim([-40, 80])

axs[1].tick_params(axis='both', which='both', labelbottom=True)

axs[2].set_visible(False)  # add spacer

# --------- Lower Plots ---------
# Get localized trajectories
K = 70  # total signal length
K_refs = [20, 40, 60]  # trajectory locations (indices)
COLS_W = ['black', 'gray', 'lightgray']

wins = lm.map_windows(costs.windows(), K_refs, K, merge_seg=True)
axs[3].set_title('lm.map_trajectories( costs.trajectories(xs) )')
for n, win in enumerate(wins):
    axs[3].plot(win, '-', color=COLS_W[n], lw=1.5, label=r"winodw weights $w_{k-K_{ref}}$")
trajectory = lm.map_trajectories(costs.trajectories(xs), K_refs, K)

for n in range(len(trajectories)):
    axs[4].plot(trajectory[n][0], lw=2, label='y_hat')

for k_ref in K_refs:
    axs[4].axvline(k_ref + a, color="gray", linestyle="-", lw=0.5)
    axs[4].axvline(k_ref + b, color="gray", linestyle="-", lw=0.5)
    axs[4].axvline(k_ref, color="black", linestyle="--", lw=0.5)

axs[4].set_xlabel('Evaluation Index $k$')
axs[4].set_ylim([-40, 80])

for ax in axs:
    ax.legend(fontsize=7)
plt.show()
