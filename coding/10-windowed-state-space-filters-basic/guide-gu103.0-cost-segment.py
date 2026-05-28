"""
Single-Segment (CostSegment) Models: Trajectories and Windows [gu103.0]
=======================================================================

Defines a cost segment consisting of an ALSSM and a left-sided,
exponentially decaying window, then visualises the window weights and
the resulting signal trajectories for a set of state vectors.

Two plot groups are shown:

* **Upper plots** — the window and trajectories in the relative index
  domain (centred at :math:`j=0`).
* **Lower plots** — the same trajectories placed at absolute signal
  positions ``K_refs`` in a length-``K`` output vector.

See also:
:class:`~lmlib.statespace.cost.CostSegment`,
:class:`~lmlib.statespace.segment.Segment`
"""
import matplotlib.pyplot as plt
import lmlib as lm
import numpy as np

# Defining a second order polynomial ALSSM
alssm_poly = lm.AlssmPoly(poly_degree=3, label="alssm-polynomial")

# Defining a segment with a left-sided, exponentially decaying window
a = -10  # left boundary
b = 5  # right boundary
g = 8  # effective number of weighted samples under the window (controls window size)
left_seg = lm.Segment(a, b, lm.FORWARD, g, label="left-decaying")

# creating the cost segment, combining window (segment) and model (ALSSM).
costs = lm.CostSegment(alssm_poly, left_seg, label="costs segment for polynomial model")

# print internal structure
print('-- Print --')
print(costs)

# get trajectory from initial state
xs = np.array([[-1, 2, 0, 0],  # polynomial coefficients of trajectories
      [-1, 2, .6, 0],
      [-1, -1, .4, -.08]])

# --------- Upper Plots ---------
# get window weights

# Window
windows = lm.Window.eval(costs)
js, w = windows[0]

# Trajectories
trajectories = lm.Trajectory.eval(costs, xs, merged_ks=False)

# plot
fig, axs = plt.subplots(5, sharex='all', gridspec_kw={'hspace': 0.1}, figsize=(6, 6))
axs[0].plot(js,w, '-', c='k', lw=1.5, label=r'window weights $w_j = \gamma^j$')
axs[0].set_title('costs.trajectories(xs)')
axs[0].axvline(0, color="black", linestyle="--", lw=1.0)
axs[0].axhline(1, color="black", linestyle="--", lw=0.5)
axs[0].axvline(a, color="gray", linestyle="-", lw=0.5)
axs[0].axvline(b, color="gray", linestyle="-", lw=0.5)
axs[0].set(ylim=[0, 2.1])

for n, trajectory_p in enumerate(trajectories):
    js, trajectory = trajectory_p
    axs[1].plot(js, trajectory, lw=1.5, label=r'trajectory $s_j(x_' + str(n) + ') = cA^jx_' + str(n) + '$')

# ideas for alternative plotting functions:
# ax = traj_obj.plot_merged(axs[1], lw=1.5, label='trajectory $s_j$')
# ax = traj_obj.plot_merged_segments(axs[1], lw=1.5, label=['trajectory $s_j$' for ..])
# ax = traj_obj.plot_merged_states(axs[1], lw=1.5, label=['trajectory $s_j$' for ..])

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

windows = lm.Window.eval_y(costs, K_refs, K, merged_seg=True)
axs[3].plot(windows,  lw=1.5, label=r"window weights $w_{k-K_{ref}}$")

trajectories = lm.Trajectory.eval_y(costs, xs, K_refs, K, merged_seg=True)

axs[4].plot(trajectories, lw=2, label='y_hat')

for k_ref in K_refs:
    axs[4].axvline(k_ref + a, color="gray", linestyle="-", lw=0.5)
    axs[4].axvline(k_ref + b, color="gray", linestyle="-", lw=0.5)
    axs[4].axvline(k_ref, color="black", linestyle="--", lw=0.5)

axs[4].set_xlabel('Evaluation Index $k$')
axs[4].set_ylim([-40, 80])

for ax in axs:
    ax.legend(fontsize=7)
plt.show()
