import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm
from lmlib.utils import gen_sine, gen_wgn

K = 4000
y = gen_sine(K, 200) #+ gen_wgn(K, 0.1)
ks = [K//2]
# --------------- main -----------------------


# Defining segments with a left-resp. right-sided decaying window and a center segment with nearly rectangular window

a = -250
b = 250
g = 100

segment_left = lm.Segment(a=a, b=-1, direction=lm.FW, g=g)
segment_right = lm.Segment(a=0, b=b-1, direction=lm.BW, g=g)

polynomial_orders = np.arange(1, 13, 2, dtype=int)
trajs = np.zeros((K,  len(polynomial_orders)))

for i, N in enumerate(polynomial_orders):
    # Defining ALSSM models
    alssm = lm.AlssmPoly(poly_degree=int(N-1))

    # Defining Cost Model
    cost = lm.CompositeCost([alssm], [segment_left, segment_right], F=[[1, 1]])
    win = lm.map_windows(cost.windows([0, 1]), ks , K, True, True)

    P = lm.poly_dilation_coef_L(expo=np.arange(alssm.N), eta=float(b-a))
    cost_t = cost.transform(P)

    # filter signal with transformation
    rlst = lm.RLSAlssm(cost_t)
    zs = rlst.filter_minimize_x(y)
    xst = np.einsum('nm, km->kn', np.linalg.inv(P), zs)
    y_hat_t = cost_t.eval_alssm_output(zs)
    trajs[:, i] = lm.map_trajectories(cost.trajectories(xst[ks]), ks, K, True, True)


fig, axs = plt.subplots(3, sharex='all')
axs[0].plot(win, c='k', lw=1.0, label='window weights')
axs[1].plot(y, c='grey', lw=0.7, label=r"$y$")
for traj, N  in zip(trajs, polynomial_orders):
    axs[1].plot(traj, c='b', lw=1, label=rf"$s_j(x_k)$ order={N} (transformation)")

axs[1].set_ylim([min(y)-0.1, max(y)+0.1])
axs[2].plot(np.max(rlst._W, axis=(1, 2)), c='b', lw=1, label=r"$max(W_k)$")

for ax in axs:
    ax.legend(loc=1)
plt.show()
