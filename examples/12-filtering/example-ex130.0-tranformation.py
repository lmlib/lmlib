import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm
from lmlib.utils import gen_sine
np.random.seed(61)
K = 2000
y = 1/10*np.cumsum(np.random.randn(K)) + gen_sine(K, 200) + 0.1*gen_sine(K, 40)
ks = [1000]
# --------------- main -----------------------

len_pulse = 20  # number of samples of the pulse width
g_sp = 15000  # pulse window weight
g_bl = 50  # baseline window weight

# Defining ALSSM models
alssm = lm.AlssmPoly(poly_degree=10)

# Defining segments with a left-resp. right-sided decaying window and a center segment with nearly rectangular window
segment_left = lm.Segment(a=-200, b=-1, direction=lm.FW, g=100)
segment_right = lm.Segment(a=0, b=199, direction=lm.BW, g=100)

cost = lm.CompositeCost([alssm], [segment_left, segment_right], F=[[1, 1]])

P = cost.get_steady_state_W_sqrt(method='limited_sum')
# P = cost.get_steady_state_W_sqrt(method='closed_form') #  todo : not invertible

cost_t = cost.transform(P)

# filter signal without transformation
rls = lm.RLSAlssm(cost)
xs = rls.filter_minimize_x(y)
y_hat = cost.eval_alssm_output(xs)
traj = lm.map_trajectories(cost.trajectories(xs[ks]), ks, K, True, True)

# filter signal with transformation
rlst = lm.RLSAlssm(cost_t)
zs = rlst.filter_minimize_x(y)
xst = np.einsum('nm, km->kn', np.linalg.inv(P), zs)
y_hat_t = cost_t.eval_alssm_output(zs)
traj_t = lm.map_trajectories(cost.trajectories(xst[ks]), ks, K, True, True)

fig, axs = plt.subplots(2)
axs[0].plot(y, c='grey', lw=0.7, label=r"$y$")
axs[0].plot(traj, c='r', lw=2, label=r"$s_j(x_k)$")
axs[0].plot(traj_t, c='b', lw=1, label=rf"$s_j(x_k)$ order={alssm.N} (transformation)")
axs[0].set_ylim([min(y)-0.1, max(y)+0.1])
axs[1].plot(np.max(rlst._W, axis=(1, 2)), c='b', lw=1, label=r"$max(W_k)$")

for ax in axs:
    ax.legend(loc=1)
plt.show()
