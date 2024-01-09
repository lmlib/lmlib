"""
Polynomial to Polynomial Fit [ex210.0]
======================================

Fits a polynomial of lower order to a polynomial of higher order by minimizing the squared error over the given window.
This example is derived and published in [Wildhaber2020]_ .

"""

import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from lmlib.utils.generator import gen_sine, gen_rand_walk

# Generate a Signal
K = 1000  # signal length
k = np.arange(K)  # sample index vector
y = gen_sine(K, k_periods=400) + 0.1 * gen_rand_walk(K, seed=1000)  # sinusoidal with additive random walk


# --- Defining a Model ---

# Polynomial ALSSM
Q = 4  # LSSM System Order
alssm_poly_Q = lm.AlssmPoly(poly_degree=Q-1)  # Q-1 polynomial degree

# Segment (Exponential Window starting at a ends at b-1, the decay is given by g (area) direction defines the
# computation direction)
segment_right = lm.Segment(a=-100, b=100, direction=lm.BACKWARD, g=200)

# Cost Segment connects ALSSM model with Segment
costs_Q = lm.CostSegment(alssm_poly_Q, segment_right)

# filter signal and take the approximation
rls = lm.RLSAlssm(costs_Q)  # data storage object

# se_param.set_transform(  se_param.get_steady_state_root() )


# filter data with the cost model defined above and minimize the costs using squared error filter parameters,
# xs are the polynomial coefficients
xs = rls.filter_minimize_x(y)
y_hat = costs_Q.eval_alssm_output(xs)  # signal estimate

# --- Polynomial to Polynomial Approximation ---

# constant calculation
a = segment_right.a  # left boundary a has to be finite
b = segment_right.b  # right boundary b has to be finite
q = np.arange(Q)  # exponent vector from 0 to Q-1

R = 3  # Polynomial order (degree +1) of the polynomial approximation ## This is the lower order polynomial
r = np.arange(R)  # exponent vector from 0 to R-1

# Constant Calculation (See. Signal Analysis Using Local Polynomial
# Approximations. Appendix)
s = np.concatenate([q, r])
A = np.concatenate([np.eye(Q), np.zeros((R, Q))], axis=0)
B = np.concatenate([np.zeros((Q, R)), np.eye(R)], axis=0)
Ms = lm.poly_square_expo(s)
L = lm.poly_int_coef_L(Ms)
c = np.power(b, Ms + 1) - np.power(a, Ms + 1)
vec_C = L.T @ c
C = vec_C.reshape(np.sqrt(len(vec_C)).astype(int), -1)

# Solves the Equation by setting derivative to zero.
Lambda = np.linalg.inv(2 * B.T @ C @ B) @ (2 * B.T @ C @ A)
betas = np.einsum('ij, nj -> ni', Lambda, xs)  # approximated low order polynomial coefficients

# ----------------  Plot  -----------------
ks = [200, 550, 800]  # show trajectory and windows at the indices in ks
wins = lm.map_windows(costs_Q.windows(), ks, K, merge_seg=True)  # windows

# trajectories of the higher order polynomial
trajs_Q = lm.map_trajectories(costs_Q.trajectories(xs[ks]), ks, K, merge_ks=True, merge_seg=True)

# trajectories of the lower order polynomial
costs_R = lm.CostSegment(lm.AlssmPoly(R - 1), segment_right)
trajs_R = lm.map_trajectories(costs_R.trajectories(betas[ks]), ks, K, merge_ks=True, merge_seg=True)


# Generate the plot
fig, axs = plt.subplots(2, 1, sharex='all')
axs[0].plot(k, wins[0], lw=1, c='k', ls='-')
axs[0].set_ylabel('window weights')
axs[0].legend([f'windows at {ks}'])
axs[1].plot(k, y, lw=0.5, c='grey', label=r'$y$')
axs[1].plot(k, y_hat, lw=0.4, c='brown', label=r'$\hat{y}$')
axs[1].plot(k, trajs_Q, lw=1.5, c='darkgreen', label=f'traj_Q at {ks}')
axs[1].plot(k, trajs_R, '--', lw=1.5, c='blue', label=f'traj_R at {ks}')
axs[1].set(xlabel='k', ylabel='amplitude')
axs[1].legend()

plt.show()
