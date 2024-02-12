"""
Alternative ALSSM Outputs [ex107.0]
===================================

lmlib offers different possibilities to generate an ALSSM output from estimates.
This example shows different methods.

"""
import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm
from lmlib.utils import gen_wgn, gen_sine, gen_rand_walk, k_period_to_omega

# create signal
seed = 12345
K = 1000
k_periods = 300
ys = gen_sine(K, k_periods) + 0.05 * gen_rand_walk(K, seed=seed) + gen_wgn(K, 0.1, seed=seed)

# create model
alssms = (lm.AlssmPoly(1), lm.AlssmSin(k_period_to_omega(k_periods)))
segments = (lm.Segment(-100, -1, lm.FW, g=100), lm.Segment(0, 99, lm.BW, g=100))
F = np.ones((2, 2))
cost = lm.CompositeCost(alssms, segments, F)

# filter and minimize
rls = lm.RLSAlssm(cost)
xs = rls.filter_minimize_x(ys)

# ------------------------------
# normal output defined by the model

# output by cost model
ys_hat_normal = cost.eval_alssm_output(xs)

# output by RLSAlssm*
ys_hat_normal = rls.filter_minimize_yhat(ys)

# ------------------------------
# alternative outputs with alssm weights parameter

# output by cost model
ys_hat_aw = cost.eval_alssm_output(xs, alssm_weights=(0, 1))

# output by RLSAlssm*
ys_hat_aw = rls.filter_minimize_yhat(ys, alssm_weights=(0, 1))

# ------------------------------
# alternative outputs with c0 parameter

# output by cost model
ys_hat_c0 = cost.eval_alssm_output(xs, c0s=([0, 1], [0, 1]))
ys_hat_c0 = cost.eval_alssm_output(xs, c0s=[1, 0, 1, 0])
ys_hat_c0 = cost.eval_alssm_output(xs, c0s=None)

# output by RLSAlssm*
ys_hat_c0 = rls.filter_minimize_yhat(ys, c0s=([0, 1], None))
