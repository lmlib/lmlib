"""
Alternative ALSSM Outputs [gu107.0]
===================================

lmlib offers different possibilities to generate an ALSSM output from estimates.
This guide script shows different methods.

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
rls = lm.RLSAlssm(cost, backend='numpy')
rls.filter(ys)
xs = rls.minimize_x()

# ------------------------------
# normal output defined by the model
ys_hat_normal = cost.eval_alssm_output(xs)

# ------------------------------
# alternative outputs with alssm weights parameter
ys_hat_aw = cost.eval_alssm_output(xs, alssm_weights=(0, 1))

