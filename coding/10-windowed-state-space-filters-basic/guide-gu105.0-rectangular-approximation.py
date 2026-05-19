"""
Local Signal Approximation and Trajectories [gu105.0]
=====================================================

Local Signal Approximation and Trajectories using :class:`~lmlib.statespace.model.PolyAlssm`.
"""
import matplotlib.pyplot as plt
import lmlib as lm
import numpy as np
from lmlib.utils.generator import gen_wgn, gen_rect

K = 2100
y = gen_rect(K, 570, 200) + gen_wgn(K, 0.01)

pd=3
cost = lm.CostSegment(lm.AlssmPoly(poly_degree=pd),
                      lm.Segment(0, 200, lm.BW, 500))
cost_wide = lm.CostSegment(lm.AlssmPoly(poly_degree=pd),
                           lm.Segment(0 - 20, 220 + 20, lm.BW, 500))

rls = lm.RLSAlssm(cost) 
rls.filter(y)
xs = rls.minimize_x()

K_refs = [500, 1130, 1800]
trajs = lm.Trajectory.eval_y(cost, xs[K_refs], K_refs, K, thd=0.01, merged_ks=True, merged_seg=True)
trajs_wide =lm.Trajectory.eval_y(cost_wide,xs[K_refs], K_refs, K, thd=0.01,merged_ks=True, merged_seg=True)

plt.title('Local Signal Approximation: RLSAlssm.filter_minimize_x(y)')
plt.plot(y, lw=0.3, c='k', label='y')
plt.plot(trajs, lw=2, c='b', label='y_hat')
plt.plot(trajs_wide, lw=1, ls='--', c='b', label='y_hat')
plt.ylim([-3, 3])

plt.show()
