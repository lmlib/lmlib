"""
Multi Segment [ex124.0]
=======================

Pyramid style model stacks with multiple segments

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
import lmlib as lm
from lmlib.utils import load_lib_csv


y = load_lib_csv('EECG_BASELINE_1CH_10S_FS2400HZ.csv')
K = len(y)
ks = [3780]

bounds_r = np.arange(0, 100, 10)
bounds_l = -np.flip(bounds_r)
print(bounds_r, bounds_l)
segs_l = []
for i, a in enumerate(bounds_l[:-1]):
    b = bounds_l[i + 1]
    segs_l.append(lm.Segment(int(a), int(b-1), lm.FW, g=100, delta=int(b-1)))

segs_r = []
for i, b in enumerate(bounds_r[1:], start=1):
    a = bounds_r[i - 1]
    segs_r.append(lm.Segment(int(a), int(b-1), lm.BW, g=100, delta=int(a)))


segs = segs_l + segs_r

n_segs_l = len(segs_l)
n_segs_r = len(segs_r)
for s in segs:
    print(s)


alssm = lm.AlssmPoly(1)
alssms = [alssm]*len(segs)

F_l = np.rot90(np.tril(np.ones((n_segs_l, n_segs_l))))
F_r = np.tril(np.ones((n_segs_r, n_segs_r)))
F = np.concatenate([F_l, F_r], axis=-1)
F = block_diag(F_l, F_r)


cost = lm.CompositeCost(alssms, segs, F)
print('F= \n', F)


rls = lm.RLSAlssm(cost)
xs = rls.filter_minimize_x(y)
error = rls.eval_errors(xs)
traj = lm.map_trajectories(cost.trajectories(xs[ks], F=F), ks, K, merge_ks=True, merge_seg=True)

trajs = []
for m, f in enumerate(F):
    F_ = np.zeros_like(F)
    F_[m] = f
    t_ = lm.map_trajectories(cost.trajectories(xs[ks], F=F_), ks, K, merge_ks=True, merge_seg=True)
    trajs.append(t_)


fig, (ax1, ax2) = plt.subplots(2, sharex='all')
ax1.plot(y, c='k', lw=0.8)
# ax1.plot(y_hat)
for t_ in trajs:
    ax1.plot(t_, lw='0.8')
ax1.plot(traj, c='b')
ax2.plot(error)
ax1.set_xlim(left=3600, right=4000)
plt.show()