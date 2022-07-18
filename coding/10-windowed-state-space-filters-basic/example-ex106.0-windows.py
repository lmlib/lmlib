"""
Windows [ex106.0]
=================

Plotting of different windows

See also:
:meth:`~lmlib.statespace.model.Segment.windows`,

"""

import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm

K = 1000
ks = [500]
k = range(K)
thd = 0.5

segment_left_infinite = lm.Segment(a=-np.inf, b=-20, direction=lm.FORWARD, g=200, delta=-20)
segment_left_finite = lm.Segment(a=-19, b=-1, direction=lm.FORWARD, g=1e6, delta=0)
segment_right_finite = lm.Segment(a=0, b=19, direction=lm.BACKWARD, g=1e6, delta=0)
segment_right_infinite = lm.Segment(a=20, b=np.inf, direction=lm.BACKWARD, g=200, delta=20)

cost = lm.CompositeCost((lm.AlssmPoly(3),),
                        [segment_left_infinite, segment_left_finite, segment_right_finite, segment_right_infinite],
                        F=[[1, 1, 1, 1]])
wins = lm.map_windows(cost.windows([0, 1, 2, 3], thd), ks, K, merge_ks=True, merge_seg=False)
win_no_thd = lm.map_windows(cost.windows([0, 3]), ks, K, merge_ks=True, merge_seg=True)

# plot
for p, win in enumerate(wins):
    plt.plot(k, win, '-', lw=.6, label=f'Segment: {p}')
plt.plot(k, win_no_thd, ':', lw=.6, label=f'Infinite Segments')
plt.axhline(thd, lw=0.6, c='k', ls='--', label='threshold for infinite windows')
plt.xlabel('time index $k$')
plt.ylabel('window weight')
plt.title(f'Window Weights at $k={ks}$')
plt.legend(loc=1, fontsize=8)
plt.show()
