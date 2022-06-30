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
ks = [100, 300, 500, 700, 900]
k = range(K)

segment_left_infinite = lm.Segment(a=-np.inf, b=-20, direction=lm.FORWARD, g=1000, delta=-20)
segment_left_finite = lm.Segment(a=-19, b=-1, direction=lm.FORWARD, g=100, delta=0)
segment_right_finite = lm.Segment(a=0, b=19, direction=lm.BACKWARD, g=1000, delta=0)
segment_right_infinite = lm.Segment(a=20, b=np.inf, direction=lm.BACKWARD, g=1000, delta=20)

cost = lm.CompositeCost((lm.AlssmPoly(3),),
                        [segment_left_infinite, segment_left_finite, segment_right_finite, segment_right_infinite],
                        F=[[1, 1, 1, 1]])
wins = lm.map_windows(cost.windows([1, 1, 1, 1], thd=0.01), ks, K, merge_ks=True, merge_seg=False)


# plot

for p, win in enumerate(wins):
    plt.plot(k, win, '.-', lw=.5, label='Segment: {p}')
plt.xlabel('Evaluation index $j$')
plt.ylabel('$s_j(x_0)$')
plt.title('Polynomial ALSSM Evaluation $s_j(x_0)$')
plt.legend()
plt.show()
