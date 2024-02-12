"""
ALSSM Trajectory [ex102.0]
==========================

Evaluation of a ALSSM over a time range with a given initial state.

See also:
:meth:`~lmlib.statespace.model.Alssm.trajectory`,
:meth:`~lmlib.statespace.model.Alssm.trajectories`

"""

import matplotlib.pyplot as plt
import lmlib as lm


js = range(-20, 20)  # ALSSM evaluation range

alssm = lm.AlssmPoly(poly_degree=1, label='1th degree')
x0_d1 = [-1, 2]  # initial state vector
sx0_d1 = alssm.trajectory(x0_d1, js)

alssm = lm.AlssmPoly(poly_degree=2, label='2nd degree')
x0_d2 = [-1, 2, .1]  # initial state vector
sx0_d2 = alssm.trajectory(x0_d2, js)

alssm = lm.AlssmPoly(poly_degree=3, label='3nd degree')
x0_d3 = [-1, 2, .1, -.01]  # initial state vector
sx0_d3 = alssm.trajectory(x0_d3, js)

# Printing Model to Console
print("--DUMP--\n", alssm.dump_tree())
print("--PRINT--\n", alssm)

# plot
plt.plot(js, sx0_d1, '.-', lw=.5, label=r'$x_0 = ' + str(x0_d1) + '^\mathrm{T}$')
plt.plot(js, sx0_d2, '.-', lw=.5, label=r'$x_0 = ' + str(x0_d2) + '^\mathrm{T}$')
plt.plot(js, sx0_d3, '.-', lw=.5, label=r'$x_0 = ' + str(x0_d3) + '^\mathrm{T}$')
plt.xlabel('Evaluation index $j$')
plt.ylabel('$s_j(x_0)$')
plt.title('Polynomial ALSSM Evaluation $s_j(x_0)$')
plt.legend()
plt.show()
