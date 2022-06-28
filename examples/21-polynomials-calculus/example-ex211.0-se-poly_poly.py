"""
Squared error between two polynomials [ex211.0]
=====================================

Closed form computation of the squared error between two time-continuous polynomials.
"""

import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm

# parameters
alpha = [2, 0.2, 1.5]
beta = [-3.5, -0.5, 2.5]
q = [0, 1, 2]
a = -3
b = 3

# constant calculation
qt = lm.poly_square_expo(q)
L = lm.poly_int_coef_L(qt)
constant = np.dot(L.T, np.power(b, qt) - np.power(a, qt))

# observation
observation = np.kron(alpha, alpha) - np.kron(beta, alpha) - np.kron(alpha, beta) + np.kron(beta, beta)

# Squared Error
J = np.dot(observation, constant)

# ------------------ plot ------------------

x = np.arange(-4, 4.01, 0.01)
p1 = lm.Poly(alpha, q)
p2 = lm.Poly(beta, q)

y1 = p1.eval(x)
y2 = p2.eval(x)

fig, ax = plt.subplots(1, 1)
ax.plot(x, y1, 'r--', label='$p_{\\alpha}$')
ax.plot(x, y2, 'g--', label='$p_{\\beta}$')
ax.fill_between(x, y1, y2, where=np.bitwise_and(x < b, x >= a), color=(0.9, 0.9, 0.9),
                label='$J(\\alpha, \\beta)={:10.2f}$'.format(J))
ax.axvline(a, c='k', lw=0.5)
ax.axvline(b, c='k', lw=0.5)
ax.legend()
ax.set_xlabel('$x$')
ax.set_title('Squared Error Between Two Polynomials Within a Given Interval')
plt.show()
