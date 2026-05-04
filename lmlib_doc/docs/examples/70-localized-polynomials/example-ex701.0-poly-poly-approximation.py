"""
Approximating a Polynomial by a Polynomial of Lower Degree [ex701.0]
====================================================================

Fits a polynomial of lower order to a polynomial of higher order by minimizing the squared error over the given window.
This example is published in [Wildhaber2020]_ .

"""

import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm

# set up a high degree polynomial
alpha = np.array([0.2+0.195348430289480, 0.0346630187098931, -0.00473267554870773, 0.000187361113911066,
                  -3.22693894846170e-06, 2.53205688827560e-08, -7.43776537580248e-11])
Q = len(alpha)

# window settings
x0 = 65  # window center index
a = x0-30
b = x0+30

# --- Polynomial to Polynomial Approximation ---

# constant calculation
q = np.arange(Q)  # exponent vector from 0 to Q-1

R = 4  # Polynomial order (degree +1) of the polynomial approximation ## This is the lower order polynomial
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
beta = Lambda@alpha  # approximated low order polynomial coefficients

# ----------------  Plot  -----------------
z_range = x0+np.arange(-64, 65)
traj_Q = np.array([alpha.T@np.power(z, q) for z in z_range])
traj_R = np.array([beta.T@np.power(z, r) for z in z_range])

fig, ax = plt.subplots()
ax.plot(z_range, traj_Q, ls='--', c='b', lw=1.0, label=r'$\alpha^T z^q$')
ax.plot(z_range, traj_R, ls='-', c='k', lw=1.0, label=r'$\beta^T z^r$')
ax.axvline(a, ls=':', c='k', lw=0.5)
ax.axvline(b, ls=':', c='k', lw=0.5)
ax.set_ylim([-0.01, 0.35])
ax.set_xlim([15, 115])
ax.set_xlabel('x')
ax.set_xticks([a, b])
ax.set_xticklabels(['a','b'])
ax.set_yticks([])
ax.set_title('Approximating a Polynomial by a Polynomial of Lower Degree')
ax.legend(loc=1, fontsize=9)
plt.show()

