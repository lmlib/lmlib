"""
Structure and Evaluation of Multivariate Polynomials [gu201.1]
==============================================================

Demonstrates how to create and evaluate a multivariate polynomial using
:class:`~lmlib.polynomial.poly.MPoly` in vector exponent notation
:math:`\\tilde{\\alpha}^\\mathsf{T}(x^q \\otimes y^r)`.

Six evaluation cases are shown:

1. Univariate :class:`MPoly` (single variable, no cross terms) — print.
2. Evaluate at a scalar.
3. Evaluate over a 1-D array.
4. Bivariate :class:`MPoly` (factorised coefficients) — scalar evaluation.
5. Array-valued variable inputs — output shape matches input shape.
6. High-dimensional array inputs — confirm broadcast output shape.

"""
import numpy as np
import lmlib as lm

expos = ([0, 2, 4],)
coefs = ([-0.3, 0.08, -0.004],)
m_poly = lm.MPoly(coefs, expos)

print("-- 1 --")
print(m_poly)

print("-- 2 --")
print(m_poly.eval((3,)))

print("-- 3 --")
print(m_poly.eval((np.arange(5),)))

expos = ([0, 1, 2], [0, 2, 4])
coefs = ([0.1, -0.03, 0.01], [-0.3, 0.08, -0.004])
m_poly = lm.MPoly(coefs, expos)

print("-- 4 --")
# Scalar variable inputs yields a scalar output
print(m_poly.eval((1, 3)))

print("-- 5 --")
# Array_like variable inputs yields into a array_like output of the same shape
variables = ([1, 2, 3, 4], [3, 2, 1, 0])
out = m_poly.eval(variables)
print(out)
print('shape: ', len(out))

print("-- 6 --")
x = np.arange(3*2*5).reshape([3, 2, 5])
y = np.arange(3*2*5).reshape([3, 2, 5])-10
print('shape: ', m_poly.eval((x, y)).shape)
