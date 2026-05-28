"""
Structure and Evaluation of Univariate Polynomials [gu201.0]
============================================================

Demonstrates how to create and evaluate a univariate polynomial using
:class:`~lmlib.polynomial.poly.Poly` in vector exponent notation
:math:`\\alpha^\\mathsf{T} x^q`.

Four evaluation cases are shown:

1. Print the polynomial's coefficient and exponent vectors.
2. Evaluate at a scalar :math:`x = 3`.
3. Evaluate over a 1-D array of :math:`x` values.
4. Evaluate over a 3-D array and confirm the output shape.

"""
import numpy as np
import lmlib as lm

expo = [0, 2, 4]
coef = [-0.3, 0.08, -0.004]
poly = lm.Poly(coef, expo)

print("-- (1) --")
print(poly)

print("-- (2) --")
print(poly.eval(3))

print("-- (3) --")
print(poly.eval(np.arange(5)))

print("-- (4) --")
var = np.arange(2*4*6).reshape([2, 4, 6])
print(poly.eval(var).shape)
