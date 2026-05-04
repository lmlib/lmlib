"""
Structure and Evaluation of Univariate Polynomials [ex201.1]
============================================================

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
