"""
Calculus with Polynomials [code203.0]
===================================

Demonstrates arithmetic operations on univariate polynomials using the
[`lmlib.polynomial.poly`][lmlib.polynomial.poly] calculus functions.

**Operations shown:**

* Sum of two polynomials ([`poly_sum`][lmlib.polynomial.poly.poly_sum])
* Product of two polynomials ([`poly_prod`][lmlib.polynomial.poly.poly_prod])
* Square of a polynomial ([`poly_square`][lmlib.polynomial.poly.poly_square])
* Shift of a polynomial ([`poly_shift`][lmlib.polynomial.poly.poly_shift])
* Dilation of a polynomial ([`poly_dilation`][lmlib.polynomial.poly.poly_dilation])
* Indefinite integral of a polynomial ([`poly_int`][lmlib.polynomial.poly.poly_int])
* Derivative of a polynomial ([`poly_diff`][lmlib.polynomial.poly.poly_diff])

"""
import lmlib as lm

print("Defining uni-variate polynomials")
p1 = lm.Poly([1, 3, 5], [0, 1, 2])
p2 = lm.Poly([2, -1], [0, 1])

print(p1)
print(p2)
print("\n"+"-"*40+"\n")
print("Sum of polynomials p1 and p2\n")
print(lm.poly_sum((p1, p2)), '\n')

print("Product of polynomials p1 and p2\n")
print(lm.poly_prod((p1, p2)), '\n')


print("Square polynomial p1\n")
print(lm.poly_square(p1), '\n')


print("Shift polynomial p1\n")
gamma = 2
print(lm.poly_shift(p1, gamma), '\n')


print("Dilation polynomial p1\n")
eta = -5
print(lm.poly_dilation(p1, eta), '\n')


print("Integral of polynomial p1\n")
print(lm.poly_int(p1), '\n')


print("Differentiation polynomial p1\n")
print(lm.poly_diff(p1), '\n')
