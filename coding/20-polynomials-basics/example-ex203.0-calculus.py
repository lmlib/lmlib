"""
Calculus with Polynomials [ex203.0]
===================================

This example demonstrates the use of arithmetic manipulations on polynomials.

**Arithmetic Operations**

* Summation of polynomials
* Product of polynomials
* Integral of polynomials over a single or multiple variables
* Derivative of polynomials
* Shifting a polynomial to the left or right
* Dilating a polynomial

"""
import lmlib as lm

print("Defining univariate polynomials")
p1 = lm.Poly([1, 3, 5], [0, 1, 2])
p2 = lm.Poly([2, -1], [0, 1])

print(p1)
print(p2)
print("\n"+"-"*40+"\n")
print("Adding two polynomials p1 and p2\n")
print(lm.poly_sum((p1, p2)), '\n')

print("Multiply two polynomials p1 and p2\n")
print(lm.poly_prod((p1, p2)), '\n')


print("Square polynomial p1\n")
print(lm.poly_square(p1), '\n')


print("Shift polynomial p1\n")
gamma = 2
print(lm.poly_shift(p1, gamma), '\n')


print("Dilation polynomial p1\n")
eta = -5
print(lm.poly_dilation(p1, eta), '\n')


print("Integration polynomial p1\n")
print(lm.poly_int(p1), '\n')


print("Differentiation polynomial p1\n")
print(lm.poly_diff(p1), '\n')
