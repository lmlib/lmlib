"""
Basic ALSSM Coding [ex101.0]
=====================

This example demonstrates how to set up a autonomous linear state space model ALSSM.

See also:
:ref:`Module statespace <module_statespace>`,
:ref:`ALSSM Classes <classes_alssms>`

"""
import lmlib as lm

# Example 1: a native ALSSM with explicit A and C definition
A = [[1, 1], [0, 1]]
C = [1, 0]
alssm_line = lm.Alssm(A, C, label="my-native-alssm")

print("-- Print --")
print(alssm_line)

# Example 2: using built-in ALSSM generator for polynomial ALSSMs
alssm_poly = lm.AlssmPoly(poly_degree=3, label="my-polynomial-alssm")

print("\n-- Print --")
print(alssm_poly)


