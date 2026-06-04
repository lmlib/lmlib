"""
Basic ALSSM Coding [gu101.0]
============================

Demonstrates how to set up an autonomous linear state space model (ALSSM).

Two approaches are shown:

1. **Native ALSSM** — explicit definition of the state transition matrix
   $A$ and output matrix $C$ using [`Alssm`][lmlib.statespace.model.Alssm].
2. **Built-in polynomial ALSSM** — using the convenience class
   [`AlssmPoly`][lmlib.statespace.model.AlssmPoly], which constructs the Pascal
   upper-triangular $A$ and default $C$ from a polynomial degree.

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


