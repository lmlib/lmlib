"""
Superposition (Stacking) of ALSSMs [gu101.1]
============================================

Generates two discrete-time autonomous linear state space models and stacks
them into a single combined model that produces a two-channel stacked output.

[`dump_tree`][lmlib.statespace.model.ModelBase.dump_tree] returns the internal
structure of the stacked ALSSM and helps with debugging.

Note: replacing [`AlssmStacked`][lmlib.statespace.model.AlssmStacked] with
[`AlssmSum`][lmlib.statespace.model.AlssmSum] generates a summed (scalar)
output instead of a stacked (multi-channel) output.

"""
import lmlib as lm

A = [[1, 1], [0, 1]]
C = [1, 0]
alssm_line = lm.Alssm(A, C, label="alssm-line")

alssm_poly = lm.AlssmPoly(poly_degree=3, label="alssm-polynomial")

# stacking ALSSM
alssm_stacked = lm.AlssmStacked((alssm_poly, alssm_line), label="alssm-stacked")

# print content and structure
print("-- Print --")
print(alssm_stacked)

print("\n-- Structure --")
print(alssm_stacked.dump_tree())
