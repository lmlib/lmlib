"""
Superposition (Stacking) of ALSSMs [ex101.1]
============================================

Generating two discrete-time autonomous linear state space models and stack them to a single model,
generating a two-channel stacked output.

Note that :py:meth:`~lmlib.statespace.model.Alssm.dump_tree()` returns the internal structure of the stacked LSSM and helps debugging.

Note: replacing :py:class:`~lmlib.statespace.model.AlssmStacked()` by
:py:class:`~lmlib.statespace.model.AlssmSum()` generates a summed (instead of a stacked) output.

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
