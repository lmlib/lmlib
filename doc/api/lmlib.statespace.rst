.. _lmlib.statespace:

lmlib.statespace
================


**Package Abstract ::** This module provides methods to define autonomous linear state space models (ALSSMs)
and to define squared error cost functions based on such ALSSMs.
ALSSMs are input-free linear state space models (LSSMs),
i.e., the model outputs are fully defined by a single state vector,
often denoted as initial state vector *x*.
The output vector of such a deterministic model forms the signal model used.
Cost functions based on ALSSM  are internally efficiently computed using recursive computation rules.

This module implements the methods published in
[Wildhaber2018]_  :download:`PDF <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8355586#page=5>`,
with extensions from [Zalmai2017]_ and [Wildhaber2019]_.



Modules
--------------------

.. autosummary::
   :toctree: _autosummary
   :template: module
   
   lmlib.statespace.model
   lmlib.statespace.cost
   lmlib.statespace.backend   
   lmlib.statespace.applications




Introductory Example
--------------------


.. plot:: _autoexamples/11-detection/example-ex110.0-edge-detection.py
   :include-source:







   











