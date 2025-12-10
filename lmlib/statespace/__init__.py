"""
This module provides methods to define autonomous linear state space models (ALSSMs)
and to optimize squared error cost functions based on ALSSMs in a sliding window manner.
ALSSMs are input-free linear state space models (LSSMs),
i.e., the model outputs are fully defined by once the state vector is fixed,
often denoted as the initial state vector.
The output of such a deterministic model is then used as the signal model in quadratic cost terms.
Cost functions based on ALSSM  are internally efficiently computed using recursive computation rules.

This module implements the methods published in
[Wildhaber2018]_  :download:`PDF <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8355586#page=5>`,
with extensions from [Zalmai2017]_ , [Wildhaber2019]_ and [Baeriswyl2025]_.



.. Currentmodule:: lmlib.statespace

.. Autosummary::
    :toctree: ../_generated

    Model
    cost
    rls
    backend
    applications

"""

from .model import *
# from .cost import *
from .cost_v2 import *
from .trajectory import *
from .rls import *
from .applications import *
from .backend import *
from .segment import *

