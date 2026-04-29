"""
This module is a collection of utility functions accelerate algorithm development and/or for educative purposes.


Signal Loading Functions
------------------------
These function are used to load a collection of recorded biological signals, see :ref:`biosignals_catalog`

.. currentmodule:: lmlib.utils.generator

.. autosummary::
    :toctree: ../_generated

    load_lib_csv
    load_lib_csv_mc

    load_csv
    load_csv_mc

Synthetic Signal Generation Functions
-------------------------------------
These function are used to generate synthetic signals. See :ref:`generator_catalog` for an overview.

.. autosummary::
    :toctree: ../_generated

    gen_sine
    gen_rect
    gen_tri
    gen_saw
    gen_pulse
    gen_exp
    gen_steps
    gen_slopes
    gen_wgn
    gen_rand_walk
    gen_rand_pulse
    gen_conv

"""

from .check import *
from .generator import *
from . import profiling
from .colors import NORD
