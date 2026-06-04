r"""
This module is a collection of utility functions accelerate algorithm development and/or for educative purposes.


Signal Loading Functions
------------------------
These function are used to load a collection of recorded biological signals, see biosignals_catalog


- [`load_lib_csv`][lmlib.utils.load_lib_csv]
- [`load_lib_csv_mc`][lmlib.utils.load_lib_csv_mc]
- [`load_csv`][lmlib.utils.load_csv]
- [`load_csv_mc`][lmlib.utils.load_csv_mc]

Synthetic Signal Generation Functions
-------------------------------------
These function are used to generate synthetic signals. See generator_catalog for an overview.

- [`gen_sine`][lmlib.utils.gen_sine]
- [`gen_rect`][lmlib.utils.gen_rect]
- [`gen_tri`][lmlib.utils.gen_tri]
- [`gen_saw`][lmlib.utils.gen_saw]
- [`gen_pulse`][lmlib.utils.gen_pulse]
- [`gen_exp`][lmlib.utils.gen_exp]
- [`gen_steps`][lmlib.utils.gen_steps]
- [`gen_slopes`][lmlib.utils.gen_slopes]
- [`gen_wgn`][lmlib.utils.gen_wgn]
- [`gen_rand_walk`][lmlib.utils.gen_rand_walk]
- [`gen_rand_pulse`][lmlib.utils.gen_rand_pulse]
- [`gen_conv`][lmlib.utils.gen_conv]
"""

from .check import *
from .generator import *
from .colors import NORD
from . import profiling
