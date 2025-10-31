.. _backend:

Backend Selection (Numpy, lfilter, JIT)
----------------------------------------
Todo: explain backend

The module :mod:`lmlib.statespace.backend` provides methods to enable Just-in-Time (JIT) compilation for time-critical routines in the package :ref:`lmlib_api_statespace`.
Turning on JIT significantly increases execution speed of filtering applications. 
To activate JIT, the Python ``numba`` package must be installed.
The following examples demonstrate the usage of JIT. 