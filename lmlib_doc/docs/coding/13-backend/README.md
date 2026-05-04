# Backend Selection (Numpy, lfilter, JIT) 

Todo: explain backend

The module `lmlib.statespace.backend`{.interpreted-text role="mod"}
provides methods to enable Just-in-Time (JIT) compilation for
time-critical routines in the package
`lmlib.statespace`{.interpreted-text role="mod"}. Turning on JIT
significantly increases execution speed of filtering applications. To
activate JIT, the Python `numba` package must be installed. The
following examples demonstrate the usage of JIT.
