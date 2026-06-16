r"""Selection tool to switch between Python interpreter (default) and JIT (Just-in-Time) compilation execution for time-critical routines in the `statespace` package. """

import importlib.util
import sys
from lmlib.statespace.cost import NDCompositeCost

__all__ = ['set_backend', 'is_backend_available', 'get_backend', 'BACKEND_TYPES', 'available_backends',
           'set_gpu_dtype', 'get_gpu_dtype']

_backend = 'lfilter' # current backend selection (global)

BACKEND_TYPES = ('jit', 'numpy', 'lfilter', 'cupy') # known backends
r"""tuple of str : All backend names known to lmlib (``'jit'``, ``'numpy'``, ``'lfilter'``, ``'cupy'``), whether or not they are installed on this system."""

available_backends = ('numpy', 'lfilter') # available backends
r"""tuple of str : Backend names actually available on this system. ``'jit'`` is appended at import time when the optional ``numba`` package is installed; ``'cupy'`` is appended when the optional ``cupy`` package is installed *and* a CUDA device is visible."""


def set_backend(backend):
    r"""
    Selects one out of multiple available backends (to optimize execution performance).

    Parameters
    ----------
    backend : str ("jit", "py", "python", "numpy", "lfilter")

          - "numpy" : for State Space Backend in Python (default)
          - "lfilter" : for Transfer Function Backend in Python
          - "jit": Just-in-Time compilation (if available)
          - "cupy": GPU Transfer Function Backend using CuPy / cupyx (if available)
          - "python" or "py" : Deprecated. (same as numpy)


    If the selected backend is not available, an assert is risen. 
    Use [`is_backend_available`][lmlib.statespace.backend.is_backend_available] to check availability first. 
    """
    backend = backend.lower()
    global _backend
    assert backend in BACKEND_TYPES, "Unknown backend name."
    if backend == 'jit':
        assert backend in available_backends, "jit backend not available. Check that numba package is installed!"
        _backend = 'jit'
    if backend == 'cupy':
        assert backend in available_backends, "cupy backend not available. Check that the cupy package is installed and a CUDA device is visible!"
        _backend = 'cupy'
    if backend in ('py', 'python'):
        DeprecationWarning("backend name 'py' and 'python' is deprecated and will be removed. Use 'numpy' instead.")
        _backend = 'numpy'
    if backend == 'numpy':
        _backend = 'numpy'
    if backend == 'lfilter':
        _backend = 'lfilter'

def set_gpu_dtype(dtype):
    r"""
    Set the on-device compute precision of the ``cupy`` GPU backend.

    Parameters
    ----------
    dtype : {'float32', 'float64'} or numpy dtype
        ``'float64'`` (default) gives ~1e-13 parity with the ``lfilter`` backend.
        ``'float32'`` is much faster on GPUs with reduced FP64 throughput
        (most consumer / laptop cards) at ~1e-6 relative accuracy (the error
        grows with ALSSM order).

    Notes
    -----
    Host buffers (``xi`` / ``kappa`` / ``W``) stay float64; only device math
    changes. The steady-state ``W`` is computed on the host and is unaffected.
    Requires the ``cupy`` backend to be available.
    """
    assert 'cupy' in available_backends, \
        "cupy backend not available; cannot set GPU precision."
    from lmlib.statespace.backends.rec_cupy import set_gpu_dtype as _set
    return _set(dtype)


def get_gpu_dtype():
    r"""Return the active ``cupy``-backend device compute dtype (numpy scalar type)."""
    assert 'cupy' in available_backends, "cupy backend not available."
    from lmlib.statespace.backends.rec_cupy import get_gpu_dtype as _get
    return _get()


def is_backend_available(backend):
    r"""
    Checks if the backend `backend` is available on this system. 

    Parameters
    ----------
    backend : str 
              or a list of valid backends, see [`set_backend`][lmlib.statespace.backend.set_backend]

    Returns
    ----------
    output : bool
             `True` or `False`
    """
    
    return backend in available_backends

def get_backend(cost_term):
    r"""
    Returns the name of the currently selected backend.


    Returns
    ----------
    output : str
             for a list of valid backends, see [`set_backend`][lmlib.statespace.backend.set_backend]
    """
        
    global _backend
    if isinstance(cost_term,NDCompositeCost):
        return 'numpy' #backend unspecified, selecting 'numpy' for ND
    else: 
        return _backend

# check if numba is installed, when yes import and add to available_backends.
if (spec := importlib.util.find_spec('numba')) is not None:
    # Import numba
    module = importlib.util.module_from_spec(spec)
    sys.modules['numba'] = module
    spec.loader.exec_module(module)
    available_backends = available_backends + ('jit',)

# check if cupy is installed AND a CUDA device is visible; only then advertise
# the GPU backend. Importing cupy can be slow / may raise on a driverless box,
# so guard the whole probe.
if importlib.util.find_spec('cupy') is not None:
    try:
        import cupy as _cupy_probe
        if _cupy_probe.cuda.runtime.getDeviceCount() > 0:
            available_backends = available_backends + ('cupy',)
        del _cupy_probe
    except Exception:
        pass
