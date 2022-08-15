"""Selection tool to switch between Python interpreter (default) and JIT (Just-in-Time) compilation execution for time-critical routines in package :mod:`lmlib.statespace`. """

import importlib.util
import sys

__all__ = ['set_backend', 'is_backend_available', 'get_backend', 'BACKEND_TYPES', 'available_backends']

_backend = 'py' # current backend selection (global)

BACKEND_TYPES = ('jit', 'py', 'python') # known backends
available_backends = ('py',) # available backends


def set_backend(backend):
    """
    Selects one out of multiple available backends (to optimize execution performance).

    Parameters
    ----------
    backend : str ("jit", "py", "python")
        
          - "py" (default), "python": Plain Python 
          - "jit": Just-in-Time compilation (if available)
 
    
    If the selected backend is not available, an assert is risen. 
    Use :meth:`is_backend_available` to check availability first. 
    """    
    backend = backend.lower()
    global _backend
    assert backend in BACKEND_TYPES, "Unknown backend name."
    if backend == 'jit':
        assert backend in available_backends, "jit backend not available. Check that numba package is installed!"
        _backend = 'jit'
    if backend in ('py', 'python'):
        _backend = 'py'


def is_backend_available(backend):
    """
    Checks if the backend :code:`backend` is available on this system. 

    Parameters
    ----------
    backend : str 
              or a list of valid backends, see :meth:`set_backend`

    Returns
    ----------
    output : bool
             :code:`True` or :code:`False`
    """    
    
    return (backend in available_backends)


def get_backend():
    """
    Returns the name of the currently selected backend.


    Returns
    ----------
    output : str
             for a list of valid backends, see :meth:`set_backend`
    """    
        
    global _backend
    return _backend


# check if numba is installed, when yes import and add to available_backends.
if (spec := importlib.util.find_spec('numba')) is not None:
    # Import numba
    module = importlib.util.module_from_spec(spec)
    sys.modules['numba'] = module
    spec.loader.exec_module(module)
    available_backends = available_backends + ('jit',)
