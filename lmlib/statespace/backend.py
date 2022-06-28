import importlib.util
import sys

__all__ = ['set_backend', 'get_backend', 'BACKEND_TYPES', 'AVAILABLE_BACKENDS']

_backend = 'py'

BACKEND_TYPES = ('jit', 'py')
AVAILABLE_BACKENDS = ('py',)


def set_backend(backend):
    backend = backend.lower()
    global _backend
    assert backend in BACKEND_TYPES + ('python',), "Wrong backend name."
    if backend == 'jit':
        assert backend in AVAILABLE_BACKENDS + ('python',), "jit backend not availalbe. Install numba package!"
        _backend = 'jit'
    if backend in ('py', 'python'):
        _backend = 'py'


def get_backend():
    global _backend
    return _backend


# check if numba is installed, when yes import and add to AVAILABLE_BACKENDS.
if (spec := importlib.util.find_spec('numba')) is not None:
    # Import numba
    module = importlib.util.module_from_spec(spec)
    sys.modules['numba'] = module
    spec.loader.exec_module(module)
    AVAILABLE_BACKENDS = AVAILABLE_BACKENDS + ('jit',)
