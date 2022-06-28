import warnings
import functools
import numpy as np

__all__ = ['is_square', 'is_2dim', 'is_1dim', 'is_array_like', 'is_string',
           'info_str_found_shape', 'info_str_found_type', 'common_C_dim',
           'deprecated']


def is_2dim(arr): return True if np.ndim(arr) == 2 else False


def is_1dim(arr): return True if np.ndim(arr) == 1 else False


def is_square(arr):
    if not is_2dim(arr):
        return False
    return True if np.diff(np.shape(arr)) == 0 else False


def is_array_like(arr):
    return isinstance(arr, (list, tuple, np.ndarray))


def info_str_found_shape(arr):
    return f'found shape: {np.shape(arr)}'


def common_C_dim(alssms):
    C_ndim = [alssm.C.ndim for alssm in alssms]
    C_L = [np.atleast_2d(alssm.C).shape[0] for alssm in alssms]
    return sum(np.diff(C_ndim)) == 0 and sum(np.diff(C_L)) == 0


def is_string(s):
    return isinstance(s, str)


def info_str_found_type(s):
    return f'found type: {type(s)}'


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func
