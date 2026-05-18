import numpy as np

from ..backend import available_backends
from .rec_numpy import *
from .rec_lfilter import *
if 'jit' in available_backends:
    from .rec_jit import *
import warnings
from .statespace_tools import kron_q

__all__ = ['xi_q_recursion', 'xi_q_asterisk_l_recursion']


def xi_q_asterisk_l_recursion(xi_curr, q, alssm, segment, xi_prev, v, beta, backend, filter_form, numdenom, cascade_params=None):
    # Equation 47 in Baeriswyl2025

    Nq_prev = xi_prev.shape[-1]
    INq = np.eye(Nq_prev)
    A = kron_q(alssm.A, q)
    C = kron_q(alssm.C, q)

    if backend in ('jit', 'lfilter'):
        warnings.warn(
            "nD costs currently only support numpy backend. Defaulting to numpy.",
            SyntaxWarning,
            stacklevel=2,
        )
    if backend in ('numpy', 'jit', 'lfilter'):
        numpy_xi_asterisk_l_recursion(xi_curr, A, C,
                                      segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                      INq, xi_prev,
                                      v, beta)
    else:
        raise ValueError("unknown backend: '{}'".format(backend))


def xi_q_recursion(xi, q, alssm, segment, y, v, beta, backend, filter_form, numdenom, cascade_params=None):
    # Equation 18 in Baeriswyl2025

    if backend == 'numpy':
        if q == 2:
            numpy_recursion_xi2(xi,
                                alssm.A, alssm.C,
                                segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                y, v, beta)
        elif q == 1:
            numpy_recursion_xi1(xi,
                                alssm.A, alssm.C,
                                segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                y, v, beta)
        elif q == 0:
            numpy_recursion_xi0(xi,
                                alssm.A, alssm.C,
                                segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                y, v, beta)
        else:
            raise ValueError("q value not supported: '{}'".format(q))

    elif backend == 'jit':
        if q == 2:
            jit_recursion_xi2(xi,
                              alssm.A, alssm.C,
                              segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                              y, v, beta)
        elif q == 1:

            jit_recursion_xi1(xi,
                              alssm.A, alssm.C,
                              segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                              y, v, beta)
        elif q == 0:
            jit_recursion_xi0(xi,
                                alssm.A, alssm.C,
                                segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                y, v, beta)
        else:
            raise ValueError("q value not supported: '{}'".format(q))

    elif backend == 'lfilter':
        if filter_form == 'cascade':
            if q == 2:
                lfilter_cascade_xi2(xi,
                                    alssm.A, alssm.C,
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta)
            elif q == 1:
                lfilter_cascade_xi1(xi,
                                    alssm.A, alssm.C,
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta, cascade_params)
            elif q == 0:
                lfilter_cascade_xi0(xi,
                                    alssm.A, alssm.C,
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta)
            else:
                raise ValueError("q value not supported: '{}'".format(q))



        elif filter_form == 'parallel':
            if q == 2:
                lfilter_parallel_xi2(xi,
                                    numdenom[0], numdenom[1], numdenom[2],
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta)
            elif q == 1:
                # numdenom[5] and [6] carry per-row reduced IIR SOS lists from QZ-based
                # PZ cancellation; numdenom[7] and [8] carry the corresponding pole counts
                # for the gamma-shift IIR (Strategy A).  All present only when the QZ path
                # was used; absent for user-supplied numdenom dicts (5-element format).
                _iir_b = numdenom[5] if len(numdenom) > 5 else None
                _iir_a = numdenom[6] if len(numdenom) > 6 else None
                _np_b  = numdenom[7] if len(numdenom) > 7 else None
                _np_a  = numdenom[8] if len(numdenom) > 8 else None
                lfilter_parallel_xi1(xi,
                                    numdenom[0], numdenom[1], numdenom[2],
                                    numdenom[3], numdenom[4],
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta,
                                    _iir_b, _iir_a, _np_b, _np_a)
            elif q == 0:
                # lfilter_parallel_xi0 delegates to the cascade implementation
                # internally and does not use numdenom at all.
                lfilter_parallel_xi0(xi,
                                    None, None, None,
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta)
            else:
                raise ValueError("q value not supported: '{}'".format(q))
        else:
            raise ValueError("unknown filter-form: '{}'".format(filter_form))
    else:
        raise ValueError("unknown backend: '{}'".format(backend))

