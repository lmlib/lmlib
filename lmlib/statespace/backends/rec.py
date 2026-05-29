import numpy as np

from ..backend import available_backends
from .rec_numpy import *
from .rec_lfilter import *
from .rec_lfilter import _compute_cascade_params_asterisk, _build_parallel_ast_sos
if 'jit' in available_backends:
    from .rec_jit import *
import warnings
from .statespace_tools import kron_q

__all__ = ['xi_q_recursion', 'xi_q_asterisk_l_recursion']


def xi_q_asterisk_l_recursion(xi_curr, q, alssm, segment, xi_prev, v, beta, backend, filter_form, numdenom, cascade_params=None):
    """
    Run the :math:`\\xi^{(q)*l}` multi-dimensional recursion (Eq. 47 in [Baeriswyl2025]).

    Computes the cross-dimensional xi terms needed for N-dimensional cost
    functions by applying the Kronecker-structured recursion in the numpy
    backend (jit/lfilter fall back to numpy with a warning).

    Parameters
    ----------
    xi_curr : ndarray
        Output buffer; updated in-place.
    q : int
        Recursion order (0, 1, or 2).
    alssm : ModelBase
        ALSSM defining A and C matrices.
    segment : Segment
        Segment defining the window parameters and direction.
    xi_prev : ndarray
        Previously computed xi values from the lower-dimensional recursion.
    v : ndarray
        Sample weights.
    beta : float
        Cost scaling factor.
    backend : str
        Computational backend (``'numpy'``, ``'jit'``, or ``'lfilter'``).
    filter_form : str
        Filter structure (``'cascade'`` or ``'parallel'``).
    numdenom : list or None
        Pre-built transfer-function coefficients for the lfilter backend.
    cascade_params : list or None, optional
        Pre-built cascade parameters for the lfilter cascade backend.
    """

    Nq_prev = xi_prev.shape[-1]
    INq = np.eye(Nq_prev)
    A = kron_q(alssm.A, q)
    C = kron_q(alssm.C, q)

    if backend == 'jit':
        jit_xi_asterisk_l_recursion(xi_curr, A, C,
                                    segment.a, segment.b, segment.direction,
                                    segment.delta, segment.gamma,
                                    INq, xi_prev,
                                    v, beta)
        return

    if backend == 'numpy':
        numpy_xi_asterisk_l_recursion(xi_curr, A, C,
                                      segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                      INq, xi_prev,
                                      v, beta)
    elif backend == 'lfilter':
        if filter_form == 'cascade':
            # q==0 and q==1: lfilter cascade supports upper-triangular A.
            # q==2 falls through to numpy.
            if q in (0, 1):
                # q==0: kron_q(A,0)=[[1.]], kron_q(C,0)=[[1.]] -- scalar IIR,
                #        identical to q==1 cascade with A=[[1.]], C=[[1.]].
                # q==1: base-model A and C directly.
                try:
                    cp_ast = _compute_cascade_params_asterisk(
                        A, C,
                        segment.a, segment.b, segment.delta, segment.gamma,
                        Nq_prev, segment.direction,
                    )
                    if segment.direction == 'fw':
                        lfilter_xi_asterisk_l_forward_cascade_recursion(
                            xi_curr, cp_ast, segment.a, segment.b, xi_prev, v, beta)
                    else:
                        lfilter_xi_asterisk_l_backward_cascade_recursion(
                            xi_curr, cp_ast, segment.a, segment.b, xi_prev, v, beta)
                except ValueError:
                    # A is not upper triangular — fall back to numpy
                    numpy_xi_asterisk_l_recursion(xi_curr, A, C,
                                                  segment.a, segment.b, segment.direction,
                                                  segment.delta, segment.gamma,
                                                  INq, xi_prev, v, beta)
            else:
                # q==2: fall back to numpy
                numpy_xi_asterisk_l_recursion(xi_curr, A, C,
                                              segment.a, segment.b, segment.direction,
                                              segment.delta, segment.gamma,
                                              INq, xi_prev, v, beta)

        elif filter_form == 'parallel':
            # q==0 and q==1: build SOS on-the-fly from A, C and apply parallel filter.
            # A=kron_q(A,0)=[[1.]] for q==0 is automatically handled (scalar IIR).
            # q==2 falls through to numpy.
            if q in (0, 1):
                try:
                    nd_ast = _build_parallel_ast_sos(
                        A, C,
                        segment.a, segment.b, segment.delta, segment.gamma,
                        segment.direction,
                    )
                    if segment.direction == 'fw':
                        lfilter_xi_asterisk_l_forward_parallel_recursion(
                            xi_curr, nd_ast,
                            segment.a, segment.b, segment.delta, segment.gamma,
                            xi_prev, v, beta)
                    else:
                        lfilter_xi_asterisk_l_backward_parallel_recursion(
                            xi_curr, nd_ast,
                            segment.a, segment.b, segment.delta, segment.gamma,
                            xi_prev, v, beta)
                except Exception:
                    # Any failure (non-invertible A, numerical issues) — fall back
                    numpy_xi_asterisk_l_recursion(xi_curr, A, C,
                                                  segment.a, segment.b, segment.direction,
                                                  segment.delta, segment.gamma,
                                                  INq, xi_prev, v, beta)
            else:
                # q==2: fall back to numpy
                numpy_xi_asterisk_l_recursion(xi_curr, A, C,
                                              segment.a, segment.b, segment.direction,
                                              segment.delta, segment.gamma,
                                              INq, xi_prev, v, beta)
        else:
            raise ValueError("unknown filter-form: '{}'".format(filter_form))


def xi_q_recursion(xi, q, alssm, segment, y, v, beta, backend, filter_form, numdenom, cascade_params=None):
    """
    Run the :math:`\\xi^{(q)}` recursion to the selected backend (Eq. 18 in [Baeriswyl2025]).

    Parameters
    ----------
    xi : ndarray
        Output buffer; updated in-place.
    q : int
        Recursion order: 2 for W, 1 for xi, 0 for kappa.
    alssm : ModelBase
        ALSSM providing A and C.
    segment : Segment
        Window parameters and direction.
    y : ndarray
        Input signal samples.
    v : ndarray
        Per-sample weights.
    beta : float
        Cost scaling factor.
    backend : str
        Computational backend (``'numpy'``, ``'lfilter'``, or ``'jit'``).
    filter_form : str
        Internal filter structure (``'cascade'`` or ``'parallel'``).
    numdenom : list or None
        Pre-built transfer-function coefficients (lfilter parallel only).
    cascade_params : list or None, optional
        Pre-built cascade parameters (lfilter cascade + q==1 only).
    """

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
