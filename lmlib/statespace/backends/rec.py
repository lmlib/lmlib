import numpy as np

from ..backend import available_backends
from .rec_numpy import *
from .rec_lfilter import *
from .rec_lfilter import _compute_cascade_params_asterisk, _build_parallel_ast_sos
if 'jit' in available_backends:
    from .rec_jit import *
if 'cupy' in available_backends:
    from .rec_cupy import *
import warnings
from .statespace_tools import kron_q

__all__ = ['xi_q_recursion', 'xi_q_asterisk_l_recursion']


def xi_q_asterisk_l_recursion(xi_curr, q, alssm, segment, xi_prev, v, beta, backend, filter_form, block_sizes=None):
    r"""
    Run the $\xi^{(q)*l}$ multi-dimensional recursion (Eq. 47 in [Baeriswyl2025]).

    Computes the cross-dimensional xi terms for N-dimensional cost functions via
    the Kronecker-structured recursion.  One combined ``AlssmSum`` per segment is
    passed in; the backend realizes it.

    Backends: ``numpy``/``jit`` use the dense Kronecker recursion.  For
    ``lfilter`` the ``q==1`` asterisk step uses the genuine **parallel**
    realization when ``filter_form='parallel'`` (per-ALSSM-block transfer
    functions + Kronecker scatter), and the **cascade** realization otherwise
    (and for ``q==0``).  

    Parameters
    ----------
    xi_curr : ndarray
        Output buffer; updated in-place.
    q : int
        Recursion order (0, 1, or 2).
    alssm : ModelBase
        Combined ALSSM (block-diagonal A) for the current dimension/segment.
    segment : Segment
        Window parameters and direction.
    xi_prev : ndarray
        Accumulated xi from the lower-dimensional recursion.
    v : ndarray
        Sample weights.
    beta : float
        Cost scaling factor.
    backend : str
        ``'numpy'``, ``'jit'``, or ``'lfilter'``.
    filter_form : str
        ``'cascade'`` or ``'parallel'``
    block_sizes : list or None
        Per-ALSSM state orders of the combined ``A``; used by the parallel
        ``q==1`` path to split the recursion per ALSSM block.
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
        return

    if backend == 'cupy':
        # The cross-dimensional (ND) asterisk recursion is not reimplemented on
        # the GPU; the GPU backend targets the 1-D cascade path. Delegate to the
        # numpy realization so ND costs still work with backend='cupy'.
        numpy_xi_asterisk_l_recursion(xi_curr, A, C,
                                      segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                      INq, xi_prev,
                                      v, beta)
        return

    if backend == 'lfilter':
        # q==1 parallel: genuine parallel realization (Option A) — per-ALSSM-block
        # split + Kronecker scatter, falling back to numpy on any failure.
        if filter_form == 'parallel' and q == 1:
            try:
                lfilter_parallel_xi_asterisk_split(
                    xi_curr, A, C,
                    segment.a, segment.b, segment.delta, segment.gamma,
                    segment.direction, xi_prev, v, beta, block_sizes)
            except Exception:
                numpy_xi_asterisk_l_recursion(xi_curr, A, C,
                                              segment.a, segment.b, segment.direction,
                                              segment.delta, segment.gamma,
                                              INq, xi_prev, v, beta)
            return

        # cascade (and parallel q==0): realize the asterisk via the cascade form.
        # q==0/q==1 use the cascade asterisk (upper-triangular A); q==2 (and
        # non-upper-triangular A) fall back to numpy.
        if q in (0, 1):
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
        return

    raise ValueError("unknown backend: '{}'".format(backend))


def xi_q_recursion(xi, q, alssm, segment, y, v, beta, backend, filter_form, block_sizes=None, parallel_plan=None):
    r"""
    Run the $\xi^{(q)}$ recursion on the selected backend (Eq. 18 in [Baeriswyl2025]).

    A single combined ``AlssmSum`` (block-diagonal ``A``) is passed per segment;
    the backend chooses the realization:

    - ``numpy`` / ``jit`` : dense recursion on the combined ``A``.
    - ``lfilter`` / ``cascade`` : block-aware single-pass cascade (``block_sizes``
      lets it skip the structurally-zero cross-block feed-forward terms).
    - ``lfilter`` / ``parallel`` : per-ALSSM split (``parallel_plan`` carries the
      pre-built per-block transfer-function coefficients from
      [`build_parallel_numdenom`][lmlib.statespace.backends.rec_lfilter.build_parallel_numdenom]).

    Parameters
    ----------
    xi : ndarray
        Output buffer; updated in-place.
    q : int
        Recursion order: 2 for W, 1 for xi, 0 for kappa.
    alssm : ModelBase
        Combined ALSSM providing A and C.
    segment : Segment
        Window parameters and direction.
    y : ndarray
        Input signal samples.
    v : ndarray
        Per-sample weights.
    beta : float
        Cost scaling factor.
    backend : str
        ``'numpy'``, ``'lfilter'``, or ``'jit'``.
    filter_form : str
        ``'cascade'`` or ``'parallel'`` (lfilter only).
    block_sizes : list or None
        Per-ALSSM state orders of the combined ``A`` (lfilter cascade, q==1).
    parallel_plan : list or None
        Per-block transfer-function coefficients (lfilter parallel, q==1);
        see [`build_parallel_numdenom`][lmlib.statespace.backends.rec_lfilter.build_parallel_numdenom].
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
                                    y, v, beta, block_sizes)
            elif q == 0:
                lfilter_cascade_xi0(xi,
                                    alssm.A, alssm.C,
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta)
            else:
                raise ValueError("q value not supported: '{}'".format(q))

        elif filter_form == 'parallel':
            if q == 2:
                # Parallel W is not implemented; steady-state mode avoids this path.
                raise NotImplementedError("lfilter_parallel_xi2 not implemented yet.")
            elif q == 1:
                lfilter_parallel_xi1_split(xi, parallel_plan,
                                           segment.a, segment.b, segment.direction,
                                           segment.delta, segment.gamma, y, v, beta)
            elif q == 0:
                # lfilter_parallel_xi0 delegates to the cascade implementation
                # internally and does not use the ALSSM at all.
                lfilter_parallel_xi0(xi,
                                    None, None, None,
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta)
            else:
                raise ValueError("q value not supported: '{}'".format(q))
        else:
            raise ValueError("unknown filter-form: '{}'".format(filter_form))
    elif backend == 'cupy':
        # GPU cascade backend (1-D, upper-triangular A). The parallel filter
        # form is not reimplemented on the GPU; route it to the numpy backend.
        if filter_form == 'cascade':
            if q == 2:
                cupy_cascade_xi2(xi,
                                 alssm.A, alssm.C,
                                 segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                 y, v, beta)
            elif q == 1:
                cupy_cascade_xi1(xi,
                                 alssm.A, alssm.C,
                                 segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                 y, v, beta, block_sizes)
            elif q == 0:
                cupy_cascade_xi0(xi,
                                 alssm.A, alssm.C,
                                 segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                 y, v, beta)
            else:
                raise ValueError("q value not supported: '{}'".format(q))
        else:
            # parallel form on GPU is not implemented; fall back to numpy.
            if q == 2:
                numpy_recursion_xi2(xi, alssm.A, alssm.C,
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta)
            elif q == 1:
                numpy_recursion_xi1(xi, alssm.A, alssm.C,
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta)
            elif q == 0:
                numpy_recursion_xi0(xi, alssm.A, alssm.C,
                                    segment.a, segment.b, segment.direction, segment.delta, segment.gamma,
                                    y, v, beta)
            else:
                raise ValueError("q value not supported: '{}'".format(q))
    else:
        raise ValueError("unknown backend: '{}'".format(backend))
