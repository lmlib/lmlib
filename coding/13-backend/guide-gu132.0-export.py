"""
Export of Transfer-Function Coefficients [gu132.0]
==================================================

Shows how to extract the IIR/FIR transfer-function coefficients from an
:class:`~lmlib.statespace.cost.CostSegment` and use them to implement the
RLS filter directly with :func:`scipy.signal.lfilter`.

The exported coefficients ``(q_a, q_b, p)`` encode the boundary FIR
numerators (at indices ``a`` and ``b``) and the shared IIR denominator ``p``.
The helper function ``filter_direct_form`` reproduces the
:meth:`~lmlib.statespace.rls.RLSAlssm.filter` output from these coefficients
without using lmlib at all, making the result portable to other environments.

"""
import numpy as np
from scipy.signal import ss2tf, lfilter

import lmlib as lm


K = 1_000_000
y = np.random.randn(K)

# setup model
alssm = lm.AlssmPoly(poly_degree=1)
seg_l = lm.Segment(a=-21, b=-1, direction=lm.FW, g=100)
cost = lm.CostSegment(alssm, seg_l)



def export_solver_task(cost, filter_from=None):
    """
    Extract transfer-function coefficients from a CostSegment.

    Computes the boundary FIR numerator vectors ``q_a`` and ``q_b`` and the
    shared IIR denominator ``p`` that together implement the ALSSM recursive
    filter as a set of :func:`scipy.signal.lfilter` calls.

    Parameters
    ----------
    cost : lm.CostSegment
        Cost segment whose ALSSM and segment define the filter.
    filter_from : ignored
        Reserved for future use; currently unused.

    Returns
    -------
    q_a : ndarray of shape (N, N)
        FIR numerator matrix at boundary ``a``.
    q_b : ndarray of shape (N, N)
        FIR numerator matrix at boundary ``b``.
    p : ndarray of shape (N,)
        IIR denominator (shared across all rows).
    shift_a : int
        Sample-index shift associated with boundary ``a``.
    shift_b : int
        Sample-index shift associated with boundary ``b``.

    Raises
    ------
    TypeError
        If ``cost`` is not a :class:`~lmlib.statespace.cost.CostSegment`.
    """

    alssm = cost.alssm
    segment = cost.segment

    A = alssm.A
    C = alssm.C
    N = alssm.N

    Css = np.eye(N)
    Dss = np.zeros((N, 1))

    # TF of system using ss2tf
    # ------- a boundary --------

    # Bss = np.linalg.matrix_power(gamma * A, a).T @ C.reshape(1, N).T  # use this if the input signal is filtered directly
    Bssgamma1 = np.linalg.matrix_power(A, segment.a - 1).T @ C.reshape(1, N).T # use this if the input signal is multiplied with gamma^s
    Ass = np.linalg.inv(A * segment.gamma).T
    q_a, p = ss2tf(Ass, Bssgamma1, Css, Dss)

    # ------- b boundary --------
    Css = np.eye(N)
    Dss = np.zeros((N, 1))

    # Bss = np.linalg.matrix_power(gamma * A, b).T @ C.reshape(1, N).T  # use this if the input signal is filtered directly
    Bssgamma1 = np.linalg.matrix_power(A, segment.b).T @ C.reshape(1, N).T # use this if the input signal is multiplied with gamma^s
    Ass = np.linalg.inv(A * segment.gamma).T
    q_b, _ = ss2tf(Ass, Bssgamma1, Css, Dss)

    shift_a = segment.a-1
    shift_b = segment.b

    return q_a, q_b, p, shift_a, shift_b

# Single channel
lm.set_backend('lfilter')
rls = lm.RLSAlssm(cost, filter_form='cascade', steady_state=False, calc_W=False, calc_kappa=False, calc_nu=False)
rls.filter(y)

q_a, q_b, p, shift_a, shift_b = export_solver_task(cost)



# EXAMPLE IMPLEMENTATION IN PYTHON
def filter_direct_form(q_a, q_b, p, gamma, shift_a, shift_b):
    """
    Implement the ALSSM RLS filter directly using scipy lfilter calls.

    Reproduces the output of :meth:`~lmlib.statespace.rls.RLSAlssm.filter`
    from the exported transfer-function coefficients, without any lmlib
    dependency at runtime.  Intended as a reference implementation for
    embedding in external systems.

    Parameters
    ----------
    q_a : ndarray of shape (N, N)
        FIR numerator matrix at boundary ``a``, as returned by
        :func:`export_solver_task`.
    q_b : ndarray of shape (N, N)
        FIR numerator matrix at boundary ``b``.
    p : ndarray of shape (N,)
        IIR denominator (shared across rows).
    gamma : float
        Window decay factor of the segment.
    shift_a : int
        Sample-index shift for boundary ``a``.
    shift_b : int
        Sample-index shift for boundary ``b``.

    Returns
    -------
    xi : ndarray of shape (K, N)
        Reconstructed cross-correlation :math:`\\xi_k`, equivalent to
        :attr:`~lmlib.statespace.rls.RLSAlssm.xi`.
    """

    def FIR_filter(b, y):
        N = np.shape(b)[0]
        out = np.zeros(np.insert(np.shape(y), 1, N))
        a = np.zeros_like(b[0])
        a[0] = 1
        for n in range(N):
            out[:, n] = lfilter(b[n], a, y)
        return out

    def shift_signal(y, a):
        y_shifted = np.zeros_like(y)
        # shift fir outputs
        s_signal = (
                a + 1
        )  # signal is shifted +1 compared to state space system (check recursion formulas)
        if s_signal == 0:
            return y
        if s_signal > 0:
            y_shifted[: K - s_signal] = y[s_signal:]
        if s_signal < 0:
            y_shifted[-s_signal:] = y[: K + s_signal]
        return y_shifted

    def IIR_filter(a, y):
        N = np.shape(y)[-1]
        out = np.zeros_like(y)
        b = np.zeros_like(a)
        b[0] = 1
        for n in range(N):
            out[:, n] = lfilter(b, a, y[:, n])
        return out

    FIR_b = shift_signal(FIR_filter(q_b, y * gamma**shift_b), shift_b)
    FIR_a = shift_signal(FIR_filter(q_a, y * gamma**shift_a), shift_a)
    FIR_diff = FIR_b - FIR_a
    xi = IIR_filter(p, FIR_diff)
    return xi


xi = filter_direct_form(q_a, q_b, p, seg_l.gamma, shift_a, shift_b)
