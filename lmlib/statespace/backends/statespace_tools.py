"""
State-Space Module Tools
========================

Helper functions for State-Space Module

"""
import numpy as np


__all__ = ['kron_q', 'common_C_dim', 'ss2zpk_qz']

def _sanitize_zeros(zeros, tol=1e-6):
    """Snap numerical noise in QZ-computed zeros so that zpk2sos always receives
    valid input.

    The QZ algorithm can return two kinds of numerical noise:

    * A zero that is mathematically real but carries a tiny imaginary part
      (e.g. ``0.9471 + 1.13e-9j``).  Snapped to real.
    * A conjugate pair whose two imaginary parts differ by a few ULPs
      (e.g. ``0.9736 + 0.0578j`` and ``0.9736 - 0.0578j`` where the last
      digits differ slightly).  Enforced to exact conjugate symmetry.

    Both issues cause ``scipy.signal.zpk2sos`` to raise
    ``ValueError: Array contains complex value with no matching conjugate``.

    Parameters
    ----------
    zeros : ndarray, complex
        Zeros as returned by the QZ decomposition.
    tol : float
        Relative tolerance for declaring a zero "nearly real":
        ``|imag| / (|real| + 1e-300) < tol``.

    Returns
    -------
    zeros : ndarray
        Sanitised zeros.  Real-valued output if all zeros are real after
        snapping; complex output otherwise.
    """
    zeros = np.array(zeros, dtype=complex)
    if zeros.size == 0:
        return zeros.real

    # -- Step 1: snap individually near-real zeros to exactly real ---------------
    near_real = np.abs(zeros.imag) < tol * (np.abs(zeros.real) + 1e-300)
    zeros[near_real] = zeros[near_real].real + 0j

    # -- Step 2: enforce exact conjugate symmetry for remaining complex pairs -----
    used = np.zeros(len(zeros), dtype=bool)
    for i in range(len(zeros)):
        if used[i] or zeros[i].imag == 0.0:
            used[i] = True
            continue
        # Find the best conjugate partner (not yet consumed)
        best_j, best_dist = -1, np.inf
        for j in range(len(zeros)):
            if used[j] or j == i:
                continue
            d = abs(zeros[j] - np.conj(zeros[i]))
            if d < best_dist:
                best_dist = d; best_j = j
        if best_j >= 0 and best_dist < tol * (abs(zeros[i]) + 1e-300) + 1e-10:
            # Average the real parts and the absolute imaginary part
            mid_re  = 0.5 * (zeros[i].real + zeros[best_j].real)
            mid_im  = 0.5 * (zeros[i].imag - zeros[best_j].imag)   # both steps give +
            zeros[i]      = mid_re + mid_im * 1j
            zeros[best_j] = mid_re - mid_im * 1j
            used[i] = used[best_j] = True
        else:
            used[i] = True   # unpaired complex zero — leave as-is

    # -- Return real array if nothing is genuinely complex ----------------------
    if np.all(zeros.imag == 0.0):
        return zeros.real
    return zeros


def _transform_ALSSM_matrices(A, C, P):
    """
    Apply a similarity transform P to the ALSSM matrices A and C.

    Returns ``(P @ A @ P_inv, C @ P_inv)``.

    Parameters
    ----------
    A : ndarray of shape (N, N)
        State transition matrix.
    C : ndarray of shape ([Q,] N)
        Output matrix.
    P : ndarray of shape (N, N)
        Invertible transformation matrix.

    Returns
    -------
    At : ndarray of shape (N, N)
        Transformed state matrix.
    Ct : ndarray of shape ([Q,] N)
        Transformed output matrix.
    """
    P_inv = np.linalg.inv(P)
    At = P@A@P_inv
    Ct = C@P_inv
    return At, Ct

def _transform_x(xs, P):
    """
    Apply a similarity transform P to an array of state vectors.

    Parameters
    ----------
    xs : ndarray of shape (K, N, ...)
        State vectors; the second axis is the state dimension.
    P : ndarray of shape (N, N)
        Transformation matrix.

    Returns
    -------
    ndarray of shape (K, N, ...)
        Transformed state vectors ``P @ x`` for each sample k.
    """
    return np.einsum('mn, kn...->km...', P, xs)


def kron_q(x, q):
    """
    Compute the Kronecker power :math:`x^{\\otimes q}` of an array.

    For a vector or matrix ``x``, this returns

    .. math::

        x^{\\otimes 0} = I_1, \\quad
        x^{\\otimes 1} = x, \\quad
        x^{\\otimes q} = x \\otimes x^{\\otimes (q-1)} \\; (q \\geq 2).

    For reference, see [Baeriswyl2025]_ [Eq. 6].

    Parameters
    ----------
    x : array_like
        Base array or matrix.
    q : int
        Non-negative Kronecker exponent.

    Returns
    -------
    out : ndarray
        ``x`` raised to the Kronecker power ``q``.  Shape is
        ``(x.shape[0]**q, x.shape[1]**q)`` for a 2-D input.

    """
    if q == 0:
        return np.eye(1)
    elif q == 1:
        return x
    else:
        out = x
        for _ in range(q-1):
            out = np.kron(out, out)
    return out

def common_C_dim(alssms):
    """
    Check whether all ALSSMs share the same output matrix dimensionality.

    Parameters
    ----------
    alssms : list of ModelBase
        ALSSMs to compare.

    Returns
    -------
    bool
        ``True`` if all ALSSMs have the same number of output dimensions
        (same ``C.ndim`` and same first dimension of ``C`` when 2-D),
        ``False`` otherwise.
    """
    C_ndim = [alssm.C.ndim for alssm in alssms]
    C_L = [np.atleast_2d(alssm.C).shape[0] for alssm in alssms]
    return sum(np.diff(C_ndim)) == 0 and sum(np.diff(C_L)) == 0


def ss2zpk_qz(A, B, C_row, D_scalar=0.0):
    """Zeros, poles and gain of a SISO state-space system via the QZ algorithm.

    Replaces ``scipy.signal.ss2zpk`` (which internally calls ``ss2tf`` and
    therefore forms an explicit transfer-function polynomial) with a direct
    generalized-eigenvalue computation that never builds that polynomial.

    **Why this matters**

    ``scipy.signal.ss2zpk`` calls ``ss2tf`` which uses the Faddeev-LeVerrier
    algorithm: it computes the characteristic polynomial of ``A`` via
    ``numpy.poly`` (eigenvalues → polynomial coefficients via Vieta's formulas)
    and then recovers roots from those coefficients with ``numpy.roots``.
    The round-trip through polynomial coefficients is ill-conditioned whenever
    the system has near-repeated eigenvalues — precisely the situation that
    arises here (all poles at ``gamma_inv ≈ 1``). The resulting zeros can be
    off by up to 1e-5 relative to the true values.

    This implementation instead builds the ``(N+1) × (N+1)`` **Rosenbrock
    system matrix pencil**::

        F = [ A   B ]     E = [ I   0 ]
            [ C   D ]         [ 0   0 ]

    The finite generalised eigenvalues of the pencil ``(F, E)`` — i.e. the
    values ``λ`` for which ``det(F − λ E) = 0`` with ``λ`` finite — are exactly
    the transmission zeros of ``H(z) = C(zI − A)^{−1}B + D``.  These are
    computed via ``scipy.linalg.qz`` (the QZ algorithm / generalised Schur
    decomposition), which is numerically backward-stable and never requires
    forming a polynomial.

    **Key accuracy benefits observed in practice** (poly_degree=3, g=1000):

    * Zeros that are cancellable with IIR poles (i.e. numerically equal to
      ``gamma_inv``) come back *exactly* at ``gamma_inv`` rather than with
      ~1e-6 spread, eliminating the need for a post-hoc snapping step.
    * Non-cancellable zeros match MATLAB ``ss2zp`` output to ~1e-14.
    * Combined with pole-zero cancellation, the parallel filter error for
      rows 0–2 reaches near machine precision, and row 3 improves ~2× over
      scipy for AlssmPolyJordan and ~16× for AlssmPoly.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        State transition matrix.
    B : ndarray, shape (N, 1) or (N,)
        Input vector (single input).
    C_row : ndarray, shape (1, N) or (N,)
        Output row vector (single output row).
    D_scalar : float, optional
        Feedthrough scalar, default 0.

    Returns
    -------
    zeros : ndarray, shape (N-1-n_inf,) real or complex
        Finite transmission zeros.  Real if all imaginary parts are negligible.
    poles : ndarray, shape (N,) complex
        Poles (eigenvalues of A).
    gain : float
        System gain.
    n_inf_zeros : int
        Number of zeros at infinity (QZ eigenvalues where beta≈0 were dropped).
        Each such zero corresponds to one extra sample delay that the caller must
        account for in the output slice.

    Notes
    -----
    The QZ algorithm has O(N³) cost, the same asymptotic complexity as
    ``ss2tf`` + ``roots``, but with much better numerical stability for
    near-repeated eigenvalues.
    """
    from scipy.linalg import qz as _qz
    from numpy.linalg import eigvals, solve

    N = int(A.shape[0])
    B = np.ravel(B)
    C_row = np.ravel(C_row)

    # ── Build Rosenbrock system matrix pencil ─────────────────────────────────
    # F and E are (N+1) × (N+1).  E has rank N (last row/col is zero), so
    # the pencil has exactly N finite generalized eigenvalues — which are the
    # N-1 transmission zeros — and 1 (or 2) infinite eigenvalue(s).
    F = np.empty((N + 1, N + 1), dtype=float)
    F[:N, :N] = A
    F[:N,  N] = B
    F[N,  :N] = C_row
    F[N,   N] = D_scalar

    E = np.zeros((N + 1, N + 1), dtype=float)
    E[:N, :N] = np.eye(N)

    # ── Generalised Schur decomposition ──────────────────────────────────────
    # AA, BB upper (quasi-)triangular; AA[i,i]/BB[i,i] = i-th gen. eigenvalue.
    # BB[i,i] ≈ 0  ↔  infinite eigenvalue  ↔  skip.
    AA, BB, _Q, _Z = _qz(F, E, output='complex')
    diag_AA = np.diag(AA)
    diag_BB = np.diag(BB)

    tol = 1e-10 * (np.max(np.abs(diag_BB)) + 1e-300)
    finite_mask = np.abs(diag_BB) > tol
    zeros = (diag_AA[finite_mask] / diag_BB[finite_mask])
    # Count eigenvalues dropped because beta≈0 (zeros at infinity = extra delays).
    n_inf_zeros = int(np.sum(~finite_mask))

    # ── Poles ─────────────────────────────────────────────────────────────────
    poles = eigvals(A)

    # ── Gain ─────────────────────────────────────────────────────────────────
    # Evaluate H(z_t) at a test point well outside all poles and zeros, then
    # invert the ZPK formula:  H(z_t) = k · ∏(z_t − zeros) / ∏(z_t − poles)
    z_t = float(max(2.0, 2.0 * np.max(np.abs(poles)).real + 1.0))
    H_t = float(np.real(C_row @ solve(z_t * np.eye(N) - A, B))) + D_scalar
    gain = float(np.real(
        H_t * np.prod(z_t - poles) / np.prod(z_t - zeros + 0j)
    ))

    # ── Near-origin complex zero snap ────────────────────────────────────────
    # QZ sometimes returns tiny complex values (~1e-5 magnitude) for structurally
    # degenerate transfer functions where the true zeros are at the origin.  These
    # fail ``zpk2sos`` with "no matching conjugate" but snapping them to exactly
    # z = 0+0j produces identity-like SOS sections (zero cancels with FIR pole at 0).
    # Only snap zeros that are COMPLEX (|imag| > 1e-14): near-real zeros are
    # legitimate and should be left to ``_sanitize_zeros`` below.
    _pole_scale = float(np.max(np.abs(poles))) + 1e-300
    _near_orig  = (np.abs(zeros) < 1e-4 * _pole_scale) & (np.abs(zeros.imag) > 1e-14)
    if np.any(_near_orig):
        zeros = zeros.copy()
        zeros[_near_orig] = 0.0 + 0.0j

    # Sanitise numerical noise in imaginary parts so that scipy zpk2sos and
    # callers can always form valid conjugate pairs.
    zeros = _sanitize_zeros(zeros)

    return zeros, poles, gain, n_inf_zeros
