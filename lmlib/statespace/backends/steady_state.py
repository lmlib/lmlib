import numpy as np
from numpy.linalg import matrix_power, cond, inv
from numpy.polynomial.legendre import legvander as _legvander
import scipy.linalg as _sp

__all__ = [
    'covariance_matrix_closed_form',
    'covariance_matrix_schur',
    'covariance_matrix_legendre',
    'covariance_matrix_limited_sum',
]


def _is_legendre_shift(A, tol=1e-10):
    r"""Return ``(True, h)`` if *A* is a Legendre shift matrix, else ``(False, None)``.

    A Legendre shift matrix satisfies

    .. math::  \phi(t + h) = \phi(t)\,A

    for some step size :math:`h = 2/(W-1)`.  It is upper-triangular with ones on
    the diagonal, and every column *n* equals the Taylor expansion of
    :math:`P_n(t + h)` in the Legendre basis (see :class:`AlssmPolyLegendre`).

    The test verifies ALL columns of *A* against the exact Taylor expansion of
    :math:`P_n(t+h)` in the Legendre basis, computed via successive applications
    of :func:`numpy.polynomial.legendre.legder`.  This is the only check that
    is both necessary and sufficient: earlier versions that only tested columns
    1 and 2 produced false positives for:

    * Jordan blocks (``h=1``, ``A[0,2]=0``): caught by the column-2 check.
    * Block-diagonal Legendre matrices (e.g. from :class:`CompositeCost` wrapping
      two Legendre ALSSMs into one ``AlssmSum``): the top-left block passes
      all scalar checks, but columns ``n ≥ N_sub`` (belonging to the second
      block) fail the full Taylor-expansion check.
    """
    from numpy.polynomial.legendre import legder as _legder_local
    N = A.shape[0]
    if N < 2:
        return False, None
    # Upper triangular with ones on diagonal
    if not np.allclose(np.tril(A, -1), 0, atol=tol):
        return False, None
    if not np.allclose(np.diag(A), 1.0, atol=tol):
        return False, None
    # Recover h from the super-diagonal entry A[0, 1]
    h = A[0, 1]
    if h <= 0 or h > 2.0:
        return False, None
    # Verify ALL columns against the exact Taylor expansion of P_n(t+h).
    # This rejects Jordan blocks (A[0,2]=0 ≠ 3h²/2 for h=1, N≥3) AND
    # block-diagonal matrices whose off-diagonal blocks break column n≥N_sub.
    for n in range(1, N):
        c_deriv = np.zeros(N); c_deriv[n] = 1.0
        expected_col = np.zeros(N)
        h_pow = 1.0
        for m in range(n + 1):
            pad = N - len(c_deriv)
            expected_col += h_pow * np.pad(c_deriv, (0, pad))
            c_deriv = _legder_local(c_deriv)
            h_pow *= h / (m + 1)
        if not np.allclose(A[:, n], expected_col, atol=tol, rtol=1e-6):
            return False, None
    return True, h


def covariance_matrix_legendre(A, C, gamma, a, b, delta):
    r"""Exact steady-state Gram matrix for :class:`AlssmPolyLegendre` via direct summation.

    Exploits the algebraic identity

    .. math::

        C\,A^j = \phi\!\left(-1 + j\,h\right)
                 = \bigl[P_0(-1+jh),\;\ldots,\;P_D(-1+jh)\bigr]

    so that the design-matrix row at lag *j* is just one row of
    :func:`numpy.polynomial.legendre.legvander` evaluated at :math:`t_j = -1 + jh`.
    No matrix powers are computed; the result is therefore **exact to machine
    precision** regardless of degree:

    .. math::

        W = \gamma^{-\delta}
            \sum_{j=a}^{b} \gamma^j\,\phi(t_j)^{\top}\,\phi(t_j)
          = \gamma^{-\delta}\,V^{\top}\,\mathrm{diag}(\gamma^j)\,V

    where :math:`V \in \mathbb{R}^{(b-a+1) \times N}` is the Legendre Vandermonde
    matrix on :math:`t_j \in [-1 + a\,h,\;\ldots,\;-1 + b\,h]`.

    Parameters
    ----------
    A, C, gamma, a, b, delta
        Same as :func:`covariance_matrix_schur`.
    """
    is_leg, h = _is_legendre_shift(np.asarray(A, dtype=float))
    if not is_leg:
        raise ValueError('covariance_matrix_legendre requires a Legendre shift matrix A.')

    deg    = np.asarray(A).shape[0] - 1
    a_int  = int(a)
    b_int  = int(b)
    j_vals = np.arange(a_int, b_int + 1, dtype=float)

    # Recover the segment offset encoded in C.
    # When AlssmPolyLegendre is initialised with a_seg != 0 its output vector is
    # C_new = phi(-1) A^{-a_seg}, so C_new[1] = P_1(-1 + (-a_seg)*h) = -1 + (-a_seg)*h.
    # Rearranging:  offset = -a_seg = round((C[1] + 1) / h).
    # For the standard case (a_seg=0):  C[1] = -1, offset = 0.
    C_vec = np.atleast_1d(np.asarray(C, dtype=float)).ravel()
    offset = int(round((C_vec[1] + 1.0) / h)) if deg >= 1 else 0

    # Design-matrix rows:  V[j-a_int, :] = C A^j = phi(-1 + (j + offset)*h)
    # = phi(-1 + (j - a_seg)*h), which spans [-1, +1] when offset == -a_seg == |a|.
    t_j = -1.0 + (j_vals + offset) * h

    V = _legvander(t_j, deg)                     # (b-a+1, N), exact
    w = gamma ** j_vals                          # exponential weights
    W = V.T @ (w[:, np.newaxis] * V)             # (N, N) = V^T diag(w) V

    return float(gamma ** (-delta)) * W


def covariance_matrix_schur(A, C, gamma, a, b, delta):
    r"""Numerically stable steady-state Gram matrix W via a Schur-based solver.

    Computes the same matrix as :func:`covariance_matrix_closed_form`

    .. math::

        W = \gamma^{-\delta}
            \bigl(\mathbf{I}_N \otimes C\bigr)
            \,M\,
            \bigl(C^{\top} \otimes \mathbf{I}_N\bigr)

    where :math:`M` is the finite geometric series

    .. math::

        M = \sum_{t=a}^{b} \gamma^t \,
            (A^t)^{\!\top} \otimes A^t
          = \gamma^a (A^a)^{\!\top} \!\otimes A^a
            + \cdots
            + \gamma^b (A^b)^{\!\top} \!\otimes A^b

    which satisfies the Stein equation
    :math:`M - \gamma\,(A^{\top} \!\otimes A)\,M\,=\,P_a - P_{b+1}` (or
    the reverse inequality for :math:`\gamma > 1`).

    **AlssmPolyLegendre fast path**

    When *A* is detected to be a Legendre shift matrix (upper-triangular, unit
    diagonal, valid step size *h*), this function automatically delegates to
    :func:`covariance_matrix_legendre` instead of solving the Stein equation.

    That path computes :math:`W = V^{\top}\,\mathrm{diag}(\gamma^j)\,V` using
    :func:`numpy.polynomial.legendre.legvander`, which is exact to machine
    precision for **any** polynomial degree and avoids the ill-conditioned
    :math:`N^2 \times N^2` Kronecker system entirely.

    **Why this is more stable than** :func:`covariance_matrix_closed_form`

    The closed-form function inverts the matrix
    :math:`I_{N^2} - \gamma\,(A^{\top} \!\otimes A)` (or its inverse for
    :math:`\gamma > 1`) via :func:`numpy.linalg.inv`.  Forming the explicit
    inverse of a Sylvester/Stein coefficient matrix amplifies rounding errors
    by :math:`\kappa^2`, where :math:`\kappa` is its condition number.

    This function instead solves the Stein equation

    .. math::

        X - \gamma\,(A^{\top} \!\otimes A)\,X = P_a - P_{b+1}

    directly with :func:`scipy.linalg.solve`, exploiting the fact that the
    coefficient matrix is available explicitly.  The linear solve (LU
    factorisation) needs only :math:`O(N^6)` work (same as the explicit
    inverse) but achieves a backward error of order :math:`u \kappa` rather
    than :math:`u \kappa^2`.

    For further robustness the right-hand side matrices :math:`P_a` and
    :math:`P_{b+1}` are formed via :func:`numpy.linalg.matrix_power` (which
    itself uses repeated squaring and is no more expensive than in the
    closed-form variant).

    Parameters
    ----------
    A : array_like of shape (N, N)
        State-transition matrix of the ALSSM.
    C : array_like of shape (N,) or (1, N)
        Output row vector of the ALSSM.
    gamma : float
        Exponential window decay factor.  ``gamma < 1`` for backward windows,
        ``gamma > 1`` for forward windows.
    a, b : int or float
        Segment boundaries (relative sample indices).  Use ``-np.inf`` /
        ``np.inf`` for one-sided infinite windows.
    delta : int
        Window reference offset.

    Returns
    -------
    W : ndarray of shape (N, N)
        Steady-state Gram matrix.

    Warns
    -----
    UserWarning
        Emitted when the Stein coefficient matrix is itself ill-conditioned
        (condition number > 1e15), which means the result will still carry
        significant numerical error even with the stable solver.  This
        typically occurs when ``g`` is chosen too large or the segment is very
        wide; the same warning is also produced by
        :func:`covariance_matrix_closed_form`.
        Not emitted when the Legendre fast path is taken (that path is always
        exact to machine precision).
    """
    A = np.asarray(A, dtype=float)

    # ── Fast path: AlssmPolyLegendre ─────────────────────────────────────────────
    # When A is a Legendre shift matrix and gamma < 1, the Stein equation in
    # the N^2 × N^2 Kronecker space becomes severely ill-conditioned (singular
    # values spanning 10^40+ at deg=20) despite having well-separated eigenvalues
    # (all equal to 1-gamma).  The direct formula W = V^T diag(gamma^j) V via
    # legvander is exact to machine precision for any degree and O(W*N^2).
    if not np.isinf(a) and not np.isinf(b):
        is_leg, _ = _is_legendre_shift(A)
        if is_leg:
            return covariance_matrix_legendre(A, C, gamma, a, b, delta)

    # ── General path: Stein equation in N^2 × N^2 ───────────────────────────
    N = int(np.shape(A)[0])
    N2 = N * N
    gATA = gamma * np.kron(np.transpose(A), A)   # (N^2, N^2)

    # Stein equation (telescoping identity, valid for ALL gamma on finite segments):
    #   M - gATA @ M = gATA^a - gATA^(b+1)
    #   (I - gATA) M = gATA^a - gATA^(b+1)
    # No separate gamma > 1 branch is needed: the coefficient matrix (I - gATA)
    # is always non-singular for finite [a, b] because the finite sum is well-defined.
    coeff = np.eye(N2) - gATA                     # coefficient matrix
    P_lo  = matrix_power(gATA, a)        if not np.isinf(a) else np.zeros((N2, N2))
    P_hi  = matrix_power(gATA, b + 1)   if not np.isinf(b) else np.zeros((N2, N2))

    cond_coeff = cond(coeff)
    if cond_coeff > 1e15:
        import warnings
        warnings.warn(
            f'Badly conditioned Stein coefficient matrix (cond={cond_coeff:.2e}): '
            'W may be inaccurate.  Consider using smaller segment boundaries or '
            'a lower g value.',
            UserWarning, stacklevel=3,
        )

    # Solve the linear system instead of forming an explicit inverse.
    # _sp.solve uses LU factorisation; backward error ~ u * kappa, not u * kappa^2.
    rhs = P_lo - P_hi
    M = _sp.solve(coeff, rhs, check_finite=False)

    # Project back to (N, N): W = gamma^{-delta} * (I_N ⊗ C) M (C^T ⊗ I_N)
    kron_IC  = np.kron(np.eye(N), np.atleast_2d(C))         # (N, N^2)
    kron_CtI = np.kron(np.atleast_2d(C).T, np.eye(N))        # (N^2, N)
    return float(gamma ** (-delta)) * (kron_IC @ M @ kron_CtI)


   


def covariance_matrix_closed_form(A, C, gamma, a, b, delta):
    N = np.shape(A)[0]
    gATA = gamma * np.kron(np.transpose(A), A)

    if gamma > 1:
        gATA_a = matrix_power(gATA, a - 1) if ~(np.isinf(a)) else np.zeros_like(gATA, dtype=float)
        gATA_b = matrix_power(gATA, b) if ~(np.isinf(b)) else np.zeros_like(gATA, dtype=float)
        cond_coeff = cond(inv(gATA) - np.eye(N * N))
        if cond_coeff > 1e15:
            import warnings
            warnings.warn(
                f'Badly conditioned Steady State Matrix (cond={cond_coeff:.2e}): '
                'W may be inaccurate. Consider using smaller segment boundaries or '
                'a lower g value or use AlssmPolyLegendre',
                UserWarning, stacklevel=3,
            )

        return np.dot(gamma ** (-delta),
                      np.kron(np.eye(N), np.atleast_2d(C)) @
                      (inv(inv(gATA) - np.eye(N * N)) @ (gATA_a - gATA_b)) @
                      np.kron(np.atleast_2d(C).T, np.eye(N))
                      )
    else:
        gATA_a = matrix_power(gATA, a) if ~(np.isinf(a)) else np.zeros_like(gATA)
        gATA_b = matrix_power(gATA, b + 1) if ~(np.isinf(b)) else np.zeros_like(gATA)
        cond_coeff = cond(np.eye(N * N) - gATA)
        if cond_coeff > 1e15:
            import warnings
            warnings.warn(
                f'Badly conditioned Steady State Matrix (cond={cond_coeff:.2e}): '
                'W may be inaccurate. Consider using smaller segment boundaries or '
                'a lower g value or use AlssmPolyLegendre',
                UserWarning, stacklevel=3,
            )
        return np.dot(gamma ** (-delta),
                      np.kron(np.eye(N), np.atleast_2d(C)) @
                      (inv(np.eye(N * N) - gATA) @ (gATA_a - gATA_b)) @
                      np.kron(np.atleast_2d(C).T, np.eye(N))
                      )



def covariance_matrix_limited_sum(A, C, gamma, a, b, delta):
    raise NotImplementedError("limited_sum is not implemented yet.")
