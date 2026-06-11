import numpy as np
from numpy.linalg import matrix_power, cond, inv
from numpy.polynomial.legendre import legvander as _legvander
import scipy.linalg as _sp

from lmlib._warnings import WConditionNumberWarning
import warnings

# Maximum number of terms (b - a) for which the exact direct-sum Gram matrix
# (`covariance_matrix_limited_sum`) is preferred over the Stein-equation solve
# on finite segments.  The direct sum is O((b-a) N^3) and exact; wider finite
# windows fall back to the Stein solver.
_DIRECT_SUM_MAX_TERMS = 8192

__all__ = [
    'covariance_matrix_closed_form',
    'covariance_matrix_schur',
    'covariance_matrix_legendre',
    'covariance_matrix_meixner',
    'covariance_matrix_limited_sum',
]


def _is_legendre_shift(A, tol=1e-10):
    r"""
    Return ``(True, h)`` if *A* is a Legendre shift matrix, else ``(False, None)``.

    A Legendre shift matrix satisfies

    $$
    \phi(t + h) = \phi(t)\,A
    $$

    for some step size $h = 2/(W-1)$.  It is upper-triangular with ones on
    the diagonal, and every column *n* equals the Taylor expansion of
    $P_n(t + h)$ in the Legendre basis (see [`AlssmPolyLegendre`][lmlib.statespace.model.AlssmPolyLegendre]).

    The test verifies ALL columns of *A* against the exact Taylor expansion of
    $P_n(t+h)$ in the Legendre basis, computed via successive applications
    of [`legder`][numpy.polynomial.legendre.legder].  This is the only check that
    is both necessary and sufficient: earlier versions that only tested columns
    1 and 2 produced false positives for:

    * Jordan blocks (``h=1``, ``A[0,2]=0``): caught by the column-2 check.
    * Block-diagonal Legendre matrices (e.g. from [`CompositeCost`][lmlib.statespace.cost.CompositeCost] wrapping
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


def _is_meixner_shift(A, tol=1e-10):
    r"""
    Return ``(True, g)`` if *A* is a Meixner shift matrix, else ``(False, None)``.

    A Meixner shift matrix for effective window size $g$ satisfies

    $$
    A = I_N - \frac{1}{g-1}\,\triu(\mathbf{1}_N,\,1),
    $$

    i.e., it is upper-triangular with ones on the diagonal and a single constant
    value $-1/(g-1)$ on every super-diagonal entry.

    The test verifies the diagonal, the strict upper triangle, and the lower
    triangle.
    """
    N = A.shape[0]
    if N < 1:
        return False, None
    # Diagonal must be all ones
    if not np.allclose(np.diag(A), 1.0, atol=tol):
        return False, None
    # Lower triangle must be zero
    if not np.allclose(np.tril(A, -1), 0.0, atol=tol):
        return False, None
    if N == 1:
        # Degree-0: A = [[1]], g is arbitrary — return True with g=None sentinel
        return True, None
    # All strict upper-triangle entries must be equal and negative
    upper_vals = A[np.triu_indices(N, k=1)]
    if not np.allclose(upper_vals, upper_vals[0], atol=tol):
        return False, None
    val = upper_vals[0]
    if val >= 0.0:
        return False, None
    # Recover g: val = -1/(g-1)  =>  g = 1 - 1/val
    g = 1.0 - 1.0 / val
    if g <= 1.0:
        return False, None
    return True, g


def covariance_matrix_meixner(A, C, gamma, a, b, delta):
    r"""
    Exact steady-state Gram matrix for [`AlssmPolyMeixner`][lmlib.statespace.model.AlssmPolyMeixner].

    Exploits the closed-form orthogonality of the Meixner polynomials
    $M_n(j;\,1,\gamma)$ under the geometric weight $\gamma^j$:

    $$
    \sum_{j=a}^{b} \gamma^j\, M_m(j)\, M_n(j) = W_n\,\delta_{mn},
    \qquad W_n = \frac{1}{(1-\gamma)\,\gamma^n}
    $$

    for the **full semi-infinite** backward segment ($a=0, b=\infty$).

    For finite segments or forward infinite segments, the function falls back to a
    direct Vandermonde sum using [`hyp2f1`][scipy.special.hyp2f1] to evaluate the
    Meixner basis exactly.

    Parameters
    ----------
    A, C, gamma, a, b, delta
        Same as [`covariance_matrix_schur`][lmlib.statespace.backends.steady_state.covariance_matrix_schur].

    Returns
    -------
    W : ndarray of shape (N, N)
        Steady-state Gram matrix.
    """
    from scipy.special import hyp2f1

    A = np.asarray(A, dtype=float)
    is_m, g = _is_meixner_shift(A)
    if not is_m:
        raise ValueError('covariance_matrix_meixner requires a Meixner shift matrix A.')

    N = A.shape[0]

    # Degree-0 is trivially diagonal regardless of g
    if N == 1 or g is None:
        # W = C[0]^2 * sum_{j=a}^{b} gamma^j
        # The C[0]^2 factor is required for any ALSSM whose output scalar C[0]
        # differs from 1 (e.g. after a whitening/transformation step).
        c_sq = float(np.atleast_1d(np.asarray(C, dtype=float)).ravel()[0]) ** 2
        if np.isinf(a) and a < 0 and np.isinf(b) and b > 0:
            val = 1.0 / (1.0 - gamma) + 1.0 / (1.0 / gamma - 1.0) - 1.0
        elif np.isinf(b) and b > 0:
            a_int = max(int(a), 0)
            val = gamma ** a_int / (1.0 - gamma)
        elif np.isinf(a) and a < 0:
            b_int = int(b)
            val = gamma ** b_int / (1.0 / gamma - 1.0) if gamma > 1 else 0.0
            if gamma > 1:
                val = gamma ** (-b_int - 1) / (gamma - 1.0)
            else:
                # forward segment: weight = gamma^{-j}, j = 1..inf, but gamma<1 means
                # gamma^{-j} grows — use forward gamma = 1/gamma
                fg = 1.0 / gamma
                val = fg ** (-b_int) / (fg - 1.0) if fg > 1 else 0.0
        else:
            a_int, b_int = int(a), int(b)
            js = np.arange(a_int, b_int + 1, dtype=float)
            val = np.sum(gamma ** js)
        return np.array([[float(gamma ** (-delta)) * c_sq * val]])

    # ---------- Full semi-infinite backward segment (a=0, b=+inf or vice versa) ----------
    # This is the canonical case for which the closed-form diagonal is exact.
    if not np.isinf(a) and not np.isinf(b):
        # Finite segment: direct Vandermonde sum
        a_int, b_int = int(a), int(b)
        js = np.arange(a_int, b_int + 1, dtype=float)
        V = np.zeros((len(js), N))
        for n in range(N):
            c = np.zeros(N); c[n] = 1.0
            V[:, n] = np.array([float(hyp2f1(-n, -j, 1, 1.0 - 1.0/gamma)) for j in js])
        w = gamma ** js
        return float(gamma ** (-delta)) * (V.T @ (w[:, None] * V))

    # One or both bounds are infinite.
    # For the backward semi-infinite case starting at a_int:
    # W[n,n] = sum_{j=a_int}^{inf} gamma^j M_n(j)^2
    #        = gamma^{a_int} * W_n_canonical  (by the shift property)
    # The off-diagonal terms are zero by orthogonality.
    #
    # For forward infinite (a=-inf, b=b_int): mirror via gamma -> 1/gamma
    # and adjust sign of delta.

    if np.isinf(b) and b > 0:
        # Backward semi-infinite: [a_int, +inf)
        a_int = 0 if np.isinf(a) else int(a)
        ns = np.arange(N, dtype=float)
        # Shift the canonical formula: extra gamma^{a_int} factor from the sum start
        diag_vals = (gamma ** (a_int - delta)) / ((1.0 - gamma) * gamma ** ns)
        return np.diag(diag_vals)

    if np.isinf(a) and a < 0:
        # Forward semi-infinite: (-inf, b_int]
        # Weights are gamma^{-j} for j = b_int, b_int-1, ..., -inf  (gamma > 1 here)
        # OR equivalently: forward segment uses gamma_fw = g/(g-1) > 1, so
        # the window decays as gamma_fw^{j-delta} going forward into the past.
        # gamma passed in is the SEGMENT gamma, which for a forward segment is > 1.
        # Canonical closed-form with gamma > 1:
        # W[n,n] = sum_{j=-inf}^{b_int} gamma^j M_n(j; 1, gamma)^2
        # = gamma^{b_int} / (gamma-1) / gamma^n  (analogous formula, mirrored)
        b_int = int(b)
        ns = np.arange(N, dtype=float)
        diag_vals = (gamma ** (b_int - delta)) / ((gamma - 1.0) * gamma ** (-ns))
        # double-check sign: for gamma>1, sum_{j=-inf}^{b} gamma^j = gamma^b/(gamma-1)
        # and M_n orthog under gamma^j with gamma>1 has norm 1/((gamma-1)*gamma^{-n})
        # = gamma^n / (gamma-1)
        diag_vals = (gamma ** (b_int - delta)) * (gamma ** ns) / (gamma - 1.0)
        return np.diag(diag_vals)

    raise ValueError('Unexpected (a, b) combination in covariance_matrix_meixner.')


def covariance_matrix_legendre(A, C, gamma, a, b, delta):
    r"""
    Exact steady-state Gram matrix for [`AlssmPolyLegendre`][lmlib.statespace.model.AlssmPolyLegendre] via direct summation.

    Exploits the algebraic identity

    $$
    C\,A^j = \phi\!\left(-1 + j\,h\right)
             = \bigl[P_0(-1+jh),\;\ldots,\;P_D(-1+jh)\bigr]
    $$

    so that the design-matrix row at lag *j* is just one row of
    [`legvander`][numpy.polynomial.legendre.legvander] evaluated at $t_j = -1 + jh$.
    No matrix powers are computed; the result is therefore **exact to machine
    precision** regardless of degree:

    $$
    W = \gamma^{-\delta}
        \sum_{j=a}^{b} \gamma^j\,\phi(t_j)^{\top}\,\phi(t_j)
      = \gamma^{-\delta}\,V^{\top}\,\mathrm{diag}(\gamma^j)\,V
    $$

    where $V \in \mathbb{R}^{(b-a+1) \times N}$ is the Legendre Vandermonde
    matrix on $t_j \in [-1 + a\,h,\;\ldots,\;-1 + b\,h]$.

    Parameters
    ----------
    A, C, gamma, a, b, delta
        Same as [`covariance_matrix_schur`][lmlib.statespace.backends.steady_state.covariance_matrix_schur].
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
    r"""
    Numerically stable steady-state Gram matrix W via a Schur-based solver.

    Computes the same matrix as [`covariance_matrix_closed_form`][lmlib.statespace.backends.steady_state.covariance_matrix_closed_form]

    $$
    W = \gamma^{-\delta}
        \bigl(\mathbf{I}_N \otimes C\bigr)
        \,M\,
        \bigl(C^{\top} \otimes \mathbf{I}_N\bigr)
    $$

    where $M$ is the finite geometric series

    $$
    M = \sum_{t=a}^{b} \gamma^t \,
        (A^t)^{\!\top} \otimes A^t
      = \gamma^a (A^a)^{\!\top} \!\otimes A^a
        + \cdots
        + \gamma^b (A^b)^{\!\top} \!\otimes A^b
    $$

    which satisfies the Stein equation
    $M - \gamma\,(A^{\top} \!\otimes A)\,M\,=\,P_a - P_{b+1}$ (or
    the reverse inequality for $\gamma > 1$).

    **AlssmPolyLegendre fast path**

    When *A* is detected to be a Legendre shift matrix (upper-triangular, unit
    diagonal, valid step size *h*), this function automatically delegates to
    [`covariance_matrix_legendre`][lmlib.statespace.backends.steady_state.covariance_matrix_legendre] instead of solving the Stein equation.

    That path computes $W = V^{\top}\,\mathrm{diag}(\gamma^j)\,V$ using
    [`legvander`][numpy.polynomial.legendre.legvander], which is exact to machine
    precision for **any** polynomial degree and avoids the ill-conditioned
    $N^2 \times N^2$ Kronecker system entirely.

    **Why this is more stable than** [`covariance_matrix_closed_form`][lmlib.statespace.backends.steady_state.covariance_matrix_closed_form]

    The closed-form function inverts the matrix
    $I_{N^2} - \gamma\,(A^{\top} \!\otimes A)$ (or its inverse for
    $\gamma > 1$) via [`inv`][numpy.linalg.inv].  Forming the explicit
    inverse of a Sylvester/Stein coefficient matrix amplifies rounding errors
    by $\kappa^2$, where $\kappa$ is its condition number.

    This function instead solves the Stein equation

    $$
    X - \gamma\,(A^{\top} \!\otimes A)\,X = P_a - P_{b+1}
    $$

    directly with [`solve`][scipy.linalg.solve], exploiting the fact that the
    coefficient matrix is available explicitly.  The linear solve (LU
    factorisation) needs only $O(N^6)$ work (same as the explicit
    inverse) but achieves a backward error of order $u \kappa$ rather
    than $u \kappa^2$.

    For further robustness the right-hand side matrices $P_a$ and
    $P_{b+1}$ are formed via [`matrix_power`][numpy.linalg.matrix_power] (which
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
        [`covariance_matrix_closed_form`][lmlib.statespace.backends.steady_state.covariance_matrix_closed_form].
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

    # ── Fast path: AlssmPolyMeixner ──────────────────────────────────────────────
    # The Meixner basis is orthogonal under the geometric weight gamma^j, so the
    # Gram matrix is exactly diagonal with closed-form entries. Avoids the N^2×N^2
    # Stein equation entirely, giving exact results for any degree and segment type.
    is_m, _ = _is_meixner_shift(A)
    if is_m:
        return covariance_matrix_meixner(A, C, gamma, a, b, delta)


    # ── Exact direct-sum path for finite segments ───────────────────────────
    # On a finite window W is a finite sum; evaluating it directly is exact to
    # machine precision and avoids the N^2 x N^2 Stein solve below, which loses
    # several digits for short, high-degree polynomial segments (e.g.
    # AlssmPolyJordan, which is caught by neither the Legendre nor the Meixner
    # fast path).  The O((b-a) N^3) loop is capped so very wide finite windows
    # still fall through to the Stein solver.
    if not np.isinf(a) and not np.isinf(b) and (b - a) <= _DIRECT_SUM_MAX_TERMS:
        return covariance_matrix_limited_sum(A, C, gamma, a, b, delta)

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
        # warnings.warn(
        #     f'Badly conditioned Stein coefficient matrix (cond={cond_coeff:.2e}): '
        #     'W may be inaccurate.  Consider using smaller segment boundaries or '
        #     'a lower g value.',
        #     WConditionNumberWarning, stacklevel=3,
        # )
        warnings.warn(
            'Badly conditioned Stein coefficient matrix: W may be inaccurate. '
            'Consider using smaller segment boundaries or a lower g value.',
            WConditionNumberWarning,
            stacklevel=3,
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
    r"""
    Compute the steady-state Gram matrix W in closed form via a Stein equation.

    Uses the analytic matrix inversion approach.  May issue a
    [`WConditionNumberWarning`][lmlib._warnings.WConditionNumberWarning] when the Stein coefficient matrix is
    badly conditioned (condition number > 1e15).

    Parameters
    ----------
    A : ndarray of shape (N, N)
        State transition matrix.
    C : ndarray of shape ([Q,] N)
        Output matrix.
    gamma : float
        Window decay factor.
    a, b : int or np.inf
        Segment boundaries.
    delta : int
        Window normalisation index.

    Returns
    -------
    W : ndarray of shape (N, N)
        Steady-state Gram matrix.
    """
    N = np.shape(A)[0]
    gATA = gamma * np.kron(np.transpose(A), A)

    if gamma > 1:
        gATA_a = matrix_power(gATA, a - 1) if ~(np.isinf(a)) else np.zeros_like(gATA, dtype=float)
        gATA_b = matrix_power(gATA, b) if ~(np.isinf(b)) else np.zeros_like(gATA, dtype=float)
        cond_coeff = cond(inv(gATA) - np.eye(N * N))
        if cond_coeff > 1e15:
            warnings.warn(
                'Badly conditioned Stein coefficient matrix: W may be inaccurate. '
                'Consider using smaller segment boundaries or a lower g value.',
                WConditionNumberWarning,
                stacklevel=3,
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
            warnings.warn(
                'Badly conditioned Stein coefficient matrix: W may be inaccurate. '
                'Consider using smaller segment boundaries or a lower g value.',
                WConditionNumberWarning,
                stacklevel=3,
                )

        return np.dot(gamma ** (-delta),
                      np.kron(np.eye(N), np.atleast_2d(C)) @
                      (inv(np.eye(N * N) - gATA) @ (gATA_a - gATA_b)) @
                      np.kron(np.atleast_2d(C).T, np.eye(N))
                      )



def covariance_matrix_limited_sum(A, C, gamma, a, b, delta):
    r"""
    Exact steady-state Gram matrix W for a finite segment via direct summation.

    Computes the windowed Gram matrix directly from its definition

    $$
    W = \sum_{t=a}^{b} \gamma^{\,t-\delta}\,(A^t)^{\top} C^{\top} C\, A^t .
    $$

    Because this is exactly the finite sum that defines ``W`` on a finite
    segment ``[a, b]``, the result is accurate to machine precision.  Unlike the
    Stein-equation path it never forms or solves the $N^2 \times N^2$ Kronecker
    system, so it stays accurate even when that system is ill-conditioned --
    e.g. short, high-degree :class:`AlssmPolyJordan` segments, where the Stein
    solve can lose several digits.  Cost is $O\big((b-a)\,N^3\big)$.

    Parameters
    ----------
    A, C, gamma, a, b, delta
        See [`covariance_matrix_closed_form`][lmlib.statespace.backends.steady_state.covariance_matrix_closed_form].
        ``a`` and ``b`` must be finite.

    Returns
    -------
    W : ndarray of shape (N, N)
    """
    if np.isinf(a) or np.isinf(b):
        raise ValueError("covariance_matrix_limited_sum requires finite a and b.")
    A = np.asarray(A, dtype=float)
    C = np.atleast_2d(np.asarray(C, dtype=float))            # (1, N)
    N = A.shape[0]
    Q = C.T @ C                                              # (N, N)
    a, b = int(a), int(b)
    At = matrix_power(A, a)                                  # A^a
    W = np.zeros((N, N))
    for t in range(a, b + 1):
        W += (gamma ** (t - delta)) * (At.T @ Q @ At)
        At = At @ A
    return W
