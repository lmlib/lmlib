"""
lmlib.statespace.sos_matlab
===========================
Tools for importing SOS (second-order section) filter coefficients that were
computed in MATLAB instead of scipy.

Why bother?
-----------
scipy's ``ss2tf`` / ``zpk2sos`` pipeline can be numerically inaccurate for
large state-space matrices (ill-conditioned polynomial expansion).  MATLAB's
``ss2zp`` + ``zp2sos`` pipeline is significantly more accurate.  This module
lets users:

1. **Generate** a self-contained MATLAB script from their cost/segment
   parameters (no .mat files needed).
2. **Run** the script in MATLAB and copy-paste the printed Python dict.
3. **Import** the dict back into lmlib with ``sos_from_matlab()``, which
   validates a hash key and converts to the internal ``_numdenom`` format.

Typical usage
-------------
::

    # In your Python script — print the MATLAB code:
    import lmlib as lm
    alssm = lm.AlssmPolyJordan(poly_degree=3)
    seg   = lm.Segment(a=-12, b=-1, direction=lm.FW, g=100)
    cost  = lm.CompositeCost([alssm], [seg], F=[[1]])
    lm.print_matlab_sos_script(cost)

    # Run the printed script in MATLAB, copy the output, paste here:
    numdenom_matlab = {
        'key': 'ad4c4c5a',
        'sos_iir': [...],
        'sos_a':   [...],
        'sos_b':   [...],
    }

    # Build the RLS filter using the MATLAB coefficients:
    nd = lm.sos_from_matlab(cost, numdenom_matlab)
    rls = lm.RLSAlssm(cost, backend='lfilter', filter_form='parallel',
                      numdenom=nd, supress_pzinstruction=True)
    rls.filter(y)
"""

import hashlib
import json

import numpy as np
from numpy.linalg import inv, matrix_power, eigvals
from scipy.signal import zpk2sos

from lmlib.statespace.backends.rec_lfilter import _make_num_sos


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = ['print_matlab_sos_script', 'sos_from_matlab', 'sos_from_matlab_multi']


def print_matlab_sos_script(cost, dim_index=0):
    """Print a self-contained MATLAB script that computes SOS coefficients.

    Run the printed script in MATLAB.  The CLI output is a Python dict that
    can be passed directly to :func:`sos_from_matlab`.

    Parameters
    ----------
    cost : CompositeCost or CostSegment
        The cost whose ALSSM/segment parameters drive the script.
    dim_index : int, optional
        Which dimension index this cost belongs to (for labelling only).
    """
    print(_build_matlab_script(cost, dim_index))


def sos_from_matlab(cost, numdenom_matlab):
    """Convert a MATLAB-generated SOS dict into the ``_numdenom`` format.

    Parameters
    ----------
    cost : CompositeCost or CostSegment
        Must be the *same* cost used to generate the MATLAB script.
    numdenom_matlab : dict
        The dict printed by the MATLAB script.  Must contain
        ``'key'``, ``'sos_iir'``, ``'num_a'``, ``'num_b'``.

    Returns
    -------
    numdenom : list
        A ``_numdenom`` structure suitable for ``RLSAlssm(..., numdenom=...)``.

    Raises
    ------
    ValueError
        If the key in ``numdenom_matlab`` does not match the cost parameters,
        indicating that the coefficients are stale.
    """
    from lmlib.statespace.cost import CostSegment
    if isinstance(cost, CostSegment):
        from lmlib.statespace.cost import CompositeCost
        cost = CompositeCost(
            [cost.alssm], [cost.segment],
            np.ones((1, 1), dtype=int),
            betas=np.array([cost.beta]),
        )

    # _numdenom shape: [dim][seg][alssm] — for a single cost there is 1 dim
    numdenom = [[[None] * cost.M for _ in range(cost.P)]]

    for p, segment in enumerate(cost.segments):
        for m, alssm_m in enumerate(cost.alssms):
            if cost.F[m, p] == 0.0:
                continue   # inactive node — stays None

            A         = np.atleast_2d(alssm_m.A)
            C         = np.atleast_1d(alssm_m.C).ravel()
            direction = segment.direction
            a         = segment.a
            b         = segment.b
            gamma     = segment.gamma
            f_mp      = float(cost.F[m, p])

            # ── validate key ─────────────────────────────────────────────────
            expected_key = _sos_key(A, C, a, b, gamma, direction)
            provided_key = numdenom_matlab.get('key', '')
            if provided_key != expected_key:
                raise ValueError(
                    f"SOS key mismatch for segment {p}, alssm {m}.\n"
                    f"  Expected : '{expected_key}'\n"
                    f"  Got      : '{provided_key}'\n"
                    "The MATLAB coefficients were generated for different "
                    "system parameters.  Re-run print_matlab_sos_script() "
                    "and regenerate the coefficients in MATLAB."
                )

            # ── parse ─────────────────────────────────────────────────────────
            sos_iir    = np.array(numdenom_matlab['sos_iir'])
            # num_a/num_b: N polynomial rows from MATLAB ss2tf — ss2tf format:
            # each row has poly[0]=0 (artefact), then the numerator coefficients.
            # Using polynomials preserves the relative degree (integer delay),
            # which zp2sos would lose for rows with no finite zeros.
            num_rows_a = [np.array(r) for r in numdenom_matlab['num_a']]
            num_rows_b = [np.array(r) for r in numdenom_matlab['num_b']]

            sos_b_list, db_list = [], []
            sos_a_list, da_list = [], []
            N = A.shape[0]
            for n_ in range(N):
                sb, db = _make_num_sos(num_rows_b[n_] * f_mp)
                sa, da = _make_num_sos(num_rows_a[n_] * f_mp)
                sos_b_list.append(sb); db_list.append(db)
                sos_a_list.append(sa); da_list.append(da)

            numdenom[0][p][m] = [sos_iir, sos_b_list, sos_a_list,
                                  db_list, da_list]

    return numdenom


# ─────────────────────────────────────────────────────────────────────────────
# Key / hash
# ─────────────────────────────────────────────────────────────────────────────

def _sos_key(A, C, a, b, gamma, direction):
    """Stable 8-character hex key from parameters that determine SOS coefficients.

    C is normalised to a 1-D list so that a row vector ``(1, N)`` and a flat
    vector ``(N,)`` produce the same key.
    """
    payload = json.dumps({
        'A':     np.array(A).tolist(),
        'C':     np.array(C).ravel().tolist(),   # always 1-D → stable across shapes
        'a':     int(a),
        'b':     int(b),
        'gamma': float(gamma),
        'dir':   str(direction),
    }, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(payload.encode()).hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────────────
# MATLAB script generator
# ─────────────────────────────────────────────────────────────────────────────

def _build_matlab_script(cost, dim_index=0):
    """Return the MATLAB script string for one cost."""
    from lmlib.statespace.cost import CostSegment
    if isinstance(cost, CostSegment):
        from lmlib.statespace.cost import CompositeCost
        cost = CompositeCost(
            [cost.alssm], [cost.segment],
            np.ones((1, 1), dtype=int),
            betas=np.array([cost.beta]),
        )

    blocks = []
    blocks.append(_matlab_header())

    for p, segment in enumerate(cost.segments):
        for m, alssm_m in enumerate(cost.alssms):
            if cost.F[m, p] == 0.0:
                continue

            A         = alssm_m.A
            C         = alssm_m.C
            direction = segment.direction
            a         = segment.a
            b         = segment.b
            gamma     = segment.gamma
            key       = _sos_key(A, C, a, b, gamma, direction)

            blocks.append(
                _matlab_block(A, C, a, b, gamma, direction,
                               key, dim_index, p, m)
            )

    return '\n'.join(blocks)


def _matlab_header():
    return (
        "% ═══════════════════════════════════════════════════════════════════\n"
        "% lmlib SOS coefficient generator\n"
        "% Run this script in MATLAB, then copy EACH printed block into your\n"
        "% Python source file as shown and pass it to lm.sos_from_matlab().\n"
        "% ═══════════════════════════════════════════════════════════════════\n"
        "format long g\n"
    )


def _matlab_block(A, C, a, b, gamma, direction, key, dim_idx, seg_idx, alssm_idx):
    """Return the MATLAB code for one (segment, alssm) combination."""
    N = A.shape[0]

    if direction == 'fw':
        gAT_expr   = "inv(A * gamma)'"
        boundary_a = a - 1
        boundary_b = b
    else:  # bw
        gAT_expr   = "(A * gamma)'"
        boundary_a = a
        boundary_b = b + 1

    label = f"dim{dim_idx}_seg{seg_idx}_alssm{alssm_idx}"

    # Build the script as a list of plain strings (no f-strings with nested quotes)
    lbl     = label
    k       = key
    ga_expr = gAT_expr
    ba      = boundary_a
    bb      = boundary_b
    ga      = gamma

    script_lines = [
        "",
        f"% -- segment {seg_idx}, alssm {alssm_idx}  (direction={direction}, a={a}, b={b}, gamma~{ga:.6g}) --",
        _mat_assign(A, "A"),
        _mat_assign(C, "C"),
        f"gamma = {ga:.17g};",
        "",
        f"N          = size(A, 1);",
        f"gAT        = {ga_expr};",
        f"boundary_a = {ba};",
        f"boundary_b = {bb};",
        "",
        "% IIR denominator SOS (poles only, shared across all rows)",
        "poles        = eig(gAT);",
        "[sos_iir, ~] = zp2sos(zeros(N, 1), poles, 1);",
        "",
        f"fprintf('numdenom_matlab_{lbl} = {{\\n');",
        f"fprintf('    ''key'':     ''{k}'',\\n');",
        "",
        "% sos_iir",
        "fprintf('    ''sos_iir'': [');",
        "for s = 1 : size(sos_iir, 1)",
        "    if s > 1; fprintf(', '); end",
        "    fprintf('[%.17g, %.17g, %.17g, %.17g, %.17g, %.17g]', ...",
        "        sos_iir(s,1), sos_iir(s,2), sos_iir(s,3), ...",
        "        sos_iir(s,4), sos_iir(s,5), sos_iir(s,6));",
        "end",
        "fprintf('],\\n');",
        "",
        "% Numerator polynomials via ss2tf -- preserves relative degree (delay)",
        "for bnd_idx = 1:2",
        "    if bnd_idx == 1; boundary = boundary_a; bnd_name = 'a';",
        "    else;            boundary = boundary_b; bnd_name = 'b'; end",
        "    B_bnd = (A ^ boundary)' * C';",
        "    [num_bnd, ~] = ss2tf(gAT, B_bnd, eye(N), zeros(N, 1));",
        "    fprintf('    ''num_%s'': [\\n', bnd_name);",
        "    for n_ = 1 : N",
        "        fprintf('        [');",
        "        for c = 1 : size(num_bnd, 2)",
        "            if c > 1; fprintf(', '); end",
        "            fprintf('%.17g', num_bnd(n_, c));",
        "        end",
        "        if n_ < N; fprintf('],\\n');",
        "        else;      fprintf(']\\n');  end",
        "    end",
        "    if bnd_idx < 2; fprintf('    ],\\n');",
        "    else;           fprintf('    ]\\n');  end",
        "end",
        "fprintf('};\\n');",
        "",
        f"% -- Python usage (paste the printed block above, then): --",
        f"% nd = lm.sos_from_matlab(cost, numdenom_matlab_{lbl})",
        "% rls = lm.RLSAlssm(cost, backend='lfilter', filter_form='parallel',",
        "%                   numdenom=nd, supress_pzinstruction=True)",
    ]
    return '\n'.join(script_lines)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mat_assign(M, name):
    """Format a numpy array as a MATLAB matrix assignment."""
    M = np.atleast_2d(np.array(M, dtype=float))
    rows = '; '.join(' '.join(f'{v:.17g}' for v in row) for row in M)
    return f"{name} = [{rows}];"


def _sos_combined_to_num_poly(sos_combined):
    """Reconstruct numerator polynomial from the b-coefficients of a combined SOS.

    MATLAB's ``zp2sos`` returns sections where each row encodes both numerator
    and denominator: ``[b0 b1 b2 a0 a1 a2]``.  We multiply out the b-parts to
    recover the full numerator polynomial, then prepend a zero to match the
    ``ss2tf`` output convention that ``_make_num_sos`` expects.

    ``zp2sos`` always pads each section to degree 2 even when fewer zeros are
    present, so trailing near-zero coefficients are stripped before returning.
    """
    poly = np.array([1.0])
    for section in np.atleast_2d(sos_combined):
        poly = np.convolve(poly, section[:3])
    # Strip trailing near-zero padding inserted by zp2sos
    nz_end = len(poly) - np.argmax(np.abs(poly[::-1]) > 1e-12)
    poly = poly[:nz_end]
    # Prepend a zero: _make_num_sos expects the ss2tf output format where
    # poly[0] is always 0 (the z^{-1} normalisation artefact).
    return np.concatenate([[0.0], poly])

def sos_from_matlab_multi(cost, numdenom_list):
    """Convert a list of MATLAB-generated SOS dicts (one per ALSSM) into ``_numdenom``.

    Use this when a ``CompositeCost`` has more than one ALSSM and you ran the
    MATLAB script separately for each ALSSM, producing one dict per ALSSM.

    Parameters
    ----------
    cost : CompositeCost or CostSegment
        Must be the *same* cost used to generate the MATLAB scripts.
    numdenom_list : list of dict
        One dict per ALSSM, **in the same order as** ``cost.alssms``.
        Each dict must contain ``'key'``, ``'sos_iir'``, ``'num_a'``, ``'num_b'``.

    Returns
    -------
    numdenom : list
        A ``_numdenom`` structure suitable for ``RLSAlssm(..., numdenom=...)``.

    Examples
    --------
    ::

        nd = lm.sos_from_matlab_multi(cost, [
            numdenom_matlab_dim0_seg0_alssm0,   # alssm_sp
            numdenom_matlab_dim0_seg0_alssm1,   # alssm_bl
        ])
        rls = lm.RLSAlssm(cost, backend='lfilter', filter_form='parallel',
                          numdenom=nd, supress_pzinstruction=True)
    """
    from lmlib.statespace.cost import CostSegment
    if isinstance(cost, CostSegment):
        from lmlib.statespace.cost import CompositeCost
        cost = CompositeCost(
            [cost.alssm], [cost.segment],
            np.ones((1, 1), dtype=int),
            betas=np.array([cost.beta]),
        )

    if len(numdenom_list) != cost.M:
        raise ValueError(
            f"numdenom_list has {len(numdenom_list)} entries but cost has "
            f"{cost.M} ALSSMs.  Provide one dict per ALSSM in cost.alssms order."
        )

    # _numdenom shape: [dim][seg][alssm]  — single dimension for a plain CompositeCost
    numdenom = [[[None] * cost.M for _ in range(cost.P)]]

    for p, segment in enumerate(cost.segments):
        for m, alssm_m in enumerate(cost.alssms):
            if cost.F[m, p] == 0.0:
                continue   # inactive node — stays None

            A         = np.atleast_2d(alssm_m.A)
            C         = np.atleast_1d(alssm_m.C).ravel()
            direction = segment.direction
            a         = segment.a
            b         = segment.b
            gamma     = segment.gamma
            f_mp      = float(cost.F[m, p])

            d = numdenom_list[m]

            # ── validate key ─────────────────────────────────────────────────
            expected_key = _sos_key(A, C, a, b, gamma, direction)
            provided_key = d.get('key', '')
            if provided_key != expected_key:
                raise ValueError(
                    f"SOS key mismatch for segment {p}, alssm {m}.\n"
                    f"  Expected : '{expected_key}'\n"
                    f"  Got      : '{provided_key}'\n"
                    "The MATLAB coefficients were generated for different system "
                    "parameters.  Re-run print_matlab_sos_script() and regenerate."
                )

            # ── parse ─────────────────────────────────────────────────────────
            sos_iir    = np.array(d['sos_iir'])
            num_rows_a = [np.array(r) for r in d['num_a']]
            num_rows_b = [np.array(r) for r in d['num_b']]

            sos_b_list, db_list = [], []
            sos_a_list, da_list = [], []
            N = A.shape[0]
            for n_ in range(N):
                sb, db = _make_num_sos(num_rows_b[n_] * f_mp)
                sa, da = _make_num_sos(num_rows_a[n_] * f_mp)
                sos_b_list.append(sb); db_list.append(db)
                sos_a_list.append(sa); da_list.append(da)

            numdenom[0][p][m] = [sos_iir, sos_b_list, sos_a_list, db_list, da_list]

    return numdenom

