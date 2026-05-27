"""
Polynomial ALSSM Benchmark v2
==============================
Config 1: finite BW [0,200], g=1000  – AlssmPoly | AlssmPolyJordan | AlssmPolyLegendre | scipy ref
Config 2: infinite BW [0,∞], g=100  – AlssmPoly | AlssmPolyJordan | AlssmPolyMeixner | direct-sum ref

Panel A – Filter output error  max_k |C x̂[k] − y_true[k]|
Panel B – cond(W)            (steady_state_method='schur': exact for Legendre/Meixner)
Panel C – ξ recursion drift  max_k ‖ξ_rec−ξ_ref‖/‖ξ_ref‖
Panel D – Max usable degree summary
"""
import sys, warnings, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.polynomial.legendre import legvander

sys.path.insert(0, '/home/claude')
import lmlib as lm

# ─── parameters ──────────────────────────────────────────────────────────────
K1, K2 = 1000, 10000  
DEGREES = list(range(1, 26))
TRIALS  = 8
NOISE   = 1e-8
K_EVAL  = np.arange(50, 350)    # used for config 1
K_EVAL2 = np.arange(1000, 1100)  # config 2: well into steady state, ample future samples
THR     = 1e-4

A1, B1, G1 = 0, 200, 1000;  W1 = 201;  GAMMA1 = (G1-1)/G1
A2, B2, G2 = 0, np.inf, 100;            GAMMA2 = (G2-1)/G2

CFGS1 = [('AlssmPoly',           'poly',       '#e15759', '-',  'o'),
         ('AlssmPolyJordan',      'jordan',     '#f28e2b', '--', 's'),
         ('AlssmPolyLegendre',    'legendre',   '#4e79a7', '-',  'P'),
         ('scipy Legendre (ref)', 'scipy',      '#59a14f', '-',  'D')]
CFGS2 = [('AlssmPoly',            'poly',       '#e15759', '-',  'o'),
         ('AlssmPolyJordan',      'jordan',     '#f28e2b', '--', 's'),
         ('AlssmPolyMeixner',     'meixner',    '#76b7b2', '-',  '^'),
         ('Meixner ref (direct)', 'meixner_ref','#59a14f', '-',  'D')]


def max_ok(vals):
    lo = 0
    for i, d in enumerate(DEGREES):
        v = vals[i]
        if not np.isnan(v) and v < THR: lo = d
        else: break
    return lo


def make_cost(method, deg, cfg_idx):
    a, b, g = (A1,B1,G1) if cfg_idx==1 else (A2,B2,G2)
    seg = lm.Segment(a, b, lm.BW, g=g)
    if   method == 'poly':     al = lm.AlssmPoly(poly_degree=deg)
    elif method == 'jordan':   al = lm.AlssmPolyJordan(poly_degree=deg)
    elif method == 'legendre': al = lm.AlssmPolyLegendre(poly_degree=deg, a_seg=a, b_seg=b)
    elif method == 'meixner':  al = lm.AlssmPolyMeixner(poly_degree=deg, segment=seg)
    else: return None, None
    return lm.CostSegment(al, seg), al

def run_rls(cost, y):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rls = lm.RLSAlssm(cost, steady_state=True, steady_state_method='schur')
        rls.filter(y)
    return rls

def build_phi(A, C, horizon):
    """(horizon, N) matrix of C @ A^j rows — equivalent to one row per lag of Trajectory.eval_y."""
    N = A.shape[0]
    phi = np.empty((horizon, N))
    Aj = np.eye(N)
    for j in range(horizon):
        phi[j] = C @ Aj
        Aj = Aj @ A
    return phi

def xi_ref_vectorized(A, C, a, b, gamma, y, K_EVAL):
    """
    Vectorised direct-sum ξ_ref[k] = Σ_{j=a}^{min(b,K-k-1)} γ^j A^{j,T} C^T y[k+j]
    for all k in K_EVAL simultaneously.
    """
    K = len(y); N = A.shape[0]
    j_min = int(max(a, 0))
    # For finite b: use b; for infinite: use K-1-min(K_EVAL) so all k have same horizon
    if np.isinf(b):
        j_max = K - 1 - int(K_EVAL.min())   # all k in K_EVAL have at least this many future samples
    else:
        j_max = int(b)
    j_max = min(j_max, K - 1)

    # Build ATC[j] = A^{j,T} @ C  for j = j_min..j_max
    ATC = np.empty((j_max - j_min + 1, N))
    Aj = np.linalg.matrix_power(A, j_min)
    for i in range(len(ATC)):
        ATC[i] = Aj.T @ C
        Aj = Aj @ A

    gj = gamma ** np.arange(j_min, j_max + 1)   # (J,)

    # Y[i, ki] = y[K_EVAL[ki] + j_min + i]  for i=0..J-1, ki=0..len(K_EVAL)-1
    J = j_max - j_min + 1
    idx = K_EVAL[np.newaxis, :] + np.arange(j_min, j_max+1)[:, np.newaxis]  # (J, len)
    valid = (idx >= 0) & (idx < K)
    Y = np.where(valid, y[np.clip(idx, 0, K-1)], 0.0)   # (J, len)

    # xi_ref = (ATC.T @ (gj[:,None] * Y)).T    shape: (len, N)
    return (ATC.T @ (gj[:, None] * Y)).T




execute_section_B=True
execute_section_C=True
execute_section_D=True

#execute_section_B=False
#execute_section_C=False
#execute_section_D=False

# ─── Section B: cond(W) ───────────────────────────────────────────────────────
if execute_section_B:
    print("\n"+"="*70); print("SECTION B – cond(W)"); print("="*70)
    cond_W = {1: {c[0]: [] for c in CFGS1}, 2: {c[0]: [] for c in CFGS2}}
    np.random.seed(7)
    probes = {1: np.random.randn(K1), 2: np.random.randn(K2)}
    
    for cfg_idx, CFGS, g, gamma, W_len in [
        (1, CFGS1, G1, GAMMA1, W1), (2, CFGS2, G2, GAMMA2, None)]:
        for deg in DEGREES:
            for lbl, method, *_ in CFGS:
                try:
                    if method == 'scipy':
                        Vs = legvander(2*np.arange(W_len,dtype=float)/(W_len-1)-1, deg)
                        cond_W[cfg_idx][lbl].append(
                            np.linalg.cond(Vs.T @ np.diag(gamma**np.arange(W_len)) @ Vs))
                    elif method == 'meixner_ref':
                        ns = np.arange(deg+1, dtype=float)
                        cond_W[cfg_idx][lbl].append(
                            np.linalg.cond(np.diag(1/((1-gamma)*gamma**ns))))
                    else:
                        cost, _ = make_cost(method, deg, cfg_idx)
                        rls = run_rls(cost, probes[cfg_idx])
                        cond_W[cfg_idx][lbl].append(np.linalg.cond(rls.W))
                except Exception:
                    cond_W[cfg_idx][lbl].append(np.nan)

# ─── Section C: ξ drift ──────────────────────────────────────────────────────
if execute_section_C:
    print("\n"+"="*70); print("SECTION C – ξ drift"); print("="*70)
    xi_drift   = {1: {c[0]: [] for c in CFGS1}, 2: {c[0]: [] for c in CFGS2}}
    xi_drift_k = {1: {c[0]: [] for c in CFGS1}, 2: {c[0]: [] for c in CFGS2}}
    np.random.seed(13)
    y_xi = {1: np.random.randn(K1), 2: np.random.randn(K2)}
    
    for cfg_idx, CFGS, K, a_seg, b_seg, g, gamma in [
        (1, CFGS1, K1, A1, B1, G1, GAMMA1),
        (2, CFGS2, K2, A2, B2, G2, GAMMA2),
    ]:
        print(f"\n--- Config {cfg_idx}  K={K} ---")
        print(f'{"deg":>4}' + ''.join(f' {c[0][:18]:>20}' for c in CFGS))
        for deg in DEGREES:
            t0 = time.time(); row = f'{deg:4d}'
            for lbl, method, *_ in CFGS:
                if method in ('scipy', 'meixner_ref'):
                    xi_drift[cfg_idx][lbl].append(np.nan)
                    xi_drift_k[cfg_idx][lbl].append(np.nan)
                    row += f' {"N/A":>20}'; continue
                try:
                    cost, al = make_cost(method, deg, cfg_idx)
                    A = al.A
                    C = al.C.flatten() if al.C.ndim > 1 else al.C
                    rls = run_rls(cost, y_xi[cfg_idx])
                    xi_rec = rls._xi1   # (K, N)
                    ke = K_EVAL if cfg_idx == 1 else K_EVAL2
                    xi_ref = xi_ref_vectorized(A, C, a_seg, b_seg, gamma, y_xi[cfg_idx], ke)
                    nref = np.linalg.norm(xi_ref, axis=1)
                    nref = np.where(nref < 1e-300, 1e-300, nref)
                    rel  = np.linalg.norm(xi_rec[ke] - xi_ref, axis=1) / nref
                    k_mx = int(ke[np.argmax(rel)])
                    mx   = float(np.max(rel))
                    xi_drift[cfg_idx][lbl].append(mx)
                    xi_drift_k[cfg_idx][lbl].append(k_mx)
                    row += f' {mx:16.2e}@k={k_mx:<3}{"!" if mx > THR else " "}'
                except Exception as e:
                    xi_drift[cfg_idx][lbl].append(np.nan)
                    xi_drift_k[cfg_idx][lbl].append(np.nan)
                    row += f' {"ERR":>20}'
            print(row + f'  ({time.time()-t0:.1f}s)')



# ─── Panel D precomputation: distance-based SNR metric on ECG template matching ──
#
# Distance metric:
#   distance[k] = (xs[k,1:] - xs_ref[1:]) @ W[1:,1:] @ (xs[k,1:] - xs_ref[1:])
#   → 0 when the local shape matches the reference template exactly.
#   → Quadratic (always ≥ 0 by construction; negatives = numerical breakdown, skipped).
#
# SNR ratio:
#   numerator   = min(distance) over non-spike, non-guard samples (values < 0 excluded)
#                 i.e. the noise sample most similar to the template
#   denominator = max of per-spike local minima (worst/hardest spike to detect)
#   ratio > 1 → detectable; ratio = 1 → boundary; ratio < 1 → false positives
#
# Panel layout:
#   col 0  → D1:  finite two-sided  [a_seg_fin,−1] ∪ [0,b_seg_fin],  g=250
#   col 1  → D2a: one-sided infinite          [0,∞) BACKWARD,        g=30,  K_REF=1825
#          → D2b: two-sided infinite  (−∞,−1] ∪ [0,∞),              g=20,  K_REF=1865

def _compute_snr_ratio(dist, true_spikes, spike_mask, K, spk_win):
    """Shared helper: compute SNR ratio from a distance array."""
    _spike_local_mins = []
    for _p in true_spikes:
        _lo = max(0, _p - spk_win)
        _hi = min(K, _p + spk_win)
        _local = dist[_lo:_hi]
        _valid = _local[_local >= 0]
        if len(_valid) > 0:
            _spike_local_mins.append(float(_valid.min()))
    if len(_spike_local_mins) == 0:
        return np.nan
    _guard_mask = np.zeros(K, dtype=bool)
    _guard_mask[:500]  = True
    _guard_mask[-500:] = True
    _noise = dist[~spike_mask & ~_guard_mask]
    _noise = _noise[_noise >= 0]
    if len(_noise) == 0:
        return np.nan
    _ratio = float(_noise.min()) / max(float(np.max(_spike_local_mins)), 1e-300)
    if not np.isfinite(_ratio) or _ratio > 1e10:
        return np.nan
    return _ratio

if execute_section_D:
    print("\n"+"="*70); print("PANEL D – distance-based SNR metric on ECG template matching"); print("="*70)
    import warnings as _w

    # panel_d_data2 keys: 0=D1 (finite), 1=D2a (one-sided inf), 2=D2b (two-sided inf)
    panel_d_data2  = {}
    _panel_d_ok = False
    try:
        _degs_d = list(range(1, 26))

        _file = 'EECG_BASELINE_1CH_10S_FS2400HZ.csv'
        _K = 10000
        _y = lm.utils.generator.load_lib_csv(_file, _K)
        _SPK_WIN = 200
        _TRUE_SPIKES = np.array([1865, 4209, 6553, 8889])
        _spike_mask = np.zeros(_K, dtype=bool)
        for _p in _TRUE_SPIKES:
            _spike_mask[max(0,_p-_SPK_WIN):min(_K,_p+_SPK_WIN)] = True

        # ── shared parameters ──
        _K_REF_FIN  = 1865;  _g_sp_fin = 250   # D1:  finite two-sided
        _K_REF_1S   = 1825;  _g_sp_1s  = 30    # D2a: one-sided infinite
        _K_REF_2S   = 1865;  _g_sp_2s  = 20    # D2b: two-sided infinite

        a_seg_fin = -51
        b_seg_fin = 50
        _segment_d1 = lm.Segment(a=0,b=np.inf,direction=lm.BACKWARD,g=_g_sp_1s)

        _ALSSM_CFGS_D1 = [
            ('AlssmPoly',        lm.AlssmPoly,         {},                                       '#e15759', '-',  'o'),
            ('AlssmPolyJordan',  lm.AlssmPolyJordan,   {},                                       '#f28e2b', '--', 's'),
            ('AlssmPolyLegendre',lm.AlssmPolyLegendre, {'a_seg': a_seg_fin, 'b_seg': b_seg_fin}, '#4e79a7', '-',  '^'),
        ]
        # D2a and D2b share the same ALSSM classes; Meixner g is overridden per sub-panel
        _ALSSM_CFGS_D2 = [
            ('AlssmPoly',        lm.AlssmPoly,        {},                  '#e15759', '-',  'o'),
            ('AlssmPolyJordan',  lm.AlssmPolyJordan,  {},                  '#f28e2b', '--', 's'),
            ('AlssmPolyMeixner', lm.AlssmPolyMeixner, {'segment': _segment_d1},     '#76b7b2', '-',  '^'),
        ]

        # ── D1: finite two-sided ──────────────────────────────────────────────
        print("\n--- D1: finite two-sided ---")
        panel_d_data2[0] = {}
        for _cls_lbl, _cls, _kw, *_ in _ALSSM_CFGS_D1:
            _vals = []
            for _deg in _degs_d:
                try:
                    _al  = _cls(poly_degree=_deg, **_kw)
                    _sL  = lm.Segment(a=a_seg_fin, b=-1,       direction=lm.FORWARD,  g=_g_sp_fin)
                    _sR  = lm.Segment(a=0,         b=b_seg_fin, direction=lm.BACKWARD, g=_g_sp_fin)
                    _cc  = lm.CompositeCost((_al,), (_sL, _sR), [[1, 1]])
                    _rl  = lm.RLSAlssm(_cc, steady_state=True, backend='lfilter')
                    with _w.catch_warnings(): _w.simplefilter('ignore')
                    _rl.filter(_y)
                    _xs   = _rl.minimize_x()
                    _xs_r = _xs[_K_REF_FIN]
                    _W    = _rl.W
                    _delta = _xs[:, 1:] - _xs_r[1:]
                    _dist  = np.einsum('ki,ij,kj->k', _delta, _W[1:,1:], _delta)
                    _vals.append(_compute_snr_ratio(_dist, _TRUE_SPIKES, _spike_mask, _K, _SPK_WIN))
                except Exception:
                    _vals.append(np.nan)
            panel_d_data2[0][_cls_lbl] = _vals
            _v = [v for v in _vals if not np.isnan(v)]
            print(f"  {_cls_lbl}: max SNR={max(_v):.3f} at D={_degs_d[_vals.index(max(_v))]}" if _v else f"  {_cls_lbl}: all NaN")

        # ── D2a: one-sided infinite (right segment only, BACKWARD) ───────────
        print("\n--- D2a: one-sided infinite [0,inf) BACKWARD ---")
        panel_d_data2[1] = {}
        for _cls_lbl, _cls, _kw, *_ in _ALSSM_CFGS_D2:
            _vals = []
            # override Meixner g to match one-sided g
            _kw_1s = {k: (_g_sp_1s if k == 'g' else v) for k, v in _kw.items()}
            for _deg in _degs_d:
                try:
                    _al  = _cls(poly_degree=_deg, **_kw_1s)
                    _sR  = lm.Segment(a=0, b=np.inf, direction=lm.BACKWARD, g=_g_sp_1s)
                    _cc  = lm.CompositeCost((_al,), (_sR,), [1])
                    _rl  = lm.RLSAlssm(_cc, steady_state=True, backend='lfilter')
                    with _w.catch_warnings(): _w.simplefilter('ignore')
                    _rl.filter(_y)
                    _xs   = _rl.minimize_x()
                    _xs_r = _xs[_K_REF_1S]
                    _W    = _rl.W
                    _delta = _xs[:, 1:] - _xs_r[1:]
                    _dist  = np.einsum('ki,ij,kj->k', _delta, _W[1:,1:], _delta)
                    _vals.append(_compute_snr_ratio(_dist, _TRUE_SPIKES, _spike_mask, _K, _SPK_WIN))
                except Exception:
                    _vals.append(np.nan)
            panel_d_data2[1][_cls_lbl] = _vals
            _v = [v for v in _vals if not np.isnan(v)]
            print(f"  {_cls_lbl}: max SNR={max(_v):.3f} at D={_degs_d[_vals.index(max(_v))]}" if _v else f"  {_cls_lbl}: all NaN")

        # ── D2b: two-sided infinite (−∞,−1] ∪ [0,∞) ─────────────────────────
        # Two separate ALSSMs (one per segment) with F=[[1,0],[0,1]].
        # For AlssmPolyMeixner, the left ALSSM uses segment= so that
        # A_L = inv(A_bw), making the forward recursion contract correctly.
        # Applied uniformly to Poly and Jordan too, for a fair comparison.
        #
        # State layout: xs = [xs_L (N cols) | xs_R (N cols)], N = deg+1.
        # Distance uses shape indices only — skipping DC of each sub-ALSSM:
        #   shape_idx = [1..N-1] + [N+1..2N-1]
        print("\n--- D2b: two-sided infinite (two ALSSMs, Meixner uses segment=) ---")
        panel_d_data2[2] = {}
        for _cls_lbl, _cls, _kw, *_ in _ALSSM_CFGS_D2:
            _vals = []
            for _deg in _degs_d:
                try:
                    _N   = _deg + 1
                    _sL  = lm.Segment(a=-np.inf, b=-1,     direction=lm.FORWARD,  g=_g_sp_2s)
                    _sR  = lm.Segment(a=0,       b=np.inf, direction=lm.BACKWARD, g=_g_sp_2s)
                    # For Meixner: pass segment= so A_L = inv(A_bw) (forward basis)
                    # For Poly/Jordan: segment= is not used, fall back to plain constructor
                    if _cls is lm.AlssmPolyMeixner:
                        _alL = _cls(poly_degree=_deg, segment=_sL)
                        _alR = _cls(poly_degree=_deg, segment=_sR)
                    else:
                        _alL = _cls(poly_degree=_deg, **_kw)
                        _alR = _cls(poly_degree=_deg, **_kw)
                    _cc  = lm.CompositeCost((_alL, _alR), (_sL, _sR), F=[[1, 0], [0, 1]])
                    _rl  = lm.RLSAlssm(_cc, steady_state=True, backend='lfilter')
                    with _w.catch_warnings(): _w.simplefilter('ignore')
                    _rl.filter(_y)
                    _xs   = _rl.minimize_x()          # (K, 2*N)
                    _xs_r = _xs[_K_REF_2S]
                    _W    = _rl.W                      # (2*N, 2*N)
                    # Shape indices: skip DC of left (col 0) and DC of right (col N)
                    _sidx = list(range(1, _N)) + list(range(_N + 1, 2 * _N))
                    _delta = _xs[:, _sidx] - _xs_r[_sidx]
                    _Wsh   = _W[np.ix_(_sidx, _sidx)]
                    _dist  = np.einsum('ki,ij,kj->k', _delta, _Wsh, _delta)
                    _vals.append(_compute_snr_ratio(_dist, _TRUE_SPIKES, _spike_mask, _K, _SPK_WIN))
                except Exception:
                    _vals.append(np.nan)
            panel_d_data2[2][_cls_lbl] = _vals
            _v = [v for v in _vals if not np.isnan(v)]
            print(f"  {_cls_lbl}: max SNR={max(_v):.3f} at D={_degs_d[_vals.index(max(_v))]}" if _v else f"  {_cls_lbl}: all NaN")

        _panel_d_ok = True
    except Exception as _panel_d_err:
        import traceback; traceback.print_exc()
        print(f"  Panel D failed: {_panel_d_err}")
        _panel_d_ok = False

# ─── plot ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 22))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.36)
CTITLES = {
    1: f'Config 1: finite BW  [0,{B1}], g={G1}  (W={W1})',
    2: f'Config 2: infinite BW  [0,∞], g={G2}  (K={K2}, K_eval=1000–1099)',
}

for col, (cfg_idx, CFGS, CFGS_D) in enumerate([(1, CFGS1,_ALSSM_CFGS_D1), (2, CFGS2,_ALSSM_CFGS_D2)]):
    K_plot = K1 if cfg_idx == 1 else K2


    # Panel A
    if execute_section_B:
        ax = fig.add_subplot(gs[0, col])
        for lbl, method, color, ls, marker in CFGS:
            ax.semilogy(DEGREES, cond_W[cfg_idx][lbl],
                        color=color, ls=ls, lw=2, marker=marker, ms=5, label=lbl)
        ax.axhline(1/np.finfo(float).eps, color='red', ls=':', lw=1.2, label='1/ε (machine)')
        ax.set_xticks(DEGREES[::2]); ax.set_xlabel('degree D'); ax.set_ylabel('cond(W)')
        ax.set_title(f'Panel A{col+1} — Gram matrix cond(W)\n{CTITLES[cfg_idx]}',
                     fontsize=8.5, fontweight='bold')
        ax.set_ylim(1e-1, 1e24)
        ax.legend(fontsize=7.5, loc='upper left')

    # Panel B — ξ recursion drift
    if execute_section_C:
        max_xi   = {1: {c[0]: max_ok(xi_drift[1][c[0]]) for c in CFGS1},
                    2: {c[0]: max_ok(xi_drift[2][c[0]]) for c in CFGS2}}
        # The colored tick-marks on the right axis show which sample k produced the
        # worst drift for each method; they share the line color of the corresponding curve.
        ax  = fig.add_subplot(gs[1, col])
        ax2 = ax.twinx()
        k_scatter_handles = []   # collect for legend
        for lbl, method, color, ls, marker in CFGS:
            if method in ('scipy', 'meixner_ref'): continue
            ax.semilogy(DEGREES, xi_drift[cfg_idx][lbl],
                        color=color, ls=ls, lw=2, marker=marker, ms=5, label=lbl)
            pts = [(DEGREES[i], xi_drift_k[cfg_idx][lbl][i]) for i in range(len(DEGREES))
                   if xi_drift_k[cfg_idx][lbl][i] is not None
                   and not (isinstance(xi_drift_k[cfg_idx][lbl][i], float)
                            and np.isnan(xi_drift_k[cfg_idx][lbl][i]))]
            if pts:
                degs_p, ks_p = zip(*pts)
                sc = ax2.scatter(degs_p, ks_p, color=color, marker='|', s=40,
                                 linewidths=1.5, zorder=3, label=f'k @ max drift ({lbl})')
                k_scatter_handles.append(sc)
        ax.axhline(THR,                 color='orange', ls='--', lw=1.5,
                   label=f'threshold ({THR:.0e})')
        ax.axhline(np.finfo(float).eps, color='gray',   ls=':', lw=1.2, label='machine ε')
        ax.set_xticks(DEGREES[::2]); ax.set_xlabel('degree D')
        ax.set_ylabel('max_k ‖ξ_rec − ξ_ref‖ / ‖ξ_ref‖')
        ax2.set_ylabel('k where max drift occurs  (tick marks, right axis)', fontsize=7)
        ax2.set_ylim(0, K_plot)
        ax.set_title(f'Panel B{col+1} — ξ recursion drift\n{CTITLES[cfg_idx]}',
                     fontsize=8.5, fontweight='bold')
        ax.set_ylim(1e-16, 1e16)
        # Combined legend: left-axis curves + right-axis tick-mark explanation
        handles_l, labels_l = ax.get_legend_handles_labels()
        if k_scatter_handles:
            from matplotlib.lines import Line2D
            dummy = Line2D([0],[0], marker='|', color='gray', markersize=7,
                           linewidth=0, label='tick: k at max drift (right axis)')
            handles_l.append(dummy); labels_l.append('k at max drift (right axis)')
        ax.legend(handles_l, labels_l, fontsize=7.5, loc='upper left')

    # Panel C — SNR ratio metric on ECG template-matching task
    if execute_section_D:
        if col == 0:
            # D1: finite two-sided — single axis
            ax = fig.add_subplot(gs[2, 0])
            _cfgs_plot  = _ALSSM_CFGS_D1
            _data_key   = 0
            _title      = (f'Panel C1 — Template Matching via Euclidean Distance. SNR ratio, File {_file}\n'
                           f'finite two-sided segs [{a_seg_fin},−1],[0,{b_seg_fin}], g={_g_sp_fin}, K_REF={_K_REF_FIN}, ')
            if _panel_d_ok:
                for _cls_lbl,_,_, _color, _ls, _marker in _cfgs_plot:
                    _v = panel_d_data2[_data_key].get(_cls_lbl, [np.nan]*len(_degs_d))
                    ax.plot(_degs_d, _v, color=_color, ls=_ls, lw=2, marker=_marker, ms=5, label=_cls_lbl)
                ax.axhline(1.0, color='orange', ls='--', lw=1.5, label='ratio=1 (indistinguishable)')
                ax.set_xticks(_degs_d[::2]); ax.set_xlabel('polynomial degree D')
                ax.set_ylabel('min(noise dist) / max(spike dist)\n[SNR ratio; >1 = detectable]', fontsize=7.5)
                ax.legend(fontsize=7.5, loc='upper right')
                ax.set_ylim(bottom=0)
            else:
                ax.text(0.5, 0.5, 'ECG data unavailable', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9); ax.axis('off')
            ax.set_title(_title, fontsize=8, fontweight='bold')

        else:
            # col == 1: D2a (one-sided) and D2b (two-sided) stacked
            _inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, 1], hspace=0.5)
            ax_2a = fig.add_subplot(_inner[0])
            ax_2b = fig.add_subplot(_inner[1])

            _sub_panels = [
                (ax_2a, 1, _ALSSM_CFGS_D2,
                 f'Panel C2a — one-sided infinite seg [0,∞) BACKWARD, g={_g_sp_1s}, K_REF={_K_REF_1S}'),
                (ax_2b, 2, _ALSSM_CFGS_D2,
                 f'Panel C2b — two-sided infinite segs (−∞,−1], [0,∞), g={_g_sp_2s}, K_REF={_K_REF_2S}'),
            ]
            for _ax, _dkey, _cfgs_plot, _title in _sub_panels:
                if _panel_d_ok:
                    for _cls_lbl,_,_, _color, _ls, _marker in _cfgs_plot:
                        _v = panel_d_data2[_dkey].get(_cls_lbl, [np.nan]*len(_degs_d))
                        _ax.plot(_degs_d, _v, color=_color, ls=_ls, lw=2, marker=_marker, ms=5, label=_cls_lbl)
                    _ax.axhline(1.0, color='orange', ls='--', lw=1.5, label='ratio=1')
                    _ax.set_xticks(_degs_d[::2]); _ax.set_xlabel('polynomial degree D')
                    _ax.set_ylabel('SNR ratio', fontsize=7.5)
                    _ax.legend(fontsize=7, loc='upper right')
                    _ax.set_ylim(bottom=0)
                else:
                    _ax.text(0.5, 0.5, 'ECG data unavailable', ha='center', va='center',
                             transform=_ax.transAxes, fontsize=9); _ax.axis('off')
                _ax.set_title(_title, fontsize=8, fontweight='bold')

fig.suptitle(
    'Polynomial ALSSM Benchmark v2\n'
    f'K1={K1} (config1), K2={K2} (config2, K2≫g={G2})  |  K_eval 50–349  |  '
    f'{TRIALS} trials/deg  |  noise={NOISE:.0e}  |  threshold={THR:.0e}\n'
    "Panel A: filter output error  |  Panel D: distance = (xs[k,1:]-xs_ref[1:])^T W[1:,1:] (xs[k,1:]-xs_ref[1:])",
    fontsize=10, fontweight='bold', y=1.005,
)
out = './benchmark_poly_alssm.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nPlot saved: {out}')
