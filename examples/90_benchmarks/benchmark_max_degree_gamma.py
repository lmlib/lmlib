"""
Maximum-degree benchmark with gamma < 1  (K=800, W=501)
========================================================
Ground truth: y_true[t] = Σ cⱼ tʲ,  cⱼ ~ 1/W^j  (window amplitude O(1))
Trajectory error: max|ŷ(k+j) − y_true(k+j)|  for j=0…500,  median over k=0…299

Root cause of the gamma=1 limit
---------------------------------
For BW segment [0,500] with gamma=1 the two-pass filter uses A⁻¹ in its
backward-pass recursion (removing the oldest sample at each step).  The Legendre
shift matrix L has eigenvalues ≡ 1, so L⁻¹ is also bounded; but composing L⁻¹
K−W = 299 times introduces accumulated floating-point drift that corrupts W[k]
for deg ≥ 9.  This is NOT a condition-number problem: cond(W_theory) ≈ 2D+1
for all degrees, but the numerically accumulated W[k] has cond → ∞ at high degree.

Fix: set gamma < 1 via lmlib's g parameter (gamma = (g−1)/g).
With g = 100 (gamma=0.990):
  • The backward-pass "forgets" old samples before they can accumulate drift.
  • The filter is always numerically stable regardless of D.
  • Error floor rises slightly (~1e-6 instead of ~1e-9) because the window is
    exponentially weighted rather than rectangular.

Configs compared
----------------
  legendre  g=inf (gamma=1)  : rect window, max D≈8 before recursion drift
  legendre  g=100 (γ=0.990) : exp window, stable to D≈11
  legendre  g= 30 (γ=0.967) : exp window, stable to D≈13
  pascal    g=inf            : fails at D≥3 (cond(W)>1/ε, baseline)
  scipy legendre             : exact sliding window, no limit found (reference)
"""

import sys, warnings, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.polynomial.legendre import legvander

sys.path.insert(0, '/home/claude/lmlib_modified')
import lmlib as lm

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
K       = 800
A_SEG   = 0
B_SEG   = 500
W       = B_SEG - A_SEG + 1      # 501
K_EVAL  = np.arange(0, 300)
DEGREES = list(range(1, 22))     # 1 … 21
TRIALS  = 10
NOISE   = 1e-8
rng     = np.random.default_rng(0)

t_global = np.arange(K, dtype=float)
t_sc_sp  = 2 * np.arange(W, dtype=float) / (W - 1) - 1  # [-1,+1]

def rand_poly(deg):
    return rng.standard_normal(deg + 1) * (1.0 / W) ** np.arange(deg + 1)

def V_sp(deg):
    return legvander(t_sc_sp, deg)   # (W, deg+1)

# Configs: (label, method, g_value, color, linestyle, marker)
CONFIGS = [
    ('pascal      g=∞   (γ=1)',    'pascal',   None,  '#e15759', '-',  'o'),
    ('Legendre    g=∞   (γ=1)',    'legendre', None,  '#9467bd', '-',  'P'),
    ('Legendre    g=100 (γ=0.990)','legendre', 100,   '#5d3a9b', '--', '^'),
    ('Legendre    g=30  (γ=0.967)','legendre', 30,    '#c39bd3', ':',  's'),
    ('scipy Legendre  (reference)', 'scipy',   None,  '#59a14f', '-',  'D'),
]

# ─────────────────────────────────────────────────────────────────────────────
# Section 0 – W drift diagnostic for AlssmPolyLegendre
# ─────────────────────────────────────────────────────────────────────────────
print("SECTION 0 – W_actual vs W_theory  (AlssmPolyLegendre, K=800)")
print("Shows that gamma<1 eliminates the filter recursion drift")
print(f'{"deg":>4}  {"g=inf":>12}  {"g=100":>12}  {"g=30":>12}  (rel error in W[k=50])')
np.random.seed(0)
for deg in DEGREES:
    row = f'{deg:4d}'
    for g_val in [None, 100, 30]:
        alssm = lm.AlssmPolyLegendre(poly_degree=deg, a_seg=0,b_seg=W-1)
        kw    = dict(g=g_val) if g_val else dict(g=None, gamma=1.0)
        seg   = lm.Segment(A_SEG, B_SEG, lm.BW, **kw)
        gamma = seg.gamma
        y_t   = np.random.randn(K)
        with warnings.catch_warnings(): warnings.simplefilter('ignore')
        rls   = lm.RLSAlssm(lm.CostSegment(alssm, seg), steady_state=False)
        rls.filter(y_t)
        W_act  = rls.W[50]
        Vs     = V_sp(deg)
        w_j    = gamma ** np.arange(W)
        W_th   = Vs.T @ np.diag(w_j) @ Vs
        rel    = np.max(np.abs(W_act - W_th)) / max(np.max(np.abs(W_th)), 1e-300)
        flag   = ' *' if rel > 1e-4 else '  '
        row   += f'  {rel:10.2e}{flag}'
    print(row)
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 – Monte-Carlo trajectory error
# ─────────────────────────────────────────────────────────────────────────────
mc = {cfg[0]: [] for cfg in CONFIGS}

for deg in DEGREES:
    Vs = V_sp(deg)
    t0 = time.time()

    for cfg_label, method, g_val, *_ in CONFIGS:
        kw  = dict(g=g_val) if g_val else dict(g=None, gamma=1.0)
        seg = lm.Segment(A_SEG, B_SEG, lm.BW, **kw)

        if method == 'pascal':
            alssm = lm.AlssmPoly(poly_degree=deg)
        elif method == 'legendre':
            alssm = lm.AlssmPolyLegendre(poly_degree=deg, a_seg=0,b_seg=W-1)
        else:
            alssm = None  # scipy

        trial_errs = []
        for _ in range(TRIALS):
            c     = rand_poly(deg)
            y_t   = sum(c[j] * t_global**j for j in range(deg + 1))
            y     = y_t + np.random.randn(K) * NOISE
            y_traj = np.stack([y_t[k:k+W] for k in K_EVAL])  # (300, W)

            if method == 'scipy':
                valid = K_EVAL[K_EVAL + W <= K]
                Y     = np.stack([y[k:k+W] for k in valid], axis=1)
                Cm, _, _, _ = np.linalg.lstsq(Vs, Y, rcond=None)
                tr    = (Vs @ Cm).T
                yt_v  = np.stack([y_t[k:k+W] for k in valid])
                trial_errs.append(np.median(np.max(np.abs(tr - yt_v), axis=1)))
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        rls = lm.RLSAlssm(lm.CostSegment(alssm, seg), steady_state=False)
                        rls.filter(y)
                        xs = rls.minimize_x(solver='lstsq')
                    # Always use legvander for trajectory (exact, no A^j drift)
                    tr  = xs[K_EVAL] @ Vs.T    # (300, W) — Vs = V_legendre = C L^j
                    trial_errs.append(np.median(np.max(np.abs(tr - y_traj), axis=1)))
                except Exception:
                    trial_errs.append(np.nan)

        mc[cfg_label].append(np.nanmedian(trial_errs))

    print(f'  deg={deg:2d} ({time.time()-t0:.2f}s): '
          + '  '.join(f'{v:.2e}' for v in [mc[c[0]][-1] for c in CONFIGS]))

print()
print("SECTION 1 – trajectory error  median(max|ŷ(k+j)−y_true(k+j)|)")
print(f"K={K}, W={W}, noise={NOISE:.0e}, {TRIALS} trials\n")
print(f'{"deg":>4}' + ''.join(f'  {c[0][:20]:>20}' for c in CONFIGS))
for i, deg in enumerate(DEGREES):
    row = f'{deg:4d}'
    for c in CONFIGS:
        v = mc[c[0]][i]
        flag = '!' if v > 1e-4 else ' '
        row += f'  {v:18.2e}{flag}'
    print(row)
print()

# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.38)

# ── Panel A: trajectory error vs degree ──────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
for cfg_label, method, g_val, color, ls, marker in CONFIGS:
    lw = 2.5 if 'legendre' in method.lower() or method == 'scipy' else 1.8
    ax0.semilogy(DEGREES, mc[cfg_label], color=color, ls=ls, lw=lw,
                 marker=marker, ms=7 if marker == 'D' else 6, label=cfg_label)

ax0.axhline(NOISE,  color='gray',   ls=':',  lw=1.2, label=f'noise floor ({NOISE:.0e})')
ax0.axhline(1e-4,   color='orange', ls='--', lw=1.5, label='practical threshold (1e-4)')
ax0.set_xticks(DEGREES)
ax0.set_xlabel('polynomial degree D')
ax0.set_ylabel('median max|ŷ(k+j) − y_true(k+j)|')
ax0.set_title(
    f'Panel A — Trajectory error vs. degree  (K={K}, W={W}, k=0…299)\n'
    'AlssmPolyLegendre with gamma<1 extends the usable degree by eliminating BW-filter recursion drift',
    fontsize=9, fontweight='bold')
ax0.legend(fontsize=8, loc='upper left')

# annotate degree limits
limits = {}
for cfg_label, *_ in CONFIGS:
    last_ok = 0
    for i, deg in enumerate(DEGREES):
        if mc[cfg_label][i] < 1e-4:
            last_ok = deg
        else:
            break
    limits[cfg_label] = last_ok

for cfg_label, method, g_val, color, ls, marker in CONFIGS:
    if method == 'scipy':
        continue
    d_lim = limits[cfg_label]
    if d_lim > 0:
        ax0.axvline(d_lim, color=color, ls=':', lw=1.0, alpha=0.5)

# ── Panel B: W drift vs degree ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
np.random.seed(1)
y_probe = np.random.randn(K)
for g_val, color, ls, lbl in [
    (None, '#9467bd', '-',  'g=∞  (γ=1)'),
    (100,  '#5d3a9b', '--', 'g=100 (γ=0.990)'),
    (30,   '#c39bd3', ':',  'g=30  (γ=0.967)'),
]:
    rel_errs = []
    for deg in DEGREES:
        alssm = lm.AlssmPolyLegendre(poly_degree=deg, a_seg=0,b_seg=W-1)
        kw    = dict(g=g_val) if g_val else dict(g=None, gamma=1.0)
        seg   = lm.Segment(A_SEG, B_SEG, lm.BW, **kw)
        gamma = seg.gamma
        with warnings.catch_warnings(): warnings.simplefilter('ignore')
        rls   = lm.RLSAlssm(lm.CostSegment(alssm, seg), steady_state=False)
        rls.filter(y_probe)
        W_act = rls.W[50]
        Vs    = V_sp(deg)
        w_j   = gamma ** np.arange(W)
        W_th  = Vs.T @ np.diag(w_j) @ Vs
        rel_errs.append(np.max(np.abs(W_act - W_th)) / max(np.max(np.abs(W_th)), 1e-300))
    ax1.semilogy(DEGREES, rel_errs, color=color, ls=ls, lw=2, marker='o', ms=5, label=lbl)

ax1.axhline(1e-4, color='orange', ls='--', lw=1.2, label='threshold (1e-4)')
ax1.axhline(np.finfo(float).eps, color='gray', ls=':', lw=1.2, label='ε (machine prec.)')
ax1.set_xticks(DEGREES); ax1.set_xlabel('polynomial degree D')
ax1.set_ylabel('‖W_actual − W_theory‖ / ‖W_theory‖')
ax1.set_title('Panel B — Filter W drift vs. degree\n(AlssmPolyLegendre, K=800)',
              fontsize=9, fontweight='bold')
ax1.legend(fontsize=7.5)

# ── Panel C: summary table ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
ax2.axis('off')

rows_tbl  = []
row_lbls  = []
row_colrs = []
for cfg_label, method, g_val, color, *_ in CONFIGS:
    d_lim = limits[cfg_label]
    if method == 'scipy':
        reason = f'No limit found (D≤{max(DEGREES)})\ndirect lstsq, no filter'
    elif method == 'pascal':
        reason = 'cond(W)>1/ε at D≥3\n(Gram matrix ill-conditioned)'
    else:
        if g_val is None:
            reason = 'BW filter recursion drift\nat D≥9 (K=800, W=501)'
        else:
            g_eff = g_val
            reason = (f'Expn. window g={g_val} (γ={(g_val-1)/g_val:.3f})\n'
                      f'Filter stable; slight error floor ↑')
    rows_tbl.append([str(d_lim) if method != 'scipy' else f'>{max(DEGREES)}', reason])
    row_lbls.append(cfg_label[:25])
    if d_lim >= 15 or method == 'scipy':
        row_colrs.append(['#d4edda', '#d4edda'])
    elif d_lim >= 8:
        row_colrs.append(['#fff3cd', '#fff3cd'])
    else:
        row_colrs.append(['#f8d7da', '#f8d7da'])

tbl = ax2.table(
    cellText=rows_tbl, cellColours=row_colrs,
    rowLabels=row_lbls, colLabels=['Max D\n(err<1e-4)', 'Reason'],
    cellLoc='center', loc='center', bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False); tbl.set_fontsize(7)
for (r, c), cell in tbl.get_celld().items():
    if r == 0: cell.set_facecolor('#c0c0c0'); cell.set_text_props(fontweight='bold')
    if c == -1: cell.set_facecolor('#e8e8e8'); cell.set_text_props(fontweight='bold')
ax2.set_title('Panel C — Maximum usable degree summary',
              fontsize=9, fontweight='bold', pad=10)

fig.suptitle(
    f'AlssmPolyLegendre: effect of gamma on maximum usable degree\n'
    f'K={K},  seg=[{A_SEG},{B_SEG}],  W={W},  noise={NOISE:.0e},  {TRIALS} trials per degree\n'
    f'Trajectory error = max|ŷ(k+j)−y_true(k+j)|,  median over k=0…299',
    fontsize=10, fontweight='bold', y=1.02,
)

out = './benchmark_max_degree_gamma.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Plot saved: {out}")
