"""
Trajectory plots for AlssmPoly variants at degrees 3, 7, 15
3×2 subplot grid:
  rows = degree (3, 7, 15)
  left  = Config 1: finite BW  [0,200], g=1000
  right = Config 2: infinite BW [0,∞],  g=100
"""
import sys, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legvander

sys.path.insert(0, '/home/claude')
import lmlib as lm

# ── parameters ─────────────────────────────────────────────────────────────
DEGREES_PLOT = [3, 7, 15]
NOISE  = 1e-8
SEED   = 7

K1, A1, B1, G1  = 1000, 0, 200, 1000
W1 = 201;  GAMMA1 = (G1-1)/G1
K_DISPLAY_1 = 400   # window [400 .. 600]

K2, A2, B2, G2  = 5000, 0, np.inf, 100
GAMMA2 = (G2-1)/G2
HORIZON2 = int(5*G2)     # 500 lags
K_DISPLAY_2 = 400   # window [400 .. 900]

CFGS1 = [
    ('AlssmPoly',          'poly',        '#e15759', '-',  1.5),
    ('AlssmPolyJordan',    'jordan',      '#f28e2b', '--', 1.5),
    ('AlssmPolyLegendre',  'legendre',    '#4e79a7', '-',  2.2),
    ('scipy ref',          'scipy',       '#59a14f', ':',  2.2),
]
CFGS2 = [
    ('AlssmPoly',          'poly',        '#e15759', '-',  1.5),
    ('AlssmPolyJordan',    'jordan',      '#f28e2b', '--', 1.5),
    ('AlssmPolyMeixner',   'meixner',     '#76b7b2', '-',  2.2),
    ('Meixner ref',        'meixner_ref', '#59a14f', ':',  2.2),
]

# ── helpers ─────────────────────────────────────────────────────────────────
def build_phi(A, C, horizon):
    N = A.shape[0]
    phi = np.empty((horizon, N))
    Aj = np.eye(N)
    for j in range(horizon):
        phi[j] = C @ Aj
        Aj = Aj @ A
    return phi

def run_rls(cost, y):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rls = lm.RLSAlssm(cost, steady_state=True, steady_state_method='schur')
        rls.filter(y)
    return rls

def make_cost(method, deg, cfg_idx):
    a, b, g = (A1, B1, G1) if cfg_idx == 1 else (A2, B2, G2)
    seg = lm.Segment(a, b, lm.BW, g=g)
    if   method == 'poly':     al = lm.AlssmPoly(poly_degree=deg)
    elif method == 'jordan':   al = lm.AlssmPolyJordan(poly_degree=deg)
    elif method == 'legendre': al = lm.AlssmPolyLegendre(poly_degree=deg, a_seg=a, b_seg=b)
    elif method == 'meixner':  al = lm.AlssmPolyMeixner(poly_degree=deg, segment=seg)
    else: return None, None
    return lm.CostSegment(al, seg), al

def poly_signal(K, W, deg, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(K, dtype=float)
    c = rng.standard_normal(deg+1) * (1.0/W)**np.arange(deg+1)
    y_true = sum(c[j]*t**j for j in range(deg+1))
    y = y_true + rng.standard_normal(K)*NOISE
    return y, y_true

# ── compute trajectories ────────────────────────────────────────────────────
def trajectories_cfg1(deg, k):
    y, y_true = poly_signal(K1, W1, deg, SEED)
    t_sc = 2*np.arange(W1, dtype=float)/(W1-1) - 1
    Vs = legvander(t_sc, deg)
    w  = np.sqrt(GAMMA1**np.arange(W1))
    out = {'ground truth': y_true[k:k+W1], 'noisy signal': y[k:k+W1]}
    for lbl, method, *_ in CFGS1:
        if method == 'scipy':
            cf, _, _, _ = np.linalg.lstsq(Vs*w[:,None], y[k:k+W1]*w, rcond=None)
            out[lbl] = Vs @ cf
        else:
            cost, al = make_cost(method, deg, 1)
            rls = run_rls(cost, y)
            xs  = rls.minimize_x(solver='lstsq')
            out[lbl] = build_phi(al.A, al.C, W1) @ xs[k]
    return out

def trajectories_cfg2(deg, k):
    y, y_true = poly_signal(K2, HORIZON2, deg, SEED)
    a, b, g = (A2, B2, G2)
    seg = lm.Segment(a, b, lm.BW, g=g)
    al_ref = lm.AlssmPolyMeixner(poly_degree=deg, segment=seg)
    phi_ref = build_phi(al_ref.A, al_ref.C, HORIZON2)
    w = np.sqrt(GAMMA2**np.arange(HORIZON2))
    out = {'ground truth': y_true[k:k+HORIZON2], 'noisy signal': y[k:k+HORIZON2]}
    for lbl, method, *_ in CFGS2:
        if method == 'meixner_ref':
            cf, _, _, _ = np.linalg.lstsq(phi_ref*w[:,None], y[k:k+HORIZON2]*w, rcond=None)
            out[lbl] = phi_ref @ cf
        else:
            cost, al = make_cost(method, deg, 2)
            rls = run_rls(cost, y)
            xs  = rls.minimize_x(solver='lstsq')
            out[lbl] = build_phi(al.A, al.C, HORIZON2) @ xs[k]
    return out

# ── plotting helper ──────────────────────────────────────────────────────────
def plot_panel(ax, j_axis, trajs, cfgs, deg, cfg_label):
    gt  = trajs['ground truth']
    sig = trajs['noisy signal']

    # y-limits driven by ground truth + a little padding
    ylo = gt.min(); yhi = gt.max()
    pad = max((yhi - ylo) * 0.12, np.abs(yhi)*1e-6 + 1e-12)
    ylim = (ylo - pad, yhi + pad)

    ax.plot(j_axis, sig, color='#cccccc', lw=0.8, zorder=1, label='noisy signal')
    ax.plot(j_axis, gt,  color='black',   lw=0.5, zorder=1, label='ground truth')

    out_of_range_labels = []
    for lbl, method, color, ls, lw in cfgs:
        tr = trajs[lbl]
        # Check how much of the trajectory is inside ylim
        inside = (tr >= ylim[0]) & (tr <= ylim[1])
        frac_inside = inside.mean()
        if frac_inside < 0.05:
            # Essentially invisible — add to annotation, still plot but off-scale
            out_of_range_labels.append((lbl, color, 'out of range'))
        elif frac_inside < 0.8:
            out_of_range_labels.append((lbl, color, f'{100*frac_inside:.0f}% in range'))

        ax.plot(j_axis, tr, color=color, ls=ls, lw=lw, zorder=4,
                label=lbl, alpha=0.9)

    ax.set_ylim(ylim)
    ax.set_xlabel('lag  j', fontsize=8)
    ax.set_ylabel('y', fontsize=8)

    # Build legend; add out-of-range warnings
    leg = ax.legend(fontsize=7, loc='best', ncol=1,
                    framealpha=0.85, handlelength=2.0)

    # Annotate any out-of-range methods as italic note
    if out_of_range_labels:
        notes = '\n'.join(f'[{l}: {note}]' for l, _, note in out_of_range_labels)
        ax.text(0.02, 0.02, notes, transform=ax.transAxes,
                fontsize=6.5, color='#555555', va='bottom',
                style='italic', bbox=dict(fc='white', alpha=0.7, pad=2))
    return out_of_range_labels

# ── main plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 13),
                         gridspec_kw={'hspace': 0.52, 'wspace': 0.28})

for row, deg in enumerate(DEGREES_PLOT):
    print(f'Computing D={deg}...')

    # Config 1 (left)
    t1 = trajectories_cfg1(deg, K_DISPLAY_1)
    ax1 = axes[row, 0]
    plot_panel(ax1, np.arange(W1), t1, CFGS1, deg, 'Config 1')
    ax1.set_title(
        f'D={deg}  —  Config 1: finite BW  [0,{B1}], g={G1},  W={W1}\n'
        f'k={K_DISPLAY_1},  window k+[0,{W1-1}]',
        fontsize=8.5, fontweight='bold')

    # Config 2 (right)
    t2 = trajectories_cfg2(deg, K_DISPLAY_2)
    ax2 = axes[row, 1]
    plot_panel(ax2, np.arange(HORIZON2), t2, CFGS2, deg, 'Config 2')
    ax2.set_title(
        f'D={deg}  —  Config 2: infinite BW  [0,∞], g={G2}\n'
        f'k={K_DISPLAY_2},  horizon={HORIZON2}',
        fontsize=8.5, fontweight='bold')

fig.suptitle(
    'Fitted trajectories  C·Aʲ·x̂[k]  vs. ground truth  —  one representative k per panel\n'
    'Polynomial coefficients scaled as (1/W)ⁿ per degree  |  '
    f'noise={NOISE:.0e}  |  seed={SEED}  |  '
    'Out-of-ylim trajectories noted in italic',
    fontsize=10, fontweight='bold',
)

out = './benchmark_trajectories.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out}')
