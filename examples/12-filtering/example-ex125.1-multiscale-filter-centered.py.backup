r"""
Centered Multi-Scale Polynomial (Savitzky-Golay) Smoothing [ex125.1]
====================================================================

Companion to [ex125.0] (``example-ex125.0-multiscale-filter.py``), but using the
symmetric, zero-lag centered windows that Savitzky-Golay smoothing normally uses
(FORWARD segment $a = -L/2$, $b = L/2 - 1$) instead of one-sided BACKWARD
windows. As before, two mathematically equivalent methods are compared:

* **Per-scale RLS** — one [`RLSAlssm`][lmlib.statespace.rls.RLSAlssm].fit() over a
  [`CompositeCost`][lmlib.statespace.cost.CompositeCost] per scale.
* **State reuse** — one recursion at the base scale, then each larger scale is
  built by *centered* dyadic composition of the filter state $(\xi, W)$. With the
  half-shift $h = L/2$, a length-$2L$ centered window is two length-$L$ windows
  centered at $k-h$ and $k+h$; each is propagated to the common reference $k$ by
  $A^{\pm h}$ and re-weighted:

$$
\xi_{2L}[k] = \gamma^{-h}\, \xi_L[k+h]\, A^{h} + \gamma^{+h}\, \xi_L[k-h]\, A^{-h}, \qquad
W_{2L} = \gamma^{-h}\, A^{h\mathsf{T}} W_L\, A^{h} + \gamma^{+h}\, A^{-h\mathsf{T}} W_L\, A^{-h}.
$$

(The scalar weights are $\gamma_\mathrm{fwd}^{\pm h}$; for a FORWARD segment lmlib
uses $\gamma_\mathrm{fwd} = 1/(1 - 1/g)$, so $\gamma_\mathrm{fwd}^{\pm h} =
\gamma^{\mp h}$ with $\gamma = 1 - 1/g$.) The script checks that both methods
agree numerically and reports the speed-up of state reuse.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm
from lmlib.utils.generator import gen_wgn, load_csv_mc

# ----------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------
data = load_csv_mc("amananandrai_eeg_s00.csv")  # ECG, fs=500Hz
CH       = 0
K        = 2000
k        = np.arange(K)
OVERLAP  = 800
data = data[:, CH]
data = data / np.nanmax(np.abs(data))
kstart = 24500
y = data[kstart:kstart + K + OVERLAP] + gen_wgn(K + OVERLAP, sigma=0.1, seed=3142)

# ----------------------------------------------------------------------
# Model and scales
# ----------------------------------------------------------------------
N      = 4                                   # poly_degree = 3
G      = 800
GAMMA  = 1.0 - 1.0 / G
BASE   = 10
SCALES = [BASE * 2**j for j in range(6)]     # 10,20,40,80,160,320


def build_rls(L):
    seg  = lm.Segment(a=-L // 2, b=L // 2 - 1, direction=lm.FORWARD, g=G)
    cost = lm.CompositeCost((lm.AlssmPolyJordan(poly_degree=N - 1),), (seg,), F=[[1]])
    return lm.RLSAlssm(cost, steady_state=True, backend='lfilter')

# ----------------------------------------------------------------------
# METHOD 1 -- per-scale RLS (no reuse of intermediate results)
# ----------------------------------------------------------------------
def method_per_scale(y):
    out = {}
    for L in SCALES:
        rls = build_rls(L)
        out[L] = rls.fit(y, output='y_hat')
    return out

# ----------------------------------------------------------------------
# METHOD 2 -- single recursion + dyadic centered xi/W reuse
# ----------------------------------------------------------------------
def method_reuse(y):
    rls0 = build_rls(BASE)
    A    = np.asarray(rls0._cost_terms.alssms[0].A, dtype=float)
    C    = np.asarray(rls0._cost_terms.alssms[0].C, dtype=float)
    rls0.filter(y)
    xi   = np.asarray(rls0.xi).copy()
    W    = np.asarray(rls0.W).copy()
    out  = {BASE: xi @ np.linalg.inv(W) @ C}
    L = BASE
    while L * 2 <= SCALES[-1]:
        h   = L // 2
        Ah  = np.linalg.matrix_power(A, h)        # A^{+h}
        Ai  = np.linalg.inv(Ah)                   # A^{-h}
        gp  = GAMMA ** (-h)                        # weight for the k+h half
        gm  = GAMMA ** (h)                         # weight for the k-h half
        v   = K + OVERLAP - h

        # centered composition of xi (needs k+h and k-h) and of the constant W
        xitilde = np.full_like(xi, np.nan)
        xitilde[h:v] = gp * (xi[2 * h:v + h] @ Ah) + gm * (xi[0:v - h] @ Ai)
        Wtilde       = gp * (Ah.T @ W @ Ah) + gm * (Ai.T @ W @ Ai)

        # smoothed output  y_hat = C x_hat = C W^{-1} xi
        out[L * 2] = xitilde @ np.linalg.inv(Wtilde) @ C

        # next layer
        xi = xitilde
        W  = Wtilde
        L *= 2
    return out

# ----------------------------------------------------------------------
# Equivalence
# ----------------------------------------------------------------------
A_out, B_out = method_per_scale(y), method_reuse(y)
sl = slice(300, K - 300)
print("=== Numerical equivalence  (interior k in [%d, %d)) ===" % (sl.start, sl.stop))
worst = max(np.nanmax(np.abs(A_out[L][sl] - B_out[L][sl])) for L in SCALES)
for L in SCALES:
    print(f"   L={L:4d}   max|per-scale - reuse| = "
          f"{np.nanmax(np.abs(A_out[L][sl]-B_out[L][sl])):.3e}")
print(f"   worst over all scales = {worst:.3e}\n")

# ----------------------------------------------------------------------
# Timing (repeated)
# ----------------------------------------------------------------------
def benchmark(fn, reps):
    fn(y)
    t = np.array([(lambda t0: (fn(y), time.perf_counter() - t0)[1])(time.perf_counter())
                  for _ in range(reps)]) * 1e3
    return t.mean(), t.std(), t.min()

REPS = 30
timing_separate_rls = benchmark(method_per_scale, REPS)
timing_with_reuse   = benchmark(method_reuse, REPS)
print(f"=== Timing over {REPS} runs  (M={len(SCALES)} scales, K+overlap={K+OVERLAP}) ===")
print(f"   Method 1  per-scale RLS  : mean {timing_separate_rls[0]:7.2f} ms  (+/- {timing_separate_rls[1]:4.2f})   best {timing_separate_rls[2]:7.2f} ms")
print(f"   Method 2  xi reuse       : mean {timing_with_reuse[0]:7.2f} ms  (+/- {timing_with_reuse[1]:4.2f})   best {timing_with_reuse[2]:7.2f} ms")
print(f"   mean speed-up            : {timing_separate_rls[0]/timing_with_reuse[0]:5.2f}x\n")

# ----------------------------------------------------------------------
# Verification plot
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
offset = 1
ax.plot(k, y[k], lw=0.5, alpha=0.5, c='gray', label='$y$')
for i, L in enumerate(SCALES):
    ax.plot(k, B_out[L][k] - (i + 1) * offset, lw=1.0, c='tab:blue', label=r'$\hat y$' if i == 0 else None)
    ax.text(8, A_out[L][0:200].max() - (i + 1) * offset + 0.05, f"$L={L}$", size=8)
ax.set_xlim([0, K]); ax.set_ylabel('amplitude')
ax.set_title('Multi-scale polynomial smoothing (centered)')
ax.legend(loc='upper right'); ax.set_xlabel('k')
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout()
plt.show()
