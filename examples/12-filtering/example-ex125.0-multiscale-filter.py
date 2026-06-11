"""
ex805.0 multi-scale polynomial (Savitzky-Golay-type) smoothing.

Two ways to obtain the smoothed signal at a dyadic family of window lengths
L in {10,20,40,...,640}:

  METHOD 1  "per-scale RLS"  (the ex805 approach)
      For every scale: create_rls -> filter(y)  -> read (W, xi).
      => the O(K) recursion is run M times.

  METHOD 2  "xi reuse"       (Proof_Multiscale.py idea)
      Run the recursion ONCE at the base scale, then build every larger
      scale by dyadic composition of (xi, W).  A length-2L window is two
      stacked length-L windows; the trailing one is propagated to the common
      reference by A^L and re-weighted by the segment decay gamma^L:

          xi_{2L}[k] = xi_L[k] + gamma^L * ( xi_L[k+L] @ A^L )
          W_{2L}     = W_L      + gamma^L * ( A^L.T @ W_L @ A^L )

Windows are one-sided BACKWARD segments (a=0, b=L-1).
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm
from lmlib.utils.generator import gen_wgn, load_csv_mc

# ----------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------

data = load_csv_mc("amananandrai_eeg_s00.csv") #ECG Data, fs is 500Hz , https://www.kaggle.com/datasets/amananandrai/complete-eeg-dataset/data
#ECG Data, fs is 500Hz , https://www.kaggle.com/datasets/amananandrai/complete-eeg-dataset/data
# channels are Fp1, Fp2, F3, F4, F7, F8, T3,T4, C3, C4, T5, T6, P3, P4, O1, O2, Fz, Cz, Pz. 
CH = 0 
K        = 2000
k        = np.arange(K)
OVERLAP  = 800                       # headroom for the widest forward window
data = data[:,CH]
data = data/np.nanmax(np.abs(data))
kstart = 24500
y        = data[kstart:kstart+ K + OVERLAP] + gen_wgn(K + OVERLAP, sigma=0.1, seed=3142)

# ----------------------------------------------------------------------
# Model and scales
# ----------------------------------------------------------------------
N      = 4                                   # poly_degree = 3
G      = 800
GAMMA  = 1.0 - 1.0 / G
BASE   = 10
SCALES = [BASE * 2**j for j in range(6)]     # 10,20,40,80,160,320


def build_rls(L):
    seg  = lm.Segment(a=0, b=L-1, direction=lm.BACKWARD, g=G)
    cost = lm.CompositeCost((lm.AlssmPolyJordan(poly_degree=N-1),), (seg,), F=[[1]])
    return lm.RLSAlssm(cost, steady_state=True, backend='lfilter')

# ----------------------------------------------------------------------
# METHOD 1 -- per-scale RLS (the usual concept, no reuse of intermediate results)
# ----------------------------------------------------------------------
def method_per_scale(y):
    out = {}
    for L in SCALES:
        rls = build_rls(L)
        out[L] = rls.fit(y,output='y_hat') #replaced rls.filter(y), rls.minimize(x), rls.eval
    return out

# ----------------------------------------------------------------------
# METHOD 2 -- single recursion + dyadic xi/W reuse
# ----------------------------------------------------------------------
def method_reuse(y):
    rls0 = build_rls(BASE)
    A      = np.asarray(rls0._cost_terms.alssms[0].A, dtype=float)
    C      = np.asarray(rls0._cost_terms.alssms[0].C, dtype=float)
    rls0.filter(y)
    xi   = np.asarray(rls0.xi).copy()
    W    = np.asarray(rls0.W).copy()
    out  = {BASE: xi @ np.linalg.inv(W) @ C}
    L = BASE
    while L * 2 <= SCALES[-1]:
        AL  = np.linalg.matrix_power(A, L) 
        gL  = GAMMA ** L
        v   = K + OVERLAP - L 
        
        #calculate xitilde and Wtilde from shorter window
        xitilde = np.full_like(xi, np.nan)
        xitilde[:v] = xi[:v] + gL * (xi[L:L+v] @ AL)
        Wtilde   = W + gL * (AL.T @ W @ AL)

        #calculate y hat
        out[L*2] = xitilde @ np.linalg.inv(Wtilde) @ C
        
        #prepare the next layer --> xitilde becomes xi, Wtilde becomes W, L is doubled.
        xi  = xitilde
        W  = Wtilde
        L  *= 2
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
    fn(y)                                     # warm-up
    t = np.array([(lambda t0: (fn(y), time.perf_counter()-t0)[1])(time.perf_counter())
                  for _ in range(reps)]) * 1e3
    return t.mean(), t.std(), t.min()

REPS = 30
timing_separate_rls = benchmark(method_per_scale, REPS)
timing_with_reuse = benchmark(method_reuse,     REPS)
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
    #ax.plot(k, A_out[L][k] - (i+1)*offset, lw=1.1, ls=(0,(4,3)), c='tab:red', label='Method 1 (per-scale)' if i == 0 else None)
    ax.plot(k, B_out[L][k] - (i+1)*offset,  lw=1.0, c='tab:blue', label=r'$\hat y$' if i == 0 else None)
    ax.text(8, A_out[L][0:200].max() - (i+1)*offset + 0.05, f"$L={L}$", size=8)
ax.set_xlim([0, K]); ax.set_ylabel('amplitude')
ax.set_title('Multi-scale polynomial smoothing')
ax.legend(loc='upper right')
ax.set_xlabel('k')
for s in ('top', 'right'): ax.spines[s].set_visible(False)

plt.show()