"""
GPU phase profiler  [run on the GPU machine]
============================================

Breaks the batched GPU cascade (the work behind one xi^(1) recursion) into its
four phases and times each with CUDA events:

    H2D    : upload of the signal batch (host -> device)
    einsum : signal-shaping contractions that build y_diff
    iir    : the per-state cupyx.scipy.signal.lfilter cascade (sequential in time)
    D2H    : download of the result (device -> host)

This answers the question that the float32 sweep raised: at large channel counts,
is the backend bound by data movement (H2D + D2H) or by the IIR kernel? That
decides whether the "keep accumulators on device" optimization (#2, which removes
per-call transfers) is worth implementing.

Run:
    python gpu_profile.py                      # default K=50000, poly=1
    python gpu_profile.py --K=50000 --poly=2
"""
import sys
import numpy as np
import lmlib as lm

if not lm.is_backend_available('cupy'):
    print("cupy backend not available."); sys.exit(0)
import cupy as cp
from lmlib.statespace.backends.rec_lfilter import _compute_cascade_params
from lmlib.statespace.backends.rec_cupy import _first_order_lfilter, _block_ranges, _dt

K = 50_000
POLY = 1
for a in sys.argv:
    if a.startswith('--K='):
        K = int(a.split('=')[1])
    if a.startswith('--poly='):
        POLY = int(a.split('=')[1])

S_LIST = [8, 32, 128, 512, 1024]
REPEAT = 30

alssm = lm.AlssmPoly(poly_degree=POLY)
seg = lm.Segment(a=-21, b=-1, direction=lm.FW, g=100)
A, C = alssm.A, np.atleast_2d(alssm.C)     # force_MC: C is 2-D in the real path
cp_ = _compute_cascade_params(A, C, seg.a, seg.b, seg.delta, seg.gamma, 'fw')
a, b = seg.a, seg.b


def ev():
    e = cp.cuda.Event(); return e


def profile(K, S):
    Y = np.random.randn(K, 1, S)               # (K, Q=1, S)  device layout
    # host buffers / params
    gAinvT = cp.asarray(cp_['gAinvT'], dtype=_dt())
    Abc = cp.asarray(cp_['Abc'], dtype=_dt())
    Aac = cp.asarray(cp_['Aac'], dtype=_dt())
    N = cp_['N']
    poles = cp.diag(gAinvT).copy()
    gAinvT = gAinvT - cp.diagflat(cp.diag(gAinvT))
    gamma_a, gamma_b = cp_['gamma_a'], cp_['gamma_b']
    window_width = b - a + 1
    K_append = max(window_width, b + 1) if b >= 0 else window_width

    t = {k: 0.0 for k in ('h2d', 'einsum', 'iir', 'd2h')}
    # warm-up
    _ = cp.asarray(Y, dtype=_dt()); cp.cuda.Device().synchronize()

    for _ in range(REPEAT):
        e0, e1, e2, e3, e4 = ev(), ev(), ev(), ev(), ev()
        e0.record()
        yd = cp.asarray(Y, dtype=_dt())                       # (K, Q, S)  H2D
        yw = yd * cp.ones((K, 1, S), dtype=_dt())             # weighting (sw=1)
        e1.record()
        ydl_b = cp.zeros((K + K_append, 1, S), dtype=_dt()); ydl_b[0:K] = yw
        y_diff = cp.einsum('klB, nl->knB', ydl_b, gamma_b * Abc)
        ydl_a = cp.zeros((K + K_append, 1, S), dtype=_dt()); ydl_a[b - a + 1:b - a + 1 + K] = yw
        y_diff -= cp.einsum('klB, nl->knB', ydl_a, gamma_a * Aac)
        y_diff = cp.swapaxes(y_diff, 0, 1)                    # (N, K+app, S)
        e2.record()
        xi_add = cp.zeros((K + K_append, N, S), dtype=_dt(), order='F')
        for s, e in _block_ranges(None, N):
            xi_add[:, s] = _first_order_lfilter(poles[s], y_diff[s].T).T
            for n_ in range(s + 1, e):
                y_diff[n_, 1:] += cp.einsum('knB, n->kB', xi_add[:-1, s:e], gAinvT[n_, s:e])
                xi_add[:, n_] = _first_order_lfilter(poles[n_], y_diff[n_].T).T
        e3.record()
        if b >= 0:
            res = xi_add[b:b + K]
        else:
            res = cp.zeros((K, N, S), dtype=_dt()); res[-b:] = xi_add[0:K + b]
        out = cp.asnumpy(cp.moveaxis(res, -1, 0))             # D2H
        e4.record(); e4.synchronize()
        t['h2d'] += cp.cuda.get_elapsed_time(e0, e1)
        t['einsum'] += cp.cuda.get_elapsed_time(e1, e2)
        t['iir'] += cp.cuda.get_elapsed_time(e2, e3)
        t['d2h'] += cp.cuda.get_elapsed_time(e3, e4)
    total = sum(t.values())
    return t, total


print(f"poly_degree={POLY}  K={K}  GPU={cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
print(f"dtype={np.dtype(_dt()).name}  (set lm.set_gpu_dtype('float32') to profile f32)")
hdr = f"{'S':>6} | {'H2D %':>7} | {'einsum %':>9} | {'iir %':>7} | {'D2H %':>7} | {'ms/call':>8}"
print(hdr); print("-" * len(hdr))
for S in S_LIST:
    np.random.seed(0)
    t, total = profile(K, S)
    pct = {k: 100 * v / total for k, v in t.items()}
    print(f"{S:>6} | {pct['h2d']:>6.1f} | {pct['einsum']:>8.1f} | {pct['iir']:>6.1f} | "
          f"{pct['d2h']:>6.1f} | {total/REPEAT:>8.2f}")

print("\nInterpretation: if (H2D+D2H) dominates at large S, the keep-on-device")
print("optimization (#2) is worth it. If 'iir' dominates, we're kernel-bound and")
print("transfer tricks won't help much on this model order / GPU.")
