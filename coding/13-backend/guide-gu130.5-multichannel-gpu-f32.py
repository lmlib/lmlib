"""
Multichannel GPU: float32 vs float64 precision/speed tradeoff  [gu130.5]
=======================================================================

Compares, for a batch of ``S`` channels of length ``K``:

* ``lfilter``  : CPU reference (per-channel loop), float64
* ``cupy/f64`` : batched GPU, float64 device math (exact parity)
* ``cupy/f32`` : batched GPU, float32 device math (lever #1)

and reports both the throughput (MS/s on total samples) **and** the float32
relative error vs the lfilter reference, so the speed gain and the accuracy cost
are visible together. On GPUs with reduced FP64 throughput (most consumer/laptop
cards, incl. the RTX A2000) float32 is typically several times faster.

Toggle precision via the public API: ``lm.set_gpu_dtype('float32' | 'float64')``.

Run on the GPU machine:
    python guide-gu130.5-multichannel-gpu-f32.py
    python guide-gu130.5-multichannel-gpu-f32.py --poly=2 --K=50000
"""
import sys
import json
import timeit
import numpy as np
import lmlib as lm

lm.WARNING_NOT_STEADY_STATE = False

POLY_DEGREE = 1
K = 50_000
for arg in sys.argv:
    if arg.startswith('--poly='):
        POLY_DEGREE = int(arg.split('=')[1])
    if arg.startswith('--K='):
        K = int(arg.split('=')[1])

if not lm.is_backend_available('cupy'):
    print("cupy backend not available — nothing to compare.")
    sys.exit(0)
import cupy as cp

alssm = lm.AlssmPoly(poly_degree=POLY_DEGREE)
seg = lm.Segment(a=-21, b=-1, direction=lm.FW, g=100)
cost = lm.CompositeCost([alssm], [seg], F=[[1]])

S_LIST = [8, 32, 128, 512, 1024]
results = {"env": {"poly_degree": POLY_DEGREE, "K": K,
                   "gpu": cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                   "cupy": cp.__version__}, "rows": []}


def time_backend(Y, n_exe, gpu=False):
    rls = lm.RLSAlssm(cost, backend='cupy' if gpu else 'lfilter')
    rls.filter(Y)
    if gpu:
        cp.cuda.Device().synchronize()
        def _run():
            rls.filter(Y); cp.cuda.Device().synchronize()
        t = timeit.timeit(_run, number=n_exe)
    else:
        t = timeit.timeit(lambda: rls.filter(Y), number=n_exe)
    return t, rls


def relerr(ref, other):
    a, b = np.asarray(ref.xi, float), np.asarray(other.xi, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-12))


print(f"poly_degree={POLY_DEGREE}  K={K}  GPU={results['env']['gpu']}")
hdr = (f"{'S':>6} | {'lfilter':>9} | {'cupy/f64':>9} | {'cupy/f32':>9} | "
       f"{'f64 sp':>7} | {'f32 sp':>7} | {'f32 relerr':>11}")
print(hdr); print("-" * len(hdr))

for S in S_LIST:
    np.random.seed(0)
    Y = np.random.randn(K, S)
    total = K * S
    n_exe = max(1, min(20, int(3_000_000 / total)))

    cp.get_default_memory_pool().free_all_blocks()
    t_lf, r_lf = time_backend(Y, n_exe, gpu=False)
    ms_lf = total * n_exe * 1e-6 / t_lf

    lm.set_gpu_dtype('float64')
    cp.get_default_memory_pool().free_all_blocks()
    t64, r64 = time_backend(Y, n_exe, gpu=True)
    ms64 = total * n_exe * 1e-6 / t64

    lm.set_gpu_dtype('float32')
    cp.get_default_memory_pool().free_all_blocks()
    t32, r32 = time_backend(Y, n_exe, gpu=True)
    ms32 = total * n_exe * 1e-6 / t32
    lm.set_gpu_dtype('float64')  # restore default

    e32 = relerr(r_lf, r32)
    print(f"{S:>6} | {ms_lf:>9.2f} | {ms64:>9.2f} | {ms32:>9.2f} | "
          f"{ms64/ms_lf:>6.2f}x | {ms32/ms_lf:>6.2f}x | {e32:>11.2e}")
    results["rows"].append({"S": S, "total": total,
                            "ms_lfilter": ms_lf, "ms_cupy_f64": ms64, "ms_cupy_f32": ms32,
                            "speedup_f64": ms64 / ms_lf, "speedup_f32": ms32 / ms_lf,
                            "f32_relerr": e32})

print("\nSUMMARY_JSON:" + json.dumps(results))
