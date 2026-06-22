"""
Multichannel throughput, PARALLEL form: CPU lfilter vs batched GPU  [gu130.6]
============================================================================

Companion to gu130.4 (cascade), for the **parallel** filter form. A 1-D
``CompositeCost`` is applied to ``S`` parallel channels of length ``K`` at once.

* ``lfilter`` (CPU): one scipy pass per channel (looped internally).
* ``cupy`` (GPU): all ``S`` channels in a **single GPU sweep** — the batched
  parallel path (``cupy_xi_q_recursion_parallel_batch``), one set of
  ``sosfilt``/FIR launches + one device->host transfer for the whole batch,
  with adaptive memory chunking.

The parallel form is the one to use for non-upper-triangular models (e.g.
``AlssmSin``); for polynomials the cascade form (gu130.4) is more accurate and
usually faster. This benchmark lets you see the parallel form's batching payoff
as ``S`` grows. Pass ``--model=sin`` to benchmark the complex-pole model, which
is the parallel form's real use case.

Prints a parseable ``SUMMARY_JSON:`` line. Run on the GPU machine:

    python guide-gu130.6-multichannel-parallel-gpu.py
    python guide-gu130.6-multichannel-parallel-gpu.py --model=sin
    python guide-gu130.6-multichannel-parallel-gpu.py --poly=2
"""
import sys
import json
import timeit
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lmlib as lm

lm.WARNING_NOT_STEADY_STATE = False
lm.set_gpu_dtype('float32')
MODEL = 'poly'
POLY_DEGREE = 1
for arg in sys.argv:
    if arg.startswith('--poly='):
        POLY_DEGREE = int(arg.split('=')[1])
    if arg.startswith('--model='):
        MODEL = arg.split('=')[1]

HAVE_GPU = lm.is_backend_available('cupy')
if HAVE_GPU:
    import cupy as cp
else:
    print("cupy backend not available — running CPU-only (no speedup numbers).")

if MODEL == 'sin':
    alssm = lm.AlssmSin(omega=0.1)
    seg = lm.Segment(a=-40, b=-1, direction=lm.FW, g=200)
    model_label = "AlssmSin(omega=0.1)"
else:
    alssm = lm.AlssmPoly(poly_degree=POLY_DEGREE)
    seg = lm.Segment(a=-21, b=-1, direction=lm.FW, g=100)
    model_label = f"AlssmPoly(pd={POLY_DEGREE})"
cost = lm.CompositeCost([alssm], [seg], F=[[1]])

K_LIST = [5_000, 50_000]
S_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

results = {"env": {"model": model_label, "filter_form": "parallel"}, "sweeps": []}
if HAVE_GPU:
    results["env"]["gpu"] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    results["env"]["cupy"] = cp.__version__


def bench_lfilter(Y, n_exe):
    rls = lm.RLSAlssm(cost, backend='lfilter', filter_form='parallel')
    rls.filter(Y)  # warm-up / cache fill
    return timeit.timeit(lambda: rls.filter(Y), number=n_exe)


def bench_cupy(Y, n_exe):
    rls = lm.RLSAlssm(cost, backend='cupy', filter_form='parallel')
    rls.filter(Y); cp.cuda.Device().synchronize()  # warm-up
    def _run():
        rls.filter(Y); cp.cuda.Device().synchronize()
    return timeit.timeit(_run, number=n_exe)


for K in K_LIST:
    print(f"\n=== K = {K}  ({model_label}, parallel) ===")
    header = f"{'S':>6} | {'total Msamp':>11} | {'lfilter':>10} | {'cupy':>10} | {'speedup':>8}"
    print(header); print("-" * len(header))
    sweep = {"K": K, "rows": []}
    for S in S_LIST:
        np.random.seed(0)
        Y = np.random.randn(K, S)
        total = K * S
        n_exe = max(1, min(20, int(3_000_000 / total)))

        t_lf = bench_lfilter(Y, n_exe)
        ms_lf = total * n_exe * 1e-6 / t_lf

        if HAVE_GPU:
            cp.get_default_memory_pool().free_all_blocks()
            try:
                t_cp = bench_cupy(Y, n_exe)
                ms_cp = total * n_exe * 1e-6 / t_cp
                sp = ms_cp / ms_lf
                print(f"{S:>6} | {total/1e6:>11.2f} | {ms_lf:>10.2f} | {ms_cp:>10.2f} | {sp:>7.2f}x")
                sweep["rows"].append({"S": S, "total": total, "ms_lfilter": ms_lf, "ms_cupy": ms_cp, "speedup": sp})
            except cp.cuda.memory.OutOfMemoryError:
                cp.get_default_memory_pool().free_all_blocks()
                print(f"{S:>6} | {total/1e6:>11.2f} | {ms_lf:>10.2f} | {'OOM':>10} | {'—':>8}")
                sweep["rows"].append({"S": S, "total": total, "ms_lfilter": ms_lf, "ms_cupy": None, "speedup": None})
        else:
            print(f"{S:>6} | {total/1e6:>11.2f} | {ms_lf:>10.2f} | {'—':>10} | {'—':>8}")
            sweep["rows"].append({"S": S, "total": total, "ms_lfilter": ms_lf})
    results["sweeps"].append(sweep)

if HAVE_GPU:
    fig, ax = plt.subplots(figsize=(8, 5))
    for sweep in results["sweeps"]:
        pts = [(r["S"], r["speedup"]) for r in sweep["rows"] if r.get("speedup") is not None]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', label=f"K={sweep['K']}")
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('number of channels S')
    ax.set_ylabel('cupy speedup vs lfilter (x)')
    ax.set_title(f'Batched multichannel GPU speedup — parallel form ({model_label})')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('gu130.6-multichannel-parallel-gpu.png', dpi=120)
    print("\nsaved plot -> gu130.6-multichannel-parallel-gpu.png")

print("\nSUMMARY_JSON:" + json.dumps(results))
