"""
Multichannel throughput: CPU lfilter vs batched GPU  [gu130.4]
==============================================================

This is the benchmark for the **batched** GPU path. A 1-D ``CompositeCost`` is
applied to ``S`` parallel channels of length ``K`` at once.

* ``lfilter`` (CPU): ``RLSAlssm.filter`` loops over the ``S`` channels internally
  (one scipy ``lfilter`` pass per channel).
* ``cupy`` (GPU): all ``S`` channels are processed in a **single GPU sweep**
  (one set of kernel launches + one device->host transfer for the whole batch).

Throughput is reported on the *total* sample count ``K*S`` (MS/s). The GPU
amortises its launch/transfer overhead as ``S`` grows, so the speedup should
climb with the number of channels even at modest ``K``.

Prints a parseable ``SUMMARY_JSON:`` line. Run on the GPU machine:

    python guide-gu130.4-multichannel-gpu.py            # default sweep
    python guide-gu130.4-multichannel-gpu.py --poly=2   # different model order
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

POLY_DEGREE = 1
for arg in sys.argv:
    if arg.startswith('--poly='):
        POLY_DEGREE = int(arg.split('=')[1])

HAVE_GPU = lm.is_backend_available('cupy')
if HAVE_GPU:
    import cupy as cp
else:
    print("cupy backend not available — running CPU-only (no speedup numbers).")

alssm = lm.AlssmPoly(poly_degree=POLY_DEGREE)
seg = lm.Segment(a=-21, b=-1, direction=lm.FW, g=100)
cost = lm.CompositeCost([alssm], [seg], F=[[1]])

K_LIST = [5_000, 50_000]
S_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

results = {"env": {"poly_degree": POLY_DEGREE}, "sweeps": []}
if HAVE_GPU:
    results["env"]["gpu"] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    results["env"]["cupy"] = cp.__version__


def bench_lfilter(Y, n_exe):
    rls = lm.RLSAlssm(cost, backend='lfilter')
    rls.filter(Y)  # warm-up / cache fill
    t = timeit.timeit(lambda: rls.filter(Y), number=n_exe)
    return t


def bench_cupy(Y, n_exe):
    rls = lm.RLSAlssm(cost, backend='cupy')
    rls.filter(Y); cp.cuda.Device().synchronize()  # warm-up
    def _run():
        rls.filter(Y); cp.cuda.Device().synchronize()
    return timeit.timeit(_run, number=n_exe)


for K in K_LIST:
    print(f"\n=== K = {K} ===")
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
            # reset the device pool between configs so a large prior allocation
            # does not starve this one (the backend also chunks to fit memory).
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

# ── plot speedup vs channel count ────────────────────────────────────────────
if HAVE_GPU:
    fig, ax = plt.subplots(figsize=(8, 5))
    for sweep in results["sweeps"]:
        pts = [(r["S"], r["speedup"]) for r in sweep["rows"] if r.get("speedup") is not None]
        if not pts:
            continue
        S = [p[0] for p in pts]
        sp = [p[1] for p in pts]
        ax.plot(S, sp, marker='o', label=f"K={sweep['K']}")
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('number of channels S')
    ax.set_ylabel('cupy speedup vs lfilter (x)')
    ax.set_title(f'Batched multichannel GPU speedup (poly_degree={POLY_DEGREE})')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('gu130.4-multichannel-gpu.png', dpi=120)
    print("\nsaved plot -> gu130.4-multichannel-gpu.png")

print("\nSUMMARY_JSON:" + json.dumps(results))
