"""
Throughput vs signal length K  [gu130.3]  — run on the GPU machine
==================================================================

Sweeps the signal length K and measures MS/s for the lfilter / numpy / cupy
backends, to locate the crossover where the GPU backend pulls ahead (and by how
much). Prints a parseable SUMMARY_JSON line at the end.

GPU timing is warmed-up and device-synchronised. numpy is only run for the
smaller K (it is ~300x slower and would make the sweep crawl); pass --with-numpy
to force it everywhere.
"""
import sys
import json
import timeit
import numpy as np
import lmlib as lm

lm.WARNING_NOT_STEADY_STATE = False

WITH_NUMPY = '--with-numpy' in sys.argv
POLY_DEGREE = 1
for a in sys.argv:
    if a.startswith('--poly='):
        POLY_DEGREE = int(a.split('=')[1])

backends = ['lfilter']
if lm.is_backend_available('cupy'):
    backends.append('cupy')
    import cupy as cp
backends.append('numpy')  # handled specially below

Ks = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 4_000_000]
NUMPY_MAX_K = 100_000  # skip numpy above this unless --with-numpy

alssm = lm.AlssmPoly(poly_degree=POLY_DEGREE)
seg = lm.Segment(a=-21, b=-1, direction=lm.FW, g=100)
cost = lm.CompositeCost([alssm], [seg], F=[[1]])

print(f"poly_degree={POLY_DEGREE}  backends={backends}")
print(f"{'K':>10} | " + " | ".join(f"{b:>10}" for b in backends) + "   (MS/s)")
print("-" * (12 + 13 * len(backends)))

results = {"env": {"poly_degree": POLY_DEGREE}, "rows": []}
if lm.is_backend_available('cupy'):
    results["env"]["gpu"] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()

for K in Ks:
    np.random.seed(0)
    y = np.random.randn(K)
    # pick number of repetitions so each timing is ~>=0.3s but bounded
    n_exe = max(2, min(50, int(2_000_000 / K)))
    row = {"K": K, "n_exe": n_exe, "msps": {}}
    cells = []
    for be in backends:
        if be == 'numpy' and not WITH_NUMPY and K > NUMPY_MAX_K:
            cells.append(f"{'—':>10}")
            continue
        rls = lm.RLSAlssm(cost, backend=be)
        if be == 'cupy':
            rls.filter(y); cp.cuda.Device().synchronize()
            def _run():
                rls.filter(y); cp.cuda.Device().synchronize()
            t = timeit.timeit(_run, number=n_exe)
        else:
            rls.filter(y)
            t = timeit.timeit('rls.filter(y)', globals={'rls': rls, 'y': y}, number=n_exe)
        msps = K * n_exe * 1e-6 / t
        row["msps"][be] = msps
        cells.append(f"{msps:10.2f}")
    print(f"{K:>10} | " + " | ".join(cells))
    results["rows"].append(row)

# speedup summary
print("\ncupy speedup vs lfilter:")
for r in results["rows"]:
    if 'cupy' in r["msps"] and 'lfilter' in r["msps"]:
        sp = r["msps"]['cupy'] / r["msps"]['lfilter']
        print(f"  K={r['K']:>9}: {sp:5.2f}x")

print("\nSUMMARY_JSON:" + json.dumps(results))
