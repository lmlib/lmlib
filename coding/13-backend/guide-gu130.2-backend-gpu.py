"""
Benchmarking State-Space vs. Transfer-Function vs. GPU Backend [gu130.2]
=======================================================================

Extends [gu130.0] by adding the GPU (``cupy``) transfer-function backend to the
throughput comparison (mega-samples per second, MS/s) of the
[`RLSAlssm`][lmlib.statespace.rls.RLSAlssm] filter on a single-channel signal.

The ``cupy`` backend mirrors the ``lfilter`` cascade realization but runs the
per-sample signal shaping and the IIR cascade on the GPU via
[`cupyx.scipy.signal.lfilter`][cupyx.scipy.signal.lfilter].

Notes on fair GPU timing
------------------------
* CuPy kernel launches are asynchronous, so the device is synchronised around
  each timed call (``cp.cuda.Device().synchronize()``).
* The first GPU call pays one-off allocation/compilation costs; a warm-up call
  is issued before timing.
* The current backend transfers each segment's result back to host once per
  ``filter()`` call (a device->host copy). For short signals this transfer can
  dominate; the GPU advantage grows with signal length ``K``. Sweep ``K`` to see
  the crossover for your hardware.
"""
import timeit
import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm

lm.WARNING_NOT_STEADY_STATE = False
n_exe = 5

# include 'cupy' only if a GPU is actually available
backends = ['lfilter']
if lm.is_backend_available('cupy'):
    backends.append('cupy')
    import cupy as cp
else:
    print("cupy backend not available — benchmarking CPU backends only.")

color_map = {'lfilter': 'k', 'numpy': 'b', 'cupy': 'g'}

K = 1000000
y = np.random.randn(K)

alssm = lm.AlssmPoly(poly_degree=1)
seg_l = lm.Segment(a=-21, b=-1, direction=lm.FW, g=100)
cost = lm.CompositeCost([alssm], [seg_l], F=[[1]])

mspsecs_dict = {}
for backend in backends:
    rls = lm.RLSAlssm(cost, backend=backend,calc_kappa=False, calc_W=False, steady_state=False)

    if backend == 'cupy':
        # warm-up (allocations + first-call overhead) and synchronise
        rls.filter(y)
        cp.cuda.Device().synchronize()

        def _run():
            rls.filter(y)
            cp.cuda.Device().synchronize()
        proc_time = timeit.timeit(_run, number=n_exe)
    else:
        rls.filter(y)  # warm-up / cache fill
        proc_time = timeit.timeit('rls.filter(y)', globals=globals(), number=n_exe)

    mspsec = K * n_exe * 1e-6 / proc_time
    mspsecs_dict[backend] = mspsec
    print(f"{backend:8s}: {mspsec:8.2f} MS/s")

# ── plot ──────────────────────────────────────────────────────────────────────
labels = ('RLSAlssm.filter(y, steady_state=True)\n1 Channel',)
locs = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 4))
for i, backend in enumerate(backends):
    rect_ = ax.barh(locs + width * i, [mspsecs_dict[backend]], width,
                    label=backend, color=color_map.get(backend))
    ax.bar_label(rect_, fmt="%0.2f", padding=4)

max_val = max(mspsecs_dict.values())
ax.set_xlim(0, max_val * 1.20)
ax.invert_yaxis()
ax.set_xlabel('MS/s')
ax.set_title('Benchmarking of ALSSM filters:\n'
             'State-Space vs. Transfer Function vs. GPU backends', pad=10)
ax.set_yticks(locs + width, labels, fontsize=8)
ax.legend()
fig.tight_layout()
plt.savefig('gu130.2-backend-gpu.png', dpi=120)
plt.show()
