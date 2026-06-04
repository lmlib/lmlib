"""
Benchmarking State-Space vs. Transfer-Function Backend (ND) [gu130.1]
======================================================================

Measures the throughput (in mega-samples per second, MS/s) of the
[`RLSAlssm`][lmlib.statespace.rls.RLSAlssm] filter using an
[`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost] for the ``numpy``,
``lfilter``, and (if available) ``jit`` backends.

An [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost] over a 2-D signal (image) is built using
the same separable polynomial basis as in example-ex801.0-Text-Recognition.py.

Each backend processes a (K1 x K2) image and results are reported in
mega-samples per second (MS/s = K1 * K2 * n_exe / proc_time * 1e-6).

"""
import timeit
import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm

lm.WARNING_NOT_STEADY_STATE = False

# ---- Signal & model setup --------------------------------------------------
K1 = 200   # image rows
K2 = 200   # image columns
Y  = np.random.randn(K1, K2)
n_exe = 2  # number of filter executions per timing measurement

g          = 100
l_side     = 35
poly_degree = 2

alssm_l = lm.AlssmPolyLegendre(poly_degree=poly_degree, a_seg=-l_side, b_seg=-1)
alssm_r = lm.AlssmPolyLegendre(poly_degree=poly_degree, a_seg=0, b_seg=l_side)
seg_l   = lm.Segment(a=-l_side, b=-1, direction=lm.FW, g=g)
seg_r   = lm.Segment(a=0, b=l_side, direction=lm.BW, g=g)
F       = [[1, 0], [0, 1]]

cost_d1 = lm.CompositeCost([alssm_l, alssm_r], [seg_l, seg_r], F)
cost_d2 = lm.CompositeCost([alssm_l, alssm_r], [seg_l, seg_r], F)
nd_cost = lm.NDCompositeCost([cost_d1, cost_d2])

# ---- Backend selection -----------------------------------------------------
backends     = ['numpy', 'lfilter']
color_codes  = 'k', 'b', 'g'
if lm.is_backend_available('jit'):
    backends.append('jit')

mspsecs_dict = {k: [] for k in backends}

# ---- Benchmark loop --------------------------------------------------------
for backend in backends:
    rls = lm.RLSAlssm(nd_cost, steady_state=True, backend=backend, filter_form='cascade')
    rls.filter(Y)                               # warm-up / JIT compile
    proc_time = timeit.timeit('rls.filter(Y)', globals=globals(), number=n_exe)
    mspsec    = K1 * K2 * n_exe * 1e-6 / proc_time
    mspsecs_dict[backend].append(mspsec)
    print(f"  {backend:8s}  {mspsec:.3f} MS/s")

# ---- Plot ------------------------------------------------------------------
labels = (f'NDCompositeCost.filter(Y)\n{K1}×{K2} image, poly_degree={poly_degree}',)
locs   = np.arange(len(labels))
width  = 0.25

fig, ax = plt.subplots(figsize=(9, 4))

for i, (backend, color) in enumerate(zip(backends, color_codes)):
    rect_ = ax.barh(locs + width * i, mspsecs_dict[backend], width,
                    label=backend, color=color)
    ax.bar_label(rect_, fmt="%0.2f", padding=4)

max_val = max(v[0] for v in mspsecs_dict.values())
ax.set_xlim(0, max_val * 1.25)
ax.invert_yaxis()
ax.set_xlabel('MS/s')
ax.set_title(
    'Benchmarking of ND ALSSM filters:\nState-Space vs. Transfer Function backends',
    pad=10,
)
ax.set_yticks(locs + width * (len(backends) - 1) / 2, labels, fontsize=9)
ax.legend()

fig.tight_layout()
plt.show()
