"""
Benchmarking State-Space vs. Transfer-Function Backend for ND Costs [gu131.1]
==============================================================================

Measures and compares the throughput (MS/s) of the
[`RLSAlssm`][lmlib.statespace.rls.RLSAlssm] filter across the ``numpy``,
``lfilter``, and (if available) ``jit`` backends for four configurations
using a 2-D [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost]:

* Small image (100×100), low poly degree (1)
* Small image (100×100), high poly degree (2)
* Large image (300×300), low poly degree (1)
* Large image (300×300), high poly degree (2)

The ND cost uses a separable polynomial basis identical to the one used
in example-ex801.0-Text-Recognition.py (AlssmPolyLegendre, left + right
segments, F=[[1,0],[0,1]]).

Results are displayed as a horizontal bar chart.

"""
import timeit
import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm

lm.WARNING_NOT_STEADY_STATE = False

n_exe = 2  # number of filter executions per timing cell

backends    = ['numpy', 'lfilter']
color_codes = 'k', 'b', 'g'
if lm.is_backend_available('jit'):
    backends.append('jit')

mspsecs_dict = {k: [] for k in backends}


def build_nd_cost(poly_degree, l_side=35, g=100):
    """Build a 2-D NDCompositeCost with left+right PolyLegendre segments."""
    alssm_l = lm.AlssmPolyLegendre(poly_degree=poly_degree, a_seg=-l_side, b_seg=-1)
    alssm_r = lm.AlssmPolyLegendre(poly_degree=poly_degree, a_seg=0, b_seg=l_side)
    seg_l   = lm.Segment(a=-l_side, b=-1, direction=lm.FW, g=g)
    seg_r   = lm.Segment(a=0, b=l_side, direction=lm.BW, g=g)
    F       = [[1, 0], [0, 1]]
    cost_d1 = lm.CompositeCost([alssm_l, alssm_r], [seg_l, seg_r], F)
    cost_d2 = lm.CompositeCost([alssm_l, alssm_r], [seg_l, seg_r], F)
    return lm.NDCompositeCost([cost_d1, cost_d2])


# ---- Benchmark configurations ----------------------------------------------
configs = [
    dict(K1=100, K2=100, poly_degree=1),
    dict(K1=100, K2=100, poly_degree=2),
    dict(K1=300, K2=300, poly_degree=1),
    dict(K1=300, K2=300, poly_degree=2),
]

labels = tuple(
    f'NDCompositeCost  {c["K1"]}×{c["K2"]} image\npoly_degree={c["poly_degree"]}'
    for c in configs
)

for cfg in configs:
    K1, K2, pd = cfg['K1'], cfg['K2'], cfg['poly_degree']
    Y       = np.random.randn(K1, K2)
    nd_cost = build_nd_cost(pd)

    for backend in backends:
        rls = lm.RLSAlssm(nd_cost, steady_state=True, backend=backend, filter_form='cascade')
        rls.filter(Y)                         # warm-up / JIT compile
        proc_time = timeit.timeit('rls.filter(Y)', globals=globals(), number=n_exe)
        mspsec    = K1 * K2 * n_exe * 1e-6 / proc_time
        mspsecs_dict[backend].append(mspsec)
        print(f"  {backend:8s}  {K1}×{K2}  poly={pd}  {mspsec:.3f} MS/s")

# ---- Plot ------------------------------------------------------------------
locs  = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(7, 6))

for i, (backend, color) in enumerate(zip(backends, color_codes)):
    rect_ = ax.barh(locs + width * i, mspsecs_dict[backend], width,
                    label=backend, color=color)
    ax.bar_label(rect_, fmt="%0.2f", padding=3)

max_val = max(max(v) for v in mspsecs_dict.values())
ax.set_xlim(0, max_val * 1.25)
ax.invert_yaxis()
ax.set_xlabel('MS/s')
ax.set_title(
    'Benchmarking of ND ALSSM filters:\nState-Space vs. Transfer Function backends',
    pad=10,
)
ax.set_yticks(locs + width * (len(backends) - 1) / 2, labels, fontsize=7)
ax.legend()

fig.tight_layout()
plt.show()
