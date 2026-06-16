"""
Benchmarking State-Space vs. Transfer-Function Backend [code910.0]
================================================================

Measures the throughput (in mega-samples per second, MS/s) of the
[`RLSAlssm`][lmlib.statespace.rls.RLSAlssm] filter for the ``numpy`` and
``lfilter`` backends on a single-channel signal.

The ``lfilter`` backend converts the state-space recursion to a cascade of
IIR/FIR transfer functions and uses [`scipy.signal.lfilter`][scipy.signal.lfilter], which
can be significantly faster for long signals when the model order is low.

"""
import timeit
import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm

lm.WARNING_NOT_STEADY_STATE = False
n_exe = 2
backends = ['lfilter', 'numpy']
color_codes = 'k', 'b'
mspsecs_dict = {k: [] for k in backends}

K = 100000
y = np.random.randn(K)

alssm = lm.AlssmPoly(poly_degree=1)
seg_l = lm.Segment(a=-21, b=-1, direction=lm.FW, g=100)
cost = lm.CompositeCost([alssm], [seg_l], F=[[1]])

for backend in backends:
    rls = lm.RLSAlssm(cost, backend=backend)
    rls.filter(y)
    proc_time = timeit.timeit('rls.filter(y)', globals=globals(), number=n_exe)
    mspsec = K * n_exe * 1e-6 / proc_time
    mspsecs_dict[backend].append(mspsec)

labels = ('RLSAlssm.filter(y, steady_state=True)\n1 Channel',)
locs = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 4))

for i, (backend, color) in enumerate(zip(backends, color_codes)):
    rect_ = ax.barh(locs + width * i, mspsecs_dict[backend], width, label=backend, color=color)
    ax.bar_label(rect_, fmt="%0.2f", padding=4)  # padding pushes label outside bar

# Extend x-axis to give label text room to breathe
max_val = max(v[0] for v in mspsecs_dict.values())
ax.set_xlim(0, max_val * 1.20)  # 20% headroom on the right

ax.invert_yaxis()
ax.set_xlabel('MS/s')
ax.set_title(
    'Benchmarking of ALSSM filters:\nState-Space vs. Transfer Function backends',
    pad=10
)
ax.set_yticks(locs + width / 2, labels, fontsize=8)  # center ticks between bars
ax.legend()

fig.tight_layout()
plt.show()
