"""
Benchmarking State-Space vs. Transfer-Function Backend [gu131.0]
================================================================

Measures and compares the throughput (MS/s) of the
[`RLSAlssm`][lmlib.statespace.rls.RLSAlssm] filter across the ``numpy``,
``lfilter``, and (if available) ``jit`` backends for four configurations:

* Single-channel, non-steady-state
* Single-channel, steady-state
* Multi-channel parallel (M=6), non-steady-state
* Multi-channel parallel (M=6), steady-state

Results are displayed as a horizontal bar chart.

"""
import timeit
import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm

lm.WARNING_NOT_STEADY_STATE = False

n_exe = 2  # number of filter executions

backends =   ['numpy', 'lfilter' ]
if lm.is_backend_available('jit'): backends.append('jit')
color_codes = 'k', 'b', 'g'
mspsecs_dict = {k: [] for k in backends}

K = 1_000
M = 6
y = np.random.randn(K)

# setup model
alssm = lm.AlssmPoly(poly_degree=2)
seg_l = lm.Segment(a=-50, b=-1, direction=lm.FW, g=100)
seg_r = lm.Segment(a=0, b=49, direction=lm.BW, g=100)
cost = lm.CompositeCost([alssm], [seg_l], F=[[1]])

# Single channel
for backend in backends:
    rls = lm.RLSAlssm(cost, backend=backend, steady_state=False)
    rls.filter(y)
    proc_time = timeit.timeit('rls.filter(y)', globals=globals(), number=n_exe)
    mspsec = K * n_exe * 1e-6 / proc_time
    mspsecs_dict[backend].append(mspsec)

# Single channel Steady State
for backend in backends:
    rls = lm.RLSAlssm(cost, backend=backend, steady_state=True)
    rls.filter(y)
    proc_time = timeit.timeit('rls.filter(y)', globals=globals(), number=n_exe)
    mspsec = K * n_exe * 1e-6 / proc_time
    mspsecs_dict[backend].append(mspsec)

# Multi channel Set
y = np.random.randn(K, M)

for backend in backends:
    rls = lm.RLSAlssm(cost, backend=backend, steady_state=False)
    rls.filter(y)
    proc_time = timeit.timeit('rls.filter(y)', globals=globals(), number=n_exe)
    mspsec = K * M * n_exe * 1e-6 / proc_time
    mspsecs_dict[backend].append(mspsec)

# Multi channel Set Steady State
for backend in backends:
    if backend != 'jit':
        rls = lm.RLSAlssm(cost, backend=backend, steady_state=True)
        rls.filter(y)
        proc_time = timeit.timeit('rls.filter(y)', globals=globals(), number=n_exe)
        mspsec = K * M * n_exe * 1e-6 / proc_time
        mspsecs_dict[backend].append(mspsec)
    else:
        mspsecs_dict[backend].append(np.nan)


labels = ('RLSAlssm.filter(y, steady_state=False) \n 1 Channel',
          'RLSAlssm.filter(y, steady_state=True) \n 1 Channel',
          'RLSAlssm.filter(y, steady_state=False) \n Multi Set (Parallel)',
          'RLSAlssm.filter(y, steady_state=True) \n Multi Set (Parallel)')

locs = np.arange(len(labels))  # the label locations
width = 0.25

fig, ax = plt.subplots(figsize=(6, 5))

for i, (backend, color) in enumerate(zip(backends, color_codes)):
    rect_ = ax.barh(locs+width*i, mspsecs_dict[backend], width, label=backend, color=color)
    ax.bar_label(rect_, fmt="%0.2f", padding=0)

ax.invert_yaxis()  # labels read top-to-bottom

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('MS/s')
ax.set_title('Benchmarking of ALSSM filters: State-Space vs. Transfer Function backends')
ax.set_yticks(locs, labels, fontsize=7)
ax.legend()

fig.tight_layout()

plt.show()
