"""
Benchmarking of Stace-Space vs. Transfer-Function Backend [ex131.0]
===================================================================

This example demonstrates the usage of transfer-function (tf) backend in RLSAlssm* classes.
"""
import timeit
import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm
from lmlib.utils import load_lib_csv, load_lib_csv_mc


n_exe = 3  # number of filter executions


# setting all recursions to JIT backend default
backends = 'py-ss', 'py-tf', 'jit'
color_codes = 'k', 'b', 'g'
mspsecs_dict = {k: [] for k in backends}


y = load_lib_csv('EECG_BASELINE_1CH_10S_FS2400HZ.csv')
K = len(y)


# setup model
alssm = lm.AlssmPoly(poly_degree=2)
seg_l = lm.Segment(a=-50, b=-1, direction=lm.FW, g=100)
seg_r = lm.Segment(a=0, b=49, direction=lm.BW, g=100)
cost = lm.CompositeCost([alssm], [seg_l, seg_r], F=[[1, 1]])

# Single channel
for backend in backends:
    rls = lm.RLSAlssm(cost, backend=backend, steady_state=False)
    rls.filter_minimize_x(y)
    proc_time = timeit.timeit('rls.filter_minimize_x(y)', globals=globals(), number=n_exe)
    mspsec = K * n_exe * 1e-6 / proc_time
    mspsecs_dict[backend].append(mspsec)

# Single channel Steady State
for backend in backends:
    rls = lm.RLSAlssm(cost, backend=backend, steady_state=True)
    rls.filter_minimize_x(y)
    proc_time = timeit.timeit('rls.filter_minimize_x(y)', globals=globals(), number=n_exe)
    mspsec = K * n_exe * 1e-6 / proc_time
    mspsecs_dict[backend].append(mspsec)

# Multi channel Set
y = load_lib_csv_mc('EECG_FILT_9CH_10S_FS2400HZ.csv')
K, M = np.shape(y)

for backend in backends:
    rls = lm.RLSAlssm(cost, backend=backend, steady_state=False)
    rls.filter_minimize_x(y)
    proc_time = timeit.timeit('rls.filter_minimize_x(y)', globals=globals(), number=n_exe)
    mspsec = K * M * n_exe * 1e-6 / proc_time
    mspsecs_dict[backend].append(mspsec)

# Multi channel Set Steady State
for backend in backends:
    rls = lm.RLSAlssm(cost, backend=backend, steady_state=True)
    rls.filter_minimize_x(y)
    proc_time = timeit.timeit('rls.filter_minimize_x(y)', globals=globals(), number=n_exe)
    mspsec = K * M * n_exe * 1e-6 / proc_time
    mspsecs_dict[backend].append(mspsec)

mspsecs_dict['jit'][-1] = 0  # not implemented

labels = (r'$W$, $\xi$, $\kappa$, $\nu$'+' \n 1 Channel',
          r'Steady State, $\xi$, $\kappa$, $\nu$'+' \n 1 Channel',
          r'$W$, $\xi$, $\kappa$, $\nu$ '+f'\n {M} Channels',
          r'Steady State, $\xi$, $\kappa$, $\nu$'+f' \n {M} Channels')

locs = np.arange(len(labels))  # the label locations
width = 0.25

fig, ax = plt.subplots(figsize=(6, 5))

for i, (backend, color) in enumerate(zip(backends, color_codes)):
    rect_ = ax.barh(locs+width*i, mspsecs_dict[backend], width, label=backend, color=color)
    ax.bar_label(rect_, fmt="%0.2f", padding=3)

ax.invert_yaxis()  # labels read top-to-bottom

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('MS/s')
ax.set_title('Benchmarking of ALSSM filters: State-Space vs. Transfer Function backends')
ax.set_yticks(locs, labels)
ax.legend()

fig.tight_layout()

plt.show()
