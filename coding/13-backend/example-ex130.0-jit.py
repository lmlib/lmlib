"""
Benchmarking of Python vs. JIT Backend [ex130.0]
===============================================

This example demonstrates the usage of Just-in-Time (JIT) compilation in RLSAlssm* classes.


"""
import timeit
import numpy as np
import matplotlib.pyplot as plt

import lmlib as lm
from lmlib.utils import load_data, load_data_mc
n_exe = 3 # number of filter executions

# setting all recursions to JIT backend
lm.set_backend('jit')

y = load_data('EECG_BASELINE_1CH_10S_FS2400HZ.csv')
K = len(y)
alssm = lm.AlssmPoly(poly_degree=2)
seg_l = lm.Segment(a=-50, b=-1, direction=lm.FW, g=100)
seg_r = lm.Segment(a=0, b=49, direction=lm.BW, g=100)
cost = lm.CompositeCost([alssm], [seg_l, seg_r], F=[[1, 1]])
cost = lm.CostSegment(alssm, seg_l)

# Single channel
rls_py = lm.RLSAlssm(cost)
rls_py.set_backend('py') # set individual RLSAlssm back to python backend

proc_time_py = timeit.timeit('rls_py.filter_minimize_x(y)', globals=globals(), number=n_exe)
mspsec_py = K*n_exe*1e-6/proc_time_py

rls_jit = lm.RLSAlssm(cost)
xs_jit = rls_jit.filter_minimize_x(y) # create just in time compilation

proc_time_jit = timeit.timeit('rls_jit.filter_minimize_x(y)', globals=globals(), number=n_exe)
mspsec_jit = K*n_exe*1e-6/proc_time_jit


# Single channel Steady State
rls_pyss = lm.RLSAlssmSteadyState(cost)
rls_pyss.set_backend('py') # set individual RLSAlssm back to python backend

proc_time_pyss = timeit.timeit('rls_pyss.filter_minimize_x(y)', globals=globals(), number=n_exe)
mspsec_pyss = K*n_exe*1e-6/proc_time_pyss

rls_jitss = lm.RLSAlssmSteadyState(cost)
xs_jitss = rls_jitss.filter_minimize_x(y) # create just in time compilation

proc_time_jitss = timeit.timeit('rls_jitss.filter_minimize_x(y)', globals=globals(), number=n_exe)
mspsec_jitss = K*n_exe*1e-6/proc_time_jitss


# Multi channel Set
y = load_data_mc('EECG_FILT_9CH_10S_FS2400HZ.csv')
K, M = np.shape(y)

rls_py_set = lm.RLSAlssmSet(cost)
rls_py_set.set_backend('py')

proc_time_py_set = timeit.timeit('rls_py_set.filter_minimize_x(y)', globals=globals(), number=n_exe)
mspsec_py_set = K*M*n_exe*1e-6/proc_time_py_set

rls_jit_set = lm.RLSAlssmSet(cost)
xs_jit_set = rls_jit_set.filter_minimize_x(y) # create just in time compilation

proc_time_jit_set = timeit.timeit('rls_jit_set.filter_minimize_x(y)', globals=globals(), number=n_exe)
mspsec_jit_set = K*M*n_exe*1e-6/proc_time_jit_set


# Multi channel Set Steady State

rls_py_set_ss = lm.RLSAlssmSetSteadyState(cost)
rls_py_set_ss.set_backend('py')

proc_time_py_set_ss = timeit.timeit('rls_py_set_ss.filter_minimize_x(y)', globals=globals(), number=n_exe)
mspsec_py_set_ss = K*M*n_exe*1e-6/proc_time_py_set_ss


# Not Yet Implemented
# rls_jit_set_ss = lm.RLSAlssmSetSteadyState(cost)
# xs_jit_set_ss = rls_jit_set_ss.filter_minimize_x(y) # create just in time compilation
#
# proc_time_jit_set_ss = timeit.timeit('rls_jit_set_ss.filter_minimize_x(y)', globals=globals(), number=n_exe)
proc_time_jit_set_ss = np.nan
mspsec_jit_set_ss = K*M*n_exe*1e-6/proc_time_jit_set_ss


labels = ('RLSAlssm() \n 1 Channel', 'RLSAlssmSteadyState() \n  1 Channel', f'RLSAlssmSet() \n {M} Channels', f'RLSAlssmSetSteadyState() \n {M} Channels')
mspsecs_py = (mspsec_py, mspsec_pyss, mspsec_py_set, mspsec_py_set_ss)
mspsecs_jit = (mspsec_jit, mspsec_jitss, mspsec_jit_set, mspsec_jit_set_ss)
locs = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(6, 5))

rect0 = ax.barh(locs - width/2, mspsecs_py, width, label='py', color='k')
rect1 = ax.barh(locs + width/2, mspsecs_jit, width, label='jit', color='b')
ax.bar_label(rect0, fmt="%0.2f", padding=3)
ax.bar_label(rect1, fmt="%0.2f", padding=3)
ax.invert_yaxis()  # labels read top-to-bottom


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('MS/s')
ax.set_title('Benchmarking of ALSSM filters: Python vs. JIT backends')
ax.set_yticks(locs, labels)
ax.legend()

fig.tight_layout()

plt.show()