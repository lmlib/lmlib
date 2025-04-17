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

n_exe = 1  # number of filter executions

# setting all recursions to JIT backend
lm.set_backend('py-tf')

y = load_lib_csv('EECG_BASELINE_1CH_10S_FS2400HZ.csv')
K = len(y)
alssm = lm.AlssmPoly(poly_degree=2)
seg_l = lm.Segment(a=-50, b=-1, direction=lm.FW, g=100)
seg_r = lm.Segment(a=0, b=49, direction=lm.BW, g=100)
cost = lm.CompositeCost([alssm], [seg_l, seg_r], F=[[1, 1]])

# Single channel
rls_py_ss = lm.RLSAlssm(cost)
rls_py_ss.set_backend('py-ss')  # set individual RLSAlssm back to python backend

proc_time_py_ss = timeit.timeit('rls_py_ss.filter_minimize_x(y)', globals=globals(), number=n_exe)
mspsec_py_ss = K * n_exe * 1e-6 / proc_time_py_ss

rls_py_tf = lm.RLSAlssm(cost)
xs_py_tf = rls_py_tf.filter_minimize_x(y)  # create just in time compilation

proc_time_py_tf = timeit.timeit('rls_py_tf.filter_minimize_x(y)', globals=globals(), number=n_exe)
mspsec_py_tf = K * n_exe * 1e-6 / proc_time_py_tf

# Single channel Steady State
rls_py_ssss = lm.RLSAlssmSteadyState(cost)
rls_py_ssss.set_backend('py-ss')  # set individual RLSAlssm back to python backend

proc_time_py_ssss = timeit.timeit('rls_py_ssss.filter_minimize_x(y)', globals=globals(), number=n_exe)
mspsec_py_ssss = K * n_exe * 1e-6 / proc_time_py_ssss

rls_py_tfss = lm.RLSAlssmSteadyState(cost)
xs_py_tfss = rls_py_tfss.filter_minimize_x(y)  # create just in time compilation

proc_time_py_tfss = timeit.timeit('rls_py_tfss.filter_minimize_x(y)', globals=globals(), number=n_exe)
mspsec_py_tfss = K * n_exe * 1e-6 / proc_time_py_tfss

# Multi channel Set
y = load_lib_csv_mc('EECG_FILT_9CH_10S_FS2400HZ.csv')
K, M = np.shape(y)

rls_py_ss_set = lm.RLSAlssmSet(cost)
rls_py_ss_set.set_backend('py-ss')

proc_time_py_ss_set = timeit.timeit('rls_py_ss_set.filter_minimize_x(y)', globals=globals(), number=n_exe)
mspsec_py_ss_set = K * M * n_exe * 1e-6 / proc_time_py_ss_set

rls_py_tf_set = lm.RLSAlssmSet(cost)
xs_py_tf_set = rls_py_tf_set.filter_minimize_x(y)  # create just in time compilation

proc_time_py_tf_set = timeit.timeit('rls_py_tf_set.filter_minimize_x(y)', globals=globals(), number=n_exe)
mspsec_py_tf_set = K * M * n_exe * 1e-6 / proc_time_py_tf_set

# Multi channel Set Steady State

rls_py_ss_set_ss = lm.RLSAlssmSetSteadyState(cost)
rls_py_ss_set_ss.set_backend('py-ss')

proc_time_py_ss_set_ss = timeit.timeit('rls_py_ss_set_ss.filter_minimize_x(y)', globals=globals(), number=n_exe)
mspsec_py_ss_set_ss = K * M * n_exe * 1e-6 / proc_time_py_ss_set_ss

# Not Yet Implemented
# rls_py_tf_set_ss = lm.RLSAlssmSetSteadyState(cost)
# xs_py_tf_set_ss = rls_py_tf_set_ss.filter_minimize_x(y) # create just in time compilation
#
# proc_time_py_tf_set_ss = timeit.timeit('rls_py_tf_set_ss.filter_minimize_x(y)', globals=globals(), number=n_exe)
proc_time_py_tf_set_ss = np.nan
mspsec_py_tf_set_ss = K * M * n_exe * 1e-6 / proc_time_py_tf_set_ss

labels = ('RLSAlssm() \n 1 Channel', 'RLSAlssmSteadyState() \n  1 Channel', f'RLSAlssmSet() \n {M} Channels',
          f'RLSAlssmSetSteadyState() \n {M} Channels')
mspsecs_py_ss = (mspsec_py_ss, mspsec_py_ssss, mspsec_py_ss_set, mspsec_py_ss_set_ss)
mspsecs_py_tf = (mspsec_py_tf, mspsec_py_tfss, mspsec_py_tf_set, mspsec_py_tf_set_ss)
locs = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(6, 5))

rect0 = ax.barh(locs - width / 2, mspsecs_py_ss, width, label='py-ss', color='k')
rect1 = ax.barh(locs + width / 2, mspsecs_py_tf, width, label='py-tf', color='b')
ax.bar_label(rect0, fmt="%0.2f", padding=3)
ax.bar_label(rect1, fmt="%0.2f", padding=3)
ax.invert_yaxis()  # labels read top-to-bottom

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('MS/s')
ax.set_title('Benchmarking of ALSSM filters: State-Space vs. Transfer Function backends')
ax.set_yticks(locs, labels)
ax.legend()

fig.tight_layout()

plt.show()
