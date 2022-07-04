"""
JIT VS Python Backend [ex130.0]
===============================

Comparison between jit (just in time) and python backend

"""
import timeit

import lmlib as lm
from lmlib.utils import load_data, load_data_mc

# setting all recursions to JIT backend
lm.set_backend('jit')

print('Single channel')
y = load_data('EECG_BASELINE_1CH_10S_FS2400HZ.csv')
alssm = lm.AlssmPoly(3)
seg_l = lm.Segment(-50, -1, lm.FW, 1000)
seg_r = lm.Segment(0, 49, lm.BW, 1000)
cost = lm.CompositeCost([alssm], [seg_l, seg_r], F=[[1, 1]])

rls_py = lm.RLSAlssm(cost)
rls_py.set_backend('py') # set individual RLSAlssm back to python backend
xs_py = rls_py.filter_minimize_x(y)

rls_jit = lm.RLSAlssm(cost)
xs_jit = rls_jit.filter_minimize_x(y) # create just in time compilation

proc_time_py = timeit.timeit('rls_py.filter_minimize_x(y)', globals=globals(), number=10)
proc_time_jit = timeit.timeit('rls_jit.filter_minimize_x(y)', globals=globals(), number=10)

print(f'python backend: {proc_time_py:.3} sec')
print(f'jit backend: {proc_time_jit:.3} sec')
print(f'speed up factor of jit: {proc_time_py/proc_time_jit}')


print('Multi channel Set')
y = load_data_mc('EECG_FILT_9CH_10S_FS2400HZ.csv')
alssm = lm.AlssmPoly(3)
seg_l = lm.Segment(-50, -1, lm.FW, 1000)
seg_r = lm.Segment(0, 49, lm.BW, 1000)
cost = lm.CompositeCost([alssm], [seg_l, seg_r], F=[[1, 1]])

rls_py = lm.RLSAlssmSet(cost)
rls_py.set_backend('py')
xs_py = rls_py.filter_minimize_x(y)

rls_jit = lm.RLSAlssmSet(cost)
xs_jit = rls_jit.filter_minimize_x(y) # create just in time compilation

proc_time_py = timeit.timeit('rls_py.filter_minimize_x(y)', globals=globals(), number=10)
proc_time_jit = timeit.timeit('rls_jit.filter_minimize_x(y)', globals=globals(), number=10)

print(f'python backend: {proc_time_py:.3} sec')
print(f'jit backend: {proc_time_jit:.3} sec')
print(f'speed up factor of jit: {proc_time_py/proc_time_jit}')

