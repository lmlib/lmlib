"""
Benchmarking of Stace-Space vs. Transfer-Function Backend [ex131.0]
===================================================================

This example demonstrates the usage of transfer-function (tf) backend in RLSAlssm* classes. This is done for one single channel.
"""
import timeit
import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm

lm.WARNING_NOT_STEADY_STATE = False

n_exe = 2  # number of filter executions

# setting all recursions to JIT backend default
backends = ['lfilter', 'numpy']
color_codes = 'k', 'b', 'g'
mspsecs_dict = {k: [] for k in backends}

K = 1_000_000
y = np.random.randn(K)

# setup model
alssm = lm.AlssmPoly(poly_degree=1)
seg_l = lm.Segment(a=-21, b=-1, direction=lm.FW, g=100)
cost = lm.CompositeCost([alssm], [seg_l], F=[[1]])

# Single channel
for backend in backends:
    rls = lm.RLSAlssm(cost, backend=backend, steady_state=False, calc_W=False, calc_kappa=False, calc_nu=False)
    rls.filter(y)
    proc_time = timeit.timeit('rls.filter(y)', globals=globals(), number=n_exe)
    mspsec = K * n_exe * 1e-6 / proc_time
    mspsecs_dict[backend].append(mspsec)


labels = ('RLSAlssm.filter(y, steady_state=False) \n 1 Channel',)
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
