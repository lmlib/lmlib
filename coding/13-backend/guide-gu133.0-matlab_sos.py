"""
Multi-Channel Spike Detection [ex112.0]
=======================================

This example shows a spike detection algorithm that uses autonomous linear state space models together with
exponentially decaying windows. Given is a multi-channel signal containing multiple spikes
(sinusoidal cycle with decaying amplitude) with additive white Gaussian noise and a baseline.

"""

import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from scipy.linalg import block_diag
from scipy.signal import find_peaks

from lmlib.utils.generator import gen_conv, gen_sine, gen_exp, gen_pulse, gen_wgn, k_period_to_omega

# signal generation
K = 550
L = 1  # number of channels
spike_length = 20
spike_decay = 0.88
spike_locations = [100, 240, 370]
spike = gen_sine(spike_length, spike_length) * gen_exp(spike_length, spike_decay)
y_sp = gen_conv(gen_pulse(K, spike_locations), spike)
y = np.column_stack([0.8 * y_sp + gen_wgn(K, sigma=0.2, seed=10000 - l) for l in range(L)]).reshape(K, L)
y = y.flatten()

# Model
alssm_sp = lm.AlssmSin(k_period_to_omega(spike_length), spike_decay)
alssm_bl = lm.AlssmPoly(poly_degree=3)

# Segments
g_bl = 500
g_sp = 5000
len_sp = spike_length
len_bl = int(1.5 * spike_length)
segment_middle = lm.Segment(a=0, b=len_sp, direction=lm.BACKWARD, g=g_sp)

# Cost
F = [[1],
     [1]]
cost = lm.CompositeCost((alssm_sp, alssm_bl), (segment_middle,), F)

rls_np = lm.RLSAlssm(cost,backend='numpy')
rls_np.filter(y)

rls_parallel = lm.RLSAlssm(cost,backend='lfilter')
rls_parallel.filter(y)



numdenom_matlab_dim0_seg0_alssm0 = {
    'key':     'e81eee05',
    'sos_iir': [[1, 0, 0, 1, -1.6735246967857345, 0.77409027097600003]],
    'num_a': [
        [0, 1, -0.83676234839286723],
        [0, 0, -0.27188056805894373]
    ],
    'num_b': [
        [0, 0.064914608274257768, -0.060052614467935368],
        [0, -0.021092034801549836, -7.9479467472501895e-17]
    ]
};
numdenom_matlab_dim0_seg0_alssm1 = {
    'key':     '7a9ee815',
    'sos_iir': [[1, 0, 0, 1, -1.9996, 0.99960004000000002], [1, 0, 0, 1, -1.9996, 0.99960004000000002]],
    'num_a': [
        [0, 1, -2.9994000000000001, 2.9988001200000003, -0.99940011999200007],
        [0, 0, 0.99980000000000002, -1.99920008, 0.99940011999199951],
        [0, 0, 0.99980000000000002, 3.3300029400606946e-16, -0.99940011999199951],
        [0, 0, 0.99980000000000002, 3.9984001599999974, 0.99940011999200096]
    ],
    'num_b': [
        [0, 1.0000000000003879, -2.9994000000007595, 2.9988001200004137, -0.99940011999203648],
        [0, 21, -61.987600000000441, 60.975602440001161, -19.988002399840671],
        [0, 441, -1279.7440000000006, 1238.5044495600007, -399.76004799679941],
        [0, 9261, -26390.720799999996, 25130.94460563999, -7995.2009599359935]
    ]
};

nd = lm.sos_matlab.sos_from_matlab_multi(cost, [
    numdenom_matlab_dim0_seg0_alssm0,   # alssm_sp (index 0)
    numdenom_matlab_dim0_seg0_alssm1,   # alssm_bl (index 1)
])
rls_parallel_matlab = lm.RLSAlssm(cost, backend='lfilter', filter_form='parallel',
                                   numdenom=nd, supress_pzinstruction=True)
rls_parallel_matlab.filter(y)


# Errors (time-domain)
print("\n=== Errors ===")
for n in range(rls_np.xi.shape[1]):
    tests = [
        (rls_parallel.xi, 'Parallel Python'),
    ]
    if rls_parallel_matlab is not None:
        tests.append((rls_parallel_matlab.xi, 'Parallel Matlab'))

    for xi_test, label in tests:
        rChan = np.arange(0,n+1)
        error = rls_np.xi[:,rChan] - xi_test[:,rChan]   
        error_normalized = np.zeros_like(error)
        for n_ in rChan:
            sn = np.sqrt(np.mean(rls_np.xi[:,n_]**2))
            error_normalized[:,n_] = error[:,n_] / sn
        max_abs_error = np.max(np.abs(error_normalized))
        vnrmse = np.sqrt(np.mean(error_normalized**2))
        
        printerrors=True
        if printerrors:
            print(f"{label}, n={n}: Max absolute error: {max_abs_error:.2e}, VNRMSE: {vnrmse:.2e}")


# Plot
fig, axs = plt.subplots(3, 1, figsize=(9, 8),  sharex='all')

axs[0].plot(range(K), y, c='gray', lw=1.0)
axs[0].legend(loc=1)

# Ref Signals
axs[1].plot(range(K), rls_np.xi - rls_parallel.xi, lw=1.0,label='parallel np')
axs[1].legend(loc=1)

# LCR
axs[2].plot(range(K),  rls_np.xi - rls_parallel_matlab.xi, lw=1.0,label='parallel matlab')
axs[2].legend(loc=1)


plt.show()
