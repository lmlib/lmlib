"""
Polynomial Clustering [ex115.0]
===============================


"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import lmlib as lm
from lmlib.utils import gen_sine, gen_exp, gen_wgn

# -------------- Signal Generation -------------

# signal with 3 different pulses in random order
K = 2000
y = np.zeros(K)
pulse_amplitudes = [0.27, 0.8, 0.3]
pulse_periods = [40, 50, 70]
pulse_lengths = [80, 100, 100]
pulse_decays = [0.8, 0.9, 0.7]
pulse_map = [[200, 250, 500, 900, 1300, 1750],
             [300, 550, 1000, 1050, 1500, 1650],
             [600, 750, 1150, 1550, 1850]]
pulses = [a_ * gen_sine(l_, k_) * gen_exp(l_, g_) for a_, l_, k_, g_ in
          zip(pulse_amplitudes, pulse_lengths, pulse_periods, pulse_decays)]

for pulse, ks in zip(pulses, pulse_map):
    for k in ks:
        y[k + np.arange(len(pulse))] = pulse

# -------------- Signal Approximation -------------

a, b = -35, 35  # outer segment boarder
g = 100  # window area weight

alssm_poly = lm.AlssmPoly(poly_degree=3)
segment_left = lm.Segment(a, -1, lm.FW, g)
segment_right = lm.Segment(0, b - 1, lm.BW, g)
cost = lm.CompositeCost([alssm_poly], [segment_left, segment_right], F=[[1, 1]])
rls = lm.RLSAlssm(cost)
xs = rls.filter_minimize_x(y)

# -------------- Feature Space Transformation -------------

K, N = np.shape(xs)
x1 = np.ones(N)
zs = np.empty_like(xs)
zs_mean_centered = np.empty_like(xs)
for k in range(K):
    V = np.linalg.cholesky(rls.W[k]).T
    z1 = V@x1
    zs[k] = V@xs[k]
    zs_mean_centered[k] = zs[k]-np.inner(zs[k], z1)/np.inner(z1, z1) * z1


# -------------- FF Approximation -------------

a, b = -40, 40  # outer segment boarder
g = 100  # window area weight

alssm_poly = lm.AlssmPoly(poly_degree=3)
segment_left = lm.Segment(a, -1, lm.FW, g)
segment_right = lm.Segment(0, b - 1, lm.BW, g)
cost = lm.CompositeCost([alssm_poly], [segment_left, segment_right], F=[[1, 1]])
rls = lm.RLSAlssmSet(cost)
zzs = rls.filter_minimize_x(zs).reshape(K, N*alssm_poly.N)

# -------------- Feature Space Clustering  -------------
n_clusters = 6
kmeans = KMeans(n_clusters).fit(zs)
n_clusters = 4
kmeans2 = KMeans(n_clusters).fit(zzs)
# -------------- Plot  -------------
fig, axs = plt.subplots(3, sharex='all')
axs[0].plot(y, label='y')
sct = axs[1].scatter(range(K), y, c=kmeans.labels_, cmap='tab20', label=['cluster %d' % d for d in range(n_clusters)])
sct2 = axs[2].scatter(range(K), y, c=kmeans2.labels_, cmap='tab20', label=['cluster %d' % d for d in range(n_clusters)])
axs[0].set_title('Polynomial Clustering')
axs[-1].set_xlabel('k')
axs[0].legend(loc=1)
# produce a legend with the unique colors from the scatter
legend1 = axs[1].legend(*sct.legend_elements(), loc="lower left", title="Clusters")
axs[1].add_artist(legend1)
plt.show()
