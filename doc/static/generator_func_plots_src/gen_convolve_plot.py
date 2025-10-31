import matplotlib.pyplot as plt
from lmlib.utils.generator import *

K = 200
y_impulse = gen_rand_pulse(K, n_pulses=4)
y_template = gen_sine(K=10, k_periods=10)
y = gen_conv(y_impulse, y_template)

fig, ax = plt.subplots(figsize=(6, 3))
ax.set(xlabel='k', ylabel='y', title='Convolve Random Unit Impulse Signal with Sinusoidal')
ax.plot(range(K), y)

plt.tight_layout()
plt.show()
