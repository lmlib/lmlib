import matplotlib.pyplot as plt
from lmlib.utils.generator import *

K = 150
y = gen_rand_pulse(K, n_pulses=5, length=10)

fig, ax = plt.subplots(figsize=(6, 3))
ax.set(xlabel='k', ylabel='y', title='Random Pulse Signal Generation')
ax.plot(range(K), y)

plt.tight_layout()
plt.show()
