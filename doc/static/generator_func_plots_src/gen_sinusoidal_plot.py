import matplotlib.pyplot as plt
from lmlib.utils.generator import *

K = 100
y = gen_sine(K, k_periods=36, k0s=0)

fig, ax = plt.subplots(figsize=(6, 3))
ax.set(xlabel='k', ylabel='y', title='Sinusoidal Signal Generation')
ax.plot(range(K), y)

plt.tight_layout()
plt.show()
