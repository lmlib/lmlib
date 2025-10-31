import matplotlib.pyplot as plt
from lmlib.utils.generator import *

K = 100
ks = [15, 33, 50, 60, 77]
deltas = [5, -2.5, -1, -3, 2]
y = gen_slopes(K, ks, deltas)

fig, ax = plt.subplots(figsize=(6, 3))
ax.set(xlabel='k', ylabel='y', title='Slopes Signal Generation')
ax.plot(range(K), y)

plt.tight_layout()
plt.show()
