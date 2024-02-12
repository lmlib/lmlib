import matplotlib.pyplot as plt
from lmlib.utils.generator import *

K = 100
ks = [5, 33, 50, 60, 77]
deltas = [-0.3, 0.6, 0.2, -0.5, -0.5]
y = gen_steps(K, ks, deltas)

fig, ax = plt.subplots(figsize=(6, 3))
ax.set(xlabel='k', ylabel='y', title='Steps Signal Generation')
ax.plot(range(K), y)

plt.tight_layout()
plt.show()
