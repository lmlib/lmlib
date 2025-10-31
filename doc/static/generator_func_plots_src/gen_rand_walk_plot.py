import matplotlib.pyplot as plt
from lmlib.utils.generator import *

K = 100
y = gen_rand_walk(K)

fig, ax = plt.subplots(figsize=(6, 3))
ax.set(xlabel='k', ylabel='y', title='Random Walk Signal Generation')
ax.plot(range(K), y)

plt.tight_layout()
plt.show()
