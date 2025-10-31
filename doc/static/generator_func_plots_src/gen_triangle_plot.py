import matplotlib.pyplot as plt
from lmlib.utils.generator import *

K = 100
y = gen_tri(K, k_period=30)

fig, ax = plt.subplots(figsize=(6, 3))
ax.set(xlabel='k', ylabel='y', title='Triangle Signal Generation')
ax.plot(range(K), y)

plt.tight_layout()
plt.show()
