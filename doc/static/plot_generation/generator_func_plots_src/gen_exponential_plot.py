import matplotlib.pyplot as plt
from lmlib.utils.generator import *

K = 70
decay = 0.95
k = 20
y = gen_exp(K, decay, k)

fig, ax = plt.subplots(figsize=(6, 3))
ax.set(xlabel='k', ylabel='y', title='Exponential Signal Generation')
ax.plot(range(K), y)
ax.scatter(k, y[k])

plt.tight_layout()
plt.show()
