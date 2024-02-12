import matplotlib.pyplot as plt
from lmlib.utils.generator import *

K = 100
y = gen_rect(K, k_period=30, k_on=20)

fig, ax = plt.subplots(figsize=(6, 3))
ax.set(xlabel='k', ylabel='y', title='Rectangle Signal Generation')
ax.plot(range(K), y)

plt.tight_layout()
plt.show()
