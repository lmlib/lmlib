import matplotlib.pyplot as plt
from lmlib.utils.generator import *

K = 100
y = gen_wgn(K, sigma=0.5)

fig, ax = plt.subplots(figsize=(6, 3))
ax.set(xlabel='k', ylabel='y', title='White Gaussian Noise Signal Generation')
ax.plot(range(K), y)

plt.tight_layout()
plt.show()
