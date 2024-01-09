import matplotlib.pyplot as plt
from lmlib.utils.generator import *

K = 65
y = gen_pulse(K, ks=[10, 17, 59, 33])

fig, ax = plt.subplots(figsize=(6, 3))
ax.set(xlabel='k', ylabel='y', title='Unit Impulse Signal Generation')
ax.stem(range(K), y)

plt.tight_layout()
plt.show()
