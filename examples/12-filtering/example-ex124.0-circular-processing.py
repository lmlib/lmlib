"""
Circular Processing for Online Applications [ex124.0]
=====================================================

TODO

"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import lmlib as lm
from lmlib.utils.generator import gen_rect, gen_wgn

# using JIT for fast processing
lm.set_backend('py')


# --- Generating test signal ---
K = 2000
Xp, Yp = 8, 8
S = Xp*Yp
S_mark = S//2
y = np.column_stack([gen_rect(K, 250, 125, k0=s) for s in range(S)])
# y += gen_wgn((K, S), sigma=0.001)

# buffer length for circular processing
buffer_len = 101

# setup simple model
segment_left = lm.Segment(a=-np.inf, b=-0, direction='fw', g=5)
alssm = lm.AlssmPoly(1)
cost = lm.CostSegment(alssm, segment_left)

# setup RLSAlssmSet with circular buffer
rls = lm.RLSAlssmSet(cost, circular=True)
rls.setup_buffer(y[:buffer_len].shape)

y_hat = np.zeros((K, S))
for k_buffer in range(0, K, buffer_len):
    buffer_range = range(k_buffer, k_buffer+buffer_len)
    print("Processing Range: ", buffer_range)

    # processing new signal part into fixed buffer
    rls.filter(y[buffer_range])
    xs = rls.minimize_x()
    y_hat[buffer_range] = cost.eval_alssm_output(xs)


# test signal estimation
rls_test = lm.RLSAlssm(cost)
y_test_hat = rls_test.filter_minimize_yhat(y[:, S_mark])





frames = [] # for storing the generated images
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax3.plot(range(K), y[:, S_mark], c='k', lw=0.8, label=r'$y$')
ax3.plot(range(K), y_hat[:, S_mark], c='b', lw=1, label=r'$\hat{y}$')
ax3.plot(range(K), y_test_hat, c='g', lw=1, label=r'$\tilde{y}$')
ax3.set_xlabel('k')
ax3.legend()

for k in range(K):
    img_ax1 = y[k].reshape((Xp, Yp))
    img_ax2 = y_hat[k].reshape((Xp, Yp))

    frames.append([ax1.imshow(img_ax1, cmap=plt.get_cmap('Greys_r'), animated=True),
                   ax2.imshow(img_ax2, cmap=plt.get_cmap('Blues_r'), animated=True)])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=1000)
# ani.save('movie.mp4')
plt.show()