"""
Circular Processing for Online Applications [ex124.0]
=====================================================

TODO

Change internal structure?
has to be infinite, b=0,  and forward (IIR)
init_parameters where fixed to zero


Create simple forward recursion for a=inf and b=0, can be done samplewise nearly no storage

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
y = np.column_stack([gen_rect(K, 200, 100, k0=s) for s in range(S)])
# y += gen_wgn((K, S), sigma=0.001)

# buffer length for circular processing
buffer_len = 50

# setup simple model
segment_left = lm.Segment(a=-np.inf, b=-0, direction='fw', g=10)
alssm = lm.AlssmPoly(1)
cost = lm.CostSegment(alssm, segment_left)

# setup RLSAlssmSet with circular buffer
rls = lm.RLSAlssmSet(cost, circular=True)
# rls.setup_buffer(y[:buffer_len].shape)

y_hat = np.zeros((K, S))
for k_buffer in range(0, K, buffer_len):
    buffer_range = range(k_buffer, min(k_buffer+buffer_len, K))
    print("Processing Range: ", buffer_range)

    # processing new signal part into fixed buffer
    rls.filter(y[buffer_range])
    xs = rls.minimize_x()

    rls._W[1:] = 0.0
    rls._xi[1:] = 0.0
    rls._kappa[1:] = 0.0
    rls._nu[1:] = 0.0

    y_hat[buffer_range] = cost.eval_alssm_output(xs[:len(buffer_range)])


# test signal estimation
rls_test = lm.RLSAlssmSet(cost)
y_test_hat = rls_test.filter_minimize_yhat(y)


frames = [] # for storing the generated images
fig = plt.figure()
gs = fig.add_gridspec(2,3)
ax11 = fig.add_subplot(gs[0, 0])
ax12 = fig.add_subplot(gs[0, 1])
ax13 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, :])


ax3.plot(range(K), y[:, S_mark], c='k', lw=0.8, label=r'$y$ raw')
ax3.plot(range(K), y_hat[:, S_mark], c='b', lw=1, label=r'$\hat{y}$ circular')
ax3.plot(range(K), y_test_hat[:, S_mark], 'g-s', lw=0.5, markersize=1, label=r'$\tilde{y}$ normal')
ax3.set_xlabel('k')
ax3.legend()
ax11.set_title('Raw Video')
ax12.set_title('Circular Processed Video')
ax13.set_title('Normal Processed Video')
for k in range(K):
    img_ax1 = y[k].reshape((Xp, Yp))
    img_ax2 = y_hat[k].reshape((Xp, Yp))
    img_ax3 = y_test_hat[k].reshape((Xp, Yp))

    frames.append([ax11.imshow(img_ax1, cmap=plt.get_cmap('Greys_r'), animated=True),
                   ax12.imshow(img_ax2, cmap=plt.get_cmap('Blues_r'), animated=True),
                   ax13.imshow(img_ax3, cmap=plt.get_cmap('Greens_r'), animated=True)])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=1000)
# ani.save('movie.mp4')
plt.show()
