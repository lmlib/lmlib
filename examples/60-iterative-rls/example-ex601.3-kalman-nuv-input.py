"""
Kalman Smoother with Scalar NUV Input [ex601.2]
=============================================


.. container:: twocol

    .. container:: col-fig
       
       .. image:: ./do-not-remove/../../../../../_static/gallery/example-ex601.0-kalman-nuv-input.png   
          :width: 400


    .. container:: col-text

       Implementation of Kalman Smoother with NUV Prior on input *U*.
       All given equation references refer to:
       *"H.-A. Loeliger, L. Bruderer, H. Malmberg, F. Wadehn, and N. Zalmai: On sparsity by NUV-EM, Gaussian message passing, and Kalman smoothing"*, 2016   Information Theory & Applications Workshop (ITA), La Jolla, CA, Jan. 31 - Feb. 5, 2016
       [Loeliger2016]_.
"""


import matplotlib.pyplot as plt
import numpy as np

import lmlib as lm
from lmlib.utils.generator import (gen_wgn, gen_steps)
from lmlib.irrls import *


# --------------- generate test signal ---------------
K = 800  # number of samples (length of test signal)
#y_ref = np.cumsum(gen_steps(K, [200, 250, 450, 590, 690], [1, .4, -1.1, -.4, -.5]))
y_ref = np.cumsum(gen_steps(K, [200, 250, 450, 590, 690], [.02, -.02, .03, -.04, -.03]))
y = y_ref + gen_wgn(K, sigma=0.2, seed=1000)

y_ref = gen_steps(K, [200, 250, 450, 590, 690], [1, .4, -1.1, -.4, -.5])
y = y_ref + gen_wgn(K, sigma=0.2, seed=1000)


# --------------- Main ------------------------------------------------------
#  Factor Graph with Blocks "(..)" and Variables "--X[k]-->" 
# ---------------------------------------------------------------------------
#                           (N(0,1))
#                              |
#                              v
#             sigmaU2^0.5 --> (x)                              
#                              |
#                              v
#                             BU[k]
#                              |
#                              v
#   X[k-1]--> (@A) --AX[k]--> (+) --AX_[k]--> (=) --> X[k]
#                                             |
#                                             v
#                                            (C) 
#                                             |
#                                            Y_[k]
#                                             |
#                      [N(0, sigmaZ2*I)] --> (+)
#                                             |
#                                             v 
#                                            y[k]                                    
# ---------------------------------------------------------------------------


# Configuration iterations and initial values
N = 1 # signal model order to use
I_MAX = 100  # 200 # Number of iterations
sigmaZ2 = 0.12 # initial value for Sigma Z^2
sigmaU2 = [0.001,] # initial value for Sigma U^2

DISPPLAY_U_THERSHOLD = 0.001 # threshold value to sparcify input (for visualization only)



# --- Defining Blocks & Messages ---

# Block:   AX := A@X[k-1]
A = lm.AlssmPoly( poly_degree=N-1 ).A
#block_A = MBF.Block_System_A(A) 
fX  = Messages_MV(K, A.shape[0]) # forward MV message on X
dX  = Messages_XW(K, A.shape[0])  # dual XW message on X

# Block: A_ := AX+B@N(0, sigmaU2)
B = np.concatenate( (np.zeros((N-1, 1)), np.ones((1, 1))) ) # generate input matrix of the form [0, ..., 0, 1]^T
#block_input_NUV = MBF.Block_Input_NUV(B)
fU  = Messages_MV(K, B.shape[1], sigma2=sigmaU2, m=0, border_suppression = True) # # forward MV message on U, initial value for Sigma_U2
dU  = Messages_XW(K, B.shape[1]) # dual XW message on U

X_fMVs  = Messages_MV(K, B.shape[1]) # dual XW message on U
mU  = Messages_MV(K, B.shape[1] ) # # marginals MV on U, initial value for Sigma_U2


# Block:  X[k] := A_ = C(y[k] + N(0, simgaZ2))
C = lm.AlssmPoly(poly_degree=N-1).C.reshape((1, -1)) #workaraound. C must be 2D

sigmaU2s = np.ones( (K, B.shape[1], B.shape[1]) )*sigmaU2

block_A = MBF.Block_System_A( A ) 
block_input_NUV = MBF.Block_Input_NUV(B, sigmaU2s, fU, mU)
block_output_Y = MBF.Block_Output_Y( C, y, fX, sigmaZ2 ) 
block_marginal = MBF.Block_Marginal( K, N )

section = Section( K )
section.add_block(block_A, 'block_A')
section.add_block(block_input_NUV, 'block_input_NUV')
section.add_block(block_output_Y, 'block_Output_Y')
section.add_block(block_marginal, 'block_marginal')

section.optimize(max_itr=I_MAX, msg_fwrd_x0 = fX[0], msg_bwrd_x0 = dX[0])

# ---  computing marginals for final plotting ---
mU = block_input_NUV.get_marginals_U()
mX = block_marginal.get_marginals()



## -- Plotting ---
ks = np.arange(K)
fig, axs = plt.subplots(3, 1, figsize=(8, 4), gridspec_kw={'height_ratios': [2, 1, 1]}, sharex='all')

# Remove horizontal space between axes
fig.subplots_adjust(hspace=0.1)
fig.tight_layout()

line_1, = axs[0].plot(ks, y, color="grey", linestyle="-", lw=1.0, label='$y$ (Observations)')
line_1b = axs[0].plot(ks, y_ref, color="k", linestyle="--", lw=1.00, label='$y_{ref}$ (Reference)')
line_2, = axs[0].plot(ks, (C@np.transpose(mX._m))[0], color="b", lw=1.25, label='posterior $m(X)$')
#line_2, = axs[0].plot(ks, fX._m, color="b", lw=1.25, label='temp: mean')
axs[0].legend(loc='upper right')


ind = (np.abs(mU._m)>DISPPLAY_U_THERSHOLD).flatten()
if (max(ind) > 0):
    markerline, stemlines, baseline = axs[1].stem(ks[ind], mU._m[ind], 'b', markerfmt='bo', label='$u$ (m_U)')
    plt.setp(stemlines, 'color', 'grey')
    plt.setp(markerline, markersize=5)
    plt.setp(baseline, linestyle="", color="b", linewidth=.0)
axs[1].axhline(y = 0, color = 'k', linestyle = '-', linewidth=0.5)
axs[1].set(ylabel='Input Mean')
axs[1].legend(loc='upper right')

line_4, = axs[2].plot(ks, mU._V[:,0,0], color="black", lw=0.75, label='$\sigma_U^2$ (sigmaU2)')
axs[2].set(ylabel='Input Var.')
axs[2].legend(loc='upper right')

plt.show()



