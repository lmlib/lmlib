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


# --------------- Main ------------------------------------------------------
#  Factor Graph with Nodes "(..)" and Variables "--X[k]-->" 
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
#   X[k-1]--> (@A) --AX[k]--> (+) --X_[k]--> (=) --> X[k]
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
N = 2 # signal model order to use
I_MAX = 200 # Number of iterations
sigmaZ2 = 0.12 # initial value for Sigma Z^2
sigmaU2 = [0.001,] # initial value for Sigma U^2

DISPPLAY_U_THERSHOLD = 0.001 # threshold value to sparcify input (for visualization only)



# --- Defining Nodes & Messages ---

# Node:   AX := A@X[k-1]
A = lm.AlssmPoly( poly_degree=N-1 ).A
node_A = Node_Mul_A(A) 
fX   = Messages_MV(K, node_A.dim_X) # forward MV message on X
dX  = Messages_XW(K, node_A.dim_X)  # dual XW message on X

# Node: A_ := AX+B@N(0, sigmaU2)
B = np.concatenate( (np.zeros((N-1, 1)), np.ones((1, 1))) ) # generate input matrix of the form [0, ..., 0, 1]^T
node_input_NUV = Node_Input_NUV(B)
fU   = Messages_MV(K, node_input_NUV.dim_U, sigma2=sigmaU2, m=0, border_suppression = True) # # forward MV message on U, initial value for Sigma_U2
dU  = Messages_XW(K, node_input_NUV.dim_U) # dual XW message on U

# Node:  X[k] := A_ = C(y[k] + N(0, simgaZ2))
C = lm.AlssmPoly(poly_degree=N-1).C.reshape((1, -1)) #workaraound. C must be 2D
node_output_Y = Node_Output_Y(C) 




# --- Interrative Optimization
for ii in range(0,I_MAX):
    for k in range(1,K):
        # 1) Forward Kalman
        fAX_k  = node_A.get_Y_forwardMV( fX[k-1])
        fAX__k = node_input_NUV.get_Z_forwardMV( fAX_k, fU._V[k], m=[0,])
        fX[k]   = node_output_Y.get_Z_forwardMV( fAX__k, y[k], sigmaZ2)

    for k in range(K-1,0,-1):
        # 2) Backward
        dAX__k = node_output_Y.get_X_dualXW( dX[k], fX[k], y[k], sigmaZ2)
        dAX_k = node_input_NUV.get_X_dualXW( dAX__k ) # ... is actually a dummy call ...           
        dX[k-1] = node_A.get_X_dualXW( dAX_k)            
        
        # 4) Posterior Mean (marginals), State U estimate
        dU[k] = node_input_NUV.get_U_dualXW( dAX_k)
        mU_k = Node.marginal_MV( fU[k], dU[k] )
        fU._V[k] = node_input_NUV.update_variance_U_EM( mU_k[0] )




# ---  computing marginals for final plotting ---
mX    = Messages_MV(K, node_A.dim_X) # marginal on X
mU    = Messages_MV(K, node_input_NUV.dim_U) # marginal on U
for k in range(K-1,0,-1): 
    mX[k] = Node.marginal_MV( fX[k], dX[k] )
    mU[k] = Node.marginal_MV( fU[k], dU[k] )




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



