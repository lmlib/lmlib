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
y_ref = gen_steps(K, [200, 250, 450, 590, 690], [1, .4, -1.1, -.4, -.5])
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
I_MAX = 200 # Number of iterations
sigmaZ2 = 0.12 # initial value for Sigma Z^2
sigmaU2 = [0.001,] # initial value for Sigma U^2

N = 1 # signal model order to use


A = lm.AlssmPoly(poly_degree=N-1).A
C = lm.AlssmPoly(poly_degree=N-1).C.reshape( (1,-1) ) #workaraound. C should be 2D
B = np.zeros( (N, 1) ) # generate input matrix
B[-1,0] = 1

SZ_B = B.shape
SZ_C = C.shape


# Messages 
fMV_X   = Messages_MV(K, N)

fMV_U   = Messages_MV(K, N, sigma2=sigmaU2, m=0) # initial value for Sigma_U2
fMV_U._V[-1] = 0 # first and last index is not used in simulation; left at zero for nice plot 
fMV_U._V[0] = 0 # first and last index is not used in simulation; left at zero for nice plot 

dXW_X  = Messages_XW(K, N)

dXW_U  = Messages_XW(K, SZ_B[1])
dXW_Y_ = Messages_XW(K, SZ_B[1])

# Marginals
MV_X    = Messages_MV(K, N)
MV_U    = Messages_MV(K, SZ_B[1])


# Boxes
node_A = Node_Mul_A()
node_input_NUV = Node_Input_NUV()
node_output_Y = Node_Output_Y()


# --- Interrative Optimization
for ii in range(0,I_MAX):

    for k in range(1,K):
        # 1) Forward Kalman
        AX_fMVk  = node_A.get_Y_forwardMV( fMV_X[k-1], A)
        fMV_AX_k = node_input_NUV.get_Z_forwardMV( AX_fMVk, B, fMV_U._V[k], m=[0,])
        fMV_X[k]   = node_output_Y.get_Z_forwardMV( fMV_AX_k, y[k], C, sigmaZ2)

    for k in range(K-1,0,-1):
        # 2) Backward
        dXW_AX_k = node_output_Y.get_X_dualXW( dXW_X[k], fMV_X[k], y[k], C, sigmaZ2)
        # dXW_AX_k = node_output_Y.bw_get_in1_dXW_opt2( dXW_X[k], fMV_X[k], y[k], C, sigmaZ2) # Alternative implementation
        dXW_AXk = node_input_NUV.get_X_dualXW( dXW_AX_k ) # dummy, nothing to do ...            
        dXW_X[k-1] = node_A.get_X_dualXW( dXW_AXk , A)            
        
        # for input estiamte 
        dXW_U[k] = node_input_NUV.get_U_dualXW( dXW_AX_k, B)
        
        # 4) Posterior Mean (marginals)
        # State X and U estimate
        MV_X[k] = Node.marginal_MV( fMV_X[k], dXW_X[k] )
        MV_U[k] = Node.marginal_MV( fMV_U[k], dXW_U[k] )
 
        fMV_U._V[k] = node_input_NUV.update_variance_U_EM( MV_U._m[k] )



## -- Plotting ---
ks = np.arange(K)
fig, axs = plt.subplots(3, 1, figsize=(8, 4), gridspec_kw={'height_ratios': [2, 1, 1]}, sharex='all')

# Remove horizontal space between axes
fig.subplots_adjust(hspace=0.1)
fig.tight_layout()

line_1, = axs[0].plot(ks, y, color="grey", linestyle="-", lw=1.0, label='$y$ (Observations)')
line_1b = axs[0].plot(ks, y_ref, color="k", linestyle="--", lw=1.00, label='$y_{ref}$ (Reference)')
line_2, = axs[0].plot(ks, (C@np.transpose(MV_X._m))[0], color="b", lw=1.25, label='posterior $m(X)$')
#line_2, = axs[0].plot(ks, fMV_X._m, color="b", lw=1.25, label='temp: mean')
axs[0].legend(loc='upper right')


ind = (np.abs(MV_U._m)>0.01).flatten()
markerline, stemlines, baseline = axs[1].stem(ks[ind], MV_U._m[ind], 'b', markerfmt='bo', label='$u$ (m_U)')
plt.setp(stemlines, 'color', 'grey')
plt.setp(markerline, markersize=5)
plt.setp(baseline, linestyle="", color="b", linewidth=.0)
axs[1].axhline(y = 0, color = 'k', linestyle = '-', linewidth=0.5)
axs[1].set(ylabel='Input Mean')
axs[1].legend(loc='upper right')

line_4, = axs[2].plot(ks, MV_U._V[:,0,0], color="black", lw=0.75, label='$\sigma_U^2$ (sigmaU2)')
axs[2].set(ylabel='Input Var.')
axs[2].legend(loc='upper right')

plt.show()



