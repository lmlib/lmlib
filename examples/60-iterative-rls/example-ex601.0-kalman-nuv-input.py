"""
Kalman Smoother with Scalar NUV Input [ex601.0]
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
sigmaU2 = 0.001 # initial value for Sigma U^2

SYS_ORDER = 1 # signal model order to use


if SYS_ORDER == 1: # N=1 (first order system)
    A = np.array([[1,],])
    C = np.array([[1,],])
    B = np.array(
              [ [1, ], ]
           )
elif SYS_ORDER ==2: # N=2 (Second order system)
    A = np.array([[1, 1],
                  [0, 1]])
    C = np.array([[1, 0]])
    B = np.array(
          [[0, ],
          [1, ]]
        )
else:
      print("Select valid system order")
      exit() 



N = A.shape[0]

AT = np.transpose(A)
CT = np.transpose(C)
BT = np.transpose(B)

SZ_B = B.shape
SZ_C = C.shape

mF_X = np.zeros([K, N])     # Forward mean at X
VF_X = np.zeros([K, N, N] ) # Forward variance at X

mF_AX = np.zeros([K, N])
VF_AX = np.zeros([K, N, N] )

mF_AX_ = np.zeros([K, N])
VF_AX_ = np.zeros([K, N, N] )

mF_U = np.zeros([K, 1])
VF_U = np.zeros([K, 1, 1] )

mB_Y_ = np.zeros([K, SZ_C[0]])
VB_Y_ = np.zeros([K, SZ_C[0], SZ_C[0]] )


XI_AX_ = np.zeros([K, N])
W_AX_ = np.zeros([K, N, N] )

XI_X = np.zeros([K, N])
W_X = np.zeros([K, N, N] )

XI_U = np.zeros([K, SZ_B[1]])
W_U = np.zeros([K, SZ_B[1], SZ_B[1]] )

m_X = np.zeros([K, N])  # Posterior mean at X
V_X = np.zeros([K, N, N])  # Posterior variance at X

m_U = np.zeros([K, SZ_B[1]])  # Posterior variance at U
V_U = np.zeros([K, SZ_B[1], SZ_B[1]])  # Posterior variance at X



VF_U[1:-1] = sigmaU2  # first and last index is not used in simulation; left at zero for nice plot 
mF_U[:] = 0




# Interrative Optimization
for ii in range(0,I_MAX):

    # 1) Forward Kalman
    for k in range(1,K):
       # "A" node
       mF_AX[k] = A@mF_X[k-1]     # III.1
       VF_AX[k] = A@VF_X[k-1]@AT   # III.2

       # "AX+B@U=AX_" Node
       mF_AX_[k] = mF_AX[k] + B@mF_U[k] # II.1, III.1
       VF_AX_[k] = VF_AX[k] + B@VF_U[k]@BT  # II.2, III.2

       # "Y_+Z=Y" Node
       mB_Y_[k] = y[k]
       VB_Y_[k] = sigmaZ2


       # "C=" Node
       G = np.linalg.inv(VB_Y_[k] + C@VF_AX_[k]@CT)  #  V.3
       mF_X[k] = mF_AX_[k]+VF_AX_[k]@CT@G@(mB_Y_[k] - C@mF_AX_[k]) #  V.1
       VF_X[k] = VF_AX_[k]-VF_AX_[k]@CT@G@C@VF_AX_[k] #  V.2


    # 2) Backward
    for k in range(K-1,0,-1):

        W_Y_ = np.array([[1/sigmaZ2,],])
        mB_y_ = y[k]

        # ---
        F = np.eye(SZ_C[1])-VF_X[k]@CT@W_Y_@C #  V.8
        FT = F.transpose()

        # "AX_ = X = Y"
        # --- Option 1
        XI_AX_[k] = FT@XI_X[k]+CT@W_Y_@(C@mF_X[k]-mB_y_)  #  V.4
        W_AX_[k] = FT@W_X[k]@F+CT@W_Y_@C@F  #  V.6
        # --- Option 2 (reusing G)
        # G = np.linalg.inv(VB_Y_[k] + C @ VF_AX_[k] @ CT)  # V.3
        # XI_AX_[k] = FT@XI_X[k]+CT@G@(C@mF_AX_[k]-mB_y_)  #  V.5
        # W_AX_[k] = FT@W_X[k]@F+CT@G@C  #  V.7
        # ---

        # "AX + BU = AX_"
        XI_X[k-1] = AT@XI_AX_[k] #  II.6, III.7
        W_X[k-1] = AT@W_AX_[k]@A #  II.7, III.8

        # 3) Input U_k estimate (not needed for Kalman smoothing with Gaussian inputs)
        # XI_U[k] = BT @ XI_X[k]   # II.6, III.7  --> wrong xi and W_tilde, we need the ones at the plus node
        # W_U[k] = BT @ W_X[k] @ B # II.7, III.8
        XI_U[k] = BT @ XI_AX_[k]   # II.6, III.7
        W_U[k] = BT @ W_AX_[k] @ B # II.7, III.8


    # 4) Posterior Mean (marginals)
    for k in range(K):
        # State X estimate
        m_X[k] = mF_X[k] - VF_X[k]@XI_X[k]     # IV.9
        V_X[k] = VF_X[k] - VF_X[k] @ W_X[k] @ VF_X[k]   # IV.13

        # Input U estimate
        m_U[k] =  mF_U[k] - VF_U[k]@XI_U[k]  # IV.9
        V_U[k] =  VF_U[k] - VF_U[k]@W_U[k]@VF_U[k] # IV.13


        # 1)  vanilla NUV prior using EM
        # VF_U[k] = V_U[k] + m_U[k]**2   # Update sigmaU_k according to Eq. (13)
        
        # 2)  vanilla NUV prior using AM
        VF_U[k] = m_U[k]**2   # Update sigmaU_k according to Eq. (13)

        # 3)  box prior to constraint m_U[k] \in [a, b]
        # a = -0.007  # lower bound
        # b = 0.004  # upper bound 
        # gamma = 100  # design param, make this smaller and observe how the constraint will fail
        # eps = 1e-6  # to avoid numerical issues
        # VF_U[k] = 1 / (gamma * ( 1/(eps + np.abs(m_U[k] - a)) + 1/(eps + np.abs(m_U[k] - b)) ) )   # According to Table 7.1, diss keusch
        # mF_U[k] = gamma * VF_U[k] * ( a/(eps + np.abs(m_U[k] - a)) + b/(eps + np.abs(m_U[k] - b)))



## -- Plotting ---
ks = np.arange(K)
fig, axs = plt.subplots(3, 1, figsize=(8, 4), gridspec_kw={'height_ratios': [2, 1, 1]}, sharex='all')

# Remove horizontal space between axes
fig.subplots_adjust(hspace=0.1)
fig.tight_layout()

line_1, = axs[0].plot(ks, y, color="grey", linestyle="-", lw=1.0, label='$y$ (Observations)')
line_1b = axs[0].plot(ks, y_ref, color="k", linestyle="--", lw=1.00, label='$y_{ref}$ (Reference)')
line_2, = axs[0].plot(ks, (C@np.transpose(m_X))[0], color="b", lw=1.25, label='posterior $m(X)$')
axs[0].legend(loc='upper right')


ind = (np.abs(m_U)>0.01).flatten()
markerline, stemlines, baseline = axs[1].stem(ks[ind], m_U[ind], 'b', markerfmt='bo', label='$u$ (m_U)')
plt.setp(stemlines, 'color', 'grey')
plt.setp(markerline, markersize=5)
plt.setp(baseline, linestyle="", color="b", linewidth=.0)
axs[1].axhline(y = 0, color = 'k', linestyle = '-', linewidth=0.5)
axs[1].set(ylabel='Input Mean')
axs[1].legend(loc='upper right')

line_4, = axs[2].plot(ks, VF_U[:,0,0], color="black", lw=0.75, label='$\sigma_U^2$ (sigmaU2)')
axs[2].set(ylabel='Input Var.')
axs[2].legend(loc='upper right')

plt.show()

