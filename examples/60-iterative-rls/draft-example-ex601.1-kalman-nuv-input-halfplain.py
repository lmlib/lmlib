"""
Kalman Smoother with Scalar NUV Input [ex601.0]
=============================================


.. container:: twocol

    .. container:: col-fig
       
       .. image:: ./do-not-remove/../../../../../_static/gallery/example-ex501.0-kalman-nuv-input.png   
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
y = y_ref + gen_wgn(K, sigma=0.02, seed=1000)


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
I_MAX = 600 # Number of iterations
sigmaZ2 = 0.12 # initial value for Sigma Z^2
sigmaU2 = [0.001,] # initial value for Sigma U^2

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


# +
class MSG_MV():
    def __init__(self, K, N, sigma2=0, m=0):
        self._m = np.zeros([K, N])
        self._V = np.zeros([K, N, N])
        
        # sigma2: if scalar, inital value for diagonal of V
        if sigma2 != 0:
            self._V[:,:,:] = np.eye(N)*sigma2

    def __setitem__(self, key, value): 
        (self._m[key], self._V[key]) = value

    def __getitem__(self, key): 
        return (self._m[key], self._V[key]) 
        
        
class MSG_XW():
    def __init__(self, K, N):        
        self._xi = np.zeros([K, N])
        self._W  = np.zeros([K, N, N])
  
    def __setitem__(self, key, value): 
        (self._xi[key], self._W[key]) = value

    def __getitem__(self, key): 
        return (self._xi[key], self._W[key]) 
    





class Box:
# Box (Base Class)    
    
    def __init__(self):
        pass

    @staticmethod
    def marginal_MV( fMV, dXW ):
        # m_X[k] = mF_X[k] - VF_X[k]@XI_X[k]     # IV.9
        m_X = fMV[0] - fMV[1]@dXW[0]     # IV.9
        # V_X[k] = VF_X[k] - VF_X[k] @ W_X[k] @ VF_X[k]   # IV.13        
        V_X = fMV[1] - fMV[1] @ dXW[1] @ fMV[1]   # IV.13        
        return (m_X, V_X)
    



class Box_Lin_Transition(Box):

# linear transition block as e.g. block A in Fig. 2

    #
    # in1 --(@A)--> in2
    #

    def __init__(self):
        pass
        
    @staticmethod
    def fw_get_in2_fMV(in1_fMV, A):
       # in1 = (m, V)
       # (mF, VF) = mV
       # "A" node
        
       in2_fM = A@in1_fMV[0]     # III.1
       in2_fV = A@in1_fMV[1]@AT   # III.2         
       return (in2_fM, in2_fV)
    #

    @staticmethod
    def bw_get_in1_dXW(in2_dXW, A):
        # "A*X = AX", but also "AX + BU = AX_"
        # in1: AX[k]
        # XI_X[k-1] = AT@XI_AX_[k] #  II.6, III.7
        in1_dX = AT@in2_dXW[0] #  II.6, III.7
        # W_X[k-1] = AT@W_AX_[k]@A #  II.7, III.8
        in1_dW = AT@in2_dXW[1]@A #  II.7, III.8
        return (in1_dX, in1_dW)



class Box_NUV_Input(Box):
    
# NUV Input as in Fig. 3
    
#                           (N(0,1))
#                              |
#                              v
#             sigmaU2^0.5 --> (x)          
#                              |
#                              v
#                             BU[k]
#                              |
#                              v
#                     in1 --> (+) -->  in2

    def __init__(self):
        pass
    
    @staticmethod
    def fw_get_in2_fMV(in1_fMV, B, sigmaU2, m=[0,]):
        # (mF_A, VF_A) = mV_A
        # "AX+B@U=AX_" Node    
        mF_A_ = in1_fMV[0] + B@m # II.1, III.1
        VF_A_ = in1_fMV[1] + B@sigmaU2@BT  # II.2, III.2     
        return (mF_A_, VF_A_)

    @staticmethod
    def bw_get_in1_dXW(in2_dXW):
        # XiW3: AX_[k]  \tilde xi, \thilde W
        # II.6, II.7  # nothing to do (dummy)
        return (in2_dXW)

    @staticmethod
    def bw_get_U_dXW(in2_dXW , B):
        # 3) Input U_k estimate (not needed for Kalman smoothing with Gaussian inputs)
        # XI_U[k] = BT @ XI_AX_[k]   # II.6, III.7
        Xi2 = B.transpose()@in2_dXW[0]   # II.6, III.7
        # W_U[k] = BT @ W_AX_[k] @ B # II.7, III.8
        W2 = B.transpose()@in2_dXW[1]@ B # II.7, III.8
        return (Xi2, W2)
        
    @staticmethod
    def update_variance_U_EM( M_U ):
        """ Updates variance of U using EM method according to Eq. (13) """
        return(M_U**2)   # Update sigmaU_k according to Eq. (13)
    



class Box_Scalar_Output(Box):
    
# Scalar output Block as in Fig. 2 (lower branch)
    
#                                     in1 ---(=) --> in2
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
    
    
    def __init__(self):
        pass
        
    @staticmethod
    def  fw_get_in2_fMV(in1_fMV, y, C, sigmaZ2):
        # "Y_+Z=Y" Node
        #mB_Y_[k] = y
        #VB_Y_[k] = sigmaZ2

        # "C=" Node
        G = np.linalg.inv(sigmaZ2 + C@in1_fMV[1]@CT)  #  V.3
        mF_ = in1_fMV[0]+in1_fMV[1]@CT@G@(y - C@in1_fMV[0]) #  V.1
        VF_ = in1_fMV[1]-in1_fMV[1]@CT@G@C@in1_fMV[1] #  V.2    
        return (mF_, VF_)
    #
        

    @staticmethod
    def  bw_get_in1_dXW(in2_dXW, in2_fMV, y, C, sigmaZ2):
       # in1 : AX_[k]
       # in2 : X[k]
       W_Y_ = np.array([[1/sigmaZ2,],])
   
       # ---
       #F = np.eye(SZ_C[1])-VF_X[k]@CT@W_Y_@C #  V.8
       F = np.eye(SZ_C[1])-in2_fMV[1]@CT@W_Y_@C #  V.8
       FT = F.transpose()
       
       # "AX_ = X = Y"
       # --- Option 1
       # XI_AX_[k] = FT@XI_X[k]+CT@W_Y_@(C@mF_X[k]-mB_y_)  #  V.4
       in1_dX = FT@in2_dXW[0]+CT@W_Y_@(C@in2_fMV[0]-y)  #  V.4
       # W_AX_[k] = FT@W_X[k]@F+CT@W_Y_@C@F  #  V.6
       in1_dW = FT@in2_dXW[1]@F+CT@W_Y_@C@F  #  V.6
        
       return (in1_dX, in1_dW)
        

       
    @staticmethod
    def  bw_get_in1_dXW_opt2(in2_dXW, in2_fMV, y, C, sigmaZ2):
        W_Y_ = np.array([[1/sigmaZ2,],])
        
        #F = np.eye(SZ_C[1])-VF_X[k]@CT@W_Y_@C #  V.8
        F = np.eye(SZ_C[1])-in2_fMV[1]@CT@W_Y_@C #  V.8
        FT = F.transpose()
        
        # --- Option 2 (reusing G)
        # G = np.linalg.inv(VB_Y_[k] + C @ VF_AX_[k] @ CT)  # V.3
        G = np.linalg.inv(sigmaZ2 + C @ in2_fMV[1] @ CT)  # V.3
        # XI_AX_[k] = FT@XI_X[k]+CT@G@(C@mF_AX_[k]-mB_y_)  #  V.5
        in1_dX = FT@in2_dXW[0]+CT@G@(C@in2_fMV[0]-y)  #  V.5
        # W_AX_[k] = FT@W_X[k]@F+CT@G@C  #  V.7
        in1_dW = FT@in2_dXW[1]@F+CT@G@C  #  V.7
        # ---
        return (in1_dX, in1_dW)
        





# Messages 
fMV_X   = MSG_MV(K, N)
fMV_AX  = MSG_MV(K, N)
fMV_AX_ = MSG_MV(K, N)

fMV_U   = MSG_MV(K, N, sigma2=sigmaU2, m=0) # initial value for Sigma_U2
fMV_U._V[-1] = 0 # first and last index is not used in simulation; left at zero for nice plot 
fMV_U._V[0] = 0 # first and last index is not used in simulation; left at zero for nice plot 

dXW_X  = MSG_XW(K, N)
dXW_AX = MSG_XW(K, N)
dXW_AX_= MSG_XW(K, N)

dXW_U  = MSG_XW(K, SZ_B[1])
dXW_Y_ = MSG_XW(K, SZ_B[1])

# Marginals
MV_X    = MSG_MV(K, N)
MV_U    = MSG_MV(K, SZ_B[1])


# Boxes
box_A = Box_Lin_Transition()
box_BNUV = Box_NUV_Input()
box_Y = Box_Scalar_Output()


# Interrative Optimization
for ii in range(0,I_MAX):

    # 1) Forward Kalman
    for k in range(1,K):
        fMV_AX[k]  = box_A.fw_get_in2_fMV( fMV_X[k-1], A)
        fMV_AX_[k] = box_BNUV.fw_get_in2_fMV( fMV_AX[k], B, fMV_U._V[k], m=[[0,],])
        fMV_X[k]   = box_Y.fw_get_in2_fMV( fMV_AX_[k], y[k], C, sigmaZ2)


    # 2) Backward
    for k in range(K-1,0,-1):
        dXW_AX_[k] = box_Y.bw_get_in1_dXW( dXW_X[k], fMV_X[k], y[k], C, sigmaZ2)
        # dXW_AX_[k] = box_Y.bw_get_in1_dXW_opt2( dXW_X[k], fMV_X[k], y[k], C, sigmaZ2)
        dXW_AXk = box_BNUV.bw_get_in1_dXW( dXW_AX_[k] ) # dummy, nothing to do ...            
        dXW_X[k-1] = box_A.bw_get_in1_dXW( dXW_AXk , A)            
        
        # for input estiamte 
        dXW_U[k] = box_BNUV.bw_get_U_dXW( dXW_AX_[k], B)
        
    # 4) Posterior Mean (marginals)
    for k in range(K):
        # State X and U estimate
        MV_X[k] = box_BNUV.marginal_MV( fMV_X[k], dXW_X[k] )
        MV_U[k] = box_BNUV.marginal_MV( fMV_U[k], dXW_U[k] )
 
        fMV_U._V[k] = box_BNUV.update_variance_U_EM( MV_U._m[k] )
        
        
    # 3)  box prior to constraint m_U[k] \in [a, b]
        if True:
           a = -4.0  # lower bound
           b = 4.0  # upper bound 
           gamma = 100  # design param, make this smaller and observe how the constraint will fail
           eps = 1e-6  # to avoid numerical issues
           
           # Box prior
           fMV_U._V[k] = 1 / (gamma * ( 1/(eps + np.abs(MV_U[k][0] - a)) + 1/(eps + np.abs(MV_U[k][0] - b)) ) )   # According to Table 7.1, diss keusch
           fMV_U._m[k] = gamma * fMV_U[k][1] * ( a/(eps + np.abs(MV_U[k][0] - a)) + b/(eps + np.abs(MV_U[k][0] - b)))   
        
           # Half-Plane - working, more or less ...  # Keusch, NUV Halfplain 2023, Tab. II 
           fMV_U._m[k] = a + np.abs(MV_U[k][0] - a)
           fMV_U._V[k] = np.abs(MV_U[k][0] - a) / gamma                     



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



