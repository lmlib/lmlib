"""
Kalman Smoother with Gaussian Input [ex600.0]
=============================================


.. container:: twocol

    .. container:: col-fig
       
       .. image:: ./do-not-remove/../../../../../_static/gallery/example-ex600.0-kalman-gaussian-input.png   
          :width: 400

    .. container:: col-text

       **Education Examples (Edu):** This is a full implementation of recursive least squares filter with Gaussian input according to Loeliger2016]_.
       For ease of comprehension and for teaching purposes, the use of library functions is omitted here.
       However, for any practical applications, we highly recommend to use the more flexible, ready made module :ref:`lmlib.irrls` instead, as demonstrated in [Example ex601.3] and others.  

       Implementation of Kalman Smoother with Gaussian input *U*.
       All given equation references refer to:
       *"H.-A. Loeliger, L. Bruderer, H. Malmberg, F. Wadehn, and N. Zalmai: On sparsity by NUV-EM, Gaussian message passing, and Kalman smoothing"*, 2016   Information Theory & Applications Workshop (ITA), La Jolla, CA, Jan. 31 - Feb. 5, 2016
       [Loeliger2016]_.
"""


import matplotlib.pyplot as plt
import numpy as np

import lmlib as lm
from lmlib.utils.generator import (gen_wgn, gen_steps, gen_rand_walk, gen_rect)
from lmlib.irrls import *

# --------------- generate test signal ---------------
K = 2*400  # number of samples (length of test signal)
y = np.hstack([2*gen_rect(K//2, k_period=100, k_on=50), gen_rand_walk(K//2, seed=12345678)])
print(y.shape)

# --------------- Main ------------------------------------------------------
#  Factor Graph with Nodes "(..)" and Variables "--X[k]-->" 
# ---------------------------------------------------------------------------
#                         (N(0,sigmaU2))
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


# Initial values
N = 1  # signal model order to use
I_MAX = 1  # 200 # Number of iterations
sigmaZ2 = 1  # initial value for scalar Sigma Z^2
sigmaU2 = [10, ]  # initial value for Sigma U^2 (diagonal Elements of covariance matrix)

# --- Defining Blocks  ---

# Block:   AX := A@X[k-1]
A = lm.AlssmPoly(poly_degree=N - 1).A
block_A = MBF.Block_System_A(A)

# Block: A_ := AX+B@N(0, sigmaU2)
B = np.concatenate((np.zeros((N - 1, 1)), np.ones((1, 1))))  # generate input matrix of the form [0, ..., 0, 1]^T
block_input_NUV = MBF.Block_Input_NUV(B, sigmaU2, K=K, border_suppression=True)

# Block:  X[k] := A_ = C(y[k] + N(0, sigmaZ2))
C = lm.AlssmPoly(poly_degree=N - 1, force_MC=True).C
block_output_Y = MBF.Block_Output_Y(K, C, y, sigmaZ2)
block_marginals = MBF.Block_Marginals(K, N)

section = Section(K, N)
section.append_block(block_A, 'block_A')
section.append_block(block_input_NUV, 'block_input_NUV')
section.append_block(block_output_Y, 'block_output_Y')
section.append_block(block_marginals, 'block_marginals')

section.optimize(max_itr=I_MAX)

# ---  computing marginals for final plotting ---
mU = block_input_NUV.get_input_marginals()
mX = block_marginals.get_state_marginals()

## -- Plotting ---
ks = np.arange(K)
fig, axs = plt.subplots(3, 1, figsize=(8, 4), gridspec_kw={'height_ratios': [2, 1, 1]}, sharex='all')

# Remove horizontal space between axes
fig.subplots_adjust(hspace=0.1)
fig.tight_layout()

line_1, = axs[0].plot(ks, y, color="gray", linestyle="-", lw=1.0, label='$y$ (Observations)')
line_2, = axs[0].plot(ks, mX._m, color="b", lw=1.25, label='posterior $m(X)$')
axs[0].legend(loc='upper right')


axs[1].plot(ks, mU._m, 'b', label='$u$ (m_U)')
axs[1].axhline(y = 0, color = 'k', linestyle = '-', linewidth=0.5)
axs[1].set(ylabel='Input Mean')
axs[1].legend(loc='upper right')

line_4, = axs[2].plot(ks, mU._V[:, 0, 0], color="black", lw=0.75, label='$\sigma_U^2$ (sigmaU2)')
axs[2].set(ylabel='Input Var.')
axs[2].legend(loc='upper right')

plt.show()

