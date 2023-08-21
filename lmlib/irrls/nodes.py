"""This module provides iteratively reweighted leasts squares computation closely related to Bayesian Learning.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
import numpy as np

__all__ = ['Messages_MV', 'Messages_XW', 
           'Section', 'Block','MBF']
#, 'Block', 'Block_System_A', 'Block_Input_BU', 'Block_Input_NUV', 'Block_Output_Y']


class Section():
    """ Defines the (repeating) part of a factor graph prapagating messages from time index k-1 to k """
    def __init__(self, K, N):
       self._K = K
       self._N = N
       self._block_entries = [] # list of tuples of sequential blocks with (block object, block label)
       pass

    def add_block(self, block, label = None):
       """ Addes a new block to the end of the previous blocks """
       self._block_entries.append( (label, block) )

    def optimize(self, max_itr, msg_fwrd_x0 = None , msg_bwrd_x0 = None):
        """ Optimizes the factor graph defined by the section by (iterrative) message passing """

        if msg_fwrd_x0 == None:
            msg_fwrd_x0 = Messages_MV(1, self._N)[0] # todo: do this more elegant

        if msg_bwrd_x0 == None:
            msg_bwrd_x0 = Messages_XW(1, self._N)[0] # todo: do this more elegant

        # --- Interrative Optimization
        for ii in range(0, max_itr):
            msg_fwrd = msg_fwrd_x0
            for k in range(1, self._K):
                for (label, block) in self._block_entries:                
                    # 1) Forward Kalman
                    msg_fwrd = block.pass_forward( msg_fwrd, k)

            msg_bwrd = msg_bwrd_x0
            for k in range(self._K-1,0,-1):
                for (label, block) in reversed(self._block_entries):                
                    # 2) Backward and 3) marginals
                    msg_bwrd = block.pass_backward( msg_bwrd, k )


         
           

        
class Block():
    """
    Base class for all blocks (i.e., specific groups of nodes, also denoted as closed boxes).
    Each nodes is connected to one or several branches.
    
    .. container:: twocol

        .. container:: col-fig
       
           .. image:: ./do-not-remove/../../../../../_static/lmlib/irrls/example-ex601.0-kalman-nuv-input.png   
              :width: 400



        .. container:: col-text

           Implementation of Kalman Smoother with NUV Prior on input *U*.
           All given equation references refer to:
           *"H.-A. Loeliger, L. Bruderer, H. Malmberg, F. Wadehn, and N. Zalmai: On sparsity by NUV-EM, Gaussian message passing, and Kalman smoothing"*, 2016   Information Theory & Applications Workshop (ITA), La Jolla, CA, Jan. 31 - Feb. 5, 2016
           [Loeliger2016]_.
    
    

    """
    
    def __init__(self):
        pass

    @abstractmethod
    def pass_forward(self, msg, k):
        """
        Computes forward message propagated forward from input (left branch) to output (right branch).

        Abstract method

        Parameters
        ----------
        X_fMV : :class:`Messages_MV`
                Forward message on branch X
        

        Returns
        -------
        Y_fMV : :class:`Messages_MV` 
                Forward message on branch Y
        """
    
        pass

    @abstractmethod
    def pass_backward(self, msg, k):
        """
        Computes forward message propagated forward from input (left branch) to output (right branch).

        Abstract method

        Parameters
        ----------
        X_fMV : :class:`Messages_MV`
                Forward message on branch X
        

        Returns
        -------
        Y_fMV : :class:`Messages_MV` 
                Forward message on branch Y
        """
    
        pass





    @staticmethod
    def marginal_MV( fMV, dXW ):
       """
       Computes marginals on single branche based on forward plus dual message on branch.

       Parameters
       ----------
       fMV : :class:`Messages_MV`
             Forward message 
       dXW : :class:`Messages_MV`
             XXX Messages_XW 
        
       Returns
       -------
       mv : :class:`Messages_MV` 
            Marginal distribution
       """
        
       # m_X[k] = mF_X[k] - VF_X[k]@XI_X[k]     # IV.9
       m_X = fMV[0] - fMV[1]@dXW[0]     # IV.9
       # V_X[k] = VF_X[k] - VF_X[k] @ W_X[k] @ VF_X[k]   # IV.13        
       V_X = fMV[1] - fMV[1] @ dXW[1] @ fMV[1]   # IV.13        
       return (m_X, V_X)



class MBF():
    """ Implementation of Modified Bryson–Frazier (MBF) algorithm according to Loeliger et al. 2016, "On Sparsity by NUV-EM, Gaussian Message Passing, and Kalman Smoothing". """
  
    class Block_Marginals():
        """
        Dummy node X=Y, enables marginals computation at block location by storing all forward- and backward messages.

        .. container:: twocol

            .. container:: col-fig
        
            .. image:: ./do-not-remove/../../../../../_static/lmlib/irrls/Block_Marginals.png   
                :width: 250



            .. container:: col-text

            Implementation of Kalman Smoother with NUV Prior on input *U*.
            All given equation references refer to:
            *"H.-A. Loeliger, L. Bruderer, H. Malmberg, F. Wadehn, and N. Zalmai: On sparsity by NUV-EM, Gaussian message passing, and Kalman smoothing"*, 2016   Information Theory & Applications Workshop (ITA), La Jolla, CA, Jan. 31 - Feb. 5, 2016
            [Loeliger2016]_.
        
        """
            
        def __init__(self, K, N, X_dXWs = None, Y_fMVs = None):
            """
            Constructor
            
            Parameters
            ----------
                X_dXWs : Memory to buffer dual messages or 'None'. If buffer is 'None', a new buffer will be alocated. Default: 'None'
                Y_fMVs : Memory to buffer backward messages or 'None'. If buffer is 'None', a new buffer will be alocated. Default: 'None' 
            """   

            if X_dXWs == None:
                X_dXWs = Messages_XW(K, N)

            if Y_fMVs == None:
                Y_fMVs = Messages_MV(K, N)     

            assert X_dXWs._K == Y_fMVs._K, "Buffers are not of equal length!"
            assert X_dXWs._N == Y_fMVs._N, "Buffers are not of equal dimension!"

            self._X_dXWs = X_dXWs
            self._Y_fMVs = Y_fMVs

            

        def pass_forward(self, X_fMV, k):
            """
            Computes forward message propagated forward from X to Y with `Y X`.
            
            Parameters
            ----------
            X_fMV : :class:`Messages_MV`
                    Forward message on branch X
            
            Returns
            -------
            Y_fMV : :class:`Messages_MV` 
                    Forward message on branch Y
            """
                
            self._Y_fMVs[k] = X_fMV
            return (X_fMV)


        def pass_backward(self, Y_dXW, k):
            """
            Computes dual message propagated backward from Y to X with `Y = AX`.
            
            Parameters
            ----------
            Y_dXW : :class:`Messages_XW`
                    Dual message on branch Y

            Returns
            -------
            X_dMV : :class:`Messages_MV` 
                    Dual message on branch X
            """            
            
            self._X_dXWs[k] = Y_dXW
            return (Y_dXW)
        
        def get_state_marginals(self):
            K = self._X_dXWs._K
            fX  = Messages_MV( K, self._X_dXWs._N ) # forward MV message on X
           
            for k in range(0,K): 
                fX[k] = Block.marginal_MV( self._Y_fMVs[k], self._X_dXWs[k] )      

            return fX       



    class Block_System_A():
        """
        Multiplication with arbitrary square and constant matrix A, i.e., constraint Y = AX.


        .. container:: twocol

            .. container:: col-fig
        
            .. image:: ./do-not-remove/../../../../../_static/lmlib/irrls/Block_System_A.png   
                :width: 400



            .. container:: col-text

            Implementation of Kalman Smoother with NUV Prior on input *U*.
            All given equation references refer to:
            *"H.-A. Loeliger, L. Bruderer, H. Malmberg, F. Wadehn, and N. Zalmai: On sparsity by NUV-EM, Gaussian message passing, and Kalman smoothing"*, 2016   Information Theory & Applications Workshop (ITA), La Jolla, CA, Jan. 31 - Feb. 5, 2016
            [Loeliger2016]_.
        
        """
            
        def __init__(self, A):
            """
            Constructor
            
            Parameters
            ----------
            A     :  array_like, shape=(N, N),
                    matrix to multiply with, i.e., Y  = AX (also known as transition matrix in linear systems)
            """        
            
            assert A.shape[0] == A.shape[1], "Matrix A must be a square matrix, but A of shape="+str(A.shape)+" was given instead."
            
            self._A = A # Transition matrix
            self._N = A.shape[0] # dimension
            
            
        @property
        def dim_X(self):
            """int :  Dimension of input `X` """
            return self._N

        @property
        def dim_Y(self):
            """int :  Dimension of input `X` """
            return self._N
                    

        def pass_forward(self, X_fMV, k):
            """
            Computes forward message propagated forward from X to Y with `Y = AX`.
            
            Parameters
            ----------
            X_fMV : :class:`Messages_MV`
                    Forward message on branch X
            

            Returns
            -------
            Y_fMV : :class:`Messages_MV` 
                    Forward message on branch Y
            """
                
            # in1 = (m, V)
            # (mF, VF) = mV
            # "A" node
            
            A = self._A
            AT = self._A.transpose()

            assert A.shape == X_fMV[1].shape, f"Matrix A is of dimension "+str(X_fMV[0].shape)+" instead of ("+str(X_fMV[1].shape)+"."
    
            Y_fM = A@X_fMV[0]     # III.1
            Y_fV = A@X_fMV[1]@AT   # III.2         
            return (Y_fM, Y_fV)


        def pass_backward(self, Y_dXW, k):
            """
            Computes dual message propagated backward from Y to X with `Y = AX`.
            
            Parameters
            ----------
            Y_dXW : :class:`Messages_XW`
                    Dual message on branch Y

            Returns
            -------
            X_dMV : :class:`Messages_MV` 
                    Dual message on branch X
            """            
            
            A = self._A
            AT = self._A.transpose()
                    
            assert A.shape == Y_dXW[1].shape, f"Matrix A is of dimension "+str(value[0].shape)+" instead of ("+str(Y_dXW[1].shape)+"."
            

            # "A*X = AX", but also "AX + BU = AX_"
            # in1: AX[k]
            # XI_X[k-1] = AT@XI_AX_[k] #  II.6, III.7
            
            X_dX = AT@Y_dXW[0] #  II.6, III.7
            # W_X[k-1] = AT@W_AX_[k]@A #  II.7, III.8
            X_dW = AT@Y_dXW[1]@A #  II.7, III.8
            return (X_dX, X_dW) # corresponds to k-1 !!




    class Block_Input_BU():
        """
        Additive Gaussian Input Signal


        .. container:: twocol

            .. container:: col-fig
        
            .. image:: ./do-not-remove/../../../../../_static/lmlib/irrls/Block_Input_BU.png   
                :width: 300


            .. container:: col-text

            Implementation of Kalman Smoother with NUV Prior on input *U*.
            All given equation references refer to:
            *"H.-A. Loeliger, L. Bruderer, H. Malmberg, F. Wadehn, and N. Zalmai: On sparsity by NUV-EM, Gaussian message passing, and Kalman smoothing"*, 2016   Information Theory & Applications Workshop (ITA), La Jolla, CA, Jan. 31 - Feb. 5, 2016
            [Loeliger2016]_.
        

        """
            
        # Gaussian Input as in Fig. 3
        
        #                           (N(0,1))
        #                              |
        #                              v
        #                             BU[k]
        #                              |
        #                              v
        #                       X --> (+) -->  Z


        def __init__(self, B ):
            """
            Constructor
            
            Parameters
            ----------
            B      :  array_like, shape=(N, M),
                    matrix to multiply with, i.e., Z  = X + BU (also known as input matrix B in linear systems)
            """
            self._B = B # matrix B
            self._M = B.shape[1] # input U dimension
            self._N = B.shape[0] # input X and Z dimension
            
            
        @property
        def dim_X(self):
            """int :  Dimension of input `X` """
            return self._N

        @property
        def dim_Z(self):
            """int :  Dimension of input `Z` """
            return self._N        

        @property
        def dim_U(self):
            """int :  Dimension of input `U` """
            return self._M        
            
                
        def pass_forward(self, X_fMV, k, m=[0,]):
            """
            Computes forward message propagated forward from X to Z.
            
            Parameters
            ----------
            X_fMV : :class:`Messages_MV`
                    Forward message on branch X
            sigmaU2 : float
                    Covariance matrix for input U
            m     : float
                    Mean vector for input U, optional, default: 0

            Returns
            -------
            Z_fMV : :class:`Messages_MV` 
                    Forward message on branch Y
            """
            sigmaU2 = self._sigmaU2

            B = self._B 
            BT = self._B.transpose()

            # (mF_A, VF_A) = mV_A
            # "AX+B@U=AX_" Node    
            mF_A_ = X_fMV[0] + B@m # II.1, III.1
            VF_A_ = X_fMV[1] + B@sigmaU2@BT  # II.2, III.2     
            return (mF_A_, VF_A_)

        @staticmethod
        def pass_backward(Z_dXW, k):
            """
            Computes dual message propagated backward from Z to X.
            (This function only exists for the sake of completness since it only returns `Z_dXW`)
            
            Parameters
            ----------
            Z_dXW : :class:`Messages_XW`
                    Dual message on branch Z            

            Returns
            -------
            X_fMV : :class:`Messages_MV` 
                    Forward message on branch X
            """
            
            # XiW3: AX_[k]  \tilde xi, \thilde W
            # II.6, II.7  # nothing to do (dummy)
            return (Z_dXW)

        def get_U_dualXW(self, Z_dXW):
            """
            Computes dual message propagated backward from X to U.
            
            Parameters
            ----------
            Y_dXW : :class:`Messages_XW`
                    Dual message on branch Y            

            Returns
            -------
            U_fMV : :class:`Messages_MV` 
                    Forward message on branch U
            """
            
            B = self._B 
            
            # 3) Input U_k estimate (not needed for Kalman smoothing with Gaussian inputs)
            # XI_U[k] = BT @ XI_AX_[k]   # II.6, III.7
            Xi2 = B.transpose()@Z_dXW[0]   # II.6, III.7
            # W_U[k] = BT @ W_AX_[k] @ B # II.7, III.8
            W2 = B.transpose()@Z_dXW[1]@ B # II.7, III.8
            return (Xi2, W2)


            
        
    class Block_Input_NUV(Block_Input_BU):
        """
        Additive Gaussian input with NUV prior.


        .. container:: twocol

            .. container:: col-fig
        
            .. image:: ./do-not-remove/../../../../../_static/lmlib/irrls/Block_Output_Y.png   
                :width: 400


            .. container:: col-text

            Implementation of Kalman Smoother with NUV Prior on input *U*.
            All given equation references refer to:
            *"H.-A. Loeliger, L. Bruderer, H. Malmberg, F. Wadehn, and N. Zalmai: On sparsity by NUV-EM, Gaussian message passing, and Kalman smoothing"*, 2016   Information Theory & Applications Workshop (ITA), La Jolla, CA, Jan. 31 - Feb. 5, 2016
            [Loeliger2016]_.
        

        """
        
        
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
        #                       X --> (+) -->  Z

        def __init__(self, B, sigmaU2, K = None, fU_mV = None, mU_VU = None, border_suppression = False):
            """
            Constructor
            
            Parameters
            ----------
            B      :  array_like, shape=(N, M),
                      matrix to multiply with, i.e., Z  = X + BU (also known as input matrix B in linear systems)

            sigmaU2 : array_like, shape=(M,) or array_like, shape=(M, K)
                      input variance(s) initial values (for M>1, the diagonal elements of the covariance matrices are expected)
                      if dimension is (M,) array is extended to shape (M, K) by repetition over dimension K-times.
                      (not fully implented yet)

            border_suppression:  Boolean
                      If `True`, variance at k=0 and k=K-1 are set to zero (avoids border artefacts), optional, default: False
                       
            """
            #self._B = B
            
            M = B.shape[1] # input dimension
            if K == None:
                assert False , "Signal length parameter K must be defined" 

            if fU_mV is None:
                self._fU_mV =  Messages_MV(K, M, sigma2=sigmaU2, m=0, border_suppression=border_suppression) # forward MV message on X
            else:
                assert M == fU_mV._N, "Message container fU_mV not of correct shape." 
                self._fU_mV = fU_mV

            if mU_VU is None:
                self._mU_UV =  Messages_MV(K, M) # marginals MV on U
            else:
                assert M == fU_mV._N, "Message container fU_mV not of correct shape." 
                self._mU_UV = mU_VU # only in last round where marginals are computed needed (see comment below)


            super().__init__( B )

        def pass_forward(self, X_fMV, k):
            """
            Computes forward message propagated forward from X to Z.
            
            Parameters
            ----------
            X_fMV : :class:`Messages_MV`
                    Forward message on branch X
            sigmaU2 : float
                    Covariance matrix for input U
            m     : float
                    Mean vector for input U, optional, default: 0

            Returns
            -------
            Z_fMV : :class:`Messages_MV` 
                    Forward message on branch Y
            """
            sigmaU2 = self._fU_mV._V[k]
            m = self._fU_mV._m[k]

            B = self._B 
            BT = self._B.transpose()

            # (mF_A, VF_A) = mV_A
            # "AX+B@U=AX_" Node    
            mF_A_ = X_fMV[0] + B@m # II.1, III.1
            VF_A_ = X_fMV[1] + B@sigmaU2@BT  # II.2, III.2     

            return (mF_A_, VF_A_)
        

        def pass_backward(self, Z_dXW, k):
            """
            Computes dual message propagated backward from Z to X.
            (This function only exists for the sake of completness since it only returns `Z_dXW`)
            
            Parameters
            ----------
            Z_dXW : :class:`Messages_XW`
                    Dual message on branch Z            

            Returns
            -------
            X_fMV : :class:`Messages_MV` 
                    Forward message on branch X
            """

            dAX_k = Z_dXW
 
            # 4) Posterior Mean (marginals), State U estimate
            dUk = self.get_U_dualXW( dAX_k )
            mU_k = Block.marginal_MV( self._fU_mV[k], dUk )

            self._mU_UV[k] = mU_k # Todo: only necessery for displaying at the end --> should not always be computed

            # 1)  vanilla NUV prior using EM
            # VF_U[k] = V_U[k] + m_U[k]**2   # Update sigmaU_k according to Eq. (13)
        
            # 2)  vanilla NUV prior using AM
            self._fU_mV._V[k] = self.update_variance_U_EM( mU_k[0] )
            # VF_U[k] = mU_k**2   # Update sigmaU_k according to Eq. (13)


            # XiW3: AX_[k]  \tilde xi, \thilde W
            # II.6, II.7  # nothing to do (dummy)
            return (Z_dXW)   



        def get_input_marginals(self):  
            """
            Returns marginals on input U.
            Todo: If marginals are not needed, it is still beeing stored. This could be optimized.
            

            Returns
            -------
            mU_UV : Message UV 
                Marginal on U
            """      

            return (self._mU_UV)



        @staticmethod
        def update_variance_U_EM( M_U ):
            """
            Updates variance of U using EM method according to Eq. (13) 
            
            Parameters
            ----------
            M_U : float
                Marginal mean on branch U (scalar)

            Returns
            -------
            V_U : float 
                Variance estimate of (scalar) U using EM iterration step.
            """        

            # TODO Is this EM or AM? Check!
            # 1)  vanilla NUV prior using EM
            # VF_U[k] = V_U[k] + m_U[k]**2   # Update sigmaU_k according to Eq. (13)
        
            # 2)  vanilla NUV prior using AM
            # VF_U[k] = mU_k**2   # Update sigmaU_k according to Eq. (13)


            
            return (M_U**2)   # Update sigmaU_k according to Eq. (13)
            
        def get_U_dualXW(self, Z_dXW):
            """
            Computes dual message propagated backward from X to U.
            
            Parameters
            ----------
            Y_dXW : :class:`Messages_XW`
                    Dual message on branch Y            

            Returns
            -------
            U_fMV : :class:`Messages_MV` 
                    Forward message on branch U
            """
            
            B = self._B 
            
            # 3) Input U_k estimate (not needed for Kalman smoothing with Gaussian inputs)
            # XI_U[k] = BT @ XI_AX_[k]   # II.6, III.7
            Xi2 = B.transpose()@Z_dXW[0]   # II.6, III.7
            # W_U[k] = BT @ W_AX_[k] @ B # II.7, III.8
            W2 = B.transpose()@Z_dXW[1]@ B # II.7, III.8
            return (Xi2, W2)        
            

    class Block_Output_Y():
        """
        Additive Gaussian Input Signal


        .. container:: twocol

            .. container:: col-fig
        
            .. image:: ./do-not-remove/../../../../../_static/lmlib/irrls/Block_Output_Y.png   
                :width: 250


            .. container:: col-text

            Implementation of Kalman Smoother with NUV Prior on input *U*.
            All given equation references refer to:
            *"H.-A. Loeliger, L. Bruderer, H. Malmberg, F. Wadehn, and N. Zalmai: On sparsity by NUV-EM, Gaussian message passing, and Kalman smoothing"*, 2016   Information Theory & Applications Workshop (ITA), La Jolla, CA, Jan. 31 - Feb. 5, 2016
            [Loeliger2016]_.
        

        """
            
            
        def __init__(self, K, C, y, sigmaZ2, Z_fMVs = None):
            """
            Constructor
            
            Parameters
            ----------
            C     :  array_like, shape=(M, N),
                    matrix to multiply with, i.e., Y  = CX + N(0, sigmaZ2) (also known as input matrix B in linear systems)


            """
            self._C = C # matrix B
            self._M = C.shape[0] # input Y dimension
            self._N = C.shape[1] # input X and Z dimension 

            self._y = y  # input signal
            self._sigmaZ2 = sigmaZ2 # sigmaZ^2

            if Z_fMVs is None:
                self._Z_fMVs =  Messages_MV(K, self._N) # memory for MV forwared message on branch Z
            else:
                assert Z_fMVs._M == self._M, "Message container Z_fMVs not of correct shape." 
                assert K == Z_fMVs._K, "Message container Z_fMVs not of correct shape." 
                self._Z_fMVs = Z_fMVs # memory for MV forwared message on branch Z

            # self._X_fMVs = X_fMVs # memory for MV forwared message on branch X
            
            
        @property
        def dim_X(self):
            """int :  Dimension of input `X` """
            return self._N

        @property
        def dim_Z(self):
            """int :  Dimension of input `Z` """
            return self._N        

        @property
        def dim_Y(self):
            """int :  Dimension of input `Y` """
            return self._M             
        
        
        # Scalar output Block as in Fig. 2 (lower branch)
        
        #                                       X ---(=) --> Z
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

            
        def  pass_forward(self, X_fMV, k):
            """
            Computes forward message propagated forward from Branch X to Z.
            
            Parameters
            ----------
            X_fMV : :class:`Messages_MV`
                    Forward Message on branch X            

            Returns
            -------
            Z_fMV : :class:`Messages_MV` 
                    Forward message on branch Z
            """
            
            # "Y_+Z=Y" Node
            #mB_Y_[k] = y
            #VB_Y_[k] = sigmaZ2

            sigmaZ2 = self._sigmaZ2
            yk = self._y[k]
            
            C = self._C
            CT = self._C.transpose()

            # "C=" Node
            G = np.linalg.inv(sigmaZ2 + C@X_fMV[1]@CT)  #  V.3
            mF_ = X_fMV[0]+X_fMV[1]@CT@G@(yk - C@X_fMV[0]) #  V.1
            VF_ = X_fMV[1]-X_fMV[1]@CT@G@C@X_fMV[1] #  V.2   

            self._Z_fMVs[k] = (mF_, VF_)
          #  self._X_fMVs[k] = X_fMV

            return self._Z_fMVs[k]
            

        def  pass_backward(self, Z_dXW, k):
            """
            Computes dual message propagated back from Branch Z to X.
            
            
            Parameters
            ----------
            Z_dXW : :class:`Messages_MV`
                    Dual Message on branch Z            
            Z_fMV : :class:`Messages_MV`
                    Forward Message on branch Z            


            Returns
            -------
            X_dualMV : :class:`Messages_MV` 
                    Forward message on branch Z
            """

            
            Z_fMV = self._Z_fMVs[k]

            yk = self._y[k]
            sigmaZ2 = self._sigmaZ2
            
            SZ_C = self._C.shape
            
            C = self._C
            CT = self._C.transpose()        
            
            W_Y_ = np.array([[1/sigmaZ2,],])

            
            # ---
            #F = np.eye(SZ_C[1])-VF_X[k]@CT@W_Y_@C #  V.8
            F = np.eye(SZ_C[1])-Z_fMV[1]@CT@W_Y_@C #  V.8
            FT = F.transpose()
        
            # "AX_ = X = Y"
            # --- Option 1 - Using F
            # XI_AX_[k] = FT@XI_X[k]+CT@W_Y_@(C@mF_X[k]-mB_y_)  #  V.4
            X_dX = FT@Z_dXW[0]+CT@W_Y_@(C@Z_fMV[0]-yk)  #  V.4
            # W_AX_[k] = FT@W_X[k]@F+CT@W_Y_@C@F  #  V.6
            X_dW = FT@Z_dXW[1]@F+CT@W_Y_@C@F  #  V.6
            
            return (X_dX, X_dW)
            

        
        def  pass_backward_opt2(self, Z_dXW, k):
            """
            Computes dual message propagated backward from Z to X.
            
            Parameters
            ----------
            Z_dXW : :class:`Messages_MV`
                    Dual Message on branch Z            
            Z_fMV : :class:`Messages_MV`
                    Forward Message on branch Z            


            Returns
            -------
            X_dualMV : :class:`Messages_MV` 
                    Forward message on branch Z
            """

            Z_fMV = self._Z_fMVs[k]

            yk = self._y[k]
            sigmaZ2 = self._sigmaZ2
            
            SZ_C = self._C.shape
            CT = self._C.transpose()
            
            C = self._C
            
            W_Y_ = np.array([[1/sigmaZ2,],])
            
            #F = np.eye(SZ_C[1])-VF_X[k]@CT@W_Y_@C #  V.8
            F = np.eye(SZ_C[1])-Z_fMV[1]@CT@W_Y_@C #  V.8
            FT = F.transpose()
            
            # --- Option 2 (reusing G)
            # G = np.linalg.inv(VB_Y_[k] + C @ VF_AX_[k] @ CT)  # V.3
            G = np.linalg.inv(sigmaZ2 + C @ VF_AX_[1] @ CT)  # V.3 # TODO VF_AX_ not yet stored
            # XI_AX_[k] = FT@XI_X[k]+CT@G@(C@mF_AX_[k]-mB_y_)  #  V.5
            X_dX = FT@Z_dXW[0]+CT@G@(C@Z_fMV[0]-yk)  #  V.5
            # W_AX_[k] = FT@W_X[k]@F+CT@G@C  #  V.7
            X_dW = FT@Z_dXW[1]@F+CT@G@C  #  V.7
            # ---
            return (X_dX, X_dW)
            
        

        




class Messages_MV():
    """
    Array storing K tuples (=messages) with mean vector `m` and a covariance matrice `V`.
    Such tuples (`m`, `V`) represent a single, not normalized Gaussian distribution of order `N`.
    
    This class provides access to any tuple at index `k` by
    ::
       mv = Message_MV(1000, 3) # initializing new message array
       ...
       mv[k] = (m, V)  # setting a value using setter function
       (m, V) = mv[k]  # getting a value using getter function
    
    Todo: Setter and getter are not shown in doc. Why?
    
    """

    def __init__(self, K, N, sigma2=0, m=0, border_suppression = False):
       """
       Constructor 
        
       Parameters
       ----------
       K : int
           number of messages (mostly the number of samples to be processed)
       N : int
           message dimension
       sigma2 : float
          variance used to initialize all diagonal elements of `V` over all `k`s, default: '0'
       m : float
          mean value used to initialize all elements of `m` over all `k`s, default: '0'
       border_suppression:  Boolean
          If `True`, variance at k=0 and k=K-1 are set to zero (avoids border artefacts), optional, default: False
       """
        
       self._N = N # system dimension
       self._K = K # array size
        
       self._m = np.zeros([K, N])
       self._V = np.zeros([K, N, N])
       if sigma2 != 0:
          self._V[:,:,:] = np.eye(N)*sigma2
       if border_suppression == True:
           self._V[-1] = 0 # first and last index is not used in simulation; left at zero for nice plot 
           self._V[0] = 0 # first and last index is not used in simulation; left at zero for nice plot 
           

    def __setitem__(self, k, value): 
#    def xx(self, k, value): 
       """
       Sets mean `m` and variance `V` at array index `k` of the array.
       Parameters
       ----------
       k : int
           Array index `0..K-1`
       value : tuple 
              (`m`, `V`) with `m` of :class:`~numpy.ndarray` of shape=(N,) and with `V` of :class:`~numpy.ndarray` of shape=(N,N)
       """
       assert k < self._K, f"Array index "+str(k)+" is out of range of 0 to "+str(self._K)+"."
       assert value[0].shape == (self._N,), f"Mean value dimension of "+str(value[0].shape)+" does not match required mean array element dimension of ("+str(self._N)+",)."
       assert value[1].shape == (self._N,self._N), f"Variance value dimension of "+str(value[0].shape)+" does not match required variance array element dimension of ("+str(self._N)+","+str(self._N)+")."
       
       (self._m[k], self._V[k]) = value

    def __getitem__(self, k): 
       """
       Gets mean `m` and variance `V` from array index `k` of the array.

       Parameters
       ----------
       k : int
           Array index `0..K-1`
        
       Returns
       -------
       x : tuple 
           (`m`, `V`) with `m` of :class:`~numpy.ndarray` of shape=(N,) and with `V` of :class:`~numpy.ndarray` of shape=(N,N)
       """
       assert k < self._K, f"Array index "+str(k)+" is out of range of 0 to "+str(self._K)+"."
       
       return (self._m[k], self._V[k]) 
    
    
class Messages_XW():
    """
    Array storing K tuples (=messages) with weighted mean vector `xi` and a precision matrice `V`.
    Each such tuple (`xi`, `W`) represents a single Gaussian distribution of order `N`.
    """
 
    def __init__(self, K, N):        
      self._N = N # system dimension
      self._K = K # array size

      self._xi = np.zeros([K, N])
      self._W  = np.zeros([K, N, N])
  
    def __setitem__(self, key, value): 
        (self._xi[key], self._W[key]) = value

    def __getitem__(self, key): 
        return (self._xi[key], self._W[key]) 
        
        
 