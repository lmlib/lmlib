"""
This module implements the MassagePassing Algorithms (MBF, BIFM)

Switching between MBF and BFIM in the FactorGraph is done using the state design pattern.
MassagePassing is the state and FactorGraph the context


"""

from abc import ABC, abstractmethod
import numpy as np

__all__ = ['MBF']


def allocate_gaussian_random_variable(K, N, str_mean='m', str_covariance='V'):
    """
    Allocate a Gaussian random variable over `K` with mean vector and covariance matrix

    The mean vector can be accessed by ``X.m` and the covariance matrix by ``X.V``.

    Parameters
    ----------
    K : int
        Length of the Gaussian random variable
    N : int
        Dimensions of Mean and Covariance matrix
    str_mean : str, optional
        Access name for mean default='m'
    str_covariance : str, optional
        Access name for mean default='V'

    Returns
    -------
    out : :class:`~numpy.recarray`
        Gaussian Random Variable of length K containing K Means of size N and K covariance matrix of size (N, N)

    """
    return np.recarray((K,), dtype=[(str_mean, 'f8', (N,)), (str_covariance, 'f8', (N, N))])


class MassagePassingBlock(ABC):
    """
    Abstract base for Message Passing Blocks


    Parameters
    ----------
    block : :class:`~lmlib.BlockBase`
        block
    K : int
        Length of the Factor Graph
    """

    def __init__(self, block, K):
        self.block = block
        self.block_label = block.label
        self.K = K
        self.memory = dict()

        # malloc
        if self.block.save_marginals:
            self.memory['X'] = allocate_gaussian_random_variable(self.K, self.block.N)

    @abstractmethod
    def propagate_forward(self, k, msg_fw):
        pass

    @abstractmethod
    def propagate_backward(self, k, msg_bw):
        pass

    @abstractmethod
    def propagate_forward_save_states(self, k, msg_fw):
        pass

    @abstractmethod
    def propagate_backward_save_states(self, k, msg_bw):
        pass

    @abstractmethod
    def get_block_by_label(self, label, out_dict):
        pass

    @abstractmethod
    def get_block_by_obj(self, obj, out_dict):
        pass

    def get_marginals(self):
        """
        Returns the marginals `X` of the block

        Returns
        -------
        out :  :class:`~numpy.recarray`
            marginals of the mp_block.
            `X.m` calls the mean of shape (K, N) and `X.V` calls the covariance of shape (K, N, N)
        """

        assert self.block.save_marginals, f"No marginals saved! Set save_marginals=True on {self.block}"
        return self.memory['X']

    def get_X(self):
        """
        See :func:`get_marginals`
        """
        return self.get_marginals()


class MessagePassing(ABC):
    """
    Abstract base for Message Passing Algorithm
    """
    @classmethod
    def get_forward_initial_state(cls, N, m=0, V=0):
        """
        Returns the initial state at k=0 for the message passing algorithm for the forward recursion

        Parameters
        ----------
        N : int
            dimension of state
        m : float, array_like of shape (N,)
            value of the initial mean vector
        V : float, array_like of shape (N, N)
            value of the initial covariance matrix. if `V` is scalar the diagnoal will be filled with `V`

        Returns
        -------
        out : :class:`~numpy.recarray`
            Single Message Passing State
        """
        m = np.full((N,), m)
        V = np.eye(N) * V
        return np.array((m, V), dtype=[('m', 'f8', (N,)), ('V', 'f8', (N, N))]).view(np.recarray)

    @classmethod
    def get_backward_initial_state(cls, N, xi=0, W=0):
        """
        Returns the initial state at k=K-1 for the message passing algorithm for the backward recursion

        Parameters
        ----------
        N : int
            dimension of state
        xi : float, array_like of shape (N,)
            value of the initial mean vector
        W : float, array_like of shape (N, N)
            value of the initial covariance matrix. if `W` is scalar the diagnoal will be filled with `W`

        Returns
        -------
        out : :class:`~numpy.recarray`
            Single Message Passing State
        """
        xi = np.full((N,), xi)
        W = np.eye(N) * W
        return np.array((xi, W), dtype=[('xi', 'f8', (N,)), ('W', 'f8', (N, N))]).view(np.recarray)

    @classmethod
    def create_mp_block(cls, block, K):
        """
        Creates the message passing block structure of the recursion and sets the length `K`,

        This resamples the section in [Loeliger2016]_ .

        Parameters
        ----------
        block : :class:`~lmlib.BlockBase`
            The message passing block to apply the recursions for each step k
        K : int
            length of recursion / factor graph

        Returns
        -------
        out : :class:`~lmlib.MassagePassingBlock`
            MassagePassingBlock
        """
        return getattr(cls, block.__class__.__name__)(block, K)


class MBF(MessagePassing):
    class BlockBaseMBF(MassagePassingBlock, ABC):
        """
        Base class for message passing block MBF
        """

        def propagate_forward_save_states(self, k, msg_fw):
            self.propagate_forward(k, msg_fw)

            if self.block.save_marginals:
                self.memory['X'][k] = msg_fw  # IV.9 & IV.13 (forward update with temporary result)

        def propagate_backward_save_states(self, k, msg_bw):
            self.propagate_backward(k, msg_bw)

            if self.block.save_marginals:
                VF_X = self.memory['X'][k].V

                # (backward update based on temporary result from forward-path)
                self.memory['X'][k].m -= VF_X @ msg_bw.xi  # IV.9 (backward update)
                self.memory['X'][k].V -= VF_X @ msg_bw.W @ VF_X  # IV.13 (backward update)

        def _check_and_set_sigma2_init(self, variable, n):

            # check sigma2 input type
            sigma2_init = self.block.sigma2_init
            if np.isscalar(sigma2_init):
                variable.V[:] = np.eye(n) * sigma2_init
            elif np.shape(sigma2_init) == (n, n):
                variable.V[:] = sigma2_init
            else:
                raise TypeError(f'Unexpected sigma2_init type/shape in {self.block}, \n '
                                f'expected scalar or {(n, n)}, got {np.shape(sigma2_init)}')

        def _check_and_set_sigma2_k_init(self, variable, K, n):

            # check sigma2 input type
            sigma2_init = self.block.sigma2_init
            if np.isscalar(sigma2_init):
                variable.V[:] = np.repeat([np.eye(n) * sigma2_init], K, axis=0)
            elif np.shape(sigma2_init) == (n, n):
                variable.V[:] = np.repeat([sigma2_init], K, axis=0)
            elif np.shape(sigma2_init) == (K,):
                variable.V[:] = np.einsum('knm, k->knm', [np.eye(n)], sigma2_init)
            elif np.shape(sigma2_init) == (K, n, n):
                variable.V[:] = sigma2_init
            else:
                raise TypeError(f'Unexpected sigma2_init type/shape in {self.block}, \n '
                                f'expected scalar or shape {(n, n)}, {(K,)}, {(K, n, n)}, got {np.shape(sigma2_init)}')

        def get_block_by_label(self, label, out_dict):
            """
            subroutine: Get a message passing block graph by label
            """
            if self.block.label == label:
                out_dict['mp_block'] = self
                print(out_dict)

        def get_block_by_obj(self, obj, out_dict):
            """
            subroutine: Get a message passing block graph by object
            """
            if self.block is obj:
                out_dict['mp_block'] = self

    class BlockSystem(BlockBaseMBF):
        """
        MBF System Block

        Details in [Loeliger2016]_
        --------------------------
        - TABLE III, GAUSSIAN MESSAGE PASSING THROUGH A MATRIX MULTIPLIER NODE WITH ARBITRARY REAL MATRIX A.

        """
        def __init__(self, block, K):
            super().__init__(block, K)

        def propagate_forward(self, k, msg_fw):
            A = self.block.A
            msg_fw.m[:] = A @ msg_fw.m  # III.1
            msg_fw.V[:] = A @ msg_fw.V @ A.T  # III.2

        def propagate_backward(self, k, msg_bw):
            A = self.block.A
            AT = A.T
            msg_bw.xi[:] = AT @ msg_bw.xi  # II.6, III.7
            msg_bw.W[:] = AT @ msg_bw.W @ A  # II.7, III.8

    class BlockInput(BlockBaseMBF):
        """
        MBF Input Block with fix variance over k

        Details in [Loeliger2016]_
        --------------------------
        - TABLE II, GAUSSIAN MESSAGE PASSING THROUGH AN ADDER NODE.
        - TABLE III, GAUSSIAN MESSAGE PASSING THROUGH A MATRIX MULTIPLIER NODE WITH ARBITRARY REAL MATRIX A.
        - TABLE IV GAUSSIAN SINGLE-EDGE MARGINALS (m, V ) AND THEIR DUALS (ξ ̃, W ̃ ). (marginal and input/output estimation)
        """

        def __init__(self, block, K):
            super().__init__(block, K)

            # malloc
            if self.block.estimate_input:
                self.memory['U'] = allocate_gaussian_random_variable(self.K, self.block.M)

            # algorithm memory
            self._initialize_fw_U()

        def _initialize_fw_U(self):
            self._fw_U = allocate_gaussian_random_variable(1, self.block.M)[0]
            self._fw_U.m.fill(0)
            self._check_and_set_sigma2_init(self._fw_U, self.block.M)

        def propagate_forward(self, k, msg_fw):
            B = self.block.B
            msg_fw.m += B @ self._fw_U.m  # II.1, III.1
            msg_fw.V += B @ self._fw_U.V @ B.T  # II.2, III.2

        def propagate_backward(self, k, msg_bw):
            pass  # II.6, II.7  # nothing to do

        def propagate_forward_save_states(self, k, msg_fw):
            super().propagate_forward_save_states(k, msg_fw)

            if self.block.estimate_input:
                self.memory['U'][k] = self._fw_U  # IV.9 & IV.13 (forward update)

        def propagate_backward_save_states(self, k, msg_bw):
            super().propagate_backward_save_states(k, msg_bw)

            # save
            if self.block.estimate_input:
                B = self.block.B
                VF_U = self.memory['U'][k].V
                self.memory['U'][k].m -= VF_U @ B.T @ msg_bw.xi  # III.7 & IV.9 (backward update)
                self.memory['U'][k].V -= VF_U @ B.T @ msg_bw.W @ B @ VF_U  # III.8 & IV.13 (backward update)

        def get_input_estimate(self):
            """
            Returns the input estimate `U` of the input block

            Returns
            -------
            marginals : np.ndarray of shape (K, ...)
                marginals of the mp_block.
                `marginals.m` calls the mean of shape (K, N) and `marginals.V` calls the covariance of shape (K, N, N)
            """

            assert self.block.estimate_input, f"No input estimate saved! Set estimate_input=True on {self.block}"
            return self.memory['U']

        def get_U(self):
            """
            See :func:`get_input_estimate`
            """
            return self.get_input_estimate()

    class BlockInput_k(BlockBaseMBF):
        """
        MBF Input Block with variable variance over k

        Details in [Loeliger2016]_
        --------------------------
        - TABLE II, GAUSSIAN MESSAGE PASSING THROUGH AN ADDER NODE.
        - TABLE III, GAUSSIAN MESSAGE PASSING THROUGH A MATRIX MULTIPLIER NODE WITH ARBITRARY REAL MATRIX A.
        - TABLE IV GAUSSIAN SINGLE-EDGE MARGINALS (m, V ) AND THEIR DUALS (ξ ̃, W ̃ ). (marginal and input/output estimation)
        """
        def __init__(self, block, K):
            super().__init__(block, K)

            # malloc
            if self.block.estimate_input:
                self.memory['U'] = allocate_gaussian_random_variable(self.K, self.block.M)

            # algorithm memory
            self._initialize_fw_U()

        def _initialize_fw_U(self):
            self._fw_U = allocate_gaussian_random_variable(self.K, self.block.M)
            self._fw_U.m.fill(0)
            self._check_and_set_sigma2_k_init(self._fw_U, self.K, self.block.M)

        def propagate_forward(self, k, msg_fw):
            B = self.block.B
            fw_U = self._fw_U[k]
            msg_fw.m += B @ fw_U.m  # II.1, III.1
            msg_fw.V += B @ fw_U.V @ B.T  # II.2, III.2

        def propagate_backward(self, k, msg_bw):
            pass  # II.6, II.7  # nothing to do

        def propagate_forward_save_states(self, k, msg_fw):
            super().propagate_forward_save_states(k, msg_fw)

            if self.block.estimate_input:
                self.memory['U'][k] = self._fw_U[k]  # IV.9 & IV.13 (forward update)

        def propagate_backward_save_states(self, k, msg_bw):
            super().propagate_backward_save_states(k, msg_bw)

            # save
            if self.block.estimate_input:
                B = self.block.B
                VF_U = self.memory['U'][k].V
                self.memory['U'][k].m -= VF_U @ B.T @ msg_bw.xi  # III.7 & IV.9 (backward update)
                self.memory['U'][k].V -= VF_U @ B.T @ msg_bw.W @ B @ VF_U  # III.8 & IV.13 (backward update)

        def get_input_estimate(self):
            """
            Returns the input estimate `U` of the input block

            Returns
            -------
            out : np.ndarray of shape (K, ...)
                input estimate of the mp_block.
                `U.m` calls the mean of shape (K, N) and `U.V` calls the covariance of shape (K, N, N)
            """

            assert self.block.estimate_input, f"No input estimate saved! Set estimate_input=True on {self.block}"
            return self.memory['U']

        def get_U(self):
            """
            See :func:`get_input_estimate`
            """
            return self.get_input_estimate()

    class BlockInputNUV(BlockInput_k):
        """
        MBF NUV Input Block fix variance over k

        Details in [Loeliger2016]_
        --------------------------
        - TABLE II, GAUSSIAN MESSAGE PASSING THROUGH AN ADDER NODE.
        - TABLE III, GAUSSIAN MESSAGE PASSING THROUGH A MATRIX MULTIPLIER NODE WITH ARBITRARY REAL MATRIX A.
        - TABLE IV GAUSSIAN SINGLE-EDGE MARGINALS (m, V ) AND THEIR DUALS (ξ ̃, W ̃ ). (marginal and input/output estimation)
        """
        def __init__(self, block, K):
            super().__init__(block, K)

            # malloc
            if self.block.save_deployed_sigma2:
                self.memory['U_fw'] = allocate_gaussian_random_variable(self.K, self.block.M)

        def propagate_backward(self, k, msg_bw):
            super().propagate_backward(k, msg_bw)
            self.update_NUV(k, msg_bw)

        def propagate_backward_save_states(self, k, msg_bw):
            if self.block.save_deployed_sigma2:
                self.memory['U_fw'][k] = self._fw_U[k]

            super().propagate_backward_save_states(k, msg_bw)

        def update_NUV(self, k, msg_bw):

            B = self.block.B
            fw_U = self._fw_U[k]
            U_m = fw_U.m - fw_U.V @ B.T @ msg_bw.xi  # III.7 & IV.9 (backward update)

            # 1)  vanilla NUV prior using EM
            U_V = fw_U.V - fw_U.V @ B.T @ msg_bw.W @ B @ fw_U.V  # III.8 & IV.13 (backward update)
            self._fw_U[k].V = np.outer(U_m, U_m) + U_V  # Update sigmaU_k according to Eq. (13)

            # 2)  vanilla NUV prior using AM
            # self._fw_U[k].V = np.inner(U_m, U_m)  # Update sigmaU_k according to Eq. (13) TODO check EQ BUG?

        def get_deployed_sigma2(self):
            """
            Returns the Random Variable of deployed_sigma2 `U_fw` of the input NUV block

            Returns
            -------
            out : np.ndarray of shape (K, ...)
                updated input estimate of forward path of the mp_block.
                `U_fw.m` calls the mean of shape (K, N) and `U_fw.V` calls the covariance of shape (K, N, N)
            """

            assert self.block.estimate_input, f"No updated input estimate saved! Set get_deployed_sigma2=True on {self.block}"
            return self.memory['U_fw']

        def get_U_fw(self):
            """
            See :func:`get_deployed_sigma2`
            """
            return self.get_deployed_sigma2()

    class BlockOutput(BlockBaseMBF):
        """
        MBF Output Block fix variance over k

        Details in [Loeliger2016]_
        --------------------------
        - TABLE V GAUSSIAN MESSAGE PASSING THROUGH AN OBSERVATION BLOCK.
        - TABLE IV GAUSSIAN SINGLE-EDGE MARGINALS (m, V ) AND THEIR DUALS (ξ ̃, W ̃ ). (marginal and input/output estimation)
        """
        def __init__(self, block, K):
            super().__init__(block, K)

            # malloc
            if self.block.estimate_output:
                self.memory['Yt'] = allocate_gaussian_random_variable(self.K, self.block.L)

            # algorithm memory
            self._X_fw = allocate_gaussian_random_variable(self.K, self.block.N)
            self._initialize_Z()

        def _initialize_Z(self):
            self._Z = allocate_gaussian_random_variable(1, self.block.L)[0]
            self._Z.m.fill(0)
            self._check_and_set_sigma2_init(self._Z, self.block.L)

        def propagate_forward(self, k, msg_fw):
            C = self.block.C
            CT = C.T
            y = self.block.y[k]
            Z = self._Z

            G = np.linalg.inv(Z.V + C @ msg_fw.V @ CT)  # V.3
            msg_fw.m += msg_fw.V @ CT @ G @ (y - C @ msg_fw.m)  # V.1
            msg_fw.V -= msg_fw.V @ CT @ G @ C @ msg_fw.V  # V.2

            self._X_fw[k] = msg_fw  # for backward propagation

        def propagate_backward(self, k, msg_bw):
            C = self.block.C
            CT = C.T
            y = self.block.y[k]
            Z = self._Z
            X_fw = self._X_fw[k]

            W_Y = np.linalg.inv(Z.V)
            F = np.eye(self.block.N) - X_fw.V @ CT @ W_Y @ C  # V.8

            msg_bw.xi[:] = F.T @ msg_bw.xi + CT @ W_Y @ (C @ X_fw.m - y)  # V.4
            msg_bw.W[:] = F.T @ msg_bw.W @ F + CT @ W_Y @ C @ F  # V.6

        def propagate_backward_save_states(self, k, msg_bw):

            super().propagate_backward_save_states(k, msg_bw)

            if self.block.estimate_output:
                C = self.block.C
                X = self.memory['X'][k]
                self.memory['Yt'][k].m = C @ X.m  # IV.9 (backward update)
                self.memory['Yt'][k].V = C @ X.V @ C.T  # IV.13 (backward update)

        def get_output_estimate(self):
            """
            Returns the output_estimate `\tilde{Y}` of the output block

            Returns
            -------
            out : np.ndarray of shape (K, ...)
                updated output estimate of forward path of the mp_block.
                `Y.m` calls the mean of shape (K, N) and `Y.V` calls the covariance of shape (K, N, N)
            """

            assert self.block.estimate_input, f"No output estimate saved! Set estimate_output=True on {self.block}"
            return self.memory['Yt']

        def get_Yt(self):
            """
            See :func:`get_output_estimate`
            """
            return self.get_output_estimate()

        def get_Z(self):
            """
            Returns the output prior `\tilde{Z}` of the output block

            Returns
            -------
            out : np.ndarray of shape (K, ...)
                updated output estimate of forward path of the mp_block.
                `Y.m` calls the mean of shape (K, N) and `Y.V` calls the covariance of shape (K, N, N)
            """

            assert self.block.estimate_input, f"No output estimate saved! Set estimate_output=True on {self.block}"
            return self.memory['Z']

    class BlockOutputOutlier(BlockBaseMBF):
        def __init__(self, block, K):
            super().__init__(block, K)

            self._X = allocate_gaussian_random_variable(self.K, self.block.N)

            # malloc
            if self.block.estimate_output:
                self.memory['Yt'] = allocate_gaussian_random_variable(self.K, self.block.L)
            if self.block.save_outlier_estimate:
                self.memory['S'] = allocate_gaussian_random_variable(self.K, self.block.L)

            # algorithm memory
            self._X_fw = allocate_gaussian_random_variable(self.K, self.block.N)
            self._initialize_Z()
            self._initialize_S()
            self._Znew = np.zeros((self.block.L,self.block.L))


        def _initialize_Z(self):
            self._Z = allocate_gaussian_random_variable(1, self.block.L)[0]
            self._Z.m.fill(0)
            self._check_and_set_sigma2_init(self._Z, self.block.L)

        def _initialize_S(self):
            self.outlier = np.zeros(self.K)
            self.n_outlier = 0
            self._S = allocate_gaussian_random_variable(self.K, self.block.L)
            self._S.m.fill(0)
            self._check_and_set_sigma2_k_init(self._S, self.K, self.block.L)

        def propagate_forward(self, k, msg_fw):
            C = self.block.C
            CT = C.T
            y = self.block.y[k]
            Z = self._Z
            S = self._S[k]

            G = np.linalg.inv(Z.V + S.V + C @ msg_fw.V @ CT)  # V.3
            msg_fw.m += msg_fw.V @ CT @ G @ (y - C @ msg_fw.m)  # V.1
            msg_fw.V -= msg_fw.V @ CT @ G @ C @ msg_fw.V  # V.2

            self._X_fw[k] = msg_fw  # for backward propagation
            self._X[k] = msg_fw  # IV.9 & IV.13 (forward update with temporary result)

        def propagate_backward(self, k, msg_bw):
            C = self.block.C
            CT = C.T
            y = self.block.y[k]
            Z = self._Z
            S = self._S[k]
            X_fw = self._X_fw[k]
            X = self._X[k]
            out_fac = self.block.outlier_threshold_factor
            # print(k, (Z.V, S.V), self.n_outlier)

            Y_W = np.linalg.inv(Z.V + S.V) #  IV.4   !!! not sure if correct
            F = np.eye(self.block.N) - X_fw.V @ CT @ Y_W @ C  # V.8

            msg_bw.xi[:] = F.T @ msg_bw.xi + CT @ Y_W @ (C @ X_fw.m - y)  # V.4
            msg_bw.W[:] = F.T @ msg_bw.W @ F + CT @ Y_W @ C @ F  # V.6

            # (backward update based on temporary result from forward-path)
            X.m -= X.V @ msg_bw.xi  # IV.9 (backward update)
            X.V -= X.V @ msg_bw.W @ X.V  # IV.13 (backward update)

            # outlier estimation
            # ------------------

            # A. Expectation Step
            X_mu_II = X.V + np.outer(X.m, X.m)

            # B. Maximization Step
            _tmp = y**2-2*np.dot(y, C@X.m) + C@X_mu_II@CT  # EQ 20
            S.V[:] = max(_tmp - Z.V, 0)  # EQ 20
            self._S[k].V = S.V

            # D. Noise Floor Estimation:
            if S.V <= out_fac * self.block.sigma2_init:
                self._Znew += _tmp
                # self.outlier[k] = 0
            else:
                # self.outlier[k] = 1
                self.n_outlier += 1
            if k == 0:
                self._Z.V = self._Znew / (self.K - self.n_outlier)
                self.n_outlier = 0

        def propagate_backward_save_states(self, k, msg_bw):

            super().propagate_backward_save_states(k, msg_bw)

            if self.block.estimate_output:
                C = self.block.C
                X = self.memory['X'][k]
                self.memory['Yt'][k].m = C @ X.m  # IV.9 (backward update)
                self.memory['Yt'][k].V = C @ X.V @ C.T  # IV.13 (backward update)

            if self.block.save_outlier_estimate:
                self.memory['S'][k] = self._S[k]

        def get_output_estimate(self):
            """
            Returns the output_estimate `\tilde{Y}` of the output block

            Returns
            -------
            out : np.ndarray of shape (K, ...)
                updated output estimate of forward path of the mp_block.
                `Y.m` calls the mean of shape (K, N) and `Y.V` calls the covariance of shape (K, N, N)
            """

            assert self.block.estimate_output, f"No output estimate saved! Set estimate_output=True on {self.block}"
            return self.memory['Yt']

        def get_Yt(self):
            """
            See :func:`get_output_estimate`
            """
            return self.get_output_estimate()

        def get_outlier_estimate(self):
            """
            Returns the outlier estimate `S` of the output block

            Returns
            -------
            out : np.ndarray of shape (K, ...)
                updated outlier estimate of forward path of the mp_block.
                `Y.m` calls the mean of shape (K, N) and `Y.V` calls the covariance of shape (K, N, N)
            """

            assert self.block.save_outlier_estimate, f"No outlier estimate saved! Set save_outlier_estimate=True on {self.block}"
            return self.memory['S']

        def get_S(self):
            """
            See :func:`get_outlier_estimate`
            """
            return self.get_outlier_estimate()

    class BlockContainer(BlockBaseMBF):
        """
        MBF Block Container

        Details in [Loeliger2016]_
        --------------------------
        - TABLE IV GAUSSIAN SINGLE-EDGE MARGINALS (m, V ) AND THEIR DUALS (ξ ̃, W ̃ ). (marginal and input/output estimation)
        """
        def __init__(self, block, K):
            super().__init__(block, K)

            self.mp_blocks = []
            for block in block.blocks:
                self.mp_blocks.append(MBF.create_mp_block(block, K))

        def propagate_forward(self, k, msg_fw):
            for mp_block in self.mp_blocks:
                mp_block.propagate_forward(k, msg_fw)

        def propagate_backward(self, k, msg_bw):
            for mp_block in reversed(self.mp_blocks):
                mp_block.propagate_backward(k, msg_bw)

        def propagate_forward_save_states(self, k, msg_fw):
            for mp_block in self.mp_blocks:
                mp_block.propagate_forward_save_states(k, msg_fw)

            if self.block.save_marginals:
                self.memory['X'][k] = msg_fw  # IV.9 & IV.13 (forward update)

        def propagate_backward_save_states(self, k, msg_bw):

            if self.block.save_marginals:
                VF_X = self.memory['X'][k].V
                self.memory['X'][k].m -= VF_X @ msg_bw.xi  # IV.9 (backward update)
                self.memory['X'][k].V -= VF_X @ msg_bw.W @ VF_X  # IV.13 (backward update)

            for mp_block in reversed(self.mp_blocks):
                mp_block.propagate_backward_save_states(k, msg_bw)

        def get_block_by_label(self, label, out_dict):
            if self.block.label == label:
                out_dict['mp_block'] = self
            else:
                for mp_block in self.mp_blocks:
                    mp_block.get_block_by_label(label, out_dict)

        def get_block_by_obj(self, obj, out_dict):
            if self.block is obj:
                out_dict['mp_block'] = self
            else:
                for mp_block in self.mp_blocks:
                    mp_block.get_block_by_obj(obj, out_dict)
