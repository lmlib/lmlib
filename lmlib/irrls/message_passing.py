"""
This module implements the MassagePassing Algorithms (MBF, BIFM)

Switching between MBF and BFIM in the FactorGraph is done using the state design pattern.
MassagePassing is the state and FactorGraph the context


"""

from abc import ABC, abstractmethod
import numpy as np

__all__ = ['init_fw_msg', 'init_bw_msg',
           'MBF_MessagePassingBase', 'MBF_SectionSystem', 'MBF_SectionContainer',
           'MBF_SectionInput', 'MBF_SectionInput_k',
           'MBF_SectionOutput',
           'MBF_SectionInput_NUV', 'MBF_SectionInput_sNUV',
           'MBF_SectionInput_L1', 'MBF_SectionInput_sL1',
           'MBF_SectionInput_Binary',
           'BIFM_MessagePassingBase', 'BIFM_SectionContainer',
           'BIFM_SectionSystem',
           'BIFM_SectionInput', 'BIFM_SectionInput_k',
           'BIFM_SectionOutput']


def random_var(mean, covariance, str_mean, str_covariance):
    msg = array_random_var(1, np.shape(covariance)[0], str_mean, str_covariance)
    msg[str_mean] = mean
    msg[str_covariance] = covariance
    return msg[0]


def array_random_var(K, N, str_mean, str_covariance):
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


def init_fw_msg(N, prior):
    """
    Returns the initial state at k=0 for the message passing algorithm for the forward recursion

    Parameters
    ----------
    N : int
        dimension of state
    prior : tuple of length 2
        m : float, array_like of shape (N,)
            value of the initial mean vector
        V float, array_like of shape (N, N)
            value of the initial covariance matrix. if `V` is scalar the diagnoal will be filled with `V`

    Returns
    -------
    out : :class:`~numpy.recarray`
        Single Message Passing State
    """
    mean = np.full((N,), prior[0]) if np.isscalar(prior[0]) else prior[0]
    covariance = np.eye(N).dot(prior[1])
    return random_var(mean, covariance, 'm', 'V')


def init_bw_msg(N, prior):
    """
    Returns the initial state at k=K-1 for the message passing algorithm for the backward recursion

    Parameters
    ----------
    N : int
        dimension of state
    prior : tuple of length 2
        xi : float, array_like of shape (N,)
            value of the initial precision matrix weighted mean vector
        W : float, array_like of shape (N, N)
            value of the initial precision matrix. if `W` is scalar the diagnoal will be filled with `W`

    Returns
    -------
    out : :class:`~numpy.recarray`
        Single Message Passing State
    """
    mean = np.full((N,), prior[0]) if np.isscalar(prior[0]) else prior[0]
    covariance = np.eye(N).dot(prior[1])
    return random_var(mean, covariance, 'xi', 'W')


class MBF_MessagePassingBase(ABC):
    """
    MBF Message Passing Base Class

    Each Section Types have its own Message Passing Class that implements forward and backward methods.
    Message Passing Classes owning the memory and will save the marginals into it if needed.
    Message Passing Classes with prior update inherit from :class:`.MBF_SectionInputUpdate`.


    Parameters
    ----------
        owner : :class:`~lmlib.irrls.section.SectionBase`
            Reference to Owner Section
        K : int
            Length of the Factor Graph
    """

    def __init__(self, owner, K):
        self._owner = owner
        self.memory = dict()
        self.identity_N = np.eye(self._owner.N)

        self._fw_save_state_callbacks = []
        self._bw_save_state_callbacks = []

        if self._owner.save_state_marginal:
            self.memory['X'] = array_random_var(K, self._owner.N, 'm', 'V')
            self._fw_save_state_callbacks.append(self._forward_save_state_marginal)
            self._bw_save_state_callbacks.append(self._backward_save_state_marginal)

    @abstractmethod
    def propagate_forward(self, k, fw_msg):
        """
        Propagate forward message

        Parameters
        ----------
        k : int
            time index
        fw_msg : :class:`~numpy.recarray`
            forward message
                - :code:`fw_msg.m` (mean)
                - :code:`fw_msg.V` (co-variance)
        """
        pass

    @abstractmethod
    def propagate_backward(self, k, bw_msg):
        """
        Propagate backward message

        Parameters
        ----------
        k : int
            time index
        bw_msg : :class:`~numpy.recarray`
            backward message
                - :code:`bw_msg.xi` (inverse co-variance weighted mean)
                - :code:`fw_msg.W` (inverse co-variance)
        """
        pass

    def propagate_forward_save_states(self, k, fw_msg):
        """
        Propagate the forward message and then runs backward callback functions to save marginals.

        Parameters
        ----------
        k : int
            time index
        fw_msg : :class:`~numpy.recarray`
            forward message
                - :code:`fw_msg.m` (mean)
                - :code:`fw_msg.V` (co-variance)
        """

        self.propagate_forward(k, fw_msg)
        for f in self._fw_save_state_callbacks:
            f(k, fw_msg)

    def propagate_backward_save_states(self, k, bw_msg):
        """
        Runs backward callback functions to save marginals and then propagate the backward message.

        Parameters
        ----------
        k : int
            time index
        bw_msg : :class:`~numpy.recarray`
            backward message
                - :code:`bw_msg.xi` (inverse co-variance weighted mean)
                - :code:`fw_msg.W` (inverse co-variance)
        """

        for f in self._bw_save_state_callbacks:
            f(k, bw_msg)
        self.propagate_backward(k, bw_msg)

    def _forward_save_state_marginal(self, k, fw_msg):
        # (forward update with temporary result)
        self.memory['X'][k] = fw_msg  # IV.9 & IV.13

    def _backward_save_state_marginal(self, k, bw_msg):

        VF_X = self.memory['X'][k].V
        # (backward update based on temporary result from forward-path)
        self.memory['X'][k].m -= VF_X @ bw_msg.xi  # IV.9
        self.memory['X'][k].V -= VF_X @ bw_msg.W @ VF_X  # IV.13


class MBF_SectionContainer(MBF_MessagePassingBase):
    """
    MBF section Container

    .. image:: /static/lmlib/irrls/irrls-MP_SectionContainer.svg
        :height: 200
        :align: center

    Details in [Loeliger2016]_
    --------------------------
    - TABLE IV GAUSSIAN SINGLE-EDGE MARGINALS (m, V ) AND THEIR DUALS (ξ̃, W̃ ). (marginal and input/output estimation)
    """

    def propagate_forward(self, k, fw_msg):
        for s in self._owner.subsections:
            s.mp.propagate_forward(k, fw_msg)

    def propagate_backward(self, k, bw_msg):
        for s in reversed(self._owner.subsections):
            s.mp.propagate_backward(k, bw_msg)

    def propagate_forward_save_states(self, k, fw_msg):
        for f in self._fw_save_state_callbacks:
            f(k, fw_msg)
        for s in self._owner.subsections:
            s.mp.propagate_forward_save_states(k, fw_msg)

    def propagate_backward_save_states(self, k, bw_msg):
        for f in self._bw_save_state_callbacks:
            f(k, bw_msg)
        for s in reversed(self._owner.subsections):
            s.mp.propagate_backward_save_states(k, bw_msg)


class MBF_SectionSystem(MBF_MessagePassingBase):
    """MBF System Section"""

    def __init__(self, owner, K):
        super().__init__(owner, K)
        self.A = owner.A
        self.AT = self.A.T

    def propagate_forward(self, k, fw_msg):
        fw_msg.m[:] = self.A @ fw_msg.m  # III.1
        fw_msg.V[:] = self.A @ fw_msg.V @ self.AT  # III.2

    def propagate_backward(self, k, bw_msg):
        bw_msg.xi[:] = self.AT @ bw_msg.xi  # II.6, III.7
        bw_msg.W[:] = self.AT @ bw_msg.W @ self.A  # II.7, III.8


class MBF_SectionInputBase(MBF_MessagePassingBase, ABC):
    """MBF Input Section Base Class"""

    def __init__(self, owner, K):
        super().__init__(owner, K)
        self.fw_U = None
        self.B = self._owner.B
        self.BT = self.B.T
        self.identity_M = np.eye(self._owner.M)

        if self._owner.save_input_marginal:
            self.memory['U'] = array_random_var(K, self._owner.M, 'm', 'V')
            self._fw_save_state_callbacks.append(self._forward_save_input_marginal)
            self._bw_save_state_callbacks.append(self._backward_save_input_marginal)

    def _forward_save_input_marginal(self, k, fw_msg):
        pass

    def _backward_save_input_marginal(self, k, bw_msg):
        pass

    def propagate_backward(self, k, bw_msg):
        pass  # II.6, II.7  # nothing to do


class MBF_SectionInput(MBF_SectionInputBase):
    """MBF Input Section"""

    def __init__(self, owner, K):
        super().__init__(owner, K)
        self.fw_U_V = self.identity_M.dot(self._owner.sigma2)

    def propagate_forward(self, k, fw_msg):
        fw_msg.V += self.B @ self.fw_U_V @ self.BT  # II.2, III.2

    def _backward_save_input_marginal(self, k, bw_msg):
        fw_U_V = self.fw_U_V
        self.memory['U'][k].m = - fw_U_V @ self.BT @ bw_msg.xi  # III.7 & IV.9
        self.memory['U'][k].V = fw_U_V - fw_U_V @ self.BT @ bw_msg.W @ self.B @ fw_U_V  # III.8 & IV.13


class MBF_SectionInput_k(MBF_SectionInputBase):
    """MBF Input Section for Variable Co-Variances over Time"""

    def __init__(self, owner, K):
        super().__init__(owner, K)
        self.fw_U_V = np.asarray([self.identity_M.dot(s_) for s_ in self._owner.sigma2])

    def propagate_forward(self, k, fw_msg):
        fw_msg.V += self.B @ self.fw_U_V[k] @ self.BT  # II.2, III.2

    def _backward_save_input_marginal(self, k, bw_msg):
        fw_U_V = self.fw_U_V[k]
        self.memory['U'][k].m = - fw_U_V @ self.BT @ bw_msg.xi  # III.7 & IV.9
        self.memory['U'][k].V = fw_U_V - fw_U_V @ self.BT @ bw_msg.W @ self.B @ fw_U_V  # III.8 & IV.13


class MBF_SectionOutput(MBF_MessagePassingBase):
    """MBF Output Section"""

    def __init__(self, owner, K):
        super().__init__(owner, K)

        self.C = self._owner.C
        self.CT = self.C.T
        self.bw_Y_m = self._owner.y
        self.fw_X = array_random_var(K, self._owner.N, 'm', 'V')  # buffer
        self.identity_L = np.eye(self._owner.L)
        self.fw_Z_V = self.identity_L.dot(self._owner.sigma2)
        self.bw_Yt_W = np.linalg.inv(self.fw_Z_V)

        if self._owner.save_output_marginal:
            self.memory['Y_tilde'] = array_random_var(K, self._owner.L, 'm', 'V')
            self._fw_save_state_callbacks.append(self._forward_save_output_marginal)
            self._bw_save_state_callbacks.append(self._backward_save_output_marginal)

    def propagate_forward(self, k, fw_msg):
        # observation block
        G = np.linalg.inv(self.fw_Z_V + self.C @ fw_msg.V @ self.CT)  # V.3
        fw_msg.m += fw_msg.V @ self.CT @ G @ (self.bw_Y_m[k] - self.C @ fw_msg.m)  # V.1
        fw_msg.V -= fw_msg.V @ self.CT @ G @ self.C @ fw_msg.V  # V.2

        # save state for backward recursion
        self.fw_X[k].m = fw_msg.m
        self.fw_X[k].V = fw_msg.V

    def propagate_backward(self, k, bw_msg):
        # recall state from forward recursion
        fw_X_m = self.fw_X[k].m
        fw_X_V = self.fw_X[k].V

        # equality constraint
        F = self.identity_N - fw_X_V @ self.CT @ self.bw_Yt_W @ self.C  # V.8
        bw_msg.xi = F.T @ bw_msg.xi + self.CT @ self.bw_Yt_W @ (self.C @ fw_X_m - self.bw_Y_m[k])  # V.4
        bw_msg.W = F.T @ bw_msg.W @ F + self.CT @ self.bw_Yt_W @ self.C @ F  # V.6

    def _forward_save_output_marginal(self, k, fw_msg):
        pass

    def _backward_save_output_marginal(self, k, bw_msg):
        # recall state from forward recursion
        fw_X_m = self.fw_X[k].m
        fw_X_V = self.fw_X[k].V
        self.memory['Y_tilde'][k].m = self.C @ (fw_X_m - fw_X_V @ bw_msg.xi)  # IV.9 & III.5
        self.memory['Y_tilde'][k].V = self.C @ (fw_X_V - fw_X_V @ bw_msg.W @ fw_X_V) @ self.CT  # IV.13 & III.6


class MBF_SectionInputUpdate(MBF_SectionInputBase, ABC):
    """MBF Input Section Update Base Class"""

    def __init__(self, owner, K):
        super().__init__(owner, K)

        if self._owner.update_method == 'EM':
            self._forward_update = self._forward_update_EM
            self._backward_update = self._backward_update_EM
        if self._owner.update_method == 'AM':
            self._forward_update = self._forward_update_AM
            self._backward_update = self._backward_update_AM

    def _forward_update_EM(self, k, fw_msg):
        """Forward update for EM Algorithm"""
        pass

    def _backward_update_EM(self, k, bw_msg):
        """Backward update for EM Algorithm"""
        pass

    def _forward_update_AM(self, k, fw_msg):
        """Forward update for AM Algorithm"""
        pass

    def _backward_update_AM(self, k, bw_msg):
        """Backward update for AM Algorithm"""
        pass


class MBF_SectionInput_NUV(MBF_SectionInputUpdate):
    """MBF Input Section with NUV-Prior"""

    def __init__(self, owner, K):
        super().__init__(owner, K)
        self.memory['sigma2'] = array_random_var(K, self._owner.M, 'm', 'V')
        self.fw_U_V = np.zeros((K, self._owner.M, self._owner.M))
        self.sigma2 = np.zeros((K, self._owner.M, self._owner.M))
        self.sigma2[:] = self.identity_M * self._owner.sigma2_init
        self.beta_inv = 1 / self._owner.beta

    def propagate_forward(self, k, fw_msg):
        self.fw_U_V[k] = self.sigma2[k]
        fw_msg.V += self.B @ self.fw_U_V[k] @ self.BT  # II.2, III.2

        self._forward_update(k, fw_msg)

    def propagate_backward(self, k, bw_msg):
        self._backward_update(k, bw_msg)

    def _backward_save_input_marginal(self, k, bw_msg):
        fw_U_V = self.fw_U_V[k]
        self.memory['U'][k].m = - fw_U_V @ self.BT @ bw_msg.xi  # III.7 & IV.9
        self.memory['U'][k].V = fw_U_V - fw_U_V @ self.BT @ bw_msg.W @ self.B @ fw_U_V  # III.8 & IV.13

    def _backward_update_EM(self, k, bw_msg):
        fw_U_V = self.fw_U_V[k]
        U_m = - fw_U_V @ self.BT @ bw_msg.xi
        U_V = fw_U_V - fw_U_V @ self.BT @ bw_msg.W @ self.B @ fw_U_V

        # Update Input Variance
        self.sigma2[k] = np.outer(U_m, U_m) + U_V

    def _backward_update_AM(self, k, bw_msg):
        U_m = -  self.fw_U_V[k] @ self.BT @ bw_msg.xi

        # Update Input Variance
        self.sigma2[k] = np.outer(U_m, U_m) * self.beta_inv


class MBF_SectionInput_sNUV(MBF_SectionInput_NUV):

    def __init__(self, owner, K):
        super().__init__(owner, K)
        self.r2 = self._owner.r2

    def propagate_forward(self, k, fw_msg):
        self.fw_U_V[k] = self.sigma2[k]
        fw_msg.V += self.B @ self.fw_U_V[k] @ self.BT  # II.2, III.2

        self._forward_update(k, fw_msg)

    def propagate_backward(self, k, bw_msg):
        self._backward_update(k, bw_msg)

    def _backward_save_input_marginal(self, k, bw_msg):
        fw_U_V = self.fw_U_V[k]
        self.memory['U'][k].m = - fw_U_V @ self.BT @ bw_msg.xi  # III.7 & IV.9
        self.memory['U'][k].V = fw_U_V - fw_U_V @ self.BT @ bw_msg.W @ self.B @ fw_U_V  # III.8 & IV.13

    def _backward_update_EM(self, k, bw_msg):
        fw_U_V = self.fw_U_V[k]
        U_m = - fw_U_V @ self.BT @ bw_msg.xi
        U_V = fw_U_V - fw_U_V @ self.BT @ bw_msg.W @ self.B @ fw_U_V

        # Update Input Variance
        self.sigma2[k] = max(self.r2, np.outer(U_m, U_m) + U_V)

    def _backward_update_AM(self, k, bw_msg):
        U_m = -  self.fw_U_V[k] @ self.BT @ bw_msg.xi

        # Update Input Variance
        self.sigma2[k] = max(self.r2, np.outer(U_m, U_m) * self.beta_inv)


class MBF_SectionInput_L1(MBF_SectionInput_NUV):
    def _backward_update_EM(self, k, bw_msg):
        fw_U_V = self.fw_U_V[k]
        U_m = - fw_U_V @ self.BT @ bw_msg.xi
        U_V = fw_U_V - fw_U_V @ self.BT @ bw_msg.W @ self.B @ fw_U_V

        # Update Input Variance
        self.sigma2[k] = np.sqrt(np.outer(U_m, U_m) + U_V)

    def _backward_update_AM(self, k, bw_msg):
        U_m = -  self.fw_U_V[k] @ self.BT @ bw_msg.xi

        # Update Input Variance
        self.sigma2[k] = np.sqrt(np.outer(U_m, U_m)) * self.beta_inv


class MBF_SectionInput_sL1(MBF_SectionInput_sNUV):

    def _backward_update_EM(self, k, bw_msg):
        fw_U_V = self.fw_U_V[k]
        U_m = - fw_U_V @ self.BT @ bw_msg.xi
        U_V = fw_U_V - fw_U_V @ self.BT @ bw_msg.W @ self.B @ fw_U_V

        # Update Input Variance
        self.sigma2[k] = max(self.r2, np.sqrt(np.outer(U_m, U_m) + U_V))

    def _backward_update_AM(self, k, bw_msg):
        U_m = -  self.fw_U_V[k] @ self.BT @ bw_msg.xi

        # Update Input Variance
        self.sigma2[k] = max(self.r2, np.sqrt(np.outer(U_m, U_m)) * self.beta_inv)


class MBF_SectionInput_Binary(MBF_SectionInputUpdate):

    def __init__(self, owner, K):
        super().__init__(owner, K)
        # self.memory['sigma2a'] = array_random_var(K, self._owner.M, 'm', 'V')
        # self.memory['sigma2b'] = array_random_var(K, self._owner.M, 'm', 'V')
        self.fw_U = array_random_var(K, self._owner.M, 'm', 'V')
        self.a = self._owner.a
        self.b = self._owner.b
        self.sigma2a = np.zeros((K, self._owner.M, self._owner.M))
        self.sigma2b = np.zeros((K, self._owner.M, self._owner.M))
        self.sigma2a[:] = self.identity_M * self._owner.sigma2a_init
        self.sigma2b[:] = self.identity_M * self._owner.sigma2b_init
        self.beta_inv = 1 / self._owner.beta

    def propagate_forward(self, k, fw_msg):
        inv_s2a = 1 / self.sigma2a[k]
        inv_s2b = 1 / self.sigma2b[k]
        self.fw_U[k].V = 1 / (inv_s2a + inv_s2b)  # (10.11)
        self.fw_U[k].m = self.fw_U[k].V * (self.a * inv_s2a + self.b * inv_s2b)  # (10.12)

        fw_msg.V += self.B @ self.fw_U[k].V @ self.BT  # II.2, III.2
        fw_msg.m += self.B @ self.fw_U[k].m  # II.1 & III.1

        self._forward_update(k, fw_msg)

    def propagate_backward(self, k, bw_msg):
        self._backward_update(k, bw_msg)

    def _backward_save_input_marginal(self, k, bw_msg):
        fw_U_V = self.fw_U[k].V
        fw_U_m = self.fw_U[k].m
        self.memory['U'][k].m = fw_U_m - fw_U_V @ self.BT @ bw_msg.xi  # III.7 & IV.9
        self.memory['U'][k].V = fw_U_V - fw_U_V @ self.BT @ bw_msg.W @ self.B @ fw_U_V  # III.8 & IV.13

    def _backward_update_EM(self, k, bw_msg):
        fw_U_V = self.fw_U[k].V
        fw_U_m = self.fw_U[k].m

        U_m = fw_U_m - fw_U_V @ self.BT @ bw_msg.xi  # III.7 & IV.9
        U_V = fw_U_V - fw_U_V @ self.BT @ bw_msg.W @ self.B @ fw_U_V  # III.8 & IV.13

        # Update Input Variance
        self.sigma2a[k] = U_V + (U_m - self.a) ** 2  # (10.19)
        self.sigma2b[k] = U_V + (U_m - self.b) ** 2  # (10.20)

    def _backward_update_AM(self, k, bw_msg):
        fw_U_V = self.fw_U[k].V
        fw_U_m = self.fw_U[k].m

        U_m = fw_U_m - fw_U_V @ self.BT @ bw_msg.xi

        # Update Input Variance
        self.sigma2a[k] = self.beta_inv * (U_m - self.a) ** 2
        self.sigma2b[k] = self.beta_inv * (U_m - self.b) ** 2


class BIFM_MessagePassingBase(ABC):

    def __init__(self, owner, K):
        self._owner = owner
        self.memory = dict()
        self.identity_N = np.eye(self._owner.N)

        self._fw_save_state_callbacks = []
        self._bw_save_state_callbacks = []

        if self._owner.save_state_marginal:
            self.memory['X'] = array_random_var(K, self._owner.N, 'm', 'V')
            self._fw_save_state_callbacks.append(self._forward_save_state_marginal)
            self._bw_save_state_callbacks.append(self._backward_save_state_marginal)

    @abstractmethod
    def propagate_forward(self, k, fw_msg):
        pass

    @abstractmethod
    def propagate_backward(self, k, bw_msg):
        pass

    def propagate_forward_save_states(self, k, fw_msg):
        self.propagate_forward(k, fw_msg)
        for f in self._fw_save_state_callbacks:
            f(k, fw_msg)

    def propagate_backward_save_states(self, k, bw_msg):
        for f in self._bw_save_state_callbacks:
            f(k, bw_msg)
        self.propagate_backward(k, bw_msg)

    def _forward_save_state_marginal(self, k, fw_msg):
        self.memory['X'][k] = fw_msg  # IV.9 & IV.13

    def _backward_save_state_marginal(self, k, bw_msg):
        pass


class BIFM_SectionContainer(BIFM_MessagePassingBase):
    """
    MBF section Container

    .. image:: /static/lmlib/irrls/irrls-MP_SectionContainer.svg
        :height: 200
        :align: center

    Details in [Loeliger2016]_
    --------------------------
    - TABLE IV GAUSSIAN SINGLE-EDGE MARGINALS (m, V ) AND THEIR DUALS (ξ̃, W̃ ). (marginal and input/output estimation)
    """

    def propagate_forward(self, k, fw_msg):
        for s in self._owner.subsections:
            s.mp.propagate_forward(k, fw_msg)

    def propagate_backward(self, k, bw_msg):
        for s in reversed(self._owner.subsections):
            s.mp.propagate_backward(k, bw_msg)

    def propagate_forward_save_states(self, k, fw_msg):
        for f in self._fw_save_state_callbacks:
            f(k, fw_msg)
        for s in self._owner.subsections:
            s.mp.propagate_forward_save_states(k, fw_msg)

    def propagate_backward_save_states(self, k, bw_msg):
        for f in self._bw_save_state_callbacks:
            f(k, bw_msg)
        for s in reversed(self._owner.subsections):
            s.mp.propagate_backward_save_states(k, bw_msg)


class BIFM_SectionSystem(BIFM_MessagePassingBase):
    """
   BIFM System section

   .. image:: /static/lmlib/irrls/irrls-MP_SectionSystem.svg
       :height: 200
       :align: center

   Parameters
   ----------
   owner : :class:`~lmlib.SectionSystem`
       Reference to the object owner
   """

    def __init__(self, owner, K):
        super().__init__(owner, K)
        self.A = owner.A
        self.AT = self.A.T

    def propagate_forward(self, k, fw_msg):
        fw_msg.m[:] = self.A @ fw_msg.m  # III.5
        fw_msg.V[:] = self.A @ fw_msg.V @ self.AT  # III.6

    def propagate_backward(self, k, bw_msg):
        bw_msg.xi[:] = self.AT @ bw_msg.xi  # III.3
        bw_msg.W[:] = self.AT @ bw_msg.W @ self.A  # III.4


class BIFM_SectionInputBase(BIFM_MessagePassingBase, ABC):
    """
    Abstract Input Section Class

    """

    def __init__(self, owner, K):
        super().__init__(owner, K)
        self.bw_X = array_random_var(K, self._owner.N, 'xi', 'W')  # buffer
        self.B = self._owner.B
        self.BT = self.B.T
        self.identity_M = np.eye(self._owner.M)

        if self._owner.save_input_marginal:
            self.memory['U'] = array_random_var(K, self._owner.M, 'm', 'V')
            self._fw_save_state_callbacks.append(self._forward_save_input_marginal)
            self._bw_save_state_callbacks.append(self._backward_save_input_marginal)

    def _forward_save_input_marginal(self, k, fw_msg):
        pass

    def _backward_save_input_marginal(self, k, bw_msg):
        pass

    def propagate_backward(self, k, bw_msg):
        pass  # II.6, II.7  # nothing to do


class BIFM_SectionInput(BIFM_SectionInputBase):

    def __init__(self, owner, K):
        super().__init__(owner, K)
        self.fw_U_V = self.identity_M.dot(self._owner.sigma2)
        self.fw_U_W = self.identity_M.dot(1/self._owner.sigma2)

    def propagate_forward(self, k, fw_msg):
        # save state for forward recursion
        bw_X = self.bw_X[k]
        fw_U_V = self.fw_U_V

        # input block (needs bw_message X left input section)
        Ft = self.identity_N - bw_X.W @ self.B @ fw_U_V @ self.BT  # VI.8
        fw_msg.m[:] = Ft.T @ fw_msg.m + self.B @ fw_U_V @ (self.BT @ bw_X.xi)  # VI.4
        fw_msg.V[:] = Ft.T @ fw_msg.V @ Ft + self.B @ fw_U_V @ self.BT @ Ft  # VI.6

    def propagate_backward(self, k, bw_msg):
        # input block
        H = np.linalg.inv(self.fw_U_W + self.BT @ bw_msg.W @ self.B)  # VI.3
        bw_msg.xi += bw_msg.W @ self.B @ H @ (- self.B @ bw_msg.xi)  # VI.1
        bw_msg.W -= bw_msg.W @ self.B @ H @ self.BT @ bw_msg.W  # VI.2

        # save state for forward recursion
        self.bw_X[k].xi = bw_msg.xi
        self.bw_X[k].W = bw_msg.W

    def _forward_save_input_marginal(self, k, fw_msg):
        fw_U_V = self.fw_U_V
        bw_X = self.bw_X[k]

        U_xi_tilde = self.B @ (bw_X.W @ fw_msg.m - bw_X.xi)  # II.6 & III.7 & 23
        U_W_tilde = self.BT @ (bw_X.W - bw_X.W @ fw_msg.V @ bw_X.W) @ self.B  # II.7 & III.8 & IV.7
        self.memory['U'][k].m = 0 - fw_U_V @ U_xi_tilde  # IV.9
        self.memory['U'][k].V = fw_U_V - fw_U_V @ U_W_tilde @ fw_U_V  # IV.13


class BIFM_SectionInput_k(BIFM_SectionInputBase):

    def __init__(self, owner, K):
        super().__init__(owner, K)
        self.fw_U_V = np.asarray([self.identity_M.dot(s_) for s_ in self._owner.sigma2])
        self.fw_U_W = np.asarray([self.identity_M.dot(1 / s_) for s_ in self._owner.sigma2])

    def propagate_forward(self, k, fw_msg):
        # save state for forward recursion
        bw_X = self.bw_X[k]
        fw_U_V = self.fw_U_V[k]

        # input block (needs bw_message X left input section)
        Ft = self.identity_N - bw_X.W @ self.B @ fw_U_V @ self.BT  # VI.8
        fw_msg.m[:] = Ft.T @ fw_msg.m + self.B @ fw_U_V @ (self.BT @ bw_X.xi)  # VI.4
        fw_msg.V[:] = Ft.T @ fw_msg.V @ Ft + self.B @ fw_U_V @ self.BT @ Ft  # VI.6

    def propagate_backward(self, k, bw_msg):
        # input block
        H = np.linalg.inv(self.fw_U_W[k] + self.BT @ bw_msg.W @ self.B)  # VI.3
        bw_msg.xi += bw_msg.W @ self.B @ H @ (- self.B @ bw_msg.xi)  # VI.1
        bw_msg.W -= bw_msg.W @ self.B @ H @ self.BT @ bw_msg.W  # VI.2

        # save state for forward recursion
        self.bw_X[k].xi = bw_msg.xi
        self.bw_X[k].W = bw_msg.W

    def _forward_save_input_marginal(self, k, fw_msg):
        fw_U_V = self.fw_U_V[k]
        bw_X = self.bw_X[k]

        U_xi_tilde = self.B @ (bw_X.W @ fw_msg.m - bw_X.xi)  # II.6 & III.7 & 23
        U_W_tilde = self.BT @ (bw_X.W - bw_X.W @ fw_msg.V @ bw_X.W) @ self.B  # II.7 & III.8 & IV.7
        self.memory['U'][k].m = 0 - fw_U_V @ U_xi_tilde  # IV.9
        self.memory['U'][k].V = fw_U_V - fw_U_V @ U_W_tilde @ fw_U_V  # IV.13


class BIFM_SectionOutput(BIFM_MessagePassingBase):
    """BIFM Output Section"""

    def __init__(self, owner, K):
        super().__init__(owner, K)

        self.C = self._owner.C
        self.CT = self.C.T
        self.bw_Y_m = np.asarray(self._owner.y)[:, None]
        self.fw_X = array_random_var(K, self._owner.N, 'm', 'V')  # buffer
        self.identity_L = np.eye(self._owner.L)
        self.fw_Z_V = self.identity_L.dot(self._owner.sigma2)
        self.bw_Yt_W = np.linalg.inv(self.fw_Z_V)  # II.4

        if self._owner.save_output_marginal:
            self.memory['Y_tilde'] = array_random_var(K, self._owner.L, 'm', 'V')
            self._fw_save_state_callbacks.append(self._forward_save_output_marginal)
            self._bw_save_state_callbacks.append(self._backward_save_output_marginal)


    def propagate_forward(self, k, fw_msg):
        # nothing to do
        pass

    def propagate_backward(self, k, bw_msg):
        bw_msg.xi += self.CT @ (self.bw_Yt_W @ self.bw_Y_m[k])  # II.3 & III.3 &  EQ. 17 & I.3
        bw_msg.W += self.CT @ self.bw_Yt_W @ self.C  # III.4 & I.4

    def _forward_save_output_marginal(self, k, fw_msg):
        self.memory['Y_tilde'][k].m = self.C @ fw_msg.m  # III.5
        self.memory['Y_tilde'][k].V = self.C @ fw_msg.V @ self.CT  # III.6

    def _backward_save_output_marginal(self, k, fw_msg):
        pass
