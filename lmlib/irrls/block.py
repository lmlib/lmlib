from abc import ABC
from functools import partial

import numpy as np
from lmlib.utils.check import is_square, is_string, is_2dim, all_equal

__all__ = ['BlockContainer', 'BlockSystem',
           'BlockInput', 'BlockInput_k', 'BlockInputNUV',
           'BlockOutput', 'BlockOutputOutlier']


def allocate_gaussian_random_variable(K, N, str_mean='m', str_covariance='V'):
    return np.recarray((K,), dtype=[(str_mean, 'f8', (N,)), (str_covariance, 'f8', (N, N))])


class BlockBase(ABC):
    """
    Abstract base class for different Blocks

    label : string, optional
        Label of the block instance
    save_marginals : boolean, optional
        Marginals of the block are saved when True, else no memory will be allocated

    """

    def __init__(self, label='n/a', save_marginals=False):
        self.label = label
        self._save_marginals = save_marginals
        self._N = 0

    @property
    def label(self):
        """str : label of the block"""
        return self._label

    @label.setter
    def label(self, label):
        assert is_string(label)
        self._label = label

    @property
    def N(self):
        """int : model order"""
        return self._N

    @property
    def save_marginals(self):
        """bool : whether to save marginals of the block"""
        return self._save_marginals


class BlockContainer(BlockBase):
    """
    Block Container

    Inherits from the BlockBase class, but offers the option of collecting several blocks into a superblock.

    Parameters
    ----------
    blocks : array_like of BlockBase
        List of Blocks
    """

    def __init__(self, blocks, **kwargs):
        super().__init__(**kwargs)

        self._blocks = []
        for block in blocks:
            self.append_block(block)

        # check model dimensions integrity
        assert all_equal([block.N for block in self.blocks]), "model dimension doesnt match between blocks"
        self._N = self.blocks[0].N

    @property
    def blocks(self):
        """list : sub-blocks of the block"""
        return self._blocks

    def append_block(self, block):
        """
        Appends a block to the internal block list

        Parameters
        ----------
        block : BlockBase
            block to append

        """
        assert isinstance(block, BlockBase), 'block must be a BlockBase object'
        self._blocks.append(block)


class BlockSystem(BlockBase):
    """
    Block System A (multiplier node)

    .. image:: /static/lmlib/irrls/BlockSystem.svg
        :width: 300
        :align: center

    Parameters
    ----------
    A : array_like of shape (N, N)
        System Matrix
    **kwargs
        Forwarded to :class:`.Block`
    """

    def __init__(self, A, **kwargs):
        super().__init__(**kwargs)
        self.A = A

    @property
    def A(self):
        """np.ndarray : State Transition Matrix"""
        return self._A

    @A.setter
    def A(self, A):
        assert is_square(A)
        self._A = np.asarray(A)
        self._N = self._A.shape[0]


class BlockInput(BlockBase):
    """
    Block Input Normal Prior

    .. image:: /static/lmlib/irrls/BlockInput.svg
        :height: 400
        :align: center

    Parameters
    ----------
    B : array_like of shape (N, M)
        Input Matrix
    sigma2_init : scalar or array_like of shape (M, M), optional
        Variance of zero-mean Gaussian, default=1.0
    estimate_input : bool, optional
        Enables input estimation :math:`U`, default=False
    **kwargs
        Forwarded to :class:`.Block`
    """

    def __init__(self, B, sigma2_init=1.0, estimate_input=False, **kwargs):
        super().__init__(**kwargs)
        self.B = B
        self._estimate_input = estimate_input
        self._sigma2_init = sigma2_init

    @property
    def B(self):
        """np.ndarray : Input Matrix"""
        return self._B

    @B.setter
    def B(self, B):
        assert is_2dim(B)
        self._B = np.asarray(B)
        self._N = self._B.shape[0]
        self._M = self._B.shape[1]

    @property
    def M(self):
        """int : input order"""
        return self._M

    @property
    def estimate_input(self):
        """bool : whether to estimate inputs"""
        return self._estimate_input

    @property
    def sigma2_init(self):
        """float, array_like : initial variance ot covariance"""
        return self._sigma2_init


class BlockInput_k(BlockInput):
    """
    Block input with time variable :math:`\sigma^2` (co-variance)

    .. image:: /static/lmlib/irrls/BlockInput_k.svg
        :height: 400
        :align: center

    Parameters
    ----------
    B : array_like of shape (N, M)
        Input Matrix
    sigma2_init : array_like of shape ([K,], M, M)
        Co-/Variance of zero-mean Gaussian
    **kwargs
        Forwarded to :class:`.InputBlock`
    """
    def __init__(self, B, sigma2_init, **kwargs):
        super().__init__(B, sigma2_init, **kwargs)


class BlockInputNUV(BlockInput_k):
    """
    Block input with NUV Prior

    Input block with a normal distribution with unknown variance (NUV).
    The EM algorithm method searches recursively for the least square solution and such estimate the prior variance.

    .. image:: /static/lmlib/irrls/BlockInputNUV.svg
        :height: 400
        :align: center


    Parameters
    ----------
    B : array_like of shape (N, M)
        Input Matrix
    sigma2_init : array_like of shape ([K,], M, M)
        Co-/Variance of zero-mean Gaussian
    save_deployed_sigma2 : bool, optional
        Whether to save the deployed (not updated) variance before update. Necessary to calculate the cost.
    **kwargs
        Forwarded to :class:`.InputBlock_k`

    """
    
    def __init__(self, B, sigma2_init, save_deployed_sigma2=False, **kwargs):
        super().__init__(B, sigma2_init, **kwargs)
        self._save_deployed_sigma2 = save_deployed_sigma2

    @property
    def save_deployed_sigma2(self):
        """bool : whether to save the deployed (not updated) variance"""
        return self._save_deployed_sigma2


class BlockOutput(BlockBase):
    """
    Block Output with additive noise

    .. image:: /static/lmlib/irrls/BlockOutput.svg
        :height: 400
        :align: center

    Parameters
    ----------
    C : array_like of shape (N, N)
        Output Matrix
    sigma2_init : scalar, array_like of shape (L, L)
        Variance of zero-mean Gaussian
    y : array_like of shape (K,[L])
        Observed Signal
    estimate_output : bool, optional
        Enables output estimation :math:`\tilde{Y}`, default=False
    **kwargs
        Forwarded to :class:`.Block`
    """

    def __init__(self, C, sigma2_init, y, estimate_output=False, **kwargs):
        super().__init__(**kwargs)
        self.C = C
        self._sigma2_init = sigma2_init
        self.y = y
        self._estimate_output = estimate_output
        self._save_marginals |= estimate_output

    @property
    def C(self):
        """np.ndarray : Output Matrix of shape (L, N)"""
        return self._C

    @C.setter
    def C(self, C):
        assert is_2dim(C)
        self._C = np.asarray(C)
        self._N = self._C.shape[1]
        self._L = self._C.shape[0]

    @property
    def L(self):
        """int : output order"""
        return self._L

    @property
    def estimate_output(self):
        """bool : output estimation status"""
        return self._estimate_output

    @property
    def sigma2_init(self):
        """float, array_like : initial variance ot covariance"""
        return self._sigma2_init

    @property
    def y(self):
        """np.ndarray : Observed Signal of shape (K, [L])"""
        return self._y

    @y.setter
    def y(self, y):
        if self.L > 1:
            assert np.ndim(y) == 2 and np.shape(y)[1] == self.L, "Shape of y must be compatible with C."
        self._y = np.asarray(y)


class BlockOutputOutlier(BlockOutput):
    def __init__(self, C, sigma2_init, y, outlier_threshold_factor=10, iterations=10, **kwargs):
        super().__init__(C, sigma2_init, y, **kwargs)
        self._outlier_threshold_factor = outlier_threshold_factor
        self._iterations = iterations

    @property
    def outlier_threshold_factor(self):
        return self._outlier_threshold_factor

    @property
    def iterations(self):
        return self._iterations
