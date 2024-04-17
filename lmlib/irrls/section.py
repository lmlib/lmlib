from abc import ABC
from functools import partial

import numpy as np
from lmlib.utils.check import is_square, is_string, is_2dim, all_equal

__all__ = ['SectionBase', 'SectionContainer', 'SectionSystem',
           'SectionInput', 'SectionInput_k',
           'SectionInputNUV',
           'SectionOutput', 'SectionOutputOutlier']


def allocate_gaussian_random_variable(K, N, str_mean='m', str_covariance='V'):
    return np.recarray((K,), dtype=[(str_mean, 'f8', (N,)), (str_covariance, 'f8', (N, N))])


class SectionBase(ABC):
    """
    Abstract base class of all sections

    label : string, optional
        Label of the section instance
    save_marginal : boolean, optional
        Marginal of the section are saved when True, else no memory will be allocated

    """

    def __init__(self, label='n/a', save_marginal=False):
        self.label = label
        self._save_marginal = save_marginal
        self._N = 0

    @property
    def label(self):
        """str : label of the section"""
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
    def save_marginal(self):
        """bool : whether to save marginal of the section"""
        return self._save_marginal


class SectionContainer(SectionBase):
    """
    Parents section containing a list of other sections

    Inherits from the SectionBase class, but offers the option of collecting several sections into a superSection.

    .. image:: /static/lmlib/irrls/irrls-SectionContainer.svg
        :height: 200
        :align: center

    Parameters
    ----------
    sections : array_like of SectionBase
        List of sections
    """

    def __init__(self, sections, **kwargs):
        super().__init__(**kwargs)

        self._subsections = []
        for section in sections:
            self.append_section(section)

        # check model dimensions integrity
        assert all_equal([section.N for section in self.subsections]), "model dimension doesnt match between sections"
        self._N = self.subsections[0].N

    @property
    def subsections(self):
        """list : sub-sections of the section"""
        return self._subsections

    def append_section(self, section):
        """
        Appends a section to the internal section list

        Parameters
        ----------
        section : SectionBase
            section to append

        """
        assert isinstance(section, SectionBase), 'Section must be a SectionBase object'
        self._subsections.append(section)


class SectionSystem(SectionBase):
    """
    Section System A (multiplier node)

    .. image:: /static/lmlib/irrls/irrls-SectionSystem.svg
        :height: 200
        :align: center

    Parameters
    ----------
    A : array_like of shape (N, N)
        System Matrix
    **kwargs
        Forwarded to :class:`.Section`
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


class SectionInput(SectionBase):
    """
    Section Input Normal Prior

    .. image:: /static/lmlib/irrls/irrls-SectionInput.svg
        :height: 300
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
        Forwarded to :class:`.Section`
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


class SectionInput_k(SectionInput):
    """
    Section input with time variable :math:`\sigma^2` (co-variance)

    .. image:: /static/lmlib/irrls/irrls-SectionInput_k.svg
        :height: 300
        :align: center

    Parameters
    ----------
    B : array_like of shape (N, M)
        Input Matrix
    sigma2_init : array_like of shape ([K,], M, M)
        Co-/Variance of zero-mean Gaussian
    **kwargs
        Forwarded to :class:`.InputSection`
    """
    def __init__(self, B, sigma2_init, **kwargs):
        super().__init__(B, sigma2_init, **kwargs)


class SectionInputNUV(SectionInput_k):
    """
    Section input with NUV Prior

    Input section with a normal distribution with unknown variance (NUV).
    The EM algorithm method searches recursively for the least square solution and such estimate the prior variance.

    .. image:: /static/lmlib/irrls/irrls-SectionInputNUV.svg
        :height: 300
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
        Forwarded to :class:`.InputSection_k`

    """
    
    def __init__(self, B, sigma2_init, prior_type='trivial', constraint=None, update_algo='EM', save_deployed_sigma2=False, **kwargs):
        super().__init__(B, sigma2_init, **kwargs)
        self._save_deployed_sigma2 = save_deployed_sigma2
        self.update_algo = update_algo
        self.prior_type = prior_type
        self._constraint = constraint

    @property
    def save_deployed_sigma2(self):
        """bool : whether to save the deployed (not updated) variance"""
        return self._save_deployed_sigma2

    @property
    def update_algo(self):
        return self._update_algo

    @update_algo.setter
    def update_algo(self, update_algo):
        assert update_algo in ('EM', 'AM'), 'Update algorithm must be either EM or AM'
        self._update_algo = update_algo

    @property
    def prior_type(self):
        return self._prior_type

    @prior_type.setter
    def prior_type(self, prior_type):
        assert prior_type in ('trivial', 'binary', 'discrete-phase', 'box', 'half-space'), 'Prior type must be either trivial, binary, discrete-phase, box or half-space'
        self._prior_type = prior_type

    @property
    def constraint(self):
        return self._constraint

class SectionOutput(SectionBase):
    """
    Section Output with additive noise

    .. image:: /static/lmlib/irrls/irrls-SectionOutput.svg
        :height: 300
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
        Forwarded to :class:`.Section`
    """

    def __init__(self, C, sigma2_init, y, estimate_output=False, **kwargs):
        super().__init__(**kwargs)
        self.C = C
        self._sigma2_init = sigma2_init
        self.y = y
        self._estimate_output = estimate_output
        self._save_marginal |= estimate_output

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


class SectionOutputOutlier(SectionOutput):
    def __init__(self, C, sigma2_init, y, save_outlier_estimate=False, outlier_threshold_factor=10, **kwargs):
        super().__init__(C, sigma2_init, y, **kwargs)
        self._outlier_threshold_factor = outlier_threshold_factor
        self._save_outlier_estimate = save_outlier_estimate

    @property
    def outlier_threshold_factor(self):
        return self._outlier_threshold_factor

    @property
    def save_outlier_estimate(self):
        """bool : save_outlier_estimate status"""
        return self._save_outlier_estimate
