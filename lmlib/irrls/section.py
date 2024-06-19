import warnings
from abc import ABC, abstractmethod

import numpy as np
from lmlib.utils.check import is_square, is_2dim, all_equal
from lmlib.irrls.message_passing import *

__all__ = ['SectionBase', 'SectionContainer', 'SectionSystem',
           'SectionInput',
           'SectionInput_NUV', 'SectionInput_sNUV',
           'SectionInput_L1', 'SectionInput_sL1',
           'SectionInput_Binary',
           'SectionOutput']


def validate_mp_type(f):
    def wrapper(section, mp_type, K):
        if mp_type not in ("MBF", "BIFM"):
            raise ValueError("mp_type known. \"MBF\" or \"BIFM\" available")
        return f(section, mp_type, K)

    return wrapper


class SectionBase(ABC):
    """
    Abstract base class of all sections

    Parameters
    ----------
    label : string, optional
        Label of the section instance
    save_state_marginal : boolean, optional
        Saves marginals when True, else no memory will be allocated
    """

    def __init__(self, label='n/a', save_state_marginal=False):
        self.label = label
        self._save_state_marginal = save_state_marginal
        self._N = 0
        self.mp = None

    @abstractmethod
    @validate_mp_type
    def _setup_mp(self, mp_type, K):
        pass

    def get_state_marginal(self):
        r"""
        Returns the state marginal :math:`X` of the section

        Returns
        -------
        X : :class:`np.recarray` of length K
            state marginals X
                - :code:`X.m` (mean)
                - :code:`X.V` (co-variance)

        """
        assert self.save_state_marginal, f"No state marginal saved! Set save_state_marginal=True on {self}"
        return self.mp.memory['X']

    @property
    def label(self):
        """str : label of the section"""
        return self._label

    @label.setter
    def label(self, label):
        if not isinstance(label, str):
            raise TypeError("Label is not type string.")
        self._label = label

    @property
    def N(self):
        """int : model order"""
        return self._N

    @property
    def save_state_marginal(self):
        """bool : whether to save state marginal of the section"""
        return self._save_state_marginal


class SectionContainer(SectionBase):
    """
    Superior section containing a list of other sections

    Inherits from the SectionBase class, but offers the option of collecting several sections into a super section.

    .. image:: /static/lmlib/irrls/irrls-SectionContainer.svg
        :height: 200
        :align: center

    Parameters
    ----------
    sections : array_like of :class:`.SectionBase`
        List of sections
    **kwargs : optional
        See :class:`.SectionBase`
    """

    def __init__(self, sections, **kwargs):
        super().__init__(**kwargs)

        self._subsections = []
        for section in sections:
            self.append_section(section)

        # check model dimensions integrity
        assert all_equal([s.N for s in self.subsections]), \
            "model dimension doesnt match between sections"
        self._N = self.subsections[0].N

    @property
    def subsections(self):
        """list : subsections of the section"""
        return self._subsections

    def append_section(self, section):
        """
        Appends a section to the internal section list

        Parameters
        ----------
        section : SectionBase
            section to append

        """

        if not isinstance(section, SectionBase):
            raise TypeError('Section must be a SectionBase object')
        self._subsections.append(section)

    @validate_mp_type
    def _setup_mp(self, mp_type, K):
        if mp_type == "MBF":
            self.mp = MBF_SectionContainer(self, K)
        if mp_type == "BIFM":
            self.mp = BIFM_SectionContainer(self, K)

        for s in self._subsections:
            s._setup_mp(mp_type, K)


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
        Forwarded to :class:`.SectionBase`
    """

    def __init__(self, A, **kwargs):
        """Constructor for SectionSystem"""
        super().__init__(**kwargs)
        self.A = A

    @property
    def A(self):
        """:class:`~numpy.ndarray` : State Transition Matrix"""
        return self._A

    @A.setter
    def A(self, A):
        assert is_square(A)
        self._A = np.asarray(A)
        self._N = self._A.shape[0]

    @validate_mp_type
    def _setup_mp(self, mp_type, K):
        if mp_type == "MBF":
            self.mp = MBF_SectionSystem(self, K)
        if mp_type == "BIFM":
            self.mp = BIFM_SectionSystem(self, K)


class SectionInputBase(SectionBase, ABC):
    """
    Section Input Base Class

    Parameters
    ----------
    B : array_like of shape (N, M)
        Input Matrix
    save_input_marginal : bool, optional
        Saves the input marginal U, default=False
    **kwargs
        Forwarded to :class:`.SectionBase`
    """

    def __init__(self, B, save_input_marginal=False, **kwargs):
        super().__init__(**kwargs)
        self.B = B
        self.save_input_marginal = save_input_marginal

    @property
    def B(self):
        """:class:`~numpy.ndarray` : Input Matrix"""
        return self._B

    @B.setter
    def B(self, B):
        assert is_2dim(B)
        self._B = np.asarray(B)
        self._N = self._B.shape[0]
        self._M = self._B.shape[1]

    @property
    def M(self):
        """int : Input Order"""
        return self._M

    @property
    def save_input_marginal(self):
        """bool : Whenever to save the input marginal U"""
        return self._save_input_marginal

    @save_input_marginal.setter
    def save_input_marginal(self, save_input_marginal):
        if not isinstance(save_input_marginal, bool):
            raise ValueError('save_input_marginal is not of type bool')
        self._save_input_marginal = save_input_marginal

    def get_input_marginal(self):
        r"""
        Returns the input marginal :math:`U` of the section

        Returns
        -------
        U : :class:`np.recarray` of length K
            input marginals U
                - :code:`U.m` (mean)
                - :code:`U.V` (co-variance)

        """
        assert self.save_input_marginal, f"No input marginal saved! Set save_input_marginal=True on {self}"
        return self.mp.memory['U']


class SectionInputUpdateBase(SectionInputBase, ABC):
    def __init__(self, B, update_method='AM', beta=1.0, **kwargs):
        super().__init__(B, **kwargs)
        self.update_method = update_method
        self.beta = beta

    @property
    def update_method(self):
        """str : Update method ('AM' or 'EM')"""
        return self._update_method

    @update_method.setter
    def update_method(self, update_method):
        if update_method not in ('AM', 'EM'):
            raise ValueError("update_method is not 'AM' or 'EM'")
        self._update_method = update_method

    @property
    def beta(self):
        """float : scale factor for AM variance updated (not active when EM-Method is selected)"""
        return self._beta

    @beta.setter
    def beta(self, beta):
        if not np.isscalar(beta):
            raise ValueError('beta is not scalar')
        self._beta = float(beta)
        if self._update_method == 'EM' and self._beta != 1.0:
            warnings.warn("beta is not active when EM-Method is used!")


class SectionInput(SectionInputBase):
    """
    Section Input of Zero Mean White Gaussian Noise (WGN)

    .. image:: /static/lmlib/irrls/irrls-SectionInput.svg
        :height: 300
        :align: center

    Parameters
    ----------
    B : array_like of shape (N, M)
        Input Matrix
    sigma2 : scalar or array_like of shape (M, M), optional
        (Co-)Variance of zero-mean white gaussian noise, default=1.0
    **kwargs
        Forwarded to :class:`.SectionBase`
    """

    def __init__(self, B, sigma2=1.0, **kwargs):
        super().__init__(B, **kwargs)
        self.sigma2 = sigma2

    @property
    def sigma2(self):
        """float, array_like : variance of shape (1,) or shape (K,) or covariance of shape (M, M) or  shape (K, M, M)"""
        return self._sigma2

    @sigma2.setter
    def sigma2(self, sigma2):

        if np.ndim(sigma2) == 1:
            self._sigma2 = np.asarray(sigma2)
            self._is_sigma2_k = True
            return

        if np.ndim(sigma2) == 3:
            if np.shape(sigma2)[-2:] != (self._M, self._M):
                raise ValueError("sigma2 needs to be array_like of shape (K, M, M)")
            self._sigma2 = np.asarray(sigma2)
            self._is_sigma2_k = True
            return

        if np.isscalar(sigma2):
            self._sigma2 = sigma2
            self._is_sigma2_k = False
            return

        if np.ndim(sigma2) == 2:
            if np.shape(sigma2) != (self._M, self._M):
                raise ValueError("sigma2 needs to be array_like of shape (M, M)")
            self._sigma2 = np.asarray(sigma2)
            self._is_sigma2_k = False
            return

        raise ValueError("sigma2 needs to be a scalar or array_like of shape ([K,] M, M) or shape (K,)")

    def _setup_mp(self, mp_type, K):
        if mp_type == "MBF":
            if self._is_sigma2_k:
                self.mp = MBF_SectionInput_k(self, K)
            else:
                self.mp = MBF_SectionInput(self, K)

        if mp_type == "BIFM":
            if self._is_sigma2_k:
                self.mp = MBF_SectionInput_k(self, K)
            else:
                self.mp = BIFM_SectionInput(self, K)


class SectionInput_NUV(SectionInputUpdateBase):
    """
    Section Input with NUV Prior

    .. image:: /static/lmlib/irrls/irrls-SectionInputNUV.svg
        :height: 300
        :align: center


    Parameters
    ----------
    B : array_like of shape (N, M)
        Input Matrix
    sigma2_init : scalar or array_like of shape (M, M), optional
        Initial (Co-)Variance of zero-mean white gaussian noise, default=1.0
    update_method : string, optional
        Update method either 'AM' for Alternate Maximization (default) or 'EM' for Exact-Maximization.
    save_input_marginal : bool, optional
        Saves the input marginal U, default=False
    beta : scalar
        Scaling factor of AM update method, default=1.0
    **kwargs
        Forwarded to :class:`.SectionBase`
    """

    def __init__(self, B, sigma2_init=1.0, **kwargs):
        super().__init__(B, **kwargs)
        self.sigma2_init = sigma2_init

    @property
    def sigma2_init(self):
        """float, array_like : variance or covariance of shape (M, M)"""
        return self._sigma2_init

    @sigma2_init.setter
    def sigma2_init(self, sigma2_init):
        if not np.isscalar(sigma2_init) and not (np.shape(sigma2_init) == (self._M, self._M)):
            raise ValueError("sigma2_init needs to be a scalar or array_like of shape (M, M)")
        self._sigma2_init = sigma2_init

    def get_sigma2_estimate(self):
        """
        Returns the latest input variance estimate

        Returns
        -------
        out : :class:`~numpy.ndarray` of shape (K, [M, M])
            updated sigma2 estimate
        """
        return self.mp.memory['sigma2']

    def _setup_mp(self, mp_type, K):
        if mp_type == "MBF":
            self.mp = MBF_SectionInput_NUV(self, K)
        if mp_type == "BIFM":
            self.mp = BIFM_SectionInput_NUV(self, K)


class SectionInput_sNUV(SectionInput_NUV):

    def __init__(self, B, r2, **kwargs):
        super().__init__(B, **kwargs)
        if self.M != 1:
            raise ValueError('Shape of B has to be (N, 1) such that M equals 1')
        self.r2 = r2

    @property
    def r2(self):
        """float : factor of the smooth function"""
        return self._r2

    @r2.setter
    def r2(self, r2):
        if not np.isscalar(r2):
            raise ValueError('r2 is not scalar')
        self._r2 = float(r2)

    @validate_mp_type
    def _setup_mp(self, mp_type, K):
        if mp_type == "MBF":
            self.mp = MBF_SectionInput_sNUV(self, K)
        if mp_type == "BIFM":
            raise NotImplemented("BIFM algorithm is not yet implemented")
            self.mp = BIFM_SectionInput_sNUV(self, K)


class SectionInput_L1(SectionInput_NUV):

    @validate_mp_type
    def _setup_mp(self, mp_type, K):
        if mp_type == "MBF":
            self.mp = MBF_SectionInput_L1(self, K)
        if mp_type == "BIFM":
            raise NotImplemented("BIFM algorithm is not yet implemented")
            self.mp = BIFM_SectionInput_L1(self, K)


class SectionInput_sL1(SectionInput_sNUV):

    @validate_mp_type
    def _setup_mp(self, mp_type, K):
        if mp_type == "MBF":
            self.mp = MBF_SectionInput_sL1(self, K)
        if mp_type == "BIFM":
            raise NotImplemented("BIFM algorithm is not yet implemented")
            self.mp = BIFM_SectionInput_sL1(self, K)


class SectionInput_Binary(SectionInputUpdateBase):
    """
    Section Input with NUV Prior

    .. image:: /static/lmlib/irrls/irrls-SectionInput.svg
        :height: 300
        :align: center


    Parameters
    ----------
    B : array_like of shape (N, M)
        Input Matrix
    sigma2_init : scalar or array_like of shape (M, M), optional
        (Co-)Variance of zero-mean white gaussian noise, default=1.0
    save_input_marginal : bool, optional
        Saves the input marginal U, default=False
    **kwargs
        Forwarded to :class:`.SectionBase`
    """

    def __init__(self, B, a, b, sigma2a_init=1.0, sigma2b_init=1.0, **kwargs):
        super().__init__(B, **kwargs)
        self.sigma2a_init = sigma2a_init
        self.sigma2b_init = sigma2b_init
        self.a = a
        self.b = b

    @property
    def sigma2a_init(self):
        """float, array_like : variance of level a of shape (M, M)"""
        return self._sigma2a_init

    @sigma2a_init.setter
    def sigma2a_init(self, sigma2a_init):
        if not np.isscalar(sigma2a_init) and not (np.shape(sigma2a_init) == (self._M, self._M)):
            raise ValueError("sigma2a_init needs to be a scalar or array_like of shape (M, M)")
        self._sigma2a_init = sigma2a_init

    @property
    def sigma2b_init(self):
        """float, array_like : variance of level b of shape (M, M)"""
        return self._sigma2b_init

    @sigma2b_init.setter
    def sigma2b_init(self, sigma2b_init):
        if not np.isscalar(sigma2b_init) and not (np.shape(sigma2b_init) == (self._M, self._M)):
            raise ValueError("sigma2b_init needs to be a scalar or array_like of shape (M, M)")
        self._sigma2b_init = sigma2b_init

    @property
    def beta(self):
        """float : scale factor for AM variance updated (not active when EM-Method is selected)"""
        return self._beta

    @beta.setter
    def beta(self, beta):
        if not np.isscalar(beta):
            raise ValueError('beta is not scalar')
        self._beta = float(beta)
        if self._update_method == 'EM' and self._beta != 1.0:
            warnings.warn("beta is not active when EM-Method is used!")

    @property
    def a(self):
        """float : level a"""
        return self._a

    @a.setter
    def a(self, a):
        if not np.isscalar(a):
            raise ValueError('a is not scalar')
        self._a = float(a)

    @property
    def b(self):
        """float : level b"""
        return self._b

    @b.setter
    def b(self, b):
        if not np.isscalar(b):
            raise ValueError('b is not scalar')
        self._b = float(b)

    def _setup_mp(self, mp_type, K):
        if mp_type == "MBF":
            self.mp = MBF_SectionInput_Binary(self, K)
        if mp_type == "BIFM":
            raise NotImplemented("BIFM algorithm is not yet implemented")
            self.mp = BIFM_SectionInput_Binary(self, K)


class SectionOutput(SectionBase):
    r"""
    Section Output with Additive White Gaussian Noise (AWGN)

    .. image:: /static/lmlib/irrls/irrls-SectionOutput.svg
        :height: 300
        :align: center


    Parameters
    ----------
    C : array_like of shape (L, N)
        Input Matrix
    sigma2 : scalar or array_like of shape (L, L), optional
        (Co-)Variance of zero-mean white gaussian noise, default=1.0
    save_output_marginal : bool, optional
        Saves the output marginal :math:`\tilde{Y}`, default=False
    **kwargs
        Forwarded to :class:`.SectionBase`
    """

    def __init__(self, C, y, sigma2=1.0, save_output_marginal=False, **kwargs):
        super().__init__(**kwargs)
        self.C = C
        self.y = y
        self.sigma2 = sigma2
        self.save_output_marginal = save_output_marginal

    @property
    def C(self):
        """:class:`~numpy.ndarray` : Output Matrix C"""
        return self._C

    @C.setter
    def C(self, C):
        assert is_2dim(C)
        self._C = np.asarray(C)
        self._L = self._C.shape[0]
        self._N = self._C.shape[1]

    @property
    def L(self):
        """int : Output Order"""
        return self._L

    @property
    def sigma2(self):
        """float, array_like : variance or covariance of shape (L, L)"""
        return self._sigma2

    @sigma2.setter
    def sigma2(self, sigma2):
        if not np.isscalar(sigma2) and not (np.shape(sigma2) == (self._L, self._L)):
            raise ValueError("sigma2 needs to be a scalar or array_like of shape (L, L)")
        self._sigma2 = sigma2

    @property
    def save_output_marginal(self):
        """bool : whether to save the output marginal of the section"""
        return self._save_output_marginal

    @save_output_marginal.setter
    def save_output_marginal(self, save_output_marginal):
        if not isinstance(save_output_marginal, bool):
            raise ValueError('save_output_marginal is not of type bool')
        self._save_output_marginal = save_output_marginal

    def get_output_marginal(self):
        r"""
        Returns the output marginal :math:`\tilde{Y}` of the section

        Returns
        -------
        Y_tilde : :class:`np.recarray` of length K
            output marginals Y_tilde
                - :code:`Y_tilde.m` (mean)
                - :code:`Y_tilde.V` (co-variance)

        """
        assert self.save_output_marginal, f"No output marginal saved! Set save_output_marginal=True on {self}"
        return self.mp.memory['Y_tilde']

    def _setup_mp(self, mp_type, K):
        if mp_type == "MBF":
            self.mp = MBF_SectionOutput(self, K)
        if mp_type == "BIFM":
            self.mp = BIFM_SectionOutput(self, K)
