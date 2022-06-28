"""This module provides methods to define linear state space models and methods to use them as signal models in recursive least squares problems."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
import re
import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import block_diag, pascal
from lmlib.utils import *

__all__ = ['Alssm', 'AlssmPoly', 'AlssmPolyJordan', 'AlssmSin', 'AlssmExp',
           'AlssmStacked', 'AlssmStackedSO', 'AlssmProd', 'ModelBase']


class ModelBase(ABC):
    """
    Abstract baseclass for autonomous linear state space models.

    Parameters
    ----------
    label : str, optional
        Label of Alssm, default: 'n/a'
    C_init : array_like, None, optional
        Initialized Output Matrix, default: 'None'
    """

    def __init__(self, label='n/a', C_init=None):
        self._alssms = list()
        self._deltas = list()
        self._A = None
        self._C = None
        self.label = label
        self.C_init = C_init
        self._state_var_labels = dict()

    def __str__(self):
        A_str = re.sub('\s+', ',', np.array_str(self.A).replace('\n', '')).replace('[,', '[')
        C_str = re.sub('\s+', ',', np.array_str(self.C).replace('\n', '')).replace('[,', '[')
        return f'{type(self).__name__}(A={A_str}, C={C_str}, label={self.label})'

    @abstractmethod
    def update(self):
        """
        Model update

        Updates the internal model (A and C Matrix) based on the initialization parameters of a class.
        """
        pass

    @property
    def A(self):
        """:class:`~numpy.ndarray`, shape=(N, N) : State matrix :math:`A \\in \\mathbb{R}^{N \\times N}`"""
        return self._A

    @A.setter
    def A(self, A):
        assert is_array_like(A), 'A is not array like'
        assert is_square(A), f'A is not square, {info_str_found_shape(A)}'
        self._A = np.asarray(A)

    @property
    def C(self):
        """:class:`~numpy.ndarray`, shape=([L,] N) : Output matrix :math:`C \\in \\mathbb{R}^{L \\times N}`"""
        return self._C

    @C.setter
    def C(self, C):
        assert is_array_like(C), 'C is not array like'
        assert is_1dim(C) or is_2dim(C), f'C is not 1 or 2 dimensional, {info_str_found_shape(C)}'
        self._C = np.asarray(C)

    @property
    def C_init(self):
        """:class:`~numpy.ndarray`, shape=([L,] N) : Initialized Output matrix :math:`C \\in \\mathbb{R}^{L \\times N}`"""
        return self._C_init

    @C_init.setter
    def C_init(self, C):
        if C is None:
            self._C_init = None
        else:
            assert is_array_like(C)
            assert is_1dim(C) or is_2dim(C), f'C_init is not 1 or 2 dimensional, {info_str_found_shape(C)}'
            self._C_init = np.asarray(C)

    @property
    def label(self):
        """str : Label of the model"""
        return self._label

    @label.setter
    def label(self, label):
        assert isinstance(label, str)
        self._label = label

    @property
    def N(self):
        """int : Model order :math:`N`"""
        return self._A.shape[0]

    @property
    def alssms(self):
        """list : Set of models"""
        return self._alssms

    @alssms.setter
    def alssms(self, alssms):
        assert isinstance(alssms, Iterable), 'alssms is not iterable'
        for alssm in alssms:
            assert isinstance(alssm, ModelBase), 'element in alssms is not instance nor subclass of Alssm'
        self._alssms = list(alssms)

    @property
    def deltas(self):
        """:class:`np.ndarray` : Output scaling factors for each ALSSM in `alssms`"""
        return self._deltas

    @deltas.setter
    def deltas(self, deltas):
        _n = len(self.alssms)
        deltas = [1] * _n if deltas is None else deltas
        assert isinstance(deltas, Iterable), 'deltas is not iterable'
        assert np.size(deltas) == _n, f'Output scaling factors deltas are not of length {_n}, ' \
                                      f'{info_str_found_shape(deltas)}'
        for delta in deltas:
            assert np.isscalar(delta), 'element in deltas is not scalar'
        self._deltas = np.asarray(deltas)

    @property
    def state_var_labels(self):
        """dict : Dictionary containing state variable labels and index"""
        return self._state_var_labels

    def eval_states(self, xs):
        r"""
        Evaluation of the ALSSM for an array of state vectors `xs`.

        `eval_states(...)` returns the ALSSM output

        .. math::
            s_0(x) = CA^0x = Cx

        for each state vector :math:`x` from the array `xs`


        Parameters
        ----------
        xs : array_like of shape=(XS,N[,S])
            List of length `XS` with state vectors :math:`x`.

        Returns
        -------
        s : :class:`~numpy.ndarray` of shape=(XS,[L[,S]])
            ALSSM outputs


        |def_N|
        |def_L|
        |def_S|


        Examples
        --------
        >>> A = [[1, 1], [0, 1]]
        >>> C = [1, 0]
        >>> alssm = lm.Alssm(A, C, label='line')
        >>>
        >>> xs = [[0.1, 3], [0, 1], [-0.8, 0.2], [1, -3]]
        >>> s = alssm.eval_states(xs)
        >>> print(s)
        [ 0.1  0.  -0.8  1. ]

        """
        return np.tensordot(self.C, xs, axes=(-1, 1))

    def eval_state(self, x):
        r"""
        Evaluation of the ALSSM for a state vector `x`.

        `eval_state(...)` returns the ALSSM output

        .. math::
            s_0(x) = CA^0x = Cx

        for a state vector :math:`x`.


        Parameters
        ----------
        x : array_like of shape=(N[,S])
           State vector :math:`x`

        Returns
        -------
        s : :class:`~numpy.ndarray` of shape=([L[,S]])
            ALSSM output


        |def_N|
        |def_L|
        |def_S|


        Examples
        --------
        >>> A = [[1, 1], [0, 1]]
        >>> C = [1, 0]
        >>> alssm = lm.Alssm(A, C, label='line')
        >>>
        >>> x = [0.1, 3]
        >>> s = alssm.eval_state(x)
        >>> print(s)
        0.1

        """

        return np.tensordot(self.C, x, axes=(-1, 0))

    def trajectory(self, x, js):
        r"""
        Evaluation of the ALSSM for a state vector `x` at evaluation indeces js.

        `trajectory(...)` returns the ALSSM output

        .. math::
            s_j(x) = CA^jx = Cx

        for a state vector :math:`x` and index :math:`j` in the list `js`


        Parameters
        ----------
        x : array_like of shape=(N[,S])
           State vector :math:`x`
        js : array_like of shape=(J,)
            ALSSM evaluation indices

        Returns
        -------
        s : :class:`~numpy.ndarray` of shape=(J, [L[,S]])
            ALSSM outputs


        |def_N|
        |def_L|
        |def_S|
        |def_j_index|
        |def_J|

        Examples
        --------
        >>> A = [[1, 1], [0, 1]]
        >>> C = [1, 0]
        >>> alssm = lm.Alssm(A, C, label='line')
        >>>
        >>> x = [0.1, 3]
        >>> s = alssm.trajectory(x, js=[0, 1, 2, 3, 4, 5])
        >>> print(s)

        """
        return np.asarray([np.tensordot(self.C @ matrix_power(self.A, j), x, axes=(-1, 0)) for j in js])

    def trajectories(self, xs, js):
        r"""
        Evaluation of the ALSSM for an array state vectors `xs` at evaluation indeces js.

        `trajectories(...)` returns the ALSSM output

        .. math::
            s_j(x) = CA^jx = Cx

        for a state vector :math:`x` and index :math:`j` in the list `js`


        Parameters
        ----------
        xs : array_like of shape=(XS,N[,S])
            List of length `XS` with state vectors :math:`x`.
        js : array_like of shape=(J,)
            ALSSM evaluation indices

        Returns
        -------
        s : :class:`~numpy.ndarray` of shape=(XS, J, [L[,S]])
            ALSSM outputs


        |def_N|
        |def_L|
        |def_S|
        |def_j_index|
        |def_J|
        |def_XS|

        Examples
        --------
        >>> A = [[1, 1], [0, 1]]
        >>> C = [1, 0]
        >>> alssm = lm.Alssm(A, C, label='line')
        >>>
        >>> xs = [[0.1, 3], [0, 1], [-0.8, 0.2], [1, -3]]
        >>> s = alssm.trajectories(xs, js=[0, 1, 2, 3, 4, 5])
        >>> print(s)

        """
        return np.asarray([[np.tensordot(self.C @ matrix_power(self.A, j), x, axes=(-1, 0)) for j in js] for x in xs])

    def dump_tree(self):
        """
        Returns the internal structure of the ALSSM model as a string.

        Returns
        -------
        out : str
            String representing internal model structure.

        Examples
        --------
        >>> alssm_poly = lm.AlssmPoly(4, label="high order polynomial.rst")
        >>> A = [[1, 1], [0, 1]]
        >>> C = [[1, 0]]
        >>> alssm_line = lm.Alssm(A, C, label="line")
        >>> stacked_alssm = lm.AlssmStacked((alssm_poly, alssm_line), label='stacked model')
        >>> print(stacked_alssm.dump_tree())
        └-Alssm : stacked, A: (7, 7), C: (2, 7), label: stacked model
          └-Alssm : polynomial.rst, A: (5, 5), C: (1, 5), label: high order polynomial.rst
          └-Alssm : native, A: (2, 2), C: (1, 2), label: line


        """
        return self._rec_tree(level=0)

    def set_state_var_label(self, label, indices):
        """
        Adds a label for one or multiple state variabels in the state vector.
        Such labels are used to quickly referece to single states in the state vector by its names.

        Parameters
        ----------
        label : str
            Label name
        indices : tuple
            State indices

        Examples
        --------
        >>> alssm = lm.AlssmPoly(poly_degree=1, label='slope_with_offset')
        >>> alssm.set_state_var_label('slope', (1,))
        >>> alssm.state_var_labels
        {'x': range(0, 2), 'x0': (0,), 'x1': (1,), 'slope': (1,)}
        >>> alssm.state_var_labels['slope']
        (1,)

        """
        self._state_var_labels[label] = indices

    def _init_state_var_labels(self):
        for n in range(self.N):
            self._state_var_labels['x' + str(n)] = (n,)
        self._state_var_labels['x'] = list(range(self.N))

    def _rec_tree(self, level):
        str_self = f'{type(self).__name__}, A: {str(self.A.shape)}, C: {str(self.C.shape)}, label: {self.label}'
        str_tree = ('  ' * level + '└-' + str_self)
        if len(self.alssms) != 0:
            for alssm in self.alssms:
                str_tree += '\n'
                str_tree += alssm._rec_tree(level=level + 1)
        return str_tree

    def get_state_var_labels(self):
        """
        Retruns a list of state variable labels

        Returns
        -------
        out : list
            list of state variable labels
        """
        state_list = []
        for var_label, indices in self._state_var_labels.items():
            state_list.append((self.label + '.' + var_label, indices))

        N = 0
        for alssm in self.alssms:
            for var_label, indices in alssm.get_state_var_labels():
                state_list.extend([(self.label + '.' + var_label, tuple(i+N for i in indices))])
            N += alssm.N
        return state_list

    def get_state_var_indices(self, label):
        """
        Returns the state indices for a specified label

        Parameters
        ----------
        label : str
            state label

        Returns
        -------
        out : list of int
            state indices of the label
        """
        for l, indices in self.get_state_var_labels():
            if label == l:
                return indices


class Alssm(ModelBase):
    r"""
    Generic Autonomous Linear State Space Model (ALSSM)

    This class holds the parameters of a discrete-time, autonomous (i.e., input-free), single- or multi-output linear
    state space model, defined recursively by

    .. math::
       x[k+1] &= Ax[k]

       s_k(x) = y[k] &= Cx[k],

    where :math:`A \in \mathbb{R}^{N\times N}, C \in \mathbb{R}^{L \times N}` are the fixed model parameters (matrices),
    :math:`k` the time index,
    :math:`y[k] \in \mathbb{R}^{L \times 1}` the output vector,
    and :math:`x[k] \in \mathbb{R}^{N}` the state vector.

    For more details, see also [Wildhaber2019]_ [Eq. 4.1].

    Parameters
    ----------
    A : array_like, shape=(N, N)
        State Matrix
    C : array_like, shape=([L,] N)
        Output Matrix
    **kwargs
        Forwarded to :class:`.ModelBase`


    |def_N|
    |def_L|

    Note
    ----
    The output matrix :math:`C` can be of the form (N,) for a 1-dimensional output or of the form (L, N), resulting in a
    2-dimensional output. Accordingly, the signal sample :math:`y[k]` in a cost function must be of the form of the ALSSM output.

    Examples
    --------
    >>> A = [[1, 1], [0, 1]]
    >>> C = [1, 0]
    >>> alssm = lm.Alssm(A, C, label='line')
    >>> print(alssm)
    A =
    [[1 1]
     [0 1]]
    C =
    [1 0]
    >>> alssm
    Alssm : native, A: (2, 2), C: (1, 2), label: line

    """

    def __init__(self, A, C, **kwargs):
        super().__init__(C_init=C, **kwargs)
        self.A = A
        self.C = C
        self.update()

    def update(self):
        self.C = self.C_init
        self._init_state_var_labels()


class AlssmPoly(ModelBase):
    r"""
    ALSSM with discrete-time polynomial.rst output sequence

    Representation of a `Q`-th degree polynomial.rst in `i` of the form

    .. math::
        P_Q(i) = x_0 i^0 + x_1 i^1 + ... + x_Q i^Q

    as an ALSSM, with transition matrix

    .. math::
        A =
        \begin{bmatrix}
            1 & 1 & 1 \\
            0 & 1 & 2 \\
            0 & 0 & 1
        \end{bmatrix}

    and state vector

    .. math::
        x =
        \begin{bmatrix}
            x_0 & x_1 & ... & x_N \\
        \end{bmatrix}^T

    For more details see [Wildhaber2019]_ :download:`PDF <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=48>`


    Parameters
    ----------
    poly_degree : int
        Polynomial degree `Q`. Corresponds to the highest exponent of the polynomial.rst.
        It follows a ALSSM system of order `N = Q+1`.
    C : array_like, shape=([L,] N), optional
        Output Matrix.
        If no output matrix is given, C gets initialize automatically to `[1, 0, ...]` such that the shape
        is `(N,)`. In addition with ``as_2dim_C=True`` C gets broadcated to shape `(1, N)`. (default: C=None)
    **kwargs
        Forwarded to :class:`.ModelBase`


    |def_N|
    |def_L|


    Examples
    --------
    Setting up a 4. order polynomial.rst, autonomous linear state space model.

    >>> poly_degree = 3
    >>> alssm = lm.AlssmPoly(poly_degree, label='poly')
    >>> print(alssm)
    A =
    [[1 1 1 1]
     [0 1 2 3]
     [0 0 1 3]
     [0 0 0 1]]
    C =
    [1 0 0 0]

    >>> C = [[1, 0, 0], [0, 1, 0]]
    >>> alssm = lm.AlssmPoly(poly_degree=2, C=C, label='poly')
    >>> print(alssm)
    A =
    [[1 1 1]
     [0 1 2]
     [0 0 1]]
    C =
    [[1 0 0]
     [0 1 0]]

   """

    def __init__(self, poly_degree, C=None, **kwargs):
        super().__init__(**kwargs)
        self.poly_degree = poly_degree
        self.C_init = C
        self.update()

    def update(self):
        if self.C_init is None:
            self.C = np.hstack([[1], [0] * self.poly_degree])
        else:
            self.C = self.C_init
        self.A = pascal(self.poly_degree + 1, kind='upper')
        self._init_state_var_labels()

    @property
    def poly_degree(self):
        """int : Polynomial degree :math:`Q` (highest exponent/ order - 1)"""
        return self._poly_degree

    @poly_degree.setter
    def poly_degree(self, poly_degree):
        assert isinstance(poly_degree, int), 'poly_degree is not of type int'
        assert poly_degree >= 0, 'poly_degree is not larger then 0'
        self._poly_degree = poly_degree


class AlssmPolyJordan(ModelBase):
    r"""
    ALSSM with discrete-time polynomial output sequence, in Jorandian normal form


    Discrete-time polynomial with ALSSM with transition matrix in Jordanian normal form, see [Zalmai2017]_
    :download:`PDF <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/176652/zalmai_thesis.pdf#page=41>`.


    .. math::
        A =
        \begin{bmatrix}
            1 & 1 & 0 \\
            0 & 1 & 1 \\
            0 & 0 & 1
        \end{bmatrix}


    Parameters
    ----------
    poly_degree : int
       Polynomial degree. Corresponds to the highest exponent of the polynomial. `N = poly_degree+1`.
    C : array_like, shape=(L, N), optional
        Output Matrix.
        If C is not set, C is initialized to `[[..., 0, 1]]` of shape `(1, N)`. (default: C=None)
    **kwargs
        Forwarded to :class:`.ModelBase`


    |def_N|
    |def_L|


    Examples
    --------
    Setting up a 3th degree polynomial ALSSM.

    >>> poly_degree = 3
    >>> alssm = lm.AlssmPolyJordan(poly_degree, label='poly')
    >>> print(alssm)
    A =
    [[1. 1. 0. 0.]
     [0. 1. 1. 0.]
     [0. 0. 1. 1.]
     [0. 0. 0. 1.]]
    C =
    [[1 0 0 0]]

   """

    def __init__(self, poly_degree, C=None, **kwargs):
        super().__init__(**kwargs)
        self.poly_degree = poly_degree
        self.C_init = C
        self.update()

    def update(self):
        if self.C_init is None:
            self.C = np.hstack([[1], [0] * self.poly_degree])
        else:
            self.C = self.C_init
        self.A = np.eye(self.poly_degree + 1) + np.diagflat(np.ones(self.poly_degree), 1)
        self._init_state_var_labels()

    @property
    def poly_degree(self):
        """int : Polynomial degree :math:`Q` (highest exponent/ order - 1)"""
        return self._poly_degree

    @poly_degree.setter
    def poly_degree(self, poly_degree):
        assert isinstance(poly_degree, int), 'poly_degree is not of type int'
        assert poly_degree >= 0, 'poly_degree is not larger then 0'
        self._poly_degree = poly_degree


class AlssmSin(ModelBase):
    r"""
    ALSSM with discrete-time (damped) sinusoidal output sequence.

    The class AlssmSin is defined by a decay factor `rho` and discrete-time frequency `omega`.

    .. math::
        A =
        \begin{bmatrix}
            \rho \cos{\omega} & -\rho \sin{\omega} \\
            \rho \sin{\omega} & \rho \cos{\omega}
        \end{bmatrix}

    For more details see [Wildhaber2019]_ :download:`PDF <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=48>`

    Parameters
    ----------
    omega : float, int
        Frequency :math:`\omega = 2\pi f_s
    rho : float, int, optional
        Decay factor, default: rho = 1.0
    C : array_like, shape=(L, N), optional
        Output Matrix.
        If no output matrix is given, C gets initialize automatically to `[1, 0]`, such that the shape
        is `(N,)`. In addition with ``as_2dim_C=True`` C gets broadcated to shape `(1, N)`(default: C=None)
    **kwargs
        Forwarded to :class:`.ModelBase`


    |def_N|
    |def_L|


    Notes
    -----
    To convert a continuous-time frequency to a normalized frequency, see :func:`~lmlib.utils.generator.k_period_to_omega`, e.g.,

    >>> from lmlib.utils.generator import k_period_to_omega
    >>> alssm = lm.AlssmSin(k_period_to_omega(k_period), rho)


    Examples
    --------
    Parametrization of a sinusoidal ALSSM:

    >>> omega = 0.1
    >>> rho = 0.9
    >>> alssm = lm.AlssmSin(omega, rho)
    >>> print(alssm)
    A =
    [[ 0.89550375 -0.08985007]
     [ 0.08985007  0.89550375]]
    C =
    [1 0]

    """

    def __init__(self, omega, rho=1.0, C=None, **kwargs):
        super().__init__(**kwargs)
        self.omega = omega
        self.rho = rho
        self.C_init = C
        self.update()
        self.set_state_var_label('cos', (0,))
        self.set_state_var_label('sin', (1,))

    def update(self):
        if self.C_init is None:
            self.C = np.array([1, 0])
        else:
            self.C = self.C_init
        c, s = np.cos(self.omega), np.sin(self.omega)
        self.A = self.rho * np.array([[c, -s], [s, c]])
        self._init_state_var_labels()

    @property
    def omega(self):
        """float : Frequency factor :math:`\\omega = 2\\pi f_s`"""
        return self._omega

    @omega.setter
    def omega(self, omega):
        assert np.isscalar(omega), 'Frequency factor omega is a scalar'
        self._omega = float(omega)

    @property
    def rho(self):
        """float : Decay factor :math:`\\rho`"""
        return self._rho

    @rho.setter
    def rho(self, rho):
        assert np.isscalar(rho), 'Decay factor rho is not a scalar'
        self._rho = float(rho)


class AlssmExp(ModelBase):
    """
    ALSSM with discrete-time exponentially decaying/increasing output sequence.

    Discrete-time linear state space model generating output sequences of exponentially decaying shape with decay
    factor :math:`\\gamma`.

    For more details see [Wildhaber2019]_ :download:`PDF <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=48>`

    Parameters
    ----------
    gamma : float, int
        Decay factor per sample step ( > 1.0 : left-sided decaying; < 1.0 : right-sided decaying)
    C : array_like, shape=(L, N), optional
        Output Matrix.
        If no output matrix is given, C gets initialize automatically to `[[1]]`. (default: C=None)
    **kwargs
        Forwarded to :class:`.ModelBase`


    |def_N|
    |def_L|


    Examples
    --------
    Parametrizing an exponentially ALSSM:

    >>> gamma = 0.8
    >>> alssm = lm.AlssmExp(gamma)
    >>> print(alssm)
    A =
    [[0.8]]
    C =
    [[1.]]

    """

    def __init__(self, gamma, C=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.C_init = C
        self.update()

    def update(self):
        if self.C_init is None:
            self.C = np.ones((1,))
        else:
            self.C = self.C_init
        self.A = np.array([[self.gamma]])
        self._init_state_var_labels()

    @property
    def gamma(self):
        """float :  Decay factor per sample :math:`\\gamma`"""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        assert np.isscalar(gamma), 'Decay factor gamma is not scalar'
        self._gamma = gamma


class AlssmStacked(ModelBase):
    r"""
    Creates a joined ALSSM generating a stacked output signal of multiple ALSSMs.

    For :math:`M` ALSSMs as in :class:`Alssm`,
    we get the stacked model's output  :math:`\tilde{s}_k(\tilde{x}) = \tilde{y}_k`

    .. math::
        \tilde{s}_k(\tilde{x}) = \begin{bmatrix}
            s_k^{(1)}(x_1) \\
            \vdots \\
            s_k^{(M)}(x_M) \\
        \end{bmatrix} =
        \begin{bmatrix}
            y_1[k] \\
            \vdots \\
            y_M[k] \\
        \end{bmatrix} =
        \tilde{y}[k] =
         \tilde{c} \tilde{A}^k \tilde{x} \  \ ,

    where :math:`y_m[k]` denotes the output of the m-th joined ALSSM.
    :math:`y_m[k]`  is either a scalar (for signle-channel ALSSMs) or a column vector (for multi-channel ALSSMs).
    Accordingly, the initial state vector is

    .. math::
        \tilde{x} =
        \begin{bmatrix}
            x_1 \\
            \vdots \\
            x_M \\
        \end{bmatrix}

    where :math:`x_m[k]` is the state vector of the m-th joined ALSSM.



    Therefore, the internal model matrices of the joined model are set to the block diagonals

    .. math::
        \tilde{A} =
        \left[\begin{array}{c|c|c}
            A_1 & 0 & 0 \\ \hline
            0 & \ddots & 0 \\ \hline
            0 & 0 & A_{M}
        \end{array}\right]

        \tilde{C} =
        \left[\begin{array}{c|c|c}
            \lambda_1 C_1 & 0 & 0 \\ \hline
            0 & \ddots & 0 \\ \hline
            0 & 0 & \lambda_M C_{M}
        \end{array}\right]


    where :math:`A_m` and :math:`C_m` are the transition matrices and the output vectors of the joined models, respectively, and
    :math:`\lambda_1 ... \lambda_M  \in \mathcal{R}` are additional factors to weight each output individually.

   For more details see [Wildhaber2019]_ :download:`PDF Sec: Linear Combination of M Systems <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=48>`


    Parameters
    ----------
    alssms : tuple of shape(M) of ALSSMs
        Set of `M` autonomous linear state space models
    deltas: list of shape=(M) of floats, optional
        List of `M` scalar factors for each output matrix of the ALSSM in `alssms`.
        (default: deltas = None, i.e., all scalars are set to 1)
    **kwargs
        Forwarded to :class:`.ModelBase`


    |def_M|

    Examples
    --------
    >>> alssm_poly = lm.AlssmPoly(4, label="high order polynomial")
    >>> A = [[1, 1], [0, 1]]
    >>> C = [[1, 0]]
    >>> alssm_line = lm.Alssm(A, C, label="line")
    >>> stacked_alssm = lm.AlssmStacked((alssm_poly, alssm_line), label='stacked model')
    >>> print(stacked_alssm)
    A =
    [[1. 1. 1. 1. 1. 0. 0.]
     [0. 1. 2. 3. 4. 0. 0.]
     [0. 0. 1. 3. 6. 0. 0.]
     [0. 0. 0. 1. 4. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 1. 1.]
     [0. 0. 0. 0. 0. 0. 1.]]
    C =
    [[1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0.]]

    """

    def __init__(self, alssms, deltas=None, **kwargs):
        super().__init__(**kwargs)
        self.alssms = alssms
        self.deltas = deltas
        self.update()

    def update(self):
        for alssm in self.alssms:
            alssm.update()
        A = block_diag(*[alssm.A for alssm in self.alssms])
        C = block_diag(*[g * alssm.C for g, alssm in zip(self.deltas, self.alssms)])
        self.A = A
        self.C = C
        self._init_state_var_labels()


class AlssmStackedSO(ModelBase):
    r"""
    Stacking multiple ALSSMs, generating summed output vector of ALSSMs.

    Creating a joined ALSSM generating the summed output signal of :math:`M` ALSSMs,

    .. math::

        \tilde{s}_k(\tilde{x}) = s_k^{(1)}(x_1) + ... + s_k^{(M)}(x_M) = y_1[k] + ... + y_M[k] = \tilde{y}_k = \tilde{c} \tilde{A}^k \tilde{x}


    Therefore, the internal model matrices of the generated model are

    .. math::
        \tilde{A} =
        \left[\begin{array}{c|c|c}
            A_1 & 0 & 0 \\ \hline
            0 & \ddots & 0 \\ \hline
            0 & 0 & A_{M}
        \end{array}\right]

        \tilde{C} =
        \left[\begin{array}{c|c|c}
            \lambda_1 C_1 & ... & \lambda_M C_M
        \end{array}\right]


    Parameters
    ----------
    alssms : tuple of shape=(M) of :class:`Alssm`
        Set of `M` autonomous linear state space models.
        All ALSSMs need to have the same number of output channels, see :meth:`Alssm.output_count`
    deltas: list of shape=(M) of floats, optional
        List of `M` scalar factors for each output matrix of the ALSSM in `alssms`.
        (default: deltas = None, i.e., all scalars are set to 1)
    **kwargs
        Forwarded to :class:`.ModelBase`



    |def_M|

    Examples
    --------
    >>> alssm_poly = lm.AlssmPoly(poly_degree=3)
    >>> A = [[1, 1], [0, 1]]
    >>> C = [[1, 0]]
    >>> alssm_line = lm.Alssm(A, C)
    >>> stacked_alssm = lm.AlssmStackedSO((alssm_poly, alssm_line))
    >>> print(stacked_alssm)
    A =
    [[1. 1. 1. 1. 0. 0.]
     [0. 1. 2. 3. 0. 0.]
     [0. 0. 1. 3. 0. 0.]
     [0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 1. 1.]
     [0. 0. 0. 0. 0. 1.]]
    C =
    [[1. 0. 0. 0. 1. 0.]]

    """

    def __init__(self, alssms, deltas=None, **kwargs):
        super().__init__(**kwargs)
        self.alssms = alssms
        self.deltas = deltas
        self.update()

    def update(self):
        for alssm in self.alssms:
            alssm.update()
        assert common_C_dim(self.alssms), "Alssms has not same output dimensions."
        A = block_diag(*[alssm.A for alssm in self.alssms])
        C = np.hstack([g * alssm.C for g, alssm in zip(self.deltas, self.alssms)])
        self.A = A
        self.C = C
        self._init_state_var_labels()


class AlssmProd(ModelBase):
    r"""
    Joins multiple ALSSMs generating the output product.

    Creating a joined ALSSM generating the product of all output signals of :math:`M` ALSSMs, e.g., on the example of :math:`M=2`,

    .. math::

         \tilde{s}_k (\tilde{x}) = s_k^{(1)}(x_1) \cdot s_k^{(2)}(x_2) &= (C_1 A_1^k x_1)(C_2 A_2^k x_2)\\
         &= (C_1 A_1^k x_1) \otimes (C_2 A_2^k x_2)\\
         &= (C_1 \otimes C_2) (A_1^k \otimes A_2^k) (x_1 \otimes  x_2) \ ,

    where :math:`s_k^{(1)}(x_1) = C_1 A_1^k x_1` is the first and :math:`s_k^{(2)}(x_2) = C_2 A_2^k x_2` the second ALSSM.

    For more details, see also [Zalmai2017]_ [Proposition 2],  [Wildhaber2019]_ [Eq. 4.21].


    Parameters
    ----------
    alssms : tuple of shape=(M) of :class:`Alssm`
        Set of `M` autonomous linear state space models.
        All ALSSMs need to have the same number of output channels, see :meth:`Alssm.output_count`
    deltas: list of shape=(M) of floats, optional
        List of `M` scalar factors for each output matrix of the ALSSM in `alssms`.
        (default: deltas = None, i.e., all scalars are set to 1)
    **kwargs
        Forwarded to :class:`.ModelBase`


    |def_M|

    Examples
    --------
    Multiply two ALSSMs

    >>> alssm_p = lm.AlssmPoly(poly_degree=2, label='poly')
    >>> alssm_s = lm.AlssmSin(omega=0.5, rho=0.2, label='sin')
    >>> alssm = lm.AlssmProd((alssm_s, alssm_p), label="multi")
    >>> print(alssm)
    A =
    [[ 0.17551651  0.17551651  0.17551651 -0.09588511 -0.09588511 -0.09588511]
     [ 0.          0.17551651  0.35103302 -0.         -0.09588511 -0.19177022]
     [ 0.          0.          0.17551651 -0.         -0.         -0.09588511]
     [ 0.09588511  0.09588511  0.09588511  0.17551651  0.17551651  0.17551651]
     [ 0.          0.09588511  0.19177022  0.          0.17551651  0.35103302]
     [ 0.          0.          0.09588511  0.          0.          0.17551651]]
    C =
    [[1. 0. 0. 0. 0. 0.]]

    """

    def __init__(self, alssms, deltas=None, **kwargs):
        super().__init__(**kwargs)
        self.alssms = alssms
        self.deltas = deltas
        self.update()

    def update(self):
        for alssm in self.alssms:
            alssm.update()
        A = [1]
        C = [1]
        for alssm, delta in zip(self.alssms, self.deltas):
            A = np.kron(A, alssm.A)
            C = np.kron(C, delta * alssm.C)
        self.A = A
        self.C = C
        self._init_state_var_labels()
