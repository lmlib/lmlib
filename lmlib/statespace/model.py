"""
This module provides methods to define linear state space models and methods to use them as signal models in recursive least squares problems.


.. currentmodule:: lmlib.statespace.model

.. inheritance-diagram:: lmlib.statespace.model
   :top-classes: lmlib.statespace.model.ModelBase
   :parts: 1

"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
import re
import numpy as np
import numpy.typing as npt
from numpy.linalg import matrix_power
from scipy.linalg import block_diag, pascal
from numpy.polynomial.legendre import legder as _legder

from lmlib.utils import *
from lmlib.statespace.backends.statespace_tools import *
from lmlib.statespace.segment import Segment, FORWARD, BACKWARD

__all__ = ['Alssm', 'AlssmPoly', 'AlssmPolyJordan', 'AlssmPolyLegendre', 'AlssmPolyMeixner',
           'AlssmSin', 'AlssmExp', 'AlssmStacked', 'AlssmSum', 'AlssmProd', 'ModelBase']


class ModelBase(ABC):
    """
    Abstract baseclass for autonomous linear state space models.

    Parameters
    ----------
    label : str, optional
        Label of ALSSM, default: 'n/a'
    C_init : array_like, None, optional
        Initialized Output Matrix, default: 'None'
    force_MC : bool, optional
        Broadcasts an 1-dimensional `C`-vector to a 2-dimensional array of shape (1, N)
    """

    def __init__(self, label:str='n/a', C_init=None, force_MC:bool=False):
        self._alssms = list()
        self._lambdas = np.ndarray([])
        self._A = None
        self._C = None
        self.label = label
        self.C_init = C_init
        self._state_var_labels = dict()
        self.force_MC = force_MC

    def __str__(self):
        A_str = re.sub('\n+', ',', np.array_str(self.A).replace('\n', '')).replace('[,', '[')
        C_str = re.sub('\n+', ',', np.array_str(self.C).replace('\n', '')).replace('[,', '[')
        return f'{type(self).__name__}(A={A_str}, C={C_str}, label={self.label})'

    @abstractmethod
    def update(self):
        """
        Model update

        Updates the internal model (A and C Matrix) based on the initialization parameters of a class.
        """
        pass

    @property
    def A(self) -> npt.NDArray:
        """:class:`~numpy.ndarray`, shape=(N, N) : State matrix :math:`A \\in \\mathbb{R}^{N \\times N}`"""
        return self._A

    @A.setter
    def A(self, A):
        assert is_array_like(A), 'A is not array like'
        assert is_square(A), f'A is not square, {info_str_found_shape(A)}'
        self._A = np.asarray(A)

    @property
    def C(self) -> npt.NDArray:
        """:class:`~numpy.ndarray`, shape=([Q,] N) : Output matrix :math:`C \\in \\mathbb{R}^{Q \\times N}`"""
        return self._C

    @C.setter
    def C(self, C):
        assert is_array_like(C), 'C is not array like'
        assert is_1dim(C) or is_2dim(C), f'C is not 1 or 2 dimensional, {info_str_found_shape(C)}'
        self._C = np.asarray(C)

    @property
    def C_init(self) -> npt.NDArray:
        """:class:`~numpy.ndarray`, shape=([Q,] N) : Initialized Output matrix :math:`C \\in \\mathbb{R}^{Q \\times N}`"""
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
    def label(self) -> str:
        """str : Label of the model"""
        return self._label

    @label.setter
    def label(self, label: str):
        assert isinstance(label, str)
        self._label = label

    @property
    def N(self) ->int:
        """int : Model order :math:`N`"""
        return self._A.shape[0]

    @property
    def Q(self):
        """int : Alssm output dimension :math:`Q`"""
        return self._C.shape[0] if np.ndim(self._C) == 2 else 0

    @property
    def alssms(self) -> list:
        """list : Set of models"""
        return self._alssms

    @alssms.setter
    def alssms(self, alssms):
        assert isinstance(alssms, Iterable), 'alssms is not iterable'
        for alssm in alssms:
            assert isinstance(alssm, ModelBase), 'element in alssms is not instance nor subclass of ALSSM'
        self._alssms = list(alssms)

    @property
    def lambdas(self) -> npt.NDArray:
        """:class:`np.ndarray` : Output scaling factors for each ALSSM in `alssms`"""
        return self._lambdas

    @lambdas.setter
    def lambdas(self, lambdas):
        _n = len(self.alssms)
        if lambdas is None:
            lambdas = [1] * _n
        elif np.isscalar(lambdas):
            lambdas = [lambdas] * _n

        assert isinstance(lambdas, Iterable), 'lambdas is not iterable'
        assert np.size(lambdas) == _n, f'Output scaling factors lambdas are not of length {_n}, ' \
                                      f'{info_str_found_shape(lambdas)}'
        for lambda_ in lambdas:
            assert np.isscalar(lambda_), 'element in lambdas is not scalar'
        self._lambdas = np.asarray(lambdas)

    @property
    def force_MC(self) -> bool:
        """bool : Flag to broadcast to a 2-dimensional `C` state space variable"""
        return self._force_MC

    @force_MC.setter
    def force_MC(self, force_MC: bool):
        assert isinstance(force_MC, bool), 'force_MC is not of type boolean'
        self._force_MC = force_MC

    @ property
    def is_MC(self) -> bool:
        """bool : returns True if 'C' is MultiChannel (2d)"""
        return np.ndim(self._C) == 2

    def eval_output(self, xs, js=None):
        """
        Evaluate ALSSM outputs.

        If js is None:
            s(x) = C x
        If js is provided:
            s_j(x) = C A^j x

        Parameters
        ----------
        xs : array_like, shape=(..., N)
            State vector(s)
        js : array_like, optional, shape=(J,)
            ALSSM evaluation indices

        Returns
        -------
        s : ndarray
            If js is None:
                shape=(..., [L])
            If js is provided:
                shape=(J, ..., [L])
        """
        xs = np.asarray(xs)

        # Ensure last axis is state dimension
        if xs.shape[-1] != self.N:
            raise ValueError(f"Last dimension of xs must be {self.N}")

        # No propagation: s = C x
        if js is None:
            _subscript = 'ln,...n->...l' if self.is_MC else 'n,...n->...'
            return np.einsum(_subscript, self.C, xs)

        # Propagation: s_j = C A^j x
        A_powers = [matrix_power(self.A, int(j)) for j in js]
        _subscript = 'ln,...n->...l' if self.is_MC else 'n,...n->...'
        return np.asarray([np.einsum(_subscript, self.C @ Aj, xs) for Aj in A_powers])

    def dump_tree(self) -> str:
        """
        Returns the internal structure of the ALSSM model as a string.

        Returns
        -------
        out : str
            String representing internal model structure.

        Examples
        --------
        >>> import lmlib as lm
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

    def set_state_var_label(self, label:str, indices:tuple[int]):
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

        >>> import lmlib as lm
        >>>
        >>> alssm = lm.AlssmPoly(poly_degree=1, label='slope_with_offset')
        >>> alssm.set_state_var_label('slope', (1,))
        >>> alssm._state_var_labels
        {'x': range(0, 2), 'x0': (0,), 'x1': (1,), 'slope': (1,)}
        >>> alssm._state_var_labels['slope']
        1,

        """
        self._state_var_labels[label] = indices

    def _init_state_var_labels(self):
        for n in range(self.N):
            self._state_var_labels['x' + str(n)] = (n,)
        for n in range(self.N, 0, -1):
            self._state_var_labels['x-' + str(n)] = (-n,)
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
        Returns a list of state variable labels

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
                state_list.extend([(self.label + '.' + var_label, tuple(i + N for i in indices))])
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
        return []

    def get_alssm_output_dimension(self):
        """int : Returns Alssm output dimension :math:`Q`"""
        return self.Q

    def _broadcast_C_to_multichannel(self):
        if self.force_MC:
            self.C = np.atleast_2d(self.C)


class Alssm(ModelBase):
    r"""
    Generic Autonomous Linear State Space Model (ALSSM)

    This class holds the parameters of a discrete-time, autonomous (i.e., input-free), single- or multi-output linear
    state space model, defined recursively by

    .. math::
       x[k+1] &= Ax[k]

       s_k(x) = y[k] &= Cx[k],

    where :math:`A \in \mathbb{R}^{N\times N}, C \in \mathbb{R}^{Q \times N}` are the fixed model parameters (matrices),
    :math:`k` the time index,
    :math:`y[k] \in \mathbb{R}^{L \times 1}` the output vector,
    and :math:`x[k] \in \mathbb{R}^{N}` the state vector.

    For more details, see also [Wildhaber2019]_ [Eq. 4.1].

    Parameters
    ----------
    A : array_like, shape=(N, N)
        State Matrix
    C : array_like, shape=([Q,] N)
        Output Matrix
    **kwargs
        Forwarded to :class:`.ModelBase`

    Note
    ----
    The output matrix :math:`C` can be of the form (N,) for a 1-dimensional output or of the form (Q, N), resulting in a
    2-dimensional output. Accordingly, the signal sample :math:`y[k]` in a cost function must be in the form of the ALSSM output.

    |def_N|
    |def_Q|

    Examples
    --------

    >>> import lmlib as lm
    >>>
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
        self._broadcast_C_to_multichannel()


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
        Polynomial degree. Corresponds to the highest exponent of the polynomial.rst.
        It follows an ALSSM system of order `N = poly_degree+1`.
    C : array_like, shape=([Q,] N), optional
        Output Matrix.
        If no output matrix is given, C gets initialized automatically to `[1, 0, ...]` such that the shape
        is `(N,)`. In addition with ``as_2dim_C=True`` C gets broadcated to shape `(1, N)`. (default: C=None)
    **kwargs
        Forwarded to :class:`.ModelBase`

    Notes
    -----
    |def_N|
    |def_Q|


    Examples
    --------
    Setting up a 4. order polynomial.rst, autonomous linear state space model.

    >>> import lmlib as lm
    >>> alssm = lm.AlssmPoly(poly_degree=3, label="poly")
    >>> print(alssm)
    A =
    [[1 1 1 1]
     [0 1 2 3]
     [0 0 1 3]
     [0 0 0 1]]
    C =
    [1 0 0 0]

    Setting up a 3. order polynomial, autonomous linear state space model with two outputs.

    >>> C = np.array([[1, 0, 0], [0, 1, 0]])
    >>> alssm = lm.AlssmPoly(poly_degree=2, C=C, label="poly")
    >>> print(alssm)
    A =
    [[1 1 1]
     [0 1 2]
     [0 0 1]]
    C =
    [[1 0 0]
     [0 1 0]]

    """

    def __init__(self, poly_degree:int, C=None, **kwargs):
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
        self._broadcast_C_to_multichannel()

    @property
    def poly_degree(self) -> int:
        """int : Polynomial degree (highest exponent/ order - 1)"""
        return self._poly_degree

    @poly_degree.setter
    def poly_degree(self, poly_degree):
        assert isinstance(poly_degree, int), 'poly_degree is not of type int'
        assert poly_degree >= 0, 'poly_degree is not larger then 0'
        self._poly_degree = poly_degree


class AlssmPolyJordan(ModelBase):
    r"""
    ALSSM with a discrete-time polynomial output sequence, in Jorandian normal form


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
    |def_Q|


    Examples
    --------
    Setting up a 3th degree polynomial ALSSM.

    >>> import lmlib as lm
    >>>
    >>> polynomial_degree = 3
    >>> alssm = lm.AlssmPolyJordan(polynomial_degree, label="poly")
    >>>
    print(alssm)
    A =
    [[1. 1. 0. 0.]
     [0. 1. 1. 0.]
     [0. 0. 1. 1.]
     [0. 0. 0. 1.]]
    C =
    [[1 0 0 0]]

    """

    def __init__(self, poly_degree:int, C=None, **kwargs):
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
        self._broadcast_C_to_multichannel()

    @property
    def poly_degree(self) -> int:
        """int : Polynomial degree :math:`Q` (highest exponent/ order - 1)"""
        return self._poly_degree

    @poly_degree.setter
    def poly_degree(self, poly_degree):
        assert isinstance(poly_degree, int), 'poly_degree is not of type int'
        assert poly_degree >= 0, 'poly_degree is not larger then 0'
        self._poly_degree = poly_degree


class AlssmPolyLegendre(ModelBase):
    r"""
    ALSSM whose output basis is the discrete Legendre polynomials on a finite window.

    Unlike :class:`AlssmPoly` (Pascal/monomial basis) and :class:`AlssmPolyJordan`
    (Jordan/binomial basis), the Legendre ALSSM maps time indices to the interval
    :math:`[-1, +1]` and uses the classical Legendre polynomials
    :math:`P_0, P_1, \ldots, P_D` as its basis functions.  This keeps the Gram
    matrix :math:`W` well-conditioned regardless of the window length :math:`W_{\rm size}`:

    .. math::

        \kappa(W_{\rm Legendre}) \approx 2D + 1
        \qquad \text{vs.} \qquad
        \kappa(W_{\rm Pascal}) = \mathcal{O}\!\left(W_{\rm size}^{2D}\right)

    For a window of 500 samples and polynomial degree 4, the Gram matrix condition
    number is :math:`\approx 9` instead of :math:`\approx 10^{22}`, an improvement
    of more than 20 orders of magnitude.

    **State-space parametrisation**

    The window of :math:`W_{\rm size}` samples is mapped to
    :math:`t_{\rm sc} \in [-1, +1]` via

    .. math::

        t_{\rm sc}(j) = \frac{2j}{W_{\rm size}-1} - 1,
        \qquad j = 0 \;(\text{newest}) \;\ldots\; W_{\rm size}-1 \;(\text{oldest}).

    The state vector :math:`x \in \mathbb{R}^N` (with :math:`N = D+1`) holds the
    Legendre expansion coefficients :math:`[c_0, \ldots, c_D]` of the fitted
    polynomial:

    .. math::

        \hat{y}(t_{\rm sc}) = \sum_{n=0}^{D} c_n P_n(t_{\rm sc}).

    **Transition matrix** :math:`A`

    Advancing one step from the newest sample toward the past corresponds to
    shifting :math:`t_{\rm sc}` by :math:`h = 2/(W_{\rm size}-1)`.  The resulting
    constant upper-triangular shift matrix :math:`L` satisfies

    .. math::

        \phi(t_{\rm sc} + h) = \phi(t_{\rm sc})\,L,
        \qquad \phi(t) = [P_0(t),\, P_1(t),\, \ldots,\, P_D(t)],

    and is computed analytically via a term-by-term Taylor expansion in the
    Legendre basis using :func:`numpy.polynomial.legendre.legder`:

    .. math::

        L_{:,n} = \sum_{m=0}^{n} \frac{h^m}{m!}
                  \bigl[\text{Legendre coefficients of } P_n^{(m)}\bigr].

    :math:`\kappa(L) \approx 1` for all practical window sizes.

    **Output vector** :math:`C`

    The newest sample (reference point, :math:`j = 0 \Rightarrow t_{\rm sc} = -1`)
    is evaluated by

    .. math::

        C = \phi(-1) = \bigl[P_0(-1),\, P_1(-1),\, \ldots,\, P_D(-1)\bigr]
          = \bigl[1,\, {-1},\, 1,\, {-1},\, \ldots\bigr].

    **Compatibility**

    :class:`AlssmPolyLegendre` is a drop-in replacement for :class:`AlssmPoly` in any
    :class:`~lmlib.statespace.cost.CostSegment` or composite cost.  All lmlib
    infrastructure (:meth:`~lmlib.statespace.rls.RLSAlssm.filter`,
    :meth:`~lmlib.statespace.rls.RLSAlssm.minimize_v`,
    :meth:`eval_output`) works unchanged.

    The state vector is in Legendre coefficient space, not monomial coefficient
    space, so the numerical values of :math:`x[k]` differ from those returned by
    :class:`AlssmPoly`.  The *output* :math:`\hat{y}[k] = Cx[k]` and the
    full-window trajectory via :meth:`eval_output` with ``js`` are identical in
    meaning (predicted signal value at each lag).

    Notes
    -----
    |def_N|

    The Legendre polynomials used here are the *standard* (unnormalised)
    Legendre polynomials satisfying :math:`P_n(1) = 1`, identical to those
    returned by :func:`numpy.polynomial.legendre.legval`.  They are *not*
    normalised to :math:`\|P_n\|_{L^2} = 1`; the orthonormality factor is
    :math:`\sqrt{(2n+1)/2}`.  Because the RLS filter works with
    :math:`W = V^\top V` (where :math:`V` contains the Legendre design-matrix
    rows), the normalisation cancels out in the coefficient recovery and does
    not need to be applied explicitly.

    To convert recovered Legendre coefficients :math:`c` back to standard
    monomial coefficients, premultiply by the change-of-basis matrix
    :math:`T^{-1}` where :math:`T` satisfies :math:`V_{\rm pascal}\,T = V_{\rm Legendre}`.

    Examples
    --------
    Setting up a degree-3 Legendre ALSSM for a 500-sample window:

    >>> import lmlib as lm
    >>> alssm = lm.AlssmPolyLegendre(poly_degree=3, a_seg=0, b_seg=499, label='legendre')
    >>> print(alssm)
    AlssmPolyLegendre(A=..., C=..., label=legendre)

    Using it inside a cost segment (drop-in replacement for AlssmPoly):

    >>> import numpy as np, lmlib as lm
    >>> from lmlib.utils.generator import gen_wgn, gen_rect
    >>> K = 1000
    >>> y = gen_rect(K, 300, 100) + gen_wgn(K, 0.01)
    >>> alssm = lm.AlssmPolyLegendre(poly_degree=3, a_seg=0,b_seg=199)
    >>> cost  = lm.CostSegment(alssm, lm.Segment(0, 199, lm.BW, 500))
    >>> rls   = lm.RLSAlssm(cost)
    >>> rls.filter(y)
    >>> xs = rls.minimize_x()
    >>> y_hat = alssm.eval_output(xs)
    """

    def __init__(self, poly_degree: int, a_seg: int = 0, b_seg: int = None,
                 **kwargs):
        r"""
        Parameters
        ----------
        poly_degree : int
            Polynomial degree :math:`D \geq 0`.  The model order is :math:`N = D+1`.
        a_seg : int, optional
            Left boundary of the target segment (default ``0``).
            Together with ``b_seg`` this defines the window :math:`[a, b]`:

            * The step size :math:`h = 2 / (b - a)` maps :math:`[a, b]` to
              :math:`[-1, +1]` in the Legendre domain.
            * The output vector is shifted to
              :math:`C_{\rm new} = \phi(-1)\,A^{-a}` so that the filter
              naturally accumulates the segment-relative Gram matrix
              :math:`W_{\rm rel}` with :math:`\kappa(W_{\rm rel}) \approx 2D+1`
              — regardless of where :math:`[a, b]` sits relative to :math:`j=0`.
            * The output :math:`C_{\rm new}\,x[k]` evaluates the polynomial at
              :math:`j = 0` (the current sample :math:`y[k]`).

        b_seg : int
            Right boundary of the target segment.  Must satisfy ``b_seg > a_seg``.
            The window width is ``b_seg - a_seg + 1``.
        **kwargs
            Forwarded to :class:`ModelBase`.

        Examples
        --------
        Standard backward window of 501 samples aligned at the current sample::

            alssm = lm.AlssmPolyLegendre(poly_degree=3, a_seg=0, b_seg=500)
            seg   = lm.Segment(0, 500, lm.BW, g=100)

        Backward window shifted 200 samples into the past — same window size,
        same :math:`h`, same :math:`\kappa(W) \approx 2D+1`::

            alssm = lm.AlssmPolyLegendre(poly_degree=3, a_seg=-200, b_seg=300)
            seg   = lm.Segment(-200, 300, lm.BW, g=100)

        Forward window entirely in the past::

            alssm = lm.AlssmPolyLegendre(poly_degree=3, a_seg=-501, b_seg=-1)
            seg   = lm.Segment(-501, -1, lm.FW, g=100)
        """
        # ── parameter handling / backward compat ──────────────────────────
        if b_seg is None:
            raise TypeError(
                'AlssmPolyLegendre requires b_seg. '
                'Example: AlssmPolyLegendre(poly_degree=3, a_seg=0, b_seg=500).'
            )

        super().__init__(**kwargs)
        self._poly_degree = int(poly_degree)
        self._a_seg = int(a_seg)
        self._b_seg = int(b_seg)
        assert isinstance(poly_degree, int) and poly_degree >= 0, \
            'poly_degree must be a non-negative int'
        assert self._b_seg > self._a_seg, \
            f'b_seg ({b_seg}) must be strictly greater than a_seg ({a_seg})'
        self.update()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def poly_degree(self) -> int:
        """int : Polynomial degree :math:`D`."""
        return self._poly_degree

    @poly_degree.setter
    def poly_degree(self, v: int):
        assert isinstance(v, int) and v >= 0, 'poly_degree must be a non-negative int'
        self._poly_degree = v

    @property
    def a_seg(self) -> int:
        """int : Left boundary of the target segment."""
        return self._a_seg

    @a_seg.setter
    def a_seg(self, v: int):
        self._a_seg = int(v)

    @property
    def b_seg(self) -> int:
        """int : Right boundary of the target segment."""
        return self._b_seg

    @b_seg.setter
    def b_seg(self, v: int):
        self._b_seg = int(v)

    @property
    def h(self) -> float:
        r"""float : Legendre step size :math:`h = 2\,/\,(b\_seg - a\_seg)`."""
        return 2.0 / (self._b_seg - self._a_seg)

    # ------------------------------------------------------------------
    # ModelBase interface
    # ------------------------------------------------------------------

    def update(self):
        r"""Recompute :math:`A` and :math:`C` from ``poly_degree``, ``a_seg``, ``b_seg``.

        Step 1 — build shift matrix :math:`A = L` for step size :math:`h = 2/(b-a)`.

        Step 2 — set :math:`C = \phi(-1) = [1,\,-1,\,1,\,-1,\,\ldots]`.

        Step 3 — apply segment-relative shift (only when ``a_seg != 0``):

        .. math::

            C \;\leftarrow\; C\,A^{-a_{\rm seg}}

        For ``a_seg < 0`` (common backward window case) this is a *positive*
        power of :math:`A` — always stable.
        """
        N = self._poly_degree + 1
        self.A = self._legendre_shift_matrix(N, self.h)
        self.C = np.array([(-1.0) ** n for n in range(N)])
        if self._a_seg != 0:
            self.C = self.C.astype(float) @ np.linalg.matrix_power(
                self.A.astype(float), -self._a_seg)
        self._init_state_var_labels()
        self._broadcast_C_to_multichannel()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _legendre_shift_matrix(N: int, h: float) -> np.ndarray:
        r"""
        Compute the :math:`N \times N` upper-triangular Legendre shift matrix
        :math:`L` such that

        .. math::

            \phi(t + h) = \phi(t)\,L,
            \quad \phi(t) = [P_0(t),\, \ldots,\, P_{N-1}(t)].

        Each column :math:`L_{:,n}` is obtained by Taylor-expanding
        :math:`P_n(t + h)` around :math:`t`:

        .. math::

            P_n(t + h)
            = \sum_{m=0}^{n} \frac{h^m}{m!} P_n^{(m)}(t)
            = \sum_{m=0}^{n} \frac{h^m}{m!}
              \sum_{j} \bigl[\operatorname{legder}^m(e_n)\bigr]_j P_j(t),

        where :math:`e_n` is the :math:`n`-th standard basis vector in the
        Legendre coefficient representation and :math:`\operatorname{legder}^m`
        denotes :math:`m` successive applications of
        :func:`numpy.polynomial.legendre.legder`.

        The formula is exact (not a numerical fit) and costs :math:`O(N^2)`
        operations.  :math:`\kappa(L) \approx 1` for all practical window
        sizes.

        Parameters
        ----------
        N : int
            Matrix dimension (= poly_degree + 1).
        h : float
            Step size in the scaled coordinate :math:`t_{\rm sc}`.

        Returns
        -------
        L : ndarray of shape (N, N)
        """
        L = np.zeros((N, N))
        for n in range(N):
            # Legendre coefficient vector for P_n: all zeros except position n
            c_deriv = np.zeros(N)
            c_deriv[n] = 1.0
            col = np.zeros(N)
            h_pow_over_mfact = 1.0          # h^m / m!
            for m in range(n + 1):          # P_n^{(m)} has degree n-m >= 0
                # c_deriv holds the Legendre coefficients of P_n^{(m)};
                # pad to length N in case legder shortened it.
                pad = N - len(c_deriv)
                col += h_pow_over_mfact * np.pad(c_deriv, (0, pad))
                c_deriv = _legder(c_deriv)  # one more differentiation
                h_pow_over_mfact *= h / (m + 1)
            L[:, n] = col
        return L


class AlssmPolyMeixner(ModelBase):
    r"""
    ALSSM whose output basis is the Meixner polynomials, orthogonal under the
    geometric (exponential) weight :math:`\gamma^j` on :math:`j = 0, 1, 2, \ldots`

    Unlike :class:`AlssmPoly` (Pascal/monomial basis), which suffers from Gram
    matrix condition numbers of :math:`\mathcal{O}(g^{2D})`, and
    :class:`AlssmPolyLegendre`, which requires a *finite* window specification,
    :class:`AlssmPolyMeixner` is designed for **infinite or semi-infinite exponential
    windows** and keeps the composite Gram matrix near the theoretical minimum.

    **Mathematical background**

    The Meixner polynomials :math:`M_n(j;\,1,\gamma)` satisfy

    .. math::

        \sum_{j=0}^{\infty} \gamma^j\, M_m(j)\, M_n(j) = W_{n}\,\delta_{mn},
        \qquad W_n = g\!\left(\frac{g}{g-1}\right)^{\!n} = \frac{1}{(1-\gamma)\,\gamma^n},

    where :math:`\gamma = (g-1)/g < 1` is the exponential decay of the backward
    segment and :math:`g > 1` is its effective window length.  The norms
    :math:`W_n` are **exact and closed-form**, growing by the constant factor
    :math:`g/(g-1)` per degree. :math: `delta_{mn}` is the kronecker delta.

    **Accepting a Segment**

    The recommended constructor is ``AlssmPolyMeixner(poly_degree, segment)`` which
    infers all parameters from the :class:`Segment`:

    * ``segment.direction`` → selects :math:`A`:

      * ``'bw'`` (backward): :math:`A = A_{\rm bw} = I - \tfrac{1}{g-1}\triu(\mathbf{1},1)`
        — basis :math:`C A_{\rm bw}^j x = M_n(j;\gamma)` at lag :math:`j \ge 0`.

      * ``'fw'`` (forward): :math:`A = A_{\rm fw} = A_{\rm bw}^{-1}`
        — the forward filter's internal :math:`A^{-1} = A_{\rm bw}` step recovers
        the decaying Meixner basis at lags :math:`j \le 0`.

    * ``segment.a`` (backward) or ``segment.b`` (forward) → shifts :math:`C` so
      that the Gram matrix remains :math:`W_{\rm ss}` (diagonal) even when the
      segment does not start at :math:`j=0`:

      * Backward :math:`[a, \infty)`:
        :math:`C \leftarrow [1,\ldots,1]\,A_{\rm bw}^{-a}`
        (makes :math:`C A^j x = M_n(j-a;\gamma)` for :math:`j \ge a`).

      * Forward :math:`(-\infty, b]`:
        :math:`C \leftarrow [1,\ldots,1]\,A_{\rm bw}^{-b}`
        (makes the relative basis start at the boundary :math:`j=b`).

    The :attr:`g` and :attr:`direction` attributes are always available.

    **Condition number**

    :math:`\kappa(W) = (g/(g-1))^D`, independent of the segment shift.


    Notes
    -----
    The Meixner polynomials used here are the standard (monic normalisation)
    :math:`M_n(x;\,1,\gamma) = {}_2F_1(-n,\,-x;\,1;\,1 - 1/\gamma)`.

    """

    def __init__(self, poly_degree: int, segment=None, **kwargs):
        r"""
        Parameters
        ----------
        poly_degree : int
            Polynomial degree :math:`D \geq 0`.  The model order is :math:`N = D+1`.
        segment : :class:`Segment`, optional
            Target segment.  **Recommended.**  Encodes direction, :math:`g`, and
            boundary shift in one object:

            * ``segment.g`` → window size.
            * ``segment.direction`` → ``'bw'`` selects :math:`A_{\rm bw}`;
              ``'fw'`` selects :math:`A_{\rm fw} = A_{\rm bw}^{-1}`.
            * ``segment.a`` (BW, if finite) or ``segment.b`` (FW, if finite)
              → shifts :math:`C` to restore Gram matrix orthogonality when the
              segment does not start at the canonical origin.

            Either ``segment`` or ``g`` must be given (not both).
        **kwargs
            Forwarded to :class:`ModelBase`. """

        super().__init__(**kwargs)
        assert isinstance(poly_degree, int) and poly_degree >= 0, \
            'poly_degree must be a non-negative int'
        self._poly_degree = int(poly_degree)

 
        assert isinstance(segment, Segment), \
            "'segment' must be a lmlib.statespace.segment.Segment instance."
        # ── Validate: Meixner orthogonality requires a semi-infinite segment ──
        if segment.direction == BACKWARD and np.isfinite(segment.b):
            raise ValueError(
                f"AlssmPolyMeixner requires b=+inf for BACKWARD segments "
                f"(got b={segment.b}). The Meixner polynomials are orthogonal only on "
                f"a semi-infinite support. For finite windows use AlssmPoly or "
                f"AlssmPolyLegendre instead.")
        if segment.direction == FORWARD and np.isfinite(segment.a):
            raise ValueError(
                f"AlssmPolyMeixner requires a=-inf for FORWARD segments "
                f"(got a={segment.a}). The Meixner polynomials are orthogonal only on "
                f"a semi-infinite support. For finite windows use AlssmPoly or "
                f"AlssmPolyLegendre instead.")
        self._g = float(segment.g)
        self._direction = segment.direction
        # ── Determine C shift from the finite boundary ──────────────────────
        #
        # BACKWARD [a, ∞):
        #   C ← [1..1] A_bw^{-a}  →  output at lag j = M_n(j-a; γ).
        #   This keeps W = W_ss regardless of a (the Stein boundary term Q_b
        #   = C^T C = ones at relative lag 0).
        #
        # FORWARD (-∞, b]:
        #   A = A_fw = A_bw^{-1}.
        #   C ← [1..1] A_bw^{-|b|} = [1..1] A_fw^{|b|}
        #   This makes Q_b = outer([1..1],[1..1]) in the Stein equation,
        #   keeping W = W_ss regardless of b (standard b=-1 → shift=1).
        if segment.direction == BACKWARD:
            self._shift = 0 if np.isinf(segment.a) else int(segment.a)
        else:  # FORWARD
            self._shift = 0 if np.isinf(segment.b) else int(abs(segment.b))

        self.update()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def poly_degree(self) -> int:
        """int : Polynomial degree :math:'D'."""
        return self._poly_degree

    @poly_degree.setter
    def poly_degree(self, v: int):
        assert isinstance(v, int) and v >= 0, 'poly_degree must be a non-negative int'
        self._poly_degree = v

    @property
    def g(self) -> float:
        r"""float : Effective window size :math:`g`."""
        return self._g

    @g.setter
    def g(self, v: float):
        assert np.isscalar(v) and v > 1.0, 'g must be a scalar > 1'
        self._g = float(v)

    @property
    def direction(self) -> str:
        r"""str : Segment direction ``'bw'`` (backward) or ``'fw'`` (forward)."""
        return self._direction

    @property
    def shift(self) -> int:
        r"""int : Boundary shift applied to :math:`C`; 0 for the canonical origin."""
        return self._shift

    @property
    def gamma(self) -> float:
        r"""float : Exponential decay :math:`\gamma = (g-1)/g`."""
        return (self._g - 1.0) / self._g

    # ------------------------------------------------------------------
    # ModelBase interface
    # ------------------------------------------------------------------

    def update(self):
        r"""Recompute :math:`A` and :math:`C` from all stored parameters.

        **Transition matrix** :math:`A`:

        * direction ``'bw'``:
          :math:`A = A_{\rm bw} = I_N - \tfrac{1}{g-1}\,\triu(\mathbf{1}_N, 1)`
        * direction ``'fw'``:
          :math:`A = A_{\rm fw} = A_{\rm bw}^{-1}`

        **Output vector** :math:`C` (before shift):
        :math:`C_{\rm base} = [1,\ldots,1]`.

        **Shift** (when ``shift != 0``):
        :math:`C \leftarrow C_{\rm base}\,A_{\rm bw}^{-\text{shift}}`.

        This ensures that for a backward segment :math:`[a, \infty)` with
        ``shift = a``, the output at absolute lag :math:`j` is
        :math:`M_n(j - a;\,\gamma)` — orthogonal under the shifted window weight.
        """
        N = self._poly_degree + 1
        A_bw = np.eye(N) - (1.0 / (self._g - 1.0)) * np.triu(np.ones((N, N)), 1)

        if self._direction == 'fw':
            self.A = np.linalg.inv(A_bw)
        else:
            self.A = A_bw

        C_base = np.ones(N)
        if self._shift != 0:
            # C @ A_bw^{-shift}: matrix_power handles negative exponents via inverse.
            C_base = C_base @ matrix_power(A_bw, -self._shift)
        self.C = C_base

        self._init_state_var_labels()
        self._broadcast_C_to_multichannel()


class AlssmSin(ModelBase):
    r"""
    ALSSM with a discrete-time (damped) sinusoidal output sequence.

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
        Frequency :math:`\omega = 2\pi f_s`
    rho : float, int, optional
        Decay factor, default: rho = 1.0
    C : array_like, shape=([Q,] N), optional
        Output Matrix.
        If no output matrix is given, C gets initialized automatically to `[1, 0]`, such that the shape
        is `(N,)`. In addition, with ``as_2dim_C=True`` C gets broadcated to shape `(1, N)` (default: C=None)
    **kwargs
        Forwarded to :class:`.ModelBase`

    Notes
    -----
    |def_N|
    |def_Q|


    Notes
    -----
    To convert a continuous-time frequency to a normalized frequency, see :func:`~lmlib.utils.generator.k_period_to_omega`, e.g.,

    >>> import lmlib as lm
    >>> from lmlib.utils.generator import k_period_to_omega
    >>> k_period = 20
    >>> alssm = lm.AlssmSin(k_period_to_omega(k_period), rho=0.9)


    Examples
    --------
    Parametrization of a sinusoidal ALSSM:

    >>> alssm = lm.AlssmSin(omega= 0.1, rho= 0.9)
    >>> print(alssm)
    A =
    [[ 0.89550375 -0.08985007]
     [ 0.08985007  0.89550375]]
    C =
    [1 0]

    """

    def __init__(self, omega:float, rho:float=1.0, C=None, **kwargs):
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
        self._broadcast_C_to_multichannel()

    @property
    def omega(self) -> float:
        """float : Frequency factor :math:`\\omega = 2\\pi f_s`"""
        return self._omega

    @omega.setter
    def omega(self, omega: float):
        assert np.isscalar(omega), 'Frequency factor omega is a scalar'
        self._omega = float(omega)

    @property
    def rho(self) -> float:
        """float : Decay factor :math:`\\rho`"""
        return self._rho

    @rho.setter
    def rho(self, rho: float):
        assert np.isscalar(rho), 'Decay factor rho is not a scalar'
        self._rho = float(rho)


class AlssmExp(ModelBase):
    """
    ALSSM with a discrete-time exponentially decaying/increasing output sequence.

    Discrete-time linear state space model generating output sequences of exponentially decaying shape with a decay
    factor :math:`\\gamma`.

    For more details see [Wildhaber2019]_ :download:`PDF <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=48>`

    Parameters
    ----------
    gamma : float, int
        Decay factor per sample step ( > 1.0 : left-sided decaying; < 1.0 : right-sided decaying)
    C : array_like, shape=([Q,], N), optional
        Output Matrix.
        If no output matrix is given, C gets initialize automatically to `[[1]]`. (default: C=None)
    **kwargs
        Forwarded to :class:`.ModelBase`

    Notes
    -----
    |def_N|
    |def_Q|


    Examples
    --------
    Parametrizing an exponentially ALSSM:

    >>> import lmlib as lm
    >>> alssm = lm.AlssmExp(gamma= 0.8)
    >>> print(alssm)
    A =
    [[0.8]]
    C =
    [[1.]]

    """

    def __init__(self, gamma:float, C=None, **kwargs):
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
        self._broadcast_C_to_multichannel()

    @property
    def gamma(self) -> float:
        """float :  Decay factor per sample :math:`\\gamma`"""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float):
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
    alssms : tuple of ALSSMs
        Set of `M` autonomous linear state space models
    lambdas: list of floats, optional
        List of `M` scalar factors for each output matrix of the ALSSM in `alssms`.
        (default: lambdas = None, i.e., all scalars are set to 1)
    **kwargs
        Forwarded to :class:`.ModelBase`

    Notes
    -----
    |def_M|

    Examples
    --------

    >>> import lmlib as lm
    >>>
    >>> alssm_poly = lm.AlssmPoly(poly_degree=4, label="high order polynomial")
    >>>
    >>> A = np.array([[1, 1], [0, 1]])
    >>> C = np.array([[1, 0]])
    >>> alssm_line = lm.Alssm(A, C, label="line")
    >>> stacked_alssm = lm.AlssmStacked([alssm_poly, alssm_line], label='stacked model')
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

    def __init__(self, alssms, lambdas=None, **kwargs):
        super().__init__(**kwargs)
        self.alssms = alssms
        self.lambdas = lambdas
        self.update()

    def update(self):
        for alssm in self.alssms:
            alssm.update()
        A = block_diag(*[alssm.A for alssm in self.alssms])
        C = block_diag(*[g * alssm.C for g, alssm in zip(self.lambdas, self.alssms)])
        self.A = A
        self.C = C
        self._init_state_var_labels()
        self._broadcast_C_to_multichannel()


class AlssmSum(ModelBase):
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
    lambdas: list of shape=(M) of floats, optional
        List of `M` scalar factors for each output matrix of the ALSSM in `alssms`.
        (default: lambdas = None, i.e., all scalars are set to 1)
    **kwargs
        Forwarded to :class:`.ModelBase`



    |def_M|

    Examples
    --------

    >>> import lmlib as lm
    >>>
    >>> alssm_poly = lm.AlssmPoly(poly_degree=3)
    >>>
    >>> A = np.array([[1, 1], [0, 1]])
    >>> C = np.array([[1, 0]])
    >>> alssm_line = lm.Alssm(A, C)
    >>> stacked_alssm = lm.AlssmSum((alssm_poly, alssm_line))
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

    def __init__(self, alssms, lambdas=None, **kwargs):
        super().__init__(**kwargs)
        self.alssms = alssms
        self.lambdas = lambdas if lambdas is not None else [1.0] * len(self.alssms)
        self.update()

    def update(self):
        for alssm in self.alssms:
            alssm.update()
        assert common_C_dim(self.alssms), "Alssms has not same output dimensions."
        A = block_diag(*[alssm.A for alssm in self.alssms])
        C = np.hstack([g * alssm.C for g, alssm in zip(self.lambdas, self.alssms)])
        self.A = A
        self.C = C
        self._init_state_var_labels()
        self._broadcast_C_to_multichannel()


AlssmStackedSO = DeprecationHelper(AlssmSum, "AlssmStackedSO() is deprecated. Use AlssmSum()!")


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
    lambdas: list of shape=(M) of floats, optional
        List of `M` scalar factors for each output matrix of the ALSSM in `alssms`.
        (default: lambdas = None, i.e., all scalars are set to 1)
    **kwargs
        Forwarded to :class:`.ModelBase`


    |def_M|

    Examples
    --------
    Multiply two ALSSMs

    >>> import lmlib as lm
    >>>
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

    def __init__(self, alssms, lambdas=None, **kwargs):
        super().__init__(**kwargs)
        self.alssms = alssms
        self.lambdas = lambdas
        self.update()

    def update(self):
        for alssm in self.alssms:
            alssm.update()
        A = [1]
        C = [1]
        for alssm, lambda_ in zip(self.alssms, self.lambdas):
            A = np.kron(A, alssm.A)
            C = np.kron(C, lambda_ * alssm.C)
        self.A = A
        self.C = C
        self._init_state_var_labels()
        self._broadcast_C_to_multichannel()
