r"""
This module provides methods to define linear state space models and methods to use them as signal models in recursive least squares problems.
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
from lmlib.utils.check import DeprecationHelper
from lmlib.statespace.backends.statespace_tools import *
from lmlib.statespace.segment import Segment, FORWARD, BACKWARD

__all__ = ['Alssm', 'AlssmPoly', 'AlssmPolyJordan', 'AlssmPolyLegendre', 'AlssmPolyMeixner',
           'AlssmSin', 'AlssmExp', 'AlssmStacked', 'AlssmSum', 'AlssmProd', 'ModelBase']


class ModelBase(ABC):
    r"""
    Abstract base class for autonomous linear state space models (ALSSMs).

    Parameters
    ----------
    label : str, optional
        Label of the ALSSM. Default: ``'n/a'``.
    C_init : array_like or None, optional
        Initial output matrix stored for use by [`update`][lmlib.statespace.model.ModelBase.update]. Default: None.
    force_MC : bool, optional
        If True, broadcasts a 1-dimensional ``C`` vector to a 2-dimensional
        array of shape ``(1, N)`` (multi-channel form). Default: False.
    """

    #: Polynomial basis tag consulted by the steady-state Gram-matrix solver
    #: (``lmlib.statespace.backends.steady_state``). ``None`` means "unknown",
    #: in which case the solver falls back to pattern-matching the transition
    #: matrix ``A`` (additionally guarded by ``C``). Subclasses with a known
    #: basis override this so the solver never misidentifies one basis as
    #: another (e.g. a degree-1 Jordan block is identical to a Legendre ``h=1``
    #: shift matrix).
    steady_state_basis = None

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
        """Return a compact string representation showing A, C, and the label."""
        A_str = re.sub('\n+', ',', np.array_str(self.A).replace('\n', '')).replace('[,', '[')
        C_str = re.sub('\n+', ',', np.array_str(self.C).replace('\n', '')).replace('[,', '[')
        return f'{type(self).__name__}(A={A_str}, C={C_str}, label={self.label})'

    @abstractmethod
    def update(self):
        r"""
        Recompute the internal model matrices A and C.

        Updates [`A`][lmlib.statespace.model.ModelBase.A] and [`C`][lmlib.statespace.model.ModelBase.C] from the stored initialisation parameters
        (e.g. ``poly_degree``, ``omega``, ``gamma``). Called automatically during
        ``__init__`` and should be called again after manually changing any parameter.
        """
        pass

    @property
    def A(self) -> npt.NDArray:
        r"""[`ndarray`][numpy.ndarray], shape=(N, N) : State matrix $A \in \mathbb{R}^{N \times N}$"""
        return self._A

    @A.setter
    def A(self, A):
        """Validate and set the state transition matrix A."""
        assert is_array_like(A), 'A is not array like'
        assert is_square(A), f'A is not square, {info_str_found_shape(A)}'
        self._A = np.asarray(A)

    @property
    def C(self) -> npt.NDArray:
        r"""[`ndarray`][numpy.ndarray], shape=([Q,] N) : Output matrix $C \in \mathbb{R}^{Q \times N}$"""
        return self._C

    @C.setter
    def C(self, C):
        """Validate and set the output matrix C."""
        assert is_array_like(C), 'C is not array like'
        assert is_1dim(C) or is_2dim(C), f'C is not 1 or 2 dimensional, {info_str_found_shape(C)}'
        self._C = np.asarray(C)

    @property
    def C_init(self) -> npt.NDArray:
        r"""[`ndarray`][numpy.ndarray], shape=([Q,] N) : Initialized Output matrix $C \in \mathbb{R}^{Q \times N}$"""
        return self._C_init

    @C_init.setter
    def C_init(self, C):
        r"""Store the initial output matrix used by [`update`][lmlib.statespace.model.ModelBase.update]."""
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
        """Set the model label (must be a string)."""
        assert isinstance(label, str)
        self._label = label

    @property
    def N(self) ->int:
        r"""int : Model order $N$"""
        return self._A.shape[0]

    @property
    def Q(self):
        r"""
        int : Number of output channels $Q$.

        Returns the first dimension of [`C`][lmlib.statespace.model.ModelBase.C] when ``C`` is 2-D (multi-channel),
        or ``0`` when ``C`` is 1-D (single scalar output channel).

        Note: ``Q=0`` is used as a sentinel meaning "scalar output" (a single channel).
        It does **not** mean zero channels.

        See also [`is_MC`][lmlib.statespace.model.ModelBase.is_MC].
        """
        return self._C.shape[0] if np.ndim(self._C) == 2 else 0

    @property
    def alssms(self) -> list:
        r"""list : Sub-ALSSMs that compose this model (empty for leaf nodes such as [`Alssm`][lmlib.statespace.model.Alssm])."""
        return self._alssms

    @alssms.setter
    def alssms(self, alssms):
        """Validate and set the list of sub-ALSSMs."""
        assert isinstance(alssms, Iterable), 'alssms is not iterable'
        for alssm in alssms:
            assert isinstance(alssm, ModelBase), 'element in alssms is not instance nor subclass of ALSSM'
        self._alssms = list(alssms)

    @property
    def lambdas(self) -> npt.NDArray:
        r"""[`ndarray`][numpy.ndarray] : Per-ALSSM scalar output scaling factors $\lambda_m$ applied to each sub-model's output matrix $C_m$."""
        return self._lambdas

    @lambdas.setter
    def lambdas(self, lambdas):
        """Validate and set the per-ALSSM output scaling factors."""
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
        """bool : If True, a 1-D output vector ``C`` is broadcast to a 2-D array of shape ``(1, N)`` (multi-channel form)."""
        return self._force_MC

    @force_MC.setter
    def force_MC(self, force_MC: bool):
        """Set the multi-channel broadcast flag (must be bool)."""
        assert isinstance(force_MC, bool), 'force_MC is not of type boolean'
        self._force_MC = force_MC

    @ property
    def is_MC(self) -> bool:
        """bool : True if the output matrix ``C`` is 2-D (multi-channel form), False if 1-D (scalar output)."""
        return np.ndim(self._C) == 2

    def eval_output(self, xs, js=None):
        r"""
        Evaluate the ALSSM output for one or more state vectors.

        Without evaluation index (``js=None``):

        $$
        s(x) = C x
        $$

        With evaluation indices (``js`` provided):

        $$
        s_j(x) = C A^j x
        $$

        Parameters
        ----------
        xs : array_like of shape (..., N)
            State vector(s). The last dimension must equal the model order N.
        js : array_like of shape (J,) or None, optional
            Sequence of integer evaluation indices. If None, evaluates at
            $j = 0$ only (i.e. returns $Cx$).

        Returns
        -------
        s : ndarray
            If ``js`` is None: shape ``(..., [Q])``.
            If ``js`` is provided: shape ``(J, ..., [Q])``.
            The ``[Q]`` dimension is present only when [`is_MC`][lmlib.statespace.model.ModelBase.is_MC] is True.
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
        Return the internal ALSSM tree structure as a string.

        Returns
        -------
        out : str
            Multi-line string representing the nested ALSSM structure.

        Example
        --------
        ```python
        >>> import lmlib as lm
        >>> import numpy as np
        >>> alssm_poly = lm.AlssmPoly(4, label="high order polynomial")
        >>> A = [[1, 1], [0, 1]]
        >>> C = [[1, 0]]
        >>> alssm_line = lm.Alssm(A, C, label="line")
        >>> stacked_alssm = lm.AlssmStacked((alssm_poly, alssm_line), label='stacked model')
        >>> print(stacked_alssm.dump_tree())
        └-AlssmStacked, A: (7, 7), C: (2, 7), label: stacked model
          └-AlssmPoly, A: (5, 5), C: (5,), label: high order polynomial
          └-Alssm, A: (2, 2), C: (1, 2), label: line
        ```
        """
        return self._rec_tree(level=0)

    def set_state_var_label(self, label:str, indices:tuple[int]):
        r"""
        Register a label for one or more state vector indices.

        Labels allow state components to be referenced by name rather than by
        numeric index; see [`get_state_var_indices`][lmlib.statespace.model.ModelBase.get_state_var_indices].

        Parameters
        ----------
        label : str
            Label name to register.
        indices : tuple of int
            State vector indices associated with this label.

        Example
        --------
        ```python
        >>> import lmlib as lm
        >>> alssm = lm.AlssmPoly(poly_degree=1, label='slope_with_offset')
        >>> alssm.set_state_var_label('slope', (1,))
        >>> alssm.get_state_var_indices('slope_with_offset.slope')
        (1,)
        ```
        """
        self._state_var_labels[label] = indices

    def _init_state_var_labels(self):
        """Initialise the internal state-variable label dictionary from sub-ALSSMs."""
        for n in range(self.N):
            self._state_var_labels['x' + str(n)] = (n,)
        for n in range(self.N, 0, -1):
            self._state_var_labels['x-' + str(n)] = (-n,)
        self._state_var_labels['x'] = list(range(self.N))

    def _rec_tree(self, level):
        """
        Recursively build the state-variable label tree from nested ALSSMs.

        Parameters
        ----------
        prefix : str
            Dot-separated label prefix accumulated during recursion.
        alssm : ModelBase
            Current ALSSM node to process.
        """
        str_self = f'{type(self).__name__}, A: {str(self.A.shape)}, C: {str(self.C.shape)}, label: {self.label}'
        str_tree = ('  ' * level + '└-' + str_self)
        if len(self.alssms) != 0:
            for alssm in self.alssms:
                str_tree += '\n'
                str_tree += alssm._rec_tree(level=level + 1)
        return str_tree

    def get_state_var_labels(self):
        r"""
        Return all registered state-variable labels together with their index tuples.

        Labels are accumulated recursively from all nested sub-ALSSMs, with
        each label prefixed by the current model's [`label`][lmlib.statespace.model.ModelBase.label].  The state
        indices are adjusted to reflect the position within the combined
        (block-diagonal) state vector.

        Returns
        -------
        out : list of (str, tuple of int)
            List of ``(label_string, indices)`` pairs.  ``label_string`` is a
            dot-separated path (e.g. ``'stacked.poly.x0'``) and ``indices`` is
            the corresponding tuple of integer state-vector positions.
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
        r"""
        Return the state-vector indices for a state variable identified by its label.

        Parameters
        ----------
        label : str
            Fully qualified state label (dot-separated path), as returned by
            [`get_state_var_labels`][lmlib.statespace.model.ModelBase.get_state_var_labels].

        Returns
        -------
        out : tuple of int or list of int
            State-vector indices associated with ``label``.
            Returns an empty list if ``label`` is not found.
        """

        for l, indices in self.get_state_var_labels():
            if label == l:
                return indices
        return []

    def get_alssm_output_dimension(self):
        r"""Return the ALSSM output dimension $Q$ (number of output channels)."""
        return self.Q

    def _broadcast_C_to_multichannel(self):
        """
        Broadcast a 1-D output vector C to shape (1, N) when ``force_MC`` is True.

        Has no effect when C is already 2-D or when ``force_MC`` is False.
        """
        if self.force_MC:
            self.C = np.atleast_2d(self.C)


class Alssm(ModelBase):
    r"""
    Generic Autonomous Linear State Space Model (ALSSM)

    This class holds the parameters of a discrete-time, autonomous (i.e., input-free), single- or multi-output linear
    state space model, defined recursively by

    $$
    \begin{aligned}
    x[k+1] &= A\,x[k] \\
    s_k(x) = y[k] &= C\,x[k],
    \end{aligned}
    $$

    where $A \in \mathbb{R}^{N\times N}, C \in \mathbb{R}^{Q \times N}$ are the fixed model parameters (matrices),
    $k$ the time index,
    $y[k] \in \mathbb{R}^{Q \times 1}$ the output vector,
    and $x[k] \in \mathbb{R}^{N}$ the state vector.

    For more details, see also [\[Wildhaber2019\]](../../bibliography.md#wildhaber2019) [Eq. 4.1].

    Parameters
    ----------
    A : array_like, shape=(N, N)
        State Matrix
    C : array_like of shape ([Q,] N)
        Output matrix. Use shape ``(N,)`` for a scalar (single-channel) output
        or shape ``(Q, N)`` for a Q-channel output.
    **kwargs
        Forwarded to [`ModelBase`][lmlib.statespace.model.ModelBase]

    Note
    ----
    When ``C`` has shape ``(N,)``, each signal sample ``y[k]`` in a cost function
    must be a scalar. When ``C`` has shape ``(Q, N)``, each sample must be a
    vector of length Q.

    `N` : ALSSM system order, corresponding to the number of state variables <br>
    `Q` : output order / number of signal channels <br>

    Example
    --------
    ```python
    >>> import lmlib as lm
    >>>
    >>> A = [[1, 1], [0, 1]]
    >>> C = [1, 0]
    >>> alssm = lm.Alssm(A, C, label='line')
    >>> print(alssm)
    Alssm(A=[[1 1] [0 1]], C=[1 0], label=line)
    ```
    """

    def __init__(self, A, C, **kwargs):
        super().__init__(C_init=C, **kwargs)
        self.A = A
        r"""[`ndarray`][numpy.ndarray], shape=(N, N) : State matrix $A \in \mathbb{R}^{N \times N}$"""
        self.C = C
        r"""[`ndarray`][numpy.ndarray], shape=([Q,] N) : Output matrix $C \in \mathbb{R}^{Q \times N}$"""
        self.update()

    def update(self):
        """Reapply ``C_init`` and re-broadcast C to multi-channel form if needed."""
        self.C = self.C_init
        self._init_state_var_labels()
        self._broadcast_C_to_multichannel()


class AlssmPoly(ModelBase):
    r"""
    ALSSM with discrete-time polynomial output sequence.

    Represents a degree-D polynomial in the propagation index $i$:

    $$
    P_Q(i) = x_0 i^0 + x_1 i^1 + \cdots + x_D i^D
    $$

    as an ALSSM with Pascal upper-triangular transition matrix, e.g. for $D = 2$:

    $$
    \begin{aligned}
    A =
    \begin{bmatrix}
        1 & 1 & 1 \\
        0 & 1 & 2 \\
        0 & 0 & 1
    \end{bmatrix}
    \end{aligned}
    $$

    and state vector

    $$
    \begin{aligned}
    x =
    \begin{bmatrix}
        x_0 & x_1 & ... & x_N \\
    \end{bmatrix}^T
    \end{aligned}
    $$

    where $x_n$ is the coefficient of the $i^n$ term.

    For more details see [\[Wildhaber2019\]](../../bibliography.md#wildhaber2019) [PDF](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=48)


    Parameters
    ----------
    poly_degree : int
        Polynomial degree $D$. The model order is $N = D + 1$.
    C : array_like of shape ([Q,] N), optional
        Output matrix. If not given, defaults to ``[1, 0, ..., 0]`` of shape ``(N,)``.
        With ``force_MC=True`` the shape
        is broadcast to ``(1, N)``. Default: None.
    **kwargs
        Forwarded to [`ModelBase`][lmlib.statespace.model.ModelBase]

    Notes
    -----
    `N` : ALSSM system order, corresponding to the number of state variables <br>
    `Q` : output order / number of signal channels <br>


    Example
    --------
    Setting up a degree-3 polynomial ALSSM:
    ```python
    >>> import lmlib as lm
    >>> alssm = lm.AlssmPoly(poly_degree=3, label="poly")
    >>> print(alssm)
    AlssmPoly(A=[[1 1 1 1],[0 1 2 3],[0 0 1 3],[0 0 0 1]], C=[1 0 0 0], label=poly)
    ```

    Setting up a degree-2 polynomial ALSSM with two output channels:
    ```python
    >>> import numpy as np
    >>> C = np.array([[1, 0, 0], [0, 1, 0]])
    >>> alssm = lm.AlssmPoly(poly_degree=2, C=C, label="poly")
    >>> print(alssm)
    AlssmPoly(A=[[1 1 1],[0 1 2],[0 0 1]], C=[[1 0 0],[0 1 0]], label=poly)
    ```
    """

    steady_state_basis = 'pascal'

    def __init__(self, poly_degree:int, C=None, **kwargs):
        super().__init__(C_init=C, **kwargs)
        self.poly_degree = poly_degree
        self.update()

    def update(self):
        """Recompute the Pascal upper-triangular A and default C from ``poly_degree``."""
        if self.C_init is None:
            self.C = np.hstack([[1], [0] * self.poly_degree])
        else:
            self.C = self.C_init
        self.A = pascal(self.poly_degree + 1, kind='upper')
        self._init_state_var_labels()
        self._broadcast_C_to_multichannel()

    @property
    def poly_degree(self) -> int:
        r"""int : Polynomial degree $D$ (highest exponent / model order - 1)."""
        return self._poly_degree

    @poly_degree.setter
    def poly_degree(self, poly_degree):
        """Validate and set the polynomial degree (non-negative int)."""
        assert isinstance(poly_degree, int), 'poly_degree is not of type int'
        assert poly_degree >= 0, 'poly_degree is not larger then 0'
        self._poly_degree = poly_degree


class AlssmPolyJordan(ModelBase):
    r"""
    ALSSM with a discrete-time polynomial output sequence in Jordan normal form.

    Discrete-time polynomial ALSSM with a shift-register (Jordan) transition matrix;
    see [\[Zalmai2017\]](../../bibliography.md#zalmai2017)
    [PDF](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/176652/zalmai_thesis.pdf#page=41).


    $$
    \begin{aligned}
    A =
    \begin{bmatrix}
        1 & 1 & 0 \\
        0 & 1 & 1 \\
        0 & 0 & 1
    \end{bmatrix}
    \end{aligned}
    $$

    Parameters
    ----------
    poly_degree : int
        Polynomial degree $D$. The model order is $N = D + 1$.
    C : array_like of shape ([Q,] N), optional
        Output matrix. If not given, defaults to ``[1, 0, ..., 0]`` of shape ``(N,)``.
        Default: None.
    **kwargs
        Forwarded to [`ModelBase`][lmlib.statespace.model.ModelBase]


    `N` : ALSSM system order, corresponding to the number of state variables <br>
    `Q` : output order / number of signal channels <br>


    Example
    --------
    Setting up a degree-3 polynomial ALSSM in Jordan normal form:
    ```python
    >>> import lmlib as lm
    >>> alssm = lm.AlssmPolyJordan(3, label="poly")
    >>> print(alssm)
    AlssmPolyJordan(A=[[1. 1. 0. 0.],[0. 1. 1. 0.],[0. 0. 1. 1.],[0. 0. 0. 1.]], C=[1. 0. 0. 0.], label=poly)
    ```
    """

    steady_state_basis = 'jordan'

    def __init__(self, poly_degree:int, C=None, **kwargs):
        super().__init__(C_init=C, **kwargs)
        self.poly_degree = poly_degree
        self.update()

    def update(self):
        """Recompute the Jordan-normal-form A and default C from ``poly_degree``."""
        if self.C_init is None:
            self.C = np.hstack([[1], [0] * self.poly_degree])
        else:
            self.C = self.C_init
        self.A = np.eye(self.poly_degree + 1) + np.diagflat(np.ones(self.poly_degree), 1)
        self._init_state_var_labels()
        self._broadcast_C_to_multichannel()

    @property
    def poly_degree(self) -> int:
        r"""int : Polynomial degree $D$ (highest exponent / model order - 1)."""
        return self._poly_degree

    @poly_degree.setter
    def poly_degree(self, poly_degree):
        """Validate and set the polynomial degree (non-negative int)."""
        assert isinstance(poly_degree, int), 'poly_degree is not of type int'
        assert poly_degree >= 0, 'poly_degree is not larger then 0'
        self._poly_degree = poly_degree


class AlssmPolyLegendre(ModelBase):
    r"""
    ALSSM whose output basis is the discrete Legendre polynomials on a finite window.

    Unlike [`AlssmPoly`][lmlib.statespace.model.AlssmPoly] (Pascal/monomial basis) and [`AlssmPolyJordan`][lmlib.statespace.model.AlssmPolyJordan]
    (Jordan/binomial basis), the Legendre ALSSM maps time indices to the interval
    $[-1, +1]$ and uses the classical Legendre polynomials
    $P_0, P_1, \ldots, P_D$ as its basis functions.  This keeps the Gram
    matrix $W$ well-conditioned regardless of the window length $W_{\rm size}$:

    $$
    \kappa(W_{\rm Legendre}) \approx 2D + 1
    \qquad \text{vs.} \qquad
    \kappa(W_{\rm Pascal}) = \mathcal{O}\!\left(W_{\rm size}^{2D}\right)
    $$

    For a window of 500 samples and polynomial degree 4, the Gram matrix condition
    number is $\approx 9$ instead of $\approx 10^{22}$, an improvement
    of more than 20 orders of magnitude.

    **State-space parametrisation**

    The window of $W_{\rm size}$ samples is mapped to
    $t_{\rm sc} \in [-1, +1]$ via

    $$
    t_{\rm sc}(j) = \frac{2j}{W_{\rm size}-1} - 1,
    \qquad j = 0 \;(\text{newest}) \;\ldots\; W_{\rm size}-1 \;(\text{oldest}).
    $$

    The state vector $x \in \mathbb{R}^N$ (with $N = D+1$) holds the
    Legendre expansion coefficients $[c_0, \ldots, c_D]$ of the fitted
    polynomial:

    $$
    \hat{y}(t_{\rm sc}) = \sum_{n=0}^{D} c_n P_n(t_{\rm sc}).
    $$

    **Transition matrix** $A$

    Advancing one step from the newest sample toward the past corresponds to
    shifting $t_{\rm sc}$ by $h = 2/(W_{\rm size}-1)$.  The resulting
    constant upper-triangular shift matrix $L$ satisfies

    $$
    \phi(t_{\rm sc} + h) = \phi(t_{\rm sc})\,L,
    \qquad \phi(t) = [P_0(t),\, P_1(t),\, \ldots,\, P_D(t)],
    $$

    and is computed analytically via a term-by-term Taylor expansion in the
    Legendre basis using [`legder`][numpy.polynomial.legendre.legder]:

    $$
    L_{:,n} = \sum_{m=0}^{n} \frac{h^m}{m!}
              \bigl[\text{Legendre coefficients of } P_n^{(m)}\bigr].
    $$

    $\kappa(L) \approx 1$ for all practical window sizes.

    **Output vector** $C$

    The newest sample (reference point, $j = 0 \Rightarrow t_{\rm sc} = -1$)
    is evaluated by

    $$
    C = \phi(-1) = \bigl[P_0(-1),\, P_1(-1),\, \ldots,\, P_D(-1)\bigr]
      = \bigl[1,\, {-1},\, 1,\, {-1},\, \ldots\bigr].
    $$

    **Compatibility**

    The state vector is in Legendre coefficient space, not monomial coefficient
    space, so the numerical values of $x[k]$ differ from those returned by
    [`AlssmPoly`][lmlib.statespace.model.AlssmPoly].  The *output* $\hat{y}[k] = Cx[k]$ and the
    full-window trajectory via [`eval_output`][lmlib.statespace.model.ModelBase.eval_output] with ``js`` are identical in
    meaning (predicted signal value at each lag).

    Notes
    -----
    `N` : ALSSM system order, corresponding to the number of state variables <br>

    The Legendre polynomials used here are the *standard* (unnormalised)
    Legendre polynomials satisfying $P_n(1) = 1$, identical to those
    returned by [`legval`][numpy.polynomial.legendre.legval].  They are *not*
    normalised to $\|P_n\|_{L^2} = 1$; the orthonormality factor is
    $\sqrt{(2n+1)/2}$.  Because the RLS filter works with
    $W = V^\top V$ (where $V$ contains the Legendre design-matrix
    rows), the normalisation cancels out in the coefficient recovery and does
    not need to be applied explicitly.

    To convert recovered Legendre coefficients $c$ back to standard
    monomial coefficients, premultiply by the change-of-basis matrix
    $T^{-1}$ where $T$ satisfies $V_{\rm pascal}\,T = V_{\rm Legendre}$.

    Example
    --------
    Setting up a degree-3 Legendre ALSSM for a 500-sample window:

    >>> import lmlib as lm
    >>> alssm = lm.AlssmPolyLegendre(poly_degree=3, a_seg=0, b_seg=499, label='legendre')
    >>> print(alssm)
    AlssmPolyLegendre(A=..., C=..., label=legendre)
    ```

    Using it inside a cost segment (drop-in replacement for AlssmPoly):
    ```python
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
    ```
    """

    steady_state_basis = 'legendre'

    def __init__(self, poly_degree: int, a_seg: int = 0, b_seg: int = None,
                 **kwargs):
        r"""
        Parameters
        ----------
        poly_degree : int
            Polynomial degree $D \geq 0$.  The model order is $N = D+1$.
        a_seg : int, optional
            Left boundary of the target segment (default ``0``).
            Together with ``b_seg`` this defines the window $[a, b]$:

            * The step size $h = 2 / (b - a)$ maps $[a, b]$ to
              $[-1, +1]$ in the Legendre domain.
            * The output vector is shifted to
              $C_{\rm new} = \phi(-1)\,A^{-a}$ so that the filter
              naturally accumulates the segment-relative Gram matrix
              $W_{\rm rel}$ with $\kappa(W_{\rm rel}) \approx 2D+1$
              — regardless of where $[a, b]$ sits relative to $j=0$.
            * The output $C_{\rm new}\,x[k]$ evaluates the polynomial at
              $j = 0$ (the current sample $y[k]$).

        b_seg : int
            Right boundary of the target segment.  Must satisfy ``b_seg > a_seg``.
            The window width is ``b_seg - a_seg + 1``.
        **kwargs
            Forwarded to [`ModelBase`][lmlib.statespace.model.ModelBase].

        Examples
        --------
        Standard backward window of 501 samples aligned at the current sample::

            alssm = lm.AlssmPolyLegendre(poly_degree=3, a_seg=0, b_seg=500)
            seg   = lm.Segment(0, 500, lm.BW, g=100)

        Backward window shifted 200 samples into the past — same window size,
        same $h$, same $\kappa(W) \approx 2D+1$::

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
        r"""int : Polynomial degree $D$."""
        return self._poly_degree

    @poly_degree.setter
    def poly_degree(self, v: int):
        """Validate and set the polynomial degree (non-negative int)."""
        assert isinstance(v, int) and v >= 0, 'poly_degree must be a non-negative int'
        self._poly_degree = v

    @property
    def a_seg(self) -> int:
        """int : Left boundary of the target segment."""
        return self._a_seg

    @a_seg.setter
    def a_seg(self, v: int):
        """Set the left boundary of the Legendre window segment."""
        self._a_seg = int(v)

    @property
    def b_seg(self) -> int:
        """int : Right boundary of the target segment."""
        return self._b_seg

    @b_seg.setter
    def b_seg(self, v: int):
        """Set the right boundary of the Legendre window segment."""
        self._b_seg = int(v)

    @property
    def h(self) -> float:
        r"""float : Legendre step size $h = 2\,/\,(b\_seg - a\_seg)$."""
        return 2.0 / (self._b_seg - self._a_seg)

    # ------------------------------------------------------------------
    # ModelBase interface
    # ------------------------------------------------------------------

    def update(self):
        r"""
        Recompute $A$ and $C$ from ``poly_degree``, ``a_seg``, ``b_seg``.

        Step 1 — build shift matrix $A = L$ for step size $h = 2/(b-a)$.

        Step 2 — set $C = \phi(-1) = [1,\,-1,\,1,\,-1,\,\ldots]$.

        Step 3 — apply segment-relative shift (only when ``a_seg != 0``):

        $$
        C \;\leftarrow\; C\,A^{-a_{\rm seg}}
        $$

        For ``a_seg < 0`` (common backward window case) this is a *positive*
        power of $A$ — always stable.
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
        Compute the $N \times N$ upper-triangular Legendre shift matrix
        $L$ such that

        $$
        \phi(t + h) = \phi(t)\,L,
        \quad \phi(t) = [P_0(t),\, \ldots,\, P_{N-1}(t)].
        $$

        Each column $L_{:,n}$ is obtained by Taylor-expanding
        $P_n(t + h)$ around $t$:

        $$
        P_n(t + h)
        = \sum_{m=0}^{n} \frac{h^m}{m!} P_n^{(m)}(t)
        = \sum_{m=0}^{n} \frac{h^m}{m!}
          \sum_{j} \bigl[\operatorname{legder}^m(e_n)\bigr]_j P_j(t),
        $$

        where $e_n$ is the $n$-th standard basis vector in the
        Legendre coefficient representation and $\operatorname{legder}^m$
        denotes $m$ successive applications of
        [`legder`][numpy.polynomial.legendre.legder].

        The formula is exact (not a numerical fit) and costs $O(N^2)$
        operations.  $\kappa(L) \approx 1$ for all practical window
        sizes.

        Parameters
        ----------
        N : int
            Matrix dimension (= poly_degree + 1).
        h : float
            Step size in the scaled coordinate $t_{\rm sc}$.

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
    geometric (exponential) weight $\gamma^j$ on $j = 0, 1, 2, \ldots$

    Unlike [`AlssmPoly`][lmlib.statespace.model.AlssmPoly] (Pascal/monomial basis), which suffers from Gram
    matrix condition numbers of $\mathcal{O}(g^{2D})$, and
    [`AlssmPolyLegendre`][lmlib.statespace.model.AlssmPolyLegendre], which requires a *finite* window specification,
    [`AlssmPolyMeixner`][lmlib.statespace.model.AlssmPolyMeixner] is designed for **infinite or semi-infinite exponential
    windows** and keeps the composite Gram matrix near the theoretical minimum.

    **Mathematical background**

    The Meixner polynomials $M_n(j;\,1,\gamma)$ satisfy

    $$
    \sum_{j=0}^{\infty} \gamma^j\, M_m(j)\, M_n(j) = W_{n}\,\delta_{mn},
    \qquad W_n = g\!\left(\frac{g}{g-1}\right)^{\!n} = \frac{1}{(1-\gamma)\,\gamma^n},
    $$

    where $\gamma = (g-1)/g < 1$ is the exponential decay of the backward
    segment and $g > 1$ is its effective window length.  The norms
    $W_n$ are **exact and closed-form**, growing by the constant factor
    $g/(g-1)$ per degree. $\delta_{mn}$ is the kronecker delta.

    **Accepting a Segment**

    The recommended constructor is ``AlssmPolyMeixner(poly_degree, segment)`` which
    infers all parameters from the [`Segment`][lmlib.statespace.segment.Segment]:

    * ``segment.direction`` → selects $A$:

      * ``'bw'`` (backward): $A = A_{\rm bw} = I - \tfrac{1}{g-1}\operatorname{triu}(\mathbf{1},1)$
        — basis $C A_{\rm bw}^j x = M_n(j;\gamma)$ at lag $j \ge 0$.

      * ``'fw'`` (forward): $A = A_{\rm fw} = A_{\rm bw}^{-1}$
        — the forward filter's internal $A^{-1} = A_{\rm bw}$ step recovers
        the decaying Meixner basis at lags $j \le 0$.

    * ``segment.a`` (backward) or ``segment.b`` (forward) → shifts $C$ so
      that the Gram matrix remains $W_{\rm ss}$ (diagonal) even when the
      segment does not start at $j=0$:

      * Backward $[a, \infty)$:
        $C \leftarrow [1,\ldots,1]\,A_{\rm bw}^{-a}$
        (makes $C A^j x = M_n(j-a;\gamma)$ for $j \ge a$).

      * Forward $(-\infty, b]$:
        $C \leftarrow [1,\ldots,1]\,A_{\rm bw}^{-b}$
        (makes the relative basis start at the boundary $j=b$).

    The [`g`][lmlib.statespace.model.AlssmPolyMeixner.g] and [`direction`][lmlib.statespace.model.AlssmPolyMeixner.direction] attributes are always available.

    **Condition number**

    $\kappa(W) = (g/(g-1))^D$, independent of the segment shift.


    Notes
    -----
    The Meixner polynomials used here are the standard (monic normalisation)
    $M_n(x;\,1,\gamma) = {}_2F_1(-n,\,-x;\,1;\,1 - 1/\gamma)$.
    """

    steady_state_basis = 'meixner'

    def __init__(self, poly_degree: int, segment, **kwargs):
        r"""
        Parameters
        ----------
        poly_degree : int
            Polynomial degree $D \geq 0$.  The model order is $N = D+1$.
        segment : Segment
            Target segment.  Encodes direction, $g$, and boundary shift:

            * ``segment.g`` → window size $g$ (giving decay
              $\gamma = (g-1)/g$).
            * ``segment.direction`` → ``'bw'`` selects $A_{\rm bw}$;
              ``'fw'`` selects $A_{\rm fw} = A_{\rm bw}^{-1}$.
            * ``segment.a`` (BW, if finite) or ``segment.b`` (FW, if finite)
              → shifts $C$ to restore Gram matrix orthogonality when the
              segment does not start at the canonical origin.

            Meixner orthogonality requires a semi-infinite support, so the
            segment must be backward with ``b=+inf`` or forward with ``a=-inf``.
        **kwargs
            Forwarded to [`ModelBase`][lmlib.statespace.model.ModelBase] (e.g. ``label``).
        """

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
        r"""int : Polynomial degree $D$."""
        return self._poly_degree

    @poly_degree.setter
    def poly_degree(self, v: int):
        """Validate and set the polynomial degree (non-negative int)."""
        assert isinstance(v, int) and v >= 0, 'poly_degree must be a non-negative int'
        self._poly_degree = v

    @property
    def g(self) -> float:
        r"""float : Effective window size $g$ (read-only; from the segment)."""
        return self._g

    @property
    def direction(self) -> str:
        r"""str : Segment direction ``'bw'`` (backward) or ``'fw'`` (forward)."""
        return self._direction

    @property
    def shift(self) -> int:
        r"""int : Boundary shift applied to $C$; 0 for the canonical origin."""
        return self._shift

    @property
    def gamma(self) -> float:
        r"""float : Exponential decay $\gamma = (g-1)/g$."""
        return (self._g - 1.0) / self._g

    # ------------------------------------------------------------------
    # ModelBase interface
    # ------------------------------------------------------------------

    def update(self):
        r"""
        Recompute $A$ and $C$ from all stored parameters.

        **Transition matrix** $A$:

        * direction ``'bw'``:
          $A = A_{\rm bw} = I_N - \tfrac{1}{g-1}\,\operatorname{triu}(\mathbf{1}_N, 1)$
        * direction ``'fw'``:
          $A = A_{\rm fw} = A_{\rm bw}^{-1}$

        **Output vector** $C$ (before shift):
        $C_{\rm base} = [1,\ldots,1]$.

        **Shift** (when ``shift != 0``):
        $C \leftarrow C_{\rm base}\,A_{\rm bw}^{-\text{shift}}$.

        This ensures that for a backward segment $[a, \infty)$ with
        ``shift = a``, the output at absolute lag $j$ is
        $M_n(j - a;\,\gamma)$ — orthogonal under the shifted window weight.
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

    Generates a sinusoidal sequence with angular frequency $\omega$ (radians
    per sample) and per-sample amplitude decay factor $\rho$:

    $$
    \begin{aligned}
    A =
    \begin{bmatrix}
        \rho \cos\omega & -\rho \sin\omega \\
        \rho \sin\omega &  \rho \cos\omega
    \end{bmatrix}
    \end{aligned}
    $$

    The state vector holds $[a\cos, a\sin]$ components at the current sample.
    With the default output $C = [1, 0]$, the output is the cosine component.

    For more details see [\[Wildhaber2019\]](../../bibliography.md#wildhaber2019) [PDF](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=48)

    Parameters
    ----------
    omega : float
        Normalised discrete-time angular frequency $\omega = 2\pi f / f_s$
        (radians per sample), where $f$ is the signal frequency and
        $f_s$ is the sampling frequency.
    rho : float, optional
        Per-sample amplitude decay factor. ``rho = 1.0`` gives an undamped
        oscillation. Default: 1.0.
    C : array_like of shape ([Q,] N), optional
        Output matrix. If not given, defaults to ``[1, 0]``, selecting the cosine
        component. With ``force_MC=True`` the shape is broadcast to ``(1, N)``.
        Default: None.
    **kwargs
        Forwarded to [`ModelBase`][lmlib.statespace.model.ModelBase].

    Notes
    -----
    `N` : ALSSM system order, corresponding to the number of state variables <br>
    `Q` : output order / number of signal channels <br>

    To convert a signal period in samples to a normalised angular frequency, use
    [`k_period_to_omega`][lmlib.utils.generator.k_period_to_omega]:

    ```python
    import lmlib as lm
    from lmlib.utils.generator import k_period_to_omega

    k_period = 20
    alssm = lm.AlssmSin(k_period_to_omega(k_period), rho=0.9)
    ```

    Example
    --------
    ```python
    >>> alssm = lm.AlssmSin(omega=0.1, rho=0.9)
    >>> print(alssm)
    AlssmSin(A=[[ 0.89550375 -0.08985007],[ 0.08985007  0.89550375]], C=[1 0], label=n/a)
    ```
    """

    def __init__(self, omega:float, rho:float=1.0, C=None, **kwargs):
        super().__init__(C_init=C, **kwargs)
        self.omega = omega
        self.rho = rho
        self.update()
        self.set_state_var_label('cos', (0,))
        self.set_state_var_label('sin', (1,))

    def update(self):
        r"""Recompute the 2×2 rotation/decay matrix A and default C from $\omega$ and $\rho$."""
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
        r"""float : Normalised angular frequency $\omega = 2\pi f / f_s$ (radians per sample)."""
        return self._omega

    @omega.setter
    def omega(self, omega: float):
        assert np.isscalar(omega), 'Frequency factor omega is a scalar'
        self._omega = float(omega)

    @property
    def rho(self) -> float:
        r"""float : Decay factor $\rho$"""
        return self._rho

    @rho.setter
    def rho(self, rho: float):
        assert np.isscalar(rho), 'Decay factor rho is not a scalar'
        self._rho = float(rho)


class AlssmExp(ModelBase):
    r"""
    ALSSM with a discrete-time exponential output sequence.

    Generates a scalar exponential sequence $\gamma^j$ with per-sample
    factor $\gamma$:

    $$
    A = [\gamma], \qquad s_j(x) = \gamma^j \cdot x
    $$

    where $x$ is the scalar initial amplitude. Values of $|\gamma| < 1$
    give a decaying sequence and $|\gamma| > 1$ a growing sequence.

    For more details see [\[Wildhaber2019\]](../../bibliography.md#wildhaber2019) [PDF](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=48)


    Parameters
    ----------
    gamma : float
        Per-sample exponential factor $\gamma$.
    C : array_like of shape ([Q,] N), optional
        Output matrix. If not given, defaults to ``[1.]``. Default: None.
    **kwargs
        Forwarded to [`ModelBase`][lmlib.statespace.model.ModelBase].

    Notes
    -----
    `N` : ALSSM system order, corresponding to the number of state variables <br>
    `Q` : output order / number of signal channels <br>

    Example
    --------
    ```python
    >>> import lmlib as lm
    >>> alssm = lm.AlssmExp(gamma=0.8)
    >>> print(alssm)
    AlssmExp(A=[[0.8]], C=[1.], label=n/a)
    ```
    """

    def __init__(self, gamma:float, C=None, **kwargs):
        super().__init__(C_init=C, **kwargs)
        self.gamma = gamma
        self.update()

    def update(self):
        """Recompute the scalar A matrix and default C from ``gamma``."""
        if self.C_init is None:
            self.C = np.ones((1,))
        else:
            self.C = self.C_init
        self.A = np.array([[self.gamma]])
        self._init_state_var_labels()
        self._broadcast_C_to_multichannel()

    @property
    def gamma(self) -> float:
        r"""float : Per-sample exponential decay factor $\gamma$."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float):
        r"""Validate and set the decay factor $\gamma$."""
        assert np.isscalar(gamma), 'Decay factor gamma is not scalar'
        self._gamma = gamma


class AlssmStacked(ModelBase):
    r"""
    Creates a joined ALSSM generating a stacked output signal of multiple ALSSMs.

    For $M$ ALSSMs as in [`Alssm`][lmlib.statespace.model.Alssm],
    we get the stacked model's output  $\tilde{s}_k(\tilde{x}) = \tilde{y}_k$

    $$
    \begin{aligned}
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
    \end{aligned}
    $$

    where $y_m[k]$ denotes the output of the m-th joined ALSSM.
    $y_m[k]$  is either a scalar (for single-channel ALSSMs) or a column vector (for multi-channel ALSSMs).
    Accordingly, the initial state vector is

    $$
    \begin{aligned}
    \tilde{x} =
    \begin{bmatrix}
        x_1 \\
        \vdots \\
        x_M \\
    \end{bmatrix}
    \end{aligned}
    $$

    where $x_m[k]$ is the state vector of the m-th joined ALSSM.

    Therefore, the internal model matrices of the joined model are set to the block diagonals

    $$
    \tilde{A} =
    \left[\begin{array}{c|c|c}
        A_1 & 0 & 0 \\ \hline
        0 & \ddots & 0 \\ \hline
        0 & 0 & A_{M}
    \end{array}\right]
    $$

    $$
    \tilde{C} =
    \left[\begin{array}{c|c|c}
        \lambda_1 C_1 & 0 & 0 \\ \hline
        0 & \ddots & 0 \\ \hline
        0 & 0 & \lambda_M C_{M}
    \end{array}\right]
    $$

    where $A_m$ and $C_m$ are the transition matrices and the output vectors of the joined models, respectively, and
    $\lambda_1 ... \lambda_M  \in \mathcal{R}$ are additional factors to weight each output individually.

    For more details see [\[Wildhaber2019\]](../../bibliography.md#wildhaber2019) [PDF Sec: Linear Combination of M Systems](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=48)

    Parameters
    ----------
    alssms : iterable of ModelBase, length M
        Set of M autonomous linear state space models whose outputs are stacked.
    lambdas : list of float or None, optional
        List of M scalar factors $\\lambda_m$ weighting each ALSSM's output
        matrix $C_m$. Default: None (all factors set to 1).
    **kwargs
        Forwarded to [`ModelBase`][lmlib.statespace.model.ModelBase].

    Notes
    -----
    `M` : number of ALSSMs <br>

    Example
    --------
    ```python
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
    ```
    """

    def __init__(self, alssms, lambdas=None, **kwargs):
        super().__init__(**kwargs)
        self.alssms = alssms
        r"""list of ModelBase : The $M$ sub-ALSSMs whose outputs are stacked."""
        self.lambdas = lambdas
        r"""ndarray : Per-ALSSM output scaling factors $\lambda_m$ (one per sub-ALSSM)."""
        self.update()

    def update(self):
        """Recompute the block-diagonal A and C from the sub-ALSSMs and their lambda scaling factors."""
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
    Joins multiple ALSSMs generating the output sum.

    Generates the summed output signal of $M$ ALSSMs:

    $$
    \tilde{s}_k(\tilde{x}) = s_k^{(1)}(x_1) + \cdots + s_k^{(M)}(x_M)
    = \tilde{C}\,\tilde{A}^k\,\tilde{x}
    $$

    with block-diagonal transition matrix $\tilde{A}$ and horizontally
    stacked output matrix $\tilde{C} = [\lambda_1 C_1 \;\cdots\; \lambda_M C_M]$.

    Parameters
    ----------
    alssms : iterable of ModelBase, length M
        Set of M autonomous linear state space models. All ALSSMs must share the
        same output dimension; see [`get_alssm_output_dimension`][lmlib.statespace.model.ModelBase.get_alssm_output_dimension].
    lambdas : list of float or None, optional
        List of M scalar factors $\lambda_m$ applied to each output matrix.
        Default: None (all factors set to 1).
    **kwargs
        Forwarded to [`ModelBase`][lmlib.statespace.model.ModelBase].

    `M` : number of ALSSMs <br>

    Example
    --------
    ```python
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
    ```
    """

    def __init__(self, alssms, lambdas=None, **kwargs):
        super().__init__(**kwargs)
        self.alssms = alssms
        r"""list of ModelBase : The $M$ sub-ALSSMs whose outputs are summed."""
        self.lambdas = lambdas if lambdas is not None else [1.0] * len(self.alssms)
        r"""ndarray : Per-ALSSM output scaling factors $\lambda_m$ (one per sub-ALSSM)."""
        self.update()

    def update(self):
        """Recompute block-diagonal A and horizontally stacked C from sub-ALSSMs."""
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

    Creating a joined ALSSM generating the product of all output signals of $M$ ALSSMs, e.g., on the example of $M=2$,

    $$
    \begin{aligned}
    \tilde{s}_k (\tilde{x}) = s_k^{(1)}(x_1) \cdot s_k^{(2)}(x_2) &= (C_1 A_1^k x_1)(C_2 A_2^k x_2)\\
    &= (C_1 A_1^k x_1) \otimes (C_2 A_2^k x_2)\\
    &= (C_1 \otimes C_2) (A_1^k \otimes A_2^k) (x_1 \otimes  x_2) \ ,
    \end{aligned}
    $$

    where $s_k^{(1)}(x_1) = C_1 A_1^k x_1$ is the first and $s_k^{(2)}(x_2) = C_2 A_2^k x_2$ the second ALSSM.

    For more details, see also [\[Zalmai2017\]](../../bibliography.md#zalmai2017) [Proposition 2],  [\[Wildhaber2019\]](../../bibliography.md#wildhaber2019) [Eq. 4.21].

    Parameters
    ----------
    alssms : iterable of ModelBase, length M
        Set of M autonomous linear state space models. For the Kronecker product
        interpretation to give a scalar output, all sub-ALSSMs should produce
        scalar outputs (i.e. 1-D ``C``).
    lambdas : list of float or None, optional
        List of M scalar factors applied to each output matrix.
        Default: None (all factors set to 1).
    **kwargs
        Forwarded to [`ModelBase`][lmlib.statespace.model.ModelBase].


    `M` : number of ALSSMs <br>

    Example
    --------
    ```python
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
    ```
    """

    def __init__(self, alssms, lambdas=None, **kwargs):
        super().__init__(**kwargs)
        self.alssms = alssms
        r"""list of ModelBase : The $M$ sub-ALSSMs whose outputs are multiplied."""
        self.lambdas = lambdas
        r"""ndarray : Per-ALSSM output scaling factors $\lambda_m$ (one per sub-ALSSM)."""
        self.update()

    def update(self):
        r"""Recompute the Kronecker-product A and C from the sub-ALSSMs and their lambda scaling factors."""
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
