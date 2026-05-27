"""
Definition of recursively computed squared error cost functions (such as *Cost Segments* and *Composite Costs*),
all based on ALSSMs

.. currentmodule:: lmlib.statespace.cost

.. inheritance-diagram:: lmlib.statespace.cost
   :top-classes: lmlib.statespace.cost.Segment, lmlib.statespace.cost.CompositeCost, lmlib.statespace.cost.NDCompositeCost, lmlib.statespace.cost.ConstrainMatrix
   :parts: 1

"""


from abc import ABC, abstractmethod
import numpy as np
from typing import Iterable, Union, List
from lmlib.utils.check import *

from lmlib.statespace.model import ModelBase, AlssmSum, AlssmProd, AlssmSin
from lmlib.statespace.segment import Segment
from lmlib.statespace.backends.steady_state import *

__all__ = ['CostSegment', 'CompositeCost', 'NDCompositeCost', 'ConstrainMatrix']


class BaseCost(ABC):
    """Base interface for all cost-like objects."""

    def __iter__(self):
        """Default iterator: yield self (leaf behavior)."""
        yield self

    def _get_sub_cost_term(self, dim=None, seg=None):
        return self

    @property
    def label(self):
        """str : Label of the Cost Model"""
        return self._label

    @label.setter
    def label(self, label):
        assert isinstance(label, str)
        self._label = label

    @abstractmethod
    def get_alssm_order(self):
        """int : Order of the (stacked) Alssm Model"""
        pass

    @abstractmethod
    def get_alssm_output_dimension(self):
        """int : Output dimension of the Alssm"""
        pass

    @abstractmethod
    def get_steady_state_W(self, dim_order=None, method='closed_form'):
        """
        Returns Steady State Matrix W

        Parameters
        ----------
        dim_order : list of int, None
            Specifies the dimensional order in a list of integers.
            If None, dim_order is equal to to range(self.L)
        method : str, optional
            If 'closed_form' is used, the steady state matrix will be calculated in a close form.
            This method can be critical, as it can produce badly conditioned matrices internally.
            If 'limited_sum' is used, the steady state matrix will be calculated brute force, with a stop condition
            on a minimum change.

        Returns
        -------
        Wss = `class:numpy.ndarray`
            Steady State Matrix W
        """
        pass

    @abstractmethod
    def get_number_of_dimensions(self):
        """int : Number of dimensions of the cost function"""
        pass


class BaseCost1d(ABC):

    @abstractmethod
    def eval_alssm_output(self, xs, alssm_weights=None):
        """
        Evaluation of the ALSSM for multiple state vectors `xs`.

        **See:** :meth:`~lmlib.statespace.models.Alssm.eval`

        Parameters
        ----------
        xs : array_like of shape=(XS, N [,S]) of floats
            List of state vectors :math:`x`
        alssm_weights : None, scalar, array_like
            CostSegments contains a single Alssm, where CompositeCost and NDCompositeCost
            may have multiple Alssms.
            - If is None no Alssm output weights are set (all equal 1).
            - If is scalar, each Alssm output has the same weight as `alssm_weights`.
            - If is array_like, each Alssm output has the weight respectively to the alssms_weights.
            Each element sets the weight of the output of the m-th ALSSM in :attr:`CompositeCost.alssms`.
        """
        pass

    @abstractmethod
    def get_state_var_indices(self, label):
        """
        Returns the state indices for a specified label of the stacked internal ALSSM

        Parameters
        ----------
        label : str
            state label

        Returns
        -------
        out : list of int
            state indices of the label
        """
        pass

    @abstractmethod
    def get_alssms(self):
        """Returns the list of internal ALSSMs."""
        pass


class CostSegment(BaseCost, BaseCost1d):
    r"""
    Quadratic cost function defined by an ALSSM and a Segment.

    A CostSegment is an implementation of [Wildhaber2019]_ :download:`PDF (Cost Segment) <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=117>`

    .. code-block:: text

        ==================
        Class: CostSegment (= 1 ALSSM + 1 Segment)
        ==================

          window weight
              ^
            2_|                   ___     exponentially rising window of a segment
            1_|              ____/     <- normalized to 1 at index delta; to
            0_|    _________/             weight the cost function

                       segment
                   +----------------+
          alssm    |                |
                   +----------------+

                   |--|-------|-----|----> k (time, relativ to the segment reference index 0)
                   a  0     delta   b

        a,b : segment interval borders a and b in N, a < b


    A cost segment is the quadratic cost function

    .. math::
        J_a^b(k,x,\theta) = \sum_{i=k+a}^{k+b} \alpha_{k+\delta}(i)v_i(y_i - cA^{i-k}x)^2

    over a fixed interval :math:`\{a, \dots, b\}` with :math:`a \in \mathbb{Z} \cup \{ - \infty \}`,
    :math:`b \in \mathbb{Z} \cup \{ + \infty\}`, and :math:`a \le b`,
    and with initial state vector :math:`x \in \mathbb{R}^{N \times 1}`.
    For more details, seen in Section 4.2.6  and Chapter 9 in [Wildhaber2019]_ .

    Parameters
    ----------
    alssm : ModelBase
        ALSSM, defining the signal model
    segment : :class:`.Segment`
        Segment, defining the window
    beta : float, optional
        Scaling factor on each cost term. beta >= 0.0, default: 1.0
    label : str, optional
        Label of CostSegment, default: 'n/a'

    Examples
    --------
    Set up a cost segment with finite boundaries and a line ALSSM.

    >>> import lmlib as lm
    >>> alssm_line = lm.AlssmPoly(poly_degree=1, label="slope with offset")
    >>> segment_left = lm.Segment(a=-30, b=0, direction=lm.FORWARD, g=20, label="finite left")
    >>> cost = lm.CostSegment(alssm_line, segment_left, label="left line")
    >>> print(cost)
    CostSegment : label: left line
      └- Alssm : polynomial, A: (2, 2), C: (1, 2), label: slope with offset,
      └- Segment : a:-30, b:0, fw, g:20, delta:0, label: finite left

    """

    def __init__(self, alssm, segment, beta=1.0, label='n/a'):
        self.alssm = alssm
        self.segment = segment
        self.beta = beta
        self.label = label

    def __str__(self):
        return f'{type(self).__name__}(label: {self.label}) \n  └- {self.alssm}, \n  └- {self.segment}'

    @property
    def alssm(self):
        """ModelBase : Alssm"""
        return self._alssm

    @alssm.setter
    def alssm(self, alssm):
        assert isinstance(alssm, ModelBase), 'alssm is not of instance ModelBase'
        self._alssm = alssm

    @property
    def segment(self):
        """Segment : Segment"""
        return self._segment

    @segment.setter
    def segment(self, segment):
        assert isinstance(segment, Segment), 'element in segments is not instance of Segment'
        self._segment = segment

    @property
    def beta(self):
        """float : Scaling factor on CostSegment"""
        return self._beta

    @beta.setter
    def beta(self, beta):
        assert np.isscalar(beta), 'beta is not scalar'
        assert beta >= 0.0, 'beta is negative'
        self._beta = float(beta)

    def _get_cost_segments(self, F=None, force_MC=False):
        """Returns a list of the updated CostSegments (modified by F or force_MC if provided)."""
        alssm = AlssmSum([self.alssm], [1], force_MC=force_MC)
        return [CostSegment(alssm, self.segment, self.beta, self.label)]

    def get_alssm_order(self) -> int:
        return self.alssm.N

    def get_number_of_dimensions(self):
        return 1

    def get_alssm_output_dimension(self) -> int:
        return self.alssm.get_alssm_output_dimension()

    def get_steady_state_W(self, dim_order=None, method='schur'):
        A, C = self.alssm.A, self.alssm.C
        gamma = self.segment.gamma
        a, b, delta = self.segment.a, self.segment.b, self.segment.delta

        if method == 'schur':
            return covariance_matrix_schur(A, C, gamma, a, b, delta)
        elif method == 'closed_form':
            return covariance_matrix_closed_form(A, C, gamma, a, b, delta)
        elif method == 'limited_sum':
            return covariance_matrix_limited_sum(A, C, gamma, a, b, delta)
        else:
            raise NotImplementedError(f'unknown method {method}')

    def get_state_var_indices(self, label):
        return self.alssm.get_state_var_indices(label)

    def eval_alssm_output(self, xs, alssm_weights=None):
        return AlssmSum([self.alssm], alssm_weights).eval_output(xs)

    def get_alssms(self):
        return [self.alssm]

class CompositeCost(BaseCost, BaseCost1d):
    r"""
    Quadratic cost function defined by a conjunction one or multiple  of :class:`~lmlib.statespace.model.Alssm` and :class:`.Segment`

    A composite costs combines multiple ALSSM models and multiple Segments in the form of a grid,
    where each row is a model and each column a Segment.
    The segments define the temporal relations of the models.
    Mapping matrix `F` enables or disables each ALSSM/Segment pair in each grid node;
    multiple active ALSSMs in one column are superimposed.

    A CompositeCost is an implementation of [Wildhaber2019]_ :download:`PDF (Composite Cost) <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=118>`


    .. code-block:: text

        ====================
        Class: CompositeCost (= M ALSSM's + P Segment's)
        ====================


        M : number of ALSSM models
        ^
        |                Segment[0]   Segment[1]    ...   Segment[P-1]
        |              +------------+------------+--//--+--------------+
        |  alssm[0]    |  F[0,0]    |            |      | F[0,P-1]     |
        |              +------------+------------+--//--+--------------+
        |    .         |     .      |     ...    |      |    .         |
        |    .         |     .      |     ...    |      |    .         |
        |              +------------+------------+--//--+--------------+
        |  alssm[M-1]  |  F[M-1,0]  |            |      | F[M-1, P-1]  |
        |              +------------+------------+--//--+--------------+``

                       |------------|-------|----|------|--------------|--> k (time)
                                            0
                                (0: common relative reference
                                 index for all segments)

        P : number of segments

        F[m,p] in R+, scalar on alssm's output. (i.e., 0 for inactive grid node,
        1 or any other scalar factor for active grid node).


    This figure shows a graphical representation of a composite cost, depicting the internal relationships between
    Segments, ALSSMs, and the mapping matrix F. F[m,p] is implemented as a scalar factor multiplied on the alssm's
    output signal.


    For more details, see Chapter 9 in [Wildhaber2019]_.
    The cost function of a composite cost is defined as

    .. math::
       J(k, x, \Theta) = \sum_{p = 1}^{P}\beta_p J_{a_p}^{b_p}(k, x, \theta_p) \ ,

    where, :math:`\Theta = (\theta_1, \theta_2,\dots, \theta_P)` and  the *segment scalars*
    :math:`\beta_p \in \mathbb{R}_+`.


    Parameters
    ----------
    alssms : tuple, shape=(M,)
        Set of ALSSMs
    segments : tuple, shape=(P,)
        Set of Segments
    F : array_like of shape=(M, P)
        Mapping matrix :math:`F`, maps models to segment
    betas : array_like of shape=(P,), optional
        Segment scalars on cost terms, default: all ones
    label : str, optional
        Label of ALSSM, default: 'n/a'


    |def_M|
    |def_P|


    Examples
    --------
    >>> import lmlib as lm
    >>>
    >>> alssm_spike = lm.AlssmPoly(poly_degree=3, label='spike')
    >>> alssm_baseline = lm.AlssmPoly(poly_degree=2, label='baseline')
    >>>
    >>> segment_left = lm.Segment(a=-50, b=-1, direction=lm.FORWARD, g=20, label="finite left")
    >>> segment_middle = lm.Segment(a=0, b=10, direction=lm.FORWARD, g=100, label="finite middle")
    >>> segment_right = lm.Segment(a=10, b=50, direction=lm.FORWARD, g=20, delta=10, label="finite right")
    >>>
    >>> F = [[0, 1, 0], [1, 1, 1]]
    >>> cost = lm.CompositeCost((alssm_spike, alssm_baseline), (segment_left, segment_middle, segment_right), F, label='spike_baseline')
    >>> print(cost)
    CostSegment : label: spike_baseline
      └- ['Alssm : polynomial, A: (4, 4), C: (1, 4), label: spike', 'Alssm : polynomial, A: (3, 3), C: (1, 3), label: baseline'],
      └- [Segment : a: -50, b: -1, direction: fw, g: 20, delta: 0, label: finite left , Segment : a: 0, b: 10, direction: fw, g: 100, delta: 0, label: finite middle , Segment : a: 10, b: 50, direction: fw, g: 20, delta: 10, label: finite right ]

    """

    def __init__(self, alssms, segments, F, betas=None, label='n/a'):
        # set alssms
        assert isinstance(alssms, Iterable), 'alssms is not iterable'
        for alssm in alssms:
            assert isinstance(alssm, ModelBase), 'element in alssms is not of instance ModelBase'
        self._alssms = list(alssms)

        # set segments
        assert isinstance(segments, Iterable), 'segments is not iterable'
        for segment in segments:
            assert isinstance(segment, Segment), 'element in segments is not instance of Segment'
        self._segments = list(segments)

        # set F
        self.F = F

        # set betas
        if betas is not None:
            betas=np.array(betas)
            assert is_array_like(betas), 'betas is not array_like'
            assert betas.shape == (self.P,), f'betas has wrong shape, {info_str_found_shape(betas)}'
            for beta in betas:
                assert np.isscalar(beta), 'beta is not scalar'
                assert beta >= 0.0, 'beta is negative'
            self._betas = betas
        else:
            self._betas = np.ones(self.P)
        self.label = label

    def __str__(self):
        return f'CompositeCost(label={self.label}) \n' \
               f'  └- {[alssm.__str__() for alssm in self.alssms]}, \n' \
               f'  └- {[segment.__str__() for segment in self.segments]} '

    def __iter__(self):
        """Iterate over cost segments."""
        return iter(self._get_cost_segments())

    def _get_cost_segments(self, F=None, force_MC=False):
        """Returns list of the updated CostSegments (modified by F if provided)."""
        F_ = self.F if F is None else np.asarray(F)
        cost_segments = []
        for p, segment in enumerate(self.segments):
            alssm = AlssmSum(self.alssms, F_[:, p], force_MC=force_MC)
            cost_segment = CostSegment(alssm, segment, self.betas[p], self.label + '-' + str(p))
            cost_segments.append(cost_segment)
        return cost_segments
    
    def _get_sub_cost_term(self, dim=None, seg=None):
        """Returns a specific CostSegment by segment index."""
        if seg is None:
            # if no segment specified, return all segments as list
            return self
        return self._get_cost_segments()[seg]

    @property
    def alssms(self):
        """Iterable : List of ALSSM"""
        return self._alssms

    @property
    def segments(self):
        """Iterable : List of Segment"""
        return self._segments

    @property
    def M(self):
        """int : Number of Alssms"""
        return len(self.alssms)

    @property
    def P(self):
        """int : Number of Segments"""
        return len(self.segments)

    @property
    def F(self):
        """:class:`~numpy.ndarray` : Mapping matrix :math:`F`, maps models to segments"""
        return self._F

    @F.setter
    def F(self, F):
        assert is_array_like(F), 'F is not array_like'
        F = np.atleast_2d(F)
        assert np.shape(F) == (self.M, self.P), f'F has wrong shape, {info_str_found_shape(F)}'
        self._F = F

    @property
    def betas(self):
        """array_like of shape=(P,) : Segment scalars on cost terms"""
        return self._betas

    def get_alssm_order(self):
        return self._get_sub_cost_term(seg=0).get_alssm_order()

    def get_alssm_output_dimension(self):
        return self._get_sub_cost_term(seg=0).get_alssm_output_dimension()

    def get_number_of_dimensions(self):
        return 1

    def get_steady_state_W(self, dim_order=None, method='schur'):
        N = self.get_alssm_order()
        W = np.zeros((N, N))

        for cost_segment in self:
            W += cost_segment.get_steady_state_W(method=method) * cost_segment.beta
        return W

    def eval_alssm_output(self, xs, alssm_weights=None):
        return AlssmSum(self.alssms, alssm_weights).eval_output(xs)

    def get_state_var_indices(self, label):
        return AlssmSum(self.alssms, label='cost').get_state_var_indices('cost.' + label)

    def get_alssms(self):
        return self.alssms

    def spline_H(self, max_continuity: int, alssm_index: int = 0) -> np.ndarray:
        r"""Build the spline continuity H-matrix for a two-ALSSM cost.

        For a :class:`CompositeCost` with **exactly two ALSSMs** and mixing matrix
        ``F = [[1,0],[0,1]]`` (one ALSSM per segment, independent states), the full
        state vector is :math:`x = [x_L\;(N),\; x_R\;(N)]`.  The H-matrix returned
        here reduces this to a lower-dimensional parameter vector :math:`v` by
        imposing continuity constraints at the knot:

        .. math::

            x = H\,v, \qquad H \in \mathbb{R}^{2N \times (2N - m_c - 1)},

        where :math:`m_c =` ``max_continuity`` and :math:`N` is the model order.

        **Continuity constraints** are derived from the *k*-th forward difference
        operator evaluated at the knot lag :math:`j = 0`:

        .. math::

            e_k = C\,(A - I)^k, \qquad k = 0, 1, \ldots

        Constraint of order *k*:

        .. math::

            e_k\,(x_R - (-1)^k\,x_L) = 0

        which encodes:

        * :math:`k=0` — value continuity: :math:`C x_R = C x_L`.
        * :math:`k=1` — first-derivative continuity: :math:`e_1(x_R + x_L) = 0`.
        * etc.

        The result is the unique minimum-rank H satisfying all constraints
        :math:`k = 0, \ldots, m_c`, computed via a stable linear solve.

        **Compatibility of the two ALSSMs**

        For constraints of order :math:`k \geq 1`, both ALSSMs must share the same
        :math:`A` and :math:`C` matrices (same basis functions).  If they differ and
        ``max_continuity >= 1``, a :class:`UserWarning` is emitted because the
        H-matrix will use the basis of ``alssm_index`` and impose it on both sides,
        which is physically wrong for the other side.

        For :class:`~lmlib.statespace.model.AlssmSin` pairs, in particular:

        * ``max_continuity = 0`` (value match only) always works.
        * ``max_continuity >= 1`` requires the same ``omega`` **and** ``rho = 1``
          (undamped oscillation) on both sides; otherwise a warning is issued.

        Parameters
        ----------
        max_continuity : int
            Highest continuity order to impose.  ``-1`` returns the identity
            (free, no constraint); ``0`` imposes :math:`C^0` (value match);
            ``D`` (= ``poly_degree``) is the maximum possible (fully smooth).
        alssm_index : int, optional
            Index (0-based) into ``self.alssms`` of the reference ALSSM whose
            :math:`A` and :math:`C` are used to build :math:`e_k`.  Relevant
            only when the two ALSSMs differ.  Default: ``0``.

        Returns
        -------
        H : ndarray of shape (2N, 2N - max_continuity - 1)
            Linear constraint matrix.  Pass to
            :meth:`~lmlib.statespace.rls.RLSAlssm.minimize_x` as ``H=``.

        Raises
        ------
        ValueError
            If the cost does not have exactly two ALSSMs.

        Warns
        -----
        UserWarning
            When ``max_continuity >= 1`` and the two ALSSMs have different
            :math:`A` or :math:`C` matrices.

        Examples
        --------
        Edge detection: test C^0 (offset-only) vs C^0+C^1 (slope join)::

            alssm_L = lm.AlssmPolyMeixner(D, segment=segL)
            alssm_R = lm.AlssmPolyMeixner(D, segment=segR)
            cost    = lm.CompositeCost((alssm_L, alssm_R), (segL, segR), [[1,0],[0,1]])

            rls.filter(y)
            xs_free  = rls.minimize_x()                       # unconstrained
            xs_cont  = rls.minimize_x(H=cost.spline_H(0))    # C^0 join
            xs_peak  = rls.minimize_x(H=cost.spline_H(1))    # C^0+C^1 join
            xs_full  = rls.minimize_x(H=cost.spline_H(D))    # fully smooth

        Works equally for AlssmPoly, AlssmPolyJordan, AlssmPolyLegendre, AlssmSin.
        For AlssmPoly(1):
        ``spline_H(0)`` reproduces ``H_Continuous`` and
        ``spline_H(1)`` reproduces ``H_Peak`` from the literature.
        """
        import warnings
        from numpy.linalg import matrix_power, solve

        if self.M != 2:
            raise ValueError(
                f"spline_H requires exactly 2 ALSSMs in the CompositeCost "
                f"(got {self.M}).  Use alssm_index to select the reference ALSSM.")

        # ── free case ────────────────────────────────────────────────────────────
        if max_continuity < 0:
            alssm_ref = self.alssms[alssm_index]
            N = alssm_ref.N
            return np.eye(2 * N)

        # ── select reference ALSSM ───────────────────────────────────────────────
        # For AlssmPolyMeixner TSLM the convention is (alssm_L[FW], alssm_R[BW]).
        # The spline e_k vectors must use A_bw (the backward matrix), so if
        # alssm_index points to a FW Meixner alssm, auto-select the BW one instead.
        _ref_idx = alssm_index
        from lmlib.statespace.model import AlssmPolyMeixner
        if (isinstance(self.alssms[_ref_idx], AlssmPolyMeixner)
                and getattr(self.alssms[_ref_idx], 'direction', 'bw') == 'fw'):
            other_idx = 1 - _ref_idx
            if (isinstance(self.alssms[other_idx], AlssmPolyMeixner)
                    and getattr(self.alssms[other_idx], 'direction', 'bw') == 'bw'):
                _ref_idx = other_idx

        alssm_ref   = self.alssms[_ref_idx]
        alssm_other = self.alssms[1 - _ref_idx]
        A = np.asarray(alssm_ref.A)
        C = np.asarray(alssm_ref.C).ravel()
        N = A.shape[0]

        # ── compatibility check ──────────────────────────────────────────────────
        if max_continuity >= 1 and isinstance(alssm_ref, AlssmSin) and isinstance(alssm_other, AlssmSin):
            same_A = np.allclose(alssm_ref.A, alssm_other.A, rtol=1e-6, atol=1e-10)
            same_C = np.allclose(
                np.asarray(alssm_ref.C).ravel(),
                np.asarray(alssm_other.C).ravel(),
                rtol=1e-6, atol=1e-10)
            if not (same_A and same_C):
                warnings.warn(
                    f"spline_H(max_continuity={max_continuity}): the two ALSSMs have "
                    f"different A or C matrices.  Constraints of order >= 1 use the "
                    f"basis of alssm[{alssm_index}] and will be physically wrong for "
                    f"alssm[{1-alssm_index}].  For C^0 (value match only), set "
                    f"max_continuity=0.",
                    UserWarning, stacklevel=2)

        # ── build constraint vectors e_k = C (A-I)^k ────────────────────────────
        nc = min(max_continuity + 1, N)
        e = [C @ matrix_power(A - np.eye(N), k) for k in range(nc)]

        # ── assemble and solve for constrained block AR_top (nc x N) ────────────
        #
        # Full state: x_full = [x_L (N), x_R (N)], free params v = [v_L (N), v_free (N-nc)]
        # x_L = v_L
        # x_R[0:nc]  = AR_top @ v_L + BR_top @ v_free   (constrained)
        # x_R[nc:]   = v_free                             (free)
        #
        # Constraint k:  e_k @ x_R = (-1)^k * e_k @ x_L  for all (v_L, v_free)
        # => (1) E_top @ AR_top = RHS          (coefficient of v_L)
        # => (2) E_top @ BR_top + E_bot = 0    (coefficient of v_free)
        #
        # where E_top = E[:, :nc], E_bot = E[:, nc:], RHS[k,:] = (-1)^k * e_k.

        E     = np.array([e[k]       for k in range(nc)])   # nc x N
        RHS   = np.array([(-1)**k * e[k] for k in range(nc)])  # nc x N
        E_top = E[:, :nc]   # nc x nc
        E_bot = E[:, nc:]   # nc x (N-nc)

        AR_top = solve(E_top, RHS)           # nc x N
        BR_top = solve(E_top, -E_bot)        # nc x (N-nc)

        # ── assemble H ───────────────────────────────────────────────────────────
        nv = N + (N - nc)
        H  = np.zeros((2 * N, nv))
        H[:N, :N]         = np.eye(N)   # x_L = v_L
        H[N:N+nc, :N]     = AR_top      # x_R[0:nc] from v_L
        H[N:N+nc, N:]     = BR_top      # x_R[0:nc] from v_free
        for j in range(N - nc):         # x_R[nc:] = v_free
            H[N + nc + j, N + j] = 1.0
        return H


class NDCompositeCost(BaseCost):

    def __init__(self, cost_terms: List[Union[CompositeCost, CostSegment]]):
        self.cost_terms = cost_terms

    def __iter__(self):
        return iter(self.cost_terms)

    def _get_sub_cost_term(self, dim=None, seg=None):
        """Access cost by dimension and/or segment."""
        if dim is None:
            return self.cost_terms
        elif seg is None:
            return self.cost_terms[dim]
        else:
            return self.cost_terms[dim]._get_sub_cost_term(seg=seg)

    @property
    def cost_terms(self):
        """Iterable : List/Tuple of CostSegment/CompositeCost"""
        return self._costs

    @cost_terms.setter
    def cost_terms(self, costs):

        # check type
        for cost in costs:
            assert isinstance(cost, (CompositeCost, CostSegment)), 'Element of costs is not a CompositeCost/CostSegment'

        # check alssm output dimensions L
        Qs = np.array([cost.get_alssm_output_dimension() for cost in costs])
        assert np.all(Qs == Qs[0]), 'Output Dimension of CompositeCosts do not match'

        self._costs = costs

    @property
    def L(self):
        """int : Number of Dimensions/Costs :math:`L`"""
        return len(self.cost_terms)

    def get_number_of_dimensions(self):
        return self.L

    def get_alssm_order(self):
        return int(np.prod([cost.get_alssm_order() for cost in self.cost_terms]))

    def get_alssm_output_dimension(self):
        return self.cost_terms[0].get_alssm_output_dimension()

    def get_steady_state_W(self, dim_order=None, method='schur'):
        if dim_order is None:
            dim_order = range(self.L)

        W = [1]
        for n in dim_order:
            W = np.kron(W, self.cost_terms[n].get_steady_state_W(method))
        return W

    def eval_alssm_output(self, xs, nd_alssm_weights=None):
        """
        Evaluate n-dimensional Alssm output

        Parameters
        ----------
        xs : array_like of shape(..., N)
            states at which to evaluate the output
        nd_alssm_weights : array_like of shape(L, M), optional
            Alssm weights for each dimension

        Returns
        -------
        ndarray of shape(..., [Q])
            Alssm output for each state in xs
        """
        alssms = []
        for l in range(self.L):
            sub_cost = self._get_sub_cost_term(dim=l)
            if nd_alssm_weights is not None:
                alssms.append(AlssmSum(sub_cost.alssms, nd_alssm_weights[l]))
            else:
                alssms.append(AlssmSum(sub_cost.alssms))
        alssm = AlssmProd(alssms)
        return alssm.eval_output(xs)

class ConstrainMatrix:
    """
    Constrain Matrix Generator

    Builder class to set up matrix `H` as a linear constraint for the squared error minimization.

    Parameters
    ----------
    cost : CompositeCost, CostSegment
        CompositeCost or CostSegment

    """

    TSLM_TYPES = ('free',
                  'continuous',
                  'straight',
                  'horizontal',
                  'left horizontal',
                  'right horizontal',
                  'peak',
                  'step'
                  )
    """tuple of string: Two Sided Line Model Types. see REF
    """

    def __init__(self, cost):
        assert isinstance(cost, (CompositeCost, CostSegment)), 'cost not of type CompositeCost,'
        self._cost = cost
        self._N = self._cost.get_alssm_order()
        self._data = np.eye(self._N)

    def constrain(self, indices, value):
        """
        Add constraining

        I.e., Apply a dependency between two indices.

        Parameters
        ----------
        indices: array_like of shape(2,)
            indices to apply a dependency
        value

        Returns
        -------
        s : ConstrainMatrix
            self
        """
        assert 0 <= indices[0] < self._N, f'first index not in range [0, {self._N - 1}]'
        assert 0 <= indices[1] < self._N, f'second index not in range [0, {self._N - 1}]'
        self._data[indices[0], indices[1]] = value
        self._data[indices[1], indices[0]] = value
        return self

    def constrain_by_labels(self, label_1, label_2, value):
        """
        Add constraining by labels

        I.e., Apply a dependency between two indices.

        Parameters
        ----------
        label_1: str
            label of the first state variable index to apply a dependency
        label_2: str
            label of the second state variable index to apply a dependency
        value: int, float
            dependency value

        Returns
        -------
        s : ConstrainMatrix
            self
        """

        index_1 = self._cost.get_state_var_indices(label_1)
        index_2 = self._cost.get_state_var_indices(label_2)
        assert len(index_1) == 1, 'index of label_1 contains more the one element'
        assert len(index_2) == 1, 'index of label_2 contains more the one element'
        return self.constrain(index_1 + index_2, value)

    def digest(self):
        """
        Reruns a "snapshot" of the constraint matrix with the applied constraints

        Returns
        -------
        H : :class:`numpy.ndarray` of shape(N, M)
            Constrain Matrix
        """
        H = self._data.copy()

        del_cols = []

        # append identical columns to del_cols
        for n in range(self._N):
            indices = np.flatnonzero(np.all(H[n, :] == H, axis=1))
            if indices.size > 1:
                del_cols.extend(np.sort(indices)[1:])

        # append zero columns to del_cols
        del_cols.extend(np.where(~H.any(axis=0))[0])

        # check if all combinations of columns in H are a mulitple of each-other when yes add last one to del_cols
        for j, c1 in enumerate(H.T):
            c1_norm = np.linalg.norm(c1)
            if c1_norm != 0.0:
                for i, c2 in enumerate(H.T):
                    c2_norm = np.linalg.norm(c2)
                    if c2_norm != 0.0:
                        is_multiple = np.all(np.divide(c1, c1_norm) - np.divide(c2, c2_norm) == 0.0)
                        is_multiple = is_multiple or np.all(
                            np.divide(-1 * c1, np.linalg.norm(-1 * c1)) - np.divide(c2, c2_norm) == 0.0)
                        if is_multiple and i > j:
                            del_cols.append(i)

        # delete columns
        if len(del_cols) > 0:
            return np.delete(H, np.unique(del_cols), 1)
        return H

    def print_table(self):
        """Prints the actual table of constraints to the console"""
        print(*[f' {c}' for c in range(self._N)])
        print(' ——' * self._N)
        for r, row in enumerate(self._data):
            print(row, ' | ', r)
