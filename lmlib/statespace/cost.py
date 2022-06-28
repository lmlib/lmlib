"""Definition of recurively computed squared error cost functions (such as *Cost Segments* and *Composite Costs*), all based on ALSSMs"""

import warnings
import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Iterable

from lmlib.statespace.model import ModelBase, AlssmStackedSO
from lmlib.statespace.recursion import *
from lmlib.statespace.backend import get_backend, BACKEND_TYPES, AVAILABLE_BACKENDS
from lmlib.utils.check import *

__all__ = ['ConstrainMatrix', 'CompositeCost', 'CostSegment', 'Segment', 'RLSAlssm',
           'RLSAlssmSet', 'RLSAlssmSteadyState', 'RLSAlssmSetSteadyState', 'FW', 'FORWARD', 'BW', 'BACKWARD',
           'map_trajectories', 'map_windows', 'create_rls']

BACKWARD = 'bw'
"""str : Sets the recursion direction in a :class:`Segment` to backward, use :const:`BACKWARD` or :const:`BW`"""
BW = BACKWARD
"""str : Sets the recursion direction in a :class:`Segment` to backward, use :const:`BACKWARD` or :const:`BW`"""
FORWARD = 'fw'
"""str : Sets the recursion direction in a :class:`Segment` to forward, use :const:`FORWARD` or :const:`FW`"""
FW = FORWARD
"""str : Sets the recursion direction in a :class:`Segment` to forward, use :const:`FORWARD` or :const:`FW`"""


def _merge_ks_seg(arr, merge_ks, merge_seg):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if merge_seg and not merge_ks:
            return np.nanmax(arr, axis=1)
        if not merge_seg and merge_ks:
            return np.nanmax(arr, axis=0)
        if merge_seg and merge_ks:
            return np.nanmax(arr, axis=(0, 1))
        if not merge_seg and not merge_ks:
            return arr


def _covariance_matrix_closed_form(A, C, gamma, a, b, delta):
    N = np.shape(A)[0]
    gATA = gamma * np.kron(np.transpose(A), A)

    if gamma > 1:
        gATA_a = np.linalg.matrix_power(gATA, a - 1) if ~(np.isinf(a)) else np.zeros_like(gATA, dtype=float)
        gATA_b = np.linalg.matrix_power(gATA, b) if ~(np.isinf(b)) else np.zeros_like(gATA, dtype=float)
        if np.linalg.cond(np.linalg.inv(gATA) - np.eye(N * N)) > 1e15:
            print(Warning('Badly Conditioned Steady State Matrix W: Use larger boundaries or lower g.'))

        return np.dot(gamma ** (-delta),
                      np.kron(np.eye(N), np.atleast_2d(C)) @
                      (np.linalg.inv(np.linalg.inv(gATA) - np.eye(N * N)) @ (gATA_a - gATA_b)) @
                      np.kron(np.atleast_2d(C).T, np.eye(N))
                      )
    else:
        gATA_a = np.linalg.matrix_power(gATA, a) if ~(np.isinf(a)) else np.zeros_like(gATA)
        gATA_b = np.linalg.matrix_power(gATA, b + 1) if ~(np.isinf(b)) else np.zeros_like(gATA)
        if np.linalg.cond(np.eye(N * N) - gATA) > 1e15:
            print(Warning('Badly Conditioned Steady State Matrix W: Use larger boundaries or lower g.'))
        return np.dot(gamma ** (-delta),
                      np.kron(np.eye(N), np.atleast_2d(C)) @
                      (np.linalg.inv(np.eye(N * N) - gATA) @ (gATA_a - gATA_b)) @
                      np.kron(np.atleast_2d(C).T, np.eye(N))
                      )


def create_rls(cost, multi_channel_set=False, steady_state=False, kappa_diag=True, steady_state_method='closed_form'):
    """
    Returns the right Recursive Least Square Obeject by configureation

    Parameters
    ----------
    cost : CostBase
        cost model
    multi_channel_set : bool
        Set to True if a RLSAlssmSet* is desired
    steady_state : bool
        Set to True if a Steaey State scheme is desired
    kappa_diag : bool
        If True a RLSAlssmSet* will performe a diagnoal kappa matrix. Only if multi_channel_set = True
    steady_state_method : str
        Type of Steady State method. Available: ('closed_form'). Only if steady_state == True

    Returns
    -------
    out : RLSAlssm, RLSAlssmSet, RLSAlssmSteadyState or RLSAlssmSetSteadyState
        Returns Recursive Least Square Obeject
    """
    if isinstance(cost, CostBase):
        if not multi_channel_set and not steady_state:
            return RLSAlssm(cost)

        if not multi_channel_set and steady_state:
            return RLSAlssmSteadyState(cost,
                                       steady_state_method=steady_state_method)

        if multi_channel_set and steady_state:
            return RLSAlssmSetSteadyState(cost,
                                          kappa_diag=kappa_diag,
                                          steady_state_method=steady_state_method)

        if multi_channel_set and not steady_state:
            return RLSAlssmSet(cost, kappa_diag=kappa_diag)

    else:
        assert False, 'cost is not subclass of CostBase()'


def map_windows(windows, ks, K, merge_ks=False, merge_seg=False, fill_value=0):
    """
    Maps the window amplitudes of one or multiple :class:`Segment` to indices `ks` into a common target output vector of length `K`.

    The parameter :attr:`windows` is commonly directly the output of one of the following methods:

    * :meth:`CostSegment.windows`
    * :meth:`CompositeCost.windows`

    Parameters
    ----------
    windows :
        * `tuple` :code:`(range, tuple)`, as output by :meth:`CostSegment.windows`.
        * `tuple of shape=(P) of tuples` :code:`(range, tuple)`, as  output by :meth:`CompositeCost.windows`.
    ks : array_like of shape=(KS) of ints
        target indices in the target output vector, where to map the windows to
    K : int
        Length of target output vector
    merge_ks : bool, optional
        If :code:`True`, all windows from different target indices :attr:`ks` are merged into a single array of length
        :attr:`K`; if the windows of two target indices overlap, only the window with the larger value remains.
    merge_seg : bool, optional
        If :code:`True`, all segments `P` are merged to a single target vector of length :attr:`K`;
        if the windows of two segments overlap,
        only window value with the larger value remains.
    fill_value : scalar or None, optional
        Default value of target output vector elements, when no window is assigned.

    Returns
    -------
    output : :class:`~numpy.ndarray` of shape=([KS,] [P,] K) of floats
        * Dimension `KS` only exists, if :attr:`merge_ks`:code:`=False`.
        * Dimension `P` only exists, if :attr:`merge_seg`:code:`=False`.



    |def_KS|
    |def_P|
    |def_K|

    Examples
    --------

    .. minigallery:: lmlib.map_windows

    """

    # return empty array if ks is empty
    if len(ks) == 0:
        return np.array([])

    w = np.full((len(ks), len(windows), K), fill_value, dtype=float)
    for p, (ab_range, weights) in enumerate(windows):
        ab_range = np.arange(ab_range.start, ab_range.stop)

        for i, k in enumerate(ks):
            boundary_mask = np.bitwise_and(0 <= ab_range+k, ab_range+k < K)
            w[i, p, ab_range[boundary_mask]+k] = weights[boundary_mask]

    return _merge_ks_seg(w, merge_ks, merge_seg)


def map_trajectories(trajectories, ks, K, merge_ks=False, merge_seg=False, fill_value=np.nan):
    """Maps trajectories at indices `ks` into a common target output vector of length `K`.

    The parameter :attr:`trajectories` is commonly directly the output of one of the following methods:

    * :meth:`CostSegment.trajectories`
    * :meth:`CompositeCost.trajectories`

    Parameters
    ----------
    trajectories :
        * `list of shape=(XS) of tuples` :code:`(range, tuple)`,
            such tuples are the output of :meth:`CostSegment.trajectories`.
        * `list of shape=(XS) of tuples of shape=(P) of tuple` :code:`(range, tuple)`,
            such tuples are the output of :meth:`CompositeCost.trajectories`.
    ks : array_like of shape=(XS) of ints
         target indices in the target output vector, where to map the windows to
    K : int
        Length of target vector
    merge_ks : bool, optional
        If :code:`True`, all trajectories from different target indices :attr:`ks` are merged into a single array of length :attr:`K`;
        if the trajectories of two target indices overlap,
        only trajectory value with the larger :class:`CostSegment` window value remains.
    merge_seg : bool, optional
        If :code:`True`, all segments `P` are merged to a single target vector of length :attr:`K`;
        if the trajectories of two segments overlap,
        only trajectory value with the larger :class:`CostSegment` window value remains.
    fill_value : scalar, optional
        Default value of target output vector elements, when no trajectory is assigned.

    Returns
    -------
    output : :class:`~numpy.ndarray` `of shape=shape of ([XS,] [P,] K, L [,S]) of floats`

        * Dimension `XS` only exists, if :attr:`merge_ks`:code:`=False`.
        * Dimension `P` only exists, if :attr:`merge_seg`:code:`=False`
        * Dimension `S` only exists, if the parmeter :attr:`xs` of :meth:`CompositeCost.trajectories` or
          :meth:`CompositeCost.trajectories` also provides dimension `S` (i.e., provides multiple signal set processing).


    |def_XS|
    |def_P|
    |def_K|
    |def_L|
    |def_S|


    Examples
    --------
    .. minigallery:: lmlib.map_trajectories


    """

    # return empty array if ks is empty
    if len(ks) == 0:
        return np.array([])

    t_ = trajectories[0][0][1]
    P = len(trajectories[0])
    XS = len(trajectories)
    assert XS == len(ks), "number of trajecotries and ks does not match up"

    if t_.ndim == 2: # check if trajecotry is from a RLSAlssmSet
        S = t_.shape[1]
        t = np.full((XS, P, K, S), fill_value)
    else:
        t = np.full((XS, P, K), fill_value)

    for i, k in enumerate(ks):
        for p, (j_range, trajectory) in enumerate(trajectories[i]):
            mask =  np.bitwise_and(k + np.array(j_range)>=0, k + np.array(j_range)<K)
            t[i, p, k + np.array(j_range)[mask], ...] = trajectory[mask]

    return _merge_ks_seg(t, merge_ks, merge_seg)


class Segment:
    r"""
    Segment represents a window of finite or infinite interval borders used to select and weight signal samples in a cost function.

    Segments are commonly used in combination with ALSSM signal models to select and weight the samples in cost
    functions, see :class:`CostSegment` or :class:`CompositeCost`. The window of a segment either has an exponentially
    decaying shape or is defined by the output of its own ALSSM model, denoted ast the `window ALSSM`.
    The selection of a window also controls the direction of a stable recursive cost computation (forward or backward).


    In cunjunction with an ALSSM,
    a Segment leads to a cost function of the form

    .. math::
        J_k(x) = \sum_{i=k+a}^{k+b} \gamma^{i-k-\delta}\big(CA^{i-k}x - y_i\big)^2 \ ,

    and when additionally using sample weights :math:`v_k`, of the form

    .. math::
        J_k(x) = \sum_{i=k+a}^{k+b} v_k  {\alpha}_{k+\delta}(k+\delta) \big(CA^{i-k}x - y_i\big)^2 \ ,

    with the sample weights :math:`v_k`
    and the window weight :math:`\alpha_k(j)` which depends on the sample weights, see Equation (14) in [Wildhaber2018]_

    See also [Wildhaber2018]_ [Wildhaber2019]_

    Parameters
    ----------
    a : int, np.inf
        Left boundary of the segment's interval
    b : int, np.inf
        Right boundary of the segment's interval
    g : int, float
        :math:`g > 0`. Effective number of samples under the window. This is used as a (more readable) surrogate for the
        window decay of exponential windows, see [Wildhaber2018]_. |br|
        :math:`g` is counted to the left or right of :math:`k+ \delta`, for for forward or backward computation
        direction, respectively.
    direction : str
        Computation direction of recursive computations (also selects a left- or right-side decaying window) |br|
        :data:`statespace.FORWARD` or `'fw'` use forward computation with forward recursions |br|
        :data:`statespace.BACKWARD` or `'bw'` use backward computation with backward recursions
    delta : int, optional
        Exponential window is normalized to 1 at relative index :code:`delta`.
    label : str, None, optional
        Segment label, useful for debugging in more complex systems (default: label = None)

    Notes
    -----
    The interval of the semgment includes both boundaries `a` and `b` into the calculations.
    i.e., if the sum runs over the interval :math:`k \in [a,b] ` it treats `b-a+1` samples.

    Examples
    --------
    >>> segment = lm.Segment(a=-20, b=-1, direction=lm.FORWARD, g=15)
    >>> print(segment)
    Segment : a:-20, b:-1, fw, g:15, delta:0, label: None

    >>> segment = lm.Segment(a=-0, b=100, direction=lm.BACKWARD, g=15, delta=30, label="right-sided window with shift")
    >>> print(segment)
    Segment : a:0, b:100, bw, g:15, delta:30, label: right-sided window with shift

    .. minigallery:: lmlib.Segment
        :add-heading:

    """

    def __init__(self, a, b, direction, g, delta=0, label=None):
        self._a = None
        self._b = None
        self.set_boundaries(a, b)
        self.direction = direction
        self.g = g
        if self.direction == FW:
            self.gamma = self.g / (self.g - 1)
        else:
            self.gamma = (self.g - 1) / self.g
        self.delta = delta
        self.label = 'n/a' if label is None else label

    def __str__(self):
        return f'{type(self).__name__}(a:{self.a}, b:{self.b}, {self.direction}, g:{self.g}, delta:{self.delta}, label: ' \
               f'{self.label})'

    @property
    def a(self):
        """int, np.inf: Left boundary of the segment's interval :math:`a`"""
        return self._a

    @property
    def b(self):
        """int, np.inf : Right boundary of the segment's interval :math:`b`"""
        return self._b

    @property
    def g(self):
        """int, float : Effective number of samples :math:`g`, setting the window with

        The effective number of samples :math:`g` is used to derive
        and set the window decay factor :math:`\\gamma` internally.

        [Wildhaber2018] [Section III.A]

        """
        return self._g

    @g.setter
    def g(self, g):
        assert isinstance(g, (int, float)), 'Effective number of samples g is not of type integer or float.'
        assert g > 1, 'Effective number of samples g has to be greater than one.'
        self._g = g

    @property
    def direction(self):
        """str : Sets the segment's recursion computation `direction`

            - :data:`FORWARD`, :data:`FW` or `'fw'` use forward computation with forward recursions
            - :data:`BACKWARD`, :data:`BW` or `'bw'` use backward computation with backward recursions
        """
        return self._direction

    @direction.setter
    def direction(self, direction):
        assert isinstance(direction, str), 'Computation direction is not of type string.'
        assert direction in (BW, FW), f'Unknown direction parameter: {self.direction}'
        self._direction = direction

    @property
    def delta(self):
        """int : Relative window shift :math:`\\delta`"""
        return self._delta

    @delta.setter
    def delta(self, delta):
        assert isinstance(delta, int), 'Relative window shift delta is not of type integer.'
        self._delta = delta

    @property
    def label(self):
        """str, None : Label of the segment"""
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def gamma(self):
        r"""float : Window decay factor :math:`\gamma`

        Window decay factor :math:`\gamma` is set internally on the initialization of a new segment object
        and is derived from the *effective number of samples* :attr:`Segment.g` as follows:

        - for a segment with forward recursions: :math:`\gamma = \frac{g}{g-1}`
        - for a segment with forward recursions: :math:`\gamma = \big(\frac{g}{g-1}\big)^{-1}`

        [Wildhaber2018] [Table IV]

        """
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        assert isinstance(gamma, float), 'Window decay factor gamma is not of type float.'
        self._gamma = gamma

    def set_boundaries(self, a, b):
        assert isinstance(a, int) or np.isinf(a), 'Left boundary is not of type integer or np.inf.'
        assert isinstance(b, int) or np.isinf(b), 'Right boundary is not of type integer or np.inf.'
        assert a <= b, 'left boundary \'a\' has to be smaller/equal to the right boundary \'b\'.'
        self._a = a
        self._b = b

    def window(self, thd=1e-6):
        r"""
        Returns the per-sample window weighs

        The return values are the window weights :math:`\alpha_{\delta}(i) \quad \forall i \in [a, b]` for a constant
        :math:`\gamma`. The window weight function is defined as

        .. math::
            w_i = \gamma^{i}

        For more details see [Wildhaber2018]_.

        Parameters
        ----------
        thd : float, None, optional
            Threshold for infinite Segment boundaries. Crops any window weight below the threshold.

        Returns
        -------
        `tuple` :code:`(range, array)`

            * :code:`range` of length `JR`: relative index range of window with respect to semgent's boundaries.
            * :code:`array` of shape=(JR) of floats: per-index window weight over the reported index range


        |def_JR|


        Examples
        --------
        .. plot:: pyplots/Segment_window_plot.py
                :include-source:

        """

        if self.direction == FW:
            a_lim = max(np.log(thd) / np.log(self.gamma) - 1, self.a)
            b_lim = self.b
        else:  # self.direction == BW:
            a_lim = self.a
            b_lim = min(np.log(thd) / np.log(self.gamma) + 1, self.b)
        ab_range = range(int(a_lim), int(b_lim)+1)
        return ab_range, np.power(self.gamma, np.array(ab_range)-self.delta)


class CostBase(ABC):
    """
    Abstract baseclass for  :class:`.CostSegment` and :class:`.CompositeCost`

    Parameters
    ----------
    label : str, optional
        Label of Alssm, default: 'n/a'

    """

    def __init__(self, label='n/a'):
        self._segments = None
        self._alssms = None
        self._F = None
        self.label = label

    @property
    def alssms(self):
        """tuple : Set of ALSSM"""
        return self._alssms

    @alssms.setter
    def alssms(self, alssms):
        assert isinstance(alssms, Iterable), 'alssms is not iterable'
        for alssm in alssms:
            assert isinstance(alssm, ModelBase), 'element in alssms is not of instance ModelBase'
        assert common_C_dim(alssms), "Alssms has not same C dimensions."
        self._alssms = list(alssms)

    @property
    def segments(self):
        """tuple : Set of Segment"""
        return self._segments

    @segments.setter
    def segments(self, segments):
        assert isinstance(segments, Iterable), 'segments is not iterable'
        for segment in segments:
            assert isinstance(segment, Segment), 'element in segments is not instance of Segment'
        self._segments = list(segments)

    @property
    def label(self):
        """str, None : Label of the Cost Model"""
        return self._label

    @label.setter
    def label(self, label):
        assert isinstance(label, str)
        self._label = label

    @property
    def F(self):
        """:class:`~numpy.ndarray` : Mapping matrix :math:`F`, maps models to segments"""
        return self._F

    @F.setter
    def F(self, F):
        assert is_array_like(F), 'F is not array_like'
        F = np.atleast_2d(F)
        assert np.shape(F) == (len(self.alssms), len(self.segments)), f'F has wrong shape, {info_str_found_shape(F)}'
        self._F = F

    def get_model_order(self):
        """int : Order of the (stacked) Alssm Model"""
        return AlssmStackedSO(self.alssms).N

    def trajectories(self, xs, F=None, thd=1e-6):
        """
        Returns the :class:`CompositeCost`'s ALSSM's trajectories (=output sequences for a fixed initial states) for
        multiple initial state vectors :attr:`xs`

        Evaluates the :class:`CompositeCost`'s ALSSM with the state vectors :attr:`xs` over the time indices defined
        by :attr:`CompositeCost.segments`.
        The segment's interval boundaries limit the evaluation interval (samples outside this interval are set to 0).
        In particular for segments with infinite interval borders, the threshold option :attr:`thds` additionally
        limits the evaluation boundaries by defining a minimal window height of the :attr:`CompositeCost.segments`.

        Parameters
        ----------
        xs : array_like of shape=(XS, N [,S]) of floats
            List of initial state vectors :math:`x`
        F : array_like, shape(M, P) of int, shape(M,) of int
            Mapping matrix. If not set to :code:`None`, the given matrix overloads the CompositeCost's internal mapping matrix as provided by the class constructor for :class:`CompositeCost`.
            If :attr:`F` is only of :code:`shape(M,)`, the vector gets enlarged to size :code:`shape(M, P)` by repeating the vector `P`-times.
            (This is a shortcut to select a single ALSSM over all segments.)
        thds : list of shape(P) of floats, scalar, None, optional
            Per segment threshold limiting the evaluation boundaries by setting a minimum window height of the
            associated :attr:`~CompositeCost.segments`
            Scalar to use the same threshold for all available segments.

        Returns
        -------
        trajectories : list of shape=(XS) of tuples of shape=(P) of tuples :code:`(range, array)`
            Each element in `trajectories` is a tuple with

            * :code:`range` `of length JR`: relative index range of trajectory with respect to semgent's boundaries
            * :code:`array` `of shape(JR, L, [S])`: trajectory values over reported range.

            Dimension `S` is only present in the output, if dimension `S` is also present in :attr:`xs` (i.e., if multiple signal sets are used)


        |def_JR|
        |def_XS|
        |def_N|
        |def_S|
        |def_M|
        |def_P|


        """
        if F is None:
            F = self.F
        M, P = np.shape(F)
        windows = CostBase.windows(self, np.arange(P), thd)
        trajectories = []
        for x in xs:
            trajectories_per_segment = []
            for alssm_weights, (ab_range, _) in zip(np.transpose(F), windows):
                y = AlssmStackedSO(self.alssms, deltas=alssm_weights).trajectory(x, ab_range)
                trajectories_per_segment.append([ab_range, y])
            trajectories.append(trajectories_per_segment)
        return trajectories

    def windows(self, segment_indices, thd=1e-6):
        """
        Returns for each selected segment its window generated by :meth:`CostSegment.windows`.

        The segments are selcted by :attr:`segment_selection`.


        Parameters
        ----------
        segment_selection : array_like, shape=(P,) of Boolean
            Enables (:code:`True`, 1) or disables (:code:`False`, 0) the evaluation of the
            p-th Segment in :attr:`CompositeCost.segments` using :meth:`CostSegment.window`
        thds : list of shape=(P,) of floats, scalar, optional
            List of per-segment threshold values or scalar to use the same threshold for all segments
            Evaluation is stopped for window weights below the given threshold.
            Set list element to :code:`None` to disable the threshold for a segment.

        Returns
        -------
        `list of shape=(P) of tuples` :code:`(range, array)`
            Each element is a tuple with

            * :code:`range` of length `JR`: relative index range of window with respect to semgent's boundaries.
            * :code:`array` of shape=(JR) of floats: per-index window weight over the reported index range


        |def_JR|
        |def_P|


        """

        assert is_array_like(segment_indices), 'segment_index is not an integer nor array_like'
        assert 0 <= min(segment_indices) and max(segment_indices) < len(self.segments), 'segment_index out of range'

        window = []
        for i in segment_indices:
            window.append(self.segments[i].window(thd))
        return window

    def get_steady_state_W(self, method='closed_form'):
        """
        Returns Steady State Matrix W

        Parameters
        ----------
        method : str, optional
            if 'closed_form' is used the steady state matrix will be calculated in a close form.
            This method can be critical, as it can produce badly conditioned matrices internally.
            if 'limited_sum' is used, the steady state matrix will be calculated brute force, with a stop condition
            on a minimum change.

        Returns
        -------
        Wss = `class:numpy.ndarray`
            Steady State Matrix W
        """
        N = self.get_model_order()
        W = np.zeros((N, N))
        for i, segment in enumerate(self.segments):
            alssm = AlssmStackedSO(self.alssms, self.F[:, i])
            if method == 'closed_form':
                W += _covariance_matrix_closed_form(alssm.A, alssm.C, segment.gamma, segment.a, segment.b, segment.delta)
        return W

    def eval_alssm_output(self, xs, alssm_weights):
        """
        Evaluation of the ALSSM for multiple state vectors `xs`.

        **See:** :meth:`~lmlib.statespace.models.Alssm.eval`

        Parameters
        ----------
        xs : array_like of shape=(XS, N [,S]) of floats
            List of state vectors :math:`x`
        alssm_selection : array_like, shape=(M,) of bool
            Each element enables (:code:`True`, 1) or disables (:code:`False`, 0) the m-th ALSSM in :attr:`CompositeCost.alssms`.

        Returns
        -------
        s : :class:`~numpy.ndarray` of shape=(XS,[J,]L[,S]) of floats
            ALSSM outputs


        |def_N|
        |def_L|
        |def_M|
        |def_S|
        |def_XS|

        Examples
        --------
        .. minigallery:: lmlib.CostBase.eval_alssm_output

        """

        return AlssmStackedSO(self.alssms, alssm_weights).eval_states(xs)

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
        return AlssmStackedSO(self.alssms, label='cost').get_state_var_indices('cost.' + label)


class CompositeCost(CostBase):
    r"""
    Quadratic cost function defined by a conjunction one or multiple  of :class:`~lmlib.statespace.model.Alssm` and :class:`.Segment`

    A composite costs combines multiple ALSSM models and multiple Segments in form of a grid,
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


    This figure shows a graphical representation of a composite cost, depicting the internal relationships between Segments, ALSSMs, and the mapping matrix F. F[m,p] is implemented as a scalar factor multiplied on the alssm's output signal.


    For more details, seen Chapter 9 in [Wildhaber2019]_.
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
    **kwargs
        Forwarded to :class:`.CostBase`


    |def_M|
    |def_P|


    Examples
    --------
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

    def __init__(self, alssms, segments, F, **kwargs):
        super().__init__(**kwargs)
        self.alssms = alssms
        self.segments = segments
        self.F = F


class CostSegment(CostBase):
    r"""
    Quadratic cost function defined by an ALSSM and a Segment.

    A CostSegment is an implementation of [Wildhaber2019]_ :download:`PDF (Cost Segment) <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=117>`

    .. code-block:: text

        ==================
        Class: CostSegment (= 1 ALSSM + 1 Segment)
        ==================

          window weight
              ^
            2_|                   ___     exponentially rising window of segment
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
    alssm : :class:`_ModelBase`
        ALSSM, defining the signal model
    segment : :class:`Segment`
        Segment, defining the window
    **kwargs
        Forwarded to :class:`.CostBase`

    Examples
    --------
    Setup a cost segment with finite boundaries and a line ALSSM.

    >>> alssm = lm.AlssmPoly(poly_degree=1, label="slope with offset")
    >>> segment = lm.Segment(a=-30, b=0, direction=lm.FORWARD, g=20, label="finite left")
    >>> cost = lm.CostSegment(alssm, segment, label="left line")
    >>> print(cost)
    CostSegment : label: left line
      └- Alssm : polynomial, A: (2, 2), C: (1, 2), label: slope with offset,
      └- Segment : a:-30, b:0, fw, g:20, delta:0, label: finite left

    """

    def __init__(self, alssm, segment, **kwargs):
        super().__init__(**kwargs)
        self.alssms = [alssm]
        self.segments = [segment]
        self.F = np.ones((1, 1))

    def __str__(self):
        return f'{type(self).__name__}(label: {self.label}) \n  └- {self.alssms[0]}, \n  └- {self.segments[0]}'

    def windows(self, thd=1e-6, *args):
        """
        Per-sample evaluation of the :class:`CostSegment`'s window.

        Returns the index range and the per-index window weights defined by the CostSegment's :code:`Segment`.

        **For parameter list see:** :meth:`Segment.windows`

        """

        return super().windows([0], thd)

    def trajectories(self, xs, alssm_weight=1.0, thd=1e-6):
        """
        Returns the ALSSM's trajectory (=output sequence for fixed initial state) for multiple initial state vectors

        Evaluates the :Class:`CostSegment`'s ALSSM with the state vectors `xs` over the time indices defined by :attr:`CostSegment.segment`.
        The segment's interval boundaries limit the evaluation interval (samples outside this interval are set to 0).
        In particular for segments with infinite interval borders, the threshold option :attr:`thd` additionally limits the evaluation boundaries by defining a minimal window height of the :attr:`CostSegment.segment`.


        Parameters
        ----------
        xs : array_like of shape=(XS, N [,S]) of float
            List of length `XS` with initial state vectors :math:`x`
        thd : float or None, optional
            Threshold setting the evaluation boundaries by setting a minimum window height

        Returns
        -------
        trajectories : list of shape=(XS) of tuples :code:`(range, array)`
            Each element in `trajectories` is a tuple with

            * :code:`range` of length `JR`: relative index range of trajectory with respect to semgent's boundaries.
            * :code:`array` of shape=(JR, L, [S]): trajectory values over reported range.

            Dimension `S` is only present in the output, if dimension `S` is present in `xs` (i.e., if multiple signal sets are used)


        |def_JR|
        |def_XS|
        |def_N|
        |def_S|
        |def_L|

        """

        return super().trajectories(xs, [[alssm_weight]], thd)

    def eval_alssm_output(self, xs, alssm_weight=1.0):
        """
        Evaluation of the :class:`CostSegment`'s ALSSM output for multiple state vectors

        **For parameter details see:** :meth:`lmlib.statespace.models.Alssm.eval`

        """
        return super().eval_alssm_output(xs, [alssm_weight])


class RLSAlssmBase(ABC):
    """
    Base Class for Recursive Least Square Alssm Classes


    Parameters
    ----------
    betas : array_like of shape=(P,) of floats, None, optional
        Segment Scalars. Factors weighting each of the `P` cost segments.
        If `betas` is not set, the weight is for each cost segment 1.
    """
    def __init__(self, betas=None):
        self._cost_model = None
        self._W = None
        self._xi = None
        self._kappa = None
        self._nu = None
        self.betas = betas
        self._backend = get_backend()

    @property
    def cost_model(self):
        """CostSegment, CompositeCost :  Cost Model"""
        return self._cost_model

    @cost_model.setter
    def cost_model(self, cost_model):
        assert isinstance(cost_model, CostBase), 'cost_model is not a subclass of CostBase'
        self._cost_model = cost_model

    @property
    def betas(self):
        """~numpy.ndarray : Segment scalars weights the cost function per segment"""
        return self._betas

    @betas.setter
    def betas(self, betas):
        if betas is None:
            self._betas = None
        else:
            assert is_array_like(betas), 'betas if not array_like'
            assert len(betas) == len(
                self._cost_model.segments), f'betas has wrong length, {info_str_found_shape(betas)}'
            self._betas = betas

    @property
    def W(self):
        """:class:`~numpy.ndarray` : Filter Parameter :math:`W`"""
        return self._W

    @property
    def xi(self):
        """:class:`~numpy.ndarray` :  Filter Parameter :math:`\\xi`"""
        return self._xi

    @property
    def kappa(self):
        """:class:`~numpy.ndarray`  : Filter Parameter :math:`\\kappa`"""
        return self._kappa

    @property
    def nu(self):
        """:class:`~numpy.ndarray`  : Filter Parameter :math:`\\nu`"""
        return self._nu

    @abstractmethod
    def _allocate_parameter_storage(self, input_shape):
        pass

    def set_backend(self, backend):
        """
        Setting the backend computations option

        Parameters
        ----------
        backend : str
            'py', for python backend, 'jit' for Just in Time backend

        """
        assert backend in BACKEND_TYPES, f'Wrong backend name {backend}'
        assert backend in AVAILABLE_BACKENDS, f'{backend} not available, check {backend} installation.'
        self._backend = backend

    @abstractmethod
    def _forward_recursion(self, segment, A, C, y, v, beta):
        pass

    @abstractmethod
    def _backward_recursion(self, segment, A, C, y, v, beta):
        pass

    def filter(self, y, v=None):
        """
        Computes the intermediate parameters for subsequent squared error computations and minimizations.

        Computes the intermediate parameters using efficient forward- and backward recursions.
        The results are stored internally, ready to solve the least squares problem using e.g., :meth:`minimize_x`
        or :meth:`minimize_v`. The parameter allocation :meth:`allocate` is called internally,
        so a manual pre-allcation is not necessary.

        Parameters
        ----------
        y : array_like
            Input signal |br|
            :class:`RLSAlssm` or :class:`RLSAlssmSteadyState`
                Single-channel signal is of `shape =(K,)` for |br|
                Multi-channel signal is of `shape =(K,L)` |br|
            :class:`RLSAlssmSet` or :class:`RLSAlssmSetSteadyState`
                Single-channel set signals is of `shape =(K,S)` for |br|
                Multi-channel set signals is of `shape =(K,L,S)` |br|
            Multi-channel-sets signal is of `shape =(K,L,S)`
        v : array_like, shape=(K,), optional
            Sample weights. Weights the parameters for a time step `k` and is the same for all multi-channels.
            By default the sample weights are initialized to 1.


        |def_K|
        |def_L|
        |def_S|

        """

        self._allocate_parameter_storage(np.shape(y))

        segments = self.cost_model.segments
        alssms = self.cost_model.alssms

        A = AlssmStackedSO(alssms).A

        if v is None:
            v = np.ones(np.shape(y)[0])

        betas = np.ones(len(segments)) if self.betas is None else self.betas

        for i, (segment, beta) in enumerate(zip(segments, betas)):

            # calculate output matrix C for the segment
            tmp_c = []
            for j, alssm in enumerate(alssms):
                tmp_c.append(self.cost_model.F[j, i] * alssm.C)
            C = np.hstack(tmp_c)

            if segment.direction == FW:
                self._forward_recursion(A, C, segment, y, v, beta)
            elif segment.direction == BW:
                self._backward_recursion(A, C, segment, y, v, beta)
            else:
                ValueError('segment.direction has wrong value.')


class RLSAlssm(RLSAlssmBase):
    """
    Filter and Data container for Recursive Least Sqaure Alssm Filters

    :class:`RLSAlssm` computes and stores intermediate values such as covariances,
    as required to efficiently solve recursive least squares problems
    between a model-based cost function :class:`CompositeCost` or :class:`CostSegment` and given observations.
    The intermediate variables are observation dependant and therefore the memory consumption of :class:`RLSAlssm`
    scales linearly with the observation vector length.

    Main intermediate variables are the covariance `W`, weighted mean `\\xi`, signal energy `\\kappa`, weighted number
    of samples `\\nu`, see Equation (4.6) in [Wildhaber2019]_
    :download:`PDF <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/357916/thesis-book-final.pdf#page=49>`


    Parameters
    ----------
    cost_model : CostSegment, CompositeCost
        Cost Model
    **kwargs
            Forwarded to :class:`.RLSAlssmBase`


    Examples
    --------
    .. minigallery:: lmlib.RLSAlssm

    """

    def __init__(self, cost_model, **kwargs):
        super().__init__(**kwargs)
        self.cost_model = cost_model

    def _allocate_parameter_storage(self, input_shape):
        K = input_shape[0]

        N = self.cost_model.get_model_order()
        self._W = np.zeros((K, N, N))
        self._xi = np.zeros((K, N))
        self._kappa = np.zeros(K)
        self._nu = np.zeros(K)

    def _forward_recursion(self, A, C, segment, y, v, beta):
        init_vars = forward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)

        if self._backend == 'py':
            forward_recursion_py(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
                                 beta, *init_vars)
        if self._backend == 'jit':
            forward_recursion_jit(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
                                  beta, *init_vars)

    def _backward_recursion(self, A, C, segment, y, v, beta):
        init_vars = backward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)

        if self._backend == 'py':
            backward_recursion_py(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
                                  beta, *init_vars)
        if self._backend == 'jit':
            backward_recursion_jit(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
                                   beta, *init_vars)

    def minimize_v(self, H=None, h=None, return_constrains=False):
        r"""
        Returns the vector `v` of the squared error minimization with linear constraints

        Minimizes the squared error over the vector `v` with linear constraints with an (optional) offset
        [Wildhaber2018]_ [TABLE V].

        **Constraint:**

        - *Linear Scalar* : :math:`x=Hv,\,v\in\mathbb{R}`

          known : :math:`H \in \mathbb{R}^{N \times 1}`

          :math:`\hat{v}_k = \frac{\xi_k^{\mathsf{T}}H}{H^{\mathsf{T}}W_k H}`

        - *Linear Combination With Offset* : :math:`x=Hv +h,\,v\in\mathbb{M}`

          known : :math:`H \in \mathbb{R}^{N \times M},\,h\in\mathbb{R}^N`

          :math:`\hat{v}_k = \big(H^{\mathsf{T}}W_k H\big)^{-1} H^\mathsf{T}\big(\xi_k - W_k h\big)`


        Parameters
        ----------
        constraint_matrix : array_like, shape=(N, M)
            Matrix for linear constraining :math:`H`
        offset_vector : array_like, shape=(N, [S]), optional
            Offset vector for linear constraining :math:`h`
        return_constrains : bool
            If set to True, the output is extened by H and h

        Returns
        -------
        v : :class:`~numpy.ndarray`, shape = (K, M),
            Least square state vector estimate for each time index.
            The shape of one state vector `x[k]` is `(N, [S])`, where k is the time index of `K` samples,
            `N` the ALSSM order.

        |def_K|
        |def_N|

        """

        N = self.cost_model.get_model_order()

        # check and init H
        H = np.eye(N) if H is None else np.asarray(H)
        assert H.shape[0] == N, ""
        M = H.shape[1]

        # check and init h
        h = np.zeros(N) if h is None else np.asarray(h)
        assert h.shape == (N,), ""

        # allocate v and minimize
        v = np.full((len(self.W), M), np.nan)
        minimize_v_py(v, self.W, self.xi, H, h)

        if return_constrains:
            return v, H, h
        return v

    def minimize_x(self, H=None, h=None):
        r"""
        Returns the state vector `x` of the squared error minimization with linear constraints

        Minimizes the squared error over the state vector `x`.
        If needed its possible to apply linear constraints with an (optional) offset.
        [Wildhaber2018]_ [TABLE V].

        **Constraint:**

        - *Linear Scalar* : :math:`x=Hv,\,v\in\mathbb{R}`

          known : :math:`H \in \mathbb{R}^{N \times 1}`

        - *Linear Combination With Offset* : :math:`x=Hv +h,\,v\in\mathbb{M}`

          known : :math:`H \in \mathbb{R}^{N \times M},\,h\in\mathbb{R}^N`

        See also :meth:`minimize_v`

        Parameters
        ----------
        H : array_like, shape=(N, M), optional
            Matrix for linear constraining :math:`H`
        h : array_like, shape=(N, [S]), optional
            Offset vector for linear constraining :math:`h`

        Returns
        -------
        xs : :class:`~numpy.ndarray` of shape = (K, N)
            Least square state vector estimate for each time index.
            The shape of one state vector `x[k]` is `(N,)`, where `k` is the time index of `K` samples,
            `N` the ALSSM order.


        |def_K|
        |def_N|

        """

        if H is None and h is None:
            # allocate x and minimize
            x = np.full_like(self.xi, np.nan)
            minimize_x_py(x, self.W, self.xi)
        else:
            v, H, h = self.minimize_v(H, h, return_constrains=True)
            x = np.einsum('nm, km->kn', H, v) + h

        return x

    def filter_minimize_x(self, y, v=None, H=None, h=None):
        """
        Combination of :meth:`RLSAlssm.filter` and :meth:`RLSAlssm.minimize_x`.

        This method has the same output as calling the methods

        .. code::

            rls.filter(y)
            xs = rls.minimize_x()


        See Also
        --------
        :meth:`RLSAlssm.filter`, :meth:`RLSAlssm.minimize_x`

        """

        self.filter(y, v)
        return self.minimize_x(H, h)

    def eval_errors(self, xs, ks=None):
        r"""
        Evaluation of the squared error for multiple state vectors `xs`.

        The return value is the squared error

        .. math::
            J(x)  = x^{\mathsf{T}}W_kx -2*x^{\mathsf{T}}\xi_k + \kappa_k

        for each state vector :math:`x` from the list `xs`.


        Parameters
        ----------
        xs : array_like of shape=(K, N)
            List of state vectors :math:`x`
        ks : None, array_like of int of shape=(XS,)
            List of indices where to evaluate the error

        Returns
        -------
        J : :class:`np.ndarray` of shape=(XS,)
            Squared Error for each state vector


        |def_K|
        |def_XS|
        |def_N|

        """
        if ks is None:
            return (np.einsum('kn, kn->k', xs, np.einsum('knm, km->kn', self.W, xs))
                    - 2 * np.einsum('kn, kn->k', self.xi, xs)
                    + self.kappa)
        else:
            return (np.einsum('kn, kn->k', xs[ks], np.einsum('knm, km->kn', self.W[ks], xs[ks]))
                    - 2 * np.einsum('kn, kn->k', self.xi[ks], xs[ks])
                    + self.kappa[ks])


class RLSAlssmSet(RLSAlssmBase):
    """
    Filter and Data container for Recursive Least Sqaure Alssm Filters using Sets (multichannel parallel processing)

    This class is the same as :class:`RLSAlssm` except that the signal `y` has an additional last dimension.
    The signals in these dimensions are processed simultaneously, as in a normal :class:`RLSAlssm` called multiple times.

    Parameters
    ----------
    cost_model : CostSegment, CompositeCost
        Cost Model
    kappa_diag : bool
        If set to False, kappa will be computed as a matrix (outer product of each signal energy) else
        its diagonal will saved
    **kwargs
        Forwarded to :class:`.RLSAlssmBase`

    """
    def __init__(self, cost_model, kappa_diag=True):
        super().__init__()
        self._kappa_diag = None
        self.cost_model = cost_model
        self.set_kappa_diag(kappa_diag)

    def set_kappa_diag(self, b):
        assert isinstance(b, bool), 'kappa_diag is not of type bool'
        self._kappa_diag = b

    def _allocate_parameter_storage(self, input_shape):

        K = input_shape[0]
        S = input_shape[-1]
        N = self.cost_model.get_model_order()
        self._W = np.zeros((K, N, N))
        self._nu = np.zeros(K)
        self._xi = np.zeros((K, N, S))
        if self._kappa_diag:
            self._kappa = np.zeros((K, S))
        else:
            self._kappa = np.zeros((K, S, S))

    def _forward_recursion(self, A, C, segment, y, v, beta):
        init_vars = forward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)

        if self._backend == 'py':
            forward_recursion_set_py(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
                                     beta, *init_vars, self._kappa_diag)
        if self._backend == 'jit':
            forward_recursion_set_jit(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
                                     beta, *init_vars, self._kappa_diag)

    def _backward_recursion(self, A, C, segment, y, v, beta):
        init_vars = backward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)

        if self._backend == 'py':
            backward_recursion_set_py(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y,
                                      v, beta, *init_vars, self._kappa_diag)
        if self._backend == 'jit':
            backward_recursion_set_jit(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y,
                                      v, beta, *init_vars, self._kappa_diag)

    def minimize_v(self, H=None, h=None, broadcast_h=True, return_constrains=False):
        r"""
        Returns the vector `v` of the squared error minimization with linear constraints

        Minimizes the squared error over the vector `v` with linear constraints with an (optional) offset
        [Wildhaber2018]_ [TABLE V].

        **Constraint:**

        - *Linear Scalar* : :math:`x=Hv,\,v\in\mathbb{R}`

          known : :math:`H \in \mathbb{R}^{N \times 1}`

          :math:`\hat{v}_k = \frac{\xi_k^{\mathsf{T}}H}{H^{\mathsf{T}}W_k H}`

        - *Linear Combination With Offset* : :math:`x=Hv +h,\,v\in\mathbb{M}`

          known : :math:`H \in \mathbb{R}^{N \times M},\,h\in\mathbb{R}^N`

          :math:`\hat{v}_k = \big(H^{\mathsf{T}}W_k H\big)^{-1} H^\mathsf{T}\big(\xi_k - W_k h\big)`


        Parameters
        ----------
        constraint_matrix : array_like, shape=(N, M)
            Matrix for linear constraining :math:`H`
        offset_vector : array_like, shape=(N, [S]), optional
            Offset vector for linear constraining :math:`h`
        broadcast_h : bool
            if True each channel has the same h vectore else h needs same shape as `x`.
        return_constrains : bool
            If set to True, the output is extened by H and h

        Returns
        -------
        v : :class:`~numpy.ndarray`, shape = (K, M, S),
            Least square state vector estimate for each time index.
            The shape of one state vector `x[k]` is `(N, [S])`, where k is the time index of `K` samples,
            `N` the ALSSM order.

        |def_K|
        |def_S|
        |def_N|

        """
        N = self.cost_model.get_model_order()
        S = np.shape(self.xi)[-1]

        # check and init H
        H = np.eye(N) if H is None else np.asarray(H)
        assert H.shape[0] == N, ""
        M = H.shape[1]

        # check and init h
        if h is None:
            h = np.zeros((N, S))
        else:
            if broadcast_h:
                h = np.repeat(h, S, axis=1)
            else:
                h = np.asarray(h)
        assert h.shape == (N, S), ""

        # allocate v and minimize
        v = np.full((len(self.W), M, S), np.nan)
        minimize_v_py(v, self.W, self.xi, H, h)

        if return_constrains:
            return v, H, h
        return v

    def minimize_x(self, H=None, h=None, broadcast_h=True):
        r"""
        Returns the state vector `x` of the squared error minimization with linear constraints

        Minimizes the squared error over the state vector `x`.
        If needed its possible to apply linear constraints with an (optional) offset.
        [Wildhaber2018]_ [TABLE V].

        **Constraint:**

        - *Linear Scalar* : :math:`x=Hv,\,v\in\mathbb{R}`

          known : :math:`H \in \mathbb{R}^{N \times 1}`

        - *Linear Combination With Offset* : :math:`x=Hv +h,\,v\in\mathbb{M}`

          known : :math:`H \in \mathbb{R}^{N \times M},\,h\in\mathbb{R}^N`

        See also :meth:`minimize_v`

        Parameters
        ----------
        H : array_like, shape=(N, M), optional
            Matrix for linear constraining :math:`H`
        h : array_like, shape=(N, [S]), optional
            Offset vector for linear constraining :math:`h`
        broadcast_h : bool
            if True each channel has the same h vectore else h needs same shape as `x`.

        Returns
        -------
        xs : :class:`~numpy.ndarray` of shape = (K, N, S)
            Least square state vector estimate for each time index.
            The shape of one state vector `x[k]` is `(N, S)`, where `k` is the time index of `K` samples,
            `N` the ALSSM order.


        |def_K|
        |def_S|
        |def_N|

        """
        if H is None and h is None:
            # allocate x and minimize
            x = np.full_like(self.xi, np.nan)
            minimize_x_py(x, self.W, self.xi)
        else:
            v, H, h = self.minimize_v(H, h, broadcast_h, return_constrains=True)
            x = np.einsum('nm, kms->kns', H, v) + h

        return x

    def filter_minimize_x(self, y, v=None, H=None, h=None, **kwargs):
        """
        Combination of :meth:`RLSAlssmSet.filter` and :meth:`RLSAlssmSet.minimize_x`.

        This method has the same output as calling the methods

        .. code::

            rls.filter(y)
            xs = rls.minimize_x()


        See Also
        --------
        :meth:`RLSAlssmSet.filter`, :meth:`RLSAlssmSet.minimize_x`

        """
        self.filter(y, v)
        return self.minimize_x(H, h, **kwargs)

    def eval_errors(self, xs, ks=None):
        r"""
        Evaluation of the squared error for multiple state vectors `xs`.

        The return value is the squared error

        .. math::
            J(x)  = x^{\mathsf{T}}W_kx -2*x^{\mathsf{T}}\xi_k + \kappa_k

        for each state vector :math:`x` from the list `xs`.


        Parameters
        ----------
        xs : array_like of shape=(K, N, S)
            List of state vectors :math:`x`
        ks : None, array_like of int of shape=(XS,)
            List of indices where to evaluate the error

        Returns
        -------
        J : :class:`np.ndarray` of shape=(XS, S [,S])
            Squared Error for each state vector


        |def_K|
        |def_XS|
        |def_N|

        """
        if ks is None:
            if self._kappa_diag:
                return (np.einsum('kns, kns->ks', xs, np.einsum('knm, kmt->knt', self.W, xs))
                        - 2 * np.einsum('kns, kns->ks', self.xi, xs)
                        + self.kappa)
            else:
                return (np.einsum('kns, knt->kst', xs, np.einsum('knm, kmt->knt', self.W, xs))
                        - 2 * np.einsum('kns, knt->kst', self.xi, xs)
                        + self.kappa)
        else:
            if self._kappa_diag:
                return (np.einsum('kns, kns->ks', xs[ks], np.einsum('knm, kmt->knt', self.W[ks], xs[ks]))
                        - 2 * np.einsum('kns, kns->ks', self.xi[ks], xs[ks])
                        + self.kappa[ks])
            else:
                return (np.einsum('kns, knt->kst', xs[ks], np.einsum('knm, kmt->knt', self.W[ks], xs[ks]))
                        - 2 * np.einsum('kns, knt->kst', self.xi[ks], xs[ks])
                        + self.kappa[ks])


class RLSAlssmSteadyState(RLSAlssmBase):
    """
    Filter and Data container for Recursive Least Sqaure Alssm Filters in Steady State Mode

    With :class:`RLSAlssmSteadyState` a common :math:`W_k = W_{steady}` is used for all samples (faster computation).
    Note that using a common :math:`W_k` potentially leads to border missmatch effects and to completely invalid results
    when samples have individual sample weights.

    See Also
    --------
    :class:`RLSAlssm`
    
    """
    def __init__(self, cost_model, steady_state_method='closed_form', **kwargs):
        super().__init__(**kwargs)
        self.cost_model = cost_model
        self._W = self.cost_model.get_steady_state_W(method=steady_state_method)

    def _allocate_parameter_storage(self, input_shape):
        K = input_shape[0]

        N = self.cost_model.get_model_order()
        self._xi = np.zeros((K, N))
        self._kappa = np.zeros(K)
        self._nu = np.zeros(K)

    def _forward_recursion(self, A, C, segment, y, v, beta):
        init_vars = forward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)

        if self._backend == 'py':
            forward_recursion_xi_kappa_nu_py(self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
                                 beta, *init_vars)
        if self._backend == 'jit':
            forward_recursion_jit(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
                                  beta, *init_vars)

    def _backward_recursion(self, A, C, segment, y, v, beta):
        init_vars = backward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)

        if self._backend == 'py':
            backward_recursion_xi_kappa_nu_py(self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
                                  beta, *init_vars)
        if self._backend == 'jit':
            backward_recursion_jit(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
                                   beta, *init_vars)

    def minimize_v(self, H=None, h=None, return_constrains=False):
        """
        Returns the state vector `v` of the squared error minimization with linear constraints

        See Also
        --------
        :class:`RLSAlssm.minimize_v`

        """
        N = self.cost_model.get_model_order()

        # check and init H
        H = np.eye(N) if H is None else np.asarray(H)
        assert H.shape[0] == N, ""
        M = H.shape[1]

        # check and init h
        h = np.zeros(N) if h is None else np.asarray(h)
        assert h.shape == (N,), ""

        # allocate v and minimize
        v = np.full((len(self.xi), M), np.nan)
        minimize_v_steady_state_py(v, self.W, self.xi, H, h)

        if return_constrains:
            return v, H, h
        return v

    def minimize_x(self, H=None, h=None):
        """
        Returns the state vector `x` of the squared error minimization with linear constraints

        See Also
        --------
        :class:`RLSAlssm.minimize_x`

        """
        if H is None and h is None:
            # allocate x and minimize
            x = np.full_like(self.xi, np.nan)
            minimize_x_steady_state_py(x, self.W, self.xi)
        else:
            v, H, h = self.minimize_v(H, h, return_constrains=True)
            x = np.einsum('nm, km->kn', H, v) + h

        return x

    def filter_minimize_x(self, y, v=None, H=None, h=None):
        """
        Combination of :meth:`RLSAlssmSteadyState.filter` and :meth:`RLSAlssmSteadyState.minimize_x`.

        See Also
        --------
        :meth:`RLSAlssmSteadyState.filter`, :meth:`RLSAlssmSteadyState.minimize_x`

        """
        self.filter(y, v)
        return self.minimize_x(H, h)

    def eval_errors(self, xs, ks=None):
        r"""
        Evaluation of the squared error for multiple state vectors `xs`.

        See Also
        --------
        :class:`RLSAlssm.eval_error`

        """
        if ks is None:
            return (np.einsum('kn, kn->k', xs, np.einsum('nm, km->kn', self.W, xs))
                    - 2 * np.einsum('kn, kn->k', self.xi, xs)
                    + self.kappa)
        else:
            return (np.einsum('kn, kn->k', xs[ks], np.einsum('nm, km->kn', self.W, xs[ks]))
                    - 2 * np.einsum('kn, kn->k', self.xi[ks], xs[ks])
                    + self.kappa[ks])


class RLSAlssmSetSteadyState(RLSAlssmBase):
    """
    Filter and Data container for Recursive Least Sqaure Alssm Filters using Sets in Steady State Mode

    With :class:`RLSAlssmSteadyState` a common :math:`W_k = W_{steady}` is used for all samples (faster computation).
    Note that using a common :math:`W_k` potentially leads to border missmatch effects and to completely invalid results
    when samples have individual sample weights.

    See Also
    --------
    :class:`RLSAlssmSet`

    """
    def __init__(self, cost_model, steady_state_method='closed_form', kappa_diag=True, **kwargs):
        super().__init__(**kwargs)
        self._kappa_diag = None
        self.cost_model = cost_model
        self.set_kappa_diag(kappa_diag)
        self._W = self.cost_model.get_steady_state_W(method=steady_state_method)

    def set_kappa_diag(self, b):
        assert isinstance(b, bool), 'kappa_diag is not of type bool'
        self._kappa_diag = b

    def _allocate_parameter_storage(self, input_shape):

        K = input_shape[0]
        S = input_shape[-1]
        N = self.cost_model.get_model_order()
        self._nu = np.zeros(K)
        self._xi = np.zeros((K, N, S))
        if self._kappa_diag:
            self._kappa = np.zeros((K, S))
        else:
            self._kappa = np.zeros((K, S, S))

    def _forward_recursion(self, A, C, segment, y, v, beta):
        init_vars = forward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)

        if self._backend == 'py':
            forward_recursion_set_xi_kappa_nu_py(self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
                                     beta, *init_vars, self._kappa_diag)
        if self._backend == 'jit':
            forward_recursion_set_jit(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y, v,
                                     beta, *init_vars, self._kappa_diag)

    def _backward_recursion(self, A, C, segment, y, v, beta):
        init_vars = backward_initialize(A, C, segment.gamma, segment.a, segment.b, segment.delta)

        if self._backend == 'py':
            backward_recursion_set_xi_kappa_nu_py(self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y,
                                      v, beta, *init_vars, self._kappa_diag)
        if self._backend == 'jit':
            backward_recursion_set_jit(self._W, self._xi, self._kappa, self._nu, segment.a, segment.b, segment.delta, y,
                                      v, beta, *init_vars, self._kappa_diag)

    def minimize_v(self, H=None, h=None, broadcast_h=True, return_constrains=False):
        """
        Returns the state vector `v` of the squared error minimization with linear constraints

        See Also
        --------
        :class:`RLSAlssmSet.minimize_v`

        """
        N = self.cost_model.get_model_order()
        S = np.shape(self.xi)[-1]

        # check and init H
        H = np.eye(N) if H is None else np.asarray(H)
        assert H.shape[0] == N, ""
        M = H.shape[1]

        # check and init h
        if h is None:
            h = np.zeros((N, S))
        else:
            if broadcast_h:
                h = np.repeat(h, S, axis=1)
            else:
                h = np.asarray(h)
        assert h.shape == (N, S), ""

        # allocate v and minimize
        v = np.full((len(self.xi), M, S), np.nan)
        minimize_v_steady_state_py(v, self.W, self.xi, H, h)

        if return_constrains:
            return v, H, h
        return v

    def minimize_x(self, H=None, h=None, broadcast_h=True):
        """
        Returns the state vector `x` of the squared error minimization with linear constraints

        See Also
        --------
        :class:`RLSAlssmSet.minimize_x`

        """
        if H is None and h is None:
            # allocate x and minimize
            x = np.full_like(self.xi, np.nan)
            minimize_x_steady_state_py(x, self.W, self.xi)
        else:
            v, H, h = self.minimize_v(H, h, broadcast_h, return_constrains=True)
            x = np.einsum('nm, kms->kns', H, v) + h

        return x

    def filter_minimize_x(self, y, v=None, H=None, h=None, **kwargs):
        """
        Combination of :meth:`RLSAlssmSetSteadyState.filter` and :meth:`RLSAlssmSetSteadyState.minimize_x`.

        See Also
        --------
        :meth:`RLSAlssmSetSteadyState.filter`, :meth:`RLSAlssmSetSteadyState.minimize_x`

        """
        self.filter(y, v)
        return self.minimize_x(H, h, **kwargs)

    def eval_errors(self, xs, ks=None):
        r"""
        Evaluation of the squared error for multiple state vectors `xs`.

        See Also
        --------
        :class:`RLSAlssm.eval_error`

        """
        if ks is None:
            if self._kappa_diag:
                return (np.einsum('kns, kns->ks', xs, np.einsum('nm, kmt->knt', self.W, xs))
                        - 2 * np.einsum('kns, kns->ks', self.xi, xs)
                        + self.kappa)
            else:
                return (np.einsum('kns, knt->kst', xs, np.einsum('nm, kmt->knt', self.W, xs))
                        - 2 * np.einsum('kns, knt->kst', self.xi, xs)
                        + self.kappa)
        else:
            if self._kappa_diag:
                return (np.einsum('kns, kns->ks', xs[ks], np.einsum('nm, kmt->knt', self.W, xs[ks]))
                        - 2 * np.einsum('kns, kns->ks', self.xi[ks], xs[ks])
                        + self.kappa[ks])
            else:
                return (np.einsum('kns, knt->kst', xs[ks], np.einsum('nm, kmt->knt', self.W, xs[ks]))
                        - 2 * np.einsum('kns, knt->kst', self.xi[ks], xs[ks])
                        + self.kappa[ks])


class ConstrainMatrix:
    """
    Constrain Matrix Generator

    Builder class to set up matrix `H` as a linear constraint for the squared error minimization

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
        self._N = self._cost.get_model_order()
        self._data = np.eye(self._N)

    def constrain(self, indices, value):
        """
        Add constrain

        I.e. Apply a dependency between two indices.
        See example below.

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
        Add constrain by labels

        I.e. Apply a dependency between two indices.
        See example below.

        Parameters
        ----------
        label_1: str
            label of first state variable index to apply a dependency
        label_2: str
            label of second state variable index to apply a dependency
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

    def constrain_TSLM_type(self, TSLM_type):
        """
        Sets the type for a Two Segment Line Model (TSLM)

        Parameters
        ----------
        TSLM_type: str
            type of two segment line model constrain, options:
            'free', continuous', 'straight','horizontal', 'left horizontal', 'right horizontal', 'peak', 'step'

        Returns
        -------
        cm : ConstrainMatrix
            self
        """
        assert len(self._cost.segments) == 2, 'Only available for two segment models'
        assert len(self._cost.alssms) == 2, 'Only available for two line models'
        assert TSLM_type in self.TSLM_TYPES, f'Unknown TSLM type {TSLM_type}'
        assert [a.N for a in self._cost.alssms] == [2, 2], 'Faulty model orders. Each line model needs N=2.'

        TSLM_type = TSLM_type.lower()
        if TSLM_type == self.TSLM_TYPES[0]:  # 'free'
            pass
        if TSLM_type == self.TSLM_TYPES[1]:  # 'continuous'
            self.constrain((0, 2), 1)
        if TSLM_type == self.TSLM_TYPES[2]:  # 'straight'
            self.constrain((0, 2), 1)
            self.constrain((1, 3), 1)
        if TSLM_type == self.TSLM_TYPES[3]:  # 'horizontal'
            self.constrain((0, 2), 1)
            self.constrain((1, 1), 0)
            self.constrain((3, 3), 0)
        if TSLM_type == self.TSLM_TYPES[4]:  # 'left horizontal'
            self.constrain((0, 2), 1)
            self.constrain((1, 1), 0)
        if TSLM_type == self.TSLM_TYPES[5]:  # 'right horizontal'
            self.constrain((0, 2), 1)
            self.constrain((3, 3), 0)
        if TSLM_type == self.TSLM_TYPES[6]:  # 'peak'
            self.constrain((0, 2), 1)
            self.constrain((1, 3), -1)
        if TSLM_type == self.TSLM_TYPES[7]:  # 'step'
            self.constrain((1, 1), 0)
            self.constrain((3, 3), 0)

        return self

    def digest(self):
        """
        Reruns a "snapshot" of the constraint matrix with the applied constrains

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

        # check if all combination of columuns in H are a mulitple of eachother when yes add last one to del_cols
        for j, c1 in enumerate(H.T):
            c1_norm = np.linalg.norm(c1)
            if c1_norm != 0.0:
                for i, c2 in enumerate(H.T):
                    c2_norm = np.linalg.norm(c2)
                    if c2_norm != 0.0:
                        is_multiple = np.all(c1/c1_norm -c2/c2_norm == 0.0)
                        is_multiple  = is_multiple or np.all(-c1/np.linalg.norm(-c1) -c2/c2_norm == 0.0)
                        if is_multiple and i>j:
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

