"""
Definition of recursively computed squared error cost functions (such as *Cost Segments* and *Composite Costs*),
all based on ALSSMs
"""


from numpy.typing import ArrayLike
from typing import Union
from collections.abc import Iterable
import warnings
import copy

import numpy as np

from lmlib.statespace.model import ModelBase, AlssmSum
from lmlib.utils.check import *
from lmlib.statespace.backends.statespace_tools import *


__all__ = ['ConstrainMatrix', 'CompositeCost', 'CostSegment', 'Segment', 'NDCompositeCost',
           'FW', 'FORWARD', 'BW', 'BACKWARD',
           'map_trajectories', 'map_windows'
           ]

BACKWARD = 'bw'
"""str : Sets the recursion direction in a :class:`Segment` to backward, use :const:`BACKWARD` or :const:`BW`"""
BW = BACKWARD
"""str : Sets the recursion direction in a :class:`Segment` to backward, use :const:`BACKWARD` or :const:`BW`"""
FORWARD = 'fw'
"""str : Sets the recursion direction in a :class:`Segment` to forward, use :const:`FORWARD` or :const:`FW`"""
FW = FORWARD
"""str : Sets the recursion direction in a :class:`Segment` to forward, use :const:`FORWARD` or :const:`FW`"""


def map_windows(windows: ArrayLike, ks, K: int, merge_ks:bool=False, merge_seg:bool=False, fill_value:Union[int, float]=0):
    """
    Maps the window amplitudes of one or multiple :class:`Segment` to indices `ks` into a common target output vector of
     length `K`.

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
    fill_value : scalar, optional
        Value of target output vector where no window is assigned, defaults to 0.

    Returns
    -------
    output : :class:`~numpy.ndarray` of shape=([KS,] [P,] K) of floats
        * Dimension `KS` only exists, if :attr:`merge_ks`:code:`=False`.
        * Dimension `P` only exists, if :attr:`merge_seg`:code:`=False`.



    |def_KS|
    |def_P|
    |def_K|

    """

    # return an empty array if ks is empty
    if len(ks) == 0:
        return np.array([])

    w = np.full((len(ks), len(windows), K), fill_value, dtype=float)
    for p, (ab_range, weights) in enumerate(windows):
        ab_range = np.arange(ab_range.start, ab_range.stop)

        for i, k in enumerate(ks):
            boundary_mask = np.bitwise_and(0 <= ab_range + k, ab_range + k < K)
            w[i, p, ab_range[boundary_mask] + k] = weights[boundary_mask]

    return _merge_ks_seg(w, merge_ks, merge_seg)


def map_trajectories(trajectories, ks, K:int, merge_ks:bool=False, merge_seg:bool=False, fill_value:Union[None, int, float]=np.nan):
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
    fill_value : None, scalar, optional
        Value of target output vector where no trajectory is assigned. Defaults is None that assigns `np.nan`.

    Returns
    -------
    output : :class:`~numpy.ndarray` `of shape=shape of ([XS,] [P,] K, L [,S]) of floats`

        * Dimension `XS` only exists, if :attr:`merge_ks`:code:`=False`.
        * Dimension `P` only exists, if :attr:`merge_seg`:code:`=False`
        * Dimension `S` only exists, if the parameter :attr:`xs` of :meth:`CompositeCost.trajectories` or
          :meth:`CompositeCost.trajectories` also provides dimension `S` (i.e., provides multiple signal set processing).


    |def_XS|
    |def_P|
    |def_K|
    |def_L|
    |def_S|

    """

    # return an empty array if ks is empty
    if len(ks) == 0:
        return np.array([])

    # TODO: change for ndRLSAlssm

    t_ = trajectories[0][0][1]
    P = len(trajectories[0])
    XS = len(trajectories)
    assert XS == len(ks), "number of trajectories and ks does not match up"

    if t_.ndim == 2:  # check if the trajectory is from a RLSAlssmSet
        S = t_.shape[1]
        t = np.full((XS, P, K, S), fill_value)
    else:
        t = np.full((XS, P, K), fill_value)

    for i, k in enumerate(ks):
        for p, (j_range, trajectory) in enumerate(trajectories[i]):
            j_list = np.array(j_range)
            mask = np.bitwise_and(k + j_list >= 0, k + j_list < K)
            t[i, p, k + j_list[mask], ...] = trajectory[mask]

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
    g : int, float, None
        :math:`g > 0`. Effective number of samples under the window. This is used as a (more readable) surrogate for the
        window decay of exponential windows, see [Wildhaber2018]_. |br|
        :math:`g` is counted to the left or right of :math:`k+ \delta`, for a forward or backward computation
        direction, respectively.
    direction : str
        Computation direction of recursive computations (also select a left- or right-side decaying window) |br|
        :data:`statespace.FORWARD` or `'fw'` use forward computation with forward recursions |br|
        :data:`statespace.BACKWARD` or `'bw'` use backward computation with backward recursions
    delta : int, optional
        Exponential window is normalized to 1 at relative index :code:`delta`.
    gamma : float, int
        Window Constant Decay, (Alternative for `g`. If gamma is set `g` has to be None)
        `gamma` is to choose dependent of the direction (forward, backward). Forward directions with `gamma <= 1` will
        raise a warning for possible instability, backwards directions with `gamma >= 1`.
        See [Wildhaber2018]_ Table IV
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

    """

    def __init__(self, a:Union[int, float], b:Union[int, float], direction:str, g:int, delta:int=0, label:str=None, gamma:Union[int, float]=None):
        self._a = None
        self._b = None
        self.set_boundaries(a, b)
        self.direction = direction
        if gamma is not None:
            assert g is None, "g is not None. If gamma is set, g has to be None."
            self._g = None
            self.gamma = gamma
            if self.direction == FW and self.gamma <= 1:
                warnings.warn('gamma <= 1 in forward direction can result into instability')
            if self.direction == BW and self.gamma >= 1:
                warnings.warn('gamma >= 1 in backward direction can result into instability')
        else:
            self.g = g
            if self.direction == FW:
                self.gamma = self.g / (self.g - 1)
            else:
                self.gamma = (self.g - 1) / self.g
        self.delta = delta
        self.label = 'n/a' if label is None else label

    def __str__(self):
        return f'{type(self).__name__}(' \
               f'a={self.a}, ' \
               f'b={self.b}, ' \
               f'direction={self.direction}, ' \
               f'g={self.g}, ' \
               f'delta={self.delta}, ' \
               f'label={self.label})'

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
        """int, float, None : Effective number of samples :math:`g`, setting the window with

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

            * :code:`range` of length `JR`: relative index range of window with respect to segment's boundaries.
            * :code:`array` of shape=(JR) of floats: per-index window weight over the reported index range


        |def_JR|

        """

        return _window_output(self.a, self.b, self.direction, self.gamma, self.delta, thd=thd)


class CompositeCost:
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


    This figure shows a graphical representation of a composite cost, depicting the internal relationships between
    Segments, ALSSMs, and the mapping matrix F. F[m,p] is implemented as a scalar factor multiplied on the alssm's
    output signal.


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

    def __init__(self, alssms, segments, F, label:str='n/a'):
        self.alssms = alssms
        self.segments = segments
        self.F = F
        self.label = label

    def __str__(self):
        return f'CompositeCost(label={self.label}) \n' \
               f'  └- {[alssm.__str__() for alssm in self.alssms]}, \n' \
               f'  └- {[segment.__str__() for segment in self.segments]} '

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
        return AlssmSum(self.alssms).N

    def get_model_output_dimension(self):
        """tuple : Alssm Output Dimension"""
        alssm = AlssmSum(self.alssms)
        return alssm.C.shape[0]  if alssm.is_MC else ()

    def trajectories(self, xs, F=None, thd:float=1e-6):
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
            Mapping matrix. If not set to :code:`None`, the given matrix overloads the CompositeCost's internal mapping
            matrix as provided by the class constructor for :class:`CompositeCost`.
            If :attr:`F` is only of :code:`shape(M,)`, the vector gets enlarged to size :code:`shape(M, P)` by repeating
            the vector `P`-times. (This is a shortcut to select a single ALSSM over all segments.)
        thd : list of shape(P) of floats, scalar, None, optional
            Per segment threshold limiting the evaluation boundaries by setting a minimum window height of the
            associated :attr:`~CompositeCost.segments`
            Scalar to use the same threshold for all available segments.

        Returns
        -------
        trajectories : list of shape=(XS) of tuples of shape=(P) of tuples :code:`(range, array)`
            Each element in `trajectories` is a tuple with

            * :code:`range` `of length JR`: relative index range of trajectory with respect to segment's boundaries
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
        else:
            M = self.alssms.__len__()
            P = self.segments.__len__()
            assert (M, P) == np.shape(F), f'F is not array_like of shape (M={M}, P={P})'
            F = np.asarray(F)

        trajectories = []
        for x in xs:
            _tmp = []
            for p, seg in enumerate(self.segments):
                alssm = AlssmSum(self.alssms, deltas=F.T[p])
                ab_range = _window_range(seg.a, seg.b, seg.direction, seg.gamma, seg.delta, thd)
                _tmp.append((ab_range, alssm.trajectory(x, ab_range)))
            trajectories.append(_tmp)
        return trajectories

    def windows(self, segment_indices:list[int], thd:float=1e-6):
        """
        Returns for each selected segment its window generated by :meth:`CostSegment.windows`.

        The segments are selected by :attr:`segment_selection`.


        Parameters
        ----------
        segment_indices : array_like, shape=(P,) of Boolean
            Enables (:code:`True`, 1) or disables (:code:`False`, 0) the evaluation of the
            p-th Segment in :attr:`CompositeCost.segments` using :meth:`CostSegment.window`
        thd : list of shape=(P,) of floats, scalar, optional
            List of per-segment threshold values or scalar to use the same threshold for all segments
            Evaluation is stopped for window weights below the given threshold.
            Set list element to :code:`None` to disable the threshold for a segment.

        Returns
        -------
        `list of shape=(P) of tuples` :code:`(range, array)`
            Each element is a tuple with

            * :code:`range` of length `JR`: relative index range of window with respect to segment's boundaries.
            * :code:`array` of shape=(JR) of floats: per-index window weight over the reported index range


        |def_JR|
        |def_P|


        """

        assert is_array_like(segment_indices), 'segment_index is not an integer nor array_like'
        assert 0 <= min(segment_indices) and max(segment_indices) < len(self.segments), 'segment_index out of range'

        windows = []
        for segment_index in segment_indices:
            seg = self.segments[segment_index]
            window = _window_output(seg.a, seg.b, seg.direction, seg.gamma, seg.delta, thd)
            windows.append(window)
        return windows

    def get_steady_state_W(self, method:str='closed_form'):
        """
        Returns Steady State Matrix W

        Parameters
        ----------
        method : str, optional
            If 'closed_form' is used the steady state matrix will be calculated in a close form.
            This method can be critical, as it can produce badly conditioned matrices internally.
            If 'limited_sum' is used, the steady state matrix will be calculated brute force, with a stop condition
            on a minimum change.

        Returns
        -------
        Wss = `class:numpy.ndarray`
            Steady State Matrix W
        """
        N = self.get_model_order()
        W = np.zeros((N, N))
        for p, segment in enumerate(self.segments):
            alssm = AlssmSum(self.alssms, self.F[:, p])
            if method == 'closed_form':
                W += _covariance_matrix_closed_form(alssm.A, alssm.C, segment.gamma, segment.a, segment.b, segment.delta)
            if method == 'limited_sum':
                W += _covariance_matrix_limited_sum(alssm.A, alssm.C, segment.gamma, segment.a, segment.b, segment.delta)
        return W

    def eval_alssm_output(self, xs, alssm_weights:list[Union[int, float]]=None, c0s=None):
        """
        Evaluation of the ALSSM for multiple state vectors `xs`.

        **See:** :meth:`~lmlib.statespace.models.Alssm.eval`

        Parameters
        ----------
        xs : array_like of shape=(XS, N [,S]) of floats
            List of state vectors :math:`x`
        alssm_weights : None, scalar, array_like, shape=(M,) of floats, optional
            Each element sets the weight of the output of the m-th ALSSM in :attr:`CompositeCost.alssms`.
            If alssm_weights is a scalar it set for each alssms the same weight.
            If None no weights are set.
        c0s : None, scalar, array_like, tuple, optional
            - None type is default and doesn't change the model output matrix
            - Scalar type will result in a 1 dimensional output matrix filled with the scalar value.
            - Array_like type expects c0s shape of ([L,] N,) and replaces the composite output matrix.
            - Tuple type c0s have an entry for each sub ALSSM in the composite model. Possible tuple entries are `None`, `scalar`, `array_like`, which are behaving like to composite c0s entries above. See Notes for more detail.


        Returns
        -------
        s : :class:`~numpy.ndarray` of shape=(XS,[J,]L[,S]) of floats
            ALSSM outputs


        |def_N|
        |def_L|
        |def_M|
        |def_S|
        |def_XS|

        Notes
        -----
        C0s of type tuple can have different element types `None`, `scalar`, `array_like`.

        - Entries are array_like and for each ALSSM resp. i.e. :code:`c0s = ([1, 2], [1, 0])` will result in the first :code:`alssms[0].C = [1, 2]` and in the second :code:`alssms[1].C = [1, 0]`.

        - Entries are scalars or `None` and for each ALSSM resp. i.e. :code:`c0s = ([1, 2], 3)`  will result in the first :code:`alssms[0].C = [1, 2]` and in the second :code:`alssms[1].C = [3, 3]`.

        - A None element won't change the ALSSM Output Matrix C. i.e. :code:`c0s = (None, 3)`  will leave alone first ALSSM.C and the second changes to :code:`alssms[1].C = [3, 3]`.

        """

        N = AlssmSum(self.alssms).N
        input_type = ''

        if c0s is None:
            input_type = 'none'
        if np.isscalar(c0s):
            input_type = 'scalar'
        if is_array_like(c0s):
            input_type = 'output_matrix'
        if isinstance(c0s, tuple):
            input_type = 'mixed'

        assert input_type != '', "c0s input unknown type"

        alssm_sum = AlssmSum(self.alssms, alssm_weights)

        if input_type == 'scalar':
            alssm_sum.C = np.full(N, fill_value=c0s)
        if input_type == 'output_matrix':
            alssm_sum.C = c0s
        if input_type == 'mixed':
            alssms = []
            for alssm, c0 in zip(self.alssms, c0s):
                alssm_ = copy.deepcopy(alssm)
                if c0 is None:
                    c0 = alssm_.C
                if np.isscalar(c0):
                    c0 = np.full_like(alssm_.C, c0)
                alssm_.C_init = c0
                alssms.append(alssm_)
            alssm_sum = AlssmSum(alssms, alssm_weights)

        return alssm_sum.eval_states(xs)

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
        return AlssmSum(self.alssms, label='cost').get_state_var_indices('cost.' + label)


class CostSegment(CompositeCost):
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
    alssm : ModelBase
        ALSSM, defining the signal model
    segment : Segment
        Segment, defining the window
    **kwargs
        Forwarded to :class:`CompositeCost`

    Examples
    --------
    Set up a cost segment with finite boundaries and a line ALSSM.

    >>> alssm = lm.AlssmPoly(poly_degree=1, label="slope with offset")
    >>> segment = lm.Segment(a=-30, b=0, direction=lm.FORWARD, g=20, label="finite left")
    >>> cost = lm.CostSegment(alssm, segment, label="left line")
    >>> print(cost)
    CostSegment : label: left line
      └- Alssm : polynomial, A: (2, 2), C: (1, 2), label: slope with offset,
      └- Segment : a:-30, b:0, fw, g:20, delta:0, label: finite left

    """

    def __init__(self, alssm: ModelBase, segment:Segment, **kwargs):
        super().__init__(alssms=[alssm], segments=[segment], F=np.ones((1, 1)), **kwargs)

    def __str__(self):
        return f'{type(self).__name__}(label: {self.label}) \n  └- {self.alssms[0]}, \n  └- {self.segments[0]}'


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

        # check if all combination of columns in H are a mulitple of each-other when yes add last one to del_cols
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



class NDCompositeCost(list):
    def __init__(self, composite_costs):
        super().__init__(composite_costs)

        # check alssm output dimensions L
        Ls = np.array([cc.get_model_output_dimension() for cc in composite_costs])
        assert np.all(Ls == Ls[0]), 'Output Dimension of composite costs do not match'


    @property
    def ND(self):
        return len(self)

    def get_model_order(self):
        """int : Order of the (stacked, dimension-combined) Alssm Model"""
        return int(np.prod([cc.get_model_order() for cc in self]))

    def get_steady_state_W(self, dim_order):
        W = [1]
        for n in dim_order:
            W = np.kron(W, self[n].get_steady_state_W())
        return W

    def get_model_output_dimension(self):
        return self[0].get_model_output_dimension()
