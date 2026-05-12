from typing import Union
import warnings
import numpy as np

__all__ = ['FW', 'FORWARD', 'BW', 'BACKWARD', 'Segment']

BACKWARD = 'bw'
"""str : Sets the recursion direction in a :class:`Segment` to backward, use :const:`BACKWARD` or :const:`BW`"""
BW = BACKWARD
"""str : Sets the recursion direction in a :class:`Segment` to backward, use :const:`BACKWARD` or :const:`BW`"""
FORWARD = 'fw'
"""str : Sets the recursion direction in a :class:`Segment` to forward, use :const:`FORWARD` or :const:`FW`"""
FW = FORWARD
"""str : Sets the recursion direction in a :class:`Segment` to forward, use :const:`FORWARD` or :const:`FW`"""


class Segment:
    r"""
    The Segment represents a window of finite or infinite interval borders used to select and weight signal samples in a cost function.

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
    The interval of the semgment includes both boundaries `a` and `b` in the calculations.
    i.e., if the sum runs over the interval :math:`k \in [a,b] ` it treats `b-a+1` samples.

    Examples
    --------
    >>> import lmlib as lm
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
        r"""float : Window decay factor \gamma

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
        assert a < b, 'left boundary \'a\' must be smaller than the right boundary \'b\'.'
        self._a = a
        self._b = b

    def _ab_range(self, thd=1e-6):

        if self.direction is FW:
            if self.gamma > 1:
                a_lim = max(np.log(thd) / np.log(self.gamma) - 1 + self.delta, self.a)
            else:
                a_lim = max(np.log(thd) / np.log(1 / self.gamma) - 1 + self.delta, self.a)
            return np.arange(int(a_lim), self.b + 1)

        if self.direction is BW:
            if self.gamma < 1:
                b_lim = min(np.log(thd) / np.log(self.gamma) + 1 + self.delta, self.b)
            else:
                b_lim = min(np.log(thd) / np.log(1 / self.gamma) + 1 + self.delta, self.b)
            return np.arange(self.a, int(b_lim) + 1)

        return None

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

        ab_range = self._ab_range(thd=thd)
        return ab_range, self.gamma ** (np.array(ab_range) - self.delta)

