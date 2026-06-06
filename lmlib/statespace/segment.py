from typing import Union
import warnings
import numpy as np

__all__ = ['FW', 'FORWARD', 'BW', 'BACKWARD', 'Segment']

BACKWARD = 'bw'
r"""str : Sets the recursion direction in a [`Segment`][lmlib.statespace.segment.Segment] to backward, use [`BACKWARD`][lmlib.statespace.segment.BACKWARD] or [`BW`][lmlib.statespace.segment.BW]"""
BW = BACKWARD
r"""str : Sets the recursion direction in a [`Segment`][lmlib.statespace.segment.Segment] to backward, use [`BACKWARD`][lmlib.statespace.segment.BACKWARD] or [`BW`][lmlib.statespace.segment.BW]"""
FORWARD = 'fw'
r"""str : Sets the recursion direction in a [`Segment`][lmlib.statespace.segment.Segment] to forward, use [`FORWARD`][lmlib.statespace.segment.FORWARD] or [`FW`][lmlib.statespace.segment.FW]"""
FW = FORWARD
r"""str : Sets the recursion direction in a [`Segment`][lmlib.statespace.segment.Segment] to forward, use [`FORWARD`][lmlib.statespace.segment.FORWARD] or [`FW`][lmlib.statespace.segment.FW]"""


class Segment:
    r"""
    Segment defining a window interval for weighting signal samples in a cost function.

    Segments are commonly used in combination with ALSSM signal models to select and
    weight the samples in cost functions; see [`CostSegment`][lmlib.statespace.cost.CostSegment] or
    [`CompositeCost`][lmlib.statespace.cost.CompositeCost]. The window has an exponentially decaying shape controlled
    by the decay factor $\gamma$. The direction of the window also determines
    whether the recursive cost computation runs forward or backward.

    In conjunction with an ALSSM, a Segment leads to a cost function of the form

    $$
    J_k(x) = \sum_{i=k+a}^{k+b} \gamma^{i-k-\delta}\big(CA^{i-k}x - y_i\big)^2 \ ,
    $$

    and when additionally using sample weights $v_k$, of the form

    $$
    J_k(x) = \sum_{i=k+a}^{k+b} v_k  {\alpha}_{k+\delta}(k+\delta) \big(CA^{i-k}x - y_i\big)^2 \ ,
    $$

    with the sample weights $v_k$
    and the window weight $\alpha_k(j)$ which depends on the sample weights, see Equation (14) in [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018)

    See also [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018) [\[Wildhaber2019\]](../../bibliography.md#wildhaber2019)

    Parameters
    ----------
    a : int or -np.inf
        Left boundary of the segment's interval (inclusive).
    b : int or np.inf
        Right boundary of the segment's interval (inclusive).
    direction : str
        Recursion direction for the cost computation; also selects whether the
        window decays toward the left or the right. <br>
        [`FORWARD`][lmlib.statespace.FORWARD] or ``'fw'``: forward computation (window decays to the left). <br>
        [`BACKWARD`][lmlib.statespace.BACKWARD] or ``'bw'``: backward computation (window decays to the right).
    g : int, float, or None
        Effective number of samples under the window, $g > 1$. Used as a
        more readable surrogate for the window decay factor; see [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018). <br>
        $g$ is counted to the right of $k+\delta$ for forward direction
        and to the left for backward direction. Must be ``None`` when ``gamma`` is
        provided instead.
    delta : int, optional
        Relative index at which the window weight is normalised to 1. Default: 0.
    gamma : float or None, optional
        Window decay factor (alternative to ``g``). When set, ``g`` must be ``None``.
        For forward direction, ``gamma > 1`` is required for stability; a warning is
        issued for ``gamma <= 1``. For backward direction, ``gamma < 1`` is required;
        a warning is issued for ``gamma >= 1``. See [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018) Table IV.
    label : str or None, optional
        Segment label, useful for debugging in more complex systems. Default: None.

    Notes
    -----
    The interval includes both boundaries ``a`` and ``b``, so the sum runs over
    $b - a + 1$ samples when both are finite.

    Examples
    --------
    >>> import lmlib as lm
    >>> segment = lm.Segment(a=-20, b=-1, direction=lm.FORWARD, g=15)
    >>> print(segment)
    Segment(a=-20, b=-1, direction=fw, g=15, delta=0, label=n/a)

    >>> segment = lm.Segment(a=0, b=100, direction=lm.BACKWARD, g=15, delta=30, label="right-sided window with shift")
    >>> print(segment)
    Segment(a=0, b=100, direction=bw, g=15, delta=30, label=right-sided window with shift)
    """

    def __init__(self, a:Union[int, float], b:Union[int, float], direction:str, g:Union[int, float, None], delta:int=0, label:str=None, gamma:Union[int, float]=None):
        self._a = None
        self._b = None
        self.set_boundaries(a, b)
        self.direction = direction
        if gamma is not None:
            assert g is None, "g is not None. If gamma is set, g has to be None."
            self._g = None
            self.gamma = np.float(gamma)
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
        """Return a human-readable summary of the segment's parameters."""
        return f'{type(self).__name__}(' \
               f'a={self.a}, ' \
               f'b={self.b}, ' \
               f'direction={self.direction}, ' \
               f'g={self.g}, ' \
               f'delta={self.delta}, ' \
               f'label={self.label})'

    @property
    def a(self):
        r"""int, np.inf: Left boundary of the segment's interval $a$"""
        return self._a

    @property
    def b(self):
        r"""int, np.inf : Right boundary of the segment's interval $b$"""
        return self._b

    @property
    def g(self):
        r"""
        int or float : Effective number of samples $g$, setting the window width.

        The effective number of samples $g$ is used to derive and set the window
        decay factor $\gamma$ internally:

        - forward direction: $\gamma = g / (g - 1) > 1$
        - backward direction: $\gamma = (g - 1) / g < 1$

        Must satisfy $g > 0$. See [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018) Section III.A.
        """
        return self._g

    @g.setter
    def g(self, g):
        assert isinstance(g, (int, float)), 'Effective number of samples g is not of type integer or float.'
        assert g > 1, 'Effective number of samples g has to be greater than one.'
        self._g = g

    @property
    def direction(self):
        r"""
        str : returns the segment's recursion computation `direction`

        - [`FORWARD`][lmlib.statespace.segment.FORWARD], [`FW`][lmlib.statespace.segment.FW] or `'fw'` use forward computation with forward recursions
        - [`BACKWARD`][lmlib.statespace.segment.BACKWARD], [`BW`][lmlib.statespace.segment.BW] or `'bw'` use backward computation with backward recursions
        """
        return self._direction

    @direction.setter
    def direction(self, direction):
        r"""
        str : Sets the segment's recursion computation `direction`

        - [`FORWARD`][lmlib.statespace.segment.FORWARD], [`FW`][lmlib.statespace.segment.FW] or `'fw'` use forward computation with forward recursions
        - [`BACKWARD`][lmlib.statespace.segment.BACKWARD], [`BW`][lmlib.statespace.segment.BW] or `'bw'` use backward computation with backward recursions
        """
        assert isinstance(direction, str), 'Computation direction is not of type string.'
        assert direction in (BW, FW), f'Unknown direction parameter: {self.direction}'
        self._direction = direction

    @property
    def delta(self):
        r"""
        int : Window normalisation index $\delta$.

        The window weight $\gamma^{i - k - \delta}$ equals 1 at relative index
        $i - k = \delta$.
        """
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
        r"""
        float : Window decay factor $\gamma$.

        The window decay factor $\gamma$ is set during initialisation and is
        derived from the effective number of samples [`g`][lmlib.statespace.segment.Segment.g] as follows:

        - forward direction: $\gamma = \frac{g}{g-1} > 1$
        - backward direction: $\gamma = \frac{g-1}{g} < 1$

        See [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018) Table IV.
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

        The return values are the window weights $\alpha_{\delta}(i) \quad \forall i \in [a, b]$ for a constant
        $\gamma$. The window weight function is defined as

        $$
        w_i = \gamma^{i-\delta}
        $$

        For more details see [\[Wildhaber2018\]](../../bibliography.md#wildhaber2018).

        Parameters
        ----------
        thd : float, optional
            Threshold below which window weights are truncated (used to clip
            infinite segment boundaries to a finite range). Default: 1e-6.

        Returns
        -------
        ab_range : ndarray of int
            Relative index range of shape ``(JR,)``, covering the indices at
            which the window weight exceeds ``thd``.
        weights : ndarray of float, shape ``(JR,)``
            Per-sample window weights $\gamma^{i - \delta}$ over ``ab_range``.

        `JR` : index range length <br>
        """

        ab_range = self._ab_range(thd=thd)
        return ab_range, self.gamma ** (np.array(ab_range) - self.delta)
