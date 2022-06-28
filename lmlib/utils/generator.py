"""Generators for deterministic and stochastic (test) signals"""

import os
import numpy as np
from .check import *

__all__ = ['gen_sine', 'gen_rect', 'gen_tri', 'gen_saw', 'gen_pulse', 'gen_exp',
           'gen_steps', 'gen_slopes' ,'gen_wgn',
           'gen_rand_walk', 'gen_rand_pulse',
           'gen_conv',
           'load_data','load_data_mc',
           'k_period_to_omega']

# Get data_path
this_dir, _ = os.path.split(__file__)
data_path = os.path.join(this_dir, "data/")

def gen_sine(K, k_periods, amplitudes=None, k0s=None):
    """
    Generates multiple sinusoidal signals and adds them to one

    Parameters
    ----------
    K : int
        Signal length
    k_periods : int or array_like of int
        signal periodicities in number of samples per period.
    amplitudes : scalar, array_like of scalars, optional
        amplitude(s) of a signal, if set to None all amlitudes are set to 1.0
    k0s : int or array_like of int, optional
        time index(es) of first zero-crossing of sinusoidal signal, if set to None all k0s are set to 1.0

    Returns
    -------
    out : :class:`numpy.ndarray`
        Sum of Sinusoidal signals of length **K**.

    Example
    -------
    .. plot:: pyplots/gen_sinusoidal_plot.py
        :include-source:
    """


    if np.isscalar(k_periods):
        amplitudes = 1.0 if amplitudes is None else amplitudes
        k0s = 0.0 if k0s is None else k0s
        assert np.isscalar(amplitudes), 'Not matching type, excpected scalar)'
        assert np.isscalar(k0s), 'Not matching type, excpected scalar)'

        return amplitudes * np.sin(2 * np.pi * (np.linspace(0, K / k_periods, K) + k0s / k_periods))
    elif is_array_like(k_periods):
        mplitudes = np.ones_like(k_periods) if amplitudes is None else amplitudes
        k0s = np.zeros_like(k_periods) if k0s is None else k0s
        assert len(k_periods) == len(amplitudes), 'amplitudes length does not match'
        assert len(k_periods) == len(k0s), 'k0s length does not match'

        out = np.zeros(K)
        for kp, a, k0 in zip(k_periods, amplitudes, k0s):
            out += a * np.sin(2 * np.pi * (np.linspace(0, K / kp, K) + k0 / kp))
        return out
    else:
        assert False, 'k_periods has wrong type'


def gen_exp(K, decay, k0=0):
    """
    Exponentially decaying signal generator

    :math:`y_k = \gamma^{k-k_0}`

    Parameters
    ----------
    K : int
        Signal length
    decay : float
        decay factor :math:`\gamma`
    k0 : int
        index shift :math:`k_0`; it follows that :math:`y_{k_0} = 1`

    Returns
    -------
    out : :class:`numpy.ndarray`
        Returns an exponential decaying signal of length **K**, normalized to 1 at index **k0**.

    Example
    -------
    .. plot:: pyplots/gen_exponential_plot.py
        :include-source:
    """
    return np.power(decay, np.arange(0 - k0, K - k0))


def gen_rect(K, k_period, k_on):
    """
    Rectangular (pulse wave) signal generator

    Parameters
    ----------
    K : int
        Signal length
    k_period: int
        periodicity, number of samples per period
    k_on : int
        Number of samples of value 1, followed by **k_period**-**k_on** samples of value 0.

    Returns
    -------
    out : :class:`~numpy.ndarray`, shape=(K,)
        Returns a rectangular wave signal of length `K`.

    Example
    -------
    .. plot:: pyplots/gen_rectangle_plot.py
        :include-source:
    """
    assert k_on <= k_period, "k_on must be smaller equal than period."
    out = np.zeros(K, )
    for p in np.arange(0, K, k_period):
        p_range = np.arange(p, min(p + k_on, K))
        out[p_range] = np.ones((p_range.size,))
    return out


def gen_saw(K, k_period):
    """
    Sawtooth signal generator

    Parameters
    ----------
    K : int
        Signal length
    k_period: int
        periodicity, number of samples per period

    Returns
    -------
    out : :class:`~numpy.ndarray`, shape=(K,)
        Returns a repetitive slope signal of length **K**. Amplitudes are normalize from 0 to 1.

    Example
    -------
    .. plot:: pyplots/gen_slope_plot.py
        :include-source:
    """
    return np.remainder(range(K), k_period) / k_period - 0.5


def gen_tri(K, k_period):
    """
    Triangular signal generator

    Parameters
    ----------
    K : int
        Signal length
    k_period: int
        periodicity, number of samples per period

    Returns
    -------
    out : :class:`~numpy.ndarray`, shape=(K,)
        Returns a triangular signal of length **K** with **k_period** samples per triangle. Amplitudes aer normalize from 0 to 1.

    Example
    -------
    .. plot:: pyplots/gen_triangle_plot.py
        :include-source:
    """
    return 1 - abs(np.remainder(range(K), k_period) - 0.5 * k_period) / (0.5 * k_period) -0.5


def gen_pulse(K, ks):
    """
    Pulse signal generator

    Parameters
    ----------
    K : int
        Signal length
    ks : list
        Time indices of unit impulses

    Returns
    -------
    out : :class:`~numpy.ndarray`, shape=(K,)
        Returns a unit impulse signal trail of length **K** with values at indices **ks** set to 1, all others to 0.

    Example
    -------
    .. plot:: pyplots/gen_pulse_plot.py
        :include-source:
    """
    out = np.zeros((K,))
    assert all(0 <= k < K for k in ks), "k is not in the range of K."
    out[ks] = 1
    return out


def gen_steps(K, ks, deltas):
    """
    Step signal generator

    Parameters
    ----------
    K : int
        Signal length
    ks : list
        Amplitude step locations (indeces)
    deltas : list
        Relative step amplitudes at indeces **ks**

    Returns
    -------
    out : :class:`~numpy.ndarray`, shape=(K,)
        Returns a step signal of length **K** with steps of relative amplitudes **deltas** at indeces **ks**.

    Example
    -------
    .. plot:: pyplots/gen_steps_plot.py
        :include-source:
    """
    out = np.zeros((K,))
    out[ks] = deltas
    return np.cumsum(out)


def gen_slopes(K, ks, deltas):
    """
    Slopes signal generator

    Parameters
    ----------
    K : int
        Signal length
    ks : list
        Indices of slope change
    deltas : list
        Slope start to end difference at each index in `ks`

    Returns
    -------
    out : :class:`~numpy.ndarray`, shape=(K,)
        Returns a signal of length `K` with chances in slope by the values `deltas` at indeces `ks`.

    Example
    -------
    .. plot:: pyplots/gen_slopes_plot.py
        :include-source:
    """
    return np.interp(np.arange(K), ks, np.cumsum(deltas))


def gen_wgn(K, sigma, seed=None):
    """
    White Gaussian noise signal generator

    Parameters
    ----------
    K : int
        Signal length
    sigma : float
        Sample variance
    seed : int, None
        random number generator seed, default = *None*.

    Returns
    -------
    out : :class:`~numpy.ndarray`, shape=(K,)
        Returns a white Gaussian noise signal of length `K`  and variance `sigma`.

    Example
    -------
    .. plot:: pyplots/gen_wgn_plot.py
        :include-source:
    """
    np.random.seed(seed)
    return np.random.normal(0, sigma, K)


def gen_rand_walk(K, seed=None):
    """Random walk generator

    Parameters
    ----------
    K : int
        Signal length
    seed : int, None
        random number generator seed, default = *None*.


    Returns
    -------
    out : :class:`~numpy.ndarray`, shape=(K,)
        Returns a signal of length `K` with a random walk

    Example
    -------
    .. plot:: pyplots/gen_rand_walk_plot.py
        :include-source:
    """
    np.random.seed(seed)
    return np.cumsum(np.sign(np.random.randn(K)) * np.random.ranf(K, ))


def gen_rand_pulse(K, n_pulses, length=1, seed=None):
    """
    Random pulse signal generator

    Parameters
    ----------
    K : int
        Signal length
    n_pulses : int
        Number of pulses in the signal
    length : int
        pulse length (number of samples per pulse set to `1`)
    seed : int, None
        random number generator seed, default = *None*.


    Returns
    -------
    out : :class:`~numpy.ndarray`, shape=(K,)
        Returns signal of length `K` with exactly `N` unity pulses of length `N` at random positions.

    Example
    -------
    .. plot:: pyplots/gen_rand_pulse_plot.py
        :include-source:
    """
    np.random.seed(seed)
    rui = np.zeros(K)
    ks = np.random.randint(0, K-1, size=n_pulses)
    rui[ks] = np.ones(n_pulses)
    return np.convolve(rui, np.ones((length,)), 'same')


def gen_conv(base, template):
    """
    Convolves two signals. The output signal shape (number of channels and signal length)  is preserved from the `base` signal.

    Parameters
    ----------
    base : array_like
        Base signal to be convolved, either single- or multi-channel.
    template : array_like
        Signal template to be convolved with `base`, either a single- or multi-channel. If `base` is multi-channel, the number of channels has to correspond to the number of channels of `base`.

    Returns
    -------
    out : :class:`~numpy.ndarray`, shape=(K,)
        If `template` is a sigle-channel signal,
        the convolution is applied to each channel of `base`, otherwise the convolution between `base` and `template`  is applied per-channel.
        The output signal is of the same dimension as `base` signal, cf. ``numpy.convolve(..., mode='same')``.


    Example
    -------
    .. plot:: pyplots/gen_convolve_plot.py
        :include-source:
    """
    y1 = np.asarray(base)
    y2 = np.asarray(template)

    assert y1.ndim <= 2, "y1 has more then two-dimensions."
    assert y2.ndim < 2, "y2 has more then one-dimension."

    if y1.ndim == 2:
        return np.stack(*[np.convolve(ych, y2, 'same') for ych in y1], axis=-1)
    return np.convolve(y1, y2, 'same')


def load_data(name, K=-1, kstart=0, chIdx=0):
    """
    Loads a single channel signal from the signal catalog, see :ref:`lmlib_signal_catalog`.

    Parameters
    ----------
    name : str
        Signal name (from signal catalog)
    K : int, optional
        Length of signal to be loaded. Default is `-1` which loads to end of the file.
        If `K` is larger than the maximal signal length, an assertion is raised.
    kstart : int, optional
        Signal load start index. Default=0
        If `k` is larger than the maximal signal length, an assertion is raised.
    chIdx : int, optional
        If the signal has multiple channels, chIdx selelcts the `chId`th channel in the signal
        Default: chIdx = 0

    Returns
    -------
    out : :class:`~numpy.ndarray`, shape=(K,)
        Signal with shape=(K,)

    """
    y_out = load_data_mc(name, K, kstart, [chIdx])
    return y_out[:, 0]


def load_data_mc(name, K=-1, kstart=0, chIdxs=None):
    """
    Loads a multi channel signal from the signal catalog, see :ref:`lmlib_signal_catalog`.

    Parameters
    ----------
    name : str
        Signal name (from signal catalog)
    K : int, optional
        Length of signal to be loaded. Default is `-1` which loads to end of the file.
        If `K` is larger than the maximal signal length, an assertion is raised.
    kstart : int, optional
        Signal load start index. Default=0
        If `k` is larger than the maximal signal length, an assertion is raised.
    chIdxs : None, array_like, optional
        List of channle index to load.
        If is None then all channles will be loaded.


    Returns
    -------
    out : :class:`~numpy.ndarray`, shape=(K, M)
        else shape=(K, M) for multi-channel signals or uf `channels` is a array_like of length `M`

    Note
    ----
    If a the files contains only one signal it will be loaded in a shapw of a multichannel signal (K, 1)

    """
    assert isinstance(name, str), "Filename is not a string."

    y = np.loadtxt(data_path+name, delimiter=",")
    if y.ndim == 2:
        K_file, M_file = y.shape
        K = K_file if K == -1 else K
        chIdxs = range(M_file) if chIdxs is None else chIdxs
        y_out = y[kstart:K][:, chIdxs]
    else:
        K_file = y.shape[0]
        K = K_file if K == -1 else K
        y_out = y[kstart:K][:, None]
    return y_out

def k_period_to_omega(k_period):
    """
    Converts sample base period  (samples per cycle) to the normalized frequency

    Parameters
    ----------
    k_period : int
        number of samples per period

    Returns
    -------
    w : float
        Normalized frequency, :math:`\omega = {2 \pi}/{k_\mathrm{period}}`

    """
    return 2 * np.pi / k_period