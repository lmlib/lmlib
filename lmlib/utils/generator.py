
import os
from warnings import warn

import numpy as np
from .check import *

__all__ = ['gen_sine', 'gen_rect', 'gen_tri', 'gen_saw', 'gen_pulse', 'gen_exp',
           'gen_steps', 'gen_slopes', 'gen_wgn',
           'gen_rand_walk', 'gen_rand_pulse',
           'gen_conv',
           'load_data', 'load_data_mc',
           'load_lib_csv', 'load_lib_csv_mc',
           'load_csv', 'load_csv_mc',
           'k_period_to_omega']

# Get data_path
this_dir, _ = os.path.split(__file__)
data_path = os.path.join(this_dir, "data/")


def gen_sine(K, k_periods, amplitudes=None, k0s=None):
    r"""
    Generates multiple sinusoidal signals and adds them to one

    Parameters
    ----------
    K : int
        Signal length
    k_periods : int or array_like of int
        signal periodicity in number of samples per period.
    amplitudes : scalar, array_like of scalars, optional
        amplitude(s) of a signal, if set to None all amplitudes are set to 1.0
    k0s : int or array_like of int, optional
        time index(es) of first zero-crossing of sinusoidal signal, if set to None all k0s are set to 1.0

    Returns
    -------
    out : ndarray
        Sum of Sinusoidal signals of length **K**.

    Example
    -------
    ![Sinusoidal Signal](../_generated/catalog/generators/example-gen_sinusoidal_plot.png)

    ```python
    --8<-- "catalog/generators/example-gen_sinusoidal_plot.py:5"
    ```
    """

    if np.isscalar(k_periods):
        amplitudes = 1.0 if amplitudes is None else amplitudes
        k0s = 0.0 if k0s is None else k0s
        assert np.isscalar(amplitudes), 'Not matching type, excpected scalar)'
        assert np.isscalar(k0s), 'Not matching type, excpected scalar)'

        return amplitudes * np.sin(2 * np.pi * (np.linspace(0, K / k_periods, K) + k0s / k_periods))
    elif is_array_like(k_periods):
        amplitudes = np.ones_like(k_periods) if amplitudes is None else amplitudes
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
    r"""
    Exponentially decaying signal generator

    $y_k = \gamma^{k-k_0}$

    Parameters
    ----------
    K : int
        Signal length
    decay : float
        decay factor $\gamma$
    k0 : int
        index shift $k_0$; it follows that $y_{k_0} = 1$

    Returns
    -------
    out : ndarray
        Returns an exponential decaying signal of length **K**, normalized to 1 at index **k0**.

    Example
    -------
    ![Exponential Signal](../_generated/catalog/generators/example-gen_exponential_plot.png)

    ```python
    --8<-- "catalog/generators/example-gen_exponential_plot.py:5"
    ```
    """
    return np.power(decay, np.arange(0 - k0, K - k0))


def gen_rect(K, k_period, k_on=None, duty_cycle=None, k0=0):
    r"""
    Rectangular (pulse wave) signal generator

    Parameters
    ----------
    K : int
        Signal length
    k_period: int
        periodicity, number of samples per period
    k_on : int, optional
        Number of samples of value 1, followed by **k_period**-**k_on** samples of value 0. Default is k_period//2 (only k_on or duty_cycle can be used)
    duty_cycle : float, optional
        Duty Cycle of a period (starts with 1). Default is k_period//2 (only k_on or duty_cycle can be used)
    k0 : int, optional
        Start shift of a period, default k0=0

    Returns
    -------
    out : ndarray, shape=(K,)
        Returns a rectangular wave signal of length `K`.

    Example
    -------
    ![Rectangle Signal](../_generated/catalog/generators/example-gen_rectangle_plot.png)

    ```python
    --8<-- "catalog/generators/example-gen_rectangle_plot.py:5"
    ```
    """

    assert k_on is None or duty_cycle is None, "Set only k_on or duty_cycle. These cannot be used at the same time."
    if k_on is None:
        k_on = k_period // 2 if duty_cycle is None else int(k_period * duty_cycle)
        assert k_on <= k_period, "duty_cycle not in range from 0 to 1"

    assert k_on <= k_period, "k_on must be smaller equal than k_period."

    out = np.zeros(K, )
    for p in np.arange(-(k0 % k_period), K, k_period):
        p_range = np.arange(p, min(p + k_on, K))
        out[p_range] = np.ones((p_range.size,))
    return out


def gen_saw(K, k_period):
    r"""
    Sawtooth signal generator

    Parameters
    ----------
    K : int
        Signal length
    k_period: int
        periodicity, number of samples per period

    Returns
    -------
    out : ndarray, shape=(K,)
        Returns a repetitive slope signal of length **K**. Amplitudes are normalize from 0 to 1.
    """
    return np.remainder(range(K), k_period) / k_period - 0.5


def gen_tri(K, k_period):
    r"""
    Triangular signal generator

    Parameters
    ----------
    K : int
        Signal length
    k_period: int
        periodicity, number of samples per period

    Returns
    -------
    out : ndarray, shape=(K,)
        Returns a triangular signal of length **K** with **k_period** samples per triangle. Amplitudes aer normalize from 0 to 1.

    Example
    -------
    ![Triangle Signal](../_generated/catalog/generators/example-gen_triangle_plot.png)

    ```python
    --8<-- "catalog/generators/example-gen_triangle_plot.py:5"
    ```
    """
    return 1 - abs(np.remainder(range(K), k_period) - 0.5 * k_period) / (0.5 * k_period) - 0.5


def gen_pulse(K, ks):
    r"""
    Pulse signal generator

    Parameters
    ----------
    K : int
        Signal length
    ks : list
        Time indices of unit impulses

    Returns
    -------
    out : ndarray, shape=(K,)
        Returns a unit impulse signal trail of length **K** with values at indices **ks** set to 1, all others to 0.

    Example
    -------
    ![Unit Impulse Signal](../_generated/catalog/generators/example-gen_pulse_plot.png)

    ```python
    --8<-- "catalog/generators/example-gen_pulse_plot.py:5"
    ```
    """
    out = np.zeros((K,))
    assert all(0 <= k < K for k in ks), "k is not in the range of K."
    out[ks] = 1
    return out


def gen_steps(K, ks, deltas):
    r"""
    Step signal generator

    Parameters
    ----------
    K : int
        Signal length
    ks : list
        Amplitude step locations (indexes)
    deltas : list
        Relative step amplitudes at indexes **ks**

    Returns
    -------
    out : ndarray, shape=(K,)
        Returns a step signal of length **K** with steps of relative amplitudes **deltas** at indexes **ks**.

    Example
    -------
    ![Steps Signal](../_generated/catalog/generators/example-gen_steps_plot.png)

    ```python
    --8<-- "catalog/generators/example-gen_steps_plot.py:5"
    ```
    """
    out = np.zeros((K,))
    out[ks] = deltas
    return np.cumsum(out)


def gen_slopes(K, ks, deltas):
    r"""
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
    out : ndarray, shape=(K,)
        Returns a signal of length `K` with chances in slope by the values `deltas` at indeces `ks`.

    Example
    -------
    ![Slopes Signal](../_generated/catalog/generators/example-gen_slopes_plot.png)

    ```python
    --8<-- "catalog/generators/example-gen_slopes_plot.py:5"
    ```
    """
    return np.interp(np.arange(K), ks, np.cumsum(deltas))


def gen_wgn(size, sigma, seed=None):
    r"""
    White Gaussian noise signal generator

    Parameters
    ----------
    size : int or tuple of ints
        Signal length when 'size' is integer. If 'size' is a tuple, the output shape corresponds to the tuple entries
    sigma : float
        Sample variance
    seed : int, None
        random number generator seed, default = *None*.

    Returns
    -------
    out : ndarray
        Returns a white Gaussian noise signal of shape like `size`  and variance `sigma`.

    Example
    -------
    ![White Gaussian Noise Signal](../_generated/catalog/generators/example-gen_wgn_plot.png)

    ```python
    --8<-- "catalog/generators/example-gen_wgn_plot.py:5"
    ```
    """
    np.random.seed(seed)
    return np.random.normal(0, sigma, size)


def gen_rand_walk(size, seed=None):
    r"""
    Random walk generator

    Parameters
    ----------
    size : int or tuple of ints
        Signal length when 'size' is integer. If 'size' is a tuple, the output shape corresponds to the tuple entries
    seed : int, None
        random number generator seed, default = *None*.

    Returns
    -------
    out : ndarray
        Returns a signal of shape `size` with a random walk

    Example
    -------
    ![Random Walk Signal](../_generated/catalog/generators/example-gen_rand_walk_plot.png)

    ```python
    --8<-- "catalog/generators/example-gen_rand_walk_plot.py:5"
    ```
    """
    np.random.seed(seed)
    size = (size,) if isinstance(size, int) else size
    return np.cumsum(np.sign(np.random.randn(*size)) * np.random.ranf(size), axis=0)


def gen_rand_pulse(size, n_pulses, length=1, seed=None):
    r"""
    Random pulse signal generator

    Parameters
    ----------
    size : int or tuple of ints
        Signal length when 'size' is integer. If 'size' is a tuple, the output shape corresponds to the tuple entries
    n_pulses : int
        Number of pulses in the per signal
    length : int
        pulse length (number of samples per pulse set to `1`)
    seed : int, None
        random number generator seed, default = *None*.


    Returns
    -------
    out : ndarray
        Returns signal of shape `size` with exactly `N` unity pulses of length `N` at random positions per signal

    Example
    -------
    ![Random Pulse Signal](../_generated/catalog/generators/example-gen_rand_pulse_plot.png)

    ```python
    --8<-- "catalog/generators/example-gen_rand_pulse_plot.py:5"
    ```
    """
    np.random.seed(seed)
    K = size[0] if isinstance(size, tuple) else size

    rui = np.zeros(K)
    ks = np.random.randint(0, K - 1, size=n_pulses)
    rui[ks] = np.ones(n_pulses)

    out = np.convolve(rui, np.ones((length,)), 'same')
    if isinstance(size, tuple):
        return np.tile(np.reshape(out, (K,) + (1,) * (len(size) - 1)), (1,) + size[1:])
    return out


def gen_conv(base, template):
    r"""
    Convolves two signals. The output signal shape (number of channels and signal length)  is preserved from the `base` signal.

    Parameters
    ----------
    base : array_like
        Base signal to be convolved, either single- or multi-channel.
    template : array_like
        Signal template to be convolved with `base`, either a single- or multi-channel. If `base` is multi-channel, the number of channels has to correspond to the number of channels of `base`.

    Returns
    -------
    out : ndarray, shape=(K,)
        If `template` is a single-channel signal,
        the convolution is applied to each channel of `base`, otherwise the convolution between `base` and `template`  is applied per-channel.
        The output signal is of the same dimension as `base` signal, cf. ``numpy.convolve(..., mode='same')``.


    Example
    -------
    ![Convolution of Two Signals](../_generated/catalog/generators/example-gen_convolve_plot.png)

    ```python
    --8<-- "catalog/generators/example-gen_convolve_plot.py:5"
    ```
    """
    y1 = np.asarray(base)
    y2 = np.asarray(template)

    assert y1.ndim <= 2, "y1 has more then two-dimensions."
    assert y2.ndim < 2, "y2 has more then one-dimension."

    if y1.ndim == 2:
        return np.stack(*[np.convolve(ych, y2, 'same') for ych in y1], axis=-1)
    return np.convolve(y1, y2, 'same')


@deprecated
def load_data(name, K=-1, kstart=0, chIdx=0):
    r"""
    Loads a single channel signal from the signal catalog, see biosignals_catalog.

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
        If the signal has multiple channels, chIdx selects the `chId` th channel in the signal
        Default: chIdx = 0

    Returns
    -------
    out : ndarray, shape=(K,)
        Signal with shape=(K,)
    """
    warn('load_data will be deprecated, use load_lib_csv instead.', DeprecationWarning, stacklevel=2)

    y_out = load_data_mc(name, K, kstart, [chIdx])
    return y_out[:, 0]


@deprecated
def load_data_mc(name, K=-1, kstart=0, chIdxs=None):
    r"""
    Loads a multi-channel signal from the signal catalog, see biosignals_catalog.

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
        List of channels index to load.
        If is None then all channels will be loaded.


    Returns
    -------
    out : ndarray, shape=(K, M)
        else shape=(K, M) for multichannel signals or uf `channels` is an array_like of length `M`

    Note
    ----
    If a file contains only one signal it will be loaded in a shape of a multi-channel signal (K, 1)
    """
    warn('load_data_mc will be deprecated, use load_lib_csv_mc instead.', DeprecationWarning, stacklevel=2)

    assert isinstance(name, str), "Filename is not a string."

    y = np.loadtxt(data_path + name, delimiter=",")
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
    r"""
    Converts sample base period  (samples per cycle) to the normalized frequency

    Parameters
    ----------
    k_period : int
        number of samples per period

    Returns
    -------
    w : float
        Normalized frequency, $\omega = {2 \pi}/{k_\mathrm{period}}$
    """
    return 2 * np.pi / k_period


def load_csv(file, K=-1, k_start=0, channel=0, ds_rate=1, **kwargs):
    """
    loads csv data as a single-channel data shape

    `load_csv` calls `numpy.genfromtxt` with a different interface.

    Parameters
    ----------
    file : str
        path to csv file (with '.csv' ending )
    K : int, optional
        signal length, default loads whole data (K=-1)
    k_start : int, optional
        start of signal, default starts at k_start=0
    channel : int, optional
        load column of csv with the index specified by `channel`
        default is 0 and loads the first column
    ds_rate : int
        down-sample rate (ds_rate >= 1)
    kwargs : optional
        keyword arguments passed to `numpy.genfromtxt`
        to exclude header add `skip_header=numbers_of_header_lines`

    Returns
    -------
    y : np.ndarray
        1 dimensional array of containing signal values over time
    """
    return load_csv_mc(file, K, k_start, (channel,), ds_rate, **kwargs)


def load_csv_mc(file, K=-1, k_start=0, channels=None, ds_rate=1, **kwargs):
    """
    loads csv data as a multi-channel data shape

    `load_csv_mc` calls `numpy.genfromtxt` with a different interface.

    Parameters
    ----------
    file : str
        path to csv file (with '.csv' ending )
    K : int, optional
        signal length, default loads whole data (K=-1)
    k_start : int, optional
        start of signal, default starts at k_start=0
    channels : list, None, optional
        load columns of csv with the index specified in `channels`
        default is None and loads all channels
    ds_rate : int
        down-sample rate (ds_rate >= 1)
    kwargs : optional
        keyword arguments passed to `numpy.genfromtxt`
        to exclude header add `skip_header=numbers_of_header_lines`

    Returns
    -------
    y : np.ndarray
        2-dimensional array, first is time dimensions, second, channels dimension
    """
    assert ds_rate >= 1, "ds_rate : down-sample rate has to larger equal 1"
    max_rows = None if K == -1 else K + k_start
    y = np.genfromtxt(fname=file, delimiter=',', usecols=channels, max_rows=max_rows, **kwargs)[k_start::ds_rate]
    return y


def load_lib_csv(filename, K=-1, k_start=0, channel=0, ds_rate=1, **kwargs):
    r"""
    loads a library-internal csv data file from the signal catalog as a single-channel data shape

    See filenames as biosignals_catalog
    `load_lib_csv` calls [`genfromtxt`][numpy.genfromtxt] with a different interface.

    Parameters
    ----------
    filename : str
        filename (with '.csv' ending ) See biosignals_catalog.
    K : int, optional
        signal length, default loads whole data (K=-1)
    k_start : int, optional
        start of signal, default starts at k_start=0
    channel : int, optional
        load column of csv with the index specified by `channel`
        default is 0 and loads the first column
    ds_rate : int
        down-sample rate (ds_rate >= 1)
    kwargs : optional
        keyword arguments passed to [`genfromtxt`][numpy.genfromtxt]
        to exclude header add `skip_header=numbers_of_header_lines`

    Returns
    -------
    y : np.ndarray
        1 dimensional array of containing signal values over time
    """
    return load_csv(data_path + filename, K, k_start, channel, ds_rate, **kwargs)


def load_lib_csv_mc(filename, K=-1, k_start=0, channels=None, ds_rate=1, **kwargs):
    r"""
    loads a library-internal csv data file from the signal catalog as a multi-channel data shape

    See filenames as biosignals_catalog
    `load_csv_mc` calls numpy.genfromtxt with a different interface.

    Parameters
    ----------
    filename : str
        filename (with '.csv' ending ) See biosignals_catalog.
    K : int, optional
        signal length, default loads whole data (K=-1)
    k_start : int, optional
        start of signal, default starts at k_start=0
    channels : list, None, optional
        load columns of csv with the index specified in `channels`
        default is None and loads all channels
    ds_rate : int
        down-sample rate (ds_rate >= 1)
    kwargs : optional
        keyword arguments passed to `numpy.genfromtxt`
        to exclude header add `skip_header=numbers_of_header_lines`

    Returns
    -------
    y : np.ndarray
        2-dimensional array, first is time dimensions, second, channels dimension
    """
    return load_csv_mc(data_path + filename, K, k_start, channels, ds_rate, **kwargs)
