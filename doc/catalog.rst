.. _lmlib_signal_catalog:



Signal Catalog
==============

Functions
^^^^^^^^^

.. currentmodule:: lmlib.utils.generator

.. autosummary::

    load_lib_csv
    load_lib_csv_mc

    load_csv
    load_csv_mc

Signals
^^^^^^^
*lmlib* provides some bioelectric signals to work with.
Just import the submodule and load one of the signals as show down below.

.. code-block::

    from lmlib.utils import load_lib_csv

    # load signal
    y = load_lib_csv('EECG_BASELINE_1CH_10S_FS2400HZ.csv')



EECG_BASELINE_1CH_10S_FS2400HZ.csv
----------------------------------

- Esophageal ECG Signal (not filtered)
- 1 channel
- 10 seconds
- Sampling rate: 2400Hz

.. plot::
    :include-source:

    from lmlib.utils import load_lib_csv
    import matplotlib.pyplot as plt

    y = load_lib_csv('EECG_BASELINE_1CH_10S_FS2400HZ.csv')

    plt.figure(figsize=(12, 6))
    plt.plot(y)
    plt.xlabel('k')
    plt.show()


EECG_FILT_1CH_10S_FS2400HZ.csv
------------------------------

- Esophageal ECG Signal (filtered)
- 1 channel
- 10 seconds
- Sampling rate: 2400Hz

.. plot::
    :include-source:

    from lmlib.utils import load_lib_csv
    import matplotlib.pyplot as plt

    y = load_lib_csv('EECG_FILT_1CH_10S_FS2400HZ.csv')

    plt.figure(figsize=(12, 6))
    plt.plot(y)
    plt.xlabel('k')
    plt.show()


EECG_FILT_9CH_10S_FS2400HZ.csv
------------------------------

- Esophageal ECG Signal (filtered)
- 9 channel
- 10 seconds
- Sampling rate: 2400Hz

.. plot::
    :include-source:

    from lmlib.utils import load_lib_csv_mc
    import matplotlib.pyplot as plt

    y = load_lib_csv_mc('EECG_FILT_9CH_10S_FS2400HZ.csv')

    plt.figure(figsize=(12, 6))
    for m in range(9):
        plt.plot(y[:, m] + 8-m, label=f'ch{m}')
    plt.legend()
    plt.xlabel('k')
    plt.show()


SECG3_FILT_HP51_3CH_20S_FS2400HZ.csv
------------------------------------

- Surface ECG Signal (filtered 5 Hz)
- 3 channel
- 20 seconds
- Sampling rate: 2400Hz

.. plot::
    :include-source:

    from lmlib.utils import load_lib_csv_mc
    import matplotlib.pyplot as plt

    y = load_lib_csv_mc('SECG3_FILT_HP51_3CH_20S_FS2400HZ.csv')

    plt.figure(figsize=(12, 6))
    for m in range(3):
        plt.plot(y[:, m] + (2-m)*1.5, label=f'ch{m}')
    plt.legend()
    plt.xlabel('k')
    plt.show()


SECG3_RAW_3CH_20S_FS2400HZ.csv
------------------------------

- Surface ECG Signal (not filtered)
- 3 channel
- 20 seconds
- Sampling rate: 2400Hz

.. plot::
    :include-source:

    from lmlib.utils import load_lib_csv_mc
    import matplotlib.pyplot as plt

    y = load_lib_csv_mc('SECG3_RAW_3CH_20S_FS2400HZ.csv')

    plt.figure(figsize=(12, 6))
    for m in range(3):
        plt.plot(y[:, m] + (2-m)*1.5, label=f'ch{m}')
    plt.legend()
    plt.xlabel('k')
    plt.show()
