:html_theme.sidebar_secondary.remove: true


lmlib documentation
===================

.. rst-class:: versiontext

   *Version:* |release| (*Documentation:* |today|) --- Open-Source and for Python.

This is the official documentation for `lmlib <https://pypi.org/project/lmLib/>`_,
a signal processing library to efficiently analyze single- and multi-channel time series using linear state space models.


**This library implements Recursive Least Squares (RLS) algorithms with**

.. grid:: 2

    .. grid-item-card:: Windowed Linear State-Space Models

        (Package  :py:mod:`lmlib.statespace`)

    .. grid-item-card:: Continuous-Time Polynomial Signal Models

        (Package :py:mod:`lmlib.polynomial`)

*This library provides methods for*

* :ref:`Recursive, Linear Filtering <filtering>`,
* :ref:`Shape Detection <detection>`,
* :ref:`Event Detection <onset>`,
* :ref:`Correlation, Convolution, and Matched Filters <convolution>` in low-dimensional vector spaces,
* and others.



.. rst-class:: maintext1

   This library is optimized for fast processing using GOU and other.

   This library is the essence of many years of `research <about>`_ documented in many publications (see :ref:`bibliography`).



.. toctree::
   :maxdepth: 1
   :caption: Contents:

    User Guide<user_guide/index>
    API<api/index>
    Examples<_gallery_examples/index>
    About Us<about_us>