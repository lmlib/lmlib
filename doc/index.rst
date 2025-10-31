:html_theme.sidebar_secondary.remove: true


lmlib documentation
===================

.. rst-class:: versiontext

   *Version:* |release| (*Documentation:* |today|) --- Open-Source and for Python.

This is the official documentation for `lmlib <https://pypi.org/project/lmLib/>`_,
a signal processing library to efficiently analyze single- and multi-channel time series.


**This library implements Recursive Least Squares (RLS) algorithms with**

.. grid:: 2

    .. grid-item-card:: Windowed Linear State-Space Models

        (Package  :py:mod:`lmlib.statespace`)

    .. grid-item-card:: Continuous-Time Polynomial Signal Models

        (Package :py:mod:`lmlib.polynomial`)

*This library provides methods for*

* :ref:`Recursive, Linear Filters <filtering>`,
* :ref:`Shape Detection <detection>`,
* :ref:`Event Detection <onset>`,
* :ref:`Correlation, Convolution, and Matched Filters <convolution>` in low-dimensional vector spaces,
* and others.



.. rst-class:: maintext1

   This library also supports :ref:`different backends <_backend>` for fast processing.

   This library is the result of many years of :ref:`research <about_us.rst>` documented in various publications (see :ref:`bibliography`).



.. toctree::
   :maxdepth: 1
   :caption: Contents:

    User Guide<user_guide/index>
    API<api/index>
    Examples<_gallery_examples/index>
    About Us<about_us>