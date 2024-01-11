.. title:: lmlib: Model-Based Signal Processing

=======================
**lmlib** Documentation
=======================

.. rst-class:: versiontext

   *Version:* |release| (*Documentation:* |today|) --- Open-Source and for Python.


This is the official documentation for `lmlib <https://pypi.org/project/lmLib/>`_,
a signal processing library to efficiently analyze single- and multi-channel time series.


**This library implements Recursive Least Squares (RLS) algorithms with**

.. grid:: 2

    .. grid-item-card:: Windowed Linear State-Space Models

        (Package  :ref:`lmlib_api_statespace`)

    .. grid-item-card:: Continuous-Time Polynomial Signal Models

        (Package :ref:`lmlib_api_polynomial`)
   
*This library provides methods for*

* :ref:`Recursive, Linear Filters <filtering>`,
* :ref:`Shape Detection <detection>`,
* :ref:`Event Detection <onset>`,
* :ref:`Correlation, Convolution, and Matched Filters <convolution>` in low-dimensional vector spaces,
* and others.





.. rst-class:: maintext1

   This library also supports :ref:`JIT (Just-in-Time) <jit>` compilation for fast processing. 

   This library is the result of many years of :ref:`research <about>` documented in various publications (see :ref:`bibliography`).



.. toctree::
   :hidden:

    User Guide <user_guide>
    API Reference <reference>
    Examples <_gallery_examples/index>
    Install <install>
    About <about>


