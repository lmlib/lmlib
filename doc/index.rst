.. lmlib documentation master file, created by
   sphinx-quickstart on Wed May 11 16:58:33 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: lmlib: Model-Based Signal Processing



=======================
**lmlib** Documentation
=======================

.. rst-class:: versiontext

   *Version:* |release| (*Documentation:* |today|) --- Open-Source and for Python.

.. rst-class:: subtitletext

   This is the official documentation for `lmlib <https://pypi.org/project/lmLib/>`_,
   a signal processing library to efficiently analyze single- and multi-channel time series.

   |

   This library implements Recurisve Least Squares (RLS) algorithms with 
   
   * **Windowed Linear State-Space Models**  (Package :ref:`lmlib.statespace`)
   * **Continuous-Time Polynomial Signal Models** (Package :ref:`lmlib.polynomial`) 
 
   |
   
   This library provides methods for
   
   * :ref:`Recursive, Linear Filters <filtering>`,
   * :ref:`Shape Detection <detection>`,
   * :ref:`Event Detection <onset>`,
   * :ref:`Correlation, Convolution, and Matched Filters <convolution>` in low-dimensional vector spaces, 
   * and others. 
  

      



.. rst-class:: maintext1

   This library is the result of many years of :ref:`research <about>` documented in various publications (see :ref:`bibliography`).


.. comment::
	.. image:: _static/images/banner.png
	   :width: 800
	   :class: img-banner

	.. rst-class:: slogan

	   "A Library for Advanced Signal :ref:`Filtering <filtering>`, :ref:`Detection <detection>`, and other Routine :ref:`Tasks <onset>`."




.. comment::
	To ensure either maximum portability or maximum processing speed, lmlib offers multiple backends:

	- Plain Python backend *(highly portable)*
	- Python with JIT--Just In Time backend *(fairly protable, fairly fast)*
	- Python with C/OpenBLAS backend *(maximum processing speed)*

	..
	   *(close-sourced, free for academic use)*






lmlib Doc Content
-----------------

.. toctree::
   :maxdepth: 2
   
   api
   _gallery_coding/index.rst
   _gallery_examples/index.rst
   catalog.rst
   install
   about
   bibliography



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

The library is provided by ETH, FHNW, and BFH, see :ref:`about`.
