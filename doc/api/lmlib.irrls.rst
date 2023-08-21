.. _lmlib.irrls:

lmlib.irrls
================



**Package Abstract ::** This module provides a calculus for uni- and multivariate polynomials using the vector exponent notation
[Wildhaber2019]_, Chapter 6.
This calculus simplifies to use polynomials in (squared error) cost functions, e.g., as localized signal models.


.. currentmodule:: lmlib.irrls.nodes

Messages Classes
------------------
.. _classes_irrls:

.. autosummary::
    :toctree: _autosummary
    :recursive:

    Messages_MV
    Messages_XW


Block Classes
------------------
.. _classes_polynomial:

.. autosummary::
    :toctree: _autosummary
    :recursive:

	MBF
	MBF.Block
	MBF.Block_Marginals
	MBF.Block_System_A
	MBF.Block_Input_BU
	MBF.Block_Input_NUV
	MBF.Block_Output_Y


List by Factor Graphs (of Block Classes)
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :recursive:
	
    MBF.Block
	
.. image:: ./../../_static/lmlib/irrls/Node_Base.png   
       :width: 200

.. autosummary::
    :toctree: _autosummary
    :recursive:
	
    MBF.Block_Marginals
	
.. image:: ./../../_static/lmlib/irrls/Node_Marginals.png   
      :width: 200

.. autosummary::
    :toctree: _autosummary
    :recursive:
	
    MBF.Block_System_A
	
.. image:: ./../../_static/lmlib/irrls/Node_System_A.png   
      :width: 200

.. autosummary::
    :toctree: _autosummary
    :recursive:
	
    MBF.Block_Input_BU
	
.. image:: ./../../_static/lmlib/irrls/Node_Input_BU.png   
      :width: 250


.. autosummary::
    :toctree: _autosummary
    :recursive:
	
    MBF.Block_Output_Y
	
.. image:: ./../../_static/lmlib/irrls/Node_Output_Y.png   
      :width: 250


.. autosummary::
    :toctree: _autosummary
    :recursive:
	
    MBF.Block_Input_NUV
	
.. image:: ./../../_static/lmlib/irrls/Node_Input_NUV.png   
      :width: 250



.. code::

   >>> import lmlib as lm
   >>>
   >>> p1 = lm.Poly([1, 3, 5], [0, 1, 2])
   >>> p2 = lm.Poly([2, -1], [0, 1])
   >>>
   >>> p_sum = lm.poly_sum((p1, p2))
   >>> print(p_sum)
   [ 1.  3.  5.  2. -1.], [0. 1. 2. 0. 1.]


