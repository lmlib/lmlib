.. _lmlib_api_irrls:

Iteratively Reweighed Recursive Least Squares (IRRLS) Algorithms
==========================================

**Package Abstract ::** This package provides a framework for Gaussian Message Passing Algorithms

This package implements multivariate Gaussian message passing on general linear state-space models and using Forney factor graphs as primary design method.
This leads to algorithms such as the the family of Kalman filters,
filters for sparse signal learning when normal priors with unknown variance (NUV),
an others.

Supported message passing methods:
   - Modified Bryson-Frazier (MBF) 
   - Forward Recursion with Marginals (BIFM)

Supported maximization methods:
   - Expectation Maximization (EM)
   - Alternate Maximization (AM)

The implementation of these methods is according to public research papers such as [Loeliger2016]_ , [Wadehn2016]_, and the Literature **TODO: insert link**.

Implementation of IRRLS Algorithms
-----------------------------------------

.. note::
   More fully working, ready-to-use examples are found here: **[LINK TO EXAMPLES]**.)


**Step 1.** Definition of the Algorithm by setting up a :class:`.FactorGraph` with its repetitive part packed in a :class:`.BlockContainer`.

.. image:: /_static/doc/irrls-notation.svg
    :width: 800
    :align: center


.. code-block:: python

   # Defining factor graph structure and comprising sections
   block_1 = lm.BlockInput( ..., label="Block 1")
   ...
   block_M = lm.BlockOutput( ..., y, label="Block M") # block with observations y
   fg = lm.FactorGraph(lm.BlockContainer(blocks=[block_1, ..., block_M], label="Block Container"),
                       left_side_prior=(mu_0, sigma_0), # left-side prior (optional)
                       right_side_prior(mu_K, sigma_K) ) # right-side prior (optional)

Note that the :code:`Block`s can be of any number, sequence, and type listed below (List: **Block Classes**)  

**Step 2.** Initialization of associated :class:`.message_passing.MessagePassing` with its necessary memory structures to store transient messages and final parameter estimates:

.. image:: /_static/doc/irrls-messagepassing.svg
    :width: 800
    :align: center	

.. role:: raw-html(raw)
   :format: html

The :raw-html:`<font color="blue">blue bars</font>` in the figure above indicate the accessible and stored parameters in each :class:`.MP_Block`

.. code-block:: python

   fg.initialize_mp(message_passing=lm.MBF, K=K) 
   fg.optimize(iterations=100) # run iterative message passing on graph (optimization) 

**Step 3.** Access to parameter estimates after optimization in Step 2. Note that the parameter estimates of a :class:`.MP_Block` get accessed via the corresponding :class:`.Block` of the :class:`.FactorGraph`:

.. code-block:: python

   # Option 1: access to generic state marginals x_k 
   X = fg.get_mp_block(bc).get_marginals() 
   
   # Option 2: access to block-class dependent parameters (here: U of input block)
   U = fg.get_mp_block(blk_input).get_U() 
   
   # Option 3: access to block-class dependent memory (here: Memory for Yt of output block)
   Yt = fg.get_mp_block(block_M).memory['Yt'] 


.. note::
   The available parameters are specific to a block class.  
   For a detailed list of block-specific parameters, please refer to corresponding class documentation. 




Factor Graph Class
------------------
.. _classes_factorgraph:

.. currentmodule:: lmlib.irrls.factorgraph

.. autosummary::
    :toctree: _autosummary
    :recursive:

    FactorGraph

Block Classes
-------------
.. _classes_block:

.. currentmodule:: lmlib.irrls.block

.. autosummary::
    :toctree: _autosummary
    :recursive:

    BlockBase
    BlockContainer
    BlockSystem
    BlockInput
    BlockInput_k
    BlockInputNUV
    BlockOutput
    BlockOutputOutlier

Message Passing Methods
--------------------------
.. _classes_message_passing:

.. currentmodule:: lmlib.irrls.message_passing

.. autosummary::
    :toctree: _autosummary
    :recursive:

    MBF
