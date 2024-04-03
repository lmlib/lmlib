.. _lmlib_api_irrls:

Iteratively Reweighed Recursive Least Squares (IRRLS) Algorithms
================================================================

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

The implementation of these methods is according to public research papers such as [Loeliger2016]_ , [Wadehn2016]_, and the :ref:`Literature <bibliography>` .

Implementation of IRRLS Algorithms
----------------------------------

.. note::
   More fully working, ready-to-use examples are found here: :ref:`irrls_examples`.)


**Step 1.** Definition of a recursive algorithm by setting up a :class:`.FactorGraph` with its repetitive part encapsulated in a :class:`.SectionContainer`.

.. image:: /static/doc/irrls-messagepassing-layer1.svg
    :width: 800
    :align: center


.. code-block:: python

   # Defining factor graph structure and comprising sections
   section_1 = lm.SectionInput(B = [[1]], label="Section 1")
   ...
   section_M = lm.SectionOutput(C = [[1]], y, label="Section M") # section with observations y
   sc = lm.SectionContainer(sections=[section_1, ..., section_M], label="Section Container")
   fg = lm.FactorGraph(section=sc, # assigning a section container as the repetitive section
                       left_side_prior=(mu_0, sigma_0), # left-side prior (optional)
                       right_side_prior(mu_K, sigma_K) ) # right-side prior (optional)


Note that the :code:`Sections` can be of any number, sequence, and type listed below (List: :ref:`classes-section`)

**Step 2.** Selection and initialization of the :code:`MessagePassing` method including allocation of the necessary memories to store transient messages and (final) parameter estimates:

.. image:: /static/doc/irrls-messagepassing-layer13.svg
    :width: 800
    :align: center

.. role:: raw-html(raw)
   :format: html

The :raw-html:`<font color="blue">blue elements</font>` in the figure above indicate the message passing structures added to iterate through the algorithm defined by the factor graph.
This message passing is initialized and iterated as

.. code-block:: python

   fg.initialize_mp(message_passing=lm.MBF, K=K) # initialization and memory allocation
   fg.optimize(iterations=100) # run iterative message passing on graph (optimization)


**Step 3.** Access to parameter estimates after iterative optimization from Step 2.
The :raw-html:`<font color="blue">blue bars with labels</font>` in the figure above indicate the stored and accessible parameters for each :class:`.MP_SectionBase` which are for the depicted example:


.. list-table::
   :widths: 25 12 40 20
   :header-rows: 1

   * - Section
     - Parameter
     - Access Method
     -
   * - :code:`MP_SectionContainer`
     - :math:`\color{blue}{\tilde \mu_k}`
     - :meth:`.MP_SectionContainer.get_marginal()`
     - right-side marginal
   * - :code:`MP_Section 1`
     - :math:`\color{blue}{\tilde \mu_k}`
     - :meth:`.MP_SectionInput.get_marginal()`
     - right-side marginal
   * -
     - :math:`\color{blue}{U_k}`
     - :meth:`.MP_SectionInput.get_U()`
     - input estimate
   * - :code:`MP_Section M`
     - :math:`\color{blue}{\tilde \mu_k}`
     - :meth:`.MP_SectionOutput.get_marginal()`
     - right-side marginal
   * -
     - :math:`\color{blue}{\tilde Y_k}`
     - :meth:`.MP_SectionOutput.get_Y_tilde()`
     - output estimate

Note that each parameter estimate of a :class:`.SectionBase` is stored in its associated :class:`.MP_SectionBase` and therefore must be accessed via this respective message passing object.


.. code-block:: python


   # Option 1: access to generic state marginals x_k
   X = fg.mp_section.get_marginals()

   # Option 2: access to section-class dependent parameters (here: U of input section)
   U = fg.mp_section.get_mp_subsection(section_1).get_U()
   # or alternatively
   U = fg.mp_section.get_mp_subsection("Section 1").get_U()

   # Option 3: access to section-class dependent memory (here: Memory for Yt of output section)
   Yt = fg.mp_section.get_mp_subsection(section_M).memory['Yt']
   # or alternatively
   Yt = fg.mp_section.get_mp_subsection("Section M").memory['Yt']

.. note::
   The available parameters are specific to a section class.
   For a detailed list of section-specific parameters, please refer to corresponding class documentation.

.. _classes_factorgraph:

Factor Graph Class
------------------


.. currentmodule:: lmlib.irrls.factorgraph

.. autosummary::
    :toctree: _autosummary
    :recursive:

    FactorGraph


.. _classes-section:

Section Classes
---------------

.. currentmodule:: lmlib.irrls.section

.. autosummary::
    :toctree: _autosummary
    :recursive:

    SectionBase
    SectionContainer
    SectionSystem
    SectionInput
    SectionInput_k
    SectionInputNUV
    SectionOutput
    SectionOutputOutlier


.. _classes_message_passing:

Message Passing Methods
-----------------------

.. currentmodule:: lmlib.irrls.message_passing

.. autosummary::
    :toctree: _autosummary
    :recursive:

    MBF
