.. _lmlib_api_irrls:

Iterative Reweighed Recursive Least Square
==========================================

**Package Abstract ::** This module provides a framework for Message Passing Algorithms

The message passing algorithms, i.e. Modified Bryson-Frazier (MBF) or Backward Recursion with Time-Reversed Information
Filter, Forward Recursion with Marginals (BIFM) use Guassian distributiuons and are parameterized by means and covariances.
The algorithms process a signal series forward and backward and estimate the messages (states)
and can be extended by using NUV priors to process sparsity.
Such algorithms are sequential in their operations as they process the signal in a stepwise manner.
Therefore, they are described by a recursive represented factor graph,
where the nodes or boxes represent the factors, while the edges denote the variables


The module implements Multivariate Message Passing for linear state space models, similar to Kalman Smoothing.
It's applied to tasks like estimating impulsive signals, detecting localized events, and removing outliers.
The approach of Normal priors with unknown variance (NUV) effectively promotes sparsity and
integrates well with parameter learning via expectation maximization (EM).

We use factor graphs, both for reasoning and for describing algorithms.
These graphs consist of nodes or boxes representing factors, while edges signify variables.
A factor graph consists of at least one block containing boxes and edges.
Blocks represent operational stages along the (time) axis :math:`k ={0, 1, 2, \dots, K-1}` and are present redundantly at every (time) step.

This framework allows for the concatenation of various blocks to represent processing at each time step.
For instance, a Kalman Smoother comprises three distinct blocks: BlockSystem, BlockInput, and BlockOutput, illustrated below.

.. image:: /static/lmlib/irrls/kalman_smoother.svg
    :width: 300
    :align: center

You will find detailed information in the papers [Loeliger2016]_ , [Wadehn2016]_ .

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

    BlockContainer
    BlockSystem
    BlockInput
    BlockInput_k
    BlockInputNUV
    BlockOutput
    BlockOutputOutlier

Message Passing Algorithms
--------------------------
.. _classes_message_passing:

.. currentmodule:: lmlib.irrls.message_passing

.. autosummary::
    :toctree: _autosummary
    :recursive:

    MBF
