# What is lmlib?
The lmlib project brings mathematical model-based signal analysis methods to the application level. It makes them freely available as open-source software. The lmlib project includes [Python source code](https://github.com/lmlib/lmlib), [documentation](../api/index.md), [examples](../_generated/examples/index.md) of how to use lmlib, and [scientific reference publications](../bibliography.md). The lmlib project was initiated by the Signal and Information Processing Laboratory (ISI) at ETH Zurich. It is supported by several [institutions](../about_us.md).

## State Space Module
This module provides methods to define Autonomous Linear State Space Models (ALSSMs) and to define squared error cost functions based on such ALSSMs. ALSSMs are input-free linear state space models (LSSMs), i.e., the model outputs are fully defined by a single state vector, often denoted as initial state vector x. The output vector of such a deterministic model forms the signal model used. Cost functions based on ALSSM are internally efficiently computed using recursive computation rules.

## Polynomial Module
This module provides a calculus for uni- and multivariate polynomials using the vector exponent notation from [\[Wildhaber2019, Chapter 6\]](../bibliography.md#wildhaber2019). This calculus simplifies to use polynomials in (squared error) cost functions, e.g., as localized signal models.

## Utils
This package is a collection of utility functions to accelerate algorithm development and/or for educative purposes:

* deterministic and stochastic signals generators
* a collection of recorded biological signals, see [Signal Catalog](../catalog/_generated_galleries/biosignals/index.md)
* experimental functions (beta)
