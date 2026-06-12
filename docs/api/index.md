# API Reference

Welcome to the **lmlib** API documentation. The reference below is generated
directly from the library's docstrings and covers the full public API.

## State Space

Autonomous linear state-space models (ALSSMs) and the recursive least-squares
machinery to fit them to signals: the models themselves, squared-error cost
functions built on them, solvers, window and segment definitions, and
higher-level applications.

- [`lmlib.statespace.model`](lmlib.statespace.model/index.md) — autonomous linear
  state-space models (ALSSMs).
- [`lmlib.statespace.cost`](lmlib.statespace.cost/index.md) — squared-error cost
  functions built on ALSSMs.
- [`lmlib.statespace.rls`](lmlib.statespace.rls/index.md) — recursive least-squares solvers.
- [`lmlib.statespace.window`](lmlib.statespace.window/index.md) — window definitions.
- [`lmlib.statespace.segment`](lmlib.statespace.segment/index.md) — segment definitions
  and direction constants.
- [`lmlib.statespace.trajectory`](lmlib.statespace.trajectory/index.md) — trajectories.
- [`lmlib.statespace.backend`](lmlib.statespace.backend.md) — backend selection.
- [`lmlib.statespace.applications`](lmlib.statespace.applications/index.md) —
  higher-level applications (e.g. `TSLM`).

## Polynomial

A calculus for univariate and multivariate polynomials in vector-exponent
notation, used to build localized polynomial signal models and to combine them
inside squared-error cost functions.

- [`lmlib.polynomial.poly`](lmlib.polynomial.poly/index.md) — uni- and multivariate
  polynomials in vector exponent notation, plus the polynomial operator calculus.

## Utils

Supporting utilities used throughout the library: synthetic test-signal
generators and signal loaders, input-validation helpers, and the lmlib color
palette.

- [`lmlib.utils.generator`](lmlib.utils.generator.md) — synthetic signal
  generators and signal loaders.
- [`lmlib.utils.check`](lmlib.utils.check.md) — validation helpers.
- [`lmlib.utils.colors`](lmlib.utils.colors.md) — color palette.

<!-- gen-api:function-groups
# Single source of truth for create_api.py's thematic grouping of a module's
# functions on its overview page. Maps a module to an ordered list of
# {section title: [name prefixes]}. A function is placed in the first section
# whose longest matching name-prefix it starts with; unmatched ones fall into a
# trailing "Other" section.
lmlib.polynomial.poly:
  - "Sum of Polynomials": [poly_sum]
  - "Product of Polynomials": [poly_prod]
  - "Square of Polynomials": [poly_square]
  - "Shift of Polynomials": [poly_shift]
  - "Dilation of Polynomials": [poly_dilation]
  - "Integration of Polynomials": [poly_int]
  - "Differentiation of Polynomials": [poly_diff]
  - "Addition of Multivariate Polynomials": [mpoly_add]
  - "Multiplication of Multivariate Polynomials": [mpoly_multiply]
  - "Product of Multivariate Polynomials": [mpoly_prod]
  - "Square of Multivariate Polynomials": [mpoly_square]
  - "Shift of Multivariate Polynomials": [mpoly_shift]
  - "Integration of Multivariate Polynomials": [mpoly_int]
  - "Differentiation of Multivariate Polynomials": [mpoly_diff]
  - "Definite Integration of Multivariate Polynomials": [mpoly_def_int]
  - "Substitution of Multivariate Polynomials": [mpoly_substitute]
  - "Independent Dilation of Multivariate Polynomials": [mpoly_dilate_ind]
  - "Dilation of Multivariate Polynomials": [mpoly_dilate]
  - "Sequences, Matrices and Basis Utilities": [kron_sequence, extend_basis, permutation_matrix, commutation_matrix, remove_redundancy, mpoly_remove_redundancy, mpoly_transformation, mpoly_extend]
-->
