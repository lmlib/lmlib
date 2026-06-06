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
