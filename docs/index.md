# lmlib Documentation

**Version:** 3.0.0 — Open-Source and for Python.

This is the official documentation for [lmlib](https://pypi.org/project/lmLib/), a signal processing library to efficiently analyze single- and multi-channel time series using linear state space models.

**This library implements Recursive Least Squares (RLS) algorithms with**

- **Windowed Linear State-Space Models** ([State Space](api/index.md))
- **Continuous-Time Polynomial Signal Models** ([Polynomial](api/index.md))

**This library provides methods for**

- [Recursive, Linear Filters](_generated/examples/index.md#filtering)
- [Shape Detection](_generated/examples/index.md#detection)
- [Event Detection](_generated/examples/index.md#event-detection-with-two-sided-line-models)
 [Correlation, Convolution, and Matched Filters](_generated/examples/index.md#convolution) in low-dimensional vector spaces,
- and others

This library is optimized for fast processing using scipy, JIT (Just-in-Time) compilation, GPU, and other.

This library is the essence of many years of research
documented in many publications (see [Bibliography](bibliography.md)).
