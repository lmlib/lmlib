# lmlib.statespace.model

::: lmlib.statespace.model
    options:
      show_root_heading: false
      members: []

## Classes

- [`ModelBase`][lmlib.statespace.model.ModelBase] — Abstract base class for autonomous linear state space models (ALSSMs).
- [`Alssm`][lmlib.statespace.model.Alssm] — Generic Autonomous Linear State Space Model (ALSSM)
- [`AlssmPoly`][lmlib.statespace.model.AlssmPoly] — ALSSM with discrete-time polynomial output sequence.
- [`AlssmPolyJordan`][lmlib.statespace.model.AlssmPolyJordan] — ALSSM with a discrete-time polynomial output sequence in Jordan normal form.
- [`AlssmPolyLegendre`][lmlib.statespace.model.AlssmPolyLegendre] — ALSSM whose output basis is the discrete Legendre polynomials on a finite window.
- [`AlssmPolyMeixner`][lmlib.statespace.model.AlssmPolyMeixner] — ALSSM whose output basis is the Meixner polynomials, orthogonal under the
- [`AlssmSin`][lmlib.statespace.model.AlssmSin] — ALSSM with a discrete-time (damped) sinusoidal output sequence.
- [`AlssmExp`][lmlib.statespace.model.AlssmExp] — ALSSM with a discrete-time exponential output sequence.
- [`AlssmStacked`][lmlib.statespace.model.AlssmStacked] — Creates a joined ALSSM generating a stacked output signal of multiple ALSSMs.
- [`AlssmSum`][lmlib.statespace.model.AlssmSum] — Joins multiple ALSSMs generating the output sum.
- [`AlssmProd`][lmlib.statespace.model.AlssmProd] — Joins multiple ALSSMs generating the output product.
