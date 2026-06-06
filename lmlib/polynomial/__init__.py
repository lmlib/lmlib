r"""
This module provides a calculus for uni- and multivariate polynomials using the vector exponent notation
[\[Wildhaber2019\]](../../bibliography.md#wildhaber2019), Chapter 6.
This calculus simplifies to use polynomials in (squared error) cost functions, e.g., as localized signal models.

.. _lmlib_api_polynomial:

.. Return parameters are highlighted in $\color{blue}{blue}$. replace:: Return parameters are highlighted in $\color{blue}{blue}$.


Polynomial Classes
------------------
.. _classes_polynomial:

- [`Poly`][lmlib.polynomial.Poly]
- [`MPoly`][lmlib.polynomial.MPoly]

Polynomial Operators
--------------------
.. _operators_polynomial:

Operators for Univariate Polynomials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Operators handling **univariate polynomials**.
Return parameters are highlighted in $\color{blue}{blue}$.


Sum of Polynomials $\alpha^\mathsf{T} x^q + \beta^\mathsf{T} x^r$
.......................................................................

- [`poly_sum`][lmlib.polynomial.poly_sum]
- [`poly_sum_coef`][lmlib.polynomial.poly_sum_coef]
- [`poly_sum_coef_Ls`][lmlib.polynomial.poly_sum_coef_Ls]
- [`poly_sum_expo`][lmlib.polynomial.poly_sum_expo]
- [`poly_sum_expo_Ms`][lmlib.polynomial.poly_sum_expo_Ms]

```
>>> import lmlib as lm
>>>
>>> p1 = lm.Poly([1, 3, 5], [0, 1, 2])
>>> p2 = lm.Poly([2, -1], [0, 1])
>>>
>>> p_sum = lm.poly_sum((p1, p2))
>>> print(p_sum)
[ 1.  3.  5.  2. -1.], [0. 1. 2. 0. 1.]
```

Product of Polynomials $\alpha^\mathsf{T} x^q \cdot \beta^\mathsf{T} x^r$
...............................................................................

- [`poly_prod`][lmlib.polynomial.poly_prod]
- [`poly_prod_coef`][lmlib.polynomial.poly_prod_coef]
- [`poly_prod_expo`][lmlib.polynomial.poly_prod_expo]
- [`poly_prod_expo_Ms`][lmlib.polynomial.poly_prod_expo_Ms]

```
>>> import lmlib as lm
>>>
>>> p1 = lm.Poly([1, 3, 5], [0, 1, 2])
>>> p2 = lm.Poly([2, -1], [0, 1])
>>>
>>> p_prod = lm.poly_prod((p1, p2))
>>> print(p_prod)
[ 2 -1  6 -3 10 -5], [0. 1. 1. 2. 2. 3.]
```

Square of a Polynomial $(\alpha^\mathsf{T} x^q)^2$
........................................................

- [`poly_square`][lmlib.polynomial.poly_square]
- [`poly_square_coef`][lmlib.polynomial.poly_square_coef]
- [`poly_square_expo`][lmlib.polynomial.poly_square_expo]
- [`poly_square_expo_M`][lmlib.polynomial.poly_square_expo_M]

```
>>> import lmlib as lm
>>>
>>> p1 = lm.Poly([1, 3, 5], [0, 1, 2])
>>>
>>> p_square = lm.poly_square(p1)
>>> print(p_square)
[ 1  3  5  3  9 15  5 15 25], [0. 1. 2. 1. 2. 3. 2. 3. 4.]
```

Shift of a Polynomial $\alpha^\mathsf{T} (x+ \gamma)^q$
.............................................................

- [`poly_shift`][lmlib.polynomial.poly_shift]
- [`poly_shift_coef`][lmlib.polynomial.poly_shift_coef]
- [`poly_shift_coef_L`][lmlib.polynomial.poly_shift_coef_L]
- [`poly_shift_expo`][lmlib.polynomial.poly_shift_expo]

```
>>> import lmlib as lm
>>>
>>> p1 = lm.Poly([1, 3, 5], [0, 1, 2])
>>> gamma = 2
>>> p_shift = lm.poly_shift(p1, gamma)
>>> print(p_shift)
[27. 23.  5.], [0 1 2]
```

Dilation of a Polynomial $\alpha^\mathsf{T} (\eta x)^q$
.............................................................

- [`poly_dilation`][lmlib.polynomial.poly_dilation]
- [`poly_dilation_coef`][lmlib.polynomial.poly_dilation_coef]
- [`poly_dilation_coef_L`][lmlib.polynomial.poly_dilation_coef_L]

```
>>> import lmlib as lm
>>>
>>> p1 = lm.Poly([1, 3, 5], [0, 1, 2])
>>> eta = -5
>>> p_dilation = lm.poly_dilation (p1, eta)
>>> print(p_dilation)
[  1 -15 125], [0 1 2]
```

Integral of a Polynomial $\int (\alpha^\mathsf{T} x^q) dx$
................................................................

- [`poly_int`][lmlib.polynomial.poly_int]
- [`poly_int_coef`][lmlib.polynomial.poly_int_coef]
- [`poly_int_coef_L`][lmlib.polynomial.poly_int_coef_L]
- [`poly_int_expo`][lmlib.polynomial.poly_int_expo]

```
>>> import lmlib as lm
>>>
>>> p1 = lm.Poly([1, 3, 5], [0, 1, 2])
>>>
>>> p_int = lm.poly_int(p1)
>>> print(p_int)
[1.         1.5        1.66666667], [1 2 3]
```

Derivative of a Polynomial $\frac{d}{dx} (\alpha^\mathsf{T} x^q)$
.......................................................................

- [`poly_diff`][lmlib.polynomial.poly_diff]
- [`poly_diff_coef`][lmlib.polynomial.poly_diff_coef]
- [`poly_diff_coef_L`][lmlib.polynomial.poly_diff_coef_L]
- [`poly_diff_expo`][lmlib.polynomial.poly_diff_expo]

```
>>> import lmlib as lm
>>>
>>> p1 = lm.Poly([1, 3, 5], [0, 1, 2])
>>>
>>> p_diff = lm.poly_diff(p1)
>>> print(p_diff)
[ 0  3 10], [0 0 1]
```

Operators for Multivariate Polynomials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Operators handling **multi variate polynomials**.
Return parameters are highlighted in $\color{blue}{blue}$.
"""

from .poly import *
