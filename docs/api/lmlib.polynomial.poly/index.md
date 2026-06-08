# lmlib.polynomial.poly

::: lmlib.polynomial.poly
    options:
      show_root_heading: false
      members: []

## Classes

- [`MPoly`][lmlib.polynomial.poly.MPoly] — Multivariate polynomials ${\tilde{\alpha}}^\mathsf{T} (x^q \otimes y^r)$, or with *factorized* coefficient
- [`Poly`][lmlib.polynomial.poly.Poly] — Univariate polynomial in vector exponent notation: $\alpha^\mathsf{T} x^q$

## Functions

- [`poly_sum`][lmlib.polynomial.poly.poly_sum] — $\alpha^\mathsf{T} x^q + \dots + \beta^\mathsf{T} x^r = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}$
- [`poly_sum_coef`][lmlib.polynomial.poly.poly_sum_coef] — $\alpha^\mathsf{T} x^q + \dots + \beta^\mathsf{T} x^r = \color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}$
- [`poly_sum_coef_Ls`][lmlib.polynomial.poly.poly_sum_coef_Ls] — $\alpha^\mathsf{T} x^q + \dots + \beta^\mathsf{T} x^r = (\color{blue}{\Lambda_1} \alpha + \dots + \color{blue}{\Lambda_N}\beta)^\mathsf{T} x^{\tilde{q}}$
- [`poly_sum_expo`][lmlib.polynomial.poly.poly_sum_expo] — $\alpha^\mathsf{T} x^q + \dots + \beta^\mathsf{T} x^r = \tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}$
- [`poly_sum_expo_Ms`][lmlib.polynomial.poly.poly_sum_expo_Ms] — $\alpha^\mathsf{T} x^q + \dots + \beta^\mathsf{T} x^r = \tilde{\alpha}^\mathsf{T} x^{\color{blue}{M_1} q + \dots +\color{blue}{M_N} r}$
- [`poly_prod`][lmlib.polynomial.poly.poly_prod] — $\alpha^\mathsf{T} x^q \cdot \beta^\mathsf{T} x^q = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}$
- [`poly_prod_coef`][lmlib.polynomial.poly.poly_prod_coef] — $\alpha^\mathsf{T} x^q \cdot \beta^\mathsf{T} x^r = \color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}$
- [`poly_prod_expo_Ms`][lmlib.polynomial.poly.poly_prod_expo_Ms] — $\alpha^\mathsf{T} x^q \cdot \beta^\mathsf{T} x^r = \tilde{\alpha}^\mathsf{T} x^{\color{blue}{M_1} q + \color{blue}{M_2} r}$
- [`poly_prod_expo`][lmlib.polynomial.poly.poly_prod_expo] — $\alpha^\mathsf{T} x^q \cdot \beta^\mathsf{T} x^r = \tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}$
- [`poly_square`][lmlib.polynomial.poly.poly_square] — $(\alpha^\mathsf{T} x^q)^2 = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}$
- [`poly_square_coef`][lmlib.polynomial.poly.poly_square_coef] — $\color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}$
- [`poly_square_expo_M`][lmlib.polynomial.poly.poly_square_expo_M] — $\tilde{\alpha}^\mathsf{T} x^{\color{blue}{M} q}$
- [`poly_square_expo`][lmlib.polynomial.poly.poly_square_expo] — $\tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}$
- [`poly_shift`][lmlib.polynomial.poly.poly_shift] — $\alpha^\mathsf{T} (x+ \gamma)^q = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}$
- [`poly_shift_coef`][lmlib.polynomial.poly.poly_shift_coef] — $\color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}$
- [`poly_shift_coef_L`][lmlib.polynomial.poly.poly_shift_coef_L] — $\color{blue}{\Lambda} \alpha^\mathsf{T} x^{\tilde{q}}$
- [`poly_shift_expo`][lmlib.polynomial.poly.poly_shift_expo] — $\tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}$
- [`poly_dilation`][lmlib.polynomial.poly.poly_dilation] — $\alpha^\mathsf{T} (\eta x)^q = \color{blue}{\tilde{\alpha}^\mathsf{T} x^q}$
- [`poly_dilation_coef`][lmlib.polynomial.poly.poly_dilation_coef] — $\alpha^\mathsf{T} (\eta x)^q =\color{blue}{\tilde{\alpha}}^\mathsf{T} x^q$
- [`poly_dilation_coef_L`][lmlib.polynomial.poly.poly_dilation_coef_L] — $\alpha^\mathsf{T} (\eta x)^q =\color{blue}{\Lambda} \alpha^\mathsf{T} x^{q}$
- [`poly_int`][lmlib.polynomial.poly.poly_int] — $\int \big(\alpha^{\mathsf{T}}x^q\big) dx = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}$
- [`poly_int_coef`][lmlib.polynomial.poly.poly_int_coef] — $\int \big(\alpha^{\mathsf{T}}x^q\big) dx  = \color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}$
- [`poly_int_coef_L`][lmlib.polynomial.poly.poly_int_coef_L] — $\int \big(\alpha^{\mathsf{T}}x^q\big) dx  = \color{blue}{\Lambda} \alpha^\mathsf{T} x^{\tilde{q}}$
- [`poly_int_expo`][lmlib.polynomial.poly.poly_int_expo] — $\int \big(\alpha^{\mathsf{T}}x^q\big) dx  = \tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}$
- [`poly_diff`][lmlib.polynomial.poly.poly_diff] — $\frac{d}{dx} \big(\alpha^{\mathsf{T}}x^q\big) = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}$
- [`poly_diff_coef`][lmlib.polynomial.poly.poly_diff_coef] — $\frac{d}{dx} \big(\alpha^{\mathsf{T}}x^q\big) =\color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}$
- [`poly_diff_coef_L`][lmlib.polynomial.poly.poly_diff_coef_L] — $\frac{d}{dx} \big(\alpha^{\mathsf{T}}x^q\big) =\color{blue}{\Lambda} \alpha^\mathsf{T} x^{\tilde{q}}$
- [`poly_diff_expo`][lmlib.polynomial.poly.poly_diff_expo] — $\frac{d}{dx} \big(\alpha^{\mathsf{T}}x^q\big) =\tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}$
- [`mpoly_add`][lmlib.polynomial.poly.mpoly_add] — Sum of two univariate polynomials different variables
- [`mpoly_add_coefs`][lmlib.polynomial.poly.mpoly_add_coefs] — Coefficients for [`mpoly_add`][lmlib.polynomial.poly.mpoly_add]
- [`mpoly_add_expos`][lmlib.polynomial.poly.mpoly_add_expos] — Exponents for [`mpoly_add`][lmlib.polynomial.poly.mpoly_add]
- [`mpoly_multiply`][lmlib.polynomial.poly.mpoly_multiply] — Product of two univariate polynomials different variables
- [`mpoly_prod`][lmlib.polynomial.poly.mpoly_prod] — Product of univariate polynomials different variables
- [`mpoly_square`][lmlib.polynomial.poly.mpoly_square] — Square of multivariate polynomial
- [`mpoly_square_coef_L`][lmlib.polynomial.poly.mpoly_square_coef_L] — Coefficient manipulation matrix for [`mpoly_square`][lmlib.polynomial.poly.mpoly_square].
- [`mpoly_square_coef`][lmlib.polynomial.poly.mpoly_square_coef] — Non-factorized coefficient vector for [`mpoly_square`][lmlib.polynomial.poly.mpoly_square]
- [`mpoly_square_expos`][lmlib.polynomial.poly.mpoly_square_expos] — Exponent vectors for [`mpoly_square`][lmlib.polynomial.poly.mpoly_square]
- [`mpoly_square_expo_Ms`][lmlib.polynomial.poly.mpoly_square_expo_Ms] — Exponent manipulation matrices $M_1, M_2, \dots M_N$ for the square of an *N*-variate polynomial.
- [`mpoly_shift`][lmlib.polynomial.poly.mpoly_shift] — Polynomial with variable shift
- [`mpoly_shift_coef`][lmlib.polynomial.poly.mpoly_shift_coef] — Coefficient vector for [`mpoly_shift`][lmlib.polynomial.poly.mpoly_shift]
- [`mpoly_shift_coef_L`][lmlib.polynomial.poly.mpoly_shift_coef_L] — Coefficient manipulation matrix for [`mpoly_shift`][lmlib.polynomial.poly.mpoly_shift]
- [`mpoly_shift_expos`][lmlib.polynomial.poly.mpoly_shift_expos] — Exponent vector for [`mpoly_shift`][lmlib.polynomial.poly.mpoly_shift]
- [`mpoly_int`][lmlib.polynomial.poly.mpoly_int] — Integral of a multivariate polynomial with respect to the scalar at a position
- [`mpoly_int_coef`][lmlib.polynomial.poly.mpoly_int_coef] — Coefficient vector for [`mpoly_int`][lmlib.polynomial.poly.mpoly_int]
- [`mpoly_int_coef_L`][lmlib.polynomial.poly.mpoly_int_coef_L] — Coefficient manipulation matrix for [`mpoly_int`][lmlib.polynomial.poly.mpoly_int]
- [`mpoly_int_expos`][lmlib.polynomial.poly.mpoly_int_expos] — Exponent vectors for [`mpoly_int`][lmlib.polynomial.poly.mpoly_int]
- [`mpoly_diff`][lmlib.polynomial.poly.mpoly_diff] — Derivative of a multivariate polynomial with respect to the variable at the given position.
- [`mpoly_diff_coef`][lmlib.polynomial.poly.mpoly_diff_coef] — Coefficient vector for [`mpoly_diff`][lmlib.polynomial.poly.mpoly_diff].
- [`mpoly_diff_coef_L`][lmlib.polynomial.poly.mpoly_diff_coef_L] — Coefficient manipulation matrix for [`mpoly_diff`][lmlib.polynomial.poly.mpoly_diff].
- [`mpoly_diff_expos`][lmlib.polynomial.poly.mpoly_diff_expos] — Exponent vectors for [`mpoly_diff`][lmlib.polynomial.poly.mpoly_diff].
- [`mpoly_def_int`][lmlib.polynomial.poly.mpoly_def_int] — Definite integral of a multivariate polynomial with respect to the scalar at a position
- [`mpoly_def_int_coef`][lmlib.polynomial.poly.mpoly_def_int_coef] — Coefficient vector for [`mpoly_def_int`][lmlib.polynomial.poly.mpoly_def_int]
- [`mpoly_def_int_coef_L`][lmlib.polynomial.poly.mpoly_def_int_coef_L] — Coefficient manipulation matrix for [`mpoly_def_int`][lmlib.polynomial.poly.mpoly_def_int]
- [`mpoly_def_int_expos`][lmlib.polynomial.poly.mpoly_def_int_expos] — Exponent vectors for [`mpoly_def_int`][lmlib.polynomial.poly.mpoly_def_int]
- [`mpoly_substitute`][lmlib.polynomial.poly.mpoly_substitute] — Substituting a variable of a multivariate polynomial by a constant
- [`mpoly_substitute_coef`][lmlib.polynomial.poly.mpoly_substitute_coef] — Coefficient vector for [`mpoly_def_int`][lmlib.polynomial.poly.mpoly_def_int]
- [`mpoly_substitute_coef_L`][lmlib.polynomial.poly.mpoly_substitute_coef_L] — Coefficient manipulation matrix for [`mpoly_def_int`][lmlib.polynomial.poly.mpoly_def_int]
- [`mpoly_substitute_expos`][lmlib.polynomial.poly.mpoly_substitute_expos] — Exponent vectors for [`mpoly_substitute`][lmlib.polynomial.poly.mpoly_substitute]
- [`mpoly_dilate_ind`][lmlib.polynomial.poly.mpoly_dilate_ind] — $\alpha^{mathsf{T}}(xy)^q = (\Delta_Q\alpha)^{mathsf{T}}(x^q \otimes y^q)$
- [`mpoly_dilate_ind_coefs`][lmlib.polynomial.poly.mpoly_dilate_ind_coefs] — Coefficient vectros for [`mpoly_dilate`][lmlib.polynomial.poly.mpoly_dilate]
- [`mpoly_dilate_ind_coef_L`][lmlib.polynomial.poly.mpoly_dilate_ind_coef_L] — Coefficient manipulation matrix for [`mpoly_dilate`][lmlib.polynomial.poly.mpoly_dilate]
- [`mpoly_dilate_ind_expos`][lmlib.polynomial.poly.mpoly_dilate_ind_expos] — Exponent vectors for [`mpoly_dilate`][lmlib.polynomial.poly.mpoly_dilate]
- [`mpoly_dilate`][lmlib.polynomial.poly.mpoly_dilate] — Dilate a multivariate polynomial by a constant eta
- [`mpoly_dilate_coefs`][lmlib.polynomial.poly.mpoly_dilate_coefs] — Coefficient vectros for [`mpoly_dilate`][lmlib.polynomial.poly.mpoly_dilate]
- [`mpoly_dilate_coef_L`][lmlib.polynomial.poly.mpoly_dilate_coef_L] — Coefficient manipulation matrix for [`mpoly_dilate`][lmlib.polynomial.poly.mpoly_dilate]
- [`mpoly_dilate_expos`][lmlib.polynomial.poly.mpoly_dilate_expos] — Exponent vectors for [`mpoly_dilate`][lmlib.polynomial.poly.mpoly_dilate]
- [`kron_sequence`][lmlib.polynomial.poly.kron_sequence] — Kroneker product of a sequence of matrices
- [`extend_basis`][lmlib.polynomial.poly.extend_basis] — (DEV) Extending the basis of a uni- or multi-variate polynomial by new variables
- [`permutation_matrix`][lmlib.polynomial.poly.permutation_matrix] — Returns permutation matrix
- [`permutation_matrix_square`][lmlib.polynomial.poly.permutation_matrix_square] — Returns permutation matrix for square matrices A and B
- [`commutation_matrix`][lmlib.polynomial.poly.commutation_matrix] — Returns commutation matrix
- [`remove_redundancy`][lmlib.polynomial.poly.remove_redundancy] — Returns the exponent vector without redundancy and the matrix to add up the coefficients of the same exponent.
- [`mpoly_remove_redundancy`][lmlib.polynomial.poly.mpoly_remove_redundancy] — Returns the tuple of exponent vectors without redundancy and
- [`mpoly_transformation_coef_L`][lmlib.polynomial.poly.mpoly_transformation_coef_L] — transformation of a uniform polynomial into the form
- [`mpoly_transformation_expos`][lmlib.polynomial.poly.mpoly_transformation_expos]
- [`mpoly_extend_coef_L`][lmlib.polynomial.poly.mpoly_extend_coef_L] — Extends the polynomial by additional variables without changing its value.

## Sum of Polynomials

::: lmlib.polynomial.poly.poly_sum
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_sum_coef
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_sum_coef_Ls
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_sum_expo
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_sum_expo_Ms
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Product of Polynomials

::: lmlib.polynomial.poly.poly_prod
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_prod_coef
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_prod_expo_Ms
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_prod_expo
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Square of Polynomials

::: lmlib.polynomial.poly.poly_square
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_square_coef
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_square_expo_M
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_square_expo
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Shift of Polynomials

::: lmlib.polynomial.poly.poly_shift
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_shift_coef
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_shift_coef_L
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_shift_expo
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Dilation of Polynomials

::: lmlib.polynomial.poly.poly_dilation
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_dilation_coef
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_dilation_coef_L
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Integration of Polynomials

::: lmlib.polynomial.poly.poly_int
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_int_coef
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_int_coef_L
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_int_expo
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Differentiation of Polynomials

::: lmlib.polynomial.poly.poly_diff
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_diff_coef
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_diff_coef_L
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.poly_diff_expo
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Addition of Multivariate Polynomials

::: lmlib.polynomial.poly.mpoly_add
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_add_coefs
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_add_expos
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Multiplication of Multivariate Polynomials

::: lmlib.polynomial.poly.mpoly_multiply
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Product of Multivariate Polynomials

::: lmlib.polynomial.poly.mpoly_prod
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Square of Multivariate Polynomials

::: lmlib.polynomial.poly.mpoly_square
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_square_coef_L
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_square_coef
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_square_expos
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_square_expo_Ms
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Shift of Multivariate Polynomials

::: lmlib.polynomial.poly.mpoly_shift
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_shift_coef
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_shift_coef_L
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_shift_expos
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Integration of Multivariate Polynomials

::: lmlib.polynomial.poly.mpoly_int
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_int_coef
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_int_coef_L
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_int_expos
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Differentiation of Multivariate Polynomials

::: lmlib.polynomial.poly.mpoly_diff
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_diff_coef
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_diff_coef_L
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_diff_expos
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Definite Integration of Multivariate Polynomials

::: lmlib.polynomial.poly.mpoly_def_int
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_def_int_coef
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_def_int_coef_L
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_def_int_expos
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Substitution of Multivariate Polynomials

::: lmlib.polynomial.poly.mpoly_substitute
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_substitute_coef
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_substitute_coef_L
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_substitute_expos
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Independent Dilation of Multivariate Polynomials

::: lmlib.polynomial.poly.mpoly_dilate_ind
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_dilate_ind_coefs
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_dilate_ind_coef_L
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_dilate_ind_expos
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Dilation of Multivariate Polynomials

::: lmlib.polynomial.poly.mpoly_dilate
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_dilate_coefs
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_dilate_coef_L
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_dilate_expos
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Sequences, Matrices and Basis Utilities

::: lmlib.polynomial.poly.kron_sequence
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.extend_basis
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.permutation_matrix
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.permutation_matrix_square
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.commutation_matrix
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.remove_redundancy
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_remove_redundancy
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_transformation_coef_L
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_transformation_expos
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: lmlib.polynomial.poly.mpoly_extend_coef_L
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
