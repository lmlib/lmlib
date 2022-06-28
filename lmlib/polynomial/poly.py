"This module provides a calculus for uni- and multivariate polynomials using the vector exponent notation."
from abc import ABC, abstractmethod

import numpy as np
from scipy.special import comb
from scipy.sparse import kron, csr_matrix, eye, spdiags

__all__ = ['kron_sequence', 'MPoly', 'Poly',
           'poly_sum', 'poly_sum_coef', 'poly_sum_coef_Ls', 'poly_sum_expo', 'poly_sum_expo_Ms',
           'poly_prod', 'poly_prod_coef', 'poly_prod_expo_Ms', 'poly_prod_expo',
           'poly_square', 'poly_square_coef', 'poly_square_expo', 'poly_square_expo_M',
           'poly_shift', 'poly_shift_coef', 'poly_shift_coef_L', 'poly_shift_expo',
           'poly_dilation', 'poly_dilation_coef', 'poly_dilation_coef_L',
           'poly_int', 'poly_int_coef', 'poly_int_coef_L', 'poly_int_expo',
           'poly_diff', 'poly_diff_coef', 'poly_diff_coef_L', 'poly_diff_expo',

           'mpoly_add', 'mpoly_add_coefs', 'mpoly_add_expos',
           'mpoly_multiply',
           'mpoly_prod',
           'mpoly_square', 'mpoly_square_coef', 'mpoly_square_coef_L', 'mpoly_square_expos', 'mpoly_square_expo_Ms',
           'mpoly_shift', 'mpoly_shift_coef', 'mpoly_shift_coef_L', 'mpoly_shift_expos',
           'mpoly_int', 'mpoly_int_coef', 'mpoly_int_coef_L', 'mpoly_int_expos',
           'mpoly_diff', 'mpoly_diff_coef', 'mpoly_diff_coef_L', 'mpoly_diff_expos',
           'mpoly_def_int', 'mpoly_def_int_coef', 'mpoly_def_int_coef_L', 'mpoly_def_int_expos',
           'mpoly_substitute', 'mpoly_substitute_coef', 'mpoly_substitute_coef_L', 'mpoly_substitute_expos',
           'mpoly_dilate', 'mpoly_dilate_coefs', 'mpoly_dilate_coef_L', 'mpoly_dilate_expos',
           'mpoly_dilate_ind', 'mpoly_dilate_ind_coefs', 'mpoly_dilate_ind_coef_L', 'mpoly_dilate_ind_expos',

           'extend_basis',
           'permutation_matrix_square',
           'permutation_matrix',
           'commutation_matrix',
           'remove_redundancy', 'mpoly_remove_redundancy',
           'mpoly_transformation_coef_L', 'mpoly_transformation_expos',
           'mpoly_extend_coef_L'
           ]


def _check_input_variables(variables, variable_count):
    if len(variables) != variable_count:
        raise ValueError(f'Number of variables doesn\'t match with polynomial. '
                         f'Input variables count: {len(variables)}, Expected variables count = {variable_count}')

    variable_shape = np.shape(variables[0])
    if any(np.shape(variable) != variable_shape for variable in variables):
        raise ValueError('Inconsistent shape of dependent variable in \'variables\'')


def kron_sequence(arrays, sparse=False):
    r"""
    Kroneker product of a sequence of matrices

    .. math::
        B = A_0 \otimes A_1 \otimes A_2 \otimes A_3 \otimes \dots A_N

    Parameters
    ----------
    arrays :  tuple of array_like, list of array_like
        sequence of matrices
    sparse : bool, optional
        if true the result calculated and returned as a sparse matrix

    Returns
    -------
    prod : array_like, int
        Kronecker product of sequence of matrices

    Examples
    --------
    >>> a1 = np.array([[2, 3, 4], [0, 1, 4]])
    >>> a2 = np.array([[9, 0], [1, 1]])
    >>> a3 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> b = kron_sequence((a1, a2, a3))
    >>> print(b)
    [[  0  18  36   0   0   0   0  27  54   0   0   0   0  36  72   0   0   0]
     [ 54  72  90   0   0   0  81 108 135   0   0   0 108 144 180   0   0   0]
     [108 126 144   0   0   0 162 189 216   0   0   0 216 252 288   0   0   0]
     [  0   2   4   0   2   4   0   3   6   0   3   6   0   4   8   0   4   8]
     [  6   8  10   6   8  10   9  12  15   9  12  15  12  16  20  12  16  20]
     [ 12  14  16  12  14  16  18  21  24  18  21  24  24  28  32  24  28  32]
     [  0   0   0   0   0   0   0   9  18   0   0   0   0  36  72   0   0   0]
     [  0   0   0   0   0   0  27  36  45   0   0   0 108 144 180   0   0   0]
     [  0   0   0   0   0   0  54  63  72   0   0   0 216 252 288   0   0   0]
     [  0   0   0   0   0   0   0   1   2   0   1   2   0   4   8   0   4   8]
     [  0   0   0   0   0   0   3   4   5   3   4   5  12  16  20  12  16  20]
     [  0   0   0   0   0   0   6   7   8   6   7   8  24  28  32  24  28  32]]
    """

    if len(arrays) == 0:
        return 1
    if len(arrays) == 1:
        return arrays[0]
    if len(arrays) > 1:
        if not sparse:
            return np.kron(arrays[0], kron_sequence(arrays[1::], sparse=False))
        return kron(arrays[0], kron_sequence(arrays[1::], sparse=True))


class _PolyBase(ABC):
    """
    Abstract baseclass for Model Container
    """

    @abstractmethod
    def __init__(self, coefs, expos):
        """
        Constructor method
        """
        self.expos = expos
        if len(coefs) == self.variable_count:
            self.coefs_fac = coefs
            self._expand_coefs()
        else:
            self.coefs = coefs
            self.coefs_fac = None
        self._check_dimensions()

    def __str__(self):
        return '{}, {}'.format(self.coefs, self.expos)

    @property
    def coefs(self):
        """tuple of :class:`~numpy.ndarray` : Coefficient vector (i.e., not factorized)"""
        return self._coefs

    @coefs.setter
    def coefs(self, coefs):
        if not isinstance(coefs, tuple):
            raise TypeError('Coefficient vector(s) \'coefs\' is not of type tuple.')
        if any(not isinstance(coef, np.ndarray) for coef in coefs):
            raise TypeError('Coefficient vector in \'coefs\' is not of type np.ndarray.')
        self._coefs = coefs

    @property
    def expos(self):
        """tuple of :class:`~numpy.ndarray` : Exponent vectors"""
        return self._expos

    @expos.setter
    def expos(self, expos):
        if not isinstance(expos, tuple):
            raise TypeError('Exponent vector(s) \'expos\' is not of type tuple.')
        if any(not isinstance(expo, np.ndarray) for expo in expos):
            raise TypeError('Exponent vector in \'expos\' is not of type np.ndarray.')

        self._expos = expos
        self._variable_count = len(expos)

    @property
    def coefs_fac(self):
        """:class:`~numpy.ndarray, None` : Factorized coefficient vectors"""
        return self._coefs_fac

    @coefs_fac.setter
    def coefs_fac(self, coefs_fac):
        self._coefs_fac = coefs_fac

    def _expand_coefs(self):
        tmp = 1
        for coefficient_vector in self.coefs_fac:
            tmp = np.kron(tmp, coefficient_vector)
        self.coefs = (tmp,)

    @property
    def variable_count(self):
        """int : Number of dependent variables"""
        return self._variable_count

    def _check_dimensions(self):
        n_var_coefficients = np.prod([coefficient_vector.shape[0] for coefficient_vector in self.coefs])
        n_var_exponents = np.prod([exponent_vector.shape[0] for exponent_vector in self.expos])
        if n_var_coefficients != n_var_exponents:
            raise ValueError('Number of coefficients elements and number of exponent elements doesn\'t match.')
        if self.coefs_fac is not None:
            n_var_coefficients_fac = np.prod([coefficient_vector.shape[0] for coefficient_vector in self.coefs_fac])
            if n_var_coefficients_fac != n_var_exponents:
                raise ValueError(
                    'Number of factorized coefficients elements and number of exponent elements doesn\'t match.')

    def eval(self, variables):
        """
        Evaluates the polynomial for given values (variables)

        Parameters
        ----------
        variables : tuple
            Dependent variables of a polynomial.
            Each element in variables has the same shape which is also the output shape.

        Returns
        -------
        out : :class:`~numpy.ndarray`
            Output of evaluated polynomial.
            Shape is identical as a dependent variable

        Example
        -------

        >>> # evaluate bivariate polynomial at multiple positions (x=.1 ... .4, y=.5)
        >>> l = MPoly(([.2, .7], [1.3, 1.4, -.9],), ([0, 1], [0, 1, 2]))
        >>> l.eval(([.1, .2, .3, .4], [.5, .5, .5, .5]))
        array([0.47925, 0.6035, 0.72775, 0.852])

        """

        _check_input_variables(variables, self.variable_count)

        variables = np.asarray(variables)
        out = np.empty_like(variables[0], dtype=float)
        it = np.nditer(out, flags=['multi_index'])
        while not it.finished:
            out[it.multi_index] = self._eval_scalar([variable[it.multi_index] for variable in variables])
            it.iternext()
        return out

    def _eval_scalar(self, variables):
        kron_var_expo = 1
        for variable, expo in zip(variables, self._expos):
            kron_var_expo = np.kron(kron_var_expo, np.power(variable, expo))
        return np.dot(self.coefs, kron_var_expo)



class MPoly(_PolyBase):
    r"""
    Multivariate polynomials :math:`{\tilde{\alpha}}^\mathsf{T} (x^q \otimes y^r)`, or with *factorized* coefficient
    vector  :math:`(\alpha \otimes \beta )^\mathsf{T} (x^q \otimes y^r)`.

    This polynomial class is for multivariate polynomials in vector exponent notation, see [Wildhaber2019]_, Chapter 6.

    Such a multivariate polynomial is in general given by

    .. math::

        p(x) = \tilde{\alpha}^\mathsf{T}(x^q \otimes y^r) \ ,

    where :math:`\tilde{\alpha} \in \mathbb{R}^{Q \times R}` is the coefficient vectors,
    :math:`q \in \mathbb{Z}_{\geq 0}^Q` and :math:`r \in \mathbb{Z}_{\geq 0}^R` the exponent vectors,
    and :math:`x \in \mathbb{R}` and :math:`y \in \mathbb{R}` the independent variables.

    As a special case,
    if the coefficient vector is in the form of a Kronecker product, i.e.,

    .. math::

        p(x) = (\alpha \otimes \beta)^\mathsf{T}(x^q \otimes y^r) \ ,

    where :math:`\alpha \in \mathbb{R}^Q` and :math:`\beta \in \mathbb{R}^R` are coefficient vectors,
    we denote a polynomial as **factorized**.
    This form often leads to algebraic simplifications (if it exists).


    Examples
    --------

    >>> # Bivariate (x,y) polynomial with factorized coefficients ([.2,.7],[-1.0,2.0,.1]) and terms x^1, x^2, y^1, y^2, y^3, and cross terms
    >>> l= MPoly(([.2,.7],[-1.0,2.0,.1]),([1,2],[1,2,3]))
    >>> l.coefs # gets coefficients
    (array([0.2, 0.7]), array([-1. ,  2. ,  0.1]))
    >>> l.coef_fac # gets factorized coefficients (if available)
    array([-0.2 ,  0.4 ,  0.02, -0.7 ,  1.4 ,  0.07])
    >>> l.eval([.3,.7])  # evaluating polynomial for x=.3 and y=.7
    array(0.0386589)

    >>> # Bivariate (x,y) polynomial with non-factorized coefficients ([.2,.7,1.3,1.4,.2,-1.6]) and terms x^1, x^2, y^1, y^2, y^3, and cross terms
    >>> l= MPoly(([.2,.7,1.3,1.4,.2,-1.6],),([1,2],[1,2,3]))
    >>> l.eval([.3,.7])  # evaluating polynomial for x=.3 and y=.7
    array(0.326298)


    Parameters
    ----------
    coefs : tuple of array_like
        Set of coefficient vector(s)
    expos : tuple of array_like
        Set of exponent vector(s)

    """

    def __init__(self, coefs, expos):
        """
        Constructor method
        """
        coefs = tuple(np.asarray(coef) for coef in coefs)
        expos = tuple(np.asarray(expo) for expo in expos)
        super(MPoly, self).__init__(coefs, expos)


class Poly(_PolyBase):
    r"""
    Univariate polyonimial in vector exponent notation: :math:`\alpha^\mathsf{T} x^q`

    Polynomial class for uni-variate polynomials in vector exponent notation, [Wildhaber2019]_, see Chapter 6.

    Such a polynomial `p(x)` in `x` is defined as

    .. math::

        p(x) &= \alpha^\mathsf{T}x^q = \begin{bmatrix}a_0& a_1& \cdots& a_{Q-1}\end{bmatrix}
             \begin{bmatrix}x^{q_0}\\ x^{q_1}\\ \vdots\\ x^{q_{Q-1}}\end{bmatrix}\\
             &= a_0 x^{q_0} + a_1 x^{q_1}+ \dots + a_{Q-1} x^{q_{Q-1}} \ ,

    with coefficient vector :math:`\alpha \in \mathbb{R}^Q`,
    exponent vector :math:`q \in \mathbb{Z}_{\geq 0}^Q`,
    and function variable :math:`x \in \mathbb{R}`.

    Parameters
    ----------
    coef : array_like, shape=(Q)
        Coefficient vector
    expo : array_like, shape=(Q)
        Exponent vector


    |def_Q|


    Examples
    --------
    >>> import lmlib as lm
    >>> p = Poly([0, 0.2, 3], [0, 1, 2])
    >>> print(p)

    """

    def __init__(self, coef, expo):
        """
        Constructor method
        """
        coefs = (np.asarray(coef),)
        expos = (np.asarray(expo),)
        super(Poly, self).__init__(coefs, expos)

    @property
    def coef(self):
        """:class:`~numpy.ndarray` : Coefficient vector :math:`\\alpha`"""
        return self._coefs[0]

    @property
    def expo(self):
        """:class:`~numpy.ndarray` : Exponent vector :math:`q`"""
        return self._expos[0]

    @property
    def Q(self):
        """int : Number of elements in exponent vector :math:`Q`"""
        return self.expo.shape[0]

    def eval(self, variable):
        """
        Evaluates the polynomial

        Parameters
        ----------
        variable : array_like, scalar
            Dependent variables of a polynomial.

        Returns
        -------
        out : :class:`~numpy.ndarray`
            Output of evaluated polynomial.
            Shape is identical as a dependent variable
        """
        return super(Poly, self).eval((variable,))


def poly_sum(polys):
    r""" :math:`\alpha^\mathsf{T} x^q + \dots + \beta^\mathsf{T} x^r = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}`

    Sum of univariate polynomials ``Poly(alpha,q),... , Poly(beta,r)``, all of common variable *x*



    Parameters
    ----------
    polys : tuple of Poly
        ``(Poly(alpha,q),... , Poly(beta,r))``, list of polynomials to be summed


    Returns
    -------
    out : Poly
        ``Poly(alpha_tilde, q_tilde)``

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.4)

    """
    return Poly(poly_sum_coef(polys), poly_sum_expo([poly.expo for poly in polys]))


def poly_sum_coef(polys):
    r""" :math:`\alpha^\mathsf{T} x^q + \dots + \beta^\mathsf{T} x^r = \color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}`

    Coefficient vector :math:`\tilde{q}` to sum of univariate polynomials ``polys``, all of common variable *x*

    Parameters
    ----------
    polys : tuple of Poly
        ``(Poly(alpha,q),... , Poly(beta,r))``, list of polynomials to be summed

    Returns
    -------
    coef : :class:`~numpy.ndarray`
        ``alpha_tilde`` - Coefficient vector :math:`\tilde{\alpha}`

    Note
    ----
    To get :math:`\tilde{q}`, see :func:`poly_sum_expo`.


    References
    ----------
    [Wildhaber2019]_ (Eq. 6.4)

    """
    Ls = poly_sum_coef_Ls([poly.expo for poly in polys])
    return np.sum([np.dot(L, coef) for L, coef in zip(Ls, [poly.coef for poly in polys])], axis=0)


def poly_sum_coef_Ls(expos):
    r""" :math:`\alpha^\mathsf{T} x^q + \dots + \beta^\mathsf{T} x^r = (\color{blue}{\Lambda_1} \alpha + \dots + \color{blue}{\Lambda_N}\beta)^\mathsf{T} x^{\tilde{q}}`

    Exponent manipulation matrices :math:`\Lambda_1, ...., Lambda_N` to  sum univariate polynomials ``polys``,
    all of common variable *x*

    Parameters
    ----------
    expos : tuple of array_like
         ``(q, ..., r)``, list of exponent vectors of polynomials to be summed

    Returns
    -------
    Ls : list of :class:`~numpy.ndarray`
        ``(Lambda_1, ..., Lambda_N)``,
        Coefficient manipulation matrices,
        see also :func:`poly_sum`


    Note
    ----
    To get :math:`\tilde{q}`, see :func:`poly_sum_expo`.

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.4)
    """
    indices_or_sections = np.cumsum([len(expo) for expo in expos[0:-1]])
    Q = sum(len(expo) for expo in expos)
    return np.split(np.eye(Q), indices_or_sections, axis=1)


def poly_sum_expo(expos):
    r""" :math:`\alpha^\mathsf{T} x^q + \dots + \beta^\mathsf{T} x^r = \tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}`

    Exponent vector :math:`\tilde{1}` of  sum  of univariate polynomials with exponent vectors ``expos``, all of common variable *x*

    Parameters
    ----------
    expos : tuple of array_like
        ``(q, ..., r)``, list of exponent vectors of polynomials to be summed

    Returns
    -------
    expo :class:`~numpy.ndarray`
        ``q_tilde``,
        exponent vector :math:`\tilde{q}`

    Note
    ----
    To get :math:`\tilde{\alpha}`, see :func:`poly_sum_coef`.



    References
    ----------
    [Wildhaber2019]_ (Eq. 6.4)
    """
    Ms = poly_sum_coef_Ls(expos)
    return np.sum([np.dot(M, expo) for M, expo in zip(Ms, expos)], axis=0)


def poly_sum_expo_Ms(expos):
    r""" :math:`\alpha^\mathsf{T} x^q + \dots + \beta^\mathsf{T} x^r = \tilde{\alpha}^\mathsf{T} x^{\color{blue}{M_1} q + \dots +\color{blue}{M_N} r}`

    Exponent manipulation matrices :math:`M_1, ... , M_N` to sum univariate polynomials with exponent vectors ``expos``, all of common variable *x*


    Parameters
    ----------
    expos : tuple of array_like
        ``(q, ..., r)``, list of exponent vectors of polynomials to be summed

    Returns
    -------
    Ms : list of :class:`~numpy.ndarray`
        ``(M_1, ..., M_N)``,
        list of exponent manipulation matrices,
        see also :func:`poly_sum`.

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.4)
    """
    indices_or_sections = np.cumsum([len(expo) for expo in expos[0:-1]])
    Q = sum(len(expo) for expo in expos)
    return np.split(np.eye(Q), indices_or_sections, axis=1)


def poly_prod(polys):
    r""" :math:`\alpha^\mathsf{T} x^q \cdot \beta^\mathsf{T} x^q = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}`

    Product of two univariate polynomials of common variable `x`

    Parameters
    ----------
    polys : tuple of Poly
        ``(poly_1, poly_2)``,
        two polynomials to be multiplied

    Returns
    -------
    out : Poly
        ``poly_tilde``, product as polynomial :math:`\tilde{\alpha}^\mathsf{T} x^\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.14)

    """
    assert len(polys) <= 2, 'Not yet implemented. Only two polynomials allowed'
    coef = poly_prod_coef(polys)
    expo = poly_prod_expo([poly.expo for poly in polys])
    return Poly(coef, expo)


def poly_prod_coef(polys):
    r""" :math:`\alpha^\mathsf{T} x^q \cdot \beta^\mathsf{T} x^r = \color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}`

    Coefficient vector :math:`\tilde{\alpha}` of product of two univariate polynomials of common variable `x`


    Parameters
    ----------
    polys : tuple of Poly
        ``(poly_1, poly_2)``,
        two polynomials to be multiplied

    Returns
    -------
    coef :class:`~numpy.ndarray`
        ``alpha_tilde``,  coefficient vector :math:`\tilde{\alpha}` of product polynomial :math:`\tilde{\alpha}^\mathsf{T} x^\tilde{q}`, see :func:`poly_prod`


    References
    ----------
    [Wildhaber2019]_ (Eq. 6.12)

    """
    return np.kron(polys[0].coef, polys[1].coef)


def poly_prod_expo_Ms(expos):
    r""" :math:`\alpha^\mathsf{T} x^q \cdot \beta^\mathsf{T} x^r = \tilde{\alpha}^\mathsf{T} x^{\color{blue}{M_1} q + \color{blue}{M_2} r}`

    Exponent manipulation matrices :math:`M_1, M_2` of product of two univariate polynomials of common variable `x`

    Parameters
    ----------
    expos : tuple of array_like
        ``(q, r)``,
        exponent vectors

    Returns
    -------
    Ms : list of :class:`~numpy.ndarray`
        ``(M_1, ..., M_N)``,
        list of exponent manipulation matrices
        for the two polynomial exponent vectors, i.e., the new exponent vector results from :math:`\tilde{q} = M_1 q + M_2 r`.
        See :func:`poly_prod`.


    References
    ----------
    [Wildhaber2019]_ (Eq. 6.16)

    """
    Q = len(expos[0])
    R = len(expos[1])
    return np.kron(np.identity(Q), np.ones((R, 1))), np.kron(np.ones((Q, 1)), np.identity(R))


def poly_prod_expo(expos):
    r""" :math:`\alpha^\mathsf{T} x^q \cdot \beta^\mathsf{T} x^r = \tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}`

    Exponent vector :math:`\tilde{q}` of product of two univariate polynomials of common variable `x`

    Parameters
    ----------
    expos : tuple of arraylike
        ``(q, r)``,
        exponent vectors of the two polynomials

    Returns
    -------
    coef :class:`~numpy.ndarray`
        ``q_tilde``,  exponent vector :math:`\tilde{q}` of product polynomial :math:`\tilde{\alpha}^\mathsf{T} x^\tilde{q}`, see :func:`poly_prod`


    References
    ----------
    [Wildhaber2019]_ (Eq. 6.16)

    """
    M1, M2 = poly_prod_expo_Ms(expos)
    return np.add(np.dot(M1, expos[0]), np.dot(M2, expos[1]))


def poly_square(poly):
    r""" :math:`(\alpha^\mathsf{T} x^q)^2 = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}`

    Square of a univariate polynomial

    Parameters
    ----------
    poly : :class:`Poly`
        ``poly``,
        polynomial to be squared

    Returns
    -------
    out : :class:`Poly`
        squared polynomial

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.11)

    """
    return poly_prod((poly, poly))


def poly_square_coef(poly):
    r""" :math:`\color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}`

    Coefficient vector :math:`\tilde{\alpha}` of squared polynomial, see :func:`poly_square`

    Parameters
    ----------
    poly : Poly
        ``poly``,
        polynomial to be squared

    Returns
    -------
    coef :class:`~numpy.ndarray`,
        ``alpha_tilde`` Coefficient vector :math:`\tilde{\alpha}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.12)

    """
    return np.kron(poly.coef, poly.coef)


def poly_square_expo_M(expo):
    r""" :math:`\tilde{\alpha}^\mathsf{T} x^{\color{blue}{M} q}`

    Exponent manipulation matrix :math:`M` of squared polynomial, see :func:`poly_square`

    Parameters
    ----------
    expo : array_like,
        `q`,
        exponent vector :math:`q`

    Returns
    -------
    M : :class:`~numpy.ndarray`,
        ``M``,
        exponent manipulation matrix :math:`M`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.10, Eq. 6.13)

    """
    return np.add(*poly_prod_expo_Ms((expo, expo)))


def poly_square_expo(expo):
    r""" :math:`\tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}`

    Exponent vector :math:`\tilde{q}` of squared polynomial, see :func:`poly_square`.

    Parameters
    ----------
    expo : array_like,
        ``q``,
        exponent vector :math:`q`

    Returns
    -------
    out :class:`~numpy.ndarray`,
        ``q_tilde``,
        exponent vector :math:`\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.12)

    """
    return np.dot(np.add(*poly_prod_expo_Ms((expo, expo))), expo)


def poly_shift(poly, gamma):
    r""" :math:`\alpha^\mathsf{T} (x+ \gamma)^q = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}`

    Shifting an univariate polynomial by constant value :math:`\gamma \in \mathbb{R}`

    Parameters
    ----------
    poly : :class:`Poly`
        polynomial to be shifted
    gamma : float
        ``gamma``,
        shift parameter :math:`\gamma`

    Returns
    -------
    out : :class:`Poly`
        shifted polynomial,
        :math:`\tilde{\alpha}^\mathsf{T} x^\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.28)

    """
    return Poly(coef=poly_shift_coef(poly, gamma), expo=poly_shift_expo(poly.expo))


def poly_shift_coef(poly, gamma):
    r""" :math:`\color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}`

    Coefficient vector of shifted polynomial, see :func:`poly_shift`

    Parameters
    ----------
    poly : :class:`Poly`
        polynomial to be shifted
    gamma : float
        ``gamma``,
        shift parameter :math:`\gamma`

    Returns
    -------
    coef : :class:`~numpy.ndarray`
        ``alpha_tilde``
        Coefficient vector :math:`\tilde{\alpha}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.29)

    """
    return np.dot(poly_shift_coef_L(poly.expo, gamma), poly.coef)


def poly_shift_coef_L(expo, gamma):
    r""" :math:`\color{blue}{\Lambda} \alpha^\mathsf{T} x^{\tilde{q}}`

    Coefficient manipulation :math:`\Lambda}` for shifted polynomial, see :func:`poly_shift`

    Parameters
    ----------
    expo : array_like
        ``q``,
        Exponent vector :math:`q`
    gamma : float
        ``gamma``,
        shift parameter :math:`\gamma`

    Returns
    -------
    L : :class:`~numpy.ndarray`
        ``L``,
        coefficient manipulation matrices :math:`\Lambda`.

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.32)

    """
    Q = len(expo)
    q_tilde = poly_shift_expo(expo)

    L = np.zeros((len(q_tilde), Q))
    for i, qi in enumerate(expo):
        L[:, i] = comb(qi, q_tilde) * np.power(gamma, np.subtract(qi, q_tilde).clip(min=0))
    return L


def poly_shift_expo(expo):
    r""" :math:`\tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}`

    Exponent vector :math:`\tilde{q}` for shifted polynomial, see :func:`poly_shift`

    Parameters
    ----------
    expo : array_like
        ``q``,
        Exponent vector :math:`q`

    Returns
    -------
    expo : :class:`~numpy.ndarray`
        ``q_tilde``,
        exponent vector :math:`\tilde{q}`.

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.30)

    """
    return np.arange(max(expo) + 1)


def poly_dilation(poly, eta):
    r""" :math:`\alpha^\mathsf{T} (\eta x)^q = \color{blue}{\tilde{\alpha}^\mathsf{T} x^q}`

    Dilation of a polynomial by scaling `x` by constant value :math:`\eta \in \mathbb{R}`

    Parameters
    ----------
    poly : :class:`Poly`
        polynomial to be scaled

    eta : float
        ``eta``,
        dilation factor :math:`\eta`


    Returns
    -------
    out : :class:`Poly`
        dilated polynomial,
        :math:`\tilde{\alpha}^\mathsf{T} x^\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.33)
    """
    return Poly(coef=poly_dilation_coef(poly, eta), expo=poly.expo)


def poly_dilation_coef(poly, eta):
    r""" :math:`\alpha^\mathsf{T} (\eta x)^q =\color{blue}{\tilde{\alpha}}^\mathsf{T} x^q`


    Coefficient vector :math:`\tilde{\alpha}` of dilated polynomial by scaling `x` by constant value :math:`\eta \in \mathbb{R}`, see  :func:`poly_dilation`.

    Parameters
    ----------
    poly : :class:`Poly`
        polynomial to be scaled

    eta : float
        ``eta``,
        dilation factor :math:`\eta`


    Returns
    -------
    coef :class:`~numpy.ndarray`
        Coefficient vector :math:`\tilde{\alpha}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.34)

    """
    return np.dot(poly_dilation_coef_L(poly.expo, eta), poly.coef)


def poly_dilation_coef_L(expo, eta):
    r""" :math:`\alpha^\mathsf{T} (\eta x)^q =\color{blue}{\Lambda} \alpha^\mathsf{T} x^{q}`

    Coefficient manipulation matrix :math:`\tilde{\Lambda}` to dilated polynomial by scaling `x` by constant value :math:`\eta \in \mathbb{R}`, see  :func:`poly_dilation`.

    Parameters
    ----------
    expo : array_like
        ``q``,
        Exponent vector :math:`q`
    eta : float
        ``eta``,
        dilation factor :math:`\eta`

    Returns
    -------
    L : :class:`~numpy.ndarray`
        ``L``,
        Coefficient Manipulation Matrices :math:`\Lambda`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.35)

    """
    return np.diag(np.power(eta, expo))


def poly_int(poly):
    r""" :math:`\int \big(\alpha^{\mathsf{T}}x^q\big) dx = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}`

    Indefinite integral of a polynomial

    Parameters
    ----------
    poly : :class:`Poly`
        polynomial to be integrated

    Returns
    -------
    out : :class:`Poly`
        polynomial,
        :math:`\tilde{\alpha}^\mathsf{T} x^\tilde{q}`


    References
    ----------
    [Wildhaber2019]_ (Eq. 6.17)
    """
    return Poly(poly_int_coef(poly), poly_int_expo(poly.expo))


def poly_int_coef(poly):
    r""" :math:`\int \big(\alpha^{\mathsf{T}}x^q\big) dx  = \color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}`

    Coefficient vector :math:`\tilde{\alpha}` of indefinite integral of a polynomial, see :func:`poly_int`

    Parameters
    ----------
    poly : :class:`Poly`
        polynomial to be integrated


    Returns
    -------
    coef : :class:`~numpy.ndarray`,
        ``alpha_tilde``,
        Coefficient vector :math:`\tilde{\alpha}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.18)
    """
    return mpoly_int_coef(poly, 0)


def poly_int_coef_L(expo):
    r""" :math:`\int \big(\alpha^{\mathsf{T}}x^q\big) dx  = \color{blue}{\Lambda} \alpha^\mathsf{T} x^{\tilde{q}}`

    Coefficient manipulation matrix :math:`\Lambda` of indefinite integral of a polynomial, see  :func:`poly_int`.

    Parameters
    ----------
    expo : array_like
        ``q``,
        Exponent vector :math:`q`

    Returns
    -------
    L : :class:`~numpy.ndarray`
        ``L``,
        coefficient Manipulation Matrices :math:`\Lambda`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.20-21)
    """
    return mpoly_int_coef_L((expo,), 0)


def poly_int_expo(expo):
    r""" :math:`\int \big(\alpha^{\mathsf{T}}x^q\big) dx  = \tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}`

    Exponent vector :math:`\tilde{q}` of indefinite integral of a polynomial, see :func:`poly_int`.

    Parameters
    ----------
    expo : array_like
        ``q``,
        exponent vector :math:`q`

    Returns
    -------
    expo : :class:`~numpy.ndarray`
        ``q_tilde``,
        exponent vector :math:`\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.19)
    """
    return mpoly_int_expos((expo,), 0)[0]


def poly_diff(poly):
    r""" :math:`\frac{d}{dx} \big(\alpha^{\mathsf{T}}x^q\big) = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}`

    Derivation of a polynomial

    Parameters
    ----------
    poly : Poly
        polynomial for derivation

    Returns
    -------
    out : Poly
        derivative
        :math:`\tilde{\alpha}^\mathsf{T} x^\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.24)
    """
    return Poly(poly_diff_coef(poly), poly_diff_expo(poly.expo))


def poly_diff_coef(poly):
    r""" :math:`\frac{d}{dx} \big(\alpha^{\mathsf{T}}x^q\big) =\color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}`

    Coefficient vector :math:`\tilde{\alpha}` of polynomial derivation, see :func:`poly_diff`

    Parameters
    ----------
    poly : Poly
        polynomial for derivation


    Returns
    -------
    coef : :class:`~numpy.ndarray`
        ``alpha_tilde``,
        coefficient vector :math:`\tilde{\alpha}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.25)
    """
    return np.dot(poly_diff_coef_L(poly.expo), poly.coef)


def poly_diff_coef_L(expo):
    r""" :math:`\frac{d}{dx} \big(\alpha^{\mathsf{T}}x^q\big) =\color{blue}{\Lambda} \alpha^\mathsf{T} x^{\tilde{q}}`

    Coefficient manipulation matrix :math:`\Lambda` of polynomial derivation, see :func:`poly_diff`

    Parameters
    ----------
    expo : array_like
        Exponent vector :math:`q`

    Returns
    -------
    L : :class:`~numpy.ndarray`
        ``L``,
        coefficient manipulation matrices :math:`\Lambda`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.27)
    """
    return np.diag(expo)


def poly_diff_expo(expo):
    r""" :math:`\frac{d}{dx} \big(\alpha^{\mathsf{T}}x^q\big) =\tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}`

    Exponent vector  :math:`\tilde{q}` of polynomial derivation, see :func:`poly_diff`

    Parameters
    ----------
    expo : array_like
        Exponent vector :math:`q`

    Returns
    -------
    expo : :class:`~numpy.ndarray`
        ``q_tilde``,
        exponent vector :math:`\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.26)
    """
    return np.subtract(expo, 1).clip(min=0)


def mpoly_int(mpoly, position):
    """
    Integral of a multivariate polynomial with respect to the scalar at a position

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.60 - 6.61)

    """
    coefs = (mpoly_int_coef(mpoly, position),)
    expos = mpoly_int_expos(mpoly.expos, position)
    return MPoly(coefs, expos)


def mpoly_int_coef(mpoly, position):
    """
    Coefficient vector for :func:`mpoly_int`

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.61)

    """
    return np.dot(mpoly_int_coef_L(mpoly.expos, position), mpoly.coefs[0])


def mpoly_int_coef_L(expos, position, sparse=False):
    """
    Coefficient manipulation matrix for :func:`mpoly_int`

    Parameters
    ----------
    expos : tuple of array_like
    position : int

    Returns
    -------
    L : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.61)

    """
    data = kron_sequence([np.ones((len(expo),)) if n != position else (1 / (expo + 1)) for n, expo in enumerate(expos)],
                         sparse=sparse)
    if sparse:
        return spdiags(data.toarray(), [0], data.shape[1], data.shape[1])
    return np.diag(kron_sequence(
        [np.ones((len(expo),)) if n != position else (1 / (expo + 1)) for n, expo in enumerate(expos)]))


def mpoly_int_expos(expos, position):
    """
    Exponent vectors for :func:`mpoly_int`

    Parameters
    ----------
    expos : tuple of array_like
    position : int

    Returns
    -------
    expos : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.61, Eq.6.19)

    """

    return expos[0:position] + (np.add(expos[position], 1),) + expos[position + 1::]


def mpoly_diff(mpoly, position, sparse=False):
    """
    Derivative of a multivariate polynomial with respect to the variable at a position

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    Todo: REF

    """
    coefs = (mpoly_diff_coef(mpoly, position, sparse),)
    expos = mpoly_diff_expos(mpoly.expos, position)
    return MPoly(coefs, expos)


def mpoly_diff_coef(mpoly, position, sparse=False):
    """
    Coefficient vector for :func:`mpoly_diff`

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    TODO: Ref

    """
    return np.dot(mpoly_diff_coef_L(mpoly.expos, position, sparse), mpoly.coefs[0])


def mpoly_diff_coef_L(expos, position, sparse=False):
    """
    Coefficient manipulation matrix for :func:`mpoly_diff`

    Parameters
    ----------
    expos : tuple of array_like
    position : int

    Returns
    -------
    L : :class:`~numpy.ndarray`

    References
    ----------
    TODO: Ref

    """
    if sparse:
        return kron_sequence(
            [eye(len(expo)) if n != position else poly_diff_coef_L(expo) for n, expo in enumerate(expos)], sparse)
    else:
        return kron_sequence(
            [np.eye(len(expo)) if n != position else poly_diff_coef_L(expo) for n, expo in enumerate(expos)])


def mpoly_diff_expos(expos, position):
    """
    Exponent vectors for :func:`mpoly_diff`

    Parameters
    ----------
    expos : tuple of array_like
    position : int

    Returns
    -------
    expos : tuple of :class:`~numpy.ndarray`

    References
    ----------
    TODO: Ref

    """

    return expos[0:position] + (poly_diff_expo(expos[position]),) + expos[position + 1::]


def mpoly_add(poly1, poly2):
    """
    Sum of two univariate polynomials different variables

    Parameters
    ----------
    poly1 : :class:`~lmlib.polynomial.poly.Poly`
    poly2 : :class:`~lmlib.polynomial.poly.Poly`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.37)

    """
    coefs = mpoly_add_coefs(poly1, poly2)
    expos = mpoly_add_expos(poly1, poly2)
    return MPoly(coefs=(np.sum(coefs, axis=0),), expos=expos)


def mpoly_add_coefs(poly1, poly2):
    """
    Coefficients for :func:`mpoly_add`

    Parameters
    ----------
    poly1 : :class:`~lmlib.polynomial.poly.Poly`
    poly2 : :class:`~lmlib.polynomial.poly.Poly`

    Returns
    -------
    out : tuple of :class:`~numpy.ndarray`
        Tuple of coefficient vectors

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.38-39)
    """
    Q = len(poly1.coefs)
    R = len(poly2.coefs)
    coef1 = np.kron(np.concatenate([[0], poly1.coef], axis=0), np.concatenate([[1], np.zeros((R,))], axis=0))
    coef2 = np.kron(np.concatenate([[1], np.zeros((Q,))], axis=0), np.concatenate([[0], poly2.coef], axis=0))
    return coef1, coef2


def mpoly_add_expos(poly1, poly2):
    """
    Exponents for :func:`mpoly_add`

    Parameters
    ----------
    poly1 : :class:`~lmlib.polynomial.poly.Poly`
    poly2 : :class:`~lmlib.polynomial.poly.Poly`

    Returns
    -------
    out : tuple of :class:`~numpy.ndarray`
        Tuple of exponent vectors

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.38-39)

    """
    return np.concatenate([[0], poly1.expo], axis=0), np.concatenate([[0], poly2.expo], axis=0)


def mpoly_multiply(poly1, poly2):
    """
    Product of two univariate polynomials different variables

    Parameters
    ----------
    poly1 : :class:`~lmlib.polynomial.poly.Poly` or :class:`~lmlib.polynomial.poly.MPoly`
    poly2 : :class:`~lmlib.polynomial.poly.Poly` or :class:`~lmlib.polynomial.poly.MPoly`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.40)

    """
    return MPoly(coefs=poly1.coefs + poly2.coefs, expos=poly1.expos + poly2.expos)


def mpoly_prod(polys):
    """
    Product of univariate polynomials different variables

    Parameters
    ----------
    polys : list of :class:`~lmlib.polynomial.poly.Poly` or :class:`~lmlib.polynomial.poly.MPoly`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.40)

    """
    coefs = sum(poly.coefs for poly in polys)
    expos = sum(poly.expo for poly in polys)
    return MPoly(coefs, expos)


def mpoly_square(mpoly, sparse=False):
    """
    Square of multivariate polynomial

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.42 - 6.43)

    """
    if len(mpoly.coefs) == 2:

        # implementation factorized polynomial [Wildhaber2019] (Eq. 6.42)
        coef = kron_sequence([mpoly.coefs[0], mpoly.coefs[0], mpoly.coefs[1], mpoly.coefs[1]])
    elif len(mpoly.coefs) == 1:

        # implementation non-factorized polynomial [Wildhaber2019] (Eq. 6.43)
        coef = mpoly_square_coef(mpoly, sparse)
    else:
        raise NotImplemented('More then 2 coefficient vectors in square is not implemented yet.')

    expos = mpoly_square_expos(mpoly.expos)
    return MPoly(coefs=(coef,), expos=expos)


def mpoly_square_coef_L(expos, sparse=False):
    """
    Coefficient manipulation matrix for :func:`mpoly_square`

    Parameters
    ----------
    expos : array_like

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    TODO: Ref

    """
    if len(expos) == 1:
        return np.eye(len(expos[0])**2)
    if len(expos) == 2:
        return permutation_matrix_square(len(expos[0]), len(expos[1]), sparse=sparse)
    else:
        N = len(expos)
        tmp_ = []
        for n in range(1, N):
            tmp_2 = (
                permutation_matrix_square(np.product([len(expo) for expo in expos[:n]]), len(expos[n]), sparse=sparse),)
            if n + 1 < N:
                tmp_2 += tuple([np.eye(len(expo) ** 2) for expo in expos[n + 1:]])
            tmp_.append(kron_sequence(tmp_2, sparse))

        R = eye(tmp_[0].shape[0])
        for R0 in tmp_:
            R = R.dot(R0)
        return R

    J = len(expos)
    if J >= 3:
        listed_mn = []
        tmp = []
        commutation_matrices = []
        for j in range(1, J):
            m = len(expos[j-1])
            n = len(expos[j])
            if (m, n) in listed_mn:
                index = listed_mn.index((m, n))
                tmp.append(commutation_matrices[index])
            else:
                commutation_matrices.append(commutation_matrix(m, n, sparse=sparse))
                tmp.append(commutation_matrices[-1])
                listed_mn.append((m, n))
        return kron_sequence([eye(len(expos[1]))] + tmp + [eye(len(expos[1]))], sparse=True)


def mpoly_square_coef(mpoly, sparse=False):
    """
    Non-factorized coefficient vector for :func:`mpoly_square`

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.44)

    """
    if not sparse:
        return np.dot(mpoly_square_coef_L(mpoly.expos, sparse=False), np.kron(mpoly.coefs[0], mpoly.coefs[0]))
    else:
        return mpoly_square_coef_L(mpoly.expos, sparse=True).dot(
            kron(mpoly.coefs[0], mpoly.coefs[0]).T).toarray().reshape(-1)


def mpoly_square_expos(expos):
    """
    Exponent vectors for :func:`mpoly_square`

    Parameters
    ----------
    expos : tuple of array_like

    Returns
    -------
    expos : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.45)

    """
    return tuple(M @ expo for M, expo in zip(mpoly_square_expo_Ms(expos), expos))


def mpoly_square_expo_Ms(expos):
    r"""
    Exponent manipulation matrices :math:`M_1, M_2, \dots M_N` of square of `N`-multivariate polynomials

    Parameters
    ----------
    expos : tuple of array_like
        ``(q, r)``,
        exponent vectors

    Returns
    -------
    Ms : list of :class:`~numpy.ndarray`
        ``(M_1, ..., M_N)``,
        list of exponent manipulation matrices

    References
    ----------
    TODO: Ref

    """
    out = []
    for expo in expos:
        out.append(np.add(*poly_prod_expo_Ms((expo, expo))))
    return out


def mpoly_shift(poly):
    """
    Polynomial with variable shift

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.49)

    """
    expos = mpoly_shift_expos(poly.expo)
    coefs = (mpoly_shift_coef(poly),)
    return MPoly(coefs, expos)


def mpoly_shift_coef(poly):
    """
    Coefficient vector for :func:`mpoly_shift`

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.50)

    """
    return np.dot(mpoly_shift_coef_L(poly.expo), poly.coef)


def mpoly_shift_coef_L(expo):
    """
    Coefficient manipulation matrix for :func:`mpoly_shift`

    Parameters
    ----------
    expo : array_like

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.52-6.53)

    """
    q_tilde = mpoly_shift_expos(expo)[0]
    L = []
    for n in np.arange(max(expo) + 1):
        G = np.zeros((len(q_tilde), len(expo)))
        for i, ri in enumerate(expo):
            for j, sj in enumerate(q_tilde):
                if ri - sj == n:
                    G[j, i] = comb(ri, sj)
        L.append(G)
    return np.concatenate(L, axis=0)


def mpoly_shift_expos(expo):
    """
    Exponent vector for :func:`mpoly_shift`

    Parameters
    ----------
    expo : array_like

    Returns
    -------
    out : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.51)

    """
    return (np.arange(max(expo) + 1),) * 2


def mpoly_def_int(mpoly, position, a, b):
    """
    Definite integral of a multivariate polynomial with respect to the scalar at a position

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int
    a : scalar
        lower integration boundary
    b : scalar
        upper integration boundary

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.62 - 6.63)

    """
    coefs = (mpoly_def_int_coef(mpoly, position, a, b),)
    expos = mpoly_def_int_expos(mpoly.expos, position)
    return MPoly(coefs, expos)


def mpoly_def_int_coef(mpoly, position, a, b):
    """
    Coefficient vector for :func:`mpoly_def_int`

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int
    a : scalar
        lower integration boundary
    b : scalar
        upper integration boundary

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.61), (Eq. 6.58)

    """
    return np.dot(mpoly_def_int_coef_L(mpoly.expos, position, a, b), mpoly.coefs[0])


def mpoly_def_int_coef_L(expos, position, a, b, sparse=False):
    """
    Coefficient manipulation matrix for :func:`mpoly_def_int`

    Parameters
    ----------
    expos : tuple of array_like
    position : int
    a : scalar
        lower integration boundary
    b : scalar
        upper integration boundary

    Returns
    -------
    L : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.61), (Eq. 6.58)

    """
    L_int = mpoly_int_coef_L(expos, position, sparse=sparse)
    expos_int = mpoly_int_expos(expos, position)
    L = kron_sequence(
        [np.eye(len(expo)) if n != position else np.atleast_2d(np.power(b, expo) - np.power(a, expo)) for n, expo in
         enumerate(expos_int)], sparse=sparse)
    if sparse:
        return L.dot(L_int)
    return np.dot(L, L_int)


def mpoly_def_int_expos(expos, position):
    """
    Exponent vectors for :func:`mpoly_def_int`

    Parameters
    ----------
    expos : tuple of array_like
    position : int

    Returns
    -------
    expos : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.61, Eq.6.19)

    """
    return mpoly_substitute_expos(expos, position)


def mpoly_substitute(mpoly, position, substitute):
    """
    Substituting a variable of a multivariate polynomial by a constant

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int
    substitute : scalar

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.62 - 6.63)

    """
    coefs = (mpoly_substitute_coef(mpoly, position, substitute),)
    expos = mpoly_substitute_expos(mpoly.expos, position)
    return MPoly(coefs, expos)


def mpoly_substitute_coef(mpoly, position, substitute):
    """
    Coefficient vector for :func:`mpoly_def_int`

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int
    substitute : scalar

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.58)

    """
    return np.dot(mpoly_substitute_coef_L(mpoly.expos, position, substitute), mpoly.coefs[0])


def mpoly_substitute_coef_L(expos, position, substitute):
    """
    Coefficient manipulation matrix for :func:`mpoly_def_int`

    Parameters
    ----------
    expos : tuple of array_like
    position : int
    substitute : scalar

    Returns
    -------
    L : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.58)

    """
    return kron_sequence(
        [np.eye(len(expo)) if n != position else np.atleast_2d(np.power(substitute, expo)) for n, expo in
         enumerate(expos)])


def mpoly_substitute_expos(expos, position):
    """
    Exponent vectors for :func:`mpoly_substitute`

    Parameters
    ----------
    expos : tuple of array_like
    position : int

    Returns
    -------
    expos : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.58)

    """
    return tuple(expo for n, expo in enumerate(expos) if n != position)


def mpoly_dilate(mpoly, position, eta):
    """
    Dilate a multivariate polynomial by a constant eta

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int
    eta: scalar

    Returns
    -------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.55)

    """
    coefs = mpoly_dilate_coefs(mpoly, position, eta)
    expos = mpoly_dilate_expos(mpoly.expos)
    return MPoly(coefs, expos)


def mpoly_dilate_coefs(mpoly, position, eta):
    """
    Coefficient vectros for :func:`mpoly_dilate`

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int
    eta: scalar

    Returns
    -------
    out : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.55)

    """
    return np.dot(mpoly_dilate_coef_L(mpoly.expos, position, eta), mpoly.coefs[0]),


def mpoly_dilate_coef_L(expos, position, eta):
    """
    Coefficient manipulation matrix for :func:`mpoly_dilate`

    Parameters
    ----------
    expos : tuple of array_like
    position : int
    eta: scalar

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.55)

    """
    return np.diag(kron_sequence(
        [np.ones_like(expo) if n != position else np.power(eta, expo) for n, expo in enumerate(expos)]))


def mpoly_dilate_expos(expos):
    """
    Exponent vectors for :func:`mpoly_dilate`

    Parameters
    ----------
    expos : tuple of array_like

    Returns
    -------
    expos : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.55)

    """
    return expos


def mpoly_dilate_ind(poly):
    r""" :math:`\alpha^{mathsf{T}}(xy)^q = (\Delta_Q\alpha)^{mathsf{T}}(x^q \otimes y^q)`

    Dilates a univariate polynomial `poly` by an indeterminate y

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`
        Univariate polynomial ``Poly(alpha, q)``

    Returns
    -------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
        Multivariate polynomial ``Poly((alpha_tilde,), (q, q))`` with dilation variable :math:`y`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.55)

    """
    coefs = mpoly_dilate_ind_coefs(poly)
    expos = mpoly_dilate_ind_expos(poly.expo)
    return MPoly(coefs, expos)


def mpoly_dilate_ind_coefs(poly):
    """
    Coefficient vectros for :func:`mpoly_dilate`

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`

    Returns
    -------
    out : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.55)

    """
    return np.dot(mpoly_dilate_ind_coef_L(poly.expo), poly.coef),


def mpoly_dilate_ind_coef_L(expo):
    """
    Coefficient manipulation matrix for :func:`mpoly_dilate`

    Parameters
    ----------
    expo : tuple of array_like

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.57)

    """
    return np.kron(np.eye(len(expo)), np.ones((len(expo), 1))) * np.kron(
        np.atleast_2d(np.eye(len(expo)).flatten('F')).T, np.ones_like(expo).T)


def mpoly_dilate_ind_expos(expo):
    """
    Exponent vectors for :func:`mpoly_dilate`

    Parameters
    ----------
    expo : array_like,
        `q`,
        exponent vector :math:`q`

    Returns
    -------
    expos : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.55)

    """
    return expo, expo


def extend_basis(mpoly, new_expos):
    r"""
    (DEV) Extending the basis of a uni- or multi-variate polynomial by new variables

    The new basis / variable are appended.

    .. math::

        \alpha^T x^q = (A\alpha)^T(x^q \otimes y^r \otimes z^s) =

        A = (I^q \otimes 0^r \otimes 0^s)


    Parameters
    ----------
    mpoly : MPoly, Poly
        Polynomial to extent the basis (variables)
        * Each of the new exponent vectors has to contain 1 zero element. **

    new_expos : tuple of array_like
        Each array in the tuple corresponds to a new variable
        ** Each of the new exponent vectors has to contain 1 zero element. **

    Returns
    -------
    mpoly : MPoly
        Multivariate polynomial with extended basis (new variales)

    References
    ----------
    TODO: Ref

    """

    _tmp = []
    for expo in mpoly.expos:
        _tmp.append(np.eye(expo))
    for expo in new_expos:
        if np.sum(expo == 0) != 1:
            raise ValueError('new expo contains non or more then one zero element.')
        _tmp.append(np.power(0, expo).T)
    return MPoly((kron_sequence(_tmp) @ mpoly.coefs[0]), mpoly.expos + new_expos)


def permutation_matrix(m, n, i, j, sparse=False):
    r"""
    Returns permutation matrix

    The permutation is given by

    .. math::

        vec(A\otimes B) = R_{m,n;i,j} \big(vec(A) \otimes vec(B)\big)

    with permutation matrix

    .. math::
        R_{m,n;i,j} = I_n \otimes K_{m,j} \otimes I_i \in \mathbb{R}^{mnij \times mnij}

    and :math:`A_{\{m,n\}} \in \mathbb{R}` and :math:`B_{\{i,j\}} \in \mathbb{R}`

    Parameters
    ----------
    m : int
        Size of first dimension of A
    n : int
        Size of second dimension of A
    i : int
        Size of first dimension of B
    j : int
        Size of second dimension of B

    Returns
    -------
    R : :class:`~numpy.ndarray`
        Commutation matrix

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.100-6.102)

    See Also
    --------
    permutation_matrix_square

    """
    K = commutation_matrix(m, j, sparse=sparse)
    return kron_sequence([np.eye(n, dtype=int), K, np.eye(i, dtype=int)], sparse=sparse)


def permutation_matrix_square(m, i, sparse=False):
    r"""
    Returns permutation matrix for square matrices A and B

    The permutation is given by

    .. math::

        vec(A\otimes B) = R_{m,n;i,j} \big(vec(A) \otimes vec(B)\big)

    with permutation matrix

    .. math::

        R_{m;i} = R_{m,m;i,i}

    and :math:`A_{\{m,m\}} \in \mathbb{R}` and :math:`B_{\{i,i\}} \in \mathbb{R}`

    Parameters
    ----------
    m : int
        Size of first dimension of A
    i : int
        Size of first dimension of B

    Returns
    -------
    R : :class:`~numpy.ndarray`
        Commutation matrix

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.101-6.103)

    See Also
    --------
    permutation_matrix

    """
    return permutation_matrix(m, m, i, i, sparse=sparse)


def commutation_matrix(m, n, sparse=False):
    r"""
    Returns commutation matrix

    .. math::

        K_{m,n}vec(A) = vec(A^\mathsf{T}) \in \mathbb{R}^{mn \times mn}

    where :math:`A_{\{m,n\}} \in \mathbb{R}`

    Parameters
    ----------
    m : int
        Size of first dimension of A
    n : int
        Size of second dimension of A

    Returns
    -------
    K : :class:`~numpy.ndarray`
        Commutation matrix squared

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.114-6.115)

    """
    if sparse:
        row = range(m * n)
        col = np.reshape(row, (n, m)).flatten('F')
        return csr_matrix((np.ones(m * n, dtype=int), (row, col)))
    else:
        K = np.zeros((m * n, m * n), dtype=int)
        eye_n = np.eye(n, dtype=int)
        eye_m = np.eye(m, dtype=int)
        for i in range(m):
            for j in range(n):
                tmp = np.kron(eye_m[[i]].T, eye_n[[j]])
                K += np.kron(tmp, tmp.T)
        return K


def remove_redundancy(expo):
    """
    Returns the exponent vector without redundancy and the matrix to add up the coefficients of the same exponent.

    Parameters
    ----------
    expo: array_like of shape=(Q,) fo int
        Exponent vector with redundant entries

    Returns
    -------
    L : :class:`~numpy.ndarray` of shape=(R,Q) of int
        Coefficient matrix
    r : :class:`~numpy.ndarray` of shape=(R,) of int
        Exponent vector

    """
    r, indices = np.unique(expo, return_inverse=True)
    L = np.zeros((len(r), len(indices)))
    for col, row in enumerate(indices):
        L[row, col] = 1
    return L, r


def mpoly_remove_redundancy(expos, sparse=False):
    """
    Returns the tuple of exponent vectors without redundancy and
    the matrix to add up the coefficients of the same exponent.

    TODO: How to address type of tuple of arrys different sizes

    Parameters
    ----------
    expos : tuple of array_like of shape=(Qn,) fo int
        Tuple of exponent vectors with redundant entries
    sparse : bool, optional
        If true the Coefficient matrix `L` gets returned as sparse matrix class

    Returns
    -------
    L : :class:`~numpy.ndarray` of shape=(Rn,Qn) of int
        Coefficient matrix
    expos_out : tuple of :class:`~numpy.ndarray` of shape=(Rn,) of int
        Tuple of exponent vectors
        """
    expos_red = ()
    tmp_ = []
    for expo in expos:
        L_tmp, expo_tmp = remove_redundancy(expo)
        expos_red += (expo_tmp,)
        tmp_.append(L_tmp)
    return kron_sequence(tmp_, sparse=sparse), expos_red


def mpoly_transformation_coef_L(q, c=0):
    r"""
    transformation of a uniform polynomial into the form
    .. math::

        \lambda \alpha^\mathsf{T}(\eta(x + c + \delta))^q = (\Lambda\alpha)^\mathsf{T}(x^q \otimes \eta^q \otimes \delta^q \otimes \lambda^s)

    for $s = [0, 1]$


    Parameters
    ----------
    expo : array_like,
        `q`,
        exponent vector :math:`q`
    c : scalar
        x-offset
    """

    Q = len(q)
    q = np.arange(Q)

    # eta (dilation) Q^2 x Q
    Lambda_ = mpoly_dilate_ind_coef_L(q)
    Lambda_eta = Lambda_  # Q^2 x Q^1

    # shift c
    Lambda_ = poly_shift_coef_L(q, c)  # Q x Q
    Lambda_shift_c = np.kron(Lambda_, np.eye(Q))  # Q^2 x Q^2

    # Delta
    Lambda_ = mpoly_shift_coef_L(q)  # Q^2 x Q
    Lambda_delta = np.kron(Lambda_, np.eye(Q))  # Q^3 x Q^2

    # lambda
    S = 2
    gamma = np.array([[0], [1]])
    s = np.arange(S)

    # change varialbe order delta with eta
    K = commutation_matrix(Q, Q)
    Lambda_commute = kron_sequence((np.eye(Q), K))

    return kron_sequence((Lambda_commute @ Lambda_delta @ Lambda_shift_c @ Lambda_eta, gamma))


def mpoly_transformation_expos(q):
    return q,q,q,np.arange(2)


def mpoly_extend_coef_L(expos, pos, sparse=False):
    """Extends the polynomial by variables without changing its result

    Basis expansion

    .. math::

        \alpha^\mathsf{T}x^q = (\Lambda\alpha)^\mathsf{T}(x^q \otimes y\^r \otimes ... \otimes z^s)

    for $s = [0, 1]$

    """
    tmp_ = []
    for n, expo in enumerate(expos):
        if n in pos:
            tmp_.append(np.power(0, expo).reshape(-1, 1))
        else:
            tmp_.append(np.eye(len(expo)))
    return kron_sequence(tmp_, sparse)
