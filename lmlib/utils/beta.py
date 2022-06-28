"""Collection of experimental functions (beta)"""

import os
import numpy as np
import csv as csv
import lmlib as lm
import itertools
import datetime as datetime


__all__ = ['find_max_mask',
           'load_source_csv',
           'zero_cross_ind',
           'diff0',
           'edge_detection',
           'poly_filter',
           'mpoly_fit_subspace1',
           'constrain_matrix_for_exponents'
           ]


"""
 This file hosts helper functions or classes which are provided as beta and will later be transferred to the library 
 when having reached a stable state. 
"""


def find_max_mask(y, msks, locs=None, range_start=None, range_end=None,  threshold=-np.inf, direction='maximum', skip_invalid=False, SHOW_DEBUG_PLOT=False):

    """
        Advanced peak finder with masked samples and search intervals.

        This peak finder finds in signal :code:`y` for every interval that sample fulfilling multiple criteria.

        Parameters
        ----------
        y : array_like, shape=(K,)
            input signal
        msks: array_like of shape=(K) of Boolean OR tuple of array_like of shape=(K) of Boolean,
            a mask or a list of masks each of length :code:`K` with :code:`True` for any valid samples.
            If multiple masks are provided, a sample is only valid, if set :code:`True` in all masks (multiple mask are merged using logic AND).
            The peak index is decided among all valid samples
        locs: array_line of shape=(P) of int,
            reference index for `P` intervals
        range_start: int
            start index of a interval relative to the reference index
        range_end: int
            end index of a interval relative to the reference index
        threshold: int
           min. value in :code:`y` for peaks
        direction: string {"forward", "backward", "maximum"}
           search direction :code:`"forward"` or :code:`"backward"` (return first value fulfilling all criteria) or :code:`"maximum"` (take maximum value of y)
        skip_invalid: bool
           :code:`True`: removes entries if no peak is found in a certain interval.
           :code:`False`: if no peak is found in a certain interval, the index is set to -1 instead.
            default: False

        Returns
        -------
        out: ndarray of shape(<=K,) of int
             index per location interval or -1 if :code:`skip_invalid == False`.
             if :code:`skip_invalid == True`, the returned vector might has less than `K` entries.

    """



    K = len(y)
#    msk_locs = np.zeros((K,), dtype=bool)
    if msks is None: # TODO: speed up, not very efficient to creat mask if not neede
        msk_merged=np.zeros((K, ), dtype=bool)
    elif type(msks) == tuple:
        msk_merged=np.ones((K, ), dtype=bool)
        for m in msks:
            msk_merged = msk_merged & m
    else:
        msk_merged = msks


    if not locs is None:
        NOF_LOCS = len(locs)
        ind = np.array([0] * NOF_LOCS)
        if range_start==None:
            msk_locs[locs] = True
        else:
            for iloc in range(len(locs)):
                loc = locs[iloc]
                # todo: do faster using convolution
                ind_start = np.minimum(np.maximum(loc+range_start, 0), K) # extract interval around current loc index
                ind_end = np.minimum(np.maximum(loc+range_end, 0), K)

                y_loc = y[ind_start:ind_end]
                k_loc = np.arange(len(y_loc))

                loc_msk = (y_loc>=threshold) & msk_merged[ind_start:ind_end]
                intLen = ind_end - ind_start


                if (len(loc_msk)) and (np.max(loc_msk)==True): # is mask not all False?
                    if direction=='forward':
                        rel_ind = np.argmax(loc_msk)
                    elif direction=='backward':
                        rel_ind = intLen - np.argmax(np.flip(loc_msk)) - 1
                    elif direction=='maximum':
                        rel_ind = k_loc[loc_msk][np.argmax(y_loc[loc_msk])]
                    else:
                        print("INVALID DIRECITON") # todo: throw error

                    ind[iloc] = rel_ind+ind_start
                else:
                    ind[iloc] = -1

                #print(y[ind_start:ind_end]>=height)
                #print(loc_msk[rel_ind])
                #print("R (loc)"+str(loc)+"rel_ind: "+str(rel_ind)+" ind[iloc]: "+ str(ind[iloc]))


                if SHOW_DEBUG_PLOT: # display debug plots
                    k = range(ind_start,ind_end)
                    _, axs = plt.subplots(5, 1, sharex='all')

                    axs[0].set(xlabel='window', label='win')
                    axs[0].plot(k, y[ind_start:ind_end],label='y')
#                    axs[0].plot(k[loc_msk], y[ind_start:ind_end][loc_msk], 'k.', label='peak')
                    axs[0].plot(ind[iloc], y[ind[iloc]], 'r.', label='peak')
                    axs[0].legend()

                    axs[1].plot(k, (y[ind_start:ind_end]>=threshold), label='y>threshold')
                    for m in msks:
                        axs[1].plot(k, m[ind_start:ind_end], label='msk')
                    axs[1].plot(k, msk_merged[ind_start:ind_end], 'r--', label='msk-merged')
                    axs[1].legend()


#                    axs[2].plot(k, (lcr0[ind_start:ind_end]) )

                    plt.show()

    if skip_invalid == True:
        return (ind[ind>=0])
    else:
        return (ind)

def load_source_csv(filename, time_format=None, K=-1, k=0, ch_select=None):
    r"""
    Loads time series from csv file with/without header section.

	File format:

    .. code-block:: text

	    # Description Line 1
	    # ...
	    # Description Line n
	    #
	    time,         signal1, signal2, ...
	    00:00:20.000, 10.34,   11.09, ...

    * Description: the file can start with an (optional) description section, prefixed with an '#'
    * Header: The data table can start with an (optional) column labels. A header line is detected, if the first data line starts with a non-nummeric character
    * Data: float values separated by either :code:`',',';','\space','\tab'`; first column commonly contains the time; either as float (seconds) or as 'hh:mm:ss.ttt'; the later will be converted to floats (in seconds).



    Parameters
    ----------
    filename : string
        csv file to be loaded
    time_format : string
        Format string to be used to decode first line:  "H:M:S" (e.g., 00:02:01.123) or None (e.g. 1.12)
    K : int
        Length of signal to be loaded or -1 to load until end of file.
        If `K` is larger than the maximal signal length, an assertion is raised.
    k : int
        Signal load start index.
        If `k` is larger than the maximal signal length, an assertion is raised.
    ch_select : int, list, None
        int: channel index to be loaded in single channel mode.
        list: List of `M` channels to be loaded in multi channel mode. Selects channel indices to be loaded from multi-channel signals. `0 <= ch_select < Number of CSV columns`.
        If set to `None`, all channels are loaded in multi channel mode and `M = Number of CSV columns`.
        First column in the CSV file is addressed as channel 0, and contains the time, if `time_format` is set accordingly.


    Returns
    -------
	out: Tuple (col_labels, data)
         with
             - col_labels: list of column labels; if csv does not provide column headers, headers are set as :code:`'col1','col2','col3', ...`  *** NOT YET IMPLEMENTED ***

             - data: (ndarray, shape=(K[, M])) – Multi-channel signal of shape=(K, M) if `ch_select` is of type `list`, single-channel signal of shape=(K, ) if `ch_select is of type `int`.
    """

    def string_to_float_array(arr):
        """ conversion of array of strings to array of floats; non-convertible values are substitutest by np.NAN """
        convertArr = []
        for s in arr.ravel():
            try:
                value = s.astype(np.float)
            except ValueError:
                value = np.NAN

            convertArr.append(value)

        return np.array(convertArr, dtype=float).reshape(arr.shape)



    if ch_select is None:
        ch_select = []
        is_multichannel = True
    elif isinstance(ch_select, int):
        ch_select = [ch_select,] # wrap int in a list
        is_multichannel = False
    elif isinstance(ch_select, list):
        is_multichannel = True
    else:
        assert True, "The attribute ch_select has an unsupported data type. Accepted types are: None -> loads all channels, list -> loads listed channels , int -> returns single channel."

    
#    print(os.path.abspath(filename) )

    with open(filename, newline='') as csvfile:
#        file_lines = csvfile.readlines()
        file_lines = [line.rstrip() for line in csvfile]


    header_str = ""

    READER_STATE = 0 # 0: reading comment header; 1: reading column headers; 2: reading values
    f_index = -1

    for data_line in file_lines:
        f_index += 1

        # update states

        if (f_index == 0) and len(data_line)>0 and (ord(data_line[0])>255): # Remove UTF Marking, if existing
            data_line = data_line[1:]


        if READER_STATE == 0:
            if data_line.startswith("#") is not True:
                READER_STATE = 1 # read header
            else:
                header_str += data_line + "\n"
                continue

        if READER_STATE == 1:
            if not data_line[:1].startswith(tuple('0123456789.-+')): # check header
                headers = data_line.split(',')
                READER_STATE = 2
                continue
            else:
                READER_STATE = 2 # read values
                break # read rest in a bunch

        if READER_STATE == 2:
            break

    K_file = len(file_lines)-f_index # total number of entries in file

    if k == 0:
        if K != -1:
            assert 0 <= K < K_file, "The signal length K is out of range." + \
                                    "\nExpected range: [0, {})".format(K_file) + \
                                    "\nFound K: {}".format(K)
    else:
        assert 0 <= k < K_file, "The signal start index k is out of range." + \
                                "\nExpected range: [0, {})".format(K_file) + \
                                "\nFound k: {}".format(k)
        if K != -1:
            assert 0 <= k + K < K_file, "The signal start + signal length is out of range." + \
                                        "\nExpected range: [0, {})".format(K_file) + \
                                        "\nReduce k to : {}".format(K_file - K - 1) + \
                                        "\nor reduce K to : {}".format(K_file - k - 1) + \
                                        "\nFound k+K: {}".format(k + K)
    
    f_index_start = f_index+k
    if K>=0:
        f_index_end = f_index_start + K
    else:
        f_index_end = K_file

       
    data = list(csv.reader(file_lines[f_index_start:f_index_end], delimiter=',', ))


    if len(data) > 0: # identify number of channels
        M = len(data[0])
    else:
        M = 0

    if time_format=="H:M:S":
        time_strs = [data_line[0].split(':') for data_line in data]

        y = np.concatenate(
                 (
                      np.array([[ ((float(time_str[0])*60+float(time_str[1])*60)+float(time_str[2])), ]  for time_str in time_strs]),
                      string_to_float_array(np.array(data)[:, 1:])
#                      np.array(data)[:, 1:].astype(
#                      np.float) # convert all columns to float except column 1
                 ), axis = 1)
    else: # no specific time format
        y = string_to_float_array(np.array(data))



    for ch in ch_select:
        assert 0 <= ch < M, "The channel selection is out of range." + \
                            "\nExpected range: [0, {})".format(M) + \
                            "\nFound channel in ch_select: {}".format(ch)

    if not ch_select:
        return y
    elif is_multichannel:
        return (y[:, ch_select]) # return multiple channels
    else:
        return (y[:, ch_select[0]]) # return single channel


def zero_cross_ind(y, threshold=0):
    """ Returns a list of the indeces of zero crossings

    Parameters
    ----------
        y: array_like of shape=(K,) of floats
             signal vector to detect zero-crossings
        threshold: float
             crossing threshold (default: 0).

    Returns
    -------
        out: ndarray of ints
            array with indices :code:`i` where :code:`y[i]<0` and :code:`y[i+1]>=0` (i.e., indices before the threshold).

    """
    return (np.where(np.diff(np.signbit(vals-zeros_at)))[0])

def diff0(y):
    """ Gets difference of consecutive elements in vector, starting with the first element

    This is the same as :code:`numpy.diff(y)`, but starting with the first element :code:`y[0]` as the first difference
    (leading to a vector of the same length as the input signal vector `y`.

    Implementation:

    .. code-block:: text

        return np.append(np.array(y[0]), np.diff(y))



    Parameters
    ----------
       y: array_like of shape(K,) of floats
           input signal vector

    Returns
    -------
       out: ndarray of shape(K,) of floats
            element-wise differences


    """
    return np.append(np.array(y[0]), np.diff(y))


def edge_detection(y, ab, g):
    r""" Performs edge detection applying joint line models

    Uses a two-line model to detect edges, according to
    example: :ref:`sphx_glr_generated_examples_11-lssm-costs-detection_example-ex110.0-edge-detection.py`


    Parameters
    ----------
    y: array_like of shape=(K,) of floats
       observation vector
    ab: tuple (a,b)
       - a (int, >0): left-sided window length (>0)
       - b (int, >0): right-sided window length (>0)
    g: tuple (g_a, g_b) of float
       Effective number of samples under the left-sided and right-sided window, see :class:`~lmlib.statespace.costfunc.CostSegment`.
         - g_a (float, >0): left-sided window length (>0)
         - g_b (float, >0): right-sided window length (>0)
    g_a: float
       Effective number of samples under the right-sided window, see :class:`~lmlib.statespace.costfunc.CostSegment`.


    Returns
    -------

    lcr: ndarray of shape(K,) of floats
         log-likelihood ratio for a detected edge
    y0: ndarray of shape(K,) of floats
        position estimate of edge on y-axis
    a0: ndarray of shape(K,) of floats
        line slope of left-sided model
    a1: ndarray of shape(K,) of floats
        line slope of right-sided model






    """

    # ALSSM models
    alssm_left = lm.AlssmPoly(poly_degree=1)  # A_L, c_L
    alssm_right = lm.AlssmPoly(poly_degree=1)  # A_R, c_R
    segment_left = lm.Segment(a=-ab[0], b=-1, direction=lm.FORWARD, g=g[0])
    segment_right = lm.Segment(a=0, b=ab[1], direction=lm.BACKWARD, g=g[1])
    F = [[1, 0], [0, 1]]  # mixing matrix, turning on and off models per segment (1=on, 0=off)
    ccost = lm.CompositeCost((alssm_left, alssm_right), (segment_left, segment_right), F, label="Edge-Model")


    # Linear Constraints

    # Forcing a continuous function of the two lines --> x_hat_edge
    H_edge = np.array([[1, 0, 0],  # x_1,left : offset of left line
                       [0, 1, 0],  # x_2,left : slope of left line
                       [1, 0, 0],  # x_1,right : offset of right line
                       [0, 0, 1]])  # x_2,right : slope of right line

    # Forcing additionally a common slope of the two lines --> x_hat_line
    H_line = np.array([[1, 0],  # x_1,left : offset of left line
                       [0, 1],  # x_2,left : slope of left line
                       [1, 0],  # x_1,right : offset of right line
                       [0, 1]])  # x_2,right : slope of right line

    # -- MODEL FITTING --
    # Filter
    separam = lm.SEParam(ccost)
    separam.filter(y)

    # constraint minimization
    x_hat_edge = separam.minimize_x(H_edge)
    x_hat_line = separam.minimize_x(H_line)

    # Square Error and LCR
    error_edge = separam.eval_errors(x_hat_edge)
    error_line = separam.eval_errors(x_hat_line)
    lcr = -1 / 2 * np.log(np.divide(error_edge, error_line))

    return (lcr, x_hat_edge[:,0], x_hat_edge[:,1], x_hat_edge[:,3])



def poly_filter(y, ab, g, poly_degree=0):
    r""" Performs edge detection applying joint line models

    Uses polynomial of oder 'poly_degree' to filter signal. (for poly_degree=0, the filter leads to a moving average filter)
    example: :ref:`sphx_glr_generated_examples_12-lssm-costs-filters-example-ex121.0-sc-filter.py`


    Parameters
    ----------
    y: array_like of shape=(K,) of floats
       observation vector
    ab: tuple (a,b)
       - a (int, >0): left-sided window length (>0)
       - b (int, >0): right-sided window length (>0)
    g: tuple (g_a, g_b) of float
       Effective number of samples under the left-sided and right-sided window, see :class:`~lmlib.statespace.costfunc.CostSegment`.
         - g_a (float, >0): left-sided window length (>0)
         - g_b (float, >0): right-sided window length (>0)
    g_a: float
       Effective number of samples under the right-sided window, see :class:`~lmlib.statespace.costfunc.CostSegment`.
    poly_degree: int
       degree of polynomial to fit; default: 0 (constant value, i.e., moving average)


    Returns
    -------

    y0: ndarray of shape(K,) of floats
         average estimate
    error: ndarray of shape(K,) of floats
        squared error of fit

    """

    # ALSSM models
    alssm = lm.AlssmPoly(poly_degree=poly_degree)
    segment_left = lm.Segment(a=-ab[0], b=-1, direction=lm.FORWARD, g=g[0])
    segment_right = lm.Segment(a=0, b=ab[1], direction=lm.BACKWARD, g=g[1])
    F = [[1, 1]]  # mixing matrix, turning on and off models per segment (1=on, 0=off)
    ccost = lm.CompositeCost((alssm, ), (segment_left, segment_right), F, label="filter-model")

    # -- MODEL FITTING --
    # Filter
    separam = lm.SEParam(ccost)
    separam.filter(y)

    # minimization
    x_hat = separam.minimize_x()

    # Square Error
    error = separam.eval_errors(x_hat)

    return (x_hat[:, 0], error)




def constrain_matrix_for_exponents(expos, threshold):
    """
    Creates a matrix which reduces exponents which are summed higher then the threshold

    Parameters
    ----------
    expos : tuple of array_like
        exponent vectors
    threshold : int
        Threshold

    Returns
    -------
    H : :class:`np.ndarray`
        H matrix
    """
    prod = list(itertools.product(*expos))
    mask = np.sum(prod, axis=1) > threshold
    H = np.eye(len(prod))
    return np.delete(H, np.argwhere(mask), 1)


def mpoly_fit_subspace1(betas, beta_expo, boundaries, coords, coord_expos, H=None, return_cost=False):
    """
    Fits a multivariate polynomial to polynomial subspace with corresponding coordinates

    Parameters
    ----------
    betas : array_like of shape (N, Q, ...)
        polynomial coefficients
    beta_expo : array_like of shape (Q,)
        exponent vector of beta
    boundaries : array_like of shape (2,)
        integral boundaries (lower, upper)
    coords : array_like of shape (N, M)
        coordinates of shape (M,) for each beta in betas
    coord_expos : tuple of shape (M,) of 1D- array_like
        exponent vector for each variable in coordinates
    H : None, array_like, optional
        output exponent reduction matrix ( max limit )
    return_cost : bool, optional
        returns additionally the cost for each sample

    Returns
    -------
    alphas : array_like
        coefficients of multivariate polynomial fit
    cost : array_like, optional
        Cost if flag `return_cost = True`

    """
    beta_expo = np.asarray(beta_expo)
    betas = np.asarray(betas)
    if betas.shape[1] != len(beta_expo):
        raise ValueError('beta is betas doesn\'t match with beta_expo. Wrong length.')
    N = betas.shape[0]
    if len(boundaries) != 2:
        raise ValueError('boundaries has not a length of 2')
    a = boundaries[0]
    b = boundaries[1]
    if a > b:
        raise ValueError('Condition not fulfilled: a <= b')
    coord_expos = np.asarray(coord_expos)
    if coord_expos.ndim != 2:
        raise ValueError('coord_expos has wrong number of dimensions. '
                         'Needs to be tuple(array_like)/tuple(array_like, array_like, ...)')
    coords = np.asarray(coords)
    if coords.shape != (N, len(coord_expos)):
        raise ValueError('coords  doesn\'t match with length of betas or with length of coord_expos.')

    J = max(beta_expo) + 1
    M = lm.poly_square_expo_M(beta_expo)
    L = lm.poly_int_coef_L(M @ beta_expo)
    c = np.power(b, M @ beta_expo + 1) - np.power(a, M @ beta_expo + 1)
    C = np.reshape(L.T @ c, (J, J))
    As = np.zeros((N, len(beta_expo), np.prod([len(expo) for expo in coord_expos])*len(beta_expo)))
    for n, coord in enumerate(coords):
        A = lm.kron_sequence([np.power(value, expo) for value, expo in zip(coord, coord_expos)])
        As[n] = np.kron(np.eye(J), A)

    if H is None:
        ACA = np.sum([A.T @ C @ A for A in As], axis=0)
        ACbeta = np.sum([np.einsum('an, n...->a...', A.T.dot(C), beta) for A, beta in zip(As, betas)], axis=0)
    else:
        ACA = np.sum([H.T @ A.T @ C @ A @ H for A in As], axis=0)
        ACbeta = np.sum([np.einsum('an, n...->a...', H.T @ A.T.dot(C), beta) for A, beta in zip(As, betas)], axis=0)
    alphas = np.einsum('ba, a...->b...', np.linalg.pinv(ACA), ACbeta)
    if return_cost:
        Js = np.einsum('a..., a...->...', np.einsum('b..., ba->a...', alphas, ACA), alphas) \
             - 2*np.einsum('b..., b...->...', ACbeta, alphas)\
             + np.einsum('ja..., ja... ->...', np.einsum('jb..., ba ->ja...', betas, C), betas)
        return alphas, Js
    return alphas

