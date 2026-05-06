import numpy as np
from numpy.linalg import inv, matrix_power
from scipy.signal import lfilter, convolve


# xi2 lfilter cascade
def lfilter_cascade_xi2(xi2, A, C, a, b, direction, delta, gamma, y, sampleweights, beta):
    _A = np.kron(A, A)
    _C = np.kron(C, C)
    _y = np.broadcast_to(1., np.shape(y))  # create an array of shape Ks, but contains only a single 1.0 in memory

    if direction == 'fw':
        lfilter_forward_cascade_xi(xi2, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi2, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi1 lfilter cascade
def lfilter_cascade_xi1(xi1, A, C, a, b, direction, delta, gamma, y, sampleweights, beta):
    if direction == 'fw':
        lfilter_forward_cascade_xi(xi1, A, C, a, b, delta, gamma, y, sampleweights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi1, A, C, a, b, delta, gamma, y, sampleweights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi0 lfilter cascade
def lfilter_cascade_xi0(xi0, A, C, a, b, direction, delta, gamma, y, sampleweights, beta):
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = y**2

    if direction == 'fw':
        lfilter_forward_cascade_xi(xi0, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi0, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# nu lfilter cascade
def lfilter_cascade_nu(nu, A, C, a, b, direction, delta, gamma, y, sampleweights, beta):
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = np.broadcast_to(1., np.shape(y))  # create an array of shape Ks, but contains only a single 1.0 in memory

    if direction == 'fw':
        lfilter_forward_cascade_xi(nu, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(nu, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# general forward cascade
def lfilter_forward_cascade_xi(xi, A, C,  a, b, delta, gamma, y, sampleweights, beta):
    """
    IIR forward calculation of xi

    Due to generalization, different input parameter shapes are possible.
    The input parameter shapes are used to enhance the performance of the function (avoidance of matrix multiplication and memory allocation).
    Therefore, A, C, y, v can be scalar or nd-arrays.

    Parameters
    ----------
    xi : np.ndarray
        shape=(K, N, [S])
    A : np.ndarray, scalar
        shape=(N, N)
    C : np.ndarray, scalar
        shape=([L,] N)
    a : int, inf
    b : int, inf
    delta : int
    gamma : float
    y : np.ndarray, scalar
        shape=(K, [L], [S]) or 1
    sampleweights : np.ndarray
        shape=(K,) or 1
    beta : float, SE Segment weight
    einsum_path : str (see RLSALssm)
    """

    # gamma pre-calculation
    gamma_inv = 1 / gamma
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)

    # state space pre-calculation separated into matrix
    A_inv = inv(A)
    gAinvT = gamma_inv * A_inv.T
    Aa = matrix_power(A, 0 if np.isinf(a) else a - 1)
    Aac = np.dot(Aa.T, C.T)
    Ab = matrix_power(A, b)
    Abc = np.dot(Ab.T, C.T)
    N = np.shape(A)[1]

    if not np.allclose(gAinvT, np.tril(gAinvT)):
        raise "State-Space Matrix A needs to be upper triangular for cascaded version"

    yweighted = y*sampleweights[:, None]
    K = yweighted.shape[0]

    # shift signal
    # insert the shifted signal: since b > a (by definition), the recursion starts with signal b only.
    K_append  = b-a+1 #this is the length of the window
    y_delayed_b = np.zeros((K + K_append, *yweighted.shape[1:]))
    y_delayed_b[0:K] = yweighted
    y_diff = np.einsum('kl, nl->kn', y_delayed_b, gamma_b * Abc)

    if not np.isinf(a):
        # insert the shifted signal: a is inserted after K_append (length of the window).
        y_delayed_a = np.zeros((K + K_append, *yweighted.shape[1:]))
        y_delayed_a[K_append:] = yweighted
        y_diff -= np.einsum('kl, nl->kn', y_delayed_a, gamma_a * Aac)

    # iterating through ALSSM (xi) elements
    y_diff = np.swapaxes(y_diff, 0, 1)  # convenient for later indexing
    xi0 = np.zeros((K + K_append, *xi.shape[1:]))
    n_ = 0
    xi0[:, n_] = lfilter([1, 0], [1, -gamma_inv], y_diff[n_].T).T
    for n_ in range(1, N):
        y_diff[n_, 1:] += np.einsum('kn..., n->k...', xi0[:-1], gAinvT[n_])
        xi0[:, n_] = lfilter([1, 0], [1, -gamma_inv], y_diff[n_].T).T
    #xi needs to be correctly inserted. since the signal y_delayed_b had an actual delay of 0, 
    #we need to shift xi0 by b.
    if b >= 0:
        xi += xi0[b:b+K]
    #  if b < 0, first few elements of xi need to be 0 (both boundaries negative)
    if b < 0: 
        xi[-b:] += xi0[0:K+b] 


    # SE weight for this cost segment
    if beta != 1:
        xi *= beta

# general backward cascade
def lfilter_backward_cascade_xi(xi, A, C,  a, b, delta, gamma, y, sampleweights, beta):
    """-
    IIR backward calculation of xi

    Due to generalization, different input parameter shapes are possible.
    The input parameter shapes are used to enhance the performance of the function (avoidance of matrix multiplication and memory allocation).
    Therefore, A, C, y, sampleweights can be scalar or nd-arrays.

    Parameters
    ----------
    xi : np.ndarray
        shape=(K, N, [S])
    A : np.ndarray, scalar
        shape=(N, N)
    C : np.ndarray, scalar
        shape=([L,] N)
    a : int, inf
    b : int, inf
    delta : int
    gamma : float
    y : np.ndarray, scalar
        shape=(K, [L], [S]) or 1
    sampleweights : np.ndarray
        shape=(K,) or 1
    beta : float, SE Segment weight
    einsum_path : str (see RLSALssm)
    """
    # gamma pre-calculation
    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)

    # state space pre-calculation separated into scalar and matrix
    gAT = gamma * A.T
    Aa = matrix_power(A, a)
    Aac = np.dot(Aa.T, C.T)
    Ab = matrix_power(A, 0 if np.isinf(b) else b + 1)
    Abc = np.dot(Ab.T, C.T)
    N = np.shape(A)[1]

    if not np.allclose(gAT, np.tril(gAT)):
        raise "State-Space Matrix A needs to be upper triangular for cascaded version"

    K = len(xi)
    yweighted = y*sampleweights[:, None]

    #time-reverse observation for backward recursion
    yweighted_flipped = yweighted[::-1]
    
    # shift signal
    # insert the shifted signal: since a < b (by definition), the backward recursion starts with signal a only.
    K_append  = b-a+1 #this is the length of the window
    y_delayed_a = np.zeros((K + K_append, *yweighted_flipped.shape[1:]))
    y_delayed_a[0:K] = yweighted_flipped
    y_diff = np.einsum('kl, nl->kn', y_delayed_a, gamma_a * Aac)

    if not np.isinf(b):
        # insert the shifted signal: b is inserted after K_append (length of the window).
        y_delayed_b = np.zeros((K + K_append, *yweighted_flipped.shape[1:]))
        y_delayed_b[K_append:] = yweighted_flipped
        y_diff -= np.einsum('kl, nl->kn', y_delayed_b, gamma_b * Abc)

    # iterating through dimensions
    y_diff = np.swapaxes(y_diff, 0, 1)  # convenient for later indexing
    xi0 = np.zeros((K + K_append, *xi.shape[1:]))
    n_ = 0
    xi0[:, n_] = lfilter([1, 0], [1, -gamma], y_diff[n_].T).T
    for n_ in range(1, N):
        y_diff[n_, 1:] += np.einsum('kn..., n->k...', xi0[:-1], gAT[n_])
        xi0[:, n_] = lfilter([1, 0], [1, -gamma], y_diff[n_].T).T
    
    #xi needs to be correctly inserted. since the signal y_delayed_a had an actual delay of 0, 
    #we need to shift xi0 by a.
    xi0_flipped=xi0[::-1]
    xi_add=np.zeros_like(xi)
    #  if a >= 0, last elements of xi need to be 0 (both boundaries positive)
    if a >= 0:
        xi_add[0:K-a] += xi0_flipped[-(K-a):] 
    if a < 0: 
        xi_add += xi0_flipped[b+1:K+b+1] #TODO WHY DOES THIS WORK? WHY DO WE USE b HERE?
        
    xi += xi_add

    # SE weight for this cost segment
    if beta != 1:
        xi *= beta
        
        
# xi2 lfilter parallel
def lfilter_parallel_xi2(xi2, denom, num_b, num_a, a, b, direction, delta, gamma, y, sampleweights, beta):
    raise NotImplementedError("lfilter_parallel_xi2 not implemented yet.")


# xi1 lfilter parallel
def lfilter_parallel_xi1(xi1, denom, num_b, num_a, a, b, direction, delta, gamma, y, sampleweights, beta):
    if direction == 'fw':
        lfilter_forward_parallel_xi(xi1,denom, num_b, num_a, a, b, delta, gamma, y, sampleweights, beta)
    elif direction == 'bw':
        lfilter_backward_parallel_xi(xi1,denom, num_b, num_a,a, b, delta, gamma, y, sampleweights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')


# xi0 lfilter parallel
def lfilter_parallel_xi0(xi0, denom, num_b, num_a, a, b, direction, delta, gamma, y, sampleweights, beta, kappa_diag=True):
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = y**2

    if direction == 'fw':
        lfilter_forward_cascade_xi(xi0, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(xi0, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')

# nu lfilter parallel
def lfilter_parallel_nu(nu, A, C, a, b, direction, delta, gamma, y, sampleweights, beta):
    _A = np.ones((1, 1))
    _C = np.ones((1, 1))
    _y = np.broadcast_to(1., np.shape(y))  # create an array of shape Ks, but contains only a single 1.0 in memory

    if direction == 'fw':
        lfilter_forward_cascade_xi(nu, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    elif direction == 'bw':
        lfilter_backward_cascade_xi(nu, _A, _C, a, b, delta, gamma, _y, sampleweights, beta)
    else:
        raise ValueError('direction must be either "forward" or "backward"')
    

def lfilter_forward_parallel_xi(xi, denom, num_b, num_a, a, b, delta, gamma, y, sampleweights, beta):   

    if not (a < 0 and b <= 0):
        raise NotImplementedError('BACKEND: a and b has to be lower then zero for forward calculated segments.')
    
    # gamma precalculation
    gamma_a = gamma ** (a - 1 - delta)
    gamma_b = gamma ** (b - delta)

    #observation weighting with individual sample weights        
    yweighted = y*sampleweights[:, None]
    yweighted = yweighted.ravel()

    # shift signal
    K = yweighted.shape[0]
    y_delayed_b = np.empty_like(yweighted)
    y_delayed_b[:-b] = 0
    y_delayed_b[-b:] = yweighted[:K+b] * gamma_b 

    if not np.isinf(a):
        y_delayed_a = np.empty_like(yweighted)
        y_delayed_a[:-a+1] = 0
        y_delayed_a[-a+1:] = yweighted[:K+a-1] * gamma_a 
  
    N = xi.shape[1]      
    allpass = np.zeros_like(denom)
    allpass[0] = 1
    
    for n_ in range(N):
        #FIR part
        #FIRdiff = scipy.signal.lfilter(num_b[n_], allpass , y_delayed_b) - scipy.signal.lfilter(num_a[n_], allpass, y_delayed_a)        
        FIRdiff = convolve(y_delayed_b, num_b[n_], mode='full') - convolve(y_delayed_a, num_a[n_], mode='full')
        
        #IIR part    
        # Parallel version has one delay more (due to unit delay normalization of filter coefficients)
        # Therefore, the result is shifted by 1 sample to the left
        xi[:,n_] += lfilter(allpass, denom, FIRdiff)[1:K+1]    #recursion: iir part
        
    
def lfilter_backward_parallel_xi(xi, denom, num_b, num_a, a, b, delta, gamma, y, sampleweights, beta):
    
    if not (a >= 0 and b > 0):
        raise NotImplementedError('BACKEND: a and b has to be higher than zero for backward calculated segments.')

    # gamma pre-calculation
    gamma_a = gamma ** (a - delta)
    gamma_b = gamma ** (b - delta + 1)

    #observation weighting with individual sample weights        
    yweighted = y*sampleweights[:, None]
    yweighted = yweighted.ravel()

    # shift signal
    K = yweighted.shape[0]
    y_delayed_a = np.empty_like(yweighted)
    y_delayed_a[-a:] = 0
    y_delayed_a[:K-a]     = yweighted[a:] * gamma_a  

    if not np.isinf(b):
        y_delayed_b = np.empty_like(yweighted)
        y_delayed_b[-b-1:] = 0
        y_delayed_b[:K-b-1]       = yweighted[b+1:] * gamma_b  
  
    N = xi.shape[1]      
    xi_flipped = np.zeros_like(xi)
    allpass = np.zeros_like(denom)
    allpass[0] = 1
    
    
    for n_ in range(N):
        #FIR part
        #FIRdiff = scipy.signal.lfilter(num_b[n_], allpass , y_delayed_b) - scipy.signal.lfilter(num_a[n_], allpass, y_delayed_a)        
        FIRdiff = convolve(y_delayed_a[::-1], num_a[n_], mode='full') - convolve(y_delayed_b[::-1], num_b[n_], mode='full')
        
        #IIR part    
        # Parallel version has one delay more (due to unit delay normalization of filter coefficients)
        # Therefore, the result is shifted by 1 sample to the left 
        xi_flipped[:,n_] += lfilter(allpass, denom, FIRdiff)[1:K+1]    #recursion: iir part
    xi += xi_flipped[::-1]    
        