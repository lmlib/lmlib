import numpy as np
import matplotlib.pyplot as plt
import os

import lmlib as lm
from numpy.linalg import matrix_power as mpow
import time

GOLDEN_REF_PATH = os.path.join(os.path.dirname(__file__), 'golden_ref.npz')

def xi_sum(alssm, seg, y):
    K = y.shape[0]
    dt = y.dtype
    xi_sum = np.zeros((K, alssm.N))
    gamma=seg.gamma

    # Pre-cast matrices
    A = alssm.A.astype(dt)
    C_T = alssm.C.astype(dt).T

    for k_ in range(K):
        for i in range(seg.a, seg.b + 1):
            if 0 <= i + k_ < K:
                gamma_i = (gamma ** i)
                A_i = mpow(A, i).astype(dt)

                contrib = (
                    gamma_i
                    * y[i + k_].astype(dt)
                    * (A_i.T @ C_T).astype(dt)
                )
                xi_sum[k_, :] += contrib.astype(dt)

    return xi_sum


def xi_sum_mp(alssm, seg, y, dps=50):
    """High-precision golden reference via mpmath (dps decimal places).

    Precomputes v_i = gamma^i * (A^i)^T @ C^T for each boundary index i
    using mpmath, then performs the final convolution in float64.
    Stores the result in GOLDEN_REF_PATH for later reuse.
    """
    import mpmath as mp
    mp.mp.dps = dps
    K, N = y.shape[0], alssm.N
    a, b = seg.a, seg.b

    gamma_mp = mp.mpf(seg.gamma)
    A_mp = mp.matrix(alssm.A.tolist())
    C_mp = mp.matrix(alssm.C.ravel().tolist())  # row vector (1 x N) → treated as (N,)

    # Precompute v_i = gamma^i * (A^i)^T @ C^T  (shape N)
    V = np.empty((b - a + 1, N), dtype=np.float64)
    for idx, i in enumerate(range(a, b + 1)):
        if i >= 0:
            Ai = A_mp ** i
        else:
            Ai = mp.inverse(A_mp ** (-i))
        AiT = Ai.T
        gamma_i = gamma_mp ** i
        for n in range(N):
            s = mp.mpf(0)
            for j in range(N):
                s += AiT[n, j] * C_mp[j]
            V[idx, n] = float(gamma_i * s)

    # Convolution (float64 — only b-a+1 terms so accumulation error ~(b-a+1)*eps)
    xi = np.zeros((K, N), dtype=np.float64)
    for idx, i in enumerate(range(a, b + 1)):
        k0 = max(0, -i)
        k1 = min(K, K - i)
        if k0 >= k1:
            continue
        xi[k0:k1, :] += y[i + k0: i + k1, np.newaxis] * V[idx]

    np.savez(GOLDEN_REF_PATH, xi=xi, y=y, a=a, b=b, gamma=seg.gamma,
             A=alssm.A, C=alssm.C, dps=dps)
    print(f"Golden reference saved to {GOLDEN_REF_PATH}  (dps={dps})")
    return xi


def load_golden_ref(alssm, seg, y):
    """Load precomputed golden reference if parameters match, else return None."""
    if not os.path.exists(GOLDEN_REF_PATH):
        return None
    d = np.load(GOLDEN_REF_PATH)
    if (np.array_equal(d['y'], y) and int(d['a']) == seg.a and int(d['b']) == seg.b
            and float(d['gamma']) == seg.gamma
            and np.array_equal(d['A'], alssm.A)
            and np.array_equal(d['C'], alssm.C)):
        print(f"Loaded golden reference from {GOLDEN_REF_PATH}  (dps={int(d['dps'])})")
        return d['xi']
    return None



# Signal
np.random.seed(42)
K = int(1e4)
y = np.random.randn(K)
#y = y.reshape(K//2,2)
k = np.arange(K)
  
# Model
polydegree=3
alssm = lm.AlssmPolyJordan(poly_degree=polydegree)
#alssm = lm.AlssmSin(omega=0.2)
N1 = alssm.N
g=1000
segment='costsegment'
if segment=='costsegment':
    segment = lm.Segment(a=-10, b=5, direction=lm.FORWARD, g=g)
    cost = lm.CostSegment(alssm, segment)

    # Golden reference: load from disk if available, else compute with mpmath
    xi_ref = load_golden_ref(alssm, segment, y)
    if xi_ref is None:
        print("Computing high-precision golden reference with mpmath (dps=50)…")
        t0 = time.perf_counter()
        xi_ref = xi_sum_mp(alssm, segment, y, dps=50)
        t_sum = time.perf_counter() - t0
    else:
        t_sum = 0.0

if segment=='compositecost':
    segment_l = lm.Segment(a=-20, b=-10, direction=lm.FORWARD, g=g)
    segment_r = lm.Segment(a=-10, b=20, direction=lm.BACKWARD, g=g)
    cost = lm.CompositeCost((alssm,alssm), (segment_l,segment_r),F=[[1, 0],[0, 1]])

# Direct Matrix Form Implementation (lmlib)
rls_directmatrix = lm.RLSAlssm(cost, calc_kappa=False, backend='numpy')
t0 = time.perf_counter()
rls_directmatrix.filter(y)
t_directmatrix = time.perf_counter() - t0

# Cascade Form Implementation 
if np.allclose(alssm.A, np.triu(alssm.A)):
    rls_cascade = lm.RLSAlssm(cost, calc_kappa=False, backend='lfilter', filter_form='cascade')
    t0 = time.perf_counter()
    rls_cascade.filter(y)
    t_cascade = time.perf_counter() - t0
else:
    print ("Can't execute cascade filter computation: State-Space Matrix A needs to be upper triangular.")
    rls_cascade=None
    t_cascade = None

# Parallel Form Implementation
execute_parallel=True
if execute_parallel:
    rls_parallel = lm.RLSAlssm(cost, calc_kappa=False, backend='lfilter', filter_form='parallel')
    t0 = time.perf_counter()
    rls_parallel.filter(y)
    t_parallel = time.perf_counter() - t0
else:
    rls_parallel=None
    t_parallel = None

if segment=='compositecost':
    xi_ref = rls_directmatrix.xi


# Print timing results
print("\n=== Timing ===")
print(f"Golden Ref (mpmath)  : {t_sum*1e3:.2f} ms" if t_sum > 0 else "Golden Ref (mpmath)  : loaded from disk")
print(f"Direct Matrix        : {t_directmatrix*1e3:.2f} ms")
print(f"Cascade              : {t_cascade*1e3:.2f} ms" if t_cascade is not None else "Cascade       : N/A")
print(f"Parallel             : {t_parallel*1e3:.2f} ms" if t_parallel is not None else "Parallel       : N/A")

# Errors (time-domain)
print("\n=== Errors ===")
for n in range(alssm.N):
    tests = [
        (rls_directmatrix.xi, 'Direct Matrix'),
    ]
    if rls_parallel is not None:
        tests.append((rls_parallel.xi, 'Parallel     '))
    if rls_cascade is not None:
        tests.append((rls_cascade.xi, 'Cascade      '))

    for xi_test, label in tests:
        rChan = np.arange(0,n+1)
        error = xi_ref[:,rChan] - xi_test[:,rChan]   
        error_normalized = np.zeros_like(error)
        for n_ in rChan:
            sn = np.sqrt(np.mean(xi_ref[:,n_]**2))
            error_normalized[:,n_] = error[:,n_] / sn
        max_abs_error = np.max(np.abs(error_normalized))
        vnrmse = np.sqrt(np.mean(error_normalized**2))
        
        printerrors=True
        if printerrors:
            print(f"{label}, n={n}: Max absolute error: {max_abs_error:.2e}, VNRMSE: {vnrmse:.2e}")

# ------------------ plotting --------------------------------------------
plt.close('all')
plot_error=False
plot_xi=False
dpi = 300 

if plot_xi:
    # -- 5.  Plotting --
    figxi, axs = plt.subplots(alssm.N+1, 1, figsize=(0.8*6, 0.8*4), sharex='all',dpi=dpi)

    # Observation
    nax=0
    axs[nax].plot(k, y[k], c='xkcd:gray', lw=1, label='$y$')
    axs[nax].legend(loc='upper right')
    axs[nax].spines['top'].set_visible(False)
    axs[nax].spines['right'].set_visible(False)

    # Collect handles/labels from the first subplot with filter errors
    handles, labels = [], []
    
    color_black = 'k'
    color_blue = 'xkcd:blue'
    color_gray = 'xkcd:gray'
    
    epsilon = np.finfo(float).eps

    for n in range(alssm.N):
        nax+=1
        sn = np.sqrt(np.mean(xi_ref[:,n]**2))
       
        axs[nax].plot(k, rls_directmatrix.xi[k,n], c=color_black, ls='-', lw=1, label='Direct Matrix',alpha=1)
       
        if rls_cascade is not None:
            axs[nax].plot(k, rls_cascade.xi[k, n], c=color_gray, lw=1, ls='-', label='Cascade', alpha=1)
            
        if rls_parallel is not None:            
            axs[nax].plot(k, rls_parallel.xi[:,n], c=color_blue, ls='-', lw=1, label='Parallel',alpha=1)
        axs[nax].set_ylabel(rf'$\xi_{n}(k)\,$[dB]')
        axs[nax].spines['top'].set_visible(False)
        axs[nax].spines['right'].set_visible(False)
        if nax==1:
            handles, labels = axs[nax].get_legend_handles_labels()
    axs[nax].set_xlim([0,K])        
    axs[nax].set_xlabel('$k$')        
    figxi.tight_layout(pad=0.3)
    
    # Add one shared legend (above subplots 2+3, i.e. centered on top)
    figxi.legend(handles, labels, loc='center right',
                #bbox_to_anchor=(0.935, 0.5), ncol=1, fontsize='small')
                bbox_to_anchor=(0.945, 0.5), ncol=1, fontsize='small')
    
    # Collect y-limits from all error axes (skip axs[0] which shows observations)
    all_mins, all_maxs = [], []
    for nax_ in range(1, alssm.N + 1):
        ymin, ymax = axs[nax_].get_ylim()
        all_mins.append(ymin)
        all_maxs.append(ymax)
    
    shared_ylim = (min(all_mins), max(all_maxs))
    for nax_ in range(1, alssm.N + 1):
        axs[nax_].set_ylim(shared_ylim)
    

if plot_error:
    # -- 5.  Plotting --
    figerror, axs = plt.subplots(alssm.N+1, 1, figsize=(0.8*6, 0.8*4), sharex='all',dpi=dpi)

    # Observation
    nax=0
    axs[nax].plot(k, y[k], c='xkcd:gray', lw=1, label='$y$')
    axs[nax].legend(loc='upper right')
    axs[nax].spines['top'].set_visible(False)
    axs[nax].spines['right'].set_visible(False)

    # Collect handles/labels from the first subplot with filter errors
    handles, labels = [], []
    
    color_black = 'k'
    color_blue = 'xkcd:blue'
    color_gray = 'xkcd:gray'
    
    epsilon = np.finfo(float).eps

    for n in range(alssm.N):
        nax+=1
        sn = np.sqrt(np.mean(xi_ref[:,n]**2))
       
        e_directmatrix = (xi_ref[k,n] - rls_directmatrix.xi[k,n]  )  /sn
        e_directmatrix_db = 20 * np.log10(np.abs(e_directmatrix) + epsilon)
        axs[nax].plot(k, e_directmatrix_db, c=color_black, ls='-', lw=1, label='Direct Matrix',alpha=1)
       
        if rls_cascade is not None:
            e_cascade = (xi_ref[k, n] - rls_cascade.xi[k, n]) / sn
            e_cascade_db = 20 * np.log10(np.abs(e_cascade) + epsilon)
            axs[nax].plot(k, e_cascade_db, c=color_gray, lw=1, ls='-', label='Cascade', alpha=1)
       
        if rls_parallel is not None:     
            e_ss2tfsos = (xi_ref[k,n] - rls_parallel.xi[:,n] ) /sn
            e_ss2tfsos_db = 20 * np.log10(np.abs(e_ss2tfsos) + epsilon)
            axs[nax].plot(k, e_ss2tfsos_db, c=color_blue, ls='-', lw=1, label='Parallel',alpha=1)
 
        axs[nax].set_ylabel(rf'$e_{n}(k)\,$[dB]')
        axs[nax].spines['top'].set_visible(False)
        axs[nax].spines['right'].set_visible(False)
        if nax==1:
            handles, labels = axs[nax].get_legend_handles_labels()
    axs[nax].set_xlim([0,K])        
    axs[nax].set_xlabel('$k$')        
    figerror.tight_layout(pad=0.3)
    
    # Add one shared legend (above subplots 2+3, i.e. centered on top)
    figerror.legend(handles, labels, loc='center right',
                #bbox_to_anchor=(0.935, 0.5), ncol=1, fontsize='small')
                bbox_to_anchor=(0.945, 0.5), ncol=1, fontsize='small')
    
    # Collect y-limits from all error axes (skip axs[0] which shows observations)
    all_mins, all_maxs = [], []
    for nax_ in range(1, alssm.N + 1):
        ymin, ymax = axs[nax_].get_ylim()
        all_mins.append(ymin)
        all_maxs.append(ymax)
    
    shared_ylim = (min(all_mins), max(all_maxs))
    for nax_ in range(1, alssm.N + 1):
        axs[nax_].set_ylim(shared_ylim)
    
    plt.show()
