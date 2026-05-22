
import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm
from scipy.signal import find_peaks
from scipy.linalg import block_diag
from lmlib.utils.generator import gen_wgn, load_csv

plt.close('all')

# Latex Style
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'   

# Constants
K = 2400  # number of samples
fs = 500  # Sampling Frequency [Hz]

k = range(K)

# load alternative ECG signal
y = load_csv('ECG_001-nohead.csv', K, channel=1)

sigma = 0.015  # Adding Gaussian Noise
y = y + gen_wgn(K, sigma, seed=233453) * np.concatenate((np.ones(K // 2),  1.0*2 * np.ones(K - K // 2)))
# Generate and add random baseline signal   
k_arr = np.array(k)
kw = 1 / (1 + np.exp(+6-k_arr/K * 13)) # increasing intensity of baseline signal
kw = (-np.cos(k_arr/K*2*np.pi)+1)/2
baseline =  kw*(-0.4 * np.cos(2 * np.pi * .6 * np.array(k) / fs) + 0.3 * np.cos(2 * np.pi * 0.4 * np.array(k) / fs) - 0.3 * np.sin(2 * np.pi * 0.2 * np.array(k) / fs))

y = y + baseline
 


def plot_T_wave(y, t, axs0, axs1, lcr_label = "LCR", V_LINE_STYLE = 'dashed', V_LINE_COLOR = 'black', LINE_COLOR = 'blue', MARKER_LINE_WIDTH = 1.0):
    """Detect T wave peaks in the ECG signal using ALSSMs and plot the results."""

    LINE_WIDTH = 1.0
    K_REF = int(1.01*500)  # time index of reference shape

    N_BASELINE= 3 # baseline model order
    N_PULSE = 3 # pulse model order
    SHAPE_LEN_2 = 45
    BL_LEN_LEFT = 65
    BL_LEN_RIGHT = 150
    g_bl = 1400
    g_sp = 1400


    # cost model to detect events
    # Defining ALSSM models
    alssm_pulse = lm.AlssmPolyJordan(poly_degree=N_PULSE - 1, label="alssm-pulse")
    alssm_baseline = lm.AlssmPoly(poly_degree=N_BASELINE - 1, label="alssm-baseline")

    # Defining segments with a left- resp. right-sided decaying window and a center segment with nearly rectangular window
    segmentLongL = lm.Segment(a=-BL_LEN_LEFT*2, b=-BL_LEN_LEFT-1, direction=lm.FORWARD, g=g_bl, delta=-BL_LEN_LEFT-1, label='long-left-segment')
    segmentL = lm.Segment(a=-BL_LEN_LEFT, b=-1 - SHAPE_LEN_2, direction=lm.FORWARD, g=g_bl, delta=-1 - SHAPE_LEN_2, label='left-segment')
    segmentC = lm.Segment(a=-SHAPE_LEN_2, b=SHAPE_LEN_2, direction=lm.FORWARD, g=g_sp, label='center-segment')
    segmentR = lm.Segment(a=SHAPE_LEN_2 + 1, b=BL_LEN_RIGHT, direction=lm.BACKWARD, g=g_bl, delta=SHAPE_LEN_2 + 1, label='right-segment')
    segmentLongR = lm.Segment(a=BL_LEN_RIGHT+1, b=int(BL_LEN_RIGHT*1.5), direction=lm.BACKWARD, g=g_bl, delta=BL_LEN_RIGHT+1, label='long-right-segment')

    # ---- COMPUTATION MODEL WITH MORE SEGMENTS
    #Defining the final cost function (a so called composite cost = CCost)
    #mapping matrix between models and segments (rows = models, columns = segments)
    F = [[0, 0, 1, 0, 0],
         [0, 1, 1, 1, 0]]
    betas = [0, 1, 1, 1, 0] #weights of the segments, will also set signal energy to 0.
    ccost = lm.CompositeCost((alssm_pulse, alssm_baseline), (segmentLongL, segmentL, segmentC, segmentR, segmentLongR), F,betas=betas)
        

    # Filter
    rls_A = lm.RLSAlssm(ccost)
    rls_A.filter(y)
    xs = rls_A.minimize_x()  # unconstrained minimization
    xs_ref = xs[K_REF]  # store state variables as reference pulse shape

    # ----

    H_A = np.transpose((block_diag(np.zeros( (0, N_PULSE) ), np.eye(N_BASELINE))))  # constrain matrix to find pulses of same shape as the reference pulse
    h_A = xs_ref.copy() 
    h_A[N_PULSE:N_PULSE+N_BASELINE] = 0.0
    H_0 = np.transpose(np.hstack([np.zeros((N_BASELINE, N_PULSE)), np.eye(N_BASELINE)]))  # constrain matrix to test for no pulse (baseline only)

    print("H_A", H_A)
    print("H_0", H_0)

    x_hat_H1_A = rls_A.minimize_x(H_A, h_A)
    #x_hat_H0_A = rls_A.minimize_x(H_0)
    x_hat_H0_A = x_hat_H1_A.copy()
    x_hat_H0_A[:, 0:N_PULSE] = 0.0

    # Square Error and LCR
    error_edge_A = rls_A.eval_errors(x_hat_H1_A)
    error_line_A = rls_A.eval_errors(x_hat_H0_A)
    lcr_A = -1 / 2 * np.log(np.divide(error_edge_A, error_line_A))

    peaks_A, _ = find_peaks(lcr_A, height=.25, distance=300)

    # Evaluate trajectories (for plotting only)
    peaks_A_1 = peaks_A[1:2]
    trajs_pulse_A = lm.Trajectory.eval_y(ccost, x_hat_H1_A, peaks_A_1, K,      F=[[0, 0, 1, 0, 0], [np.nan, np.nan, 1, np.nan, np.nan]], thd=0.01)
    trajs_baseline_A_1 = lm.Trajectory.eval_y(ccost, x_hat_H1_A, peaks_A_1, K, F=[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], thd=0.01)
    trajs_baseline_A = lm.Trajectory.eval_y(ccost, x_hat_H1_A, peaks_A, K,   F=[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], thd=0.01)

    y_hat = ccost.eval_alssm_output(x_hat_H1_A, alssm_weights=[1, 1])

    # -- Plotting ---
    if True:
        axs0.plot(t, trajs_pulse_A, c=LINE_COLOR, lw=1.25, ls='-', zorder=1)
        #axs0.plot(t, trajs_baseline_A, c='lightgray', lw=1.25, ls='--', zorder=1)
        axs0.plot(t, trajs_baseline_A_1, c=LINE_COLOR, lw=1.25, ls='--', zorder=1)
        axs0.scatter(peaks_A[:] / fs, y_hat[peaks_A], marker='v', edgecolors=LINE_COLOR, facecolors='none', s=70.0, linewidth=MARKER_LINE_WIDTH, label='T wave')
    for xp in peaks_A / fs:
        axs0.axvline(x=xp, ls=V_LINE_STYLE, c=V_LINE_COLOR, lw=LINE_WIDTH)
    

    

    axs1.plot(t, lcr_A, lw=1.0, c='k', label=lcr_label)
    for xp in peaks_A / fs:
        axs1.axvline(x=xp, ls=V_LINE_STYLE, c=V_LINE_COLOR, lw=LINE_WIDTH)    
    axs1.scatter(peaks_A / fs, lcr_A[peaks_A], marker=7, c=LINE_COLOR)
    axs1.set_ylim(bottom=.0, top=2.3)
    axs1.legend(loc=1)



def plot_Q_wave(y, t, axs0, axs1, lcr_label = "LCR", V_LINE_STYLE = 'dashed', V_LINE_COLOR = 'black', LINE_COLOR = 'green', MARKER_LINE_WIDTH = 1.0):
    LINE_WIDTH = 1.0

    # cost model to detect events
    costs_A = lm.TSLM.create_cost(ab=(-30, 7), gs=(80, 20))

    # Filter
    rls_A = lm.RLSAlssm(costs_A)
    rls_A.filter(y)

    # constraint minimization
    x_hat_H1_A = rls_A.minimize_x(lm.TSLM.H_Continuous)
    x_hat_H0_A = rls_A.minimize_x(lm.TSLM.H_Straight)

    # only accept positive slope changes
    if True:
        msk = (x_hat_H1_A[:, 3] < x_hat_H1_A[:, 1])
        x_hat_H1_A[msk,3] = x_hat_H1_A[msk,1]


    # Square Error and LCR
    error_edge_A = rls_A.eval_errors(x_hat_H1_A)
    error_line_A = rls_A.eval_errors(x_hat_H0_A)
    lcr_A = -1 / 2 * np.log(np.divide(error_edge_A, error_line_A))

    # Find LCR peaks with minimal distance and height
    peaks_A, _ = find_peaks(lcr_A, height=1, distance=250)

    # Evaluate trajectories (for plotting only)
    peaks_A_1 = peaks_A[1:2]
    trajs_edge_A = lm.Trajectory.eval_y(costs_A, x_hat_H1_A, peaks_A_1, K,  thd=0.01)
    # trajs_line_A = lm.map_trajectories(costs_A.trajectories(x_hat_H0_A[peaks_A]), peaks_A, K, merge_ks=True)

    # -- Plotting ---
    if True:
        axs0.plot(t, trajs_edge_A, c=LINE_COLOR, lw=1.25, ls='-', zorder=1)
        axs0.scatter(peaks_A[:] / fs, x_hat_H1_A[peaks_A[:], 0], marker='o', edgecolors=LINE_COLOR, facecolors='none', s=40.0, linewidth=MARKER_LINE_WIDTH, label='Q wave')

    for xp in peaks_A / fs:
        axs0.axvline(x=xp, ls=V_LINE_STYLE, c=V_LINE_COLOR, lw=LINE_WIDTH)


    # axs[nax].scatter(peaks_A, y[peaks_A], marker=7, c='b')

    axs1.plot(t, lcr_A, lw=1.0, c='k', label=lcr_label)
    for xp in peaks_A / fs:
        axs1.axvline(x=xp, ls=V_LINE_STYLE, c=V_LINE_COLOR, lw=LINE_WIDTH)    
    axs1.scatter(peaks_A / fs, lcr_A[peaks_A], marker=7, c=LINE_COLOR)
    axs1.legend(loc=1)
    axs1.set_ylim(bottom=.0, top=1.9)


def plot_P_onset(y, t, axs0, axs1, lcr_label = "LCR", V_LINE_STYLE = 'dashed', V_LINE_COLOR = 'black', LINE_COLOR = 'blue', MARKER_LINE_WIDTH = 1.0):
    LINE_WIDTH = 1.0

    # cost model to detect events
    costs_A = lm.TSLM.create_cost(ab=(-120, 40), gs=(80, 80))

    # Filter
    rls_A = lm.RLSAlssm(costs_A)
    rls_A.filter(y)

    # constraint minimization
    x_hat_H1_A = rls_A.minimize_x(lm.TSLM.H_Continuous)
    x_hat_H0_A = rls_A.minimize_x(lm.TSLM.H_Straight)

    # only accept positive slope changes
    msk = (x_hat_H1_A[:, 3] < x_hat_H1_A[:, 1])
    x_hat_H1_A[msk,3] = x_hat_H1_A[msk,1]


    # Square Error and LCR
    error_edge_A = rls_A.eval_errors(x_hat_H1_A)
    error_line_A = rls_A.eval_errors(x_hat_H0_A)
    lcr_A = -1 / 2 * np.log(np.divide(error_edge_A, error_line_A))

    # Find LCR peaks with minimal distance and height
    peaks_A, _ = find_peaks(lcr_A, height=.25, distance=250)

   # Evaluate trajectories (for plotting only)
    peaks_A_1 = peaks_A[1:2]
    trajs_edge_A = lm.Trajectory.eval_y(costs_A, x_hat_H1_A, peaks_A_1, K,  thd=0.01)
    
    # -- Plotting ---
    if True:
        axs0.plot(t, trajs_edge_A, c='b', lw=1.25, ls='-', zorder=1)  
        axs0.scatter(peaks_A[:] / fs, x_hat_H1_A[peaks_A[:], 0], marker='s', edgecolors=LINE_COLOR, facecolors='none', s=40.0, linewidth=MARKER_LINE_WIDTH, label='P onset')

    for xp in peaks_A / fs:
        axs0.axvline(x=xp, ls=V_LINE_STYLE, c=V_LINE_COLOR, lw=LINE_WIDTH)

    axs1.plot(t, lcr_A, lw=1.0, c='k', label=lcr_label)
    for xp in peaks_A / fs:
        axs1.axvline(x=xp, ls=V_LINE_STYLE, c=V_LINE_COLOR, lw=LINE_WIDTH)    
    axs1.scatter(peaks_A / fs, lcr_A[peaks_A], marker=7, c=LINE_COLOR)
    axs1.legend(loc=1)
    axs1.set_ylim(bottom=0.0, top=1.5)




# -- PLOTTING --
_, axs = plt.subplots(4, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [4, 1, 1, 1]}, sharex='all',dpi=200)
nax = 0

# ----- p onset -----
t = np.array(list(k)) / fs


axs[nax].set_title("ECG Analysis with ALSSMs: P onset, Q wave and T wave detection", fontsize=14)    


axs[nax].plot(t, y, lw=1.0, c='gray', label='$y$', zorder=0)

plot_T_wave(y, t, axs[0], axs[3], "$\mathrm{LCR}_3$", V_LINE_STYLE = 'dashed', V_LINE_COLOR = 'lightgray', LINE_COLOR = 'green', MARKER_LINE_WIDTH = 1.5)
plot_P_onset(y, t, axs[0], axs[1], "$\mathrm{LCR}_1$", V_LINE_STYLE = 'dashed', V_LINE_COLOR = 'lightgray', LINE_COLOR = 'blue', MARKER_LINE_WIDTH = 1.0)
plot_Q_wave(y, t, axs[0], axs[2], "$\mathrm{LCR}_2$", V_LINE_STYLE = 'dashed', V_LINE_COLOR = 'lightgray', LINE_COLOR = 'red', MARKER_LINE_WIDTH = 1.0)

axs[0].legend(loc='upper right')
axs[0].set_ylim(bottom=min(y), top=max(y))

# nax += 1
# axs[nax].set_visible(False)
# nax += 1

axs[0].set_xlim(left=.1, right=4.8)
#plt.subplots_adjust(bottom=0.21)

plt.show()