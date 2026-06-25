#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Channel ECG Shape Detection with a 2-D ALSSM / Cost-Ratio [ex804.0]
=========================================================================

Heartbeat shapes are detected in a multi-lead ECG by fitting a separable
2-D Autonomous Linear State-Space Model (ALSSM) over *time* (axis 0) and
*channel* (axis 1) and scoring every time index with a cost ratio

$CR = \dfrac{J_0}{J_1}$

where $J_0$ is the squared error of a flat (zero) hypothesis and $J_1$ the
squared error obtained when a *fixed reference shape* -- the model state
estimated at a hand-picked reference beat ``k_REF`` -- is compared to the observation at every
position.  $CR$ peaks wherever a beat matches the reference template.

The two models are

* time   : ``AlssmProd(AlssmPolyJordan(5), AlssmExp(gamma=1))`` on a backward
           window ``[0, b_r1]``,
* channel: ``AlssmPolyJordan(5)`` on a backward window ``[0, b_r2]``,

combined with [`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost] and
filtered with [`RLSAlssm`][lmlib.statespace.rls.RLSAlssm] in steady state
(``filter(y, dim_order=[0, 1])``).  Four reference shapes of increasing length
are evaluated.

"""
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import lmlib as lm


# ----------------------------------------------------------------------
# Load observation (time x channel) from CSV -- no wfdb dependency
# ----------------------------------------------------------------------
CSV_FILE = "y_ecg_st-petersburg-incart-12-lead-arrhythmia-database_I20.csv"
with open(CSV_FILE, newline="") as fh:
    sig_name = next(csv.reader(fh))                 # header row -> lead names
y = np.loadtxt(CSV_FILE, delimiter=",", skiprows=1)
K, M = y.shape
k = np.arange(K)


def hide_rightandtopspine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ----------------------------------------------------------------------
# Per-scale 2-D ALSSM fit + cost-ratio detection
# ----------------------------------------------------------------------
def fit_shape(y, b_r1, b_r2, K_REF, M_REF, g, poly_k1=5, poly_k2=5):
    """Return (nd_cost, xs_ref, J_0, J_A, lcr, cr, peaks) for one reference shape."""
    alssm_k1 = lm.AlssmProd((lm.AlssmPolyJordan(poly_degree=poly_k1),
                             lm.AlssmExp(gamma=1)))         # time
    alssm_k2 = lm.AlssmPolyJordan(poly_degree=poly_k2)      # channel
    seg_k1 = lm.Segment(a=0, b=b_r1, direction=lm.BACKWARD, g=g)
    seg_k2 = lm.Segment(a=0, b=b_r2, direction=lm.BACKWARD, g=g)
    nd_cost = lm.NDCompositeCost([lm.CostSegment(alssm_k1, seg_k1),
                                  lm.CostSegment(alssm_k2, seg_k2)])

    rls = lm.RLSAlssm(nd_cost, steady_state=True,backend='lfilter')   
    rls.filter(y, dim_order=[0, 1])

    xhat = rls.minimize_x()                                 # (K, M, N)
    xs_ref = xhat[K_REF, M_REF, :]                          # reference state

    J_0 = rls.eval_errors(np.zeros_like(xhat))             # flat hypothesis
    xs_A = np.tile(xs_ref[None, None, :], (K, M, 1))       # fixed-shape hypothesis
    J_A = rls.eval_errors(xs_A)

    lcr = -0.5 * np.log(J_A / J_0)
    lcr = lcr / np.nanmax(lcr[:, M_REF])
    cr = J_0[:, M_REF] / J_A[:, M_REF]

    peaks, _ = find_peaks(cr, height=CR_THD, distance=b_r1 * 3)
    peaks = peaks[(peaks < K_REF - b_r1) | (peaks > K_REF + b_r1)]
    peaks = np.append(peaks, K_REF)
    cr = cr / np.median(cr[peaks])
    return nd_cost, xs_ref, J_0, J_A, lcr, cr, peaks


# ----------------------------------------------------------------------
# Reference shapes (coarse-to-fine) and plot setup
# ----------------------------------------------------------------------
l_b_r1 = [15, 30, 30, 30]          # time half-window (backward)
l_K_REF = [88, 152, 218, 279]      # reference-beat time index
l_b_r2 = [6, 6, 6, 6]              # channel window
l_M_REF = [0, 0, 0, 0]             # reference channel
l_g = np.multiply(l_b_r1, 1.0)     # window decay
l_colors = ["xkcd:blue", "xkcd:periwinkle", "xkcd:spruce", "xkcd:green"]
l_CR_THD = [2.5, 2, 5, 5]
n_shapes = len(l_b_r1)
roman = ["I", "II", "III", "IV"]

offsets = (np.arange(M) * np.max(y)) * 0.8

hr = np.concatenate(([10], np.repeat(1, n_shapes)))
fig_multi, axs2 = plt.subplots(n_shapes + 1, 1, sharex=True, figsize=(9.6, 4.8),
                              gridspec_kw={"height_ratios": hr},
                              layout="constrained")

fig_detail, axs1 = plt.subplots(3, 1, sharex=True, figsize=(4.8, 4.8),
                             gridspec_kw={"height_ratios": [2, 0.7, 0.7]},
                             layout="constrained")

# ----------------------------------------------------------------------
# Detect + plot, one reference shape at a time
# ----------------------------------------------------------------------
for index, (b_r1, b_r2, K_REF, M_REF, g, color, CR_THD) in enumerate(
        zip(l_b_r1, l_b_r2, l_K_REF, l_M_REF, l_g, l_colors, l_CR_THD)):

    nd_cost, xs_ref, J_0, J_A, lcr, cr, peaks = fit_shape(
        y, b_r1, b_r2, K_REF, M_REF, g)
    ki = roman[index]

    # ---- right figure: shared observation panel (drawn once) -------------
    if index == 0:
        axs2[0].plot(k, y + offsets, c="xkcd:gray", lw=1,
                     label=["$y$"] + [""] * (M - 1))
        for m_ in range(M):
            axs2[0].text(10, 0.1 + offsets[m_], sig_name[m_])
        hide_rightandtopspine(axs2[0])

    # reference-shape trajectory mapped onto the full (K, M) grid via eval_y
    mapped = lm.Trajectory.eval_y(nd_cost, xs_ref, [K_REF, M_REF], [K, M])
    axs2[0].plot(k, mapped + offsets, c=color, lw=1.2,
                 label=[fr"$\hat{{y}}_{{\bullet-k_{{{ki}}}}}$"] + [""] * (M - 1))

    # ---- right figure: per-shape CR sub-panel ----------------------------
    top_cr = 1.0
    crplt = np.minimum(cr, top_cr)
    axs2[index + 1].set_ylim(bottom=0, top=top_cr + 0.25)
    axs2[index + 1].plot(k, crplt, c="xkcd:black", lw=1, label=fr"CR$_{{{ki}}}$")
    for xp in peaks:
        mark = "*" if xp == K_REF else 7
        yval = top_cr if cr[xp] > top_cr else (cr[xp] if xp == K_REF else cr[xp] + 0.07)
        axs2[index + 1].scatter(xp, yval, marker=mark, c=color, s=20, zorder=4)
        axs2[index + 1].axvline(x=xp, ls="--", c="k", lw=0.5)
        axs2[0].axvline(x=xp, ls="--", c="k", lw=0.5)
    axs2[index + 1].legend(loc="center right")
    hide_rightandtopspine(axs2[index + 1])
    axs2[n_shapes].set(xlabel=r"$k_1$")
    axs2[n_shapes].set_xlim(left=0, right=K)

    # ---- left figure: detailed view of the first reference shape ---------
    if index == 0:
        kplot = np.arange(0, 750)
        axs1[0].plot(kplot, y[kplot] + offsets, c="xkcd:gray", lw=1.25,
                     label=["$y$"] + [""] * (M - 1))
        for m_ in range(M):
            axs1[0].text(52, np.max(y[52:82, m_]) + 0.1 + offsets[m_], sig_name[m_])
        axs1[0].plot(kplot, mapped[kplot] + offsets, c=color, lw=1.25,
                     label=[fr"$\hat{{y}}_{{\bullet-k_{{{ki}}}}}$"] + [""] * (M - 1))
        axs1[0].legend(loc="upper right")
        axs1[0].set_xlim(left=50, right=kplot.max() + 1)
        hide_rightandtopspine(axs1[0])

        crsingle = J_0[:, M_REF] / J_A[:, M_REF]
        axs1[1].plot(kplot, crsingle[kplot], c="xkcd:black", lw=1, label=fr"CR$_{{{ki}}}$")
        axs1[1].legend(loc="upper right", bbox_to_anchor=(1, 1.2))
        axs1[1].set_ylim(bottom=-0.1)
        hide_rightandtopspine(axs1[1])

        axs1[2].plot(kplot, J_0[kplot, M_REF], c="xkcd:black", ls="--", lw=1.0, label="$J_0$")
        axs1[2].plot(kplot, J_A[kplot, M_REF], c="xkcd:black", ls="-", lw=1.0, label="$J_1$")
        axs1[2].legend(loc="upper right", bbox_to_anchor=(1, 1.2))
        axs1[2].set(xlabel=r"$k_1$")
        hide_rightandtopspine(axs1[2])

        for xp in peaks:
            for ax in axs1:
                ax.axvline(x=xp, ls=(":" if xp == K_REF else "--"), c="k", lw=0.5)
            mark = "*" if xp == K_REF else 7
            axs1[1].scatter(xp, crsingle[xp], marker=mark, c=color, s=20)

        # ---- optional 3-D view of the reference-shape fit (window surface)
        offs, surf = lm.Trajectory.eval(nd_cost, xs_ref)     # (b_r1+1, b_r2+1)
        rk = np.arange(K_REF, K_REF + b_r1)
        rm = np.arange(M_REF, M_REF + b_r2)
        surf_win = surf[0:b_r1, 0:b_r2]
        fig3d = plt.figure(figsize=(7, 5))
        ax3d = fig3d.add_subplot(projection="3d")
        rrk, rrm = np.meshgrid(rk, rm)
        for jj, col in enumerate(rm):
            ax3d.plot(rk, np.full_like(rk, col), y[rk, col], c="k", lw=0.6)
        ax3d.plot_surface(rrk, rrm, surf_win.T, color="xkcd:blue", alpha=0.6, shade=False)
        ax3d.plot_wireframe(rrk, rrm, surf_win.T, colors="xkcd:blue", lw=0.5)
        ax3d.plot([], [], [], c="k", label=r"$y$")
        ax3d.plot([], [], [], c="xkcd:blue", label=r"$\hat{y}$")
        ax3d.set_xlabel(r"$k_1$"); ax3d.set_ylabel(r"$k_2$"); ax3d.set_zlabel("amplitude")
        ax3d.legend(loc="upper right")
        ax3d.set_title(f"Reference shape {ki}: observation vs. ALSSM fit")

axs2[0].legend(loc="upper right")
fig_detail.suptitle("Multi-channel ECG: reference-shape detail")
fig_multi.suptitle("Multi-channel ECG: multi-shape cost-ratio detection")

plt.show()
