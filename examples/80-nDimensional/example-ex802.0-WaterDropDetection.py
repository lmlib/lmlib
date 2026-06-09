#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Scale Water-Drop Detection with 2-D ALSSM / LCR [ex802.0]
======================================================================

Round water drops in a microscope image are detected as local maxima of a
**Log-Cost Ratio (LCR)** between two competing 2-D ALSSM hypotheses, evaluated
at several scales.

For each scale an isotropic 1-D model -- an offset plus two cosine modes,
``AlssmPoly(0) + AlssmSin + AlssmSin`` -- is placed on a two-sided (forward +
backward) window and combined into a separable 2-D model with
[`NDCompositeCost`][lmlib.statespace.cost.NDCompositeCost].  Two constrained
fits are compared:

* ``H1`` -- symmetric "drop" hypothesis (offset + symmetric cosine terms),
* ``H0`` -- flat "background" hypothesis (offset only).

``LCR = -1/2 log(error_H1 / error_H0)`` peaks where a drop-like bump fits much
better than a flat patch.  Detections are accumulated coarse-to-fine: each
scale claims circular regions so finer scales only fire on what is left.

Compared with the original prototype, the hand-written 2-D recursion is
replaced by the library's native ND filtering
(``RLSAlssm(NDCompositeCost(...)).filter(Y, dim_order=[0, 1])``).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

import lmlib as lm


# ----------------------------------------------------------------------
# Helpers (peak finding / coarse-to-fine masking)
# ----------------------------------------------------------------------
def rgb2gray(rgb):
    """Rec. 709 luminance, matching skimage.color.rgb2gray.

    Integer images are scaled to [0, 1] first (as skimage's img_as_float does),
    so downstream LCR thresholds keep the same meaning.
    """
    rgb = np.asarray(rgb)
    if np.issubdtype(rgb.dtype, np.integer):
        rgb = rgb / np.iinfo(rgb.dtype).max     # uint8 JPEG -> [0, 1]
    else:
        rgb = rgb.astype(float, copy=False)
    return rgb[..., :3] @ np.array([0.2125, 0.7154, 0.0721])


def gray2rgb(gray):
    """Stack a 2-D array into 3 identical channels (skimage.color.gray2rgb)."""
    return np.stack((gray,) * 3, axis=-1)
    
def find_2d_peaks(image, height=0.0, width=2):
    """Boolean mask of local maxima above *height* (neighbourhood size *width*)."""
    local_max = maximum_filter(image, size=width) == image
    return local_max & (image > height)


def draw_circles_on_mask(seed_mask, radius):
    """Return a boolean mask with filled discs of *radius* at every seed pixel."""
    ys, xs = np.nonzero(seed_mask)
    yy, xx = np.meshgrid(np.arange(seed_mask.shape[0]),
                         np.arange(seed_mask.shape[1]), indexing='ij')
    out = np.zeros_like(seed_mask, dtype=bool)
    for yc, xc in zip(ys, xs):
        out |= (yy - yc) ** 2 + (xx - xc) ** 2 <= radius ** 2
    return out


# ----------------------------------------------------------------------
# Build the per-scale 2-D ALSSM fit and return LCR, peaks and states
# ----------------------------------------------------------------------
def fit_scale(y, a_l, b_r, L1, g=100, poly_degree=0):
    """Run the 2-D ALSSM fit for one scale; return (lcr, vhat_H1, alssm, N)."""
    alssm_1d = lm.AlssmSum((lm.AlssmPoly(poly_degree),
                            lm.AlssmSin(2 * np.pi / L1),
                            lm.AlssmSin(2 * np.pi / (0.5 * L1))))
    N = alssm_1d.N
    seg_l = lm.Segment(a=a_l, b=-1, direction=lm.FORWARD, g=g)
    seg_r = lm.Segment(a=0, b=b_r, direction=lm.BACKWARD, g=g)
    cost_1d = lm.CompositeCost((alssm_1d,), (seg_l, seg_r), [[1, 1]])
    nd_cost = lm.NDCompositeCost([cost_1d, cost_1d])

    #with warnings.catch_warnings():
        #warnings.simplefilter('ignore')
    if True:
        rls = lm.RLSAlssm(nd_cost, steady_state=True)
        rls.filter(y, dim_order=[0, 1])

        # H1: symmetric offset + cosine model (the "drop" hypothesis)
        I = np.eye(N ** 2)
        H1 = np.zeros((N ** 2, 6))
        H1[:, 0] = I[:, 0]                        # offset
        H1[:, 1] = I[:, 1] + I[:, N + 0]          # cos(omega_1), symmetric
        H1[:, 2] = I[:, 3] + I[:, 3 * N + 0]      # cos(omega_2), symmetric
        H1[:, 3] = I[:, N + 1]                    # cos1 * cos1
        H1[:, 4] = I[:, N + 3] + I[:, 3 * N + 1]  # cos1 * cos2
        H1[:, 5] = I[:, 3 * N + 3]                # cos2 * cos2
        # H0: offset only (the flat "background" hypothesis)
        H0 = np.delete(np.eye(N ** 2), list(range(poly_degree + 1, N ** 2)), axis=1)

        xhat_H1 = rls.minimize_x(H1)
        vhat_H1 = rls.minimize_v(H1)
        xhat_H0 = rls.minimize_x(H0)
        error_H1 = rls.eval_errors(xhat_H1)
        error_H0 = rls.eval_errors(xhat_H0)

    lcr = -0.5 * np.log(np.divide(error_H1, error_H0))
    return lcr, vhat_H1, xhat_H1,  N, nd_cost


# ----------------------------------------------------------------------
# Load image
# ----------------------------------------------------------------------
y = rgb2gray(plt.imread("Image_2979_part.jpg"))
K, M = y.shape
kk, mm = np.meshgrid(np.arange(K), np.arange(M), indexing='ij')

# big reference drop (row, col) for the close-up figure
kdrop, mdrop = 413, 73

# coarse-to-fine scales: (half-window, sine period, LCR height, colour)
scales = [
    dict(a=-40, b=40, L1=80, height=0.7, color='xkcd:dark red'),
    dict(a=-20, b=20, L1=40, height=0.4, color='xkcd:red'),
    dict(a=-10, b=10, L1=20, height=0.4, color='xkcd:orange'),
    dict(a=-7,  b=7,  L1=14, height=0.2, color='xkcd:gold'),
    dict(a=-5,  b=5,  L1=10, height=0.3, color='xkcd:yellowish orange'),
]

# ----------------------------------------------------------------------
# Plot Detection figure
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 9))
ax.imshow(gray2rgb(y))
ax.set_xlabel(r'$k_2$')
ax.set_ylabel(r'$k_1$', rotation=0, labelpad=10, ha='right')

#Plot rectangular box around the big drop
er = 25
ax.plot([mdrop - er, mdrop + er, mdrop + er, mdrop - er, mdrop - er],
        [kdrop - er, kdrop - er, kdrop + er, kdrop + er, kdrop - er],
        lw=1.0, ls='-', c='r')

claimed = np.zeros((K, M), dtype=bool)     # coarse-to-fine occupancy mask

for i, s in enumerate(scales):
    lcr, vhat, xhat, N, _ = fit_scale(y, s['a'], s['b'], s['L1'], poly_degree=0)

    lcr = lcr.copy()
    lcr[claimed] = np.nan                                   # suppress claimed regions
    peaks = find_2d_peaks(lcr, height=s['height'], width=s['b'] - s['a'])

    # keep only physically plausible drops: positive fundamental cosine, and a
    # second mode that is not a large negative (concave) contribution
    ys, xs = np.nonzero(peaks)
    for yc, xc in zip(ys, xs):
        v = vhat[yc, xc, :]
        if (v[1] < 0) or ((abs(v[2]) > abs(v[1])) and v[2] < 0):
            peaks[yc, xc] = False
    print(f"scale {i} (window +/-{s['b']}): {int(peaks.sum())} drops")

    claimed |= draw_circles_on_mask(peaks, radius=s['b'] * 0.5)
    ax.scatter(mm[peaks], kk[peaks], c=s['color'], s=6, marker='.',
               label=f"window $\\pm${s['b']}")

ax.legend(loc='upper right', markerscale=3)
ax.set_title('Multi-scale water-drop detection (LCR peaks, coarse-to-fine)')
fig.tight_layout()

# ----------------------------------------------------------------------
# Plot Trajectory observed surface vs. fitted 2-D ALSSM model at the big drop.
# This uses its own parameter set (window +/-20 but a longer sine period
# L1 = 80).
# ----------------------------------------------------------------------
half = 20
_, _, xhat_conf0, N_conf0, ndcost_conf0 = fit_scale(y, -half, half, L1=80, poly_degree=0)
_, mappedtraj = lm.Trajectory.eval(ndcost_conf0, xhat_conf0[kdrop,mdrop])

rk = np.arange(kdrop - half, kdrop + half + 1)
rm = np.arange(mdrop - half, mdrop + half + 1)

figv = plt.figure(figsize=(8, 6))
axv = figv.add_subplot(projection='3d')
axv.plot([rk.min(), rk.min(), rk.max(), rk.max(), rk.min()],
         [rm.min(), rm.max(), rm.max(), rm.min(), rm.min()],
         np.full(5, 0.13), c='r', lw=1)                # window outline at the base
for jj, col in enumerate(rm):                          # one line per column
    axv.plot(rk, np.full_like(rk, col), y[rk, col],
             c='k', marker='o', lw=0.5, markersize=2)
    #axv.plot(rk, np.full_like(rk, col), model_surface[:, jj],
    #         c='xkcd:blue', marker='o', lw=0.5, markersize=2)
    axv.plot(rk, np.full_like(rk, col), mappedtraj[:, jj],
             c='xkcd:blue', marker='o', lw=0.5, markersize=2)
axv.plot([], [], [], c='k', label=r'$y$')
axv.plot([], [], [], c='xkcd:blue', label=r'$\hat{y}$')
axv.set_zlim([0.13, 1])
axv.set_xlabel(r'$k_1$'); axv.set_ylabel(r'$k_2$'); axv.set_zlabel('amplitude')
axv.legend(loc='upper right')
axv.set_title(f'Big drop at (row {kdrop}, col {mdrop}): observation vs. ALSSM fit')

plt.show()
