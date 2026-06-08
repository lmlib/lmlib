"""
Savitzky Golay filter [ex124.0]
======================================================

"""
import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from lmlib.utils.generator import gen_rect
import time

# --- Generating test signal ---
K = 2000
k = np.arange(K)
y_clean = gen_rect(K, 500, 250)

# --- Adding noise ---
noise = np.random.normal(0, 0.1, K)
y = y_clean + noise

# --- Savitzky-Golay Parameter ---
window_length = 21      # has to be an odd number (L = 2*k + 1)
poly_degree = 2           

half_window = window_length // 2  # k = 50 für window_length = 101

# --- Polynom-ALSSM (Grad > 0 für Savitzky-Golay) ---
alssm_poly = lm.AlssmPoly(poly_degree=poly_degree, force_MC=False)

# --- Segment-Konfiguration (endlicher Support) ---
# Linkes Segment: von -(half_window) bis -1 (vor dem aktuellen Punkt)
segment_left = lm.Segment(
    a=-half_window, 
    b=-1, 
    direction=lm.FORWARD, 
    g=1000  # Gewichtung
)

# Rechtes Segment: von 0 bis +half_window (inklusive aktueller Punkt)
segment_right = lm.Segment(
    a=0, 
    b=half_window, 
    direction=lm.BACKWARD, 
    g=1000
)

# --- CompositeCost für symmetrisches Fenster ---
costs = lm.CompositeCost(
    (alssm_poly,), 
    (segment_left, segment_right), 
    F=[[1, 1]]  # Kombiniert beide Segmente
)

# --- Filterung durchführen ---
rls = lm.RLSAlssm(costs, steady_state=False, calc_W=True)
y_hat = rls.fit(y)

# --- Optional: Ableitungen berechnen (Savitzky-Golay Stärke!) ---
# Für die erste Ableitung ändern wir die Ausgabematrix F
# F=[[0, 1]] extrahiert den linearen Koeffizienten (erste Ableitung)
costs_deriv = lm.CompositeCost(
    (alssm_poly,), 
    (segment_left, segment_right), 
    F=[[0, 1]]  # Erste Ableitung
)
rls_deriv = lm.RLSAlssm(costs_deriv, steady_state=False, calc_W=True)
y_deriv = rls_deriv.fit(y)

# --- Visualisierung ---
fig, axes = plt.subplots(3, 1, sharex='all', figsize=(12, 8))

# Plot 1: Original vs. geglättetes Signal
axes[0].plot(k, y, lw=0.5, c='gray', alpha=0.7, label='Rauschsignal $y$')
axes[0].plot(k, y_clean, lw=1, c='green', alpha=0.7, label='Original $y_{clean}$', linestyle='--')
axes[0].plot(k, y_hat, lw=1.5, c='blue', label=f'Savitzky-Golay ($L={window_length}, m={poly_degree}$)')
axes[0].legend(loc='upper right')
axes[0].set_title('Savitzky-Golay Glättung')
axes[0].grid(True, alpha=0.3)

# Plot 2: Ableitung des Signals
axes[1].plot(k, np.gradient(y_clean), lw=1, c='green', alpha=0.7, label='Analytische Ableitung', linestyle='--')
axes[1].plot(k, y_deriv, lw=1.5, c='red', label='SG-Ableitung')
axes[1].legend(loc='upper right')
axes[1].set_title('Erste Ableitung (direkt aus Filterkoeffizienten)')
axes[1].grid(True, alpha=0.3)

# Plot 3: Fehleranalyse
error = y_hat - y_clean
axes[2].plot(k, error, lw=0.8, c='purple')
axes[2].axhline(0, color='black', linewidth=0.5)
axes[2].set_title('Glättungsfehler (geglättet - Original)')
axes[2].grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlabel('Sample index k')

plt.tight_layout()
plt.show()

# --- Vergleich mit verschiedenen Parametern ---
print(f"Fensterlänge: {window_length}")
print(f"Polynomgrad: {poly_degree}")
print(f"Anzahl Koeffizienten: {poly_degree + 1}")
print(f"Effektive Glättung: ~{window_length} Samples")