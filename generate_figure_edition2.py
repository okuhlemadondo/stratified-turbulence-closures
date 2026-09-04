"""
generate_figure_edition2.py

Generates the comprehensive 4-panel publication figure incorporating all Round 2 controls:
Panel (a): Loss trajectories for Path A (staged), Path B (direct), Path C (cold start).
Panel (b): Realizability violation trajectories for Path A, Path B, Path C.
Panel (c): Ablation bar chart isolating step size, momentum, and basis scaling.
Panel (d): Gram matrix condition numbers and scale-vs-coupling decomposition.
"""
import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('.cache/matplotlib')

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data
with open('results_curvature_experiment.json') as f:
    orig = json.load(f)
with open('results_audit_experiments.json') as f:
    audit = json.load(f)
with open('results_round2_controls.json') as f:
    r2 = json.load(f)

hA = orig['history_A']
hB = orig['history_B']
hC = audit['experiments']['cold_start']['path_C']
gram = audit['gram_diagnostics']

fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
fig.subplots_adjust(hspace=0.36, wspace=0.28)

cA = '#1F77B4'   # Blue for Path A
cB = '#D62728'   # Red for Path B
cC = '#2CA02C'   # Green for Path C

# --------------------------------------------------------------------------
# Panel (a): Loss trajectories
# --------------------------------------------------------------------------
ax = axes[0, 0]
ax.semilogy(hA['loss'], color=cA, lw=1.8, label='Path A (Staged warm-start)')
ax.semilogy(hB['loss'], color=cB, lw=1.8, label='Path B (Direct warm-start)')
ax.semilogy(hC['loss'], color=cC, lw=1.8, ls='--', label='Path C (Cold start, $\\theta=0$)')

ax.axvline(x=100, color='gray', ls=':', alpha=0.7, lw=1.0)
ax.annotate('Hop $e_2$ (Stratum 1 $\\to$ 2)\n$\\delta_R \\equiv 0$',
            xy=(100, 4.73e-9), xytext=(112, 3e-8),
            fontsize=8, color='#333333',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# Annotate Path B rebound
ax.annotate('Rebound $1.68\\times$\n(steps 21--28)',
            xy=(28, 4.40e-9), xytext=(36, 1.2e-8),
            fontsize=8, color=cB,
            arrowprops=dict(arrowstyle='->', color=cB, lw=0.8))

ax.set_xlabel('Gradient step', fontsize=9.5)
ax.set_ylabel('Total Loss $L(\\theta)$', fontsize=9.5)
ax.set_title('(a) Calibration loss trajectories ($N = 220$, $\\lambda = 150$)', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8.5, loc='upper right')
ax.set_xlim(0, 220)
ax.grid(True, which='both', ls=':', alpha=0.4)

# --------------------------------------------------------------------------
# Panel (b): Realizability violation trajectories
# --------------------------------------------------------------------------
ax = axes[0, 1]
ax.plot([v * 100 for v in hA['viol']], color=cA, lw=1.8, label='Path A (Staged)')
ax.plot([v * 100 for v in hB['viol']], color=cB, lw=1.8, label='Path B (Direct)')
ax.plot([v * 100 for v in hC['viol']], color=cC, lw=1.8, ls='--', label='Path C (Cold start)')

# Shaded scaffold
ax.axvspan(0, 100, alpha=0.08, color=cA, label='Scaffold phase (Stratum 1)')
ax.axvline(x=100, color='gray', ls=':', alpha=0.7, lw=1.0)

ax.annotate('Deployed Stratum 2 inherits\n$5.64\\%$ violation at hop',
            xy=(100, 5.64), xytext=(115, 4.5),
            fontsize=8, color=cA,
            arrowprops=dict(arrowstyle='->', color=cA, lw=0.8))

ax.annotate('Peak violation $2.00\\%$\n(step 25)',
            xy=(25, 2.00), xytext=(35, 2.6),
            fontsize=8, color=cB,
            arrowprops=dict(arrowstyle='->', color=cB, lw=0.8))

ax.annotate('Peak violation $1.22\\%$\n(Cold start)',
            xy=(28, 1.22), xytext=(45, 0.4),
            fontsize=8, color=cC,
            arrowprops=dict(arrowstyle='->', color=cC, lw=0.8))

ax.set_xlabel('Gradient step', fontsize=9.5)
ax.set_ylabel('Realizability violation fraction (%)', fontsize=9.5)
ax.set_title('(b) Realizability dynamics: scaffold inheritance vs direct excursions', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8.5, loc='upper right')
ax.set_xlim(0, 220)
ax.set_ylim(-0.2, 7.0)
ax.grid(True, ls=':', alpha=0.4)

# --------------------------------------------------------------------------
# Panel (c): Ablation bar chart (mechanism breakdown)
# --------------------------------------------------------------------------
ax = axes[1, 0]
conditions = [
    'Raw Adam\n(baseline)',
    'Raw Adam\n($\\beta_1 = 0$)',
    'Raw GD\n(conv. $\\alpha$)',
    'Norm. Adam\n($\\alpha$ resc.)',
    'Norm. Adam\n(fixed $\\alpha$)',
]

peak_viols = [
    audit['experiments']['raw_adam']['path_B']['max_viol'] * 100,
    max(r2['control_3_no_momentum']['path_B']['viol']) * 100,
    max(r2['control_4_convergent_gd']['path_B']['viol']) * 100,
    max(r2['control_5_rescaled_alpha']['path_B']['viol']) * 100,
    audit['experiments']['norm_adam']['path_B']['max_viol'] * 100,
]

bar_colors = ['#E08214', '#FDB863', '#542788', '#8073AC', '#B35806']

bars = ax.bar(range(5), peak_viols, color=bar_colors, edgecolor='black', linewidth=0.6, width=0.65)
for i, v in enumerate(peak_viols):
    ax.text(i, v + 0.4, f'{v:.2f}%', ha='center', fontsize=8.5, fontweight='bold')

ax.set_xticks(range(5))
ax.set_xticklabels(conditions, fontsize=8.5)
ax.set_ylabel('Path B peak realizability violation (%)', fontsize=9.5)
ax.set_title('(c) Mechanism ablation: optimizer, momentum, and basis scaling', fontsize=10.5, fontweight='bold')
ax.set_ylim(0, 25.0)
ax.grid(True, axis='y', ls=':', alpha=0.4)

# --------------------------------------------------------------------------
# Panel (d): Gram matrix conditioning & scale-vs-coupling decomposition
# --------------------------------------------------------------------------
ax = axes[1, 1]
strata = ['Stratum 0\n($d=1$)', 'Stratum 1\n($d=3$)', 'Stratum 2\n($d=4$)']

kappas_raw = [1.0, gram['kappa_block_01'], gram['kappa_full']]

G = np.array(gram['G_full'])
norms = gram['L2_norms']
G1_norm = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        G1_norm[i, j] = G[i, j] / (norms[i] * norms[j])
eigs_G1_norm = np.linalg.eigvalsh(G1_norm)
kappa_G1_norm = float(eigs_G1_norm[-1] / eigs_G1_norm[0])

kappas_norm = [1.0, kappa_G1_norm, gram['kappa_unit_normalized']]

x = np.arange(3)
w = 0.32
bars_raw = ax.bar(x - w/2, kappas_raw, w, label='Raw basis $\\kappa(G)$',
                  color='#3182BD', edgecolor='black', linewidth=0.6)
bars_norm = ax.bar(x + w/2, kappas_norm, w, label='Unit-normalized $\\kappa(\\tilde{G})$',
                   color='#FD8D3C', edgecolor='black', linewidth=0.6)

ax.set_xticks(x)
ax.set_xticklabels(strata, fontsize=9)
ax.set_ylabel('Condition number $\\kappa$', fontsize=9.5)
ax.set_title('(d) Gram conditioning: scale disparity vs cross-coupling', fontsize=10.5, fontweight='bold')
ax.set_yscale('log')
ax.set_ylim(0.7, 180)
ax.legend(fontsize=8.5, loc='upper left')
ax.grid(True, which='both', axis='y', ls=':', alpha=0.4)

for i, (kr, kn) in enumerate(zip(kappas_raw, kappas_norm)):
    ax.text(x[i] - w/2, kr * 1.35, f'{kr:.1f}', ha='center', fontsize=8, fontweight='bold')
    ax.text(x[i] + w/2, kn * 1.35, f'{kn:.2f}', ha='center', fontsize=8, fontweight='bold')

# Annotation explaining decomposition
ax.text(0.97, 0.22,
        'Stratum 2 decomposition:\n'
        '• Scale disparity $(||T^{(1)}||/||T^{(4)}||)^2 \\approx 30.8$\n'
        '• Cross-coupling $R_{34} = -0.714 \\to \\kappa = 5.99$\n'
        '• Joint $\\kappa \\approx 30.8 \\times 2.08 = 64.10$',
        transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='gray', alpha=0.9))

plt.savefig('curvature_results.png', dpi=250, bbox_inches='tight')
print("Successfully generated revised curvature_results.png")
plt.close()
