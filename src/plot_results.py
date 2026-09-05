"""
plot_results.py

Generates publication-quality figures:
1. Cross-sectional secondary flow streamlines and corner vortex structures.
2. Optimization trajectories comparing the 4 conditions (Cold Start, Re-fit, Exact Transport, Extended Transport).
3. Realization drift and realizability evolution.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
DATA_DIR = REPO_ROOT / "data"
PAPER_DIR = REPO_ROOT / "paper"

def plot_experiment():
    with open(DATA_DIR / "results_square_duct_prototype.json", "r") as f:
        results = json.load(f)
        
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # --------------------------------------------------------------------------
    # Panel 1: Convergence Trajectories (Log Loss vs Gradient Steps)
    # --------------------------------------------------------------------------
    ax1 = axes[0]
    colors = {
        "1. Cold Start": "#d95f02",
        "2. Standard Re-fit": "#7570b3",
        "3. Exact Transport": "#1b9e77",
        "4. Extended Transport": "#e7298a"
    }
    linestyles = {
        "1. Cold Start": "--",
        "2. Standard Re-fit": "-.",
        "3. Exact Transport": "-",
        "4. Extended Transport": ":"
    }
    
    for name, data in results.items():
        loss_hist = data["loss_history"]
        steps = np.arange(len(loss_hist))
        ax1.plot(steps, loss_hist, label=name, color=colors[name], 
                 linestyle=linestyles[name], linewidth=2.0)
        
    ax1.set_yscale('log')
    ax1.set_xlabel('Optimization Steps in Stratum 1 (Quadratic Pope)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'Task Loss $L_1(\theta_1) = \frac{1}{2} \|\tau - \tau_{\rm DNS}\|_{L^2}^2$', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Structural Edit Trajectories: Loss vs Steps', fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(frameon=True, fontsize=10)
    
    # --------------------------------------------------------------------------
    # Panel 2: Initial Realization Drift vs Final Error
    # --------------------------------------------------------------------------
    ax2 = axes[1]
    names = list(results.keys())
    drifts = [results[n]["delta_R"] for n in names]
    final_losses = [results[n]["final_loss"] for n in names]
    
    short_names = ["Cold Start", "Standard Re-fit", "Exact Transport\n(Analytical)", "Extended Transport\n(w/ State)"]
    x = np.arange(len(names))
    width = 0.35
    
    color_bars = [colors[n] for n in names]
    bars = ax2.bar(x, drifts, width, color=color_bars, alpha=0.85, edgecolor='black', linewidth=1.2)
    
    # Annotate drift values
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        if yval < 1e-12:
            ax2.text(bar.get_x() + bar.get_width()/2.0, 0.003, 'EXACT 0.0\n(Identity realization)', 
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1b9e77')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f'{yval:.3f}', 
                     ha='center', va='bottom', fontsize=9)
            
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=10, fontweight='bold')
    ax2.set_ylabel(r"Initial Realization Drift $\delta_R = \frac{\|R_{t'}(T_e(\theta)) - R_t(\theta)\|}{\|R_t(\theta)\|}$", 
                   fontsize=11, fontweight='bold')
    ax2.set_title('(b) Semantic Preservation across Structural Transition', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(drifts) * 1.25)
    ax2.grid(axis='y', ls="--", alpha=0.4)
    
    plt.tight_layout()
    output_path = PAPER_DIR / "figures" / "results_square_duct.png"
    plt.savefig(output_path)
    print(f"Figure saved successfully to '{output_path}'.")

if __name__ == "__main__":
    plot_experiment()
