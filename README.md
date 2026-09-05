# Stratified Turbulence Closures: Staged Calibration of Tensor-Basis Models

An implementation and validation framework for **staged vs. direct calibration of tensor-basis turbulence closures**, applied to constant-coefficient Pope expansions on a synthetic square duct benchmark.

---

## Overview

Data-driven turbulence closure modeling faces a practical challenge: each time a structural modification is made (adding or swapping tensor basis terms), continuous coefficients must be recalibrated. This project investigates whether staging calibration through intermediate subspaces—calibrating a quadratic truncation before extending to cubic—offers measurable advantages over direct joint calibration or an un-staged cold start.

The design space is formalized as a **Stratified Design Atlas**: nested parameter subspaces connected by zero-padding embeddings. In the convex, constant-coefficient setting studied here, this structure is algebraically trivial; the framework's value lies in organizing the design space and establishing an auditable baseline for future extensions to non-nested edits, non-convex losses, and coupled Navier--Stokes evaluation.

---

## Canonical Benchmark: Turbulent Square Duct Flow ($\text{Re}_\tau = 300$)

Turbulent square duct flow exhibits Prandtl's secondary motion of the second kind: counter-rotating streamwise vortices driven by Reynolds stress anisotropy $(\tau_{yy} - \tau_{zz} \neq 0)$ and cross-plane shear $(\tau_{yz} \neq 0)$.

Three nested strata of Pope's tensor integrity basis:
- **Stratum 0 (Linear Boussinesq, $d=1$):** Near-zero stress representability ($\cos\phi_{\text{sec}} \approx 10^{-4}$), anti-aligned vorticity forcing ($\rho = -0.157$, $29.4\%$ reversed). Recovers the classical Speziale (1987) result.
- **Stratum 1 (Quadratic Pope, $d=3$):** Full representability ($\cos\phi_{\text{sec}} = 1.0$ by construction for the synthetic target).
- **Stratum 2 (Cubic Pope, $d=4$):** Adds $T^{(4)} \propto S^2\Omega - \Omega S^2$.

---

## Key Results

### 1. Mechanism: Travel Distance and Coordinate Scaling (Not Gram Conditioning)

A comprehensive optimizer ablation crossing basis scaling (raw vs. unit-normalized) with optimizer choice (convergent plain GD, Adam with and without momentum) isolates the mechanism of realizability excursions:

| Condition | Path B Peak Violation | Rebound Ratio | Final Loss |
|:---|:---|:---|:---|
| Raw Adam (baseline) | 2.00% | 1.68× | $1.3412 \times 10^{-9}$ |
| Raw Adam ($\beta_1 = 0$, no momentum) | 1.30% | 1.05× | $1.4360 \times 10^{-9}$ |
| Raw Plain GD (convergent $\alpha \approx 7324$) | 6.25% | 1.00× | $1.3412 \times 10^{-9}$ |
| Raw Plain GD (inert $\alpha = 2 \times 10^{-3}$) | 0.00% | 1.00× | $1.2263 \times 10^{-7}$ |
| Norm. Adam ($\alpha$ rescaled to $3.6 \times 10^{-4}$) | 12.50% | 2.20× | $1.3412 \times 10^{-9}$ |
| Norm. Adam (fixed $\alpha = 2 \times 10^{-3}$) | 21.79% | 834.0× | $1.3412 \times 10^{-9}$ |

- **Convergent plain GD violates realizability (6.25%):** Excursions occur whenever an optimizer travels through parameter space at a rate sufficient to converge. The 0.00% violation of inert GD was an artifact of remaining stationary.
- **Momentum amplifies overshoot:** Setting $\beta_1 = 0$ reduces Path B's peak violation from 2.00% to 1.30% and flattens the rebound.
- **Basis normalization breaks Adam's scale invariance:** Unit-normalizing magnifies effective parameter steps along small-norm tensors, causing severe overshooting (21.79% violation).
- **Conditioning decomposition:** $\kappa(G_2) = 64.10$ decomposes into scale disparity $(\|T^{(1)}\|/\|T^{(4)}\|)^2 \approx 30.8$ and cross-coupling $R_{34} = -0.714 \to \kappa = 5.99$.

### 2. Traversal Dynamics (Budget-Matched $N = 220$)

- **Path A (Staged):** The intermediate scaffold peaks at 6.08% violation. At the transition ($\delta_R \equiv 0$), the deployed Stratum 2 model **inherits the full 5.64% violation** and repairs essentially monotonically to 1.04%.
- **Path B (Direct):** Starts at 0.00%, experiences a 2.00% transient realizability excursion, and suffers a 1.68× loss rebound (steps 21–28) before reaching 1.04%.
- **Path C (Cold Start, $\theta = \mathbf{0}$):** Achieves the **lowest peak violation in the study (1.22%)**, reaching the shared minimizer without loss rebounds.
- **All paths converge** to the identical global minimizer ($\|\theta_A^* - \theta_B^*\| = 3.16 \times 10^{-5}$, $C_{AB} = 6.93 \times 10^{-6}$), as guaranteed by strict convexity.
- **Conclusion:** In this convex setting, neither staged nor warm-started calibration confers an advantage in convergence speed, endpoint accuracy, or peak violation over an un-staged cold start.

---

## Repository Structure

```
stratified-turbulence-closures/
├── README.md                      # Executive summary, math framework, and reproduction guide
├── requirements.txt               # Dependencies (numpy, scipy, matplotlib)
├── LICENSE                        # MIT License
├── paper/
│   ├── paper.tex                  # Final Edition 2 LaTeX source (revtex4-2)
│   ├── references.bib             # Standalone BibTeX bibliography
│   ├── curvature_results.png      # 4-panel publication figure
│   ├── PAPER_DRAFT.md             # Markdown draft
│   └── figures/                   # Companion physics and ablation figures
│       ├── square_duct_physics.png
│       ├── audit_ablation_results.png
│       └── results_square_duct.png
├── src/
│   ├── prototype_square_duct.py   # Mesh, surrogate fields, and realizability checks
│   ├── curvature_experiment.py    # Main Paths A, B, C traversal and Gram conditioning
│   ├── audit_experiments.py       # 2x2 ablation matrix, cold start, frozen ablation
│   ├── round2_controls.py         # Five decisive controls (restart, beta1=0, GD, alpha)
│   ├── generate_figure_edition2.py # 4-panel publication figure generator
│   ├── plot_secondary_flow.py     # Secondary flow and stress anisotropy visualizer
│   └── plot_results.py            # Baseline convergence visualizer
└── data/
    ├── results_curvature_experiment.json   # Traversal telemetry and penalty sweep
    ├── results_audit_experiments.json      # 2x2 ablation and cold-start telemetry
    ├── results_round2_controls.json        # Decisive controls telemetry
    └── results_square_duct_prototype.json  # Initial prototype results
```

---

## Quickstart & Reproduction

```bash
# Clone the repository
git clone https://github.com/okuhlemadondo/stratified-turbulence-closures.git
cd stratified-turbulence-closures

# Install dependencies
pip install -r requirements.txt

# 1. Run the decisive Round 2 controls:
python src/round2_controls.py

# 2. Run the 2x2 ablation matrix:
python src/audit_experiments.py

# 3. Regenerate the publication figure:
python src/generate_figure_edition2.py
```
