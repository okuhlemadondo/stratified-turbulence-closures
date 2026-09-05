# Stratified Turbulence Closures: Staged Calibration of Tensor-Basis Models

An implementation and verification framework for **staged vs. direct calibration of tensor-basis turbulence closures**, applied to constant-coefficient Pope expansions on a synthetic square duct benchmark ($\text{Re}_\tau = 300$).

---

## Overview

Data-driven turbulence closure modeling faces a fundamental challenge: when structural modifications are made to a model (e.g., adding or mutating tensor basis terms), continuous parameters must be recalibrated. This project investigates whether staging calibration through intermediate subspaces—calibrating a quadratic truncation before extending to cubic—offers measurable advantages over direct joint calibration or an un-staged cold start.

The design space is formalized as a **Stratified Design Atlas**: nested parameter manifolds connected by zero-padding embeddings. In the convex, constant-coefficient setting studied here, zero-padding preserves physical predictions identically ($\delta_R \equiv 0$). Consequently, the deployed model inherits whatever realizability violations were accumulated by the intermediate scaffold.

---

## Canonical Benchmark: Turbulent Square Duct Flow ($\text{Re}_\tau = 300$)

Turbulent square duct flow exhibits Prandtl's secondary motion of the second kind: counter-rotating streamwise vortices driven by Reynolds stress anisotropy $(\tau_{yy} - \tau_{zz} \neq 0)$ and cross-plane shear $(\tau_{yz} \neq 0)$.

Three nested strata of Pope's tensor integrity basis:
- **Stratum 0 (Linear Boussinesq, $d=1$):** Near-zero stress representability ($\cos\phi_{\text{sec}} \approx 1.15 \times 10^{-4}$), anti-aligned vorticity forcing ($\rho = -0.157$, $29.38\%$ reversed). Recovers the classical Speziale (1987) result.
- **Stratum 1 (Quadratic Pope, $d=3$):** Full representability ($\cos\phi_{\text{sec}} = 1.000$). At the $\lambda=150$-regularized scaffold iterate after 100 steps, $\rho = +0.294$ ($20.36\%$ reversed).
- **Stratum 2 (Cubic Pope, $d=4$):** Adds $T^{(4)} \propto S^2\Omega - \Omega S^2$. Fully calibrated endpoint achieves $\rho = +0.999$ ($1.65\%$ reversed).

---

## Key Findings

### 1. Mechanism: Travel Distance and Coordinate Scaling (Not Gram Conditioning)

The initial conjecture that Gram matrix ill-conditioning ($\kappa(G_2) = 64.10$) governs transient loss rebounds is disproven by an extensive ablation and control suite:
- **Conditioning Decomposition:** The raw Gram condition number $\kappa(G_2) = 64.10$ decomposes into a basis scale disparity $(\|T^{(1)}\|/\|T^{(4)}\|)^2 = 30.82$ and cross-coupling $R_{34} = -0.714$, which inflates conditioning by a factor of $2.08$ to yield $\kappa(\tilde{G}_2) = 5.99$ upon unit-normalization ($30.82 \times 2.08 = 64.10$).
- **Convergent Plain GD Violates Realizability ($6.25\%$):** Operating plain gradient descent at a convergent step size ($\alpha \approx 7324$) produces a $6.25\%$ peak violation, demonstrating that excursions are an intrinsic consequence of parameter travel distance across the landscape. The $0.00\%$ violation of default GD was solely due to stationarity ($\alpha = 2 \times 10^{-3}$).
- **Momentum Amplifies Overshoot:** Setting $\beta_1 = 0$ in Adam reduces peak violation from $2.00\%$ to $1.30\%$ at nominal step size ($\alpha = 2 \times 10^{-3}$), while matching convergence rate at 76 steps isolates a $1.27\times$ momentum amplification factor ($1.65\%$ vs. $1.30\%$). Extended optimization of $\beta_1 = 0$ exhibits a slow monotone crawl along the shallow Gram direction ($1.4360 \times 10^{-9} \to 1.4324 \times 10^{-9}$) rather than achieving matched loss.
- **Scale Invariance Restored:** While global scalar rescaling ($\alpha = 3.6 \times 10^{-4}$) is insufficient ($12.50\%$ violation), per-coordinate Adam ($\alpha_n = \alpha \|T^{(n)}\|$, $\epsilon_n = \epsilon / \|T^{(n)}\|$) restores exact coordinate-wise scale invariance to machine precision (relative loss discrepancy $< 2.1 \times 10^{-15}$, parameter difference $< 3.0 \times 10^{-17}$, matching raw Adam's $2.00\%$ peak).

| Condition | Path B Peak Viol. (%) | Path B Rebound | Path C Peak Viol. (%) | Path C Rebound | Final Loss |
|:---|:---:|:---:|:---:|:---:|:---:|
| Raw Adam (baseline) | 2.00% | 1.68× | 1.22% | N/A | $1.3412 \times 10^{-9}$ |
| Raw Adam ($\beta_1 = 0$, no momentum) | 1.30% | 1.05× | — | — | $1.4360 \times 10^{-9}$ |
| Raw Plain GD (convergent $\alpha \approx 7324$) | 6.25% | 1.00× | — | — | $1.3412 \times 10^{-9}$ |
| Raw Plain GD (inert $\alpha = 2 \times 10^{-3}$) | 0.00% | 1.00× | — | — | $1.2263 \times 10^{-7}$ |
| Norm. Adam (per-coord $\alpha_n, \epsilon_n$) | 2.00% | 1.68× | — | — | $1.3412 \times 10^{-9}$ |
| Norm. Adam (scalar $\alpha$ resc., insufficient) | 12.50% | 2.20× | — | — | $1.3412 \times 10^{-9}$ |
| Norm. Adam (fixed $\alpha = 2 \times 10^{-3}$) | 21.79% | 834.0× | — | — | $1.3412 \times 10^{-9}$ |

*Note: For Path C ($\theta = \mathbf{0}$), descent is monotonic without rebound (marked N/A).*

### 2. Decisive Controls & Mechanistic Attribution

- **Matched-Loss Path B Restart (Step 18):** When Path B is restarted at step 18 where its loss first drops below Path A's hop loss ($4.73 \times 10^{-9}$) to $4.34 \times 10^{-9}$ (preceded by $5.84 \times 10^{-9}$ at step 17), the post-restart peak violation is **$1.74\%$** at step 21 (rising from $0.95\%$ at step 17), while the loss continues its monotonic descent toward tolerance without an additional rebound. This confirms that travel distance across the ill-conditioned landscape dominates the excursion, while continuous momentum provides an incremental $+0.26\%$ amplification ($1.74\% \to 2.00\%$), and that the restart merely interrupted an in-progress boundary traversal rather than preventing one.
- **Path A Carry-Over vs. Fresh Restart:** Path A with momentum carry-over deploys at **$5.64\%$** violation (never exceeding the inherited value). Path A with fresh restart exhibits a deployed peak of **$5.90\%$** at step 101. Carry-over is slightly *better* than restart because Adam's cold update at $v = 0$ acts as a full-magnitude sign step pushing the model deeper into the penalty barrier.
- **Matched-Convergence-Rate Momentum Ablation:** Comparing full Adam and Adam with $\beta_1 = 0$ at matched nominal step size ($\alpha = 2 \times 10^{-3}$) partially conflates momentum with effective traversal rate ($(1-\beta_1)^{-1} \approx 10$). At a matched convergence rate of **76 steps** to tolerance ($L < 1.4 \times 10^{-9}$), full Adam ($\alpha = 1.37 \times 10^{-3}$) incurs a **$1.65\%$** peak violation, compared to **$1.30\%$** for $\beta_1 = 0$ ($\alpha = 2 \times 10^{-3}$), isolating a **$1.27\times$** momentum amplification factor. Extended optimization of $\beta_1 = 0$ reaches $1.4324 \times 10^{-9}$ at 450 steps, exhibiting a slow monotone crawl along the shallow Gram direction above the baseline target.
- **Convergent Plain Gradient Descent:** Operating at a step size matched to the Lipschitz constant ($\alpha = 1/\lambda_{\max} \approx 7324$), Plain GD converges to $1.3412 \times 10^{-9}$ but traverses unrealizable states at **$6.25\%$** peak violation for Path B and **$6.16\%$** for Path A, confirming that excursions reflect traversal speed rather than optimizer-specific adaptivity.
- **Scale Invariance Proof:** Under unit basis normalization with per-coordinate scaling ($\alpha_n = \alpha \|T^{(n)}\|$, $\epsilon_n = \epsilon / \|T^{(n)}\|$), Adam is scale-invariant to machine precision (relative loss difference $< 3 \times 10^{-15}$, parameter difference $< 3 \times 10^{-17}$).

### 3. Traversal Dynamics and Protocol Performance

- **Path A (Staged):** Scaffold peaks at $6.08\%$. At the hop ($\delta_R \equiv 0$), the deployed model inherits $5.64\%$, rises to $5.90\%$ at step 101, descends to $0.95\%$ at step 150, and rises mildly to $1.04\%$ at step 219. Staging does not eliminate boundary violations; it inherits them, and is the slowest overall protocol ($123$ steps to tolerance).
- **Path B (Direct Warm-Start):** Direct calibration from Stratum 0 achieves the fastest convergence ($59$ steps to tolerance, saving $42$ steps over cold start), but incurs a transient peak of $2.00\%$ and a $1.68\times$ rebound before converging to $1.04\%$.
- **Path C (Cold Start, $\theta = \mathbf{0}$):** Achieves the **lowest peak violation in the study ($1.22\%$)**, reaching the identical minimizer without loss rebounds. Reaches tolerance in $101$ steps (faster than Path A's $123$ steps). Cold-start is optimal for realizability preservation, while direct warm-start minimizes step count.
- **Theoretical Basis of Realizability Violations:** Lumley's domain is convex in stress space and parameter space for linear bases. Violations reflect:
  1. *Inherited floor:* $\tau_{\text{ref}}$ carries a $1.56\%$ violation ($\min\lambda = -5.10 \times 10^{-5}$); any least-squares fit inherits this floor ($1.48\%$ at $\lambda = 0$, $1.04\%$ at $\lambda = 150$, $0.52\%$ at $\lambda = 1500$).
  2. *Transient excess:* Excursions above this floor are governed by per-step travel distance across parameter coordinates.

### 4. Penalty Modulation Sweep ($\lambda \in \{0, 150, 1500\}$)

| $\lambda$ | A scaf. (%) | A depl. (%) | A end (%) | B peak (%) | B end (%) | C peak (%) | C end (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $0$ (unconstrained) | 6.25% | 6.08% | 1.48% | 2.78% | 1.48% | 1.74% | 1.48% |
| $150$ (baseline) | 6.08% | 5.90% | 1.04% | 2.00% | 1.04% | 1.22% | 1.04% |
| $1500$ (stiff) | 5.47% | 2.95% | 0.52% | 1.30% | 0.52% | 0.87% | 0.52% |

---

## Repository Structure

```
stratified-turbulence-closures/
├── Makefile                       # Verification and automation pipeline (make verify)
├── README.md                      # Executive summary, math framework, and reproduction guide
├── requirements.txt               # Dependencies (numpy, scipy, matplotlib)
├── LICENSE                        # MIT License
├── verify_numbers.py              # Automated verification of mathematical identities and numbers
├── docs/
│   └── digit_collisions.md        # Collision audit of committed telemetry values
├── paper/
│   ├── paper.tex                  # Edition 2 LaTeX source (revtex4-2, self-contained)
│   ├── references.bib             # Complete 28-entry BibTeX bibliography
│   ├── numbers.json               # Canonical number binding dictionary
│   ├── curvature_results.png      # 4-panel publication figure
│   ├── PAPER_DRAFT.md             # Synchronized Markdown draft
│   └── figures/
│       ├── pareto_stepsize_sweep.png  # Step-size Pareto frontier (plain GD vs Adam)
│       ├── square_duct_physics.png    # Secondary flow mechanics and mesh
│       ├── audit_ablation_results.png # 2x2 ablation diagnostic figure
│       └── results_square_duct.png    # Baseline convergence
├── src/
│   ├── prototype_square_duct.py   # Mesh, surrogate fields, and realizability checks
│   ├── curvature_experiment.py    # Main Paths A, B, C traversal and Gram conditioning
│   ├── audit_experiments.py       # 2x2 ablation matrix, cold start, frozen ablation
│   ├── round2_controls.py         # Controls (matched restart, carry-over, plain GD)
│   ├── round3_investigations.py   # Round 3 pre-registered investigations (B2, B6, B7, B8)
│   ├── build_numbers_json.py      # Automated generator for paper/numbers.json
│   ├── generate_figure_edition2.py# 4-panel publication figure generator
│   └── plot_secondary_flow.py     # Secondary flow and stress anisotropy visualizer
└── data/
    ├── results_curvature_experiment.json   # Traversal telemetry and penalty sweep
    ├── results_audit_experiments.json      # 2x2 ablation and cold-start telemetry
    ├── results_round2_controls.json        # Round 2 decisive controls telemetry
    ├── results_round3_investigations.json   # Round 3 matched restart, scale inv., Pareto sweep
    └── results_square_duct_prototype.json  # Initial prototype results
```

---

## Verification & Reproduction

All experiments are fully deterministic (`np.random.seed(42)`). Every numerical claim in the manuscript is verified automatically:

```bash
# Run complete verification suite
make verify

# Or directly:
python3 verify_numbers.py
```

To re-run the complete experimental and figure pipeline from scratch:

```bash
# 1. Run core experiments and controls
python3 src/round2_controls.py
python3 src/round3_investigations.py

# 2. Rebuild numbers dictionary
python3 src/build_numbers_json.py

# 3. Regenerate publication figures
python3 src/generate_figure_edition2.py

# 4. Verify all mathematical identities and reported values
python3 verify_numbers.py
```
