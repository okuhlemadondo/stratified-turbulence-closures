# OpDiscovery-FluidMech: Staged Calibration of Tensor-Basis Turbulence Closures

An implementation and validation framework for **staged warm-starting of turbulence closure models**, applied to constant-coefficient Pope tensor-basis closures on a synthetic square duct benchmark.

## Overview

Data-driven turbulence closure modeling faces a practical challenge: each time a structural modification is made (adding or swapping tensor basis terms), the continuous coefficients must be recalibrated. This project investigates whether staging calibration through intermediate subspaces—calibrating a quadratic truncation before extending to cubic—offers measurable advantages over direct joint calibration or cold starts.

The design space is formalized as a **Stratified Design Atlas**: nested parameter subspaces connected by zero-padding embeddings. In the convex, constant-coefficient setting studied here, this structure is algebraically trivial; the framework's value lies in organizing the design space for future extensions to non-nested edits and non-convex losses.

## Canonical Benchmark: Turbulent Square Duct Flow ($\text{Re}_\tau = 300$)

Turbulent square duct flow exhibits Prandtl's secondary motion of the second kind: counter-rotating streamwise vortices driven by Reynolds stress anisotropy $(\tau_{yy} - \tau_{zz} \neq 0)$ and cross-plane shear $(\tau_{yz} \neq 0)$.

Three nested strata of Pope's tensor integrity basis:
- **Stratum 0 (Linear Boussinesq, $d=1$):** Near-zero stress representability ($\cos\phi_{\text{sec}} \approx 10^{-4}$), anti-aligned vorticity forcing ($\rho = -0.157$, 29.4% reversed). Recovers the classical Speziale (1987) result.
- **Stratum 1 (Quadratic Pope, $d=3$):** Full representability ($\cos\phi_{\text{sec}} = 1.0$ by construction for the synthetic target).
- **Stratum 2 (Cubic Pope, $d=4$):** Adds $T^{(4)} \propto S^2\Omega - \Omega S^2$.

## Key Results

### 1. Mechanism: Adam Momentum Overshoot (Not Gram Conditioning)

A $2\times 2$ ablation matrix crossing basis scaling (raw vs. unit-normalized) with optimizer choice (Adam vs. plain GD) identifies the transient instability mechanism:

| Condition | Path B Peak Violation | Rebound Ratio |
|:---|:---|:---|
| Raw Basis + Adam | 2.00% | 1.7× |
| Normalized + Adam | 21.79% | 834× |
| Raw Basis + Plain GD | 0.00% | 1.0× |
| Normalized + Plain GD | 0.61% | 1.0× |

- **Plain GD eliminates all violations** regardless of basis scaling.
- **Normalizing the basis makes Adam worse**, not better (opposite of Gram-conditioning prediction).
- The raw $\kappa = 64.1$ drops to $6.0$ after unit-normalizing (scaling artifact, not fundamental obstruction).

### 2. Traversal Dynamics (Budget-Matched $N = 220$)

- **Path A (Staged):** Scaffold violations (5.64% peak) are confined to Stratum 1; deployed Stratum 2 model repairs essentially monotonically.
- **Path B (Direct):** 2.00% peak violation, 1.39× loss rebound (steps 21–27) driven by Adam momentum overshooting the penalty barrier.
- **Path C (Cold Start):** 1.22% peak violation—lower than both warm-starts. Converges to the same final loss ($1.34 \times 10^{-9}$).
- **All paths converge** to the same minimizer ($C_{AB} = 6.93 \times 10^{-6}$), as guaranteed by convexity.

### 3. Gram Matrix Block Structure

The Gram matrix decouples exactly into $(\theta_0, \theta_1)$ and $(\theta_2, \theta_3)$ blocks (algebraic identity: $\text{Tr}(A[B,\Omega]) = 0$ for symmetric polynomials $A, B$ in $S$). Within the $(\theta_2, \theta_3)$ block, $R_{34} = -0.714$ (substantial cross-coupling).

## Repository Structure

- [`paper.tex`](paper.tex): Complete LaTeX manuscript (revtex4-2, audit-revised).
- [`audit_experiments.py`](audit_experiments.py): $2\times 2$ ablation matrix, cold-start baseline, frozen-coefficient ablation.
- [`results_audit_experiments.json`](results_audit_experiments.json): Full ablation telemetry.
- [`curvature_experiment.py`](curvature_experiment.py): Original 3-stratum budget-matched experiment.
- [`results_curvature_experiment.json`](results_curvature_experiment.json): Original traversal telemetry (Path A, Path B, λ-sweep).
- [`prototype_square_duct.py`](prototype_square_duct.py): Baseline square duct environment, mesh, and representability metrics.
- [`plot_secondary_flow.py`](plot_secondary_flow.py): Secondary flow streamlines and normal stress anisotropy visualization.
- [`audit_ablation_results.png`](audit_ablation_results.png): 6-panel ablation diagnostic figure.
- [`curvature_results.png`](curvature_results.png): Original 4-panel figure.
