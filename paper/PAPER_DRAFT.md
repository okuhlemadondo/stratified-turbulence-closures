# Staged Calibration of Tensor-Basis Turbulence Closures: Transient Realizability Dynamics in Nested Subspaces

**Okuhle Madondo**  
*Rosetta Technologies*  
`okuhlemadondo@users.noreply.github.com`  

---

## Abstract

In a convex, constant-coefficient, *a priori* calibration of a four-term Pope tensor basis on a synthetic square duct benchmark ($\text{Re}_\tau = 300$), we investigate whether staging calibration through intermediate subspaces offers measurable advantages over direct joint calibration or an un-staged cold start. Because zero-padding preserves predictions identically ($\delta_R \equiv 0$), the deployed Stratum 2 model in the staged protocol inherits the full $5.64\%$ realizability violation of the intermediate scaffold at deployment (peaking at $5.90\%$ during repair due to Adam's cold-update sign-step transient), whereas direct calibration produces a transient excursion peaking at $2.00\%$ violation (accompanied by a $1.68\times$ loss rebound), and an un-staged cold start ($\theta = \mathbf{0}$, Path C) reaches the identical minimizer with a peak violation of only $1.22\%$.

Through an extensive suite of pre-registered controls, we isolate the physical and numerical mechanisms governing these dynamics. In a matched-loss restart control at step 18 (where Path B's loss first drops below Path A's hop loss $4.73 \times 10^{-9}$ to $4.34 \times 10^{-9}$, preceded by $5.84 \times 10^{-9}$ at step 17), resetting momentum yields a post-restart peak violation of $1.74\%$ (**Branch ii occurred**), confirming that travel distance across the ill-conditioned landscape is the dominant driver of realizability excursions, while continuous momentum contributes an incremental $+0.26\%$ amplification ($1.74\% \to 2.00\%$). Comparing Adam with and without first-moment momentum ($\beta_1 = 0$) at a matched convergence rate ($76$ steps to $L < 1.4 \times 10^{-9}$) isolates a $1.27\times$ momentum amplification ($1.65\%$ vs. $1.30\%$), while a step-size sweep shows that traversal speed strongly modulates excursion magnitude. Convergent plain gradient descent traverses unrealizable states at $6.25\%$ (Path B) and $6.16\%$ (Path A). Furthermore, we prove that per-coordinate Adam on a unit-normalized basis preserves exact scale invariance to machine precision (relative loss discrepancy $< 2.1 \times 10^{-15}$, parameter difference $< 3.0 \times 10^{-17}$, peak violation $2.00\%$).

In this convex setting, direct warm-starting accelerates convergence (reaching tolerance in $59$ steps vs. $101$ for cold start), but incurs a higher peak violation ($2.00\%$ vs. $1.22\%$). Staged warm-starting (Path A) is both slower ($123$ total steps) and accumulates higher deployed violation ($5.90\%$), directly inheriting the intermediate scaffold's boundary excursion. Un-staged cold start is therefore optimal for realizability preservation, while direct warm-starting minimizes step count. We formalize this configuration space as a **Stratified Design Atlas** and delineate the necessary conditions—non-convex losses, coupled Navier–Stokes evaluation, and non-nested basis edits—under which geometric transport and path-dependence become non-trivial.

---

## 1. Introduction

Turbulence closure modeling for Reynolds-Averaged Navier–Stokes (RANS) equations remains one of the foundational open challenges in computational fluid dynamics. The Reynolds stress tensor $\tau_{ij} = \overline{u_i' u_j'}$ represents the statistical effect of turbulent fluctuations on the mean momentum field. For decades, practical engineering CFD has relied on linear eddy-viscosity models governed by the Boussinesq hypothesis:
$$\tau_{ij} - \frac{2}{3}k \delta_{ij} = -2 \nu_t S_{ij}$$
where $k$ is the turbulent kinetic energy, $\nu_t$ is the scalar eddy viscosity, and $S_{ij} = \frac{1}{2}(\partial_j U_i + \partial_i U_j)$ is the mean strain rate tensor.

While computationally robust, the Boussinesq hypothesis fails qualitatively in complex strain fields—most notoriously in non-circular duct flows, where turbulent secondary motion of the second kind (Prandtl, 1926) is driven exclusively by Reynolds normal stress anisotropy $(\tau_{yy} - \tau_{zz})$ and cross-stream shear stress $\tau_{yz}$ (Speziale, 1987). Linear models predict identical normal stresses in the cross-plane ($\tau_{yy} = \tau_{zz}$), completely suppressing corner circulation.

To overcome these deficiencies, higher-order non-linear eddy viscosity models (Pope, 1975, 2000; Craft et al., 1996; Wallin & Johansson, 2000) and data-driven symbolic formulations have been developed (Duraisamy et al., 2019; Ling et al., 2016; Weatheritt & Sandberg, 2016; Schmelzer et al., 2020; Parish & Duraisamy, 2016; Wu et al., 2018; Kochkov et al., 2021; List et al., 2022). However, automated model discovery frameworks encounter fundamental questions when mutating model structure:
* **Functional Preservation:** Does zero-initializing newly introduced terms preserve physical predictions? In linear parameterizations, zero-padding provides exact functional preservation ($\delta_R \equiv 0$), paralleling network morphism and Net2Net architectures (Chen et al., 2016; Wei et al., 2016).
* **Warm-Starting vs. Cold-Starting:** In machine learning, warm-starting neural networks often underperforms cold-starting due to momentum transients, basin traps, and catastrophic forgetting (Ash & Adams, 2020; Hinton et al., 2015; Frankle & Carbin, 2019; Draxler et al., 2018; Garipov et al., 2018; Ainsworth et al., 2023; Bengio et al., 2009).
* **Realizability Invariants:** Parameter trajectories must preserve physical realizability (non-negative normal stresses and Cauchy–Schwarz conditions; Schumann, 1977; Lumley, 1978).

### 1.1 The Stratified Design Atlas

We formalize the configuration space of candidate closures as a **Stratified Design Atlas**:
$$\mathcal{A} = \bigl(\{\Theta_s\},\, L,\, \{T_e\}\bigr)$$
where $\Theta_s$ are continuous parameter manifolds (strata) corresponding to specific tensor bases, $L$ is the calibration loss functional penalizing realizability violations, and $T_e: \Theta_s \to \Theta_{s'}$ are transport mappings carrying calibrated states across structural mutations.

---

## 2. Mathematical Formulation

### 2.1 Governing Equations & Secondary Flow Mechanics

Consider steady, incompressible turbulent flow through a straight square duct of width $h$ aligned with the $x$-axis. The mean velocity field is $\mathbf{U} = (U(y,z), V(y,z), W(y,z))$. Streamwise mean vorticity is defined as:
$$\omega_x = \frac{\partial W}{\partial y} - \frac{\partial V}{\partial z}$$
In straight square duct flow ($\partial/\partial x \equiv 0$), the transport equation for $\omega_x$ simplifies to:
$$V \frac{\partial \omega_x}{\partial y} + W \frac{\partial \omega_x}{\partial z} = \nu \nabla^2 \omega_x + f_{\text{sec}}(\mathbf{x})$$
where the secondary vorticity driving force $f_{\text{sec}}$ is defined entirely by spatial derivatives of the Reynolds stresses:
$$f_{\text{sec}}(\mathbf{x}) = \underbrace{\frac{\partial^2}{\partial y \partial z}(\tau_{yy} - \tau_{zz})}_{\text{Normal stress anisotropy}} + \underbrace{\left(\frac{\partial^2}{\partial z^2} - \frac{\partial^2}{\partial y^2}\right)\tau_{yz}}_{\text{Cross-plane shear stress}}$$

### 2.2 Tensor Integrity Basis & Nested Subspaces

Following Pope (1975, 2000), the deviatoric Reynolds stress anisotropy is expanded as $a_{ij} = \sum_{n=1}^{N} c_n T_{ij}^{(n)}$ with $c_n \equiv \theta_{n-1}$. In the duct cross-section, three nested strata are defined:
* **Stratum 0 ($\mathcal{M}_0$, Linear Boussinesq, $d_0 = 1$):**
  $$\tau^{(0)}(\mathbf{x}; \theta) = \frac{2}{3}k \mathbb{I} + \theta_0 T^{(1)}, \quad T^{(1)} = -2 \frac{k}{\omega} S$$
  where $\theta_0 = C_\mu \approx 0.09$.
* **Stratum 1 ($\mathcal{M}_1$, Quadratic Pope, $d_1 = 3$):**
  $$\tau^{(1)}(\mathbf{x}; \theta) = \frac{2}{3}k \mathbb{I} + \theta_0 T^{(1)} + \theta_1 T^{(2)} + \theta_2 T^{(3)}$$
  with $T^{(2)} = \frac{k}{(C_\mu \omega)^2} (S^2 - \frac{1}{3}\operatorname{Tr}(S^2)\mathbb{I})$ and $T^{(3)} = \frac{k}{(C_\mu \omega)^2} (S\Omega - \Omega S)$.
* **Stratum 2 ($\mathcal{M}_2$, Cubic Pope, $d_2 = 4$):**
  $$\tau^{(2)}(\mathbf{x}; \theta) = \frac{2}{3}k \mathbb{I} + \sum_{i=0}^3 \theta_i T^{(i+1)}$$
  incorporating $T^{(4)} = \frac{k}{(C_\mu \omega)^3} (S^2\Omega - \Omega S^2)$.

### 2.3 Stress-Representability & Canonical Transports

The stress-projection alignment $\cos\phi_{\text{sec}}$ measures whether secondary stress lies in the basis span:
$$\cos\phi_{\text{sec}}(\mathcal{M}_s) = \frac{\langle \tau_{\text{sec}}, \operatorname{proj}_{\mathcal{M}_s} \tau_{\text{sec}} \rangle_{L^2}}{\|\tau_{\text{sec}}\|_{L^2} \|\operatorname{proj}_{\mathcal{M}_s} \tau_{\text{sec}}\|_{L^2}}$$
The forcing correlation $\rho(\theta)$ evaluates field derivatives of the instantiated closure:
$$\rho(\theta) = \frac{\langle f_{\text{sec}}(\tau(\theta)), f_{\text{sec}}(\tau_{\text{ref}}) \rangle_{L^2}}{\|f_{\text{sec}}(\tau(\theta))\|_{L^2} \|f_{\text{sec}}(\tau_{\text{ref}})\|_{L^2}}$$

Zero-padding transport $T_e(\theta) = [\theta_0, \dots, \theta_{d-1}, 0]$ yields **exact zero realization drift**:
$$\delta_R = \frac{\|\tau^{(s')}(T_e(\theta)) - \tau^{(s)}(\theta)\|_{L^2}}{\|\tau_{\text{ref}}\|_{L^2}} \equiv 0$$

The task loss is $L_{\text{task}}(\theta) = \frac{1}{2} \int_\Omega \|\tau(\theta) - \tau_{\text{ref}}\|_F^2 \, dA$, with constant Hessian equal to the spatial Gram matrix $G_{ij} = \int_\Omega \operatorname{Tr}(T^{(i)} T^{(j)}) \, dA$. The regularized objective adds a realizability penalty:
$$L(\theta) = L_{\text{task}}(\theta) + \lambda_{\text{realiz}} \int_\Omega \max\bigl(0, -\lambda_{\min}(\tau(\mathbf{x}; \theta))\bigr)^2 \, dA$$

---

## 3. Stress-Representability & Gram Matrix Decomposition

```
STRESS-REPRESENTABILITY METRICS ACROSS STRATA
-----------------------------------------------------------------------------------------
Metric                                Stratum 0 (Linear)  Stratum 1 (Quad.)  Stratum 2 (Cubic)
-----------------------------------------------------------------------------------------
cos φ_sec (stress projection)         1.15e-4 ≈ 0         1.000              1.000
ρ(θ*) (forcing correlation)           -0.157              +0.294*            +0.999
Reversed forcing fraction             29.38%              20.36%             1.65%
-----------------------------------------------------------------------------------------
*Note: Stratum-1 ρ is evaluated at the λ=150 scaffold iterate after 100 steps.
```

### Gram Matrix Conditioning Decomposition

The Gram matrix decouples block-diagonally ($G_{13} = G_{14} = G_{23} = G_{24} = 0$) due to symmetry:
* Diagonal entries: $G_{11} = 1.365 \times 10^{-4}$, $G_{22} = 9.589 \times 10^{-6}$, $G_{33} = 1.151 \times 10^{-4}$, $G_{44} = 4.431 \times 10^{-6}$.
* Scale disparity: $G_{11} / G_{44} = (\|T^{(1)}\| / \|T^{(4)}\|)^2 = 30.82$.
* Cross-coupling: $R_{34} = -0.714$, yielding normalized 2×2 block condition ratio $(1 + |R_{34}|)/(1 - |R_{34}|) = 5.99$.
* Global condition number: $\kappa(G_2) = 64.10 = 30.82 \times 2.08$. Specifically, $5.99$ is the unit-normalized $(\theta_2, \theta_3)$-block $\kappa$, while $2.08$ is the factor by which $R_{34} = -0.714$ depresses $\lambda_{\min}$ below $G_{44}$.

---

## 4. Traversal Dynamics & Audited Telemetry

Three calibration paths evaluated under budget-matched controls ($N = 220$ gradient steps, $\lambda = 150$):
* **Path A (Staged):** 100 steps in Stratum 1 $\to$ 120 steps in Stratum 2.
* **Path B (Direct):** 220 steps in Stratum 2 initialized from $\theta = [\theta_0^*, 0, 0, 0]$.
* **Path C (Cold Start):** 220 steps in Stratum 2 initialized from $\theta = [0, 0, 0, 0]$.

```
===================================================================================================================
AUDITED TRAVERSAL TELEMETRY (data/results_curvature_experiment.json, results_audit_experiments.json)
===================================================================================================================
Protocol    Stage                               Step    Loss L(θ)      Violation    Forcing ρ    Reversed Area (%)
-------------------------------------------------------------------------------------------------------------------
Shared      Warm-start origin                      0    1.2264e-07       0.00%       -0.157           29.38%
-------------------------------------------------------------------------------------------------------------------
Path A      Scaffold                              50    5.0321e-09       5.12%       +0.293           20.57%
            Scaffold end (100th S1 step)          99    4.7298e-09       5.64%       +0.294           20.36%
            Embedding (δ_R ≡ 0, pre-grad)        100    4.7300e-09       5.64%       +0.294           20.44%
            Deployed 1st step (peak)             101    4.5361e-09       5.90%       +0.368           12.41%
            Deployed repair                      105    3.3522e-09       3.12%       +0.654            8.12%
            Deployed repair                      120    1.4889e-09       1.22%       +0.988            2.30%
            Tol. reached (L < 1.4e-9)            123    1.3851e-09       1.30%       +0.994            1.78%
            1% above final (L ≤ 1.355e-9)        126    1.3472e-09       1.13%       +0.998            1.65%
            Final                                219    1.3412e-09       1.04%       +0.999            1.65%
-------------------------------------------------------------------------------------------------------------------
Path B      Rapid descent                          1    1.1058e-07       0.00%       +0.941           10.11%
            Excursion onset                       20    2.7765e-09       1.56%       +0.955            2.91%
            Excursion trough                      21    2.6159e-09       1.65%       +0.957            2.82%
            Peak violation                        25    3.8512e-09       2.00%       +0.965            2.73%
            Rebound peak (1.68×)                  28    4.4029e-09       2.00%       +0.974            2.65%
            Tol. reached (L < 1.4e-9)             59    1.3990e-09       1.22%       +0.998            1.78%
            1% above final (L ≤ 1.355e-9)         89    1.3522e-09       1.13%       +0.998            1.74%
            Final                                219    1.3412e-09       1.04%       +0.999            1.69%
-------------------------------------------------------------------------------------------------------------------
Path C      Origin (θ = 0)                         0    6.7566e-07       0.00%        ---              ---
            Peak violation                        30    8.5443e-08       1.22%        ---              ---
            Tol. reached (L < 1.4e-9)            101    1.3963e-09       1.13%        ---              ---
            1% above final (L ≤ 1.355e-9)        111    1.3528e-09       1.04%        ---              ---
            Final                                219    1.3412e-09       1.04%        ---              ---
===================================================================================================================
```

### Dynamical Observations:
1. **Identical Minimizer:** All paths converge to $\|\theta_A^* - \theta_B^*\| = 3.16 \times 10^{-5}$ and $C_{AB} = 6.93 \times 10^{-6}$, guaranteed by strict convexity.
2. **Scaffold Inheritance:** Path A inherits $5.64\%$ violation from the scaffold at step 100, rising to $5.90\%$ at step 101 due to Adam's cold-update sign step, followed by two non-monotonicities ($5.64\% \to 5.90\%$ early rise, $0.95\% \to 1.04\%$ late rise). Staging is the slowest protocol overall ($123$ steps to tolerance).
3. **Protocol Performance & Cold Start Optimality:** Path C reaches the identical minimum with a peak violation of only **$1.22\%$**, lower than both Path B ($2.00\%$) and Path A deployed ($5.90\%$), converging to tolerance in $101$ steps without loss rebounds. Direct warm-starting (Path B) is the fastest protocol ($59$ steps, saving $42$ steps over cold start), while staged warm-starting (Path A) costs $22$ steps ($123$ steps).

---

## 5. Mechanism Identification & Pre-Registered Controls

### 5.1 Optimizer & Basis-Scaling Ablation

```
2×2 DIAGNOSTIC MATRIX + PER-COORDINATE ADAM
---------------------------------------------------------------------------------------------------
Condition                                      B Peak (%)    B Rebound    C Peak (%)    Final Loss
---------------------------------------------------------------------------------------------------
Raw Adam (baseline)                            2.00%         1.68×        1.22%         1.3412e-09
Raw Adam (β₁ = 0, no momentum)                 1.30%         1.05×        —             1.4360e-09*
Raw Plain GD (convergent α ≈ 7324)             6.25%         1.00×        —             1.3412e-09
Raw Plain GD (inert α = 2e-3)                  0.00%         1.00×        —             1.2263e-07
Norm. Adam (per-coord α_n, ε_n)                2.00%         1.68×        —             1.3412e-09
Norm. Adam (scalar α resc., insufficient)     12.50%         2.20×        —             1.3412e-09
Norm. Adam (fixed α = 2e-3)                   21.79%       834.0×         —             1.3412e-09
---------------------------------------------------------------------------------------------------
* Evaluated at N = 220. Extended optimization up to 450 steps reaches 1.4324e-09 without reaching the baseline target.
```

* **Proof of Scale Invariance:** Per-coordinate Adam ($\alpha_n = \alpha \|T^{(n)}\|$, $\epsilon_n = \epsilon / \|T^{(n)}\|$) on the normalized basis reproduces raw Adam to machine precision (relative loss difference $< 2.1 \times 10^{-15}$, parameter norm difference $< 3.0 \times 10^{-17}$, peak violation $2.00\%$).
* **Matched-Rate Momentum Ablation:** Comparing full Adam and $\beta_1 = 0$ at matched nominal $\alpha$ partially conflates momentum with effective step size ($(1-\beta_1)^{-1} \approx 10$). At a matched convergence rate of **76 steps** to tolerance ($L < 1.4 \times 10^{-9}$), full Adam ($\alpha = 1.37 \times 10^{-3}$) incurs a **$1.65\%$** peak violation, compared to **$1.30\%$** for $\beta_1 = 0$ ($\alpha = 2 \times 10^{-3}$), isolating a **$1.27\times$** momentum amplification factor.
* **Convergent Plain GD:** Demonstrates that excursions occur even in plain gradient descent ($6.25\%$ for Path B, $6.16\%$ for Path A) when operating at a step size matched to the Lipschitz constant ($\alpha = 1/\lambda_{\max} \approx 7324$).

### 5.2 Decisive Restart Controls

* **Matched-Loss B-Restart (Step 18):** Restarting Adam on Path B at step 18 (where loss drops to $4.34 \times 10^{-9} < 4.73 \times 10^{-9}$, preceded by $5.84 \times 10^{-9}$ at step 17) yields a post-restart peak violation of **$1.74\%$** at step 21 (climbing from $0.95\%$ at step 17, **Branch ii occurred**), while the loss continues its monotonic descent toward tolerance without an additional rebound. Parameter travel distance dominates the excursion, while momentum adds $+0.26\%$ ($1.74\% \to 2.00\%$), and the restart merely interrupted an in-progress boundary traversal rather than preventing one.
* **A-Carry ($5.64\%$) vs. A-Fresh ($5.90\%$):** Momentum carry-over deploys at $5.64\%$ without rising. Fresh restart triggers an initial sign step update pushing violation to $5.90\%$ at step 101.

---

## 6. Realizability Theoretical Basis & Discussion

Realizability excursions do not reflect non-convex boundary geometry:
* **Convexity of the Admissible Set:** Lumley's realizability triangle is convex in Reynolds stress space. For linear tensor parameterizations $\tau = \sum_k \theta_k T^{(k)}$, the set of parameters producing non-negative stress at every cell is an intersection of convex sets, hence strictly **convex** in $\Theta$.
* **Two Distinct Claims:**
  1. *Inherited Floor:* The reference field $\tau_{\text{ref}}$ carries a $1.56\%$ violation ($\min\lambda = -5.10 \times 10^{-5}$); any least-squares model fitting it inherits this floor ($1.48\%$ at $\lambda = 0$, $1.04\%$ at $\lambda = 150$, $0.52\%$ at $\lambda = 1500$).
  2. *Transient Excess:* Excursions above this floor ($2.00\%$ for Path B, $5.90\%$ for Path A deployed) are governed by per-step travel distance and coordinate scaling during descent across the ill-conditioned landscape.

### Summary of Contributions

1. Formalized candidate turbulence closures within a **Stratified Design Atlas** and established that in nested, convex calibrations, zero-padding provides exact functional preservation ($\delta_R \equiv 0$).
2. Demonstrated that staged calibration inherits scaffold realizability violations upon deployment ($5.64\% \to 5.90\%$) and is the slowest protocol ($123$ steps), direct warm-starting is the fastest protocol ($59$ steps, $2.00\%$ peak), and un-staged cold start ($\theta = \mathbf{0}$, Path C) achieves the lowest peak violation ($1.22\%$, $101$ steps).
3. Disproved the Gram-conditioning hypothesis, proving that excursions are driven by parameter travel distance across coordinate scales and isolated a $1.27\times$ momentum amplification at matched convergence speed.
4. Demonstrated exact machine-precision scale invariance for per-coordinate Adam on normalized bases.
5. Isolated the necessary conditions (non-convex losses, coupled Navier–Stokes solvers, non-nested edits) for geometric curvature and path-dependence to emerge.

---

## References

1. Ainsworth, S., Hayase, J., & Srinivasa, S. (2023). Git Re-Basin: Merging Models modulo Permutation Symmetries. *ICLR*.
2. Amari, S. (1998). Natural Gradient Works Efficiently in Learning. *Neural Computation*, 10(2), 251–276.
3. Ash, J., & Adams, R. P. (2020). On Warm-Starting Neural Network Training. *NeurIPS*, 33, 3884–3894.
4. Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum Learning. *ICML*, 41–48.
5. Boussinesq, J. (1877). Essai sur la théorie des eaux courantes. *Mémoires présentés par divers savants à l'Académie des Sciences de l'Institut National de France*, 23(1), 1–680.
6. Chen, T., Goodfellow, I., & Shlens, J. (2016). Net2Net: Accelerating Learning via Knowledge Transfer. *ICLR*.
7. Craft, T. J., Launder, B. E., & Suga, K. (1996). Development and application of a cubic eddy-viscosity model of turbulence. *Int. J. Heat Fluid Flow*, 17(2), 108–115.
8. Draxler, F., Veschgini, K., Salmhofer, M., & Hamprecht, F. A. (2018). Essentially No Barriers in Neural Network Energy Landscapes. *ICML*, 1309–1318.
9. Duraisamy, K., Iaccarino, G., & Xiao, H. (2019). Turbulence Modeling in the Age of Data. *Annu. Rev. Fluid Mech.*, 51, 357–377.
10. Frankle, J., & Carbin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. *ICLR*.
11. Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D. P., & Wilson, A. G. (2018). Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs. *NeurIPS*, 31, 8789–8798.
12. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *arXiv:1503.02531*.
13. Kochkov, D., Smith, J. A., Alieva, A., Wang, Q., Brenner, M. P., & Hoyer, S. (2021). Machine learning-accelerated computational fluid dynamics. *PNAS*, 118(21), e2101784118.
14. Ling, J., Jones, R., & Templeton, J. (2016). Machine learning strategies for systems with invariance; application to Reynolds stress closure modelling. *J. Fluid Mech.*, 807, 155–166.
15. List, F., Chen, L.-W., & Thuerey, N. (2022). Learned turbulent field closures with direct numerical simulation data. *Phys. Fluids*, 34(1), 015118.
16. Lumley, J. L. (1978). Computational modeling of turbulent flows. *Adv. Appl. Mech.*, 18, 123–176.
17. Parish, E. J., & Duraisamy, K. (2016). A paradigm for data-driven predictive modeling using field inversion and machine learning. *J. Comput. Phys.*, 305, 758–774.
18. Pinelli, A., Uhlmann, M., Sekimoto, A., & Kawahara, G. (2010). Reynolds number dependence of mean flow and turbulence statistics in a square duct. *J. Fluid Mech.*, 644, 107–122.
19. Pope, S. B. (1975). A more general effective-viscosity hypothesis. *J. Fluid Mech.*, 72(2), 331–340.
20. Pope, S. B. (2000). *Turbulent Flows*. Cambridge University Press.
21. Prandtl, L. (1926). Über die ausgebildete Turbulenz. *Verh. 2. Int. Kongr. Tech. Mech., Zürich*, 62–74.
22. Schmelzer, M., Dwight, R. P., & Cinnella, P. (2020). Discovery of Algebraic Reynolds-Stress Models Using Sparse Symbolic Regression. *Flow Turbul. Combust.*, 104, 579–603.
23. Schumann, U. (1977). Realizability of Reynolds stress turbulence models. *Phys. Fluids*, 20(5), 721–725.
24. Speziale, C. G. (1987). On nonlinear K-l and K-ε models of turbulence. *J. Fluid Mech.*, 178, 459–475.
25. Wallin, S., & Johansson, A. V. (2000). An explicit algebraic Reynolds stress model for incompressible and compressible turbulent flows. *J. Fluid Mech.*, 403, 89–132.
26. Weatheritt, J., & Sandberg, R. D. (2016). A novel evolutionary algorithm applied to algebraic Reynolds stress model development. *J. Fluid Mech.*, 806, 248–277.
27. Wei, T., Wang, C., Rui, Y., & Chen, C. W. (2016). Network Morphism. *ICML*, 564–572.
28. Wu, J.-L., Xiao, H., & Paterson, E. (2018). Physics-informed machine learning approach for augmenting turbulence models: A comprehensive framework. *Phys. Rev. Fluids*, 3(7), 074602.
