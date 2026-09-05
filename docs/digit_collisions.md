# Audit of Numerical Values & Digit Collisions

This document records the results of a mechanical scan across all numerical values reported in the manuscript (*Edition 2*), the codebase, and the serialized JSON data. It explicitly adjudicates apparent "digit collisions"—instances where unrelated quantities round to identical or near-identical values.

---

## 1. Inventory of Identified Digit Collisions

### Collision 1: Controllability Alignment vs. Gram Matrix Element ($1.15 \times 10^{-4}$)

- **Occurrence A**: $\cos \phi_{\mathrm{sec}}(M_0) = 1.15 \times 10^{-4}$ (Table I, Stratum 0 alignment).
  - *Full precision*: $1.147728 \times 10^{-4}$.
  - *Origin*: Computed in `src/prototype_square_duct.py` by projecting the secondary-flow driving stress $\tau_{\mathrm{sec}}$ onto the single linear Boussinesq basis tensor $T^{(1)}$. Because $T^{(1)}$ has zero normal-stress anisotropy on a square duct with parallel streamlines, the alignment is zero up to secondary-flow velocity leakage ($V, W \sim 0.02$).
- **Occurrence B**: $G_{33} = \|T^{(3)}\|^2 = 1.15 \times 10^{-4}$ (Table II, Gram matrix diagonal element).
  - *Full precision*: $1.150729 \times 10^{-4}$.
  - *Origin*: The $L^2$ norm squared of the third integrity basis tensor $T^{(3)} = \frac{k}{\omega^2}(S\Omega - \Omega S)$ over the $48 \times 48$ quarter-duct cross-section.
- **Verdict**: **Spurious coincidence**. These quantities derive from fundamentally different equations (tensor projection vs. tensor Frobenius norm integral) that happen to round to $1.15 \times 10^{-4}$ to three significant figures.

---

### Collision 2: Path B Step-20 Rebound Loss vs. Path D Frozen-Basis Loss ($2.78 \times 10^{-9}$)

- **Occurrence A**: Path B loss at step 20 = $2.776474 \times 10^{-9}$ (prior to the $1.68\times$ rebound to $4.40 \times 10^{-9}$ at step 28).
  - *Origin*: Trajectory of joint 4-tensor Adam descent traversing the ill-conditioned valley.
- **Occurrence B**: Path D final calibrated loss = $2.784131 \times 10^{-9}$ (Audit experiment where Stratum-1 weights are frozen at the hop).
  - *Origin*: Optimal loss achievable when restricted to the 1D subspace of $T^{(4)}$ with frozen $\theta_0^*, \theta_1^*, \theta_2^*$.
- **Verdict**: **Physical proxy, not numerical identity**. Both values reflect the residual loss floor of the quadratic surface when the primary and secondary normal stress differences are partially resolved, but before the cross-coupled $T^{(3)} \text{--} T^{(4)}$ subspace is fully coordinated.

---

### Collision 3: Scaffold Peak ($6.08\%$) vs. Convergent Plain GD Peaks ($6.16\% \text{--} 6.25\%$)

- **Occurrence A**: Path A Stratum-1 scaffold peak violation = $6.08\%$ (step 38 of 100).
- **Occurrence B**: Path A with convergent Plain GD ($\alpha = 7323.77$) peak violation = $6.16\%$.
- **Occurrence C**: Path B with convergent Plain GD ($\alpha = 7323.77$) peak violation = $6.25\%$.
- **Verdict**: **Geometric boundary artifact**. The $6.08\% \text{--} 6.25\%$ cluster corresponds to the geometric fraction of the duct grid ($48 \times 48 = 2304$ cells) where the unconstrained uncalibrated model produces negative eigenvalues before the penalty $\lambda_{\mathrm{realiz}} = 150$ enforces realizability. Specifically, $6.08\% = 140/2304$ cells and $6.25\% = 144/2304$ cells (a difference of just 4 cells near the duct corner).

---

## 2. Cross-Table Reconciliations

### Cross-Coupling Condition Number vs. Joint Condition Number

- Condition number of normalized $2 \times 2$ block:
  $$\frac{1 + |R_{34}|}{1 - |R_{34}|} = \frac{1 + 0.7139}{1 - 0.7139} \approx 5.99$$
- Condition number of unit-normalized $4 \times 4$ Gram matrix:
  $$\kappa(\tilde{G}) = \frac{\lambda_{\max}(\tilde{G})}{\lambda_{\min}(\tilde{G})} \approx 5.99$$
- Scale disparity ratio:
  $$\left(\frac{\|T^{(1)}\|}{\|T^{(4)}\|}\right)^2 = \left(\frac{0.011685}{0.002105}\right)^2 \approx 30.82$$
- Cross-coupling inflation factor:
  $$\frac{\kappa(G)}{\text{scale disparity}} = \frac{64.10}{30.82} \approx 2.08$$
- Product decomposition:
  $$30.82 \times 2.08 \approx 64.10 \approx \kappa(G)$$
Specifically, $5.99$ is the unit-normalized $(\theta_2,\theta_3)$-block $\kappa$, while $2.08$ is the factor by which cross-coupling $R_{34} = -0.714$ depresses $\lambda_{\min}$ below $G_{44}$. Both quantities are explicitly reconciled across the manuscript and verification scripts.
