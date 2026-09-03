"""
curvature_experiment.py

The 3-Stratum Curvature / Holonomy Experiment:
Empirically measures the geometric curvature of the design space by testing path dependence.
Demonstrates that the structural edit sequence matters for optimization stability,
realizability preservation, and parameter basin convergence.

Design Atlas Strata:
- Stratum 0: {T1}               (Linear Boussinesq, dim = 1)
- Stratum 1: {T1, T2, T3}       (Quadratic Pope, dim = 3)
- Stratum 2: {T1, T2, T3, T4}   (Cubic Pope, dim = 4)

Pathways:
- Path A (Sequential / Step-wise): Stratum 0 -> Stratum 1 -> Stratum 2
- Path B (Direct Jump):            Stratum 0 -> Stratum 2
"""

import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('.cache/matplotlib')

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prototype_square_duct import SquareDuctQuadrant, AdamOptimizer, check_realizability

np.random.seed(42)

# ==============================================================================
# 1. Domain Discretization & Extended Tensor Basis
# ==============================================================================

class ExtendedDuctMesh(SquareDuctQuadrant):
    """
    Square duct quadrant domain extending SquareDuctQuadrant with Pope's cubic tensor:
    T4 = k / (C_mu * omega)^3 * (S^2 * Omega - Omega * S^2)
    Properly non-dimensionalized with turbulent timescale tau_t = 1 / (C_mu * omega).
    """
    def __init__(self, h=1.0, Ny=48, Nz=48):
        super().__init__(h=h, Ny=Ny, Nz=Nz)
        self._construct_cubic_basis()
        self._generate_target_dns()

    def _construct_cubic_basis(self):
        # S^2
        S_sq = np.einsum('abij,abjk->abik', self.S, self.S)
        # S^2 * Omega - Omega * S^2
        S2_Om = np.einsum('abij,abjk->abik', S_sq, self.Omega)
        Om_S2 = np.einsum('abij,abjk->abik', self.Omega, S_sq)
        M_cubic = S2_Om - Om_S2  # symmetric and trace-free
        
        # Consistent Pope turbulent timescale scaling
        # tau_t = 1 / (0.09 * omega)
        c_mu_scale = 0.09
        scale_factor = (1.0 / c_mu_scale)**3
        self.T4 = scale_factor * (self.k / (self.omega**3))[:, :, None, None] * M_cubic
        
        # Verify symmetry and tracelessness
        sym_error = np.max(np.abs(self.T4 - np.swapaxes(self.T4, 2, 3)))
        tr_error = np.max(np.abs(np.trace(self.T4, axis1=2, axis2=3)))
        assert sym_error < 1e-12, f"T4 symmetry error: {sym_error}"
        assert tr_error < 1e-12, f"T4 trace error: {tr_error}"

    def _generate_target_dns(self):
        # Ground truth with linear, quadratic, and cubic components
        self.c_mu_true = 0.090
        self.c1_true = 0.045
        self.c2_true = -0.035
        self.c3_true = 0.020
        
        I3 = np.eye(3)[None, None, :, :]
        tau_iso = (2.0/3.0) * self.k[:, :, None, None] * I3
        
        # Realistic turbulence stress with unmodeled wall damping
        Y, Z = self.Y[:, :, None, None], self.Z[:, :, None, None]
        S_sq = np.einsum('abij,abjk->abik', self.S, self.S)
        unmodeled_wall_stress = 0.015 * (self.k / (self.omega**2))[:, :, None, None] * np.sin(np.pi * Y) * np.sin(np.pi * Z) * S_sq
        
        tau_dev_true = (self.c_mu_true * self.T1 + 
                        self.c1_true * self.T2 + 
                        self.c2_true * self.T3 + 
                        self.c3_true * self.T4 + 
                        unmodeled_wall_stress)
        
        self.tau_DNS = tau_iso + tau_dev_true

# ==============================================================================
# 2. Stratum Models with Realizability Penalty
# ==============================================================================

class Stratum0:
    """Stratum 0: Linear Boussinesq, theta = [C_mu]"""
    def __init__(self, mesh):
        self.mesh = mesh
        self.theta = np.array([0.01])
        self.I3 = np.eye(3)[None, None, :, :]
        self.W = mesh.W_grid[:, :, None, None]

    def predict(self, theta=None):
        if theta is None: theta = self.theta
        return (2.0/3.0) * self.mesh.k[:, :, None, None] * self.I3 + theta[0] * self.mesh.T1

    def loss_and_grad(self, theta=None):
        if theta is None: theta = self.theta
        tau = self.predict(theta)
        diff = tau - self.mesh.tau_DNS
        loss = 0.5 * np.sum(self.W * (diff**2))
        grad = np.array([np.sum(self.W * diff * self.mesh.T1)])
        return loss, grad

class Stratum1:
    """Stratum 1: Quadratic Pope, theta = [C_mu, c1, c2]"""
    def __init__(self, mesh, lambda_realiz=150.0):
        self.mesh = mesh
        self.theta = np.zeros(3)
        self.I3 = np.eye(3)[None, None, :, :]
        self.W = mesh.W_grid[:, :, None, None]
        self.lambda_realiz = lambda_realiz
        self.basis = [mesh.T1, mesh.T2, mesh.T3]

    def predict(self, theta=None):
        if theta is None: theta = self.theta
        tau_dev = sum(theta[i] * self.basis[i] for i in range(3))
        return (2.0/3.0) * self.mesh.k[:, :, None, None] * self.I3 + tau_dev

    def loss_and_grad(self, theta=None):
        if theta is None: theta = self.theta
        tau = self.predict(theta)
        diff = tau - self.mesh.tau_DNS
        
        loss_task = 0.5 * np.sum(self.W * (diff**2))
        grad_task = np.array([np.sum(self.W * diff * self.basis[i]) for i in range(3)])
        
        loss_realiz = 0.0
        grad_realiz = np.zeros(3)
        
        if self.lambda_realiz > 0:
            tau_sym = 0.5 * (tau + np.swapaxes(tau, 2, 3))
            eigvals, eigvecs = np.linalg.eigh(tau_sym)
            min_eig = eigvals[:, :, 0]
            min_vec = eigvecs[:, :, :, 0]
            
            violation_mask = (min_eig < 0)
            if np.any(violation_mask):
                viol = np.where(violation_mask, min_eig, 0.0)
                loss_realiz = self.lambda_realiz * np.sum(self.mesh.W_grid * (viol**2))
                
                for i in range(3):
                    v_T_v = np.einsum('abi,abij,abj->ab', min_vec, self.basis[i], min_vec)
                    grad_realiz[i] = 2.0 * self.lambda_realiz * np.sum(self.mesh.W_grid * viol * v_T_v)
                    
        return loss_task + loss_realiz, grad_task + grad_realiz

class Stratum2:
    """Stratum 2: Cubic Pope, theta = [C_mu, c1, c2, c3]"""
    def __init__(self, mesh, lambda_realiz=150.0):
        self.mesh = mesh
        self.theta = np.zeros(4)
        self.I3 = np.eye(3)[None, None, :, :]
        self.W = mesh.W_grid[:, :, None, None]
        self.lambda_realiz = lambda_realiz
        self.basis = [mesh.T1, mesh.T2, mesh.T3, mesh.T4]

    def predict(self, theta=None):
        if theta is None: theta = self.theta
        tau_dev = sum(theta[i] * self.basis[i] for i in range(4))
        return (2.0/3.0) * self.mesh.k[:, :, None, None] * self.I3 + tau_dev

    def loss_and_grad(self, theta=None):
        if theta is None: theta = self.theta
        tau = self.predict(theta)
        diff = tau - self.mesh.tau_DNS
        
        loss_task = 0.5 * np.sum(self.W * (diff**2))
        grad_task = np.array([np.sum(self.W * diff * self.basis[i]) for i in range(4)])
        
        loss_realiz = 0.0
        grad_realiz = np.zeros(4)
        
        if self.lambda_realiz > 0:
            tau_sym = 0.5 * (tau + np.swapaxes(tau, 2, 3))
            eigvals, eigvecs = np.linalg.eigh(tau_sym)
            min_eig = eigvals[:, :, 0]
            min_vec = eigvecs[:, :, :, 0]
            
            violation_mask = (min_eig < 0)
            if np.any(violation_mask):
                viol = np.where(violation_mask, min_eig, 0.0)
                loss_realiz = self.lambda_realiz * np.sum(self.mesh.W_grid * (viol**2))
                
                for i in range(4):
                    v_T_v = np.einsum('abi,abij,abj->ab', min_vec, self.basis[i], min_vec)
                    grad_realiz[i] = 2.0 * self.lambda_realiz * np.sum(self.mesh.W_grid * viol * v_T_v)
                    
        return loss_task + loss_realiz, grad_task + grad_realiz

# ==============================================================================
# 3. Path Traversal & Holonomy Experiment
# ==============================================================================

def run_curvature_experiment():
    print("=" * 75)
    print("3-STRATUM CURVATURE & HOLONOMY EXPERIMENT")
    print("Testing Path Dependence across the Stratified Design Atlas")
    print("=" * 75)

    mesh = ExtendedDuctMesh(h=1.0, Ny=48, Nz=48)
    W = mesh.W_grid[:, :, None, None]
    norm_dns = np.sqrt(np.sum(W * (mesh.tau_DNS**2)))

    # --------------------------------------------------------------------------
    # Step 0: Optimize Base Stratum 0 (Shared origin for both paths)
    # --------------------------------------------------------------------------
    print("\n[Step 0] Calibrating Base Stratum 0 (Linear Boussinesq)...")
    s0 = Stratum0(mesh)
    opt0 = AdamOptimizer(lr=2e-3)
    for _ in range(120):
        loss0, grad0 = s0.loss_and_grad()
        s0.theta = opt0.step(s0.theta, grad0)
        
    theta_0_star = s0.theta.copy()
    loss_0_star, _ = s0.loss_and_grad()
    print(f"  Optimized theta_0*: [{theta_0_star[0]:.6f}], Loss: {loss_0_star:.6e}")

    # ==========================================================================
    # PATH A: Sequential (Stratum 0 -> Stratum 1 -> Stratum 2)
    # ==========================================================================
    print("\n" + "-" * 60)
    print("PATH A: Sequential Traversal (0 -> 1 -> 2)")
    print("-" * 60)

    # 1. Transport e1: Stratum 0 -> Stratum 1
    theta_1_init = np.array([theta_0_star[0], 0.0, 0.0])
    s1_A = Stratum1(mesh, lambda_realiz=150.0)
    s1_A.theta = theta_1_init.copy()
    
    drift_e1 = np.sqrt(np.sum(W * ((s1_A.predict() - s0.predict(theta_0_star))**2))) / norm_dns
    print(f"  Transport e1 (0 -> 1): Realization Drift delta_R = {drift_e1:.6e}")
    
    opt1_A = AdamOptimizer(lr=2e-3)
    N1_steps = 100
    for _ in range(N1_steps):
        l1, g1 = s1_A.loss_and_grad()
        s1_A.theta = opt1_A.step(s1_A.theta, g1)
        
    theta_1_star = s1_A.theta.copy()
    print(f"  Stratum 1 Calibrated theta_1*: [{theta_1_star[0]:.4f}, {theta_1_star[1]:.4f}, {theta_1_star[2]:.4f}]")

    # 2. Transport e2: Stratum 1 -> Stratum 2
    theta_2A_init = np.array([theta_1_star[0], theta_1_star[1], theta_1_star[2], 0.0])
    s2_A = Stratum2(mesh, lambda_realiz=150.0)
    s2_A.theta = theta_2A_init.copy()
    
    drift_e2 = np.sqrt(np.sum(W * ((s2_A.predict() - s1_A.predict(theta_1_star))**2))) / norm_dns
    print(f"  Transport e2 (1 -> 2): Realization Drift delta_R = {drift_e2:.6e}")
    
    # 3. Optimize Stratum 2 along Path A
    opt2_A = AdamOptimizer(lr=2e-3)
    N2_steps = 120
    history_A = {"loss": [], "realiz_viol": []}
    
    for step in range(N2_steps):
        l2, g2 = s2_A.loss_and_grad()
        history_A["loss"].append(float(l2))
        realiz_pct = check_realizability(mesh, s2_A.predict())
        history_A["realiz_viol"].append(1.0 - realiz_pct)
        s2_A.theta = opt2_A.step(s2_A.theta, g2)
        
    theta_2A_star = s2_A.theta.copy()
    print(f"  Final Path A theta_2,A*: [{theta_2A_star[0]:.4f}, {theta_2A_star[1]:.4f}, {theta_2A_star[2]:.4f}, {theta_2A_star[3]:.4f}]")
    print(f"  Final Path A Loss:      {history_A['loss'][-1]:.6e}")

    # ==========================================================================
    # PATH B: Direct Jump (Stratum 0 -> Stratum 2)
    # ==========================================================================
    print("\n" + "-" * 60)
    print("PATH B: Direct Jump (0 -> 2)")
    print("-" * 60)

    # Direct Transport e_direct: Stratum 0 -> Stratum 2
    theta_2B_init = np.array([theta_0_star[0], 0.0, 0.0, 0.0])
    s2_B = Stratum2(mesh, lambda_realiz=150.0)
    s2_B.theta = theta_2B_init.copy()
    
    drift_direct = np.sqrt(np.sum(W * ((s2_B.predict() - s0.predict(theta_0_star))**2))) / norm_dns
    print(f"  Direct Transport (0 -> 2): Realization Drift delta_R = {drift_direct:.6e}")
    
    opt2_B = AdamOptimizer(lr=2e-3)
    history_B = {"loss": [], "realiz_viol": []}
    
    for step in range(N2_steps):
        l2, g2 = s2_B.loss_and_grad()
        history_B["loss"].append(float(l2))
        realiz_pct = check_realizability(mesh, s2_B.predict())
        history_B["realiz_viol"].append(1.0 - realiz_pct)
        s2_B.theta = opt2_B.step(s2_B.theta, g2)
        
    theta_2B_star = s2_B.theta.copy()
    print(f"  Final Path B theta_2,B*: [{theta_2B_star[0]:.4f}, {theta_2B_star[1]:.4f}, {theta_2B_star[2]:.4f}, {theta_2B_star[3]:.4f}]")
    print(f"  Final Path B Loss:      {history_B['loss'][-1]:.6e}")

    # ==========================================================================
    # Quantitative Curvature & Holonomy Computation
    # ==========================================================================
    print("\n" + "=" * 75)
    print("CURVATURE & HOLONOMY QUANTIFICATION")
    print("=" * 75)

    # 1. Parameter Space Holonomy Distance
    param_dist = np.linalg.norm(theta_2A_star - theta_2B_star)
    print(f"Parameter Holonomy ||theta_2,A* - theta_2,B*||:    {param_dist:.6f}")

    # 2. Semantic Realization Defect (Curvature in realization space)
    tau_2A = s2_A.predict(theta_2A_star)
    tau_2B = s2_B.predict(theta_2B_star)
    realiz_defect = np.sqrt(np.sum(W * ((tau_2A - tau_2B)**2))) / norm_dns
    print(f"Semantic Realization Defect C_AB (Curvature):      {realiz_defect:.6e}")

    # 3. Realizability Violation Comparison
    max_viol_A = max(history_A["realiz_viol"]) * 100
    max_viol_B = max(history_B["realiz_viol"]) * 100
    avg_viol_A = np.mean(history_A["realiz_viol"]) * 100
    avg_viol_B = np.mean(history_B["realiz_viol"]) * 100
    print(f"Peak Domain Realizability Violation (A vs B):      {max_viol_A:.1f}% vs {max_viol_B:.1f}%")
    print(f"Mean Domain Realizability Violation (A vs B):      {avg_viol_A:.1f}% vs {avg_viol_B:.1f}%")

    # 4. Final Loss
    print(f"Final Loss Ratio L_B / L_A:                         {history_B['loss'][-1] / history_A['loss'][-1]:.2f}x")

    # Save results to JSON
    results = {
        "theta_0_star": theta_0_star.tolist(),
        "theta_1_star": theta_1_star.tolist(),
        "theta_2A_star": theta_2A_star.tolist(),
        "theta_2B_star": theta_2B_star.tolist(),
        "param_holonomy": float(param_dist),
        "semantic_curvature": float(realiz_defect),
        "history_A": history_A,
        "history_B": history_B,
        "ground_truth": [mesh.c_mu_true, mesh.c1_true, mesh.c2_true, mesh.c3_true]
    }
    with open("results_curvature_experiment.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate Figure
    plot_curvature_results(history_A, history_B, theta_2A_star, theta_2B_star, 
                           [mesh.c_mu_true, mesh.c1_true, mesh.c2_true, mesh.c3_true],
                           param_dist, realiz_defect)

def plot_curvature_results(hist_A, hist_B, theta_A, theta_B, theta_true, param_dist, realiz_defect):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    steps = np.arange(len(hist_A["loss"]))
    
    # --------------------------------------------------------------------------
    # Panel 1: Loss Trajectory in Stratum 2
    # --------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(steps, hist_A["loss"], label=r'Path A: Sequential ($0 \to 1 \to 2$)', 
             color='#1b9e77', linewidth=2.4)
    ax1.plot(steps, hist_B["loss"], label=r'Path B: Direct ($0 \to 2$)', 
             color='#d95f02', linewidth=2.2, linestyle='--')
    ax1.set_yscale('log')
    ax1.set_xlabel('Optimization Steps in Stratum 2 (Cubic Pope)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'Loss $L_2(\theta) = L_{\rm task} + \lambda L_{\rm realiz}$', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Stratum 2 Loss Trajectories', fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(frameon=True, fontsize=10)

    # --------------------------------------------------------------------------
    # Panel 2: Realizability Violation History
    # --------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(steps, np.array(hist_A["realiz_viol"]) * 100, label=r'Path A: Sequential', 
             color='#1b9e77', linewidth=2.4)
    ax2.plot(steps, np.array(hist_B["realiz_viol"]) * 100, label=r'Path B: Direct Jump', 
             color='#d95f02', linewidth=2.2, linestyle='--')
    ax2.set_xlabel('Optimization Steps in Stratum 2', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Unrealizable Domain Fraction $(\tau \not\succeq 0)$ [%]', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Realizability Boundary Traversal', fontsize=12, fontweight='bold')
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend(frameon=True, fontsize=10)

    # --------------------------------------------------------------------------
    # Panel 3: Holonomy & Parameter Divergence
    # --------------------------------------------------------------------------
    ax3 = axes[2]
    labels = [r'$C_\mu$', r'$c_1$', r'$c_2$', r'$c_3$']
    x = np.arange(len(labels))
    width = 0.25

    ax3.bar(x - width, theta_true, width, label='Target Spec', color='#7570b3', alpha=0.85, edgecolor='black')
    ax3.bar(x, theta_A, width, label=r'Path A ($\theta_{2,A}^*$)', color='#1b9e77', alpha=0.85, edgecolor='black')
    ax3.bar(x + width, theta_B, width, label=r'Path B ($\theta_{2,B}^*$)', color='#d95f02', alpha=0.85, edgecolor='black')

    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax3.set_ylabel('Parameter Value', fontsize=11, fontweight='bold')
    ax3.set_title(f'(c) Holonomy: Parameter Divergence\n' + 
                  rf'$\|\theta_A - \theta_B\| = {param_dist:.4f}, \; C_{{AB}} = {realiz_defect:.2e}$', 
                  fontsize=12, fontweight='bold')
    ax3.grid(axis='y', ls="--", alpha=0.4)
    ax3.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    output_path = "curvature_results.png"
    plt.savefig(output_path)
    print(f"\nFigure saved successfully to '{output_path}'.")

if __name__ == "__main__":
    run_curvature_experiment()
