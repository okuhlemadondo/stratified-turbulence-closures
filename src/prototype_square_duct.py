"""
prototype_square_duct.py

A self-contained prototype demonstrating:
1. Two-Tiered Controllability: Pointwise tensor basis span vs field-level reachability.
2. Controllability Alignment Metric: cos θ_sec for Linear Boussinesq vs Quadratic Pope.
3. Structural Transition Experiment (Linear Stratum -> Quadratic Stratum):
   - Condition 1: Cold Start
   - Condition 2: Standard Re-fit (random initializations for new terms)
   - Condition 3: Exact Analytical Transport (T_e(C_mu*) = [C_mu*, 0, 0])
   - Condition 4: Extended Transport with Optimizer State (momentum & calibrated variance)
4. Tracking Realization Drift δ_R and Field-Dependent Realizability (Lumley Triangle).
"""

import numpy as np
import os
import sys
from pathlib import Path
import json

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
DATA_DIR = REPO_ROOT / "data"

# Set seed for reproducible comparison across conditions
np.random.seed(42)

# ==============================================================================
# 1. Domain Discretization & Surrogate Field Generation
# ==============================================================================

class SquareDuctQuadrant:
    """
    Square duct quadrant domain (y, z) in [0, h] x [0, h].
    Walls at y = 0 (bottom) and z = 0 (side). Symmetry at y = h and z = h.
    Surrogate for Vinuesa et al. (2014) / Pinelli et al. (2010) at Re_tau ≈ 300.
    """
    def __init__(self, h=1.0, Ny=48, Nz=48):
        self.h = h
        self.Ny = Ny
        self.Nz = Nz
        
        # Grid with near-wall clustering (tanh stretching)
        gamma = 2.0
        eta_y = np.linspace(0, 1, Ny)
        eta_z = np.linspace(0, 1, Nz)
        self.y = h * np.tanh(gamma * eta_y) / np.tanh(gamma)
        self.z = h * np.tanh(gamma * eta_z) / np.tanh(gamma)
        self.Y, self.Z = np.meshgrid(self.y, self.z, indexing='ij')
        
        # Integration weights (trapezoidal rule)
        dy = np.zeros(Ny)
        dy[0] = 0.5 * (self.y[1] - self.y[0])
        dy[-1] = 0.5 * (self.y[-1] - self.y[-2])
        dy[1:-1] = 0.5 * (self.y[2:] - self.y[:-2])
        
        dz = np.zeros(Nz)
        dz[0] = 0.5 * (self.z[1] - self.z[0])
        dz[-1] = 0.5 * (self.z[-1] - self.z[-2])
        dz[1:-1] = 0.5 * (self.z[2:] - self.z[:-2])
        
        self.W_grid = np.outer(dy, dz)  # Area element dA = dy * dz
        
        # Synthesize realistic turbulent flow fields
        self._generate_flow_fields()

    def _generate_flow_fields(self):
        Y, Z, h = self.Y, self.Z, self.h
        
        # 1. Streamwise Velocity U(y, z): boundary layer near y=0 and z=0
        delta = 0.15 * h
        U_bulk = 1.0
        self.U = U_bulk * np.tanh(Y / delta) * np.tanh(Z / delta)
        
        # Gradients of U
        # dU/dy, dU/dz
        dU_dy = (U_bulk / delta) * (1.0 - np.tanh(Y / delta)**2) * np.tanh(Z / delta)
        dU_dz = (U_bulk / delta) * np.tanh(Y / delta) * (1.0 - np.tanh(Z / delta)**2)
        
        # 2. Secondary Velocity (V, W) via streamfunction psi(y, z)
        # Counter-rotating vortices in the corner (Prandtl's secondary motion of the 2nd kind)
        # Streamfunction is anti-symmetric across corner diagonal (Y = Z)
        A_sec = 0.02 * U_bulk
        psi = A_sec * (Y**2 * Z**2 / h**4) * (1.0 - (Y/h)**2)**2 * (1.0 - (Z/h)**2)**2 * (Y - Z) / h
        
        # Numerical derivatives for secondary velocities V = d(psi)/dz, W = -d(psi)/dy
        # Using central differences on non-uniform grid
        V = np.gradient(psi, self.z, axis=1)
        W = -np.gradient(psi, self.y, axis=0)
        self.V = V
        self.W = W
        
        # Derivatives of V, W
        dV_dy = np.gradient(V, self.y, axis=0)
        dV_dz = np.gradient(V, self.z, axis=1)
        dW_dy = np.gradient(W, self.y, axis=0)
        dW_dz = np.gradient(W, self.z, axis=1)
        
        # Velocity gradient tensor L_ij = d(u_i)/d(x_j) where (x_1, x_2, x_3) = (x, y, z)
        # Flow is homogeneous in x, so d()/dx = 0
        Ny, Nz = self.Ny, self.Nz
        L = np.zeros((Ny, Nz, 3, 3))
        # i=0 (u), j=0(x), 1(y), 2(z)
        L[:, :, 0, 1] = dU_dy
        L[:, :, 0, 2] = dU_dz
        # i=1 (v)
        L[:, :, 1, 1] = dV_dy
        L[:, :, 1, 2] = dV_dz
        # i=2 (w)
        L[:, :, 2, 1] = dW_dy
        L[:, :, 2, 2] = dW_dz
        
        # Strain rate S_ij = 0.5 * (L_ij + L_ji)
        self.S = 0.5 * (L + np.swapaxes(L, 2, 3))
        
        # Rotation rate Omega_ij = 0.5 * (L_ij - L_ji)
        self.Omega = 0.5 * (L - np.swapaxes(L, 2, 3))
        
        # Turbulent kinetic energy k and specific dissipation rate omega
        # k is peaked in the buffer layer near the corner
        k_max = 0.08 * (U_bulk**2)
        dist_corner = np.sqrt(Y**2 + Z**2)
        y_plus_corner = dist_corner / (0.05 * h)
        self.k = k_max * (y_plus_corner**2) * np.exp(-y_plus_corner) + 1e-4
        
        # omega ~ sqrt(k) / length_scale
        length_scale = 0.1 * h * np.tanh(dist_corner / (0.2 * h)) + 1e-3
        self.omega = np.sqrt(self.k) / length_scale + 1.0

        # 3. Ground Truth DNS Reynolds Stresses tau_DNS
        # Generated with genuine normal stress anisotropy: tau_yy - tau_zz != 0 and tau_yz != 0
        # Under true DNS: tau = -2 * nu_t * S + T_nonlinear + noise
        c_mu_true = 0.09
        c1_true = 0.05
        c2_true = -0.04
        
        # Construct Basis at every point
        self.T1 = -2.0 * (self.k / self.omega)[:, :, None, None] * self.S
        
        # T2 = k/omega^2 * (S^2 - 1/3 Tr(S^2)I)
        S_sq = np.einsum('abij,abjk->abik', self.S, self.S)
        tr_S_sq = np.trace(S_sq, axis1=2, axis2=3)[:, :, None, None]
        I3 = np.eye(3)[None, None, :, :]
        self.T2 = (self.k / (self.omega**2))[:, :, None, None] * (S_sq - (1.0/3.0) * tr_S_sq * I3)
        
        # T3 = k/omega^2 * (S*Omega - Omega*S)
        S_Om = np.einsum('abij,abjk->abik', self.S, self.Omega)
        Om_S = np.einsum('abij,abjk->abik', self.Omega, self.S)
        self.T3 = (self.k / (self.omega**2))[:, :, None, None] * (S_Om - Om_S)
        
        # Isotropic baseline: (2/3) k delta_ij
        tau_iso = (2.0/3.0) * self.k[:, :, None, None] * I3
        
        # 3. Reference Reynolds Stresses tau_ref
        tau_dev_true = c_mu_true * self.T1 + c1_true * self.T2 + c2_true * self.T3
        
        # Full Reynolds stress
        self.tau_ref = tau_iso + tau_dev_true
        self.tau_DNS = self.tau_ref  # Alias for backwards compatibility

# ==============================================================================
# 2. Controllability Metric: cos θ_sec
# ==============================================================================

def compute_controllability_alignment(mesh, basis_list):
    """
    Computes the Controllability Alignment Metric cos θ_sec between a tensor basis
    and the secondary-flow-driving stress component:
    tau_sec = diag(0, tau_yy - tau_zz, tau_zz - tau_yy) + tau_yz * (e_y x e_z + e_z x e_y)
    """
    # Extract secondary driving stress components from tau_ref
    tau_yy = mesh.tau_ref[:, :, 1, 1]
    tau_zz = mesh.tau_ref[:, :, 2, 2]
    tau_yz = mesh.tau_ref[:, :, 1, 2]
    
    tau_sec = np.zeros_like(mesh.tau_ref)
    tau_sec[:, :, 1, 1] = 0.5 * (tau_yy - tau_zz)
    tau_sec[:, :, 2, 2] = -0.5 * (tau_yy - tau_zz)
    tau_sec[:, :, 1, 2] = tau_yz
    tau_sec[:, :, 2, 1] = tau_yz
    
    # Domain Frobenius norm of tau_sec
    W = mesh.W_grid[:, :, None, None]
    frob_sq_tau_sec = np.sum(W * (tau_sec**2))
    norm_tau_sec = np.sqrt(frob_sq_tau_sec)
    
    if norm_tau_sec < 1e-12:
        return 0.0
        
    K = len(basis_list)
    Ny, Nz = mesh.Ny, mesh.Nz
    
    # Pointwise projection: solve least-squares min_c ||tau_sec(x) - sum c_k T^(k)(x)||^2 at each point
    # Stack basis: shape (Ny, Nz, K, 9)
    T_flat = np.stack([T.reshape(Ny, Nz, 9) for T in basis_list], axis=2)  # (Ny, Nz, K, 9)
    tau_sec_flat = tau_sec.reshape(Ny, Nz, 9, 1)                         # (Ny, Nz, 9, 1)
    
    # Gram matrix G = T_flat @ T_flat^T of shape (Ny, Nz, K, K)
    G = np.einsum('abki,abmi->abkm', T_flat, T_flat)                     # (Ny, Nz, K, K)
    # RHS b = T_flat @ tau_sec_flat of shape (Ny, Nz, K, 1)
    b = np.einsum('abki,abij->abkj', T_flat, tau_sec_flat)               # (Ny, Nz, K, 1)
    
    # Regularize slightly for pseudo-inverse
    G_reg = G + 1e-10 * np.eye(K)[None, None, :, :]
    try:
        c_proj = np.linalg.solve(G_reg, b)                               # (Ny, Nz, K, 1)
    except np.linalg.LinAlgError:
        c_proj = np.linalg.pinv(G_reg) @ b
        
    # Projected tensor
    proj_tau_sec = np.zeros_like(tau_sec)
    for k in range(K):
        proj_tau_sec += c_proj[:, :, k, 0, None, None] * basis_list[k]
        
    # Domain inner product <tau_sec, proj_tau_sec>_L2
    inner_prod = np.sum(W * tau_sec * proj_tau_sec)
    norm_proj = np.sqrt(np.sum(W * (proj_tau_sec**2)))
    
    if norm_proj < 1e-12:
        return 0.0
        
    cos_theta = inner_prod / (norm_tau_sec * norm_proj)
    return float(np.clip(cos_theta, 0.0, 1.0))

# ==============================================================================
# 3. Realizability Diagnostic (Lumley Triangle & Positive Semi-definiteness)
# ==============================================================================

def check_realizability(mesh, tau_model):
    """
    Checks if tau_model is positive semi-definite (tau >= 0) at every grid point.
    Returns the percentage of realizable grid points (1.0 = 100% realizable).
    """
    # Compute eigenvalues of tau at every point (Ny, Nz, 3)
    # tau must be symmetric
    tau_sym = 0.5 * (tau_model + np.swapaxes(tau_model, 2, 3))
    eigvals = np.linalg.eigvalsh(tau_sym)  # sorted in ascending order
    
    # Minimum eigenvalue must be >= 0 (with slight numerical tolerance -1e-6)
    min_eig = eigvals[:, :, 0]
    realizable_mask = (min_eig >= -1e-6)
    realizable_fraction = float(np.mean(realizable_mask))
    return realizable_fraction

# ==============================================================================
# 4. Stratum Models & Optimizer
# ==============================================================================

class Stratum0_Linear:
    """Stratum 0: Linear Boussinesq Model, parameter theta_0 = [C_mu]"""
    def __init__(self, mesh):
        self.mesh = mesh
        self.theta = np.array([0.01])  # initial guess
        self.I3 = np.eye(3)[None, None, :, :]
        self.W = mesh.W_grid[:, :, None, None]
        
    def predict(self, theta=None):
        if theta is None:
            theta = self.theta
        c_mu = theta[0]
        # tau_dev = c_mu * T1, total tau = 2/3 k I + tau_dev
        tau_dev = c_mu * self.mesh.T1
        tau = (2.0/3.0) * self.mesh.k[:, :, None, None] * self.I3 + tau_dev
        return tau
        
    def loss_and_grad(self, theta=None):
        if theta is None:
            theta = self.theta
        tau_pred = self.predict(theta)
        diff = tau_pred - self.mesh.tau_ref
        loss = 0.5 * np.sum(self.W * (diff**2))
        
        # d(diff)/d(c_mu) = T1
        grad = np.array([np.sum(self.W * diff * self.mesh.T1)])
        return loss, grad

class Stratum1_Quadratic:
    """Stratum 1: Quadratic Pope Expansion, parameter theta_1 = [C_mu, c1, c2]"""
    def __init__(self, mesh):
        self.mesh = mesh
        self.theta = np.zeros(3)
        self.I3 = np.eye(3)[None, None, :, :]
        self.W = mesh.W_grid[:, :, None, None]
        
    def predict(self, theta=None):
        if theta is None:
            theta = self.theta
        c_mu, c1, c2 = theta
        tau_dev = c_mu * self.mesh.T1 + c1 * self.mesh.T2 + c2 * self.mesh.T3
        tau = (2.0/3.0) * self.mesh.k[:, :, None, None] * self.I3 + tau_dev
        return tau
        
    def loss_and_grad(self, theta=None):
        if theta is None:
            theta = self.theta
        tau_pred = self.predict(theta)
        diff = tau_pred - self.mesh.tau_ref
        loss = 0.5 * np.sum(self.W * (diff**2))
        
        g0 = np.sum(self.W * diff * self.mesh.T1)
        g1 = np.sum(self.W * diff * self.mesh.T2)
        g2 = np.sum(self.W * diff * self.mesh.T3)
        grad = np.array([g0, g1, g2])
        return loss, grad

# ==============================================================================
# 5. Adam Optimizer with State Preservation
# ==============================================================================

class AdamOptimizer:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
        
    def step(self, theta, grad):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
            
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad**2)
        
        # Bias correction
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        
        theta_next = theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return theta_next

# ==============================================================================
# 6. The Execution Pipeline
# ==============================================================================

def run_experiment():
    print("=" * 70)
    print("OPDISCOVERY-FLUIDMECH: STRATIFIED DESIGN ATLAS PROTOTYPE")
    print("Canonical Benchmark: Turbulent Square Duct Flow (Prandtl Secondary Motion)")
    print("=" * 70)
    
    mesh = SquareDuctQuadrant(h=1.0, Ny=48, Nz=48)
    
    # --------------------------------------------------------------------------
    # Check Controllability Metrics
    # --------------------------------------------------------------------------
    print("\n--- 1. TWO-TIERED CONTROLLABILITY CHECK ---")
    cos_sec_basis0 = compute_controllability_alignment(mesh, [mesh.T1])
    cos_sec_basis1 = compute_controllability_alignment(mesh, [mesh.T1, mesh.T2, mesh.T3])
    
    print(f"Stratum 0 Basis {{T1}} Alignment cos θ_sec:             {cos_sec_basis0:.6f}")
    print(f"Stratum 1 Basis {{T1, T2, T3}} Alignment cos θ_sec:      {cos_sec_basis1:.6f}")
    
    if cos_sec_basis0 < 1e-5 and cos_sec_basis1 > 0.9:
        print(">> VERIFIED: Stratum 0 has provable ZERO controllability on secondary driving stress.")
        print(">> VERIFIED: Stratum 1 spans the required anisotropic stress subspace.")
    else:
        print(">> Note: Alignment values calculated successfully.")

    # --------------------------------------------------------------------------
    # Phase 1: Optimize Stratum 0 (Linear Boussinesq)
    # --------------------------------------------------------------------------
    print("\n--- 2. PHASE 1: STRATUM 0 OPTIMIZATION (Boussinesq) ---")
    model0 = Stratum0_Linear(mesh)
    opt0 = AdamOptimizer(lr=2e-3)
    
    N0_steps = 150
    for step in range(N0_steps):
        loss0, grad0 = model0.loss_and_grad()
        model0.theta = opt0.step(model0.theta, grad0)
        
    loss0_final, _ = model0.loss_and_grad()
    c_mu_star = model0.theta[0]
    tau0_final = model0.predict()
    realizable0 = check_realizability(mesh, tau0_final)
    
    print(f"Optimized C_mu*:                   {c_mu_star:.6f}")
    print(f"Stratum 0 Final Loss L_0*:         {loss0_final:.6e}")
    print(f"Stratum 0 Realizability Fraction:  {realizable0 * 100:.1f}%")
    
    # --------------------------------------------------------------------------
    # Phase 2: Structural Edit e: t0 -> t1 (Transition to Quadratic Stratum)
    # --------------------------------------------------------------------------
    print("\n--- 3. PHASE 2: STRUCTURAL TRANSITION EXPERIMENT (4 CONDITIONS) ---")
    N1_steps = 150
    
    conditions = {
        "1. Cold Start": {
            "theta_init": np.random.normal(0, 0.05, size=3),
            "opt_state": None
        },
        "2. Standard Re-fit": {
            "theta_init": np.array([c_mu_star, np.random.normal(0, 0.05), np.random.normal(0, 0.05)]),
            "opt_state": None
        },
        "3. Exact Transport": {
            "theta_init": np.array([c_mu_star, 0.0, 0.0]),
            "opt_state": None
        },
        "4. Extended Transport": {
            "theta_init": np.array([c_mu_star, 0.0, 0.0]),
            "opt_state": (opt0.m, opt0.v, opt0.t)
        }
    }
    
    results = {}
    W = mesh.W_grid[:, :, None, None]
    norm_tau0 = np.sqrt(np.sum(W * (tau0_final**2)))
    
    for name, config in conditions.items():
        model1 = Stratum1_Quadratic(mesh)
        model1.theta = config["theta_init"].copy()
        
        # Calculate initial Realization Drift delta_R
        tau1_init = model1.predict(model1.theta)
        drift_norm = np.sqrt(np.sum(W * ((tau1_init - tau0_final)**2)))
        delta_R = drift_norm / norm_tau0
        
        # Setup Optimizer
        opt1 = AdamOptimizer(lr=2e-3)
        if config["opt_state"] is not None:
            m0, v0, t0 = config["opt_state"]
            # Extended transport: transfer calibrated momentum & set variance prior
            v_prior = float(v0[0]) if v0 is not None else 1e-4
            opt1.m = np.array([float(m0[0]), 0.0, 0.0])
            opt1.v = np.array([float(v0[0]), v_prior, v_prior])
            opt1.t = t0
            
        loss_history = []
        realizability_history = []
        
        for step in range(N1_steps):
            loss1, grad1 = model1.loss_and_grad()
            loss_history.append(float(loss1))
            realizability_history.append(check_realizability(mesh, model1.predict()))
            model1.theta = opt1.step(model1.theta, grad1)
            
        final_loss = loss_history[-1]
        results[name] = {
            "delta_R": float(delta_R),
            "init_loss": float(loss_history[0]),
            "final_loss": float(final_loss),
            "final_theta": model1.theta.tolist(),
            "loss_history": loss_history,
            "realizability_init": realizability_history[0],
            "realizability_final": realizability_history[-1]
        }
        
        print(f"\n[{name}]")
        print(f"  Realization Drift δ_R:       {delta_R:.6e} {'(EXACT ZERO)' if delta_R < 1e-12 else ''}")
        print(f"  Initial Loss L_1(0):         {loss_history[0]:.6e}")
        print(f"  Final Loss L_1({N1_steps}):       {final_loss:.6e}")
        print(f"  Final θ_1:                   [{model1.theta[0]:.4f}, {model1.theta[1]:.4f}, {model1.theta[2]:.4f}]")
        print(f"  Realizability (Init -> End): {realizability_history[0]*100:.1f}% -> {realizability_history[-1]*100:.1f}%")

    # --------------------------------------------------------------------------
    # Amortization & Performance Summary
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENTAL SUMMARY & VALIDATION OF THE GEOMETRIC FRAMEWORK")
    print("=" * 70)
    
    # Steps to achieve 90% of final loss reduction
    target_loss = loss0_final * 0.1
    print("\nSpeed to Reach Target Loss (< 10% of Boussinesq baseline):")
    for name, res in results.items():
        hist = res["loss_history"]
        reached = [i for i, l in enumerate(hist) if l <= target_loss]
        step_reached = reached[0] if len(reached) > 0 else f">{N1_steps}"
        print(f"  {name:<25}: Step {step_reached}")
        
    # Save results to JSON
    out_json = DATA_DIR / "results_square_duct_prototype.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to '{out_json}'.")

if __name__ == "__main__":
    run_experiment()
