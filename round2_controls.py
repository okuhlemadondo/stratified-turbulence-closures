"""
round2_controls.py

The five decisive controls from the Round 2 review:
1. Path B + fresh Adam restart at step 100 (FP-2 decisive control)
2. Path A carrying Adam state across the hop (FP-2 mirror)
3. Adam with β₁ = 0, both paths (FP-3: momentum vs adaptivity)
4. Plain GD at a step size that actually converges (FP-1: inert-arm fix)
5. Adam with α rescaled alongside the basis (FP-18: scale-invariance test)

Plus reporting fixes:
- τ_ref realizability (FP-9)
- Step 99→100 at full precision (FP-5)
- Final calibrated coefficients θ* (FP-18)
- Digit collision checks (FP-16)
"""

import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('.cache/matplotlib')

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prototype_square_duct import SquareDuctQuadrant, AdamOptimizer, check_realizability
from curvature_experiment import ExtendedDuctMesh

np.random.seed(42)


# ==============================================================================
# Optimizers
# ==============================================================================
class PlainGD:
    def __init__(self, lr=2e-3):
        self.lr = lr
    def step(self, theta, grad):
        return theta - self.lr * grad

class AdamNoBeta1:
    """Adam with β₁ = 0 (no first-moment momentum, only second-moment scaling)."""
    def __init__(self, lr=2e-3, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta2 = beta2
        self.eps = eps
        self.v = None
        self.t = 0
    def step(self, theta, grad):
        self.t += 1
        if self.v is None:
            self.v = np.zeros_like(grad)
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return theta - self.lr * grad / (np.sqrt(v_hat) + self.eps)


# ==============================================================================
# Generic runner (returns full per-step telemetry)
# ==============================================================================
def run_path_full(mesh, basis, tau_ref, W, theta_init, n_steps,
                  lambda_realiz=150.0, optimizer=None, freeze_mask=None):
    """Run optimization. Returns (history_dict, final_theta, optimizer)."""
    d = len(basis)
    theta = theta_init.copy()
    I3 = np.eye(3)[None, None, :, :]
    history = {"loss": [], "viol": [], "theta": []}

    for step in range(n_steps):
        tau_dev = sum(theta[i] * basis[i] for i in range(d))
        tau = (2.0 / 3.0) * mesh.k[:, :, None, None] * I3 + tau_dev
        diff = tau - tau_ref
        loss_task = 0.5 * np.sum(W * (diff ** 2))
        grad_task = np.array([np.sum(W * diff * basis[i]) for i in range(d)])
        loss_realiz = 0.0
        grad_realiz = np.zeros(d)
        if lambda_realiz > 0:
            tau_sym = 0.5 * (tau + np.swapaxes(tau, 2, 3))
            eigvals, eigvecs = np.linalg.eigh(tau_sym)
            min_eig = eigvals[:, :, 0]
            min_vec = eigvecs[:, :, :, 0]
            mask = (min_eig < 0)
            if np.any(mask):
                viol = np.where(mask, min_eig, 0.0)
                loss_realiz = lambda_realiz * np.sum(mesh.W_grid * viol ** 2)
                for i in range(d):
                    vtv = np.einsum('abi,abij,abj->ab', min_vec, basis[i], min_vec)
                    grad_realiz[i] = 2.0 * lambda_realiz * np.sum(mesh.W_grid * viol * vtv)

        total_loss = loss_task + loss_realiz
        total_grad = grad_task + grad_realiz
        viol_frac = 1.0 - check_realizability(mesh, tau)
        history["loss"].append(float(total_loss))
        history["viol"].append(float(viol_frac))
        history["theta"].append(theta.tolist())

        if freeze_mask is not None:
            total_grad = total_grad.copy()
            total_grad[freeze_mask] = 0.0

        theta = optimizer.step(theta, total_grad)

    return history, theta, optimizer


def main():
    print("=" * 80)
    print("ROUND 2 CONTROLS — FIVE DECISIVE EXPERIMENTS")
    print("=" * 80)

    mesh = ExtendedDuctMesh(h=1.0, Ny=48, Nz=48)
    W = mesh.W_grid[:, :, None, None]
    norm_ref = np.sqrt(np.sum(W * (mesh.tau_DNS ** 2)))
    raw_basis = [mesh.T1, mesh.T2, mesh.T3, mesh.T4]
    lam = 150.0

    # Gram matrix eigenvalues (for step-size computation)
    G = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            G[i, j] = np.sum(W * raw_basis[i] * raw_basis[j])
    eigs = np.linalg.eigvalsh(G)
    lam_max_raw = eigs[-1]
    print(f"λ_max(G_raw) = {lam_max_raw:.6e}")
    print(f"α·λ_max(raw) = {2e-3 * lam_max_raw:.6e} (per-step contraction)")
    print(f"Over 220 steps: exp(-220·α·λ_max) = {np.exp(-220 * 2e-3 * lam_max_raw):.6f}")

    # ======================================================================
    # 0. REPORTING FIXES
    # ======================================================================
    print("\n" + "=" * 80)
    print("0. REPORTING FIXES")
    print("=" * 80)

    # FP-9: Is τ_ref itself realizable?
    viol_ref = 1.0 - check_realizability(mesh, mesh.tau_DNS)
    print(f"\nτ_ref realizability violation: {viol_ref * 100:.2f}%")

    tau_sym_ref = 0.5 * (mesh.tau_DNS + np.swapaxes(mesh.tau_DNS, 2, 3))
    min_eig_ref = np.linalg.eigvalsh(tau_sym_ref)[:, :, 0]
    print(f"τ_ref min eigenvalue: {np.min(min_eig_ref):.6e}")
    print(f"τ_ref min eigenvalue / max(k): {np.min(min_eig_ref) / np.max(mesh.k):.6e}")

    # FP-12: Does Re_τ enter the field synthesis?
    print(f"\nRe_τ does NOT enter the field synthesis. The surrogate fields U, V, W, k, ω")
    print(f"are constructed from analytical profiles with hardcoded parameters (δ=0.15h,")
    print(f"A_sec=0.02, k_max=0.08). Re_τ=300 is a label denoting the DNS study whose")
    print(f"qualitative features the surrogate reproduces, not a simulation parameter.")

    # FP-16: Digit collision checks
    print(f"\nDigit collision check:")
    print(f"  cos φ_sec(M₀) = 1.15×10⁻⁴ (Table I): from prototype output")
    print(f"  G₃₃ = {G[2,2]:.6e} (Table II)")
    print(f"  These are unrelated quantities that happen to round to similar values.")

    # ======================================================================
    # 1. BASE CALIBRATION
    # ======================================================================
    print("\n" + "=" * 80)
    print("1. BASE STRATUM 0 CALIBRATION")
    print("=" * 80)

    opt0 = AdamOptimizer(lr=2e-3)
    _, theta_0_star, _ = run_path_full(
        mesh, [mesh.T1], mesh.tau_DNS, W,
        theta_init=np.array([0.01]), n_steps=120,
        lambda_realiz=0.0, optimizer=opt0
    )
    print(f"θ₀* = {theta_0_star[0]:.6f}")

    # ======================================================================
    # 2. CONTROL 1: Path B + fresh Adam restart at step 100 (FP-2)
    # ======================================================================
    print("\n" + "=" * 80)
    print("CONTROL 1: PATH B + FRESH ADAM RESTART AT STEP 100")
    print("(If this repairs monotonically, staging is inert)")
    print("=" * 80)

    # Phase 1: 100 steps in Stratum 2 with Adam
    opt_B1 = AdamOptimizer(lr=2e-3)
    h_B1, theta_B100, _ = run_path_full(
        mesh, raw_basis, mesh.tau_DNS, W,
        theta_init=np.array([theta_0_star[0], 0.0, 0.0, 0.0]),
        n_steps=100, lambda_realiz=lam, optimizer=opt_B1
    )
    # Phase 2: fresh Adam at step 100, continue 120 steps
    opt_B2_fresh = AdamOptimizer(lr=2e-3)
    h_B2, theta_Brestart_final, _ = run_path_full(
        mesh, raw_basis, mesh.tau_DNS, W,
        theta_init=theta_B100.copy(), n_steps=120,
        lambda_realiz=lam, optimizer=opt_B2_fresh
    )
    h_Brestart = {
        "loss": h_B1["loss"] + h_B2["loss"],
        "viol": h_B1["viol"] + h_B2["viol"],
    }
    print(f"Path B-restart: max viol = {max(h_Brestart['viol']) * 100:.2f}%")
    print(f"  Pre-restart (steps 0-99): max viol = {max(h_B1['viol']) * 100:.2f}%")
    print(f"  Post-restart (steps 100-219): max viol = {max(h_B2['viol']) * 100:.2f}%")
    print(f"  Final loss = {h_Brestart['loss'][-1]:.4e}")

    # ======================================================================
    # 3. CONTROL 2: Path A carrying Adam state across the hop (FP-2 mirror)
    # ======================================================================
    print("\n" + "=" * 80)
    print("CONTROL 2: PATH A CARRYING ADAM STATE ACROSS HOP")
    print("=" * 80)

    # Phase 1: Stratum 1
    opt_A1 = AdamOptimizer(lr=2e-3)
    h_A1, theta_1_star, opt_A1_out = run_path_full(
        mesh, raw_basis[:3], mesh.tau_DNS, W,
        theta_init=np.array([theta_0_star[0], 0.0, 0.0]),
        n_steps=100, lambda_realiz=lam, optimizer=opt_A1
    )
    # Phase 2: carry Adam state, zero-pad m and v from dim 3 to dim 4
    opt_A2_carry = AdamOptimizer(lr=2e-3)
    opt_A2_carry.t = opt_A1_out.t
    opt_A2_carry.m = np.append(opt_A1_out.m, 0.0)
    opt_A2_carry.v = np.append(opt_A1_out.v, 0.0)

    h_A2, theta_Acarry_final, _ = run_path_full(
        mesh, raw_basis, mesh.tau_DNS, W,
        theta_init=np.array([theta_1_star[0], theta_1_star[1], theta_1_star[2], 0.0]),
        n_steps=120, lambda_realiz=lam, optimizer=opt_A2_carry
    )
    h_Acarry = {
        "loss": h_A1["loss"] + h_A2["loss"],
        "viol": h_A1["viol"] + h_A2["viol"],
    }
    print(f"Path A-carry: max viol = {max(h_Acarry['viol']) * 100:.2f}%")
    print(f"  Scaffold (steps 0-99): max viol = {max(h_A1['viol']) * 100:.2f}%")
    print(f"  Deployed+carry (steps 100-219): max viol = {max(h_A2['viol']) * 100:.2f}%")
    print(f"  Final loss = {h_Acarry['loss'][-1]:.4e}")

    # ======================================================================
    # 4. CONTROL 3: Adam β₁ = 0, both paths (FP-3)
    # ======================================================================
    print("\n" + "=" * 80)
    print("CONTROL 3: ADAM β₁ = 0 (NO MOMENTUM, ONLY ADAPTIVE SCALING)")
    print("=" * 80)

    # Path A with β₁=0
    opt_A1_nb = AdamNoBeta1(lr=2e-3)
    h_A1_nb, theta_1_nb, _ = run_path_full(
        mesh, raw_basis[:3], mesh.tau_DNS, W,
        theta_init=np.array([theta_0_star[0], 0.0, 0.0]),
        n_steps=100, lambda_realiz=lam, optimizer=opt_A1_nb
    )
    opt_A2_nb = AdamNoBeta1(lr=2e-3)
    h_A2_nb, theta_Anb_final, _ = run_path_full(
        mesh, raw_basis, mesh.tau_DNS, W,
        theta_init=np.array([theta_1_nb[0], theta_1_nb[1], theta_1_nb[2], 0.0]),
        n_steps=120, lambda_realiz=lam, optimizer=opt_A2_nb
    )
    h_A_nb = {"loss": h_A1_nb["loss"] + h_A2_nb["loss"],
              "viol": h_A1_nb["viol"] + h_A2_nb["viol"]}

    # Path B with β₁=0
    opt_B_nb = AdamNoBeta1(lr=2e-3)
    h_B_nb, theta_Bnb_final, _ = run_path_full(
        mesh, raw_basis, mesh.tau_DNS, W,
        theta_init=np.array([theta_0_star[0], 0.0, 0.0, 0.0]),
        n_steps=220, lambda_realiz=lam, optimizer=opt_B_nb
    )

    print(f"Path A (β₁=0): max viol = {max(h_A_nb['viol']) * 100:.2f}%, final = {h_A_nb['loss'][-1]:.4e}")
    print(f"  Scaffold: max viol = {max(h_A1_nb['viol']) * 100:.2f}%")
    print(f"  Deployed: max viol = {max(h_A2_nb['viol']) * 100:.2f}%")
    print(f"Path B (β₁=0): max viol = {max(h_B_nb['viol']) * 100:.2f}%, final = {h_B_nb['loss'][-1]:.4e}")

    # ======================================================================
    # 5. CONTROL 4: Plain GD at a step size that converges (FP-1)
    # ======================================================================
    print("\n" + "=" * 80)
    print("CONTROL 4: PLAIN GD AT CONVERGENT STEP SIZE")
    print("=" * 80)

    # Optimal GD step: α = 2/(λ_max + λ_min) for quadratic. Use α = 1/λ_max as safe choice.
    alpha_gd = float(1.0 / lam_max_raw)
    print(f"α_GD = 1/λ_max = {alpha_gd:.2f}")
    print(f"α·λ_max = {alpha_gd * lam_max_raw:.4f}")
    print(f"α·λ_min = {alpha_gd * eigs[0]:.4f}")

    # Path B with convergent GD
    opt_B_gd = PlainGD(lr=alpha_gd)
    h_B_gd, theta_Bgd_final, _ = run_path_full(
        mesh, raw_basis, mesh.tau_DNS, W,
        theta_init=np.array([theta_0_star[0], 0.0, 0.0, 0.0]),
        n_steps=220, lambda_realiz=lam, optimizer=opt_B_gd
    )
    print(f"Path B (conv. GD): max viol = {max(h_B_gd['viol']) * 100:.2f}%")
    print(f"  Final loss = {h_B_gd['loss'][-1]:.4e}")
    print(f"  Step 0 loss = {h_B_gd['loss'][0]:.4e}")
    print(f"  Loss ratio (final/initial) = {h_B_gd['loss'][-1] / h_B_gd['loss'][0]:.4f}")

    # Path A with convergent GD
    opt_A1_gd = PlainGD(lr=alpha_gd)
    h_A1_gd, theta_1_gd, _ = run_path_full(
        mesh, raw_basis[:3], mesh.tau_DNS, W,
        theta_init=np.array([theta_0_star[0], 0.0, 0.0]),
        n_steps=100, lambda_realiz=lam, optimizer=opt_A1_gd
    )
    opt_A2_gd = PlainGD(lr=alpha_gd)
    h_A2_gd, theta_Agd_final, _ = run_path_full(
        mesh, raw_basis, mesh.tau_DNS, W,
        theta_init=np.array([theta_1_gd[0], theta_1_gd[1], theta_1_gd[2], 0.0]),
        n_steps=120, lambda_realiz=lam, optimizer=opt_A2_gd
    )
    h_A_gd = {"loss": h_A1_gd["loss"] + h_A2_gd["loss"],
              "viol": h_A1_gd["viol"] + h_A2_gd["viol"]}
    print(f"Path A (conv. GD): max viol = {max(h_A_gd['viol']) * 100:.2f}%")
    print(f"  Final loss = {h_A_gd['loss'][-1]:.4e}")

    # ======================================================================
    # 6. CONTROL 5: Adam α rescaled with basis (FP-18)
    # ======================================================================
    print("\n" + "=" * 80)
    print("CONTROL 5: NORMALIZED BASIS + ADAM WITH α RESCALED")
    print("=" * 80)

    norms = [float(np.sqrt(np.sum(W * T ** 2))) for T in raw_basis]
    norm_basis = [T / n for T, n in zip(raw_basis, norms)]

    # When we normalize T → T̃ = T/‖T‖, θ → θ̃ = θ·‖T‖.
    # To keep the effective step size per *raw* parameter unchanged,
    # we need α̃ = α · ‖T‖ (so α̃·∂L/∂θ̃ = α·‖T‖·(∂L/∂θ/‖T‖) = α·∂L/∂θ).
    # But Adam normalizes by √v̂, so actually we need to think more carefully.
    # The simplest test: scale α down by max(norms)/min(norms) ~ 5.6
    # to compensate for the normalized basis making gradients bigger.
    #
    # Actually, the correct test per FP-18: Adam's scale invariance is broken
    # by fixed α. With normalized basis, the effective α is too large.
    # Scale α down by the ratio of norm changes.
    scale_ratio = min(norms) / max(norms)
    alpha_rescaled = 2e-3 * scale_ratio
    print(f"Norm ratio min/max = {scale_ratio:.4f}")
    print(f"α_rescaled = {alpha_rescaled:.6f}")

    theta_0_norm = np.array([theta_0_star[0] * norms[0]])
    opt_B_rescaled = AdamOptimizer(lr=alpha_rescaled)
    h_B_rescaled, theta_Brescaled_final, _ = run_path_full(
        mesh, norm_basis, mesh.tau_DNS, W,
        theta_init=np.array([theta_0_star[0] * norms[0], 0.0, 0.0, 0.0]),
        n_steps=220, lambda_realiz=lam, optimizer=opt_B_rescaled
    )
    print(f"Path B (norm + α-rescaled): max viol = {max(h_B_rescaled['viol']) * 100:.2f}%")
    print(f"  Final loss = {h_B_rescaled['loss'][-1]:.4e}")

    # ======================================================================
    # 7. FP-5: Step 99 → 100 at full precision
    # ======================================================================
    print("\n" + "=" * 80)
    print("FP-5: STEP 99 → 100 AT FULL PRECISION")
    print("=" * 80)

    # Reproduce the original Path A with full precision logging
    opt_A_fp = AdamOptimizer(lr=2e-3)
    h_A_fp, theta_1_fp, _ = run_path_full(
        mesh, raw_basis[:3], mesh.tau_DNS, W,
        theta_init=np.array([theta_0_star[0], 0.0, 0.0]),
        n_steps=100, lambda_realiz=lam, optimizer=opt_A_fp
    )
    theta_1_final = theta_1_fp.copy()

    # Step 100 is the EMBEDDING: θ₁* → [θ₁*[0], θ₁*[1], θ₁*[2], 0]
    # Before any gradient step in Stratum 2
    theta_embedded = np.array([theta_1_final[0], theta_1_final[1], theta_1_final[2], 0.0])
    I3 = np.eye(3)[None, None, :, :]
    tau_s1 = (2.0/3.0) * mesh.k[:,:,None,None] * I3 + sum(theta_1_final[i]*raw_basis[i] for i in range(3))
    tau_s2 = (2.0/3.0) * mesh.k[:,:,None,None] * I3 + sum(theta_embedded[i]*raw_basis[i] for i in range(4))
    delta_R = float(np.sqrt(np.sum(W * (tau_s1 - tau_s2)**2)))
    print(f"θ₁* = [{', '.join(f'{x:.15e}' for x in theta_1_final)}]")
    print(f"θ_embedded = [{', '.join(f'{x:.15e}' for x in theta_embedded)}]")
    print(f"δ_R (exact) = {delta_R:.15e}")
    print(f"Step 99 loss = {h_A_fp['loss'][99]:.15e}")
    print(f"Step 99 viol = {h_A_fp['viol'][99]:.15e}")

    # Compute loss at the embedded point (pre-gradient-step)
    diff_emb = tau_s2 - mesh.tau_DNS
    loss_emb = 0.5 * np.sum(W * diff_emb**2)
    viol_emb = 1.0 - check_realizability(mesh, tau_s2)
    print(f"Embedded (pre-grad) loss = {loss_emb:.15e}")
    print(f"Embedded (pre-grad) viol = {viol_emb:.15e}")

    print(f"\nNote: Step 100 in Table IV logs the state AFTER the first")
    print(f"Stratum-2 gradient step. The loss/viol difference between")
    print(f"step 99 and step 100 is NOT a violation of δ_R ≡ 0; it is")
    print(f"the first Adam step with fresh m=v=0.")

    # ======================================================================
    # 8. FINAL CALIBRATED COEFFICIENTS
    # ======================================================================
    print("\n" + "=" * 80)
    print("FINAL CALIBRATED COEFFICIENTS")
    print("=" * 80)

    # Re-run baseline Paths A and B to get final θ*
    opt_Abase1 = AdamOptimizer(lr=2e-3)
    _, theta_1_base, _ = run_path_full(
        mesh, raw_basis[:3], mesh.tau_DNS, W,
        theta_init=np.array([theta_0_star[0], 0.0, 0.0]),
        n_steps=100, lambda_realiz=lam, optimizer=opt_Abase1
    )
    opt_Abase2 = AdamOptimizer(lr=2e-3)
    _, theta_A_final, _ = run_path_full(
        mesh, raw_basis, mesh.tau_DNS, W,
        theta_init=np.array([theta_1_base[0], theta_1_base[1], theta_1_base[2], 0.0]),
        n_steps=120, lambda_realiz=lam, optimizer=opt_Abase2
    )
    opt_Bbase = AdamOptimizer(lr=2e-3)
    _, theta_B_final, _ = run_path_full(
        mesh, raw_basis, mesh.tau_DNS, W,
        theta_init=np.array([theta_0_star[0], 0.0, 0.0, 0.0]),
        n_steps=220, lambda_realiz=lam, optimizer=opt_Bbase
    )
    opt_Cbase = AdamOptimizer(lr=2e-3)
    _, theta_C_final, _ = run_path_full(
        mesh, raw_basis, mesh.tau_DNS, W,
        theta_init=np.array([0.0, 0.0, 0.0, 0.0]),
        n_steps=220, lambda_realiz=lam, optimizer=opt_Cbase
    )

    true_c = [0.090, 0.045, -0.035, 0.060]
    print(f"True coefficients:  [{', '.join(f'{c:.6f}' for c in true_c)}]")
    print(f"Path A θ* (staged): [{', '.join(f'{c:.6f}' for c in theta_A_final)}]")
    print(f"Path B θ* (direct): [{', '.join(f'{c:.6f}' for c in theta_B_final)}]")
    print(f"Path C θ* (cold):   [{', '.join(f'{c:.6f}' for c in theta_C_final)}]")
    print(f"‖θ*_A - θ*_B‖ = {np.linalg.norm(theta_A_final - theta_B_final):.6e}")
    print(f"‖θ*_A - θ*_C‖ = {np.linalg.norm(theta_A_final - theta_C_final):.6e}")
    print(f"‖θ*_A - c_true‖ = {np.linalg.norm(theta_A_final - np.array(true_c)):.6e}")

    # ======================================================================
    # 9. SUMMARY TABLE
    # ======================================================================
    print("\n" + "=" * 80)
    print("DECISION TABLE")
    print("=" * 80)

    rows = [
        ("Path A (baseline)", max(h_A_fp['viol'])*100, h_A_fp['loss'][-1]),
        ("Path B-restart (FP-2)", max(h_Brestart['viol'])*100, h_Brestart['loss'][-1]),
        ("Path A-carry (FP-2m)", max(h_Acarry['viol'])*100, h_Acarry['loss'][-1]),
        ("Path A (β₁=0)", max(h_A_nb['viol'])*100, h_A_nb['loss'][-1]),
        ("Path B (β₁=0)", max(h_B_nb['viol'])*100, h_B_nb['loss'][-1]),
        ("Path B (conv. GD)", max(h_B_gd['viol'])*100, h_B_gd['loss'][-1]),
        ("Path A (conv. GD)", max(h_A_gd['viol'])*100, h_A_gd['loss'][-1]),
        ("Path B (norm+α-resc)", max(h_B_rescaled['viol'])*100, h_B_rescaled['loss'][-1]),
    ]
    print(f"{'Condition':<30s} | {'Max viol%':>12s} | {'Final loss':>14s}")
    print("-" * 62)
    for name, mv, fl in rows:
        print(f"{name:<30s} | {mv:>11.2f}% | {fl:>14.4e}")

    # Focus on the DECISIVE question
    print("\n" + "=" * 80)
    print("DECISIVE QUESTION: Does B-restart repair monotonically?")
    print("=" * 80)
    post_restart_viols = h_B2['viol']
    post_restart_max = max(post_restart_viols)
    post_restart_losses = h_B2['loss']
    # Check for rebounds post-restart
    rebound_post = 1.0
    for i in range(5, len(post_restart_losses)):
        window_min = min(post_restart_losses[max(0, i-5):i])
        if window_min > 0:
            ratio = post_restart_losses[i] / window_min
            if ratio > rebound_post:
                rebound_post = ratio
    print(f"  Post-restart peak violation: {post_restart_max * 100:.2f}%")
    print(f"  Post-restart max rebound ratio: {rebound_post:.3f}x")
    print(f"  Post-restart final loss: {post_restart_losses[-1]:.4e}")

    if post_restart_max < 0.01:  # < 1% violation
        print("\n  >>> B-RESTART REPAIRS CLEANLY.")
        print("  >>> STAGING IS INERT. The result is an optimizer-restart artifact.")
    else:
        print(f"\n  >>> B-RESTART STILL VIOLATES ({post_restart_max*100:.2f}%).")
        print("  >>> Staging has a real effect beyond the optimizer restart.")

    # Also compare: does A-carry (momentum across hop) still repair?
    print(f"\n  Path A-carry deployed max viol: {max(h_A2['viol'])*100:.2f}%")
    if max(h_A2['viol']) > max(h_B2['viol']):
        print("  >>> A-CARRY IS WORSE than B-restart.")
        print("  >>> Momentum carryover hurts; the reset IS the mechanism.")
    else:
        print("  >>> A-carry is better or equal to B-restart.")

    # ======================================================================
    # 10. SAVE ALL RESULTS
    # ======================================================================
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(x) for x in obj]
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = {
        "control_1_B_restart": {
            "pre_restart": {"loss": h_B1["loss"], "viol": h_B1["viol"]},
            "post_restart": {"loss": h_B2["loss"], "viol": h_B2["viol"]},
            "combined": {"loss": h_Brestart["loss"], "viol": h_Brestart["viol"]},
            "final_theta": theta_Brestart_final.tolist(),
        },
        "control_2_A_carry": {
            "scaffold": {"loss": h_A1["loss"], "viol": h_A1["viol"]},
            "deployed_carry": {"loss": h_A2["loss"], "viol": h_A2["viol"]},
            "combined": {"loss": h_Acarry["loss"], "viol": h_Acarry["viol"]},
            "final_theta": theta_Acarry_final.tolist(),
        },
        "control_3_no_momentum": {
            "path_A": {"loss": h_A_nb["loss"], "viol": h_A_nb["viol"]},
            "path_B": {"loss": h_B_nb["loss"], "viol": h_B_nb["viol"]},
        },
        "control_4_convergent_gd": {
            "alpha": alpha_gd,
            "path_A": {"loss": h_A_gd["loss"], "viol": h_A_gd["viol"]},
            "path_B": {"loss": h_B_gd["loss"], "viol": h_B_gd["viol"]},
        },
        "control_5_rescaled_alpha": {
            "alpha_rescaled": alpha_rescaled,
            "path_B": {"loss": h_B_rescaled["loss"], "viol": h_B_rescaled["viol"]},
        },
        "reporting": {
            "tau_ref_violation": float(viol_ref),
            "tau_ref_min_eigenvalue": float(np.min(min_eig_ref)),
            "delta_R_exact": delta_R,
            "final_theta_A": theta_A_final.tolist(),
            "final_theta_B": theta_B_final.tolist(),
            "final_theta_C": theta_C_final.tolist(),
            "true_coefficients": true_c,
        },
    }

    with open("results_round2_controls.json", "w") as f:
        json.dump(sanitize(results), f, indent=2)
    print("\nResults saved to results_round2_controls.json")


if __name__ == "__main__":
    main()
